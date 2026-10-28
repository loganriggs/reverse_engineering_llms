#!/usr/bin/env python3
"""Finite validation of recipient-conditioned semantic channel coalitions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at, summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_dynamic_semantic_coalitions.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_dynamic_semantic_coalitions.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_DYNAMIC_SEMANTIC_COALITIONS.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_dynamic_semantic_coalitions.png"


def main():
    config = json.loads(CONFIG.read_text()); old = json.loads(Path(config["source_results"]).read_text()); causal = json.loads(Path(config["source_config"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrast_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]
    contrast = torch.as_tensor(contrast_np, device="cuda", dtype=torch.float32)
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]; layer = causal["layer"]; rank = config["rank"]
    unembedding = model.lm_head.weight.float()[:len(flags)]; records = []
    for index, old_row in enumerate(old["records"]):
        feature = FEATURES.index(old_row["feature"]); vector = contrast[:, feature]
        donor_raw = np.asarray(corpus[heldout_start + old_row["donor_row"], :source["sequence_length"]], dtype=np.int64); recipient_raw = np.asarray(corpus[heldout_start + old_row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, layer); recipient_trace = forward_layers_native(model, recipient_ids, layer)
        donor_product = product_at(model, donor_trace, layer, old_row["donor_position"]); recipient_product = product_at(model, recipient_trace, layer, old_row["recipient_position"]); full = donor_product - recipient_product
        gradient = semantic_gradient(model, recipient_trace, layer, old_row["recipient_position"], vector)
        contribution = full * gradient
        absolute_indices = contribution.abs().topk(rank).indices; positive_indices = contribution.topk(rank).indices
        output_direction = torch.as_tensor(contrast_np[:, feature], device="cuda") @ unembedding
        direct = model.blocks[layer].mlp.Down.weight.float().T @ output_direction
        direct_indices = (full * direct).abs().topk(rank).indices
        deltas = {"base": torch.zeros_like(full), "full": full, "local_absolute64": selected_delta(full, absolute_indices), "local_positive64": selected_delta(full, positive_indices), "local_direct64": selected_delta(full, direct_indices)}
        with torch.no_grad(): logits = logits_for_deltas(model, layer, recipient_trace[layer], deltas, old_row["recipient_position"])
        base = class_score(logits["base"], vector); shifts = {name: class_score(value, vector) - base for name, value in logits.items() if name != "base"}
        predicted = {"local_absolute64": float(contribution[absolute_indices].sum()), "local_positive64": float(contribution[positive_indices].sum()), "full": float(contribution.sum())}
        records.append({"feature": old_row["feature"], "selection": old_row["selection"], "pair": old_row["pair"], "shifts": {**shifts, "fixed_activation64": old_row["shifts"]["supervised64"]}, "first_order": predicted, "supports": {"local_positive64": positive_indices.cpu().tolist(), "local_absolute64": absolute_indices.cpu().tolist(), "local_direct64": direct_indices.cpu().tolist()}})
        if (index + 1) % 12 == 0: print("dynamic semantic pairs", index + 1, "/", len(old["records"]), flush=True)
    methods = ("full", "fixed_activation64", "local_direct64", "local_absolute64", "local_positive64")
    summary = {method: {"all": summarize(records, method), "extreme": summarize(records, method, "extreme"), "uniform": summarize(records, method, "uniform")} for method in methods}
    actual = np.asarray([row["shifts"]["local_positive64"] for row in records]); predicted = np.asarray([row["first_order"]["local_positive64"] for row in records])
    first_order = {"positive_prediction_fraction": float(np.mean(predicted > 0)), "finite_positive_fraction": float(np.mean(actual > 0)), "sign_agreement": float(np.mean((predicted > 0) == (actual > 0))), "correlation": float(np.corrcoef(predicted, actual)[0, 1])}
    local_positive, local_absolute, fixed = summary["local_positive64"], summary["local_absolute64"], summary["fixed_activation64"]
    hypotheses = {
        "D0_local_positive_coalition_is_directional": {"supported": local_positive["all"]["directional_fraction"] >= .75},
        "D1_local_positive_generalizes_across_pair_selection": {"supported": local_positive["uniform"]["directional_fraction"] >= .70},
        "D2_local_absolute_support_tracks_full_path": {"supported": local_absolute["all"]["correlation_with_full"] is not None and local_absolute["all"]["correlation_with_full"] >= .60},
        "D3_dynamic_support_beats_fixed_semantic_support": {"supported": local_positive["all"]["mean_shift"] >= 2 * fixed["all"]["mean_shift"]},
        "D4_first_order_sign_survives_finite_patch": {"supported": first_order["sign_agreement"] >= .80},
        "D5_dynamic_selector_is_an_upper_bound": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "first_order": first_order, "records": records}, indent=2, allow_nan=False))
    lines = ["# Dynamic semantic coalitions at layer 1", "", "For each held-out same-token donor–recipient pair, the exact recipient-state derivative of the final semantic class score with respect to all 4,608 layer-1 product channels is combined with the donor activation delta. Two rank-64 supports are then validated by finite patching: largest absolute first-order contributions (approximate the full path) and largest positive contributions (move toward the donor label). This reads the endpoint activation delta and is therefore a mechanistic upper bound, not a prospective semantic decoder.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Dynamic semantic coalitions](bilin18_layer1_dynamic_semantic_coalitions.png)", "", "| Method | Directional all / uniform | Mean shift | Mean abs | Corr. with full |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        row = summary[method]["all"]; corr = "—" if row["correlation_with_full"] is None else f"{row['correlation_with_full']:.3f}"; lines.append(f"| {method} | {row['directional_fraction']:.3f} / {summary[method]['uniform']['directional_fraction']:.3f} | {row['mean_shift']:+.4f} | {row['mean_absolute_shift']:.4f} | {corr} |")
    lines += ["", f"First-order versus finite local-positive patches: sign agreement **{first_order['sign_agreement']:.1%}**, correlation **{first_order['correlation']:.3f}**.", "", "## Interpretation", "", "Success establishes that a small recipient-conditioned native-channel coalition can be computed and causally validated. It does not make the feature neuron-aligned: the selected support may change almost completely across contexts. Failure would mean either that 64 channels are insufficient or that a first-order channel decomposition is not stable for these donor-sized finite moves."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].bar(range(len(methods)), [summary[method]["all"]["directional_fraction"] for method in methods]); axes[0].set_xticks(range(len(methods)), methods, rotation=25, ha="right"); axes[0].axhline(.5, color="gray", linestyle=":"); axes[0].set(ylabel="Directional fraction", title="Fixed versus recipient-conditioned supports"); axes[0].grid(axis="y", alpha=.2)
    axes[1].scatter(predicted, actual, alpha=.7); axes[1].axhline(0, color="gray"); axes[1].axvline(0, color="gray"); axes[1].set(xlabel="Predicted first-order shift", ylabel="Finite patch shift", title="Does local tensor sensitivity survive the edit?"); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary, "first_order": first_order}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
