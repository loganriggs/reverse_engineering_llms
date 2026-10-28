#!/usr/bin/env python3
"""Full finite replication of early QK/value suffix factor programs."""

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
from run_bilin18_layer1_suffix_attention_factor_pilot import factor_program, make_specifications
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_layer_localization_pilot import summarize
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_attention_factor_replication.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_attention_factor_replication.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_ATTENTION_FACTOR_REPLICATION.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_attention_factor_replication.png"


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrast = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]
    specifications = make_specifications(config["attention_layers"]); k = config["support_rank"]; records = []; maximum_local_error = 0.0
    for site, row in enumerate(old["records"]):
        feature = FEATURES.index(row["feature"]); vector = contrast[:, feature]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64)
        recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, 1); recipient_trace = forward_layers_native(model, recipient_ids, 17)
        delta = product_at(model, donor_trace, 1, row["donor_position"]) - product_at(model, recipient_trace, 1, row["recipient_position"])
        exact = semantic_gradient(model, recipient_trace, 1, row["recipient_position"], vector)
        predictions, error = factor_program(model, recipient_trace, row["recipient_position"], vector, specifications); maximum_local_error = max(maximum_local_error, error); predictions["exact"] = exact
        supports = {name: (prediction * delta).topk(k).indices for name, prediction in predictions.items()}
        deltas = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, support) for name, support in supports.items()}}
        with torch.no_grad(): logits = logits_for_deltas(model, 1, recipient_trace[1], deltas, row["recipient_position"])
        base = class_score(logits["base"], vector); finite = {name: class_score(value, vector) - base for name, value in logits.items() if name != "base"}
        records.append({"feature": row["feature"], "selection": row["selection"], "metrics": {name: metrics(prediction, exact, delta, k) for name, prediction in predictions.items()}, "finite_shifts": finite})
        if (site + 1) % 8 == 0: print("attention-factor replication", site + 1, "/", len(old["records"]), flush=True)
    methods = list(records[0]["metrics"]); summaries = {method: summarize(records, method) for method in methods}; exact_shift = summaries["exact"]["finite_mean_shift"]
    qk = summaries["early_qk"]; value = summaries["early_value"]; full = summaries["early_full"]
    per_feature_qk = {feature: {"gradient_cosine_median": float(np.median([row["metrics"]["early_qk"]["gradient_cosine"] for row in records if row["feature"] == feature])), "finite_directional_fraction": float(np.mean([row["finite_shifts"]["early_qk"] > 0 for row in records if row["feature"] == feature]))} for feature in causal["features"]}
    hypotheses = {
        "N0_factor_program_is_exact": {"supported": maximum_local_error <= 1e-5 and all(row["metrics"]["all_full"]["gradient_cosine"] >= .999 and row["metrics"]["all_full"]["gradient_relative_error"] <= .02 for row in records)},
        "N1_qk_routing_replication": {"supported": qk["gradient_cosine_median"] >= .90 and qk["gradient_cosine_q10"] >= .75},
        "N2_qk_routing_is_causally_useful": {"supported": qk["finite_directional_fraction"] >= .95 and qk["finite_mean_shift"] >= .80 * exact_shift},
        "N3_value_transport_is_insufficient": {"supported": value["gradient_cosine_median"] < .85},
        "N4_value_adds_limited_increment_beyond_qk": {"supported": full["gradient_cosine_median"] - qk["gradient_cosine_median"] < .08},
        "N5_two_qk_factors_are_complementary": {"supported": qk["gradient_cosine_median"] - max(summaries["early_qk1"]["gradient_cosine_median"], summaries["early_qk2"]["gradient_cosine_median"]) >= .05},
        "N6_qk_routing_transfers_across_features": {"supported": all(value["gradient_cosine_median"] >= .80 and value["finite_directional_fraction"] >= .75 for value in per_feature_qk.values())},
        "N7_no_single_qk_factor_plus_value_is_sufficient": {"supported": summaries["early_qk1_value"]["gradient_cosine_median"] < .90 and summaries["early_qk2_value"]["gradient_cosine_median"] < .90},
    }
    OUTPUT.write_text(json.dumps({"config": config, "maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "summaries": summaries, "per_feature_qk": per_feature_qk, "records": records}, indent=2, allow_nan=False))
    display = ["mlp_only", "early_qk1", "early_qk2", "early_qk", "early_value", "early_qk1_value", "early_qk2_value", "early_full", "all_full"]
    lines = ["# Full replication of early QK/value suffix programs", "", "All 96 frozen semantic sites are evaluated with exact local QK1, QK2, and value VJPs and finite top-64 Layer-1 channel patches.", "", f"Maximum local recomposition error: `{maximum_local_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Attention factor replication](bilin18_layer1_suffix_attention_factor_replication.png)", "", "| Enabled early derivative | Gradient cosine / Q10 | Contribution cosine | Finite direction | Shift / exact |", "|---|---:|---:|---:|---:|"]
    for method in display:
        value = summaries[method]; lines.append(f"| {method} | {value['gradient_cosine_median']:.3f}/{value['gradient_cosine_q10']:.3f} | {value['contribution_cosine_median']:.3f} | {value['finite_directional_fraction']:.3f} | {value['finite_mean_shift'] / exact_shift:.3f} |")
    lines += ["", "## QK-only transfer by feature", "", "| Feature | Median cosine | Finite direction |", "|---|---:|---:|"]
    for feature, value in per_feature_qk.items(): lines.append(f"| {feature} | {value['gradient_cosine_median']:.3f} | {value['finite_directional_fraction']:.3f} |")
    lines += ["", "## Guardrail", "", "QK dominance here means local sensitivity routing explains the Layer-1 semantic gradient better than current-value sensitivity. The QK program still composes two routing factors across eight blocks and all heads and positions; it is not yet a sparse human-readable circuit."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained"); names = display[:-1]
    axes[0].bar(range(len(names)), [summaries[name]["gradient_cosine_median"] for name in names]); axes[0].set_xticks(range(len(names)), names, rotation=30, ha="right"); axes[0].set(ylabel="Median gradient cosine", title="Routing versus value sensitivity"); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(range(len(names)), [summaries[name]["finite_mean_shift"] / exact_shift for name in names]); axes[1].set_xticks(range(len(names)), names, rotation=30, ha="right"); axes[1].axhline(1, color="gray", linestyle=":"); axes[1].set(ylabel="Mean finite shift / exact", title="Causal strength"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "summaries": summaries, "per_feature_qk": per_feature_qk}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
