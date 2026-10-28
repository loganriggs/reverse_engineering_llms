#!/usr/bin/env python3
"""Factor pair-specific activation changes from pair-specific suffix sensitivity."""

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
from run_bilin18_layer01_supervised_feature_census import FEATURES, collect_split, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_supervised_coalition_causality import class_score, fit_supports, product_at, summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_coalition_factorial.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_coalition_factorial.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_COALITION_FACTORIAL.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_coalition_factorial.png"
ARRAYS = TENSOR_ROOT / "bilin18_layer1_coalition_factorial_arrays.npz"


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrast_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]
    contrast = torch.as_tensor(contrast_np, device="cuda", dtype=torch.float32)
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); discovery_start = source["start_row"]; heldout_start = discovery_start + source["discovery_rows"]
    discovery_rows = corpus[discovery_start:heldout_start]
    discovery_products, discovery_labels, discovery_ids, _ = collect_split(model, discovery_rows, source["positions"], flags, contrast, source["batch_size"])
    layer, rank = causal["layer"], config["rank"]; feature_indices = [FEATURES.index(name) for name in causal["features"]]
    supports = fit_supports(discovery_products[layer], discovery_labels, feature_indices, rank)
    mean_gradients = {}; discovery_gradient_rows = []; discovery_gradient_features = []; positions = source["positions"]
    for feature in feature_indices:
        negatives = np.flatnonzero(~discovery_labels[:, feature].astype(bool)); chosen, seen = [], set()
        for flat in negatives:
            token = int(discovery_ids[flat])
            if token in seen: continue
            seen.add(token); chosen.append(int(flat))
            if len(chosen) == config["gradient_sites_per_feature"]: break
        gradients = []
        for flat in chosen:
            row, position_index = divmod(flat, len(positions)); ids = torch.as_tensor(np.asarray(discovery_rows[row, :source["sequence_length"]], dtype=np.int64), device="cuda")[None]
            with torch.no_grad(): trace = forward_layers_native(model, ids, layer)
            gradients.append(semantic_gradient(model, trace, layer, positions[position_index], contrast[:, feature]))
        mean_gradients[feature] = torch.stack(gradients).mean(0)
        discovery_gradient_rows.extend([gradient.cpu().numpy().astype(np.float16) for gradient in gradients]); discovery_gradient_features.extend([feature] * len(gradients))
        print("factorial discovery", FEATURES[feature], flush=True)

    records = []; evaluation_gradients = []; evaluation_deltas = []; evaluation_features = []
    for index, old_row in enumerate(old["records"]):
        feature = FEATURES.index(old_row["feature"]); vector = contrast[:, feature]
        donor_raw = np.asarray(corpus[heldout_start + old_row["donor_row"], :source["sequence_length"]], dtype=np.int64); recipient_raw = np.asarray(corpus[heldout_start + old_row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, layer); recipient_trace = forward_layers_native(model, recipient_ids, layer)
        full = product_at(model, donor_trace, layer, old_row["donor_position"]) - product_at(model, recipient_trace, layer, old_row["recipient_position"])
        local_gradient = semantic_gradient(model, recipient_trace, layer, old_row["recipient_position"], vector)
        evaluation_gradients.append(local_gradient.cpu().numpy().astype(np.float16)); evaluation_deltas.append(full.cpu().numpy().astype(np.float16)); evaluation_features.append(feature)
        mean_delta = torch.as_tensor(supports[feature]["std"] * supports[feature]["gap"], device="cuda")
        ingredients = {
            "fixed_both64": mean_delta * mean_gradients[feature],
            "pair_delta64": full * mean_gradients[feature],
            "pair_gradient64": mean_delta * local_gradient,
            "joint64": full * local_gradient,
        }
        indices = {name: value.topk(rank).indices for name, value in ingredients.items()}
        deltas = {"base": torch.zeros_like(full), **{name: selected_delta(full, selected) for name, selected in indices.items()}}
        with torch.no_grad(): logits = logits_for_deltas(model, layer, recipient_trace[layer], deltas, old_row["recipient_position"])
        base = class_score(logits["base"], vector); shifts = {name: class_score(value, vector) - base for name, value in logits.items() if name != "base"}; shifts["full"] = old_row["shifts"]["full"]
        first_order = {name: float((full[selected] * local_gradient[selected]).sum()) for name, selected in indices.items()}
        records.append({"feature": old_row["feature"], "selection": old_row["selection"], "pair": old_row["pair"], "shifts": shifts, "first_order": first_order, "supports": {name: selected.cpu().tolist() for name, selected in indices.items()}})
        if (index + 1) % 12 == 0: print("factorial pairs", index + 1, "/", len(old["records"]), flush=True)
    methods = ("fixed_both64", "pair_delta64", "pair_gradient64", "joint64")
    summary = {}
    for method in methods:
        summary[method] = {"all": summarize(records, method), "extreme": summarize(records, method, "extreme"), "uniform": summarize(records, method, "uniform")}
        predicted = np.asarray([row["first_order"][method] for row in records]); actual = np.asarray([row["shifts"][method] for row in records]); summary[method]["first_order"] = {"sign_agreement": float(np.mean((predicted > 0) == (actual > 0))), "correlation": float(np.corrcoef(predicted, actual)[0, 1])}
    delta_direction = summary["pair_delta64"]["all"]["directional_fraction"]; gradient_direction = summary["pair_gradient64"]["all"]["directional_fraction"]; joint_direction = summary["joint64"]["all"]["directional_fraction"]
    hypotheses = {
        "F0_pair_delta_is_sufficient": {"supported": delta_direction >= .80},
        "F1_pair_gradient_is_sufficient": {"supported": gradient_direction >= .80},
        "F2_joint_conditioning_is_necessary": {"supported": joint_direction - max(delta_direction, gradient_direction) >= .10},
        "F3_activation_delta_is_dominant_router_input": {"supported": delta_direction - gradient_direction >= .10},
        "F4_local_linearization_survives": {"supported": all(summary[method]["first_order"]["sign_agreement"] >= .80 for method in methods)},
        "F5_no_evaluation_outcomes_select_support": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "records": records}, indent=2, allow_nan=False))
    np.savez_compressed(ARRAYS, discovery_gradients=np.stack(discovery_gradient_rows), discovery_features=np.asarray(discovery_gradient_features), evaluation_gradients=np.stack(evaluation_gradients), evaluation_deltas=np.stack(evaluation_deltas), evaluation_features=np.asarray(evaluation_features))
    lines = ["# Layer-1 semantic coalition factorial", "", "The successful pair-local selector combines two state-dependent objects: the donor-minus-recipient product delta and the recipient suffix gradient for the declared semantic score. This 2×2 test replaces either object with its disjoint-discovery feature mean, freezes the resulting rank-64 support before finite intervention, and evaluates the same 96 held-out patches.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Coalition factorial](bilin18_layer1_coalition_factorial.png)", "", "| Delta / gradient | Directional all / uniform | Mean shift | First-order sign / corr. |", "|---|---:|---:|---:|"]
    labels = {"fixed_both64": "mean / mean", "pair_delta64": "pair / mean", "pair_gradient64": "mean / pair", "joint64": "pair / pair"}
    for method in methods:
        row = summary[method]; lines.append(f"| {labels[method]} | {row['all']['directional_fraction']:.3f} / {row['uniform']['directional_fraction']:.3f} | {row['all']['mean_shift']:+.4f} | {row['first_order']['sign_agreement']:.3f} / {row['first_order']['correlation']:.3f} |")
    lines += ["", "## Interpretation", "", "The factorial distinguishes a router that merely predicts which channels changed from one that must also evaluate how the remaining network will use each change. Joint superiority supports the full input-conditioned-coalition account; success of one singly conditioned arm identifies a simpler prospective target."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].bar(range(len(methods)), [summary[method]["all"]["directional_fraction"] for method in methods]); axes[0].set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=20, ha="right"); axes[0].axhline(.5, color="gray", linestyle=":"); axes[0].set(ylabel="Directional fraction", title="Which input must be context-specific?"); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(range(len(methods)), [summary[method]["all"]["mean_shift"] for method in methods]); axes[1].set_xticks(range(len(methods)), [labels[m] for m in methods], rotation=20, ha="right"); axes[1].set(ylabel="Mean finite semantic shift", title="Causal strength of each factorial arm"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
