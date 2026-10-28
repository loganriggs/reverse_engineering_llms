#!/usr/bin/env python3
"""Compare activation-only, direct-weight, and suffix-gradient semantic supports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import differentiable_suffix, logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, collect_split, vocabulary_features
from run_bilin18_layer1_supervised_coalition_causality import class_score, fit_supports, product_at, summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_output_aware_coalitions.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_output_aware_coalitions.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_OUTPUT_AWARE_COALITIONS.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_output_aware_coalitions.png"


def clear_rotary(model, start):
    for block in model.blocks[start:]:
        block.attn.rotary.seq_len_cached = None; block.attn.rotary.cos_cached = None; block.attn.rotary.sin_cached = None


def semantic_gradient(model, trace, layer, position, contrast):
    decoder = model.blocks[layer].mlp.Down.weight.float()
    x0 = trace[layer]["x0"].detach().clone(); bus = trace[layer]["value_bus"].detach().clone(); output = trace[layer]["output"].detach().clone()
    clear_rotary(model, layer + 1)
    with torch.enable_grad():
        variable = torch.zeros(decoder.shape[1], device="cuda", requires_grad=True)
        mask = torch.zeros_like(output)
        mask[:, position] = F.linear(variable, decoder).to(output.dtype)
        logits = differentiable_suffix(model, layer, x0, bus, output + mask, position)[0]
        score = logits[:contrast.numel()].float() @ contrast
        gradient = torch.autograd.grad(score, variable)[0]
    return gradient.detach().float()


def main():
    red = json.loads(CONFIG.read_text()); causal_config = json.loads(Path(red["source_config"]).read_text()); old = json.loads(Path(red["source_results"]).read_text()); source = json.loads(Path(causal_config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrast_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]
    contrast = torch.as_tensor(contrast_np, device="cuda", dtype=torch.float32)
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); start = source["start_row"]
    discovery_rows = corpus[start:start + source["discovery_rows"]]; heldout_start = start + source["discovery_rows"]
    discovery_products, discovery_labels, discovery_ids, _ = collect_split(model, discovery_rows, source["positions"], flags, contrast, source["batch_size"])
    feature_indices = [FEATURES.index(name) for name in causal_config["features"]]; layer = causal_config["layer"]; rank = causal_config["rank"]
    supports = fit_supports(discovery_products[layer], discovery_labels, feature_indices, rank)
    unembedding = model.lm_head.weight.float()[:len(flags)]
    direct_supports, gradient_supports = {}, {}
    positions = source["positions"]
    for feature in feature_indices:
        delta_mean = torch.as_tensor(supports[feature]["std"] * supports[feature]["gap"], device="cuda")
        output_direction = torch.as_tensor(contrast_np[:, feature], device="cuda") @ unembedding
        direct_coefficient = model.blocks[layer].mlp.Down.weight.float().T @ output_direction
        direct_supports[feature] = (delta_mean * direct_coefficient).abs().topk(rank).indices
        negatives = np.flatnonzero(~discovery_labels[:, feature].astype(bool))
        chosen, seen = [], set()
        for flat in negatives:
            token = int(discovery_ids[flat])
            if token in seen: continue
            seen.add(token); chosen.append(int(flat))
            if len(chosen) == red["gradient_sites_per_feature"]: break
        gradients = []
        for flat in chosen:
            row, position = divmod(flat, len(positions)); ids = torch.as_tensor(np.asarray(discovery_rows[row, :source["sequence_length"]], dtype=np.int64), device="cuda")[None]
            with torch.no_grad(): trace = forward_layers_native(model, ids, layer)
            gradients.append(semantic_gradient(model, trace, layer, positions[position], contrast[:, feature]))
        average_gradient = torch.stack(gradients).mean(0)
        gradient_supports[feature] = (delta_mean * average_gradient).abs().topk(rank).indices
        print("output-aware support", FEATURES[feature], flush=True)

    records = []
    for old_row in old["records"]:
        feature = FEATURES.index(old_row["feature"])
        donor_raw = np.asarray(corpus[heldout_start + old_row["donor_row"], :source["sequence_length"]], dtype=np.int64)
        recipient_raw = np.asarray(corpus[heldout_start + old_row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, layer); recipient_trace = forward_layers_native(model, recipient_ids, layer)
        donor_product = product_at(model, donor_trace, layer, old_row["donor_position"]); recipient_product = product_at(model, recipient_trace, layer, old_row["recipient_position"]); full = donor_product - recipient_product
        activation_indices = torch.as_tensor(supports[feature]["indices"], device="cuda")
        magnitude_indices = full.abs().topk(rank).indices
        deltas = {"base": torch.zeros_like(full), "full": full, "activation64": selected_delta(full, activation_indices), "direct64": selected_delta(full, direct_supports[feature]), "gradient64": selected_delta(full, gradient_supports[feature]), "magnitude64": selected_delta(full, magnitude_indices)}
        with torch.no_grad(): logits = logits_for_deltas(model, layer, recipient_trace[layer], deltas, old_row["recipient_position"])
        vector = contrast[:, feature]; base = class_score(logits["base"], vector)
        shifts = {name: class_score(value, vector) - base for name, value in logits.items() if name != "base"}
        records.append({"feature": old_row["feature"], "selection": old_row["selection"], "pair": old_row["pair"], "shifts": shifts})
    methods = ("full", "activation64", "direct64", "gradient64", "magnitude64")
    summary = {method: {"all": summarize(records, method), "extreme": summarize(records, method, "extreme"), "uniform": summarize(records, method, "uniform")} for method in methods}
    activation, direct, gradient = summary["activation64"], summary["direct64"], summary["gradient64"]
    hypotheses = {
        "W0_suffix_gradient_improves_direction": {"supported": gradient["all"]["directional_fraction"] - activation["all"]["directional_fraction"] >= .10},
        "W1_suffix_gradient_improves_effect": {"supported": gradient["all"]["mean_shift"] >= 2 * activation["all"]["mean_shift"]},
        "W2_direct_weight_path_helps": {"supported": direct["all"]["directional_fraction"] - activation["all"]["directional_fraction"] >= .05},
        "W3_suffix_gradient_tracks_full_path": {"supported": gradient["all"]["correlation_with_full"] is not None and gradient["all"]["correlation_with_full"] >= .50},
        "W4_output_aware_effect_generalizes": {"supported": gradient["uniform"]["directional_fraction"] >= .60 and gradient["uniform"]["directional_fraction"] > activation["uniform"]["directional_fraction"]},
        "W5_selection_is_discovery_only": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summary": summary, "records": records}, indent=2, allow_nan=False))
    lines = ["# Output-aware supervised coalitions", "", "The same frozen 96 held-out donor–recipient pairs are rerun. Activation-only supports use feature predictivity. Direct supports additionally use the exact layer-1 Down weights and the raw unembedding class direction. Suffix-gradient supports use four disjoint discovery states per feature and differentiate the final clipped class score through layers 2–17. No held-out intervention outcome enters either fixed support.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Output-aware coalitions](bilin18_layer1_output_aware_coalitions.png)", "", "| Method | Directional all / uniform | Mean shift | Mean abs | Corr. with full |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        row = summary[method]["all"]; corr = "—" if row["correlation_with_full"] is None else f"{row['correlation_with_full']:.3f}"; lines.append(f"| {method} | {row['directional_fraction']:.3f} / {summary[method]['uniform']['directional_fraction']:.3f} | {row['mean_shift']:+.4f} | {row['mean_absolute_shift']:.4f} | {corr} |")
    lines += ["", "## Interpretation", "", "If suffix sensitivity improves selection, activation selectivity and output use are distinct parts of a feature explanation. If it does not, a fixed average gradient is itself too coarse: downstream use may be input-conditioned, or the feature may travel mostly outside this MLP path."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].bar(range(len(methods)), [summary[method]["all"]["directional_fraction"] for method in methods]); axes[0].set_xticks(range(len(methods)), methods, rotation=25, ha="right"); axes[0].axhline(.5, color="gray", linestyle=":"); axes[0].set(ylabel="Directional fraction", title="Which fixed support causes the semantic shift?"); axes[0].grid(axis="y", alpha=.2)
    axes[1].scatter([row["shifts"]["full"] for row in records], [row["shifts"]["gradient64"] for row in records], alpha=.7); axes[1].axhline(0, color="gray"); axes[1].axvline(0, color="gray"); axes[1].set(xlabel="Full product-patch shift", ylabel="Gradient-selected shift", title="Output-aware support versus full path"); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
