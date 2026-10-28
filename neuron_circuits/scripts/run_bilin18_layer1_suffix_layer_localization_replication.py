#!/usr/bin/env python3
"""Full finite replication of suffix attention-layer localization."""

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
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_layer_localization_pilot import make_masks, masked_attention_program, summarize
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_layer_localization_replication.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_layer_localization_replication.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_LAYER_LOCALIZATION_REPLICATION.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_layer_localization_replication.png"


def main():
    config = json.loads(CONFIG.read_text())
    causal = json.loads(Path(config["source_config"]).read_text())
    old = json.loads(Path(config["source_results"]).read_text())
    source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrast = torch.as_tensor(
        flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None],
        device="cuda", dtype=torch.float32,
    )
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r")
    heldout_start = source["start_row"] + source["discovery_rows"]
    masks = make_masks(config["window_sizes"]); k = config["support_rank"]; records = []
    for site, row in enumerate(old["records"]):
        feature = FEATURES.index(row["feature"]); vector = contrast[:, feature]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64)
        recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]
        recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad():
            donor_trace = forward_layers_native(model, donor_ids, 1)
            recipient_trace = forward_layers_native(model, recipient_ids, 17)
        delta = product_at(model, donor_trace, 1, row["donor_position"]) - product_at(model, recipient_trace, 1, row["recipient_position"])
        exact = semantic_gradient(model, recipient_trace, 1, row["recipient_position"], vector)
        predictions = masked_attention_program(model, recipient_trace, row["recipient_position"], vector, masks)
        predictions["exact"] = exact
        supports = {name: (prediction * delta).topk(k).indices for name, prediction in predictions.items()}
        deltas = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, support) for name, support in supports.items()}}
        with torch.no_grad(): logits = logits_for_deltas(model, 1, recipient_trace[1], deltas, row["recipient_position"])
        base = class_score(logits["base"], vector)
        finite = {name: class_score(value, vector) - base for name, value in logits.items() if name != "base"}
        records.append({
            "feature": row["feature"], "selection": row["selection"],
            "metrics": {name: metrics(prediction, exact, delta, k) for name, prediction in predictions.items()},
            "finite_shifts": finite,
        })
        if (site + 1) % 8 == 0: print("layer-localization replication", site + 1, "/", len(old["records"]), flush=True)

    methods = list(records[0]["metrics"]); summaries = {method: summarize(records, method) for method in methods}
    exact_shift = summaries["exact"]["finite_mean_shift"]
    source8 = summaries["source_8"]; logit8 = summaries["logit_8"]; source4 = summaries["source_4"]
    per_feature8 = {
        feature: {
            "gradient_cosine_median": float(np.median([
                row["metrics"]["source_8"]["gradient_cosine"] for row in records if row["feature"] == feature
            ])),
            "finite_directional_fraction": float(np.mean([
                row["finite_shifts"]["source_8"] > 0 for row in records if row["feature"] == feature
            ])),
        }
        for feature in causal["features"]
    }
    single_cosines = [summaries[f"single_{layer}"]["gradient_cosine_median"] for layer in range(2, 18)]
    leaveout_cosines = [summaries[f"leaveout_{layer}"]["gradient_cosine_median"] for layer in range(2, 18)]
    hypotheses = {
        "K0_full_recurrence_is_exact": {"supported": all(
            row["metrics"]["full"]["gradient_cosine"] >= .999
            and row["metrics"]["full"]["gradient_relative_error"] <= .02 for row in records
        )},
        "K1_source_eight_replication": {"supported": source8["gradient_cosine_median"] >= .90 and source8["gradient_cosine_q10"] >= .75},
        "K2_source_eight_causal_replication": {"supported": source8["finite_directional_fraction"] >= .95 and source8["finite_mean_shift"] >= .80 * exact_shift},
        "K3_early_attention_beats_late_attention": {"supported": source8["gradient_cosine_median"] - logit8["gradient_cosine_median"] >= .10 and (source8["finite_mean_shift"] - logit8["finite_mean_shift"]) / exact_shift >= .10},
        "K4_no_single_attention_block_is_sufficient": {"supported": max(single_cosines) < .80},
        "K5_no_single_attention_block_is_decisive": {"supported": min(leaveout_cosines) >= .90},
        "K6_source_eight_transfers_across_features": {"supported": all(value["gradient_cosine_median"] >= .80 and value["finite_directional_fraction"] >= .75 for value in per_feature8.values())},
        "K7_source_four_is_insufficient_control": {"supported": source4["gradient_cosine_median"] < .90 or source4["finite_mean_shift"] < .80 * exact_shift},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "per_feature_source8": per_feature8, "records": records}, indent=2, allow_nan=False))
    display = ["mlp_only", "source_1", "source_2", "source_4", "source_8", "source_12", "logit_1", "logit_2", "logit_4", "logit_8", "logit_12", "both_ends_4", "full"]
    lines = ["# Full replication of suffix attention-layer localization", "", "All 96 frozen sites are evaluated. Attention Jacobian responses are enabled in specified blocks while every residual and MLP response remains native; each program selects and finitely patches its own top-64 Layer-1 channels.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Layer localization replication](bilin18_layer1_suffix_layer_localization_replication.png)", "", "| Attention-enabled blocks | Gradient cosine / Q10 | Contribution cosine | Finite direction | Shift / exact |", "|---|---:|---:|---:|---:|"]
    for method in display:
        value = summaries[method]
        lines.append(f"| {method} | {value['gradient_cosine_median']:.3f}/{value['gradient_cosine_q10']:.3f} | {value['contribution_cosine_median']:.3f} | {value['finite_directional_fraction']:.3f} | {value['finite_mean_shift'] / exact_shift:.3f} |")
    lines += ["", "## Source-eight transfer by feature", "", "| Feature | Median gradient cosine | Finite direction |", "|---|---:|---:|"]
    for feature, value in per_feature8.items(): lines.append(f"| {feature} | {value['gradient_cosine_median']:.3f} | {value['finite_directional_fraction']:.3f} |")
    lines += ["", "## Guardrail", "", "The early-window advantage is an ordering result for local sensitivity transport. It does not by itself mean early blocks contain more semantic content: an early attention response also has more downstream MLP operators through which to be transformed."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained"); sizes = config["window_sizes"]
    axes[0].plot(sizes, [summaries[f"source_{width}"]["gradient_cosine_median"] for width in sizes], marker="o", label="nearest Layer 1")
    axes[0].plot(sizes, [summaries[f"logit_{width}"]["gradient_cosine_median"] for width in sizes], marker="o", label="nearest logits")
    axes[0].axhline(summaries["mlp_only"]["gradient_cosine_median"], color="gray", linestyle=":", label="MLP-only")
    axes[0].set(xlabel="Attention-enabled suffix blocks", ylabel="Median gradient cosine", title="Cumulative attention response"); axes[0].legend(); axes[0].grid(alpha=.2)
    layers = list(range(2, 18)); axes[1].plot(layers, single_cosines, marker="o", label="only block enabled"); axes[1].plot(layers, [1 - value for value in leaveout_cosines], marker="o", label="cosine loss when omitted")
    axes[1].set(xlabel="Suffix block", ylabel="Median cosine / cosine loss", title="No single attention bottleneck"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "key_summaries": {key: summaries[key] for key in display}, "per_feature_source8": per_feature8}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
