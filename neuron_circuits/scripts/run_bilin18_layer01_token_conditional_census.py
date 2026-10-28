#!/usr/bin/env python3
"""Measure next-token feature codes while exactly matching the current token."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_layer01_supervised_feature_census import (
    FEATURES, collect_split, vocabulary_features,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_token_conditional_census.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer01_token_conditional_census.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER01_TOKEN_CONDITIONAL_CENSUS.md"
FIGURE = TENSOR_ROOT / "bilin18_layer01_token_conditional_census.png"


def within_token_auc(labels, scores, token_ids):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    token_ids = np.asarray(token_ids)
    weighted, pairs, groups = 0.0, 0, 0
    for token in np.unique(token_ids):
        selected = token_ids == token
        y = labels[selected]
        positives, negatives = int(y.sum()), int((~y).sum())
        if not positives or not negatives: continue
        count = positives * negatives
        weighted += auc(y, scores[selected]) * count
        pairs += count
        groups += 1
    return {"auc": float(weighted / pairs) if pairs else None, "pairs": int(pairs), "groups": groups}


def fit_coalition_scores(discovery, heldout, discovery_labels, sizes):
    discovery = discovery.astype(np.float32); heldout = heldout.astype(np.float32)
    mean = discovery.mean(0); std = discovery.std(0).clip(1e-4)
    zd = (discovery - mean) / std; zh = (heldout - mean) / std
    results = []
    for feature in range(discovery_labels.shape[1]):
        y = discovery_labels[:, feature].astype(bool)
        if not y.any() or y.all():
            results.append({str(size): None for size in sizes}); continue
        gap = zd[y].mean(0) - zd[~y].mean(0)
        order = np.argsort(np.abs(gap))[::-1]
        results.append({str(size): zh[:, order[:size]] @ gap[order[:size]] for size in sizes})
    return results


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrast = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); start = source["start_row"]
    discovery_rows = corpus[start:start + source["discovery_rows"]]
    heldout_start = start + source["discovery_rows"]
    heldout_rows = corpus[heldout_start:heldout_start + source["heldout_rows"]]
    discovery_products, discovery_labels, _, _ = collect_split(model, discovery_rows, source["positions"], flags, contrast, source["batch_size"])
    heldout_products, heldout_labels, heldout_ids, heldout_output = collect_split(model, heldout_rows, source["positions"], flags, contrast, source["batch_size"])
    scores = {str(layer): fit_coalition_scores(discovery_products[layer], heldout_products[layer], discovery_labels, config["coalition_sizes"]) for layer in source["layers"]}
    rows = []
    for feature_index, name in enumerate(FEATURES):
        model_metric = within_token_auc(heldout_labels[:, feature_index], heldout_output[:, feature_index], heldout_ids)
        row = {"feature": name, "positive_examples": int(heldout_labels[:, feature_index].sum()), "model": model_metric, "layers": {}}
        for layer in source["layers"]:
            row["layers"][str(layer)] = {}
            for size in config["coalition_sizes"]:
                values = scores[str(layer)][feature_index][str(size)]
                row["layers"][str(layer)][str(size)] = {"auc": None, "pairs": model_metric["pairs"], "groups": model_metric["groups"]} if values is None else within_token_auc(heldout_labels[:, feature_index], values, heldout_ids)
        row["eligible"] = model_metric["pairs"] >= config["minimum_within_token_pairs"]
        row["competent"] = row["eligible"] and model_metric["auc"] is not None and model_metric["auc"] >= .55
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]; competent = [row for row in rows if row["competent"]]
    med = lambda values: float(np.median(values)) if values else None
    summary = {"eligible_features": len(eligible), "competent_features": len(competent), "model_auc": med([row["model"]["auc"] for row in competent]), "layers": {}}
    for layer in source["layers"]:
        key = str(layer); summary["layers"][key] = {str(size): med([row["layers"][key][str(size)]["auc"] for row in competent]) for size in config["coalition_sizes"]}
    hypotheses = {
        "T0_conditional_coverage": {"supported": len(eligible) >= 20},
        "T1_model_behavior_uses_context": {"supported": bool(competent) and summary["model_auc"] >= .60},
        "T2_layer1_coalitions_encode_context": {"supported": bool(competent) and summary["layers"]["1"]["64"] >= .60},
        "T3_layer1_approaches_model_behavior": {"supported": bool(competent) and summary["model_auc"] - summary["layers"]["1"]["256"] <= .05},
        "T4_depth_improves_conditional_code": {"supported": bool(competent) and summary["layers"]["1"]["64"] - summary["layers"]["0"]["64"] >= .02},
        "T5_conditional_code_is_not_single_neuron_aligned": {"supported": bool(competent) and summary["layers"]["1"]["64"] - summary["layers"]["1"]["1"] >= .05},
        "T6_exact_token_matching_removes_embedding_only_explanations": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "features": rows}, indent=2, allow_nan=False))
    lines = ["# Within-current-token supervised feature census", "", "Every AUROC comparison is made only between held-out examples with the exact same current GPT-2 token but different next-token feature labels. The metric is a pair-weighted average across token identities. Any predictor that is a deterministic function of the current embedding alone—including an isolated-token forward pass—scores exactly 0.5 on these pairs.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Conditional census](bilin18_layer01_token_conditional_census.png)", "", f"Eligible labels: **{len(eligible)}**; conditionally competent labels: **{len(competent)}**; median model-output conditional AUROC: **{summary['model_auc']:.3f}**.", "", "| Feature | pairs/groups | Model | L0 one / 64 / 256 | L1 one / 64 / 256 |", "|---|---:|---:|---:|---:|"]
    for row in sorted(eligible, key=lambda item: item["model"]["auc"], reverse=True):
        l0, l1 = row["layers"]["0"], row["layers"]["1"]
        lines.append(f"| {row['feature']} | {row['model']['pairs']:,}/{row['model']['groups']} | {row['model']['auc']:.3f} | {l0['1']['auc']:.3f}/{l0['64']['auc']:.3f}/{l0['256']['auc']:.3f} | {l1['1']['auc']:.3f}/{l1['64']['auc']:.3f}/{l1['256']['auc']:.3f} |")
    lines += ["", "## Red-team interpretation", "", "- Exact current-token matching removes deterministic embedding and isolated-token explanations, but not correlations with earlier context that are irrelevant to the model's actual next-token computation.", "- The output competence score is an exact mean-logit contrast over the declared token class, not a calibrated class probability.", "- Fixed supervised coalitions are descriptive decoders. Finite interventions are still needed to establish that the decoded contextual information is causally used through these channels."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    sizes = config["coalition_sizes"]
    for layer in source["layers"]: axes[0].plot(sizes, [summary["layers"][str(layer)][str(size)] for size in sizes], marker="o", label=f"layer {layer}")
    axes[0].axhline(summary["model_auc"], color="black", linestyle="--", label="model output"); axes[0].axhline(.5, color="gray", linestyle=":"); axes[0].set_xscale("log", base=2); axes[0].set(xlabel="Native product channels", ylabel="Within-token AUROC", title="Context code versus coalition size"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].scatter([row["model"]["auc"] for row in competent], [row["layers"]["1"]["256"]["auc"] for row in competent], alpha=.8)
    for row in competent:
        if abs(row["model"]["auc"] - row["layers"]["1"]["256"]["auc"]) > .1: axes[1].annotate(row["feature"], (row["model"]["auc"], row["layers"]["1"]["256"]["auc"]), fontsize=7)
    axes[1].plot([.5, 1], [.5, 1], color="black", linestyle="--"); axes[1].set(xlabel="Model output conditional AUROC", ylabel="Layer-1 rank-256 conditional AUROC", title="Does the early code track behavior?"); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
