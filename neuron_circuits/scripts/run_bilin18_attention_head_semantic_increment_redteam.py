#!/usr/bin/env python3
"""Red-team one-head semantics by conditioning on the residual-only operator state."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_attention_head_operator_roles import (
    collect_layer,
    coordinates,
    normalized_state,
    one_head_coordinates,
    operator_map,
)
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features, within_token_auc
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from transformers import AutoTokenizer


CONFIG = Path("configs/bilin18_attention_head_semantic_increment_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_attention_head_semantic_increment_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_ATTENTION_HEAD_SEMANTIC_INCREMENT_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_attention_head_semantic_increment_redteam.png"


def fit_decoder(train, labels):
    mean = train.mean(0); std = train.std(0).clamp_min(1e-4)
    normalized = (train - mean) / std
    positive = labels.bool()
    gap = normalized[positive].mean(0) - normalized[~positive].mean(0)
    return mean, std, gap


def predict_decoder(model, value):
    mean, std, gap = model
    return ((value - mean) / std) @ gap


def score(train, train_labels, test, test_labels, current=None):
    prediction = predict_decoder(fit_decoder(train, train_labels), test).numpy()
    result = {"auc": float(auc(test_labels.numpy(), prediction))}
    if current is not None:
        value, groups, pairs = within_token_auc(test_labels.numpy(), prediction, current.numpy())
        result.update({"within_token_auc": value, "within_token_groups": groups, "within_token_pairs": pairs})
    return result


@torch.inference_mode()
def layer_records(train, test, basis, eligible, config):
    mean, mapping, _ = operator_map(basis, config["rank"], "cuda")
    train_mixed = train["mixed"].cuda(); train_heads = train["heads"].cuda()
    test_mixed = test["mixed"].cuda(); test_heads = test["heads"].cuda()
    train_base = coordinates(normalized_state(train_mixed, train_heads, ()), mean, mapping).cpu()
    test_base = coordinates(normalized_state(test_mixed, test_heads, ()), mean, mapping).cpu()
    all_heads = tuple(range(train_heads.shape[1]))
    train_full = coordinates(normalized_state(train_mixed, train_heads, all_heads), mean, mapping).cpu()
    test_full = coordinates(normalized_state(test_mixed, test_heads, all_heads), mean, mapping).cpu()
    train_one = one_head_coordinates(train_mixed, train_heads, mean, mapping).cpu()
    test_one = one_head_coordinates(test_mixed, test_heads, mean, mapping).cpu()
    train_delta = train_one - train_base[:, None]
    test_delta = test_one - test_base[:, None]
    rows = config["rows_per_split"]; positions = len(config["positions"])
    row_index = torch.arange(rows).repeat_interleave(positions)
    fitting = row_index < rows // 2; validation = ~fitting
    records = []
    for local_feature, feature in enumerate(eligible):
        fit_labels = train["labels"][fitting, local_feature]
        validation_labels = train["labels"][validation, local_feature]
        baseline_validation = score(train_base[fitting], fit_labels, train_base[validation], validation_labels)["auc"]
        validation_gains = []
        for head in range(train_heads.shape[1]):
            candidate = torch.cat((train_base, train_delta[:, head]), 1)
            candidate_auc = score(candidate[fitting], fit_labels, candidate[validation], validation_labels)["auc"]
            validation_gains.append(candidate_auc - baseline_validation)
        selected = int(np.nanargmax(validation_gains))
        labels = train["labels"][:, local_feature]; evaluation_labels = test["labels"][:, local_feature]
        baseline = score(train_base, labels, test_base, evaluation_labels, test["current"])
        full = score(train_full, labels, test_full, evaluation_labels, test["current"])
        selected_train = torch.cat((train_base, train_delta[:, selected]), 1)
        selected_test = torch.cat((test_base, test_delta[:, selected]), 1)
        selected_result = score(selected_train, labels, selected_test, evaluation_labels, test["current"])
        delta_only = score(train_delta[:, selected], labels, test_delta[:, selected], evaluation_labels, test["current"])
        evaluation_gains = []
        for head in range(train_heads.shape[1]):
            candidate_train = torch.cat((train_base, train_delta[:, head]), 1)
            candidate_test = torch.cat((test_base, test_delta[:, head]), 1)
            evaluation_gains.append(score(candidate_train, labels, candidate_test, evaluation_labels)["auc"] - baseline["auc"])
        rank = 1 + sum(value > evaluation_gains[selected] for value in evaluation_gains)
        records.append({
            "feature": FEATURES[feature], "selected_head": selected,
            "validation_increment": float(validation_gains[selected]), "evaluation_head_rank": int(rank),
            "residual": baseline, "full": full, "residual_plus_head": selected_result,
            "head_delta_only": delta_only,
            "full_increment": full["auc"] - baseline["auc"],
            "head_increment": selected_result["auc"] - baseline["auc"],
            "within_token_head_increment": selected_result["within_token_auc"] - baseline["within_token_auc"] if selected_result["within_token_auc"] is not None and baseline["within_token_auc"] is not None else None,
        })
    return records


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    bases = torch.load(config["bases"], map_location="cpu", weights_only=True)
    competence = json.loads(Path(config["competence_results"]).read_text())
    eligible = [FEATURES.index(row["feature"]) for row in competence["features"] if row["competent"]]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}
    for layer in config["layers"]:
        print("increment redteam layer", layer, "discovery", flush=True)
        train = collect_layer(model, layer, config["corpus"], config["discovery_start"], config["rows_per_split"], config, flags, eligible)
        print("increment redteam layer", layer, "evaluation", flush=True)
        test = collect_layer(model, layer, config["corpus"], config["evaluation_start"], config["rows_per_split"], config, flags, eligible)
        records = layer_records(train, test, bases[str(layer)], eligible, config)
        results[str(layer)] = {"records": records, "summary": {
            "median_full_increment": float(np.median([row["full_increment"] for row in records])),
            "median_head_increment": float(np.median([row["head_increment"] for row in records])),
            "median_delta_only_auc": float(np.median([row["head_delta_only"]["auc"] for row in records])),
            "top3_selection_rate": float(np.mean([row["evaluation_head_rank"] <= 3 for row in records])),
        }}
        del train, test
    records = [row for layer in config["layers"] for row in results[str(layer)]["records"]]
    informative = [row for row in records if row["full_increment"] >= .01]
    captures = [row["head_increment"] >= .5 * row["full_increment"] for row in informative]
    confounded = [row["head_delta_only"]["auc"] <= row["residual_plus_head"]["auc"] - .05 for row in records]
    within = [row for row in records if row["within_token_head_increment"] is not None]
    large = [row for row in within if row["head_increment"] >= .03 and row["within_token_head_increment"] >= .01]
    hypotheses = {
        "S0_full_attention_adds_semantic_information": {"supported": sum(results[str(layer)]["summary"]["median_full_increment"] >= .01 for layer in config["layers"]) >= 4},
        "S1_one_head_captures_half_attention_gain": {"supported": bool(informative and np.mean(captures) >= .50), "eligible_cells": len(informative), "fraction": float(np.mean(captures)) if captures else None},
        "S2_raw_head_semantics_were_residual_confounded": {"supported": np.mean(confounded) >= .75, "fraction": float(np.mean(confounded))},
        "S3_increment_survives_current_token_control": {"supported": np.mean([row["within_token_head_increment"] >= .01 for row in within]) >= .20, "fraction": float(np.mean([row["within_token_head_increment"] >= .01 for row in within]))},
        "S4_validation_head_selection_generalizes": {"supported": np.mean([row["evaluation_head_rank"] <= 3 for row in records]) >= .50, "fraction": float(np.mean([row["evaluation_head_rank"] <= 3 for row in records]))},
        "S5_large_unique_single_head_roles_exist": {"supported": len(large) >= 10, "cells": len(large)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Incremental semantic red team for early attention heads", "", "The earlier one-head probes retained the complete mixed residual and therefore could inherit semantic information from it. Here each candidate is represented as residual-only operator coordinates plus that head's operator-coordinate delta. Heads are selected on the second half of discovery data by improvement over the residual decoder, then refitted on all discovery data and evaluated on disjoint FineWeb. Delta-only and exact-current-token controls are reported separately.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Incremental semantics](bilin18_attention_head_semantic_increment_redteam.png)", "", "| Layer | Median full-attention AUROC gain | Median selected-head gain | Median head-delta-only AUROC | Validation head top-3 rate |", "|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]["summary"]
        lines.append(f"| {layer} | {row['median_full_increment']:+.3f} | {row['median_head_increment']:+.3f} | {row['median_delta_only_auc']:.3f} | {row['top3_selection_rate']:.3f} |")
    ranked = sorted(records, key=lambda row: row["head_increment"], reverse=True)[:12]
    lines += ["", "## Largest held-out incremental roles", ""]
    for row in ranked:
        layer = next(layer for layer in config["layers"] if row in results[str(layer)]["records"])
        lines.append(f"- Layer {layer} H{row['selected_head']} `{row['feature']}`: AUROC {row['residual']['auc']:.3f} → {row['residual_plus_head']['auc']:.3f} (full {row['full']['auc']:.3f}); within-token gain {row['within_token_head_increment']:+.3f}.")
    lines += ["", "## Guardrails", "", "Incremental linear decoding is conditional evidence, not a unique causal allocation: correlated heads can make individually useful information redundant, and concatenating a head delta doubles feature dimension relative to the residual baseline. Validation-based head selection and disjoint evaluation prevent direct test leakage but do not correct for testing many feature-layer cells. A later finite head ablation must verify any specific semantic role retained here."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    layers = config["layers"]
    axes[0].plot(layers, [results[str(layer)]["summary"]["median_full_increment"] for layer in layers], marker="o", label="full attention")
    axes[0].plot(layers, [results[str(layer)]["summary"]["median_head_increment"] for layer in layers], marker="o", label="selected head")
    axes[0].axhline(0, color="black", lw=.7); axes[0].set(xlabel="Layer", ylabel="Median held-out AUROC gain", title="Information beyond residual-only state")
    axes[1].plot(layers, [results[str(layer)]["summary"]["top3_selection_rate"] for layer in layers], marker="o")
    axes[1].axhline(3/9, color="black", lw=.7, ls="--", label="chance top-3")
    axes[1].set(xlabel="Layer", ylabel="Fraction", title="Validation-selected head ranks top-3"); axes[1].legend()
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summaries": {layer: row["summary"] for layer, row in results.items()}}, indent=2))
    print(REPORT)


if __name__ == "__main__": main()
