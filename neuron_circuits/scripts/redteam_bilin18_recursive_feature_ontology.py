#!/usr/bin/env python3
"""Test recursive semantic generalization of the 50-feature MLP ontology."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_known_ontology_residual_atlas import (
    collect_panel, design_matrices, match_axes, multivariate_r2, pca, project,
    ridge_fit, subspace_overlap,
)
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_stratified_natural_route_census import token_class_from_flags


CONFIG = Path("configs/bilin18_recursive_feature_ontology_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_recursive_feature_ontology_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_RECURSIVE_FEATURE_ONTOLOGY_REDTEAM.md"


CLASSES = ["other", "continuation", "capitalized", "function", "numeric", "specialized", "punctuation", "eot", "byte_fragment"]


def vocabulary_classes(tokenizer, flags):
    result = token_class_from_flags(flags).astype("U16"); texts = np.asarray([tokenizer.decode([index]) for index in range(tokenizer.vocab_size)])
    result[np.asarray(["\ufffd" in text for text in texts])] = "byte_fragment"
    if tokenizer.eos_token_id is not None: result[tokenizer.eos_token_id] = "eot"
    return result


def interaction_design(metadata, classes):
    current = classes[metadata["current_id"]]; following = classes[metadata["next_id"]]
    first = np.stack([current == name for name in CLASSES], axis=1).astype(np.float64); second = np.stack([following == name for name in CLASSES], axis=1).astype(np.float64)
    interactions = (first[:, :, None] * second[:, None, :]).reshape(len(first), -1)
    names = [f"{left}->{right}" for left in CLASSES for right in CLASSES]
    return interactions, names


def best_interaction(scores, design, names):
    rows = []
    for index, name in enumerate(names):
        labels = design[:, index]
        if labels.sum() < 10 or labels.sum() > len(labels) - 10: continue
        value = float(auc(labels, scores)); rows.append({"feature": name, "auroc": max(value, 1 - value), "positive_pole": value >= .5})
    return max(rows, key=lambda row: row["auroc"])


def fit_residual(train_design, train_y, test_design, test_y, train_groups, alphas, rank):
    train_prediction, test_prediction, alpha, validation = ridge_fit(train_design, train_y, test_design, alphas, train_groups)
    train_residual = train_y - train_prediction; test_residual = test_y - test_prediction
    train_mean, train_basis, train_eigen = pca(train_residual, rank); test_mean, test_basis, test_eigen = pca(test_residual, rank)
    combined = test_prediction + project(test_residual, train_mean, train_basis)
    return {
        "train_prediction": train_prediction, "test_prediction": test_prediction,
        "train_residual": train_residual, "test_residual": test_residual,
        "train_mean": train_mean, "train_basis": train_basis, "test_basis": test_basis,
        "train_eigen": train_eigen, "test_eigen": test_eigen,
        "alpha": alpha, "validation_r2": validation,
        "prediction_r2": multivariate_r2(test_y, test_prediction, train_y.mean(0)),
        "combined_r2": multivariate_r2(test_y, combined, train_y.mean(0)),
        "overlap": subspace_overlap(train_basis, test_basis),
    }


def main():
    config = json.loads(CONFIG.read_text()); previous = json.loads(Path(config["source"]).read_text()); previous_arrays = np.load(config["source_arrays"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); classes = vocabulary_classes(tokenizer, flags)
    corpus = np.load(config["corpus"], mmap_mode="r"); model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    panels = {}
    for name, offset in (("discovery", config["discovery_start"]), ("third", config["third_start"])):
        values, metadata = collect_panel(model, corpus[offset:offset + config["rows_per_panel"]], offset, config["positions"], config["layers"], config["batch_size"]); base = design_matrices(metadata, flags, config["positions"])["declared"]; interaction, names = interaction_design(metadata, classes); panels[name] = {"values": values, "metadata": metadata, "base": base, "interaction": interaction, "extended": np.concatenate((base, interaction), axis=1)}
    results = {}; stable_unexplained = 0; prior_axis_rows = []
    for layer in config["layers"]:
        discovery = panels["discovery"]["values"][layer]; third = panels["third"]["values"][layer]; groups = panels["discovery"]["metadata"]["global_row"]
        base = fit_residual(panels["discovery"]["base"], discovery, panels["third"]["base"], third, groups, config["ridge_alphas"], config["residual_rank"])
        extended = fit_residual(panels["discovery"]["extended"], discovery, panels["third"]["extended"], third, groups, config["ridge_alphas"], config["residual_rank"])
        permutation, cosines = match_axes(extended["train_basis"], extended["test_basis"]); axis_rows = []
        for axis in range(config["residual_rank"]):
            scores = (extended["test_residual"] - extended["train_mean"]) @ extended["train_basis"][axis]; feature = best_interaction(scores, panels["third"]["interaction"], names); stable = abs(cosines[axis]) >= config["minimum_axis_cosine"]
            stable_unexplained += int(stable and feature["auroc"] < .75); axis_rows.append({"axis": axis, "axis_cosine": float(cosines[axis]), "replication_axis": int(permutation[axis]), "best_interaction": feature})
        old_basis = previous_arrays[f"residual_basis_L{layer}"]; old_mean, _, _ = pca(base["train_residual"], len(old_basis)); old_scores = base["test_residual"] - old_mean
        for candidate in previous["selected_candidates"]:
            if candidate["layer"] != layer: continue
            scores = old_scores @ old_basis[candidate["axis"]]; prior_axis_rows.append({"layer": layer, "axis": candidate["axis"], "best_interaction": best_interaction(scores, panels["third"]["interaction"], names)})
        old_top5 = float(np.sum(base["train_eigen"][:5]) / max(np.sum(base["train_eigen"]), 1e-30)); new_top5 = float(np.sum(extended["train_eigen"][:5]) / max(np.sum(extended["train_eigen"]), 1e-30))
        results[str(layer)] = {"base_prediction_r2": base["prediction_r2"], "extended_prediction_r2": extended["prediction_r2"], "improvement": extended["prediction_r2"] - base["prediction_r2"], "base_residual_overlap": base["overlap"], "extended_residual_overlap": extended["overlap"], "extended_combined_r2": extended["combined_r2"], "base_top5_residual_share": old_top5, "extended_top5_residual_share": new_top5, "top5_share_change": new_top5 - old_top5, "extended_alpha": extended["alpha"], "axes": axis_rows}
        print(f"recursive feature ontology L{layer}", flush=True)
    improvements = sum(row["improvement"] >= .03 for row in results.values()); prior_explained = sum(row["best_interaction"]["auroc"] >= .90 for row in prior_axis_rows); recurring = sum(row["extended_residual_overlap"] >= .70 for row in results.values())
    hypotheses = {
        "R0_third_panel_is_fresh": {"supported": config["third_start"] >= previous["config"]["replication_start"] + previous["config"]["rows_per_panel"], "range": [config["third_start"], config["third_start"] + config["rows_per_panel"] - 1]},
        "R1_recursive_interactions_improve_write_explanation": {"supported": improvements >= 2, "passing": improvements},
        "R2_recursive_features_explain_prior_residual_axes": {"supported": prior_explained >= 3, "passing": prior_explained, "axes": prior_axis_rows},
        "R3_residual_geometry_survives_recursive_expansion": {"supported": recurring >= 3, "passing": recurring},
        "R4_recursive_expansion_reduces_late_residual_concentration": {"supported": results["17"]["top5_share_change"] <= -.10, "change": results["17"]["top5_share_change"]},
        "R5_late_compact_residual_code_survives": {"supported": results["17"]["extended_combined_r2"] >= .95, "r2": results["17"]["extended_combined_r2"]},
        "R6_stable_unexplained_axes_remain": {"supported": stable_unexplained >= 2, "count": stable_unexplained},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Recursive feature-ontology red team", "", "The strongest residual examples from the primitive fifty-feature model suggested conjunctions rather than wholly unrelated concepts. Before viewing a third panel, nine mutually exclusive current-token classes and nine next-token classes were frozen; their 81 pairwise interactions were added to the original current/next feature and position design.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | Base/extended write R2 | Gain | Base/extended residual overlap | Extended known+rank16 R2 | Top-five residual share base/extended |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; lines.append(f"| {layer} | {row['base_prediction_r2']:.3f}/{row['extended_prediction_r2']:.3f} | {row['improvement']:+.3f} | {row['base_residual_overlap']:.3f}/{row['extended_residual_overlap']:.3f} | {row['extended_combined_r2']:.3f} | {row['base_top5_residual_share']:.3f}/{row['extended_top5_residual_share']:.3f} |")
    lines += ["", "## Prior frozen axes on the third panel", "", "| Axis | Best recursive interaction | AUROC |", "|---|---|---:|"]
    for row in prior_axis_rows: lines.append(f"| L{row['layer']}A{row['axis']} | {row['best_interaction']['feature']} | {row['best_interaction']['auroc']:.3f} |")
    lines += ["", "## Guardrail", "", "The interaction vocabulary is a prospective test of one recursive generalization prompted by earlier examples; it is not a license to add arbitrary labels until every axis is named. An interaction can explain axis membership without proving that the axis is causally dedicated to it. Stable residual geometry after expansion is evidence that the ontology remains incomplete."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
