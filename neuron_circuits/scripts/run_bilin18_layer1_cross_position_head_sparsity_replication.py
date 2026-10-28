#!/usr/bin/env python3
"""Corrected held-out replication of the 144-edge head sparsity frontier."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import selected_delta
from run_bilin18_layer1_cross_position_head_sparsity import head_program, make_site, median_summary
from run_bilin18_layer1_cross_position_role_redteam import cross_logits_for_deltas, extended_metrics
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_cross_position_head_sparsity_replication.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_cross_position_head_sparsity_replication.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_CROSS_POSITION_HEAD_SPARSITY_REPLICATION.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_cross_position_head_sparsity_replication.png"


def first_threshold(summaries, distance, ranks, threshold):
    for rank in ranks:
        if summaries[str(distance)][f"top_{rank}"]["gradient_cosine_median"] >= threshold:
            return rank
    return None


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source_results"]).read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    ordering = [(row["layer"], row["head"]) for row in source["ranked_edges"]]; assert len(ordering) == 144 and len(set(ordering)) == 144
    programs = {f"top_{rank}": set(ordering[:rank]) for rank in config["head_edge_ranks"]}; rng = np.random.default_rng(config["seed"])
    for rank in config["random_ranks"]:
        for replicate in range(config["random_replicates"]):
            chosen = rng.choice(len(ordering), rank, replace=False); programs[f"random_{rank}_{replicate}"] = {ordering[int(index)] for index in chosen}
    corpus = np.load(config["corpus"], mmap_mode="r"); rows = corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"]]; records = []; maximum_local_error = 0.0
    for edit in config["evaluation_edit_positions"]:
        for row_index, raw_row in enumerate(rows):
            raw = np.asarray(raw_row[:config["sequence_length"]], dtype=np.int64).copy()
            if raw[edit] == newline: continue
            trace, contrast, delta, exact = make_site(model, raw, newline, edit, config["prediction_position"]); predictions, error = head_program(model, trace, edit, config["prediction_position"], contrast, programs); maximum_local_error = max(maximum_local_error, error); predictions["exact"] = exact
            supports = {name: (prediction * delta).topk(config["support_rank"]).indices for name, prediction in predictions.items()}; patch_deltas = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, support) for name, support in supports.items()}}
            with torch.no_grad(): logits = cross_logits_for_deltas(model, trace[1], patch_deltas, edit, config["prediction_position"])
            base_score = float(logits["base"].float() @ contrast); finite = {name: float(value.float() @ contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({"distance": config["prediction_position"] - edit, "row": row_index, "metrics": {name: extended_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in predictions.items()}, "finite_shifts": finite})
        print("corrected head replication position", edit, flush=True)
    distances = sorted({row["distance"] for row in records}); summaries = {}
    for distance in distances:
        subset = [row for row in records if row["distance"] == distance]; summaries[str(distance)] = {method: median_summary(subset, method) for method in subset[0]["metrics"]}
        for rank in config["random_ranks"]:
            medians = [summaries[str(distance)][f"random_{rank}_{replicate}"]["gradient_cosine_median"] for replicate in range(config["random_replicates"])]
            summaries[str(distance)][f"random_{rank}_aggregate"] = {"median": float(np.median(medians)), "p95": float(np.quantile(medians, .95)), "maximum": float(np.max(medians))}
    cell = lambda distance, method: summaries[str(distance)][method]; thresholds = {str(distance): first_threshold(summaries, distance, config["head_edge_ranks"], .90) for distance in distances}
    hypotheses = {
        "Z0_all_144_edges_are_exact": {"supported": maximum_local_error <= 1e-5 and all(row["metrics"]["top_144"]["gradient_cosine"] >= .999 and row["metrics"]["top_144"]["gradient_relative_error"] <= .02 for row in records)},
        "Z1_frozen_top72_replication": {"supported": all(cell(distance, "top_72")["gradient_cosine_median"] >= .90 and float(np.quantile([row["metrics"]["top_72"]["gradient_cosine"] for row in records if row["distance"] == distance], .1)) >= .75 for distance in distances)},
        "Z2_frozen_top48_is_useful": {"supported": all(cell(distance, "top_48")["gradient_cosine_median"] >= .75 for distance in distances)},
        "Z3_top32_remains_insufficient": {"supported": sum(cell(distance, "top_32")["gradient_cosine_median"] < .80 for distance in distances) >= 2},
        "Z4_top72_beats_random": {"supported": all(cell(distance, "top_72")["gradient_cosine_median"] - cell(distance, "random_72_aggregate")["p95"] >= .15 for distance in distances)},
        "Z5_half_the_edges_are_sufficient": {"supported": all(thresholds[str(distance)] is not None and thresholds[str(distance)] <= 72 for distance in distances)},
        "Z6_top72_is_causally_directional": {"supported": all(cell(distance, "top_72")["finite_directional_fraction"] >= .90 for distance in distances)},
        "Z7_ordering_is_frozen": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": config, "maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "first_rank_at_0.90": thresholds, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Corrected 144-edge head-sparsity replication", "", "The earlier assay incorrectly treated 72 edges as the complete 16-block × 9-head suffix. This correction preserves its discovery-frozen ordering, evaluates all 144 edges for the exactness gate, and uses entirely new FineWeb rows for the compactness frontier.", "", f"Maximum local head recomposition error: `{maximum_local_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Corrected head sparsity](bilin18_layer1_cross_position_head_sparsity_replication.png)", "", "| Distance | First rank at cosine 0.90 | Top-32 | Top-48 | Top-72 | Top-144 | Random-72 p95 | Top-72 finite direction |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for distance in distances: lines.append(f"| {distance} | {thresholds[str(distance)] or '—'} | {cell(distance, 'top_32')['gradient_cosine_median']:.3f} | {cell(distance, 'top_48')['gradient_cosine_median']:.3f} | {cell(distance, 'top_72')['gradient_cosine_median']:.3f} | {cell(distance, 'top_144')['gradient_cosine_median']:.3f} | {cell(distance, 'random_72_aggregate')['p95']:.3f} | {cell(distance, 'top_72')['finite_directional_fraction']:.3f} |")
    lines += ["", "## Guardrail", "", "A 72-edge program is half of the available layer-head graph, not a tiny sparse circuit. Its strong advantage over random establishes structured concentration, while its size argues that native heads are still too coarse to be the final explanatory units."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained"); ranks = config["head_edge_ranks"]
    for distance in distances: axes[0].plot(ranks, [cell(distance, f"top_{rank}")["gradient_cosine_median"] for rank in ranks], marker="o", label=f"distance {distance}")
    axes[0].axhline(.9, color="gray", linestyle=":"); axes[0].set(xlabel="Frozen layer-head edges", ylabel="Median gradient cosine", title="Corrected 144-edge frontier"); axes[0].legend(); axes[0].grid(alpha=.2)
    for distance in distances: axes[1].bar(str(distance), cell(distance, "top_72")["gradient_cosine_median"] - cell(distance, "random_72_aggregate")["p95"])
    axes[1].axhline(.15, color="gray", linestyle=":"); axes[1].set(xlabel="Source-to-output distance", ylabel="Top-72 minus random p95", title="Frozen structure beyond edge count"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig); print(json.dumps({"maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "first_rank_at_0.90": thresholds, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
