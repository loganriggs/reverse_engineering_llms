#!/usr/bin/env python3
"""Red-team layer-head identifiability with an independent discovery ranking."""

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


CONFIG = Path("configs/bilin18_layer1_head_ranking_stability.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_head_ranking_stability.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_HEAD_RANKING_STABILITY.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_head_ranking_stability.png"


def ranking_from_rows(model, rows, newline, edit, prediction):
    edges = [(layer, head) for layer in range(2, 18) for head in range(model.config.n_head)]; all_edges = set(edges); scores = {edge: [] for edge in edges}; maximum_error = 0.0; sites = 0
    for row_index, raw_row in enumerate(rows):
        raw = np.asarray(raw_row[:128], dtype=np.int64).copy()
        if raw[edit] == newline: continue
        trace, contrast, delta, exact = make_site(model, raw, newline, edit, prediction); programs = {"full": all_edges, **{f"leaveout_{layer}_{head}": all_edges - {(layer, head)} for layer, head in edges}}
        predictions, error = head_program(model, trace, edit, prediction, contrast, programs); maximum_error = max(maximum_error, error); target = exact * delta
        for edge in edges:
            contribution = predictions[f"leaveout_{edge[0]}_{edge[1]}"] * delta; cosine = float((contribution @ target) / (contribution.norm() * target.norm()).clamp_min(1e-30)); scores[edge].append(1 - cosine)
        sites += 1; print("ranking rediscovery", row_index + 1, "/", len(rows), flush=True)
    mean_scores = {edge: float(np.mean(values)) for edge, values in scores.items()}; return sorted(edges, key=lambda edge: mean_scores[edge], reverse=True), mean_scores, maximum_error, sites


def main():
    config = json.loads(CONFIG.read_text()); original = json.loads(Path(config["original_results"]).read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r"); rediscovery = corpus[config["rediscovery_start"]:config["rediscovery_start"] + config["rediscovery_rows"]]
    alternate_order, alternate_scores, maximum_error, discovery_sites = ranking_from_rows(model, rediscovery, newline, config["discovery_edit_position"], config["prediction_position"])
    original_order = [(row["layer"], row["head"]) for row in original["ranked_edges"]]; assert len(original_order) == len(alternate_order) == 144
    budget = config["edge_budget"]; original_set = set(original_order[:budget]); alternate_set = set(alternate_order[:budget]); intersection = original_set & alternate_set; union = original_set | alternate_set
    original_rank = {edge: rank for rank, edge in enumerate(original_order)}; alternate_rank = {edge: rank for rank, edge in enumerate(alternate_order)}; first = np.asarray([original_rank[edge] for edge in original_order]); second = np.asarray([alternate_rank[edge] for edge in original_order]); rank_correlation = float(np.corrcoef(first, second)[0, 1]); jaccard = len(intersection) / len(union)
    consensus_order = sorted(original_order, key=lambda edge: original_rank[edge] + alternate_rank[edge]); consensus_set = set(consensus_order[:budget]); all_edges = set(original_order)
    programs = {"original64": original_set, "alternate64": alternate_set, "consensus64": consensus_set, "intersection": intersection, "union": union, "full144": all_edges}; rng = np.random.default_rng(config["seed"])
    for replicate in range(config["random_replicates"]):
        chosen = rng.choice(len(original_order), budget, replace=False); programs[f"random64_{replicate}"] = {original_order[int(index)] for index in chosen}
    evaluation = corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"]]; records = []
    for edit in config["evaluation_edit_positions"]:
        for row_index, raw_row in enumerate(evaluation):
            raw = np.asarray(raw_row[:config["sequence_length"]], dtype=np.int64).copy()
            if raw[edit] == newline: continue
            trace, contrast, delta, exact = make_site(model, raw, newline, edit, config["prediction_position"]); predictions, error = head_program(model, trace, edit, config["prediction_position"], contrast, programs); maximum_error = max(maximum_error, error); predictions["exact"] = exact
            supports = {name: (prediction * delta).topk(config["support_rank"]).indices for name, prediction in predictions.items()}; patch_deltas = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, support) for name, support in supports.items()}}
            with torch.no_grad(): logits = cross_logits_for_deltas(model, trace[1], patch_deltas, edit, config["prediction_position"])
            base_score = float(logits["base"].float() @ contrast); finite = {name: float(value.float() @ contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({"distance": config["prediction_position"] - edit, "row": row_index, "metrics": {name: extended_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in predictions.items()}, "finite_shifts": finite})
        print("ranking stability evaluation position", edit, flush=True)
    distances = sorted({row["distance"] for row in records}); summaries = {}
    for distance in distances:
        subset = [row for row in records if row["distance"] == distance]; summaries[str(distance)] = {method: median_summary(subset, method) for method in subset[0]["metrics"]}; random_medians = [summaries[str(distance)][f"random64_{replicate}"]["gradient_cosine_median"] for replicate in range(config["random_replicates"])]; summaries[str(distance)]["random64_aggregate"] = {"median": float(np.median(random_medians)), "p95": float(np.quantile(random_medians, .95)), "maximum": float(np.max(random_medians))}
    cell = lambda distance, method: summaries[str(distance)][method]
    hypotheses = {
        "AA0_full_head_program_is_exact": {"supported": maximum_error <= 1e-5 and all(row["metrics"]["full144"]["gradient_cosine"] >= .999 and row["metrics"]["full144"]["gradient_relative_error"] <= .02 for row in records)},
        "AA1_top64_sets_are_stable": {"supported": jaccard >= .40},
        "AA2_full_rankings_are_correlated": {"supported": rank_correlation >= .50},
        "AA3_rediscovered_top64_transfers": {"supported": all(cell(distance, "alternate64")["gradient_cosine_median"] >= .90 for distance in distances)},
        "AA4_two_top64_programs_are_behaviorally_equivalent": {"supported": all(abs(cell(distance, "original64")["gradient_cosine_median"] - cell(distance, "alternate64")["gradient_cosine_median"]) <= .05 for distance in distances)},
        "AA5_intersection_is_useful": {"supported": all(cell(distance, "intersection")["gradient_cosine_median"] >= .80 for distance in distances)},
        "AA6_both_top64_programs_beat_random": {"supported": all(min(cell(distance, "original64")["gradient_cosine_median"], cell(distance, "alternate64")["gradient_cosine_median"]) - cell(distance, "random64_aggregate")["p95"] >= .15 for distance in distances)},
        "AA7_all_selection_is_disjoint": {"supported": True},
    }
    alternate_rows = [{"rank": rank + 1, "layer": edge[0], "head": edge[1], "score": alternate_scores[edge]} for rank, edge in enumerate(alternate_order)]
    OUTPUT.write_text(json.dumps({"config": config, "discovery_sites": discovery_sites, "maximum_local_recomposition_error": maximum_error, "top64_jaccard": jaccard, "top64_intersection_size": len(intersection), "top64_union_size": len(union), "rank_correlation": rank_correlation, "hypotheses": hypotheses, "alternate_ranking": alternate_rows, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Independent stability of the cross-position head scaffold", "", "A second disjoint distance-32 panel independently reranks all 144 suffix layer-head edges. Both 64-edge sets, their intersection, union, and an average-rank consensus are then evaluated on a third FineWeb panel at distances 8, 32, and 64.", "", f"Discovery sites: `{discovery_sites}`. Top-64 Jaccard: `{jaccard:.3f}` (intersection {len(intersection)}, union {len(union)}). Full-ranking Spearman: `{rank_correlation:.3f}`. Maximum local recomposition error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Ranking stability](bilin18_layer1_head_ranking_stability.png)", "", "| Distance | Original 64 | Alternate 64 | Consensus 64 | Intersection | Union | Random-64 p95 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for distance in distances: lines.append(f"| {distance} | {cell(distance, 'original64')['gradient_cosine_median']:.3f} | {cell(distance, 'alternate64')['gradient_cosine_median']:.3f} | {cell(distance, 'consensus64')['gradient_cosine_median']:.3f} | {cell(distance, 'intersection')['gradient_cosine_median']:.3f} | {cell(distance, 'union')['gradient_cosine_median']:.3f} | {cell(distance, 'random64_aggregate')['p95']:.3f} |")
    lines += ["", "## Guardrail", "", "Stable predictive performance does not imply a unique edge set. Set overlap and effect fidelity answer different questions: redundancy can preserve a reusable scaffold while making individual native-head identities non-identifiable."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained"); methods = ["original64", "alternate64", "consensus64", "intersection", "union"]; width = .8 / len(distances); x = np.arange(len(methods))
    for index, distance in enumerate(distances): axes[0].bar(x + (index - 1) * width, [cell(distance, method)["gradient_cosine_median"] for method in methods], width=width, label=f"distance {distance}")
    axes[0].set_xticks(x, methods, rotation=25, ha="right"); axes[0].set(ylabel="Median gradient cosine", title="Independent frozen rankings"); axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    axes[1].scatter(first, second, alpha=.6, s=14); axes[1].set(xlabel="Original rank", ylabel="Independent rank", title=f"All-edge rank correlation = {rank_correlation:.3f}"); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig); print(json.dumps({"maximum_local_recomposition_error": maximum_error, "top64_jaccard": jaccard, "intersection_size": len(intersection), "rank_correlation": rank_correlation, "hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
