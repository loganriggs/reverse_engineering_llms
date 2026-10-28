#!/usr/bin/env python3
"""Use exact cooperative energy allocations to red-team channel sparsity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_conditional_subspace_quantization import grassmann_similarity, state_centroids
from run_bilin18_layer8_native_channel_program import (
    channel_target,
    coefficients,
    effective_support,
    first_rank,
    oracle_prediction,
    ordered_curve,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_channel_shapley_redteam.json")
SOURCE = TENSOR_ROOT / "layer8_multimetric_quantization.npz"
BASE_RESULT = TENSOR_ROOT / "layer8_native_channel_program.json"
OUTPUT = TENSOR_ROOT / "layer8_channel_shapley_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_CHANNEL_SHAPLEY_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer8_channel_shapley_redteam.png"


def shapley_scores(coefficients, down, target):
    """C_h dot sum_k C_k: exact Shapley allocation for squared output norm."""
    # Force the small contraction order. A three-operand einsum may otherwise
    # materialize an O(N*A*H*O) intermediate (tens of GiB at model width).
    projected_target = (target.reshape(-1, target.shape[-1]) @ down).reshape_as(coefficients)
    return (coefficients * projected_target).sum(1)


def cosine(first, second):
    return float(first @ second / (first.norm() * second.norm()).clamp_min(1e-30))


def signed_mass_rank(scores, fraction=.9):
    ordered = scores.sort(descending=True).values
    total = scores.sum()
    if total <= 0:
        return None
    hits = torch.nonzero(ordered.cumsum(0) >= fraction * total)
    return int(hits[0]) + 1 if len(hits) else None


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE)
    base = json.loads(BASE_RESULT.read_text())
    active = [int(value) for value in base["active_prototypes"]]
    basis = torch.from_numpy(source["basis"]).cuda().float()
    axes = torch.from_numpy(source["axes_direct_mlp"]).cuda().float()
    prototypes = torch.from_numpy(source["prototypes_direct_mlp"]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    discovery_states, _ = collect(model, 8, config["prototype_discovery"], {**config, "positions": [config["causal_position"]]})
    discovery_features = discovery_states @ basis
    discovery_features /= discovery_features.norm(dim=1, keepdim=True).clamp_min(1e-30)
    discovery_labels = grassmann_similarity(axes, prototypes).argmax(1)
    centroids = state_centroids(discovery_features, discovery_labels, config["clusters"])
    states, _ = collect(model, 8, config["natural_states"], {**config, "positions": [config["causal_position"]]})
    features = states @ basis
    features /= features.norm(dim=1, keepdim=True).clamp_min(1e-30)
    labels = (features @ centroids.T).argmax(1)
    row_ids = torch.arange(len(labels), device="cuda")
    train_mask = row_ids % 2 == 0; test_mask = ~train_mask
    block = model.blocks[8]
    down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
    down_energy = down.square().sum(0)
    results = {}
    for cluster in active:
        train_x = states[(labels == cluster) & train_mask]
        test_x = states[(labels == cluster) & test_mask]
        print("channel Shapley", cluster, len(train_x), len(test_x), flush=True)
        edit = basis @ prototypes[cluster]
        train_coefficients = coefficients(left, right, edit, train_x)
        test_coefficients = coefficients(left, right, edit, test_x)
        train_target = channel_target(train_coefficients, down)
        test_target = channel_target(test_coefficients, down)
        train_site_scores = shapley_scores(train_coefficients, down, train_target)
        test_site_scores = shapley_scores(test_coefficients, down, test_target)
        train_scores = train_site_scores.sum(0)
        test_scores = test_site_scores.sum(0)
        shapley_order = train_scores.argsort(descending=True)
        absolute_order = train_scores.abs().argsort(descending=True)
        energy_order = (train_coefficients.square().sum((0, 1)) * down_energy).argsort(descending=True)
        shapley_rows = ordered_curve(test_coefficients, down, test_target, shapley_order, config["static_ranks"])
        absolute_rows = ordered_curve(test_coefficients, down, test_target, absolute_order, config["static_ranks"])
        energy_rows = ordered_curve(test_coefficients, down, test_target, energy_order, config["static_ranks"])
        oracle_order = test_site_scores.argsort(1, descending=True)
        oracle_rows = []
        for rank in config["oracle_ranks"]:
            prediction = oracle_prediction(test_coefficients, down, oracle_order, rank, config["batch_size"])
            from run_bilin18_layer8_restricted_tensor_program import prediction_r2
            oracle_rows.append({"rank": rank, "r2": prediction_r2(test_target, prediction)})
        total_energy = test_target.square().sum()
        allocation_error = float((test_scores.sum() - total_energy).abs() / total_energy.clamp_min(1e-30))
        absolute_support = effective_support(test_site_scores.abs())
        negative_mass = -test_scores.clamp_max(0).sum() / total_energy.clamp_min(1e-30)
        results[str(cluster)] = {
            "train_sites": int(len(train_x)), "test_sites": int(len(test_x)),
            "allocation_relative_error": allocation_error,
            "train_test_score_cosine": cosine(train_scores, test_scores),
            "negative_allocation_mass": float(negative_mass),
            "signed_mass_rank90": signed_mass_rank(test_scores),
            "median_absolute_shapley_support": float(absolute_support.median()),
            "shapley_rows": shapley_rows, "shapley_rank90": first_rank(shapley_rows),
            "absolute_rows": absolute_rows, "absolute_rank90": first_rank(absolute_rows),
            "energy_rows": energy_rows, "energy_rank90": first_rank(energy_rows),
            "oracle_rows": oracle_rows, "oracle_rank90": first_rank(oracle_rows),
        }
    shapley256 = [next(item for item in row["shapley_rows"] if item["rank"] == 256)["r2"] for row in results.values()]
    energy256 = [next(item for item in row["energy_rows"] if item["rank"] == 256)["r2"] for row in results.values()]
    shapley512 = [next(item for item in row["shapley_rows"] if item["rank"] == 512)["r2"] for row in results.values()]
    oracle256 = [next(item for item in row["oracle_rows"] if item["rank"] == 256)["r2"] for row in results.values()]
    ranks90 = [row["shapley_rank90"] for row in results.values()]
    observed = [rank for rank in ranks90 if rank is not None]
    hypotheses = {
        "S1_shapley_allocation_is_exact": {"supported": max(row["allocation_relative_error"] for row in results.values()) <= 1e-5, "maximum_relative_error": max(row["allocation_relative_error"] for row in results.values())},
        "S2_shapley_ranking_improves_static": {"supported": all(first > second for first, second in zip(shapley256, energy256)), "median_gain": float(np.median(np.asarray(shapley256) - np.asarray(energy256)))},
        "S3_static_shapley512_is_sufficient": {"supported": sum(value >= .9 for value in shapley512) >= len(shapley512) / 2, "passing": sum(value >= .9 for value in shapley512)},
        "S4_oracle_shapley256_is_sufficient": {"supported": sum(value >= .9 for value in oracle256) >= len(oracle256) / 2, "passing": sum(value >= .9 for value in oracle256)},
        "S5_cooperative_ranking_transfers": {"supported": all(row["train_test_score_cosine"] >= .8 for row in results.values()), "minimum_cosine": min(row["train_test_score_cosine"] for row in results.values())},
        "S6_shapley_support_is_sparse": {"supported": float(np.median([row["median_absolute_shapley_support"] for row in results.values()])) <= 512, "median_support": float(np.median([row["median_absolute_shapley_support"] for row in results.values()]))},
        "S7_sufficiency_remains_broad": {"supported": len(observed) < len(ranks90) or float(np.median(observed)) > 512, "ranks90": ranks90},
    }
    OUTPUT.write_text(json.dumps({"config": config, "active_prototypes": active, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Cooperative-attribution red-team for Layer-8 native channels", "",
        "For the exact channel contributions `C_h`, the squared-write cooperative game has value `v(S)=||sum_{h in S} C_h||²`. Its exact Shapley allocation is `<C_h, sum_k C_k>`: self energy plus each pair interaction split equally. These signed allocations sum exactly to total derivative energy. Rankings are learned on training contexts and tested on disjoint contexts; per-state rankings remain oracle diagnostics.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Channel Shapley red-team](layer8_channel_shapley_redteam.png)", "", "| Address | Shapley/energy K90 | Shapley/energy R² @256 | Oracle Shapley @256 | Train/test score cosine | Abs. Shapley support | Negative mass |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for cluster, row in results.items():
        shapley = {item["rank"]: item["r2"] for item in row["shapley_rows"]}
        energy = {item["rank"]: item["r2"] for item in row["energy_rows"]}
        oracle = {item["rank"]: item["r2"] for item in row["oracle_rows"]}
        lines.append(f"| {cluster} | {row['shapley_rank90']} / {row['energy_rank90']} | {shapley[256]:.3f} / {energy[256]:.3f} | {oracle[256]:.3f} | {row['train_test_score_cosine']:.3f} | {row['median_absolute_shapley_support']:.1f} | {row['negative_allocation_mass']:.3f} |")
    lines += ["", "Isolated channel energy is a necessity-like statistic; Shapley allocation includes cooperation. Neither is itself a sparse-sufficiency optimizer. Failure of both fixed and per-state Shapley prefixes therefore gives stronger evidence for a broad cooperative ensemble than the energy-ranked result alone."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.2))
    for cluster, row in results.items():
        plots[0].plot([item["rank"] for item in row["shapley_rows"]], [item["r2"] for item in row["shapley_rows"]], marker="o", label=cluster)
        plots[1].plot([item["rank"] for item in row["oracle_rows"]], [item["r2"] for item in row["oracle_rows"]], marker="o", label=cluster)
    plots[2].scatter([row["median_absolute_shapley_support"] for row in results.values()], [row["negative_allocation_mass"] for row in results.values()])
    plots[0].set_title("Fixed Shapley ranking"); plots[1].set_title("Per-state Shapley oracle"); plots[2].set_title("Attribution support and conflict")
    for plot in plots[:2]:
        plot.set_xscale("log", base=2); plot.set_ylim(-.1, 1.05); plot.set_xlabel("channels"); plot.set_ylabel("held-out derivative R²"); plot.grid(alpha=.2)
    plots[2].set_xlabel("absolute-Shapley effective support"); plots[2].set_ylabel("negative allocation / total"); plots[2].grid(alpha=.2)
    plots[0].legend(fontsize=7, ncol=2)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
