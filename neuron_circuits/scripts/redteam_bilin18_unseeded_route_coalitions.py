#!/usr/bin/env python3
"""Distinguish discrete route operators from stable quantization of a continuum."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unseeded_route_coalition_geometry import best_center_matching
from run_bilin18_unsupervised_weight_circuit_atlas import spherical_kmeans


CONFIG = Path("configs/bilin18_unseeded_route_coalition_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_unseeded_route_coalition_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSEEDED_ROUTE_COALITION_REDTEAM.md"


def maximum_off_diagonal_cosine(centers):
    gram = np.asarray(centers) @ np.asarray(centers).T
    np.fill_diagonal(gram, -np.inf)
    return float(gram.max())


def cluster_replication_fidelity(values, centers):
    values = np.asarray(values, dtype=np.float64); centers = np.asarray(centers, dtype=np.float64)
    assignment = (values @ centers.T).argmax(1)
    reconstruction = centers[assignment]
    return float(1 - np.square(values - reconstruction).sum() / max(np.square(values).sum(), 1e-30))


def subspace_replication_fidelity(discovery, replication, rank):
    _, _, vh = np.linalg.svd(np.asarray(discovery, dtype=np.float64), full_matrices=False)
    basis = vh[:rank]
    values = np.asarray(replication, dtype=np.float64)
    projection = (values @ basis.T) @ basis
    return float(1 - np.square(values - projection).sum() / max(np.square(values).sum(), 1e-30))


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text()); arrays = np.load(config["arrays"])
    layers = len(source["layer_summary"]); sweep = {}; comparisons = {}
    for layer in range(layers):
        discovery = arrays[f"discovery_routes_L{layer}"]; replication = arrays[f"replication_routes_L{layer}"]
        sweep[str(layer)] = {}
        fitted = {}
        for count in config["cluster_counts"]:
            first_assignment, first_centers, first_own, first_margin = spherical_kmeans(
                torch.as_tensor(discovery), count, config["kmeans_restarts"], config["kmeans_iterations"], config["seed"] + 1009 * layer + count,
            )
            second_assignment, second_centers, second_own, second_margin = spherical_kmeans(
                torch.as_tensor(replication), count, config["kmeans_restarts"], config["kmeans_iterations"], config["seed"] + 100000 + 1009 * layer + count,
            )
            matching = best_center_matching(first_centers.numpy(), second_centers.numpy())
            row = {
                "matched_center_cosine": matching["mean_cosine"],
                "discovery_own_cosine": float(first_own.mean()), "replication_own_cosine": float(second_own.mean()),
                "discovery_margin": float(first_margin.mean()), "replication_margin": float(second_margin.mean()),
                "maximum_center_cosine": maximum_off_diagonal_cosine(first_centers.numpy()),
            }
            sweep[str(layer)][str(count)] = row; fitted[count] = (first_centers.numpy(), row)
        centers4, row4 = fitted[4]
        cluster_fidelity = cluster_replication_fidelity(replication, centers4)
        subspace_fidelity = subspace_replication_fidelity(discovery, replication, 4)
        comparisons[str(layer)] = {
            "cluster_fidelity": cluster_fidelity, "rank4_subspace_fidelity": subspace_fidelity,
            "cluster_minus_subspace": cluster_fidelity - subspace_fidelity,
        }
        print(f"coalition redteam L{layer} k4cos={row4['matched_center_cosine']:.3f} margin={min(row4['discovery_margin'],row4['replication_margin']):.3f} cluster={cluster_fidelity:.3f} pca={subspace_fidelity:.3f}", flush=True)
    privileged = []
    separated = []
    for layer in range(layers):
        row = sweep[str(layer)]; k4 = row["4"]
        if k4["matched_center_cosine"] >= row["3"]["matched_center_cosine"] + .05 and k4["matched_center_cosine"] >= row["6"]["matched_center_cosine"] + .05:
            privileged.append(layer)
        if k4["maximum_center_cosine"] <= config["maximum_center_cosine"] and min(k4["discovery_margin"], k4["replication_margin"]) >= config["minimum_margin"]:
            separated.append(layer)
    high_margin_candidates = [
        f"L{row['layer']}C{row['cluster']}" for row in source["selected_candidates"]
        if min(row["discovery_margin"], row["replication_margin"]) >= config["minimum_margin"]
    ]
    cluster_wins = [int(layer) for layer, row in comparisons.items() if row["cluster_minus_subspace"] > 0]
    continuous_layers = sorted(set(range(layers)) - set(separated))
    hypotheses = {
        "R0_k4_is_a_privileged_discrete_vocabulary": {"supported": len(privileged) >= 8, "layers": privileged},
        "R1_k4_centers_are_well_separated": {"supported": len(separated) >= 8, "layers": separated},
        "R2_high_margin_candidates_survive": {"supported": len(high_margin_candidates) >= 8, "candidates": high_margin_candidates},
        "R3_clusters_beat_parameter_matched_continuous_subspaces": {"supported": len(cluster_wins) >= 12, "layers": cluster_wins, "mean_cluster_minus_subspace": float(np.mean([row["cluster_minus_subspace"] for row in comparisons.values()]))},
        "R4_continuous_geometry_is_depth_dependent": {"supported": bool(continuous_layers) and bool(separated), "high_margin_layers": separated, "continuous_or_ambiguous_layers": continuous_layers},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "sweep": sweep, "continuous_comparison": comparisons}, indent=2, allow_nan=False))
    lines = ["# Unseeded route-coalition discreteness red team", "", "Stable k=4 centroids can be either discrete operators or a reproducible quantization of continuous anisotropic route geometry. This audit sweeps k, measures assignment margins and center separation, and compares frozen centroid reconstruction with a parameter-matched rank-4 linear subspace on replication routes.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | k=3/4/6 stability | k4 margin D/R | max center cosine | centroid fidelity | rank-4 subspace |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in range(layers):
        row = sweep[str(layer)]; k4 = row["4"]; comparison = comparisons[str(layer)]
        lines.append(f"| {layer} | {row['3']['matched_center_cosine']:.3f}/{k4['matched_center_cosine']:.3f}/{row['6']['matched_center_cosine']:.3f} | {k4['discovery_margin']:.3f}/{k4['replication_margin']:.3f} | {k4['maximum_center_cosine']:.3f} | {comparison['cluster_fidelity']:.3f} | {comparison['rank4_subspace_fidelity']:.3f} |")
    lines += ["", "## Interpretation", "", "Stable centers establish reproducible route density, not automatically a discrete operator vocabulary. High-margin, well-separated layers support coalition-like states; low-margin layers are better described as continuous cones or manifolds even when any fixed k replicates. The rank-4 comparison is parameter-matched for basis storage but not runtime code: centroids use a discrete index, whereas the subspace uses continuous coordinates."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
