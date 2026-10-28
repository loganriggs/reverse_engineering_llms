#!/usr/bin/env python3
"""Full-space red team for the rank-64 task-blind MLP atlas."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_taskblind_mlp_multiview_atlas import (
    canonical_channel_scale, frozen_cluster_fidelity, frozen_subspace_fidelity,
    match_centers,
)
from run_bilin18_unsupervised_weight_circuit_atlas import adjusted_rand


CONFIG = Path("configs/bilin18_taskblind_mlp_fullspace_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_taskblind_mlp_fullspace_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_TASKBLIND_MLP_FULLSPACE_REDTEAM.md"


def centered_unit(discovery: np.ndarray, replication: np.ndarray):
    mean = np.asarray(discovery, dtype=np.float32).mean(0)
    first = np.asarray(discovery, dtype=np.float32) - mean
    second = np.asarray(replication, dtype=np.float32) - mean
    first /= np.linalg.norm(first, axis=1, keepdims=True).clip(1e-30)
    second /= np.linalg.norm(second, axis=1, keepdims=True).clip(1e-30)
    return first, second


def spherical_kmeans_cuda(values: np.ndarray, clusters: int, restarts: int, iterations: int, seed: int):
    values = F.normalize(torch.as_tensor(values, device="cuda", dtype=torch.float32), dim=1)
    best = None
    for restart in range(restarts):
        generator = torch.Generator(device="cuda").manual_seed(seed + 1009 * restart)
        first = int(torch.randint(len(values), (), generator=generator, device="cuda"))
        chosen = [first]; nearest = 1 - values @ values[first]
        for _ in range(1, clusters):
            probability = nearest.clamp_min(0).square()
            candidate = int(torch.multinomial(probability, 1, generator=generator)) if float(probability.sum()) > 0 else int(torch.randint(len(values), (), generator=generator, device="cuda"))
            chosen.append(candidate); nearest = torch.minimum(nearest, 1 - values @ values[candidate])
        centers = values[chosen].clone(); assignment = None
        for _ in range(iterations):
            score = values @ centers.T; updated = score.argmax(1)
            if assignment is not None and torch.equal(assignment, updated): break
            assignment = updated; sums = torch.zeros_like(centers); sums.index_add_(0, assignment, values)
            empty = sums.norm(dim=1) < 1e-12
            if empty.any(): sums[empty] = values[score.max(1).values.argsort()[:int(empty.sum())]]
            centers = F.normalize(sums, dim=1)
        objective = float((values * centers[assignment]).sum())
        if best is None or objective > best[0]: best = objective, assignment.clone(), centers.clone()
    _, assignment, centers = best
    score = values @ centers.T; own = score.gather(1, assignment[:, None]).squeeze(1)
    masked = score.clone(); masked[torch.arange(len(values), device="cuda"), assignment] = -torch.inf
    return assignment.cpu().numpy(), centers.cpu().numpy(), own.cpu().numpy(), (own - masked.max(1).values).cpu().numpy()


def fullspace_fit(discovery, replication, config, seed):
    first, second = centered_unit(discovery, replication)
    da, dc, down, dmargin = spherical_kmeans_cuda(first, config["clusters"], config["kmeans_restarts"], config["kmeans_iterations"], seed)
    ra, rc, rown, rmargin = spherical_kmeans_cuda(second, config["clusters"], config["kmeans_restarts"], config["kmeans_iterations"], seed + 100000)
    permutation, cosines = match_centers(dc, rc)
    rng = np.random.default_rng(seed + 999); null = []
    for _ in range(config["null_replicates"]):
        _, values = match_centers(dc, rc[:, rng.permutation(rc.shape[1])]); null.append(float(np.mean(values)))
    frozen = (second @ dc.T).argmax(1)
    sizes = [{"discovery": int(np.sum(da == cluster)), "replication": int(np.sum(frozen == cluster)), "center_cosine": float(cosines[cluster])} for cluster in range(config["clusters"])]
    return {
        "matched_center_cosine": float(np.mean(cosines)), "center_cosines": cosines.tolist(), "null_p95": float(np.quantile(null, .95)),
        "discovery_margin": float(np.mean(dmargin)), "replication_margin": float(np.mean(rmargin)),
        "cluster_fidelity": frozen_cluster_fidelity(second, dc),
        "rank_matched_subspace_fidelity": frozen_subspace_fidelity(first, second, config["clusters"]),
        "sizes": sizes,
    }, frozen


def main():
    config = json.loads(CONFIG.read_text()); source = np.load(config["source"]); atlas = json.loads(Path(config["atlas"]).read_text())
    model = load_full_model("cuda"); model.blocks.float().eval(); results = {}; assignments = {}
    for layer in config["layers"]:
        pd = source[f"discovery_products_L{layer}"].astype(np.float32); pr = source[f"replication_products_L{layer}"].astype(np.float32)
        cd = source[f"discovery_contributions_L{layer}"].astype(np.float32); cr = source[f"replication_contributions_L{layer}"].astype(np.float32)
        down = model.blocks[layer].mlp.Down.weight.detach().float().cpu().numpy(); scale = canonical_channel_scale(down).astype(np.float32)
        views = {"canonical_allocation": (pd * scale, pr * scale), "exact_write": (pd @ down.T, pr @ down.T), "ce_allocation": (cd, cr)}
        results[str(layer)] = {}
        for view in config["views"]:
            row, assignment = fullspace_fit(*views[view], config, config["seed"] + 1009 * layer + 17 * config["views"].index(view))
            results[str(layer)][view] = row; assignments[(layer, view)] = assignment
        print(f"fullspace MLP redteam L{layer}", flush=True)
    pairs = [("canonical_allocation", "exact_write"), ("canonical_allocation", "ce_allocation"), ("exact_write", "ce_allocation")]
    cross = {str(layer): {f"{a}__{b}": adjusted_rand(assignments[(layer, a)], assignments[(layer, b)]) for a, b in pairs} for layer in config["layers"]}
    recurring = {view: sum(results[str(layer)][view]["matched_center_cosine"] > results[str(layer)][view]["null_p95"] for layer in config["layers"]) for view in config["views"]}
    adequate = sum(atlas["results"][str(layer)][view]["retained_energy"]["replication"] >= .50 for layer in config["layers"] for view in config["views"])
    cross_median = {str(layer): float(np.median(list(cross[str(layer)].values()))) for layer in config["layers"]}
    cluster_wins = sum(results[str(layer)][view]["cluster_fidelity"] > results[str(layer)][view]["rank_matched_subspace_fidelity"] for layer in config["layers"] for view in config["views"])
    eligible = []
    for layer in config["layers"]:
        for view in config["views"]:
            for cluster, row in enumerate(results[str(layer)][view]["sizes"]):
                if row["center_cosine"] >= config["minimum_center_cosine"] and min(row["discovery"], row["replication"]) >= config["minimum_candidate_size"]:
                    eligible.append({"layer": layer, "view": view, "cluster": cluster, **row})
    depth_gaps = {}
    for view in config["views"]:
        late = np.mean([results[str(layer)][view]["matched_center_cosine"] for layer in (16, 17)])
        early = np.mean([results[str(layer)][view]["matched_center_cosine"] for layer in config["layers"] if layer <= 12])
        depth_gaps[view] = float(late - early)
    cells = len(config["layers"]) * len(config["views"])
    hypotheses = {
        "R0_fullspace_clusters_recur": {"supported": sum(value >= 6 for value in recurring.values()) >= 2, "layers_beating_null": recurring},
        "R1_rank64_was_adequate": {"supported": adequate >= .75 * cells, "passing": adequate, "cells": cells},
        "R2_one_fullspace_ontology_spans_views": {"supported": sum(value >= .50 for value in cross_median.values()) >= 6, "layer_median_ari": cross_median},
        "R3_discrete_fullspace_codes_beat_continuous_subspaces": {"supported": cluster_wins >= .5 * cells, "cluster_wins": cluster_wins, "cells": cells},
        "R4_fullspace_candidates_exist": {"supported": len(eligible) >= 8, "eligible": len(eligible)},
        "R5_late_layers_are_more_canonical": {"supported": sum(value >= .10 for value in depth_gaps.values()) >= 2, "late_minus_early": depth_gaps},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results, "cross_view_ari": cross, "eligible_candidates": eligible}, indent=2, allow_nan=False))
    lines = ["# Full-space red team for the task-blind MLP atlas", "", "The rank-64 atlas is rerun directly in each complete view: 4,608 canonical product-allocation coordinates, the exact 1,152-dimensional residual write, and 4,608 channelwise CE contributions. This removes the discovery-PCA transfer bottleneck.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | View | Center/null | Margin D/R | Cluster/PCA fidelity |", "|---:|---|---:|---:|---:|"]
    for layer in config["layers"]:
        for view in config["views"]:
            row = results[str(layer)][view]; lines.append(f"| {layer} | {view} | {row['matched_center_cosine']:.3f}/{row['null_p95']:.3f} | {row['discovery_margin']:.3f}/{row['replication_margin']:.3f} | {row['cluster_fidelity']:.3f}/{row['rank_matched_subspace_fidelity']:.3f} |")
    lines += ["", "## Cross-view agreement", "", "| Layer | Allocation↔write | Allocation↔CE | Write↔CE |", "|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = cross[str(layer)]; lines.append(f"| {layer} | {row['canonical_allocation__exact_write']:.3f} | {row['canonical_allocation__ce_allocation']:.3f} | {row['exact_write__ce_allocation']:.3f} |")
    lines += ["", "## Interpretation guardrail", "", "Beating a coordinate-permutation null establishes recurrence in native tensor coordinates, not a discrete vocabulary. The rank-matched PCA comparison tests discreteness, and cross-view ARI tests whether internal allocation, communicated write, and current-prediction effect share one partition. Candidate counts are descriptive until finite interventions and semantic transfer are run."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
