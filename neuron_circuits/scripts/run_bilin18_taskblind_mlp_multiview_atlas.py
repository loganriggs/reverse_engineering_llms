#!/usr/bin/env python3
"""Discover task-blind MLP regimes under three purpose-relative ontologies.

The views distinguish a canonicalized native allocation, the exact residual write,
and the channelwise CE allocation.  All discovery precedes token decoding.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_operator_probe_convergence import linear_sum_assignment_maximum
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unsupervised_weight_circuit_atlas import adjusted_rand, spherical_kmeans


CONFIG = Path("configs/bilin18_taskblind_mlp_multiview_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_taskblind_mlp_multiview_atlas.json"
ARRAYS = TENSOR_ROOT / "bilin18_taskblind_mlp_multiview_atlas.npz"
REPORT = TENSOR_ROOT / "BILIN18_TASKBLIND_MLP_MULTIVIEW_ATLAS.md"


def canonical_channel_scale(down_weight: np.ndarray) -> np.ndarray:
    """Fix product-neuron scale and sign using its residual-write vector.

    If h_c -> a h_c and d_c -> d_c/a, h_c ||d_c|| sign(pivot(d_c)) is
    invariant for any nonzero real a.  Channel permutations remain harmless to
    within-model geometry.
    """
    down = np.asarray(down_weight, dtype=np.float64)
    pivot = np.abs(down).argmax(0)
    signs = np.sign(down[pivot, np.arange(down.shape[1])])
    signs[signs == 0] = 1
    return np.linalg.norm(down, axis=0) * signs


def dual_pca_coordinates(discovery: np.ndarray, replication: np.ndarray, rank: int):
    discovery = np.asarray(discovery, dtype=np.float64)
    replication = np.asarray(replication, dtype=np.float64)
    mean = discovery.mean(0)
    first = discovery - mean; second = replication - mean
    gram = first @ first.T
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(values)[::-1]; values = values[order]; vectors = vectors[:, order]
    keep = min(rank, int((values > 1e-12 * max(values[0], 1e-30)).sum()))
    basis = (vectors[:, :keep].T @ first) / np.sqrt(values[:keep, None].clip(1e-30))
    dcoords = first @ basis.T; rcoords = second @ basis.T
    denergy = float(np.square(dcoords).sum() / max(np.square(first).sum(), 1e-30))
    renergy = float(np.square(rcoords).sum() / max(np.square(second).sum(), 1e-30))
    return dcoords, rcoords, basis, {"discovery": denergy, "replication": renergy}


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-30)


def match_centers(first: np.ndarray, second: np.ndarray):
    score = np.asarray(first) @ np.asarray(second).T
    rows, columns = linear_sum_assignment_maximum(score)
    order = np.argsort(rows); columns = columns[order]
    cosines = score[np.arange(len(first)), columns]
    return columns, cosines


def frozen_cluster_fidelity(replication: np.ndarray, centers: np.ndarray) -> float:
    assignment = (replication @ centers.T).argmax(1)
    approximation = centers[assignment]
    return float(1 - np.square(replication - approximation).sum() / max(np.square(replication).sum(), 1e-30))


def frozen_subspace_fidelity(discovery: np.ndarray, replication: np.ndarray, rank: int) -> float:
    _, _, vh = np.linalg.svd(np.asarray(discovery, dtype=np.float64), full_matrices=False)
    basis = vh[:rank]; approximation = (replication @ basis.T) @ basis
    return float(1 - np.square(replication - approximation).sum() / max(np.square(replication).sum(), 1e-30))


def encoded_labels(values) -> np.ndarray:
    return np.unique(np.asarray(values), return_inverse=True)[1]


def loss_quartiles(discovery_loss, values):
    boundaries = np.quantile(discovery_loss, [.25, .50, .75])
    return np.digitize(values, boundaries)


def label_agreements(assignments: np.ndarray, metadata: dict, discovery_loss: np.ndarray):
    labels = {
        "current_form": metadata["current_class"],
        "next_form": metadata["next_class"],
        "position": metadata["position"],
        "loss_quartile": loss_quartiles(discovery_loss, metadata["loss"]),
    }
    return {name: adjusted_rand(assignments, encoded_labels(value)) for name, value in labels.items()}


def enrichment(assignments, cluster, metadata, discovery_loss):
    selected = assignments == cluster
    families = {
        "current": metadata["current_class"], "next": metadata["next_class"],
        "position": np.asarray([f"p{x}" for x in metadata["position"]]),
        "loss": np.asarray(["q1", "q2", "q3", "q4"])[loss_quartiles(discovery_loss, metadata["loss"])],
    }
    rows = []
    for family, values in families.items():
        for value in np.unique(values):
            inside = float(np.mean(values[selected] == value)); outside = float(np.mean(values == value))
            rows.append({"label": f"{family}:{value}", "inside": inside, "baseline": outside, "lift": inside / max(outside, 1e-12)})
    return sorted(rows, key=lambda row: (row["lift"], row["inside"]), reverse=True)[:4]


def clean(text: str) -> str:
    return text.replace("\n", "\\n").replace("\t", "\\t")


def decoded_examples(tokenizer, corpus, metadata, assignments, scores, cluster, count, before, after):
    members = np.flatnonzero(assignments == cluster)
    members = members[np.argsort(-scores[members])[:count]]
    rows = []
    for index in members:
        row = int(metadata["global_row"][index]); position = int(metadata["position"][index])
        tokens = np.asarray(corpus[row], dtype=np.int64)
        left = max(0, position - before); right = min(len(tokens), position + after + 1)
        rows.append({
            "global_row": row, "position": position, "score": float(scores[index]),
            "current": clean(tokenizer.decode([int(tokens[position])])),
            "next": clean(tokenizer.decode([int(tokens[position + 1])])),
            "context": clean(tokenizer.decode(tokens[left:right].tolist())),
        })
    return rows


def main():
    config = json.loads(CONFIG.read_text()); source = np.load(config["source"]); corpus = np.load(config["corpus"], mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda"); model.blocks.float().eval()
    dmeta = {key[len("discovery_meta_"):]: source[key] for key in source.files if key.startswith("discovery_meta_")}
    rmeta = {key[len("replication_meta_"):]: source[key] for key in source.files if key.startswith("replication_meta_")}
    rng = np.random.default_rng(config["seed"]); results = {}; arrays = {}; candidate_pool = []
    primary_assignments = {}
    for layer in config["layers"]:
        products_d = source[f"discovery_products_L{layer}"].astype(np.float32)
        products_r = source[f"replication_products_L{layer}"].astype(np.float32)
        contributions_d = source[f"discovery_contributions_L{layer}"].astype(np.float32)
        contributions_r = source[f"replication_contributions_L{layer}"].astype(np.float32)
        down = model.blocks[layer].mlp.Down.weight.detach().float().cpu().numpy()
        scale = canonical_channel_scale(down).astype(np.float32)
        raw_views = {
            "canonical_allocation": (products_d * scale, products_r * scale),
            "exact_write": (products_d @ down.T, products_r @ down.T),
            "ce_allocation": (contributions_d, contributions_r),
        }
        results[str(layer)] = {}
        for view in config["views"]:
            dcoords, rcoords, basis, retained = dual_pca_coordinates(*raw_views[view], config["signature_rank"])
            dvalues, rvalues = unit_rows(dcoords), unit_rows(rcoords)
            arrays[f"discovery_signature_L{layer}_{view}"] = dvalues.astype(np.float32)
            arrays[f"replication_signature_L{layer}_{view}"] = rvalues.astype(np.float32)
            counts = {}; fits = {}
            for count in config["cluster_counts"]:
                da, dc, down_own, down_margin = spherical_kmeans(torch.as_tensor(dvalues, dtype=torch.float32), count, config["kmeans_restarts"], config["kmeans_iterations"], config["seed"] + 1009 * layer + 31 * count + 7 * config["views"].index(view))
                ra, rc, rown, rmargin = spherical_kmeans(torch.as_tensor(rvalues, dtype=torch.float32), count, config["kmeans_restarts"], config["kmeans_iterations"], config["seed"] + 100000 + 1009 * layer + 31 * count + 7 * config["views"].index(view))
                dc = dc.numpy(); rc = rc.numpy(); permutation, cosines = match_centers(dc, rc)
                null = []
                for _ in range(config["null_replicates"]):
                    _, null_cosines = match_centers(dc, rc[:, rng.permutation(rc.shape[1])]); null.append(float(np.mean(null_cosines)))
                row = {
                    "matched_center_cosine": float(np.mean(cosines)), "center_cosines": cosines.tolist(),
                    "discovery_own_cosine": float(np.mean(down_own)), "replication_own_cosine": float(np.mean(rown)),
                    "discovery_margin": float(np.mean(down_margin)), "replication_margin": float(np.mean(rmargin)),
                    "null_p95": float(np.quantile(null, .95)),
                    "cluster_fidelity": frozen_cluster_fidelity(rvalues, dc),
                    "rank_matched_subspace_fidelity": frozen_subspace_fidelity(dvalues, rvalues, count),
                }
                counts[str(count)] = row; fits[count] = (da, dc, ra, rc, permutation, cosines)
            count = config["primary_clusters"]; da, dc, ra, rc, permutation, cosines = fits[count]
            frozen = (rvalues @ dc.T).argmax(1); primary_assignments[(layer, view)] = frozen
            agreement = label_agreements(frozen, rmeta, dmeta["loss"])
            view_row = {"retained_energy": retained, "counts": counts, "coarse_label_ari": agreement}
            results[str(layer)][view] = view_row
            for cluster in range(count):
                dsize = int(np.sum(da == cluster)); rsize = int(np.sum(frozen == cluster)); center_scores = rvalues @ dc[cluster]
                candidate_pool.append({
                    "layer": layer, "view": view, "cluster": cluster,
                    "center_cosine": float(cosines[cluster]), "discovery_size": dsize, "replication_size": rsize,
                    "mean_frozen_member_cosine": float(np.mean(center_scores[frozen == cluster])) if rsize else 0.0,
                    "enrichment": enrichment(frozen, cluster, rmeta, dmeta["loss"]),
                    "_assignments": frozen, "_scores": center_scores,
                })
        print(f"task-blind MLP multiview L{layer}", flush=True)

    cross_view = {}; pairs = [("canonical_allocation", "exact_write"), ("canonical_allocation", "ce_allocation"), ("exact_write", "ce_allocation")]
    for layer in config["layers"]:
        cross_view[str(layer)] = {f"{first}__{second}": adjusted_rand(primary_assignments[(layer, first)], primary_assignments[(layer, second)]) for first, second in pairs}

    eligible = [row for row in candidate_pool if row["center_cosine"] >= config["minimum_candidate_center_cosine"] and min(row["discovery_size"], row["replication_size"]) >= config["minimum_candidate_size"]]
    eligible.sort(key=lambda row: row["center_cosine"] * row["mean_frozen_member_cosine"], reverse=True)
    selected = eligible[:config["maximum_candidates"]]
    for row in selected:
        row["examples"] = decoded_examples(tokenizer, corpus, rmeta, row.pop("_assignments"), row.pop("_scores"), row["cluster"], config["examples_per_candidate"], config["context_tokens_before"], config["context_tokens_after"])
    for row in candidate_pool:
        row.pop("_assignments", None); row.pop("_scores", None)

    primary_cells = [(layer, view, results[str(layer)][view]["counts"][str(config["primary_clusters"])]) for layer in config["layers"] for view in config["views"]]
    recurring = {view: sum(results[str(layer)][view]["counts"][str(config["primary_clusters"])]["matched_center_cosine"] > results[str(layer)][view]["counts"][str(config["primary_clusters"])]["null_p95"] for layer in config["layers"]) for view in config["views"]}
    cross_medians = {str(layer): float(np.median(list(cross_view[str(layer)].values()))) for layer in config["layers"]}
    label_low = sum(max(results[str(layer)][view]["coarse_label_ari"].values()) < .25 for layer in config["layers"] for view in config["views"])
    cluster_wins = sum(row["cluster_fidelity"] > row["rank_matched_subspace_fidelity"] for _, _, row in primary_cells)
    privileged = sum(results[str(layer)][view]["counts"]["4"]["matched_center_cosine"] >= max(results[str(layer)][view]["counts"]["2"]["matched_center_cosine"], results[str(layer)][view]["counts"]["8"]["matched_center_cosine"]) + .05 for layer in config["layers"] for view in config["views"])
    hypotheses = {
        "A0_discovery_is_task_blind": {"supported": True, "views": config["views"], "semantic_fields_used_after_selection_only": True},
        "A1_model_derived_clusters_recur": {"supported": sum(value >= 6 for value in recurring.values()) >= 2, "layers_beating_null": recurring},
        "A2_one_cluster_ontology_spans_views": {"supported": sum(value >= .50 for value in cross_medians.values()) >= 6, "layer_median_ari": cross_medians},
        "A3_clusters_are_not_coarse_label_restatements": {"supported": label_low >= .75 * len(primary_cells), "passing": label_low, "cells": len(primary_cells)},
        "A4_discrete_codes_beat_continuous_subspaces": {"supported": cluster_wins >= .5 * len(primary_cells), "cluster_wins": cluster_wins, "cells": len(primary_cells)},
        "A5_k4_is_a_privileged_vocabulary": {"supported": privileged >= .5 * len(primary_cells), "passing": privileged, "cells": len(primary_cells)},
        "A6_replicated_unlabeled_candidates_exist": {"supported": len(eligible) >= 8, "eligible": len(eligible), "selected": len(selected)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "results": results, "cross_view_ari": cross_view, "selected_candidates": selected}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False)); np.savez_compressed(ARRAYS, **arrays)
    lines = ["# Task-blind MLP multiview atlas", "", "The same natural FineWeb sites are organized three ways after a frozen rank-64 compression: canonicalized native product allocation, exact residual write, and channelwise next-token CE allocation. Clustering uses no task labels or decoded text. Coarse labels and examples are inspected only after all coordinates and memberships are frozen.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "## Primary k=4 cells", "", "| Layer | View | Center cosine/null p95 | Margin D/R | Cluster/PCA fidelity | Max coarse-label ARI | Rank-64 energy D/R |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer, view, row in primary_cells:
        view_row = results[str(layer)][view]
        lines.append(f"| {layer} | {view} | {row['matched_center_cosine']:.3f}/{row['null_p95']:.3f} | {row['discovery_margin']:.3f}/{row['replication_margin']:.3f} | {row['cluster_fidelity']:.3f}/{row['rank_matched_subspace_fidelity']:.3f} | {max(view_row['coarse_label_ari'].values()):.3f} | {view_row['retained_energy']['discovery']:.3f}/{view_row['retained_energy']['replication']:.3f} |")
    lines += ["", "## Cross-view agreement", "", "| Layer | Allocation↔write | Allocation↔CE | Write↔CE |", "|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = cross_view[str(layer)]; lines.append(f"| {layer} | {row['canonical_allocation__exact_write']:.3f} | {row['canonical_allocation__ce_allocation']:.3f} | {row['exact_write__ce_allocation']:.3f} |")
    lines += ["", "## Frozen candidates", "", "| Candidate | Center cosine | D/R size | Member cosine | Strongest post-hoc enrichment |", "|---|---:|---:|---:|---|"]
    for row in selected:
        strongest = row["enrichment"][0]; lines.append(f"| L{row['layer']} {row['view']} C{row['cluster']} | {row['center_cosine']:.3f} | {row['discovery_size']}/{row['replication_size']} | {row['mean_frozen_member_cosine']:.3f} | {strongest['label']} ({strongest['inside']:.2f}, {strongest['lift']:.1f}x) |")
    lines += ["", "## Guardrails", "", "Stable clusters can quantize a continuous density; the cluster-versus-PCA and k-sweep tests address that directly. Canonical allocation fixes bilinear channel scale/sign with each Down column, exact write removes channel-coordinate choice entirely, and CE allocation is prediction-relative. Disagreement is therefore evidence for purpose-relative ontologies, not automatically polysemanticity. The selected JSON examples are post-hoc interpretation material and do not validate a semantic name."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
