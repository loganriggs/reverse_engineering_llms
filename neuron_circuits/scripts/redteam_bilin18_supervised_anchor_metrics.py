#!/usr/bin/env python3
"""Post-hoc metric red team for supervised-anchor/weight-motif agreement."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/bilin18_tensor_decomposition")
ARRAYS = ROOT / "bilin18_supervised_anchor_motif_comparison.npz"
MOTIFS = ROOT / "bilin18_unsupervised_weight_motif_redteam.json"
OUTPUT = ROOT / "bilin18_supervised_anchor_metric_redteam.json"
REPORT = ROOT / "BILIN18_SUPERVISED_ANCHOR_METRIC_REDTEAM.md"
FIGURE = ROOT / "bilin18_supervised_anchor_metric_redteam.png"
REPLICATES = 10_000
SEED = 20262112


def unit_rows(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-30)


def geometry(values):
    values = unit_rows(values)
    return values @ values.T


def triangle_correlation(first, second):
    triangle = np.triu_indices(len(first), 1)
    return float(np.corrcoef(first[triangle], second[triangle])[0, 1])


def motif_pairs(payload, heads=9):
    components = [[member["layer"] * heads + member["head"] for member in component]
                  for component in payload["bus_robust"]["components"]]
    membership = {node: index for index, component in enumerate(components) for node in component}
    closure = [(a, b) for component in components for i, a in enumerate(component)
               for b in component[i + 1:]]
    direct = [tuple(map(int, pair)) for pair in payload["bus_robust"]["edge_pairs"]]
    return components, membership, closure, direct


def pools_for(pairs, membership, heads=9, nodes=162):
    pools = []
    for left, right in pairs:
        layer_pair = tuple(sorted((left // heads, right // heads)))
        candidates = []
        for a in range(nodes):
            for b in range(a + 1, nodes):
                if tuple(sorted((a // heads, b // heads))) != layer_pair:
                    continue
                if membership.get(a, -1) >= 0 and membership.get(a) == membership.get(b):
                    continue
                candidates.append((a, b))
        pools.append(candidates)
    return pools


def compare(similarity, pairs, pools, rng):
    observed = float(np.mean([similarity[a, b] for a, b in pairs]))
    null = np.empty(REPLICATES)
    for replicate in range(REPLICATES):
        null[replicate] = np.mean([
            similarity[pair[0], pair[1]]
            for pool in pools
            for pair in [pool[int(rng.integers(len(pool)))]]
        ])
    return {"pairs": len(pairs), "motif_mean": observed,
            "matched_mean": float(null.mean()), "enrichment": float(observed - null.mean()),
            "one_sided_p": float((1 + np.sum(null >= observed)) / (REPLICATES + 1))}


def main():
    arrays = np.load(ARRAYS)
    labels, split = arrays["labels"], arrays["split"]
    motif = json.loads(MOTIFS.read_text())
    components, membership, closure, direct = motif_pairs(motif)
    closure_pools = pools_for(closure, membership)
    direct_pools = pools_for(direct, membership)
    results = {}
    for objective in ("semantic", "gram_matched"):
        raw = arrays[f"{objective}_site_gradients"].reshape(len(labels), -1).astype(np.float64)
        normalized = unit_rows(raw)
        results[objective] = {}
        for aggregation in ("signed", "absolute", "rms"):
            sides = []
            for side in (0, 1):
                profile = np.zeros((raw.shape[1], int(labels.max()) + 1))
                for label in range(profile.shape[1]):
                    sites = normalized[(labels == label) & (split == side)]
                    if aggregation == "signed":
                        profile[:, label] = sites.mean(0)
                    elif aggregation == "absolute":
                        profile[:, label] = np.abs(sites).mean(0)
                    else:
                        profile[:, label] = np.sqrt(np.square(sites).mean(0))
                sides.append(profile)
            for centered in (False, True):
                variant = f"{aggregation}_{'centered' if centered else 'cosine'}"
                used = [value - value.mean(1, keepdims=True) if centered else value
                        for value in sides]
                combined = unit_rows((used[0] + used[1]) / 2)
                similarity = combined @ combined.T
                rng = np.random.default_rng(SEED)
                results[objective][variant] = {
                    "split_geometry_pearson": triangle_correlation(geometry(used[0]), geometry(used[1])),
                    "component_closure": compare(similarity, closure, closure_pools, rng),
                    "direct_edges": compare(similarity, direct, direct_pools, rng),
                }
    payload = {
        "status": "post-hoc-metric-red-team-after-locked-signed-comparison",
        "method": "Signed effect tests direction; absolute/RMS tests only where a head matters. Centering removes a head's generic sensitivity level across labels.",
        "results": results,
        "decision": {
            "signed_semantic_recovery": "not validated",
            "magnitude_recovery": "strong but non-semantic: Gram-matched objectives show comparable or larger enrichment",
            "use_as_ground_truth": "no; retain labels as external silver-standard anchors",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))
    lines = ["# Supervised-anchor metric red team", "",
             "This analysis was added after the locked signed-profile comparison. It asks whether the answer changes when head effects are treated as unsigned magnitude rather than promotion versus suppression. The Gram-matched vocabulary permutation is the critical semantic control.", "",
             "| Objective | Profile | Split r | Closure enrichment / p | Direct-edge enrichment / p |",
             "|---|---|---:|---:|---:|"]
    for objective in ("semantic", "gram_matched"):
        for variant in ("signed_cosine", "signed_centered", "absolute_cosine", "absolute_centered", "rms_cosine", "rms_centered"):
            row = results[objective][variant]; closure_row = row["component_closure"]; direct_row = row["direct_edges"]
            lines.append(f"| {objective} | {variant} | {row['split_geometry_pearson']:.3f} | {closure_row['enrichment']:+.3f} / {closure_row['one_sided_p']:.4f} | {direct_row['enrichment']:+.3f} / {direct_row['one_sided_p']:.4f} |")
    lines += ["", "## Decision", "",
              "The magnitude result is real as a statement about generic causal salience: weight-motif members tend to matter on similar examples. It is not semantic validation, because the Gram-matched nonsense objectives reproduce the same effect and can be stronger after centering. The signed semantic effect is larger than its nonsense control, but it did not clear the locked significance or semantic-margin gates.", "",
              "Therefore the 32 labels should annotate frozen components but should not be folded into cluster discovery or called ground-truth circuits."]
    REPORT.write_text("\n".join(lines) + "\n")
    variants = ("signed_cosine", "signed_centered", "absolute_cosine", "absolute_centered", "rms_cosine", "rms_centered")
    x = np.arange(len(variants)); fig, axes = plt.subplots(1, 2, figsize=(15, 5), layout="constrained")
    for index, target in enumerate(("component_closure", "direct_edges")):
        width = .36
        axes[index].bar(x - width / 2, [results["semantic"][variant][target]["enrichment"] for variant in variants], width, label="semantic")
        axes[index].bar(x + width / 2, [results["gram_matched"][variant][target]["enrichment"] for variant in variants], width, label="Gram-matched null")
        axes[index].axhline(0, color="black", linewidth=1); axes[index].set_xticks(x, variants, rotation=35, ha="right")
        axes[index].set_ylabel("Layer-matched cosine enrichment"); axes[index].set_title(target.replace("_", " ")); axes[index].legend(); axes[index].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(payload["decision"], indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
