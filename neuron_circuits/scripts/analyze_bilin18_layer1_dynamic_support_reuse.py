#!/usr/bin/env python3
"""Post-hoc overlap and reuse audit for pair-local semantic supports."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


SOURCE = TENSOR_ROOT / "bilin18_layer1_dynamic_semantic_coalitions.json"
OUTPUT = TENSOR_ROOT / "bilin18_layer1_dynamic_support_reuse.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_DYNAMIC_SUPPORT_REUSE.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_dynamic_support_reuse.png"


def main():
    source = json.loads(SOURCE.read_text()); rows = source["records"]; width = 4608; rank = source["config"]["rank"]
    features = sorted(set(row["feature"] for row in rows)); labels = np.asarray([features.index(row["feature"]) for row in rows]); rng = np.random.default_rng(20262012)
    kinds = ("local_positive64", "local_absolute64", "local_direct64"); results = {}
    matrices = {}
    for kind in kinds:
        matrix = np.eye(len(rows)); within, across = [], []
        for i, j in itertools.combinations(range(len(rows)), 2):
            left, right = set(rows[i]["supports"][kind]), set(rows[j]["supports"][kind]); value = len(left & right) / len(left | right); matrix[i, j] = matrix[j, i] = value
            (within if labels[i] == labels[j] else across).append(value)
        matrices[kind] = matrix
        observed = float(np.mean(within) - np.mean(across)); null = []
        upper = np.triu_indices(len(rows), 1)
        for _ in range(2000):
            shuffled = rng.permutation(labels); same = shuffled[upper[0]] == shuffled[upper[1]]; values = matrix[upper]; null.append(float(values[same].mean() - values[~same].mean()))
        per_feature = {}
        for feature in features:
            selected = [row for row in rows if row["feature"] == feature]; counts = np.bincount([channel for row in selected for channel in row["supports"][kind]], minlength=width); total = counts.sum()
            per_feature[feature] = {"union_channels": int(np.sum(counts > 0)), "channels_in_half_or_more": int(np.sum(counts >= len(selected) / 2)), "maximum_contexts": int(counts.max()), "frequency_effective_support": float(total * total / max(counts @ counts, 1))}
        results[kind] = {"within_median_jaccard": float(np.median(within)), "within_mean_jaccard": float(np.mean(within)), "across_median_jaccard": float(np.median(across)), "across_mean_jaccard": float(np.mean(across)), "mean_gap": observed, "permutation_p_one_sided": float((1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1)), "per_feature": per_feature}
    random_expected = rank / (2 * width - rank)
    OUTPUT.write_text(json.dumps({"source": str(SOURCE), "random_expected_jaccard": random_expected, "results": results}, indent=2))
    lines = ["# How stable are the dynamic semantic coalitions?", "", "This is a post-hoc reuse audit of the 64-channel supports that were already selected and causally tested. It does not add a new preregistered success claim. Pairwise Jaccard overlap is compared within versus across semantic labels, and feature labels are permuted across the fixed support-overlap matrix as a null.", "", f"A uniformly random rank-64 support among 4,608 channels has expected pairwise Jaccard **{random_expected:.4f}**.", "", "![Dynamic support reuse](bilin18_layer1_dynamic_support_reuse.png)", "", "| Support | Within median / mean | Across median / mean | Mean gap | Permutation p | Median union | Median ≥half core |", "|---|---:|---:|---:|---:|---:|---:|"]
    for kind in kinds:
        row = results[kind]; unions = [value["union_channels"] for value in row["per_feature"].values()]; cores = [value["channels_in_half_or_more"] for value in row["per_feature"].values()]
        lines.append(f"| {kind} | {row['within_median_jaccard']:.4f}/{row['within_mean_jaccard']:.4f} | {row['across_median_jaccard']:.4f}/{row['across_mean_jaccard']:.4f} | {row['mean_gap']:+.4f} | {row['permutation_p_one_sided']:.4f} | {np.median(unions):.0f} | {np.median(cores):.0f} |")
    lines += ["", "## Reading the result", "", "The successful suffix-conditioned positive coalitions are almost completely reselected for each context: their median within-feature overlap equals the random-scale overlap, each feature uses roughly 680 distinct channels across only twelve contexts, and no channel appears in half the contexts for any feature. A small above-null mean overlap exists, but it is not a stable neuron core.", "", "The direct Down×unembedding supports are different: they have visibly higher within-feature than across-feature overlap and small recurring cores. Yet those stable weight-defined supports were much weaker causal controllers. This separates two objects: a relatively stable output-role scaffold in the weights, and a highly context-dependent subset that actually has leverage after the remaining sixteen layers.", "", "That is evidence for input-conditioned coalitions, not evidence that the model literally stores a named feature as a fixed set of 64 neurons. The selected coalition is also objective-dependent: changing the output contrast changes the suffix gradient and therefore the support."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    x = np.arange(len(kinds)); axes[0].bar(x - .18, [results[k]["within_mean_jaccard"] for k in kinds], .36, label="within feature"); axes[0].bar(x + .18, [results[k]["across_mean_jaccard"] for k in kinds], .36, label="across feature"); axes[0].axhline(random_expected, color="black", linestyle="--", label="random expectation"); axes[0].set_xticks(x, [k.replace("local_", "").replace("64", "") for k in kinds]); axes[0].set(ylabel="Mean Jaccard", title="Support identity across contexts"); axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    for offset, kind in enumerate(kinds): axes[1].scatter([offset] * len(features), [results[kind]["per_feature"][feature]["union_channels"] for feature in features], alpha=.8)
    axes[1].set_xticks(range(len(kinds)), [k.replace("local_", "").replace("64", "") for k in kinds]); axes[1].axhline(rank * 12, color="gray", linestyle=":", label="no reuse ceiling"); axes[1].set(ylabel="Distinct channels across 12 contexts", title="Feature support union"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"random_expected_jaccard": random_expected, "results": {kind: {key: value for key, value in results[kind].items() if key != "per_feature"} for kind in kinds}}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
