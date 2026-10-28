#!/usr/bin/env python3
"""Cluster-aware red team of the supervised coalition intervention result."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


SOURCE = TENSOR_ROOT / "bilin18_layer1_supervised_coalition_causality.json"
OUTPUT = TENSOR_ROOT / "bilin18_layer1_supervised_coalition_uncertainty.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUPERVISED_COALITION_UNCERTAINTY.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_supervised_coalition_uncertainty.png"


def main():
    source = json.loads(SOURCE.read_text()); records = source["records"]
    features = source["config"]["features"]; rng = np.random.default_rng(20262011)
    per_feature = []
    for feature in features:
        rows = [row for row in records if row["feature"] == feature]
        supervised = np.asarray([row["shifts"]["supervised64"] for row in rows])
        random = np.asarray([row["shifts"]["random64"] for row in rows])
        full = np.asarray([row["shifts"]["full"] for row in rows])
        per_feature.append({"feature": feature, "pairs": len(rows), "supervised_directional": float(np.mean(supervised > 0)), "random_directional": float(np.mean(random > 0)), "directional_advantage": float(np.mean(supervised > 0) - np.mean(random > 0)), "mean_supervised_shift": float(supervised.mean()), "mean_random_shift": float(random.mean()), "mean_full_shift": float(full.mean())})
    draws = []
    for _ in range(20_000):
        sampled_features = rng.choice(features, len(features), replace=True)
        directional, shift = [], []
        for feature in sampled_features:
            rows = [row for row in records if row["feature"] == feature]
            indices = rng.choice(len(rows), len(rows), replace=True)
            directional.extend([float(rows[index]["shifts"]["supervised64"] > 0) - float(rows[index]["shifts"]["random64"] > 0) for index in indices])
            shift.extend([rows[index]["shifts"]["supervised64"] - rows[index]["shifts"]["random64"] for index in indices])
        draws.append((np.mean(directional), np.mean(shift)))
    draws = np.asarray(draws)
    summary = {
        "directional_advantage": float(np.mean([row["directional_advantage"] for row in per_feature])),
        "directional_advantage_cluster_bootstrap_95": np.quantile(draws[:, 0], [.025, .975]).tolist(),
        "mean_shift_advantage": float(np.mean([row["mean_supervised_shift"] - row["mean_random_shift"] for row in per_feature])),
        "mean_shift_advantage_cluster_bootstrap_95": np.quantile(draws[:, 1], [.025, .975]).tolist(),
        "features_positive_zero_negative": [sum(row["directional_advantage"] > 0 for row in per_feature), sum(row["directional_advantage"] == 0 for row in per_feature), sum(row["directional_advantage"] < 0 for row in per_feature)],
        "robust_across_feature_sampling": bool(np.quantile(draws[:, 0], .025) > 0),
    }
    OUTPUT.write_text(json.dumps({"source": str(SOURCE), "summary": summary, "per_feature": per_feature}, indent=2))
    lines = ["# Cluster-aware uncertainty red team", "", "The intervention preregistration treats the 96 held-out pairs descriptively. This post-hoc red team resamples the eight semantic features first and the twelve pairs within each sampled feature second, so repeated text pairs do not masquerade as 96 independent conceptual replications.", "", f"The supervised-minus-random directional advantage is **{summary['directional_advantage']:+.3f}**, with a feature-cluster bootstrap 95% interval of **[{summary['directional_advantage_cluster_bootstrap_95'][0]:+.3f}, {summary['directional_advantage_cluster_bootstrap_95'][1]:+.3f}]**. The mean class-score shift advantage is **{summary['mean_shift_advantage']:+.4f}**, interval **[{summary['mean_shift_advantage_cluster_bootstrap_95'][0]:+.4f}, {summary['mean_shift_advantage_cluster_bootstrap_95'][1]:+.4f}]**.", "", f"Features with positive / zero / negative directional advantage: **{summary['features_positive_zero_negative'][0]} / {summary['features_positive_zero_negative'][1]} / {summary['features_positive_zero_negative'][2]}**.", "", "Therefore the aggregate finite intervention is promising but **not yet a feature-level replication**: the directional interval crosses zero. The locked site-level threshold remains reported as passed; the stronger claim that the effect generalizes across feature families is not validated.", "", "![Uncertainty by feature](bilin18_layer1_supervised_coalition_uncertainty.png)", "", "| Feature | Supervised | Random | Advantage | Mean supervised shift | Mean full shift |", "|---|---:|---:|---:|---:|---:|"]
    for row in per_feature: lines.append(f"| {row['feature']} | {row['supervised_directional']:.3f} | {row['random_directional']:.3f} | {row['directional_advantage']:+.3f} | {row['mean_supervised_shift']:+.4f} | {row['mean_full_shift']:+.4f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    names = [row["feature"] for row in per_feature]
    axes[0].barh(range(len(names)), [row["directional_advantage"] for row in per_feature]); axes[0].set_yticks(range(len(names)), names); axes[0].invert_yaxis(); axes[0].axvline(0, color="black"); axes[0].set(xlabel="Directional fraction: supervised − random", title="Heterogeneity across feature families"); axes[0].grid(axis="x", alpha=.2)
    axes[1].hist(draws[:, 0], bins=50, alpha=.8); axes[1].axvline(0, color="black"); axes[1].set(xlabel="Cluster-bootstrap directional advantage", ylabel="Draws", title="Feature-cluster uncertainty"); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2)); print(REPORT)


if __name__ == "__main__": main()
