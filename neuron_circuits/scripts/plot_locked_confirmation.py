#!/usr/bin/env python3
"""Render aggregate curves and preregistered endpoint comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"raw": "#d55e00", "rotation": "#777777", "pca": "#0072b2"}
LABELS = {"raw": "raw neurons", "rotation": "random rotated", "pca": "learned subspace"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    output = args.summary.parent

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for family_index, family in enumerate(("operation", "class")):
        for layer_index, layer in enumerate(("1", "2")):
            axis = axes[family_index, layer_index]
            curve = summary["families"][family]["layers"][layer]["curve"]
            ranks = [row["rank"] for row in curve]
            for method in ("raw", "rotation", "pca"):
                mean = np.array([row[method]["mean"] for row in curve])
                low = np.array([row[method]["ci95_low"] for row in curve])
                high = np.array([row[method]["ci95_high"] for row in curve])
                axis.plot(ranks, mean, marker="o", color=COLORS[method], label=LABELS[method])
                axis.fill_between(ranks, low, high, color=COLORS[method], alpha=.13)
            axis.axhline(.8, color="#999999", linestyle="--", linewidth=.8)
            axis.axhline(.9, color="#999999", linestyle=":", linewidth=.8)
            axis.set_xscale("log", base=2)
            axis.set_ylim(-.02, 1.03)
            axis.set_title(f"{family.capitalize()} counterfactual — layer {layer}")
            axis.grid(alpha=.2)
            if family_index == 1:
                axis.set_xlabel("patched dimensions")
            if layer_index == 0:
                axis.set_ylabel("donor KL recovery")
    axes[0, 1].legend(loc="lower right")
    fig.suptitle("Locked confirmation: sparse causal transfer", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "aggregate_curves.png", dpi=200)
    plt.close(fig)

    groups, axis_values, subspace_values = [], [], []
    axis_errors, subspace_errors = [], []
    for family in ("operation", "class"):
        for layer in ("1", "2"):
            row = summary["families"][family]["layers"][layer]
            groups.append(f"{family}\nL{layer}")
            for endpoint, values, errors in (
                ("axis_privilege", axis_values, axis_errors),
                ("distributed_subspace_advantage", subspace_values, subspace_errors),
            ):
                interval = row[endpoint]
                values.append(interval["mean"])
                errors.append([interval["mean"] - interval["ci95_low"], interval["ci95_high"] - interval["mean"]])
    x = np.arange(len(groups)); width = .35
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(x - width/2, axis_values, width, yerr=np.array(axis_errors).T, capsize=4, color=COLORS["raw"], label="raw − rotation")
    axis.bar(x + width/2, subspace_values, width, yerr=np.array(subspace_errors).T, capsize=4, color=COLORS["pca"], label="PCA − raw")
    axis.axhline(.10, color="#333333", linestyle="--", linewidth=1, label="preregistered substantial effect")
    axis.axhline(.05, color="#777777", linestyle=":", linewidth=1, label="preregistered negligible-axis bound")
    axis.set_xticks(x, groups); axis.set_ylabel("mean KL-recovery advantage"); axis.grid(axis="y", alpha=.2)
    axis.legend(ncol=2); fig.tight_layout(); fig.savefig(output / "endpoint_comparison.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
