#!/usr/bin/env python3
"""Make a compact explanatory figure for the grammar operator-regime results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/grammar_regimes")
OUTPUT = ROOT / "grammar_operator_regimes.png"


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    load_sweep = load("grammar_jacobian_gauge_redteam.json")
    capacity = load("grammar_jacobian_capacity_sweep.json")
    natural = load("grammar_mixed_natural_metric_redteam.json")
    explanation = load("grammar_mixed_chart_explanation.json")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    for rho, label, color in (("0.0", "0% arbitrary", "#2474b5"),
                              ("0.5", "50% arbitrary", "#e08214"),
                              ("1.0", "100% arbitrary", "#b2182b")):
        rows = []
        for seed in load_sweep["seeds"]:
            rows.append([layer["rms"]["effective_rank"] for layer in seed["models"][rho]["layers"]])
        rows = np.asarray(rows); ax.plot(range(4), rows.mean(0), marker="o", label=label, color=color)
        ax.fill_between(range(4), rows.min(0), rows.max(0), alpha=.15, color=color)
    ax.set(title="Arbitrary load expands early operator families", xlabel="Layer", ylabel="RMS-normalized effective rank")
    ax.set_xticks(range(4)); ax.legend(frameon=False)

    ax = axes[0, 1]
    widths = [64, 96, 192, 384]
    early = [capacity["summary"][str(width)]["layers"][0]["effective_rank"] for width in widths]
    late = [capacity["summary"][str(width)]["layers"][3]["effective_rank"] for width in widths]
    fractional = [early[i] / (2 * width) for i, width in enumerate(widths)]
    ax.plot(widths, early, marker="o", label="Layer 0: absolute rank", color="#b2182b")
    ax.plot(widths, late, marker="o", label="Layer 3: absolute rank", color="#2474b5")
    twin = ax.twinx(); twin.plot(widths, fractional, marker="s", ls="--", color="#4d4d4d", label="Layer 0: rank/channels")
    ax.set(title="Width dilutes rather than removes diversity", xlabel="Model width", ylabel="Effective rank")
    twin.set_ylabel("Layer-0 rank / MLP channels"); ax.set_xscale("log", base=2); ax.set_xticks(widths, labels=widths)
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8)

    ax = axes[1, 0]
    groups = ("heldout_generalized", "train_unselected", "pure_same_heldout")
    labels = ("Mixed: generalized rules", "Mixed: trained rules", "Pure rule: same prompts")
    colors = ("#b2182b", "#e08214", "#2474b5")
    for group, label, color in zip(groups, labels, colors):
        rows = np.asarray([[seed["layers"][layer][group]["fisher_covariance"]["mean"] for layer in range(4)]
                           for seed in natural["seeds"]])
        ax.plot(range(4), rows.mean(0), marker="o", label=label, color=color)
        ax.fill_between(range(4), rows.min(0), rows.max(0), alpha=.15, color=color)
    ax.set(title="Correct generalization can use the mixed regime", xlabel="Layer", ylabel="Fisher-covariance effective rank")
    ax.set_xticks(range(4)); ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for feature, label, color in (("semantic", "Rule semantics", "#2474b5"),
                                  ("output", "Answer/rival role", "#1b9e77"),
                                  ("address", "Token/entity address", "#e08214"),
                                  ("combined", "All additive features", "#6a3d9a")):
        rows = np.asarray([[seed["layers"][layer][feature]["heldout_r2"] for layer in range(4)]
                           for seed in explanation["seeds"]])
        ax.plot(range(4), rows.mean(0), marker="o", label=label, color=color)
    ax.axhline(0, color="black", lw=.7)
    ax.set(title="Simple explanations emerge only near output", xlabel="Layer", ylabel="Held-out causal-chart R²")
    ax.set_xticks(range(4)); ax.set_ylim(-.05, 1.0); ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Grammar-to-memorization operator regimes", fontsize=15)
    fig.savefig(OUTPUT, dpi=180)
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
