#!/usr/bin/env python3
"""Plot head-importance transfer and random-subset red team."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bilin18_tensor_decomposition"
SPARSITY = ARTIFACT / "bilin18_transfer_state_role_head_sparsity.json"
RANDOM = ARTIFACT / "bilin18_transfer_state_role_head_random_redteam.json"
OUTPUT = ARTIFACT / "bilin18_role_head_sparsity.png"


def main() -> None:
    sparse = json.loads(SPARSITY.read_text())
    random = json.loads(RANDOM.read_text())
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)

    importance = sparse["importance"]
    x = np.asarray([row["discovery_importance"] for row in importance])
    y = np.asarray([row["evaluation_importance"] for row in importance])
    ranks = np.asarray([row["discovery_rank"] for row in importance])
    selected = ranks <= 8
    axes[0].scatter(x[~selected], y[~selected], color="tab:gray", alpha=.65, label="other heads")
    axes[0].scatter(x[selected], y[selected], color="tab:red", label="frozen top 8")
    for row in importance:
        if row["discovery_rank"] <= 8:
            axes[0].annotate(f"L{row['layer']}H{row['head']}",
                             (row["discovery_importance"], row["evaluation_importance"]),
                             xytext=(3, 2), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("discovery leave-one-out error")
    axes[0].set_ylabel("evaluation leave-one-out error")
    axes[0].set_title("Head importance partly transfers (ρ = 0.639)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)

    positions = [1, 2]
    for index, rank in enumerate((8, 16)):
        row = random["summary"][str(rank)]
        axes[1].scatter(np.full(len(row["random_mean_fidelity"]), positions[index]),
                        row["random_mean_fidelity"], alpha=.7, color="tab:gray")
        axes[1].scatter(positions[index], row["top_mean_fidelity"], marker="*", s=180,
                        color="tab:red", zorder=3)
    axes[1].set_xticks(positions, ["rank 8", "rank 16"])
    axes[1].set_ylabel("mean held-out causal fidelity")
    axes[1].set_title("Ranked heads do not beat random behaviorally")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].legend(handles=[
        plt.Line2D([], [], marker="*", color="tab:red", linestyle="", markersize=12, label="ranked"),
        plt.Line2D([], [], marker="o", color="tab:gray", linestyle="", label="random subset"),
    ], fontsize=8)

    for index, rank in enumerate((8, 16)):
        row = random["summary"][str(rank)]
        axes[2].scatter(np.full(len(row["random_mean_adjoint_error"]), positions[index]),
                        row["random_mean_adjoint_error"], alpha=.7, color="tab:gray")
        axes[2].scatter(positions[index], row["top_mean_adjoint_error"], marker="*", s=180,
                        color="tab:blue", zorder=3)
    axes[2].set_xticks(positions, ["rank 8", "rank 16"])
    axes[2].set_ylabel("weighted suffix-adjoint error")
    axes[2].set_title("Ranked heads are special geometrically")
    axes[2].grid(axis="y", alpha=.25)
    axes[2].legend(handles=[
        plt.Line2D([], [], marker="*", color="tab:blue", linestyle="", markersize=12, label="ranked"),
        plt.Line2D([], [], marker="o", color="tab:gray", linestyle="", label="random subset"),
    ], fontsize=8)

    figure.suptitle("Native head sparsity depends on the equivalence relation", fontsize=14)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
