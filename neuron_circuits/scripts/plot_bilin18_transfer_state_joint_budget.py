#!/usr/bin/env python3
"""Plot same-rank and runtime-coordinate views of joint Fisher charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bilin18_tensor_decomposition"
SOURCE = ARTIFACT / "bilin18_transfer_state_joint_fisher_budget.json"
OUTPUT = ARTIFACT / "bilin18_transfer_state_joint_budget.png"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = []
    for target, target_rows in source["results"].items():
        for boundary, row in target_rows.items():
            rows.append((target, int(boundary), row["methods"]))

    labels = [f"{target}\nb{boundary}" for target, boundary, _ in rows]
    x = np.arange(len(rows))
    collapsed64 = np.asarray([row["collapsed"]["64"]["response_fidelity"] for _, _, row in rows])
    joint64 = np.asarray([row["joint"]["64"]["response_fidelity"] for _, _, row in rows])
    collapsed1 = np.asarray([row["collapsed"]["1"]["response_fidelity"] for _, _, row in rows])
    joint128 = np.asarray([row["joint"]["128"]["response_fidelity"] for _, _, row in rows])

    ranks = np.asarray([1, 4, 16, 64, 128])
    joint_mean = np.asarray([
        np.mean([row["joint"][str(rank)]["response_fidelity"] for _, _, row in rows])
        for rank in ranks
    ])
    collapsed_mean = np.asarray([
        np.mean([row["collapsed"][str(rank)]["response_fidelity"] for _, _, row in rows])
        for rank in ranks
    ])

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    width = 0.38
    axes[0].bar(x - width / 2, collapsed64, width, label="collapsed r64\n8192 runtime coordinates")
    axes[0].bar(x + width / 2, joint64, width, label="joint r64\n64 runtime coordinates")
    axes[0].axhline(.75, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Same nominal rank")
    axes[0].set_ylabel("final Fisher-response fidelity")
    axes[0].set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)

    axes[1].plot(ranks, joint_mean, "o-", label="joint sequence field")
    axes[1].plot(128 * ranks, collapsed_mean, "o-", label="collapsed residual chart")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlim(.8, 20000)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Runtime-coordinate frontier")
    axes[1].set_xlabel("runtime coefficients (log scale)")
    axes[1].set_ylabel("mean fidelity across 12 cells")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.25)

    axes[2].scatter(collapsed1, joint128, c=[boundary for _, boundary, _ in rows], cmap="viridis", s=60)
    axes[2].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    for index, (target, boundary, _) in enumerate(rows):
        axes[2].annotate(f"{target}/b{boundary}", (collapsed1[index], joint128[index]),
                         xytext=(4, 2), textcoords="offset points", fontsize=7)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Matched 128-coordinate budget")
    axes[2].set_xlabel("collapsed r1 fidelity")
    axes[2].set_ylabel("joint r128 fidelity")
    axes[2].grid(alpha=.25)

    figure.suptitle("Cross-position structure is expensive to ignore", fontsize=14)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
