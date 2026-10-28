#!/usr/bin/env python3
"""Visualize causal fidelity and positional diagnostics for read functions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bilin18_tensor_decomposition"
SOURCE = ARTIFACT / "bilin18_transfer_state_read_function_audit.json"
OUTPUT = ARTIFACT / "bilin18_transfer_state_read_function_audit.png"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = []
    readable = ["current_sites4", "current_previous8", "recent4x4", "recent8x4", "segment4", "causal_prefix4"]
    for target, target_rows in source["results"].items():
        for boundary, row in target_rows.items():
            rows.append({"target": target, "boundary": int(boundary), **row})
    labels = [f"{row['target']}\nb{row['boundary']}" for row in rows]
    x = np.arange(len(rows))
    joint = np.asarray([row["joint128"]["response_fidelity"] for row in rows])
    current = np.asarray([row["current_sites4"]["response_fidelity"] for row in rows])
    best = np.asarray([max(row[name]["response_fidelity"] for name in readable) for row in rows])
    fisher = np.asarray([row["fisher_position8"]["response_fidelity"] for row in rows])
    random = np.asarray([row["random_position8"]["response_fidelity"] for row in rows])

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    width = .19
    axes[0].bar(x - 1.5 * width, joint, width, label="joint field")
    axes[0].bar(x - .5 * width, best, width, label="best readable")
    axes[0].bar(x + .5 * width, fisher, width, label="learned position")
    axes[0].bar(x + 1.5 * width, random, width, label="random position")
    axes[0].set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    axes[0].set_ylim(-.05, 1)
    axes[0].set_ylabel("final Fisher-response fidelity")
    axes[0].set_title("Matched 128-coordinate programs")
    axes[0].legend(fontsize=8)

    axes[1].scatter(current, joint, c=[row["boundary"] for row in rows], cmap="viridis", s=65)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
    for index, row in enumerate(rows):
        axes[1].annotate(f"{row['target']}/b{row['boundary']}", (current[index], joint[index]),
                         xytext=(4, 2), textcoords="offset points", fontsize=7)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("current-site factorized fidelity")
    axes[1].set_ylabel("joint-field fidelity")
    axes[1].set_title("Remaining nonseparable gap")
    axes[1].grid(alpha=.25)

    boundaries = [6, 10, 14, 17]
    overlap = []
    rank90 = []
    coverage = []
    for boundary in boundaries:
        cells = [item["boundaries"][str(boundary)] for item in source["diagnostics"]]
        overlap.append(np.median([cell["position_overlap8"] for cell in cells]))
        rank90.append(np.median([cell["position_rank90"] for cell in cells]))
        coverage.append(np.median([max(cell["leading8_readable_coverage"].values()) for cell in cells]))
    axes[2].plot(boundaries, overlap, "o-", label="split-stable overlap")
    axes[2].plot(boundaries, coverage, "o-", label="readable-span coverage")
    axis2 = axes[2].twinx()
    axis2.plot(boundaries, rank90, "s--", color="tab:red", label="position rank90")
    axes[2].set_ylim(0, 1.05); axis2.set_ylim(0, max(rank90) + 2)
    axes[2].set_xticks(boundaries)
    axes[2].set_xlabel("residual boundary")
    axes[2].set_ylabel("overlap / coverage")
    axis2.set_ylabel("rank for 90% position energy", color="tab:red")
    axes[2].set_title("The positional demand is stable and small")
    handles, labels_left = axes[2].get_legend_handles_labels()
    handles2, labels_right = axis2.get_legend_handles_labels()
    axes[2].legend(handles + handles2, labels_left + labels_right, fontsize=8, loc="center right")

    figure.suptitle("Known read locations are real, but do not explain the joint code", fontsize=14)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
