#!/usr/bin/env python3
"""Plot the exact native-channel sufficiency frontier."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/grammar_regimes")


def main():
    payload = json.loads((ROOT / "grammar_last_block_channel_frontier.json").read_text())
    counts = payload["config"]["channel_counts"]
    positions = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for rho, label, color in ((0.0, "Pure reusable rules", "#2474b5"),
                              (0.5, "50% arbitrary", "#b2182b")):
        runs = [row for row in payload["runs"] if row["rho"] == rho]
        values = np.asarray([[row["frontier"][str(count)] for count in counts] for row in runs])
        ax.plot(positions, values.mean(0), marker="o", color=color, label=f"{label}: fixed bank")
        ax.fill_between(positions, values.min(0), values.max(0), color=color, alpha=.15)
        fixed32 = values[:, counts.index(32)].mean()
        random32 = np.mean([row["random32_p95"] for row in runs])
        site32 = np.mean([row["site_top32_fidelity"] for row in runs])
        position32 = counts.index(32)
        ax.scatter([position32], [random32], marker="x", s=70, color=color,
                   label=f"{label}: random-32 p95")
        ax.scatter([position32], [site32], marker="*", s=110, color=color,
                   label=f"{label}: per-site top-32")
        ax.annotate(f"fixed {fixed32:.2f}", (position32, fixed32), xytext=(7, -14),
                    textcoords="offset points", color=color, fontsize=8)
    ax.axhline(.8, color="black", lw=.8, ls="--", label="0.80 fidelity")
    ax.set(xlabel="Retained final-block bilinear channels", ylabel="Complete Layer-0 operator fidelity",
           title="Pure rules reuse a compact bank; mixed memory routes a broad bank", xlim=(-.3, len(counts) - .7), ylim=(0, 1.04))
    ax.set_xticks(positions, labels=counts); ax.legend(frameon=False, fontsize=8, ncol=2)
    destination = ROOT / "grammar_last_block_channel_frontier.png"
    fig.savefig(destination, dpi=180); print("wrote", destination)


if __name__ == "__main__":
    main()
