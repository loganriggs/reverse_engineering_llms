#!/usr/bin/env python3
"""Plot prospective suffix-Fisher chart performance and sampling red team."""

import json

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


BASE = TENSOR_ROOT / "bilin18_transfer_state_fisher_chart.json"
HIGH = TENSOR_ROOT / "bilin18_transfer_state_fisher_chart_highsample.json"
REDTEAM = TENSOR_ROOT / "bilin18_transfer_state_fisher_chart_sampling_redteam.json"
FIGURE = TENSOR_ROOT / "bilin18_transfer_state_fisher_chart.png"


def main():
    base = json.loads(BASE.read_text())
    high = json.loads(HIGH.read_text())
    redteam = json.loads(REDTEAM.read_text())
    targets = list(high["results"])
    boundaries = sorted(next(iter(high["results"].values())), key=int)
    colors = {"L0H0": "#4c78a8", "L2H2": "#f58518", "L5H3": "#54a24b"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    x = np.arange(len(boundaries))
    for target in targets:
        row = high["results"][target]
        axes[0].plot(x, [row[b]["static64_fidelity"] for b in boundaries], ls=":", alpha=.6,
                     color=colors[target], marker="o", label=f"{target} static")
        axes[0].plot(x, [row[b]["ranks"]["64"]["response_fidelity"] for b in boundaries],
                     color=colors[target], marker="o", label=f"{target} Fisher")
        axes[0].plot(x, [row[b]["oracle64_fidelity"] for b in boundaries], ls="--", alpha=.75,
                     color=colors[target], marker="o", label=f"{target} oracle")
    axes[0].axhline(.75, color="black", ls="--", lw=.7)
    axes[0].set_xticks(x, boundaries)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_xlabel("boundary layer")
    axes[0].set_ylabel("rank-64 Fisher fidelity")
    axes[0].set_title("Prospective chart closes much of gap")
    axes[0].legend(frameon=False, fontsize=7, ncol=3)

    base64 = np.asarray([row["base64"] for row in redteam["rows"]])
    high64 = np.asarray([row["high64"] for row in redteam["rows"]])
    axes[1].scatter(base64, high64, c=[colors[row["target"]] for row in redteam["rows"]], s=45)
    limits = [min(base64.min(), high64.min()) - .02, max(base64.max(), high64.max()) + .02]
    axes[1].plot(limits, limits, "k--", lw=.8)
    axes[1].set_xlim(limits); axes[1].set_ylim(limits)
    axes[1].set_xlabel("8-top / 8-tail fidelity")
    axes[1].set_ylabel("16-top / 16-tail fidelity")
    axes[1].set_title("Behavior stable across sketches")
    axes[1].text(.04, .94, f"Spearman {redteam['summary']['64']['spearman']:.3f}", transform=axes[1].transAxes, va="top")

    overlap = [redteam["overlap_by_boundary"][b]["highsample"] for b in boundaries]
    rank90 = [np.mean([high["results"][t][b]["rank90_median"] for t in targets]) for b in boundaries]
    axes[2].bar(x, overlap, color="#72b7b2", label="literal overlap")
    axes[2].set_xticks(x, boundaries)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("boundary layer")
    axes[2].set_ylabel("independent-chart overlap")
    secondary = axes[2].twinx()
    secondary.plot(x, rank90, color="#e45756", marker="o", label="Fisher rank90")
    secondary.set_ylabel("sketched Fisher rank90")
    axes[2].set_title("Axes unstable; quotient broader")
    handles, labels = axes[2].get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axes[2].legend(handles + handles2, labels + labels2, frameon=False)

    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("Candidate-independent suffix-Fisher charts", y=1.02)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(FIGURE)


if __name__ == "__main__":
    main()
