#!/usr/bin/env python3
"""Visual summary of static, native, and context-conditioned state vocabularies."""

import json

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


STATIC = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.json"
NATIVE = TENSOR_ROOT / "bilin18_transfer_state_native_control.json"
DYNAMIC = TENSOR_ROOT / "bilin18_transfer_state_dynamic_chart.json"
FIGURE = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.png"


def main():
    static = json.loads(STATIC.read_text())
    native = json.loads(NATIVE.read_text())
    dynamic = json.loads(DYNAMIC.read_text())
    targets = list(dynamic["results"])
    boundaries = [str(value) for value in static["config"]["boundaries"]]
    colors = {"L0H0": "#4c78a8", "L2H2": "#f58518", "L5H3": "#54a24b"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    x = np.arange(len(boundaries))
    for target in targets:
        axes[0].plot(x, [static["results"][target][b]["causal"]["shared64"]["response_fidelity"] for b in boundaries],
                     marker="o", ls="--", alpha=.65, color=colors[target], label=f"{target} static64")
        axes[0].plot(x, [dynamic["results"][target][b]["ranks"]["64"]["response_fidelity"] for b in boundaries],
                     marker="o", color=colors[target], label=f"{target} dynamic64")
    axes[0].axhline(.9, color="black", ls=":", lw=.8)
    axes[0].set_xticks(x, boundaries)
    axes[0].set_xlabel("absolute boundary layer")
    axes[0].set_ylabel("final Fisher-response fidelity")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("Context charts beat static axes")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    width = .23
    means = {}
    for name in ("static", "native", "dynamic64", "dynamic128"):
        means[name] = []
    for boundary in boundaries:
        means["static"].append(np.mean([static["results"][t][boundary]["causal"]["shared64"]["response_fidelity"] for t in targets]))
        means["native"].append(np.mean([native["results"][t][boundary]["causal_fidelity"]["native_basis"] for t in targets]))
        means["dynamic64"].append(np.mean([dynamic["results"][t][boundary]["ranks"]["64"]["response_fidelity"] for t in targets]))
        means["dynamic128"].append(np.mean([dynamic["results"][t][boundary]["ranks"]["128"]["response_fidelity"] for t in targets]))
    for offset, (name, values) in zip((-.345, -.115, .115, .345), means.items()):
        axes[1].bar(x + offset, values, width, label=name)
    axes[1].set_xticks(x, boundaries)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel("absolute boundary layer")
    axes[1].set_ylabel("mean Fisher fidelity")
    axes[1].set_title("Basis choice and budget")
    axes[1].legend(frameon=False, fontsize=8)

    for target in targets:
        axes[2].plot(x, [dynamic["results"][target][b]["rank90_median"] for b in boundaries],
                     marker="o", color=colors[target], label=f"{target} rank90")
    overlap = [np.mean([dynamic["results"][t][b]["mean_pairwise_rank64_overlap"] for t in targets]) for b in boundaries]
    secondary = axes[2].twinx()
    secondary.plot(x, overlap, color="black", marker="s", ls="--", label="chart overlap")
    axes[2].set_xticks(x, boundaries)
    axes[2].set_xlabel("absolute boundary layer")
    axes[2].set_ylabel("same-context local rank90")
    secondary.set_ylabel("mean cross-context overlap")
    secondary.set_ylim(0, .5)
    axes[2].set_title("Charts are broad and rotate")
    handles, labels = axes[2].get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axes[2].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("Transported tangent state: no small static vocabulary", y=1.02)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(FIGURE)


if __name__ == "__main__":
    main()
