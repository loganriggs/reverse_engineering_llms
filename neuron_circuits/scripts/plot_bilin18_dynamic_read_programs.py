#!/usr/bin/env python3
"""Compare fixed, dynamic, and attention-derived read programs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bilin18_tensor_decomposition"
OUTPUT = ARTIFACT / "bilin18_dynamic_read_programs.png"


def load(name):
    return json.loads((ARTIFACT / name).read_text())


def main() -> None:
    read = load("bilin18_transfer_state_read_function_audit.json")
    invocation = load("bilin18_transfer_state_invocation_chart.json")
    dynamic = load("bilin18_transfer_state_dynamic_read_selector.json")
    attention = load("bilin18_transfer_state_attention_address.json")
    boundaries = [6, 10, 14, 17]
    targets = list(read["results"])
    readable_names = ["current_sites4", "current_previous8", "recent4x4", "recent8x4", "segment4", "causal_prefix4"]
    invocation_names = ["current_specific32", "current_previous_specific16", "recent4_specific8", "recent8_specific4"]
    dynamic_names = ["dynamic_top4", "dynamic_top8", "dynamic_top16", "dynamic_top32"]

    series = {name: [] for name in ["joint", "shared readable", "bound calls", "dynamic calls", "QK rollout"]}
    for boundary in boundaries:
        joint = []
        shared = []
        bound = []
        selected = []
        rollout = []
        for target in targets:
            rr = read["results"][target][str(boundary)]
            ir = invocation["results"][target][str(boundary)]
            dr = dynamic["results"][target][str(boundary)]
            ar = attention["results"][target][str(boundary)]
            joint.append(rr["joint128"]["response_fidelity"])
            shared.append(max(rr[name]["response_fidelity"] for name in readable_names))
            bound.append(max(ir[name]["response_fidelity"] for name in invocation_names))
            selected.append(max(dr[name]["response_fidelity"] for name in dynamic_names))
            rollout.append(ar["rollout"]["response_fidelity"])
        for name, values in zip(series, [joint, shared, bound, selected, rollout]):
            series[name].append(np.mean(values))

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    styles = {
        "joint": ("black", "o"),
        "shared readable": ("tab:orange", "s"),
        "bound calls": ("tab:green", "^"),
        "dynamic calls": ("tab:blue", "D"),
        "QK rollout": ("tab:red", "x"),
    }
    for name, values in series.items():
        color, marker = styles[name]
        axes[0].plot(boundaries, values, marker=marker, color=color, label=name, linewidth=2)
    axes[0].set_ylim(0, 1); axes[0].set_xticks(boundaries)
    axes[0].set_xlabel("residual boundary")
    axes[0].set_ylabel("mean causal fidelity")
    axes[0].set_title("A hierarchy of 128-coordinate programs")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.25)

    direct_j = [attention["median_jaccard"]["direct"][str(boundary)] for boundary in boundaries]
    rollout_j = [attention["median_jaccard"]["rollout"][str(boundary)] for boundary in boundaries]
    axes[1].plot(boundaries, direct_j, "o-", label="direct QK")
    axes[1].plot(boundaries, rollout_j, "o-", label="QK rollout")
    axes[1].axhline(.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.05); axes[1].set_xticks(boundaries)
    axes[1].set_xlabel("residual boundary")
    axes[1].set_ylabel("top-8 Jaccard with Fisher addresses")
    axes[1].set_title("QK alone predicts addresses only late")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.25)

    labels = []
    gaps = []
    for target in targets:
        for boundary in boundaries:
            rr = read["results"][target][str(boundary)]
            dr = dynamic["results"][target][str(boundary)]
            best_dynamic = max(dr[name]["response_fidelity"] for name in dynamic_names)
            labels.append(f"{target}\nb{boundary}")
            gaps.append(rr["joint128"]["response_fidelity"] - best_dynamic)
    x = np.arange(len(gaps))
    colors = ["tab:red" if gap >= .1 else "tab:blue" for gap in gaps]
    axes[2].bar(x, gaps, color=colors)
    axes[2].axhline(.1, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(0, color="black", linewidth=.8)
    axes[2].set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    axes[2].set_ylabel("joint minus best dynamic-site fidelity")
    axes[2].set_title("Remaining nonseparable/contextual gap")

    figure.suptitle("Readable functions, bound invocations, and composed routing", fontsize=14)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
