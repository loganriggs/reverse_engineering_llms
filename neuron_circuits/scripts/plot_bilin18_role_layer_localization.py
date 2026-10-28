#!/usr/bin/env python3
"""Plot suffix-role localization and hard-address threshold sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/bilin18_tensor_decomposition"
SOURCE = ARTIFACT / "bilin18_transfer_state_role_layer_localization.json"
OUTPUT = ARTIFACT / "bilin18_role_layer_localization.png"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    widths = [2, 4, 8]
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)

    for role, color in (("kv", "tab:blue"), ("qv", "tab:orange")):
        axes[0].plot(widths, [source["median_jaccard"][f"{role}_near{width}"] for width in widths],
                     "o-", color=color, label=f"{role.upper()} near boundary")
        axes[0].plot(widths, [source["median_jaccard"][f"{role}_far{width}"] for width in widths],
                     "o--", color=color, alpha=.65, label=f"{role.upper()} near output")
    axes[0].set_xticks(widths); axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("enabled attention-role layers")
    axes[0].set_ylabel("median top-8 address Jaccard")
    axes[0].set_title("Address construction is boundary-local")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.25)

    targets = list(source["results"])
    x = np.arange(len(targets)); width_bar = .13
    methods = ["self_qv_near2", "self_qv_near4", "self_qv_near8", "self_qv_far4", "self_qv_all"]
    labels = ["near2", "near4", "near8", "far4", "all"]
    for index, (method, label) in enumerate(zip(methods, labels)):
        values = [source["results"][target][method]["response_fidelity"] for target in targets]
        axes[1].bar(x + (index - 2) * width_bar, values, width_bar, label=label)
    axes[1].set_xticks(x, targets); axes[1].set_ylim(0, .8)
    axes[1].set_ylabel("boundary-6 causal fidelity")
    axes[1].set_title("Four Q+V layers suffice behaviorally")
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=.25)

    points = []
    for context in source["diagnostics"]:
        for method in ("kv_all", "kv_near2", "kv_near4", "kv_near8", "kv_far8"):
            row = context["programs"][method]
            points.append((row["cosine"], row["top_jaccard"], method))
    colors = {"kv_all": "black", "kv_near2": "tab:green", "kv_near4": "tab:blue", "kv_near8": "tab:purple", "kv_far8": "tab:red"}
    for method in colors:
        subset = [(cosine, jaccard) for cosine, jaccard, name in points if name == method]
        axes[2].scatter([x for x, _ in subset], [y for _, y in subset], label=method, color=colors[method], s=45)
    axes[2].set_xlim(.985, 1.0005); axes[2].set_ylim(.35, 1.05)
    axes[2].set_xlabel("weighted address-demand cosine")
    axes[2].set_ylabel("hard top-8 Jaccard")
    axes[2].set_title("Hard top-k identity is threshold-sensitive")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=.25)

    figure.suptitle("Layer-local role programs: geometry, hard addresses, and behavior", fontsize=14)
    figure.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
