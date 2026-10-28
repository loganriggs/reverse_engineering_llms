#!/usr/bin/env python3
"""Plot the prospective predictor, exact order audit, and transfer conditioning."""

import json

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


PREDICTOR = TENSOR_ROOT / "bilin18_destination_path_predictor.json"
ORDER = TENSOR_ROOT / "bilin18_destination_attention_order_exact.json"
TRANSFER = TENSOR_ROOT / "bilin18_destination_transfer_conditioning.json"
FIGURE = TENSOR_ROOT / "bilin18_destination_transfer_summary.png"


def main():
    predictor = json.loads(PREDICTOR.read_text())
    order = json.loads(ORDER.read_text())
    transfer = json.loads(TRANSFER.read_text())
    colors = {"L0H0": "#4c78a8", "L2H2": "#f58518", "L5H3": "#54a24b"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    labels, analytic, raw = [], [], []
    for target, target_row in predictor["targets"].items():
        for half, row in target_row["halves"].items():
            labels.append(f"{target}/{half}")
            analytic.append(row["analytic_vs_finite"]["finite_kl_rank_correlation"])
            raw.append(row["analytic_vs_finite"]["raw_norm_finite_kl_rank_correlation"])
    x = np.arange(len(labels))
    axes[0].bar(x - .18, analytic, .36, label="tensor JVP", color="#4c78a8")
    axes[0].bar(x + .18, raw, .36, label="raw norm", color="#e45756")
    axes[0].axhline(0, color="black", lw=.7)
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_ylim(-.65, 1.08)
    axes[0].set_ylabel("Spearman with finite KL")
    axes[0].set_title("Prospective compatibility")
    axes[0].legend(frameon=False)

    for target, target_row in order["targets"].items():
        for half, row in target_row["halves"].items():
            orders = np.arange(target_row["max_order"] + 1)
            error_ratio = np.asarray([
                max(1 - row["orders"][str(value)]["response_fidelity"], 1e-8)
                for value in orders
            ])
            axes[1].plot(
                orders, error_ratio, marker="o", ms=3,
                color=colors[target], alpha=1 if half == "a" else .55,
                label=target if half == "a" else None,
            )
    axes[1].axhline(.1, color="black", ls="--", lw=.8, label="90% fidelity")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("maximum attention-composition order")
    axes[1].set_ylabel("unexplained response energy (1 − fidelity)")
    axes[1].set_title("Low-order paths can explode")
    axes[1].legend(frameon=False)

    for target, target_row in transfer["targets"].items():
        for half, row in target_row["halves"].items():
            profile = row["profile"]
            target_layer = int(target.split("H")[0][1:])
            offsets = [item["layer"] - target_layer for item in profile]
            conditions = [item["cancellation_condition"] for item in profile]
            axes[2].plot(
                offsets, conditions, marker="o", ms=3,
                color=colors[target], alpha=1 if half == "a" else .55,
                label=target if half == "a" else None,
            )
    axes[2].axhline(1, color="black", lw=.7)
    axes[2].set_xlabel("layers after destination")
    axes[2].set_ylabel("local cancellation condition")
    axes[2].set_title("Modest corrections compound")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("Destination features: exact transfer beats raw similarity and path enumeration", y=1.02)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(FIGURE)


if __name__ == "__main__":
    main()
