#!/usr/bin/env python3
"""Multi-start stability of weight-only approximate joint diagonalization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_layer0_joint_forms import ARRAYS as LAYER0_ARRAYS
from run_bilin18_layer17_joint_forms import ARRAYS as LAYER17_ARRAYS
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_joint_diagonalization_stability.json")
OUTPUT = TENSOR_ROOT / "joint_diagonalization_stability.json"
REPORT = TENSOR_ROOT / "JOINT_DIAGONALIZATION_STABILITY.md"
FIGURE = TENSOR_ROOT / "joint_diagonalization_stability.png"

ORIGINAL = ["newline", "suffix_ing", "suffix_ed", "suffix_ly", "all_caps"]


def axis_match(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    overlap = (reference.T @ candidate).abs().cpu().numpy()
    forward = overlap.argmax(1)
    backward = overlap.argmax(0)
    row_max = overlap.max(1)
    mutual = np.asarray([backward[column] == row for row, column in enumerate(forward)])
    return {
        "mean_best_absolute_overlap": float(row_max.mean()),
        "median_best_absolute_overlap": float(np.median(row_max)),
        "minimum_best_absolute_overlap": float(row_max.min()),
        "mutual_nearest_fraction": float(mutual.mean()),
        "unique_candidate_fraction": float(len(set(forward.tolist())) / len(forward)),
    }


def summarize_runs(runs: list[dict]) -> dict:
    losses = np.asarray([run["training"]["final_off_diagonal_fraction"] for run in runs])
    comparisons = [run["axis_match_to_reference"] for run in runs[1:]]
    return {
        "final_losses": losses.tolist(),
        "loss_mean": float(losses.mean()),
        "loss_std": float(losses.std()),
        "loss_coefficient_of_variation": float(losses.std() / max(losses.mean(), 1e-30)),
        "mean_best_axis_overlap": float(np.mean([row["mean_best_absolute_overlap"] for row in comparisons])),
        "median_best_axis_overlap": float(np.median([row["median_best_absolute_overlap"] for row in comparisons])),
        "mean_mutual_nearest_fraction": float(np.mean([row["mutual_nearest_fraction"] for row in comparisons])),
        "mean_unique_candidate_fraction": float(np.mean([row["unique_candidate_fraction"] for row in comparisons])),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    dimension = config["dimension"]
    layer0 = np.load(LAYER0_ARRAYS)
    layer17 = np.load(LAYER17_ARRAYS)
    families = {
        "layer0": torch.stack([
            torch.from_numpy(layer0[f"embedding_{category}_restricted_form"][:dimension, :dimension]).cuda()
            for category in ORIGINAL
        ]),
        "layer17": torch.stack([
            torch.from_numpy(layer17[f"{category}_restricted_form"][:dimension, :dimension]).cuda()
            for category in ORIGINAL
        ]),
    }
    all_runs = {}
    summaries = {}
    for family, forms in families.items():
        runs = []
        reference = None
        for start in range(config["starts"]):
            if start == 0:
                initial = torch.zeros(dimension, dimension, device="cuda", dtype=torch.float64)
            else:
                generator = torch.Generator(device="cuda").manual_seed(
                    config["seed"] + 1000 * (1 if family == "layer0" else 2) + start
                )
                initial = config["random_initial_scale"] * torch.randn(
                    dimension,
                    dimension,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float64,
                )
            rotation, training = train_rotation(forms, config, initial)
            rotation = rotation.detach()
            if reference is None:
                reference = rotation
                match = {
                    "mean_best_absolute_overlap": 1.0,
                    "median_best_absolute_overlap": 1.0,
                    "minimum_best_absolute_overlap": 1.0,
                    "mutual_nearest_fraction": 1.0,
                    "unique_candidate_fraction": 1.0,
                }
            else:
                match = axis_match(reference, rotation)
            runs.append({"start": start, "training": training, "axis_match_to_reference": match})
            print(family, start, training["final_off_diagonal_fraction"], match["mean_best_absolute_overlap"], flush=True)
        all_runs[family] = runs
        summaries[family] = summarize_runs(runs)

    hypotheses = {
        "S1_objective_stability": {
            "supported": all(row["loss_coefficient_of_variation"] < 0.05 for row in summaries.values()),
            "per_layer": {name: row["loss_coefficient_of_variation"] for name, row in summaries.items()},
        },
        "S2_layer0_axis_recovery": {
            "supported": summaries["layer0"]["mean_best_axis_overlap"] >= 0.90
            and summaries["layer0"]["mean_mutual_nearest_fraction"] >= 0.80,
            **summaries["layer0"],
        },
        "S3_layer17_axis_recovery": {
            "supported": summaries["layer17"]["mean_best_axis_overlap"] >= 0.90
            and summaries["layer17"]["mean_mutual_nearest_fraction"] >= 0.80,
            **summaries["layer17"],
        },
        "S4_depth_stability_gap": {
            "supported": summaries["layer0"]["mean_best_axis_overlap"]
            - summaries["layer17"]["mean_best_axis_overlap"] >= 0.05,
            "layer0": summaries["layer0"]["mean_best_axis_overlap"],
            "layer17": summaries["layer17"]["mean_best_axis_overlap"],
            "gap": summaries["layer0"]["mean_best_axis_overlap"]
            - summaries["layer17"]["mean_best_axis_overlap"],
        },
    }
    output = {
        "config": config,
        "summaries": summaries,
        "hypotheses": hypotheses,
        "runs": all_runs,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Multi-start joint-diagonalization stability",
        "",
        "Each family is optimized once from identity and five times from independent random skew initializations. Axis comparison quotients sign and permutation by matching absolute column overlaps to the identity-start solution. Row-best and mutual-nearest statistics are reported because no external assignment package is used.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Stability](joint_diagonalization_stability.png)",
        "",
        "| Family | Loss mean | Loss CV | Best-axis overlap | Mutual-nearest axes | Unique matched axes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family, row in summaries.items():
        lines.append(
            f"| {family} | {row['loss_mean']:.4f} | {row['loss_coefficient_of_variation']:.3f} | "
            f"{row['mean_best_axis_overlap']:.3f} | {row['mean_mutual_nearest_fraction']:.1%} | "
            f"{row['mean_unique_candidate_fraction']:.1%} |"
        )
    lines += ["", "Per-start traces and matching diagnostics are in `joint_diagonalization_stability.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for family, runs in all_runs.items():
        axes[0].plot(
            range(config["starts"]),
            [run["training"]["final_off_diagonal_fraction"] for run in runs],
            marker="o",
            label=family,
        )
        axes[1].plot(
            range(1, config["starts"]),
            [run["axis_match_to_reference"]["mean_best_absolute_overlap"] for run in runs[1:]],
            marker="o",
            label=family,
        )
    axes[0].set_xlabel("start")
    axes[0].set_ylabel("final off-diagonal fraction")
    axes[1].set_xlabel("random start")
    axes[1].set_ylabel("mean best axis overlap")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
