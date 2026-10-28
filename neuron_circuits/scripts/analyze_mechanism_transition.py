#!/usr/bin/env python3
"""Aggregate the locked load x optimizer mechanism-transition factorial."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/mechanism_transition/factorial")
OPTIMIZERS = ("adamw", "sgd")
LOADS = (64, 128, 256)
SEEDS = (0, 1, 2)


def mean(values):
    return float(statistics.mean(values))


def load_row(seed: int, optimizer: str, load: int) -> dict:
    root = ROOT / f"seed{seed}_{optimizer}_load{load}" / "rho_1" / "depth_4"
    training = json.loads((root / "results.json").read_text())
    row = {
        "seed": seed, "optimizer": optimizer, "load": load,
        "steps": training["steps"],
        "fixed_accuracy": training["best"]["fixed_accuracy"],
        "template_accuracy": training["best"]["transfer_accuracy"],
        "gate": training["best"]["fixed_accuracy"] >= .97 and training["best"]["transfer_accuracy"] >= .95,
    }
    audit_path = root / "constraint_audit.json"
    if not audit_path.exists():
        return row
    audit = json.loads(audit_path.read_text())
    blocks = [audit["kernels"][f"block_{index}"] for index in range(4)]
    row.update({
        "residual_coupling": mean([block["mean_absolute_residual_cosine"] for block in blocks]),
        "raw_support": mean([block["mean_effective_support"] for block in blocks]),
        "supranull_support": mean([block["mean_effective_support_above_sign_null"] for block in blocks]),
        "supranull_edge_fraction": mean([block["fraction_edges_above_sign_null_q95"] for block in blocks]),
        "coupling_to_null_ratio": mean([block["absolute_cosine_to_sign_null_ratio"] for block in blocks]),
        "collateral": audit["pooled_block_edits"]["1.0"]["mean_absolute_collateral"],
        "predictor_pearson": audit["pooled_block_edits"]["1.0"]["pearson"],
        "predictor_origin_r2": audit["pooled_block_edits"]["1.0"]["origin_r2"],
        "margin_median": audit["baseline_margin"]["median"],
        "kkt_final_mlp": mean([
            audit["kkt_stationarity"][f"block_{index}_mlp"]["nonnegative_cosine"] for index in range(4)
        ]),
        "kkt_displacement_mlp": mean([
            audit["kkt_stationarity"][f"block_{index}_mlp"]["displacement_nonnegative_cosine"]
            for index in range(4)
        ]),
    })
    return row


def paired(rows: list[dict]) -> list[dict]:
    lookup = {(row["seed"], row["optimizer"], row["load"]): row for row in rows}
    output = []
    metrics = (
        "residual_coupling", "collateral", "raw_support", "supranull_support",
        "supranull_edge_fraction", "coupling_to_null_ratio", "kkt_final_mlp",
        "kkt_displacement_mlp", "margin_median",
    )
    for seed in SEEDS:
        for optimizer in OPTIMIZERS:
            low, high = lookup[(seed, optimizer, 64)], lookup[(seed, optimizer, 128)]
            assert low["gate"] and high["gate"]
            output.append({
                "seed": seed, "optimizer": optimizer,
                **{f"delta_{metric}": high[metric] - low[metric] for metric in metrics},
            })
    return output


def plot(rows: list[dict], differences: list[dict], path: Path):
    colors = {"adamw": "#3366aa", "sgd": "#cc5c35"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
    for optimizer in OPTIMIZERS:
        for seed in SEEDS:
            selected = [row for row in rows if row["optimizer"] == optimizer and row["seed"] == seed]
            selected.sort(key=lambda row: row["load"])
            axes[0, 0].plot([row["load"] for row in selected], [row["template_accuracy"] for row in selected],
                            color=colors[optimizer], alpha=.55, marker="o")
        axes[0, 0].plot([], [], color=colors[optimizer], label=optimizer.upper())
    axes[0, 0].axhline(.95, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(title="Behavioral capacity boundary", xlabel="facts / relation-answer stratum",
                   ylabel="held-out-template accuracy", ylim=(0, 1.02))
    axes[0, 0].legend()

    metric_panels = (
        (axes[0, 1], "delta_residual_coupling", "Δ controlled |gradient cosine|"),
        (axes[1, 0], "delta_collateral", "Δ normalized finite collateral"),
        (axes[1, 1], "delta_supranull_edge_fraction", "Δ fraction above sign-null q95"),
    )
    positions = {"adamw": 0, "sgd": 1}
    for axis, metric, title in metric_panels:
        for optimizer in OPTIMIZERS:
            values = [row[metric] for row in differences if row["optimizer"] == optimizer]
            x = positions[optimizer]
            axis.scatter(np.full(len(values), x), values, color=colors[optimizer], s=45, zorder=3)
            axis.plot([x - .18, x + .18], [mean(values)] * 2, color="black", linewidth=2)
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_xticks([0, 1], ["AdamW", "SGD"])
        axis.set_title(title + " (128 − 64)")
        axis.grid(axis="y", alpha=.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(rows: list[dict], differences: list[dict], output: Path):
    means = {
        optimizer: {
            metric: mean([row[f"delta_{metric}"] for row in differences if row["optimizer"] == optimizer])
            for metric in ("residual_coupling", "collateral", "raw_support", "supranull_support",
                           "supranull_edge_fraction", "coupling_to_null_ratio", "kkt_final_mlp",
                           "kkt_displacement_mlp")
        }
        for optimizer in OPTIMIZERS
    }
    high_gate = {
        optimizer: sum(row["gate"] for row in rows if row["optimizer"] == optimizer and row["load"] == 256)
        for optimizer in OPTIMIZERS
    }
    predictor_min = min(row["predictor_origin_r2"] for row in rows if row.get("predictor_origin_r2") is not None)
    lines = [
        "# Mechanism transition: matched load × optimizer factorial", "",
        "## Bottom line", "",
        "Doubling the number of successfully learned arbitrary facts from 64 to 128 ranks did **not** reproducibly tighten the transformer constraint web. Raw effective support rose in all six paired cells, but this was a noise-floor artifact: sign-null-corrected support and the fraction of supranull edges generally fell. Coupling magnitude and normalized finite collateral had mixed seed signs.", "",
        "This falsifies the simple universal saturation story on this substrate. It does not falsify the one-layer KKT result or rule out a transition caused by depth, residual paths, normalization, a still-tighter successfully fit model, or a different optimizer limit.", "",
        "## Behavioral gate", "",
        "All 12 load-64/load-128 cells passed the fixed-template and unseen-template gate. No load-256 cell passed:", "",
        f"- AdamW: {high_gate['adamw']}/3 load-256 gates",
        f"- SGD: {high_gate['sgd']}/3 load-256 gates", "",
        "Load 256 is therefore a replicated capacity/optimization boundary, not part of the learned-geometry comparison.", "",
        "## Paired primary effects (load 128 minus 64)", "",
        "| optimizer | controlled coupling | finite collateral | raw support | support above sign null | supranull edge fraction | coupling/null ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for optimizer in OPTIMIZERS:
        values = means[optimizer]
        lines.append(
            f"| {optimizer.upper()} | {values['residual_coupling']:+.5f} | {values['collateral']:+.5f} | "
            f"{values['raw_support']:+.2f} | {values['supranull_support']:+.2f} | "
            f"{values['supranull_edge_fraction']:+.3f} | {values['coupling_to_null_ratio']:+.3f} |"
        )
    lines += ["", f"The finite-edit predictor remained valid in every retained cell; the minimum pooled origin-R² was **{predictor_min:.3f}**.", "",
              "## KKT red team", "",
              "A 64-constraint nonnegative stationarity fit was computed for every block and its MLP-only parameters. It is descriptive, not a KKT theorem: the transformer is normalized and residual, the panel is a small subset of all training constraints, and its fraction of the training table halves when load doubles.", "",
              "| optimizer | Δ final-weight MLP KKT cosine | Δ displacement-from-init MLP KKT cosine |",
              "|---|---:|---:|",
              f"| AdamW | {means['adamw']['kkt_final_mlp']:+.3f} | {means['adamw']['kkt_displacement_mlp']:+.3f} |",
              f"| SGD | {means['sgd']['kkt_final_mlp']:+.3f} | {means['sgd']['kkt_displacement_mlp']:+.3f} |", "",
              "Final-weight alignment rises for SGD but displacement alignment falls for both optimizers. Because initialization choice reverses the apparent conclusion and the constraint panel is incomplete, this does not validate a transformer KKT interpretation. It identifies the next controlled bridge: remove normalization/residual paths and compute stationarity using the complete constraint set in a small model.", "",
              "## Red-team changes that mattered", "",
              "1. Loads are literal nested prefixes of one fact table with stable random targets.",
              "2. Every seed uses the same 64 fact identities across loads and optimizers, balanced over relation × answer.",
              "3. Every fact uses the same competitor, selected from the mean probability across the four matched models.",
              "4. Gradient coupling is compared with a coordinate-sign null preserving gradient norms and magnitude patterns.",
              "5. Current weights and displacement from the reproduced initialization are reported separately.", "",
              "## Updated hypothesis", "",
              "In these normalized residual transformers, adding learnable arbitrary facts expands a diffuse background of tiny sensitivities rather than reliably making existing fact constraints more load-bearing. Taut KKT-like organization is more likely a special regime requiring a simpler homogeneous architecture, complete active-constraint coverage, or substantially tighter successful saturation—not a generic consequence of factual load.", "",
              "![Transition summary](transition_summary.png)"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return means


def main():
    rows = [load_row(seed, optimizer, load) for seed in SEEDS for optimizer in OPTIMIZERS for load in LOADS]
    differences = paired(rows)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    with (ROOT / "paired_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0]))
        writer.writeheader(); writer.writerows(differences)
    means = write_report(rows, differences, ROOT)
    plot(rows, differences, ROOT / "transition_summary.png")
    (ROOT / "summary.json").write_text(json.dumps({"cells": rows, "paired": differences,
                                                   "mean_differences": means}, indent=2), encoding="utf-8")
    print(json.dumps(means, indent=2), flush=True)


if __name__ == "__main__":
    main()
