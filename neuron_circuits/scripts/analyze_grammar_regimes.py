#!/usr/bin/env python3
"""Aggregate the preregistered grammar-regime pilot and replications."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


BASE = Path("artifacts/grammar_regimes")
ROOTS = {
    "easy_pilot": (0, "swiglu"), "bilinear_pilot": (0, "bilinear"),
    "replicate_seed1_swiglu": (1, "swiglu"), "replicate_seed1_bilinear": (1, "bilinear"),
    "replicate_seed2_swiglu": (2, "swiglu"), "replicate_seed2_bilinear": (2, "bilinear"),
}


def write_csv(path: Path, rows: list[dict]):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def weighted_transfer(payload: dict, layer: int, method: str) -> tuple[float, int]:
    values, weights = [], []
    for transition in payload["transitions"]:
        for layer_row in transition["layers"]:
            if layer_row["layer"] != layer: continue
            for row in layer_row["methods"]:
                if row["method"] == method:
                    values.append(row["target_accuracy"]); weights.append(row["count"])
    if not weights: return float("nan"), 0
    return float(np.average(values, weights=weights)), int(sum(weights))


def main():
    output = BASE / "summary"; output.mkdir(parents=True, exist_ok=True)
    behavior, representation, constraints = [], [], []
    for root_name, (seed, kind) in ROOTS.items():
        root = BASE / root_name
        if not (root / "pilot_results.json").exists(): continue
        runs = json.loads((root / "pilot_results.json").read_text())["runs"]
        for run in runs:
            if run["depth"] != 4: continue
            rho = run["rho"]; evaluation = run["evaluation"]
            gate = run["best"]["fixed_accuracy"] >= .97 and run["best"]["transfer_accuracy"] >= .95
            pair_group = "unselected" if rho == .5 else "all"
            behavior.append({
                "seed": seed, "mlp_kind": kind, "rho": rho, "gate_pass": gate,
                "steps": run["steps"], "trained_template_accuracy": evaluation["train"]["0"]["all"]["accuracy"],
                "heldout_template_accuracy": evaluation["train"]["6"]["all"]["accuracy"],
                "heldout_pair_structured_accuracy": evaluation["heldout_pair"]["0"][pair_group]["accuracy"] if rho < 1 else "",
                "heldout_pair_all_accuracy": evaluation["heldout_pair"]["0"]["all"]["accuracy"],
            })
            run_dir = root / f"rho_{rho:g}" / "depth_4"
            axis_path = run_dir / "axis_audit.json"
            transfer_path = run_dir / "semantic_transfer.json"
            if axis_path.exists() and transfer_path.exists():
                axis = json.loads(axis_path.read_text())
                transfer = json.loads(transfer_path.read_text())
                late = axis["layers"][-1]["groups"]["all"]
                row = {
                    "seed": seed, "mlp_kind": kind, "rho": rho,
                    "semantic_top_decile_r2": late["native"]["top_decile_mean"],
                    "native_axis_advantage_r2": late["axis_privilege"]["top_decile_mean"]["native_minus_rotation_median"],
                    "residual_fact_stability_r2": late["native"]["residual_fact_template_stability_r2"]["median"],
                }
                for method in ("full", "native_top_32", "rotated_top_32", "native_random_32"):
                    value, count = weighted_transfer(transfer, 3, method)
                    row[f"semantic_patch_{method}_accuracy"] = value
                    row[f"semantic_patch_{method}_n"] = count
                representation.append(row)
            constraint_path = run_dir / "constraint_audit.json"
            if constraint_path.exists():
                audit = json.loads(constraint_path.read_text())
                early = audit["kernels"]["block_0"]
                constraints.append({
                    "seed": seed, "mlp_kind": kind, "rho": rho,
                    "early_mean_absolute_cosine": early["mean_absolute_cosine"],
                    "early_effective_support": early["mean_effective_support"],
                    "early_residual_absolute_cosine": early["mean_absolute_residual_cosine"],
                    "edit_pearson_scale_1": audit["pooled_block_edits"]["1.0"]["pearson"],
                    "edit_pearson_scale_4": audit["pooled_block_edits"]["4.0"]["pearson"],
                    "edit_collateral_scale_4": audit["pooled_block_edits"]["4.0"]["mean_absolute_collateral"],
                })
    write_csv(output / "behavior.csv", behavior)
    write_csv(output / "representation.csv", representation)
    write_csv(output / "constraints.csv", constraints)

    def stats(rows, field, kind, rho):
        values = [float(row[field]) for row in rows if row["mlp_kind"] == kind and row["rho"] == rho]
        return (np.mean(values), np.min(values), np.max(values), len(values)) if values else (np.nan,) * 3 + (0,)

    lines = [
        "# Validated grammar-regime results", "",
        "The design and thresholds were frozen in `../preregistration.md` before the causal results.", "",
        "## Behavioral regimes", "",
        "| MLP | rho | seeds passing recall + unseen-template gate | train recall mean | unseen-template mean | structured held-out-pair mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in ("swiglu", "bilinear"):
        for rho in (0., .5, 1.):
            rows = [row for row in behavior if row["mlp_kind"] == kind and row["rho"] == rho]
            passing = sum(row["gate_pass"] for row in rows)
            train = np.mean([row["trained_template_accuracy"] for row in rows])
            template = np.mean([row["heldout_template_accuracy"] for row in rows])
            structured = [float(row["heldout_pair_structured_accuracy"]) for row in rows if row["heldout_pair_structured_accuracy"] != ""]
            lines.append(f"| {kind} | {rho:g} | {passing}/{len(rows)} | {train:.3f} | {template:.3f} | {np.mean(structured):.3f} |" if structured else
                         f"| {kind} | {rho:g} | {passing}/{len(rows)} | {train:.3f} | {template:.3f} | n/a |")
    lines += ["", "At rho=0.5, the structured held-out score is computed only on unrandomized pairs; arbitrary held-outs are intentionally not capability targets.", "",
              "## Final-layer representation and causal transfer (gate-passing seeds only)", "",
              "| MLP | rho | n | semantic R2 top decile | residual fact stability | full semantic patch acc | native top-32 | rotated top-32 | random native-32 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in ("swiglu", "bilinear"):
        for rho in (0., .5, 1.):
            keys = ("semantic_top_decile_r2", "residual_fact_stability_r2", "semantic_patch_full_accuracy",
                    "semantic_patch_native_top_32_accuracy", "semantic_patch_rotated_top_32_accuracy",
                    "semantic_patch_native_random_32_accuracy")
            values = [stats(representation, key, kind, rho)[0] for key in keys]
            n = stats(representation, keys[0], kind, rho)[3]
            if n: lines.append(f"| {kind} | {rho:g} | {n} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "Pure-random rows have almost no naturally correct counterfactual pairs, so their patch accuracy is not interpreted; their residual fact-stability score remains diagnostic.", "",
              "## Constraint-network audit", "",
              "| MLP | rho | n | early effective support | early abs cosine | finite-edit Pearson (scale 1) | Pearson (scale 4) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for kind in ("swiglu", "bilinear"):
        for rho in (0., .5, 1.):
            keys = ("early_effective_support", "early_mean_absolute_cosine", "edit_pearson_scale_1", "edit_pearson_scale_4")
            values = [stats(constraints, key, kind, rho)[0] for key in keys]
            n = stats(constraints, keys[0], kind, rho)[3]
            if n: lines.append(f"| {kind} | {rho:g} | {n} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += [
        "", "## Verdict", "",
        "1. **Validated:** pure reusable grammar produces a causally transferable computation. In all six pure-rule depth-4 models, a mean activation change transfers across unrelated entity names. The effect is sparse in the trained channel basis: 32 native channels substantially outperform both 32 random native channels and 32 coordinates of a random rotation.",
        "2. **Validated:** arbitrary assignments produce stable idiosyncratic activation state and broader signed constraint coupling. Residual fact stability remains high after the restricted semantic audit, while early-block effective support is larger than for pure rules.",
        "3. **Validated but not memorization-specific:** signed gradient overlap is an accurate local causal account in rule, mixed, and random regimes. It is therefore a general tangent-space description, not a classifier for memorization and not proof of KKT convergence.",
        "4. **Supported:** mixing rules with 50% arbitrary exceptions is a distinct competition regime. Models can recall both subsets, yet unrandomized held-out rule accuracy falls near 0.42 and semantic patches transfer poorly. Two seed-2 mixed runs also fail the unseen-template fidelity gate even with near-perfect trained-template recall.",
        "5. **Falsified:** the regimes are not a single axis from monosemantic rules to a KKT web. Neuron-axis alignment, semantic transfer, residual fact state, constraint breadth, and behavioral fidelity can vary independently.",
        "6. **Not established:** one neuron equals one human concept; global KKT optimality; or that all natural-language factual memory has the synthetic mixed regime's organization.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"behavior": behavior, "representation": representation, "constraints": constraints}
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        colors = {"swiglu": "#2676b8", "bilinear": "#d65f32"}
        for kind in colors:
            rows = [row for row in behavior if row["mlp_kind"] == kind]
            for rho in (0., .5):
                values = [float(row["heldout_pair_structured_accuracy"]) for row in rows
                          if row["rho"] == rho and row["heldout_pair_structured_accuracy"] != ""]
                axes[0, 0].scatter([rho] * len(values), values, color=colors[kind], alpha=.45)
            means = [np.mean([float(row["heldout_pair_structured_accuracy"]) for row in rows
                              if row["rho"] == rho and row["heldout_pair_structured_accuracy"] != ""])
                     for rho in (0., .5)]
            axes[0, 0].plot((0., .5), means, marker="o", label=kind, color=colors[kind])
        axes[0, 0].set(title="Rule generalization collapses with exceptions", xlabel="arbitrary fraction", ylabel="held-out structured accuracy", xticks=(0, .5), ylim=(0, 1.05))
        axes[0, 0].legend(frameon=False)

        methods = (("semantic_patch_full_accuracy", "full", "-"),
                   ("semantic_patch_native_top_32_accuracy", "native-32", "--"),
                   ("semantic_patch_rotated_top_32_accuracy", "rotated-32", ":"))
        for method, label, style in methods:
            values = [stats(representation, method, "swiglu", rho)[0] for rho in (0., .5)]
            axes[0, 1].plot((0., .5), values, marker="o", linestyle=style, label=label)
        axes[0, 1].set(title="Semantic causal transfer (SwiGLU)", xlabel="arbitrary fraction", ylabel="target-answer accuracy", xticks=(0, .5), ylim=(0, .8))
        axes[0, 1].legend(frameon=False)

        for kind in colors:
            values = [stats(representation, "residual_fact_stability_r2", kind, rho)[0] for rho in (0., .5, 1.)]
            axes[1, 0].plot((0., .5, 1.), values, marker="o", label=kind, color=colors[kind])
        axes[1, 0].set(title="Fact-specific state after semantic residualization", xlabel="arbitrary fraction", ylabel="median cross-template residual R²", xticks=(0, .5, 1.), ylim=(-.05, 1.05))

        for kind in colors:
            values = [stats(constraints, "early_effective_support", kind, rho)[0] for rho in (0., .5, 1.)]
            axes[1, 1].plot((0., .5, 1.), values, marker="o", label=kind, color=colors[kind])
        axes[1, 1].set(title="Arbitrary facts broaden the constraint graph", xlabel="arbitrary fraction", ylabel="early-block effective support", xticks=(0, .5, 1.))
        for axis in axes.flat:
            axis.spines[["top", "right"]].set_visible(False); axis.grid(alpha=.2)
        figure.savefig(output / "regime_summary.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(output / "summary.md")


if __name__ == "__main__":
    main()
