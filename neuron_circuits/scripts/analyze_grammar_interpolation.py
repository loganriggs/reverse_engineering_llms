#!/usr/bin/env python3
"""Plot the 0/5/20/50/100% grammar-to-arbitrary interpolation."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("artifacts/grammar_regimes")
OUTPUT = BASE / "interpolation_summary"
RHOS = (0.0, 0.05, 0.2, 0.5, 1.0)
COLORS = {"swiglu": "#2774ae", "bilinear": "#d65f32"}


def roots():
    for seed in range(3):
        yield seed, "swiglu", BASE / ("easy_pilot" if seed == 0 else f"replicate_seed{seed}_swiglu")
        yield seed, "bilinear", BASE / ("bilinear_pilot" if seed == 0 else f"replicate_seed{seed}_bilinear")
        for kind in ("swiglu", "bilinear"):
            yield seed, kind, BASE / f"interpolation_seed{seed}_{kind}"


def weighted_patch(payload, method, layer=3):
    values, counts = [], []
    for transition in payload["transitions"]:
        for layer_row in transition["layers"]:
            if layer_row["layer"] != layer: continue
            for row in layer_row["methods"]:
                if row["method"] == method:
                    values.append(row["target_accuracy"]); counts.append(row["count"])
    return float(np.average(values, weights=counts)) if counts else float("nan")


def load_rows():
    rows = []
    seen = set()
    for seed, kind, root in roots():
        pilot = root / "pilot_results.json"
        if not pilot.exists(): continue
        for run in json.loads(pilot.read_text())["runs"]:
            rho = float(run["rho"])
            if run["depth"] != 4 or rho not in RHOS: continue
            key = (seed, kind, rho)
            if key in seen: continue
            seen.add(key)
            evaluation = run["evaluation"]
            gate = run["best"]["fixed_accuracy"] >= .97 and run["best"]["transfer_accuracy"] >= .95
            train0 = evaluation["train"]["0"]
            heldout = evaluation["heldout_pair"]["0"]
            row = {
                "seed": seed, "mlp_kind": kind, "rho": rho, "gate_passed": gate,
                "train_accuracy": train0["all"]["accuracy"],
                "template_transfer_accuracy": evaluation["train"]["6"]["all"]["accuracy"],
                "structured_pair_accuracy": heldout[("unselected" if rho > 0 else "all")]["accuracy"] if rho < 1 else np.nan,
                "exception_recall": train0.get("changed", {}).get("accuracy", np.nan),
                "steps": run["steps"],
            }
            run_dir = root / f"rho_{rho:g}" / "depth_4"
            axis_path, transfer_path = run_dir / "axis_audit.json", run_dir / "semantic_transfer.json"
            constraint_path = run_dir / "constraint_audit.json"
            if axis_path.exists():
                axis = json.loads(axis_path.read_text())
                late = axis["layers"][-1]["groups"]["all"]
                row.update({
                    "semantic_r2": late["native"]["top_decile_mean"],
                    "axis_r2_advantage": late["axis_privilege"]["top_decile_mean"]["native_minus_rotation_median"],
                    "residual_fact_stability": late["native"]["residual_fact_template_stability_r2"]["median"],
                })
            if transfer_path.exists():
                transfer = json.loads(transfer_path.read_text())
                for method in ("full", "native_top_32", "rotated_top_32", "native_random_32"):
                    row[f"patch_{method}"] = weighted_patch(transfer, method)
            if constraint_path.exists():
                constraint = json.loads(constraint_path.read_text())
                row["constraint_support"] = constraint["kernels"]["block_0"]["mean_effective_support"]
                row["constraint_abs_cosine"] = constraint["kernels"]["block_0"]["mean_absolute_cosine"]
            rows.append(row)
    return rows


def values(rows, kind, rho, field):
    return [float(row[field]) for row in rows if row["mlp_kind"] == kind and row["rho"] == rho
            and field in row and np.isfinite(float(row[field]))]


def plot_field(axis, rows, field, title, ylabel, *, ylim=None):
    positions = np.arange(len(RHOS))
    for kind, color in COLORS.items():
        means = []
        for position, rho in enumerate(RHOS):
            current = [row for row in rows if row["mlp_kind"] == kind and row["rho"] == rho
                       and field in row and np.isfinite(float(row[field]))]
            means.append(np.mean([row[field] for row in current]) if current else np.nan)
            for row in current:
                axis.scatter(position, row[field], s=38, facecolors=color if row["gate_passed"] else "white",
                             edgecolors=color, linewidth=1.4, alpha=.9, zorder=3)
        axis.plot(positions, means, color=color, marker="o", linewidth=2, label=kind)
    axis.set(title=title, xlabel="fraction of arbitrary training facts", ylabel=ylabel,
             xticks=positions, xticklabels=("0%", "5%", "20%", "50%", "100%"))
    if ylim: axis.set_ylim(*ylim)
    axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fields = sorted({key for row in rows for key in row})
    with (OUTPUT / "interpolation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_field(axes[0, 0], rows, "structured_pair_accuracy", "Reusable rule generalization", "accuracy on new structured pairs", ylim=(0, 1.04))
    plot_field(axes[0, 1], rows, "exception_recall", "Recall of changed exceptions", "trained-exception accuracy", ylim=(0, 1.04))
    plot_field(axes[1, 0], rows, "template_transfer_accuracy", "Transfer to an unseen template", "same-fact accuracy", ylim=(0, 1.04))
    plot_field(axes[1, 1], rows, "train_accuracy", "Overall trained-template recall", "accuracy", ylim=(0, 1.04))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Behavior across the grammar-to-memorization interpolation\nfilled points pass the fidelity gate; open points are exploratory", fontsize=15)
    fig.savefig(OUTPUT / "behavior_interpolation.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_field(axes[0, 0], rows, "patch_native_top_32", "Native-neuron semantic patch transfer", "target-answer success", ylim=(-.02, .85))
    plot_field(axes[0, 1], rows, "residual_fact_stability", "Stable fact-specific state after semantic residualization", "median cross-template residual R²", ylim=(-.08, 1.04))
    plot_field(axes[1, 0], rows, "constraint_support", "Early-layer constraint-network breadth", "effective support")
    plot_field(axes[1, 1], rows, "semantic_r2", "Semantic predictability of the best native channels", "top-decile held-out R²", ylim=(-.05, 1.04))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Representation changes as arbitrary load increases\nopen points failed the strict fidelity gate and are exploratory", fontsize=15)
    fig.savefig(OUTPUT / "representation_interpolation.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, kind in zip(axes, ("swiglu", "bilinear")):
        positions = np.arange(len(RHOS))
        for field, label, style in (("patch_full", "all 192 channels", "-"),
                                    ("patch_native_top_32", "top 32 native", "--"),
                                    ("patch_rotated_top_32", "top 32 rotated", ":"),
                                    ("patch_native_random_32", "random 32 native", "-.")):
            means = []
            for rho in RHOS:
                current = values(rows, kind, rho, field)
                means.append(np.mean(current) if current else np.nan)
            ax.plot(positions, means, marker="o", linestyle=style, label=label)
        ax.set(title=kind, xlabel="fraction arbitrary", ylabel="counterfactual target accuracy",
               xticks=positions, xticklabels=("0%", "5%", "20%", "50%", "100%"), ylim=(-.02, .85))
        ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("What the patch numbers mean: fraction of prompts switched to the intended new answer", fontsize=14)
    fig.savefig(OUTPUT / "patch_interpolation.png", dpi=180); plt.close(fig)

    lines = ["# Grammar-to-memorization interpolation", "",
             "Filled graph markers pass the original recall plus unseen-template gate. Open markers are exploratory audits on naturally correct examples and are not used as matched-fidelity confirmation.", "",
             "## Seed-level behavioral table", "",
             "| MLP | arbitrary fraction | gate passes | train recall | exception recall | new structured pairs | unseen template |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for kind in ("swiglu", "bilinear"):
        for rho in RHOS:
            current = [row for row in rows if row["mlp_kind"] == kind and row["rho"] == rho]
            def mean(field):
                v = [float(row[field]) for row in current if np.isfinite(float(row[field]))]
                return f"{np.mean(v):.3f}" if v else "n/a"
            lines.append(f"| {kind} | {rho:g} | {sum(row['gate_passed'] for row in current)}/{len(current)} | {mean('train_accuracy')} | {mean('exception_recall')} | {mean('structured_pair_accuracy')} | {mean('template_transfer_accuracy')} |")
    lines += ["", "## Interpretation", "",
              "- Rule generalization declines smoothly once exceptions become numerous: near 1.00 at 0%, about 0.98 at 5%, about 0.78 at 20%, and about 0.43 at 50%.",
              "- Exception acquisition is not smooth across seeds at 5%: some models preserve the rule and largely ignore rare exceptions, while others fit exceptions but become template-specific. This is a competing-solution or rare-exception optimization effect.",
              "- At 20%, exceptions are usually learned, while structured generalization is damaged but far from destroyed. This supplies the requested intermediate regime.",
              "- Representation curves use all seeds for visualization, but open points remain exploratory because prompt-interface brittleness can contaminate mechanistic comparisons.", "",
              "## Figures", "",
              "- `behavior_interpolation.png`: task performance and gate status.",
              "- `representation_interpolation.png`: neuron transfer, instance state, constraint breadth, and semantic predictability.",
              "- `patch_interpolation.png`: direct explanation of full/native/rotated/random patch success."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT / "interpolation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(OUTPUT / "summary.md")


if __name__ == "__main__":
    main()
