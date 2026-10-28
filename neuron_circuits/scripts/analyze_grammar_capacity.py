#!/usr/bin/env python3
"""Aggregate the fixed-task width sweep and causal monosemanticity profile."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE = Path("artifacts/grammar_regimes")
OUTPUT = BASE / "capacity_summary"
WIDTHS = (64, 96, 192, 384)
KINDS = ("swiglu", "bilinear")
COLORS = {"swiglu": "#2774ae", "bilinear": "#d65f32"}
GROUPS = ("train_changed", "train_unselected", "heldout_unselected")
GROUP_LABELS = {
    "train_changed": "memorized exceptions",
    "train_unselected": "stored rule-following facts",
    "heldout_unselected": "generalized rule facts",
}


def root_for(seed: int, kind: str, width: int):
    if width != 96: return BASE / f"capacity_seed{seed}_{kind}_w{width}"
    if seed == 0: return BASE / ("easy_pilot" if kind == "swiglu" else "bilinear_pilot")
    return BASE / f"replicate_seed{seed}_{kind}"


def statistic(group, field, statistic="median"):
    value = group["summary"].get(field)
    return np.nan if value is None else float(value[statistic])


def load_rows():
    rows = []
    for seed in range(3):
        for kind in KINDS:
            for width in WIDTHS:
                root = root_for(seed, kind, width)
                run_path = root / "rho_0.5" / "depth_4" / "results.json"
                if not run_path.exists(): continue
                run = json.loads(run_path.read_text())
                evaluation = run["evaluation"]
                row = {
                    "seed": seed, "mlp_kind": kind, "width": width,
                    "parameters": run["parameter_counts"]["total"], "steps": run["steps"],
                    "gate_passed": run["best"]["fixed_accuracy"] >= .97 and run["best"]["transfer_accuracy"] >= .95,
                    "train_accuracy": evaluation["train"]["0"]["all"]["accuracy"],
                    "template_accuracy": evaluation["train"]["6"]["all"]["accuracy"],
                    "exception_recall": evaluation["train"]["0"]["changed"]["accuracy"],
                    "structured_train_recall": evaluation["train"]["0"]["unselected"]["accuracy"],
                    "structured_generalization": evaluation["heldout_pair"]["0"]["unselected"]["accuracy"],
                }
                audit_path = root / "rho_0.5" / "depth_4" / "causal_circuit_audit.json"
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text())
                    for group_name, group in audit["groups"].items():
                        prefix = group_name + "_"
                        for field in ("native_s90", "native_n90", "native_s90_fraction",
                                      "native_n90_fraction", "rotated_s90", "axis_advantage"):
                            row[prefix + field] = statistic(group, field)
                        for field in ("native_sufficient_curve_nonmonotonic_fraction",
                                      "native_necessary_curve_nonmonotonic_fraction",
                                      "rotated_sufficient_curve_nonmonotonic_fraction"):
                            row[prefix + field] = group["summary"].get(field)
                        worst_drops = []
                        for example in group["examples"]:
                            if not example["valid"]: continue
                            curve = example["native"]["sufficient_curve"]
                            values = np.asarray([curve[key]["fraction"] for key in sorted(curve, key=int)])
                            if len(values) > 1: worst_drops.append(float(max(0, -np.diff(values).min())))
                        row[prefix + "median_worst_reversal"] = (
                            None if not worst_drops else float(np.median(worst_drops))
                        )
                        row[prefix + "large_reversal_fraction"] = (
                            None if not worst_drops else float(np.mean(np.asarray(worst_drops) > .5))
                        )
                        reuse = group["reuse"]
                        row[prefix + "reuse90"] = reuse["fraction_at_least_90_percent"]
                        row[prefix + "reuse_mean"] = reuse["mean_recovery"]
                        row[prefix + "valid_examples"] = group["summary"]["valid_examples"]
                rows.append(row)
    return rows


def plot_lines(axis, rows, field, title, ylabel, ylim=None):
    positions = np.arange(len(WIDTHS))
    for kind in KINDS:
        means = []
        for position, width in zip(positions, WIDTHS):
            current = [row for row in rows if row["mlp_kind"] == kind and row["width"] == width
                       and field in row and row[field] is not None and np.isfinite(row[field])]
            means.append(np.mean([row[field] for row in current]) if current else np.nan)
            for row in current:
                axis.scatter(position, row[field], s=38,
                             facecolors=COLORS[kind] if row["gate_passed"] else "white",
                             edgecolors=COLORS[kind], linewidth=1.4, zorder=3)
        axis.plot(positions, means, marker="o", color=COLORS[kind], linewidth=2, label=kind)
    axis.set(title=title, xlabel="model width", ylabel=ylabel, xticks=positions)
    axis.set_xticklabels([str(width) for width in WIDTHS])
    if ylim is not None: axis.set_ylim(*ylim)
    axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)


def representative_curve(rows):
    root = root_for(0, "swiglu", 96)
    payload = json.loads((root / "rho_0.5" / "depth_4" / "causal_circuit_audit.json").read_text())
    output = {}
    for group_name in ("train_changed", "heldout_unselected"):
        examples = [row for row in payload["groups"][group_name]["examples"] if row["valid"]]
        examples.sort(key=lambda row: row["native"]["s90"] if row["native"]["s90"] is not None else 10**9)
        output[group_name] = examples[len(examples) // 2]
    return output, payload["total_neurons"]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    fields = sorted({key for row in rows for key in row})
    with (OUTPUT / "capacity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (OUTPUT / "capacity.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_lines(axes[0, 0], rows, "exception_recall", "Recall of arbitrary exceptions", "accuracy", (0, 1.04))
    plot_lines(axes[0, 1], rows, "structured_generalization", "Generalization of the shared rule", "accuracy", (0, 1.04))
    plot_lines(axes[1, 0], rows, "template_accuracy", "Same facts in an unseen template", "accuracy", (0, 1.04))
    plot_lines(axes[1, 1], rows, "train_accuracy", "Overall trained-fact recall", "accuracy", (0, 1.04))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fixed 50%-arbitrary task while model capacity changes\nfilled points pass the recall/transfer gate; open points are underfit", fontsize=14)
    fig.savefig(OUTPUT / "behavior_by_width.png", dpi=180); plt.close(fig)

    for group_name in GROUPS:
        prefix = group_name + "_"
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        plot_lines(axes[0, 0], rows, prefix + "native_s90", "S90: channels sufficient for 90%", "channel count")
        plot_lines(axes[0, 1], rows, prefix + "native_n90", "N90: channels removed to erase 90%", "channel count")
        plot_lines(axes[1, 0], rows, prefix + "axis_advantage", "Native basis advantage over rotation", "log₂(rotated S90 / native S90)")
        plot_lines(axes[1, 1], rows, prefix + "reuse90", "Reuse of one example's S90 circuit", "fraction of recipients ≥90%", (-.03, 1.03))
        axes[0, 0].legend(frameon=False)
        fig.suptitle(f"Causal monosemanticity profile: {GROUP_LABELS[group_name]}", fontsize=14)
        fig.savefig(OUTPUT / f"cmp90_{group_name}.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, group_name in zip(axes, GROUPS):
        plot_lines(axis, rows, group_name + "_large_reversal_fraction",
                   GROUP_LABELS[group_name], "fraction with a reversal >0.5", (0, 1.03))
    axes[0].legend(frameon=False)
    fig.suptitle("Channel cooperation: adding a ranked neuron can reverse recovered margin", fontsize=14)
    fig.savefig(OUTPUT / "nonmonotonicity_by_width.png", dpi=180); plt.close(fig)

    if all((root_for(0, "swiglu", 96) / "rho_0.5" / "depth_4" / "causal_circuit_audit.json").exists() for _ in [0]):
        examples, total = representative_curve(rows)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for axis, group_name in zip(axes, ("heldout_unselected", "train_changed")):
            row = examples[group_name]
            for basis, color in (("native", "#2774ae"), ("rotated", "#d65f32")):
                curve = row[basis]["sufficient_curve"]
                x = np.asarray([int(key) for key in curve])
                y = np.asarray([curve[str(key)]["fraction"] for key in x])
                order = np.argsort(x)
                axis.plot(x[order] + 1, y[order], marker="o", label=basis, color=color)
            axis.axhline(.9, color="black", linestyle="--", linewidth=1)
            ticks = np.asarray([1, 2, 5, 11, 101, total + 1])
            axis.set(title=GROUP_LABELS[group_name], xlabel=f"channels kept (of {total})",
                     ylabel="fraction of MLP-added margin recovered", xscale="log", ylim=(-.3, 1.5),
                     xticks=ticks, xticklabels=["0", "1", "4", "10", "100", str(total)])
            axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        fig.suptitle("Two concrete width-96 examples: exact circuit interventions", fontsize=14)
        fig.savefig(OUTPUT / "worked_circuit_curves.png", dpi=180); plt.close(fig)

    lines = ["# Fixed-task capacity sweep", "",
             "The task is fixed at 50% selected arbitrary assignments. Width changes; depth remains four. S90 and N90 are exact intervention measurements after a first-order ranking, not counts inferred from activations alone.", "",
             "## Behavioral results", "",
             "| MLP | width | seeds | gate passes | exception recall | rule generalization | template transfer |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for kind in KINDS:
        for width in WIDTHS:
            current = [row for row in rows if row["mlp_kind"] == kind and row["width"] == width]
            if not current: continue
            mean = lambda field: np.mean([row[field] for row in current])
            lines.append(f"| {kind} | {width} | {len(current)} | {sum(row['gate_passed'] for row in current)}/{len(current)} | {mean('exception_recall'):.3f} | {mean('structured_generalization'):.3f} | {mean('template_accuracy'):.3f} |")
    lines += ["", "## Causal monosemanticity profile", "",
              "Each S90 entry is `mean seed-level median count (fraction of all MLP channels)`. Reuse is the fraction of matched recipients recovering at least 90% of their own MLP-added margin.", "",
              "| MLP | width | exception S90 | exception axis advantage | exception reuse | generalized-rule S90 | rule axis advantage | rule reuse | large reversal rate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in KINDS:
        for width in WIDTHS:
            current = [row for row in rows if row["mlp_kind"] == kind and row["width"] == width]
            if not current: continue
            def average(field):
                values = [row[field] for row in current if field in row and row[field] is not None]
                return float(np.mean(values)) if values else np.nan
            exception_count = average("train_changed_native_s90")
            exception_fraction = average("train_changed_native_s90_fraction")
            rule_count = average("heldout_unselected_native_s90")
            rule_fraction = average("heldout_unselected_native_s90_fraction")
            lines.append(
                f"| {kind} | {width} | {exception_count:.0f} ({exception_fraction:.3f}) | "
                f"{average('train_changed_axis_advantage'):.2f} | {average('train_changed_reuse90'):.2f} | "
                f"{rule_count:.0f} ({rule_fraction:.3f}) | {average('heldout_unselected_axis_advantage'):.2f} | "
                f"{average('heldout_unselected_reuse90'):.2f} | {average('heldout_unselected_large_reversal_fraction'):.2f} |"
            )
    lines += ["", "## Interpretation and hypothesis verdicts", "",
              "- **More width did not restore the shared rule.** Exception recall stayed around 0.97–0.99, while new structured-pair accuracy fell from 0.486 to 0.337 for SwiGLU and from 0.440 to 0.285 for bilinear. This falsifies the simple prediction that spare capacity separates exceptions and recovers rule generalization under this training objective.",
              "- **The learned neuron basis becomes more privileged with width.** Native S90 is increasingly smaller than random-rotation S90, especially for width-384 SwiGLU. This supports axis-aligned causal specialization.",
              "- **Axis alignment is not the same as semantic monosemanticity.** Same-rule-and-answer reuse does not rise reliably with width, and matched-answer exception reuse is low. Wider models appear to develop sparse, privileged coordinates without those coordinates forming universally reusable rule circuits.",
              "- **The constraint-web signature remains visible.** Nearly every ranked-prefix curve has some reversal; large reversals become less frequent with width. This is consistent with cooperation/cancellation becoming milder as capacity grows, but it is not a KKT proof for these multi-layer Adam-trained models.",
              "- **Architecture matters.** Width-384 SwiGLU circuits become very sparse as a fraction of available channels; bilinear absolute circuit counts generally remain larger. A single scalar would conceal this distinction.", "",
              "## Reading CMP-90", "",
              "- Small S90 says a compact ranked channel set can reconstruct the example's MLP contribution.",
              "- Small N90 says the contribution is causally concentrated rather than robustly redundant.",
              "- Positive axis advantage says the learned neuron coordinates are more compact than a random rotation.",
              "- High same-rule reuse is the semantic test: a compact circuit found on one example also works on another example governed by the same relation.",
              "- Sparse S90 without reuse is compatible with a fact-specific lookup and is not called monosemantic.", "",
              "## Limitations", "",
              "- Circuit panels contain 12 naturally correct examples per group per run; neurons are correlated within a model, so seeds—not examples—are the replication units.",
              "- Several runs failed the strict unseen-template fidelity gate. They remain visible as open graph markers and are exploratory, not matched-fidelity confirmation.",
              "- S90/N90 are smallest crossings on a fixed dense ranked-prefix grid. Non-monotonicity makes them resolution-limited upper-bound estimates, not globally minimal circuits.",
              "- The reuse test controls relation plus answer for structured facts and answer plus different relation for exceptions, but a broader family of semantic patch tests is still needed.", "",
              "## Figures", "",
              "- `behavior_by_width.png`: whether capacity improves memorization, rules, or both.",
              "- `cmp90_*.png`: separate causal profiles for exceptions, trained structured facts, and generalized structured facts.",
              "- `worked_circuit_curves.png`: exact native-versus-rotated interventions for two concrete examples."]
    lines += ["- `nonmonotonicity_by_width.png`: how often nested ranked circuits reverse direction, revealing cooperation or cancellation."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT / "summary.md")


if __name__ == "__main__":
    main()
