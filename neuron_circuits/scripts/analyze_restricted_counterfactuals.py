#!/usr/bin/env python3
"""Analyze location-restricted MLP counterfactual-patching sweeps."""

from __future__ import annotations

import base64
from collections import Counter
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/restricted_counterfactuals")
BASE_ROOT = Path("artifacts/counterfactual_patching")
FAMILIES = ("ioi", "agreement", "brackets")
FAMILY_LABELS = {"ioi": "IOI", "agreement": "Agreement", "brackets": "Brackets"}
CONDITIONS = (
    "unrestricted",
    "exclude_final_layer",
    "exclude_final_four_layers",
    "exclude_final_position",
    "exclude_final_four_layers_and_position",
)
CONDITION_LABELS = {
    "unrestricted": "Unrestricted",
    "exclude_final_layer": "No layer 23",
    "exclude_final_four_layers": "No layers 20–23",
    "exclude_final_position": "No final position",
    "exclude_final_four_layers_and_position": "No layers 20–23\nor final position",
}
COLORS = ("#64748b", "#38bdf8", "#2563eb", "#f97316", "#dc2626")


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def first_crossovers(rows, family: str, condition: str | None = None):
    pair_ids = {row["pair_id"] for row in rows if row["family"] == family}
    output = {}
    for pair_id in pair_ids:
        successes = [
            row["patched"] for row in rows
            if row["pair_id"] == pair_id
            and row["method"] == "counterfactual_gradient"
            and row["resolution"] == "instance"
            and row["target_top1"]
            and (condition is None or row.get("condition") == condition)
        ]
        if successes:
            output[pair_id] = min(successes)
    return output


def collect(base_rows, restricted_rows):
    values = {}
    for family in FAMILIES:
        values[family] = {"unrestricted": first_crossovers(base_rows, family)}
        for condition in CONDITIONS[1:]:
            values[family][condition] = first_crossovers(
                restricted_rows, family, condition
            )
    return values


def plot_medians(values):
    fig, ax = plt.subplots(figsize=(12, 5.4))
    x = np.arange(len(FAMILIES))
    width = .16
    for index, (condition, color) in enumerate(zip(CONDITIONS, COLORS, strict=True)):
        medians = [statistics.median(values[family][condition].values()) for family in FAMILIES]
        bars = ax.bar(x + (index - 2) * width, medians, width, color=color, label=CONDITION_LABELS[condition].replace("\n", " "))
        ax.bar_label(bars, fmt="%g", padding=2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, [FAMILY_LABELS[family] for family in FAMILIES])
    ax.set_ylabel("Median targeted activations at strict top-1 flip")
    ax.set_title("Final-position exclusion matters more than final-layer exclusion")
    ax.grid(axis="y", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "restricted_counterfactual_medians.png", dpi=180)
    plt.close(fig)


def plot_pairs(values):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    positions = np.arange(len(CONDITIONS))
    for ax, family in zip(axes, FAMILIES, strict=True):
        pair_ids = sorted(values[family]["unrestricted"])
        for pair_id in pair_ids:
            row = [values[family][condition][pair_id] for condition in CONDITIONS]
            ax.plot(positions, row, color="#94a3b8", alpha=.55, linewidth=.8)
            ax.scatter(positions, row, c=COLORS, s=22, zorder=3)
        medians = [statistics.median(values[family][condition].values()) for condition in CONDITIONS]
        ax.plot(positions, medians, color="#172033", linewidth=2.5, marker="o", label="Median")
        ax.set_yscale("log")
        ax.set_xticks(positions, [CONDITION_LABELS[condition] for condition in CONDITIONS], rotation=28, ha="right")
        ax.set_title(FAMILY_LABELS[family])
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Targeted activations at strict top-1 flip")
    axes[0].legend(frameon=False)
    fig.suptitle("Most pairs retain sparse earlier-position control, but restrictions expose hard outliers")
    fig.tight_layout()
    fig.savefig(ROOT / "restricted_counterfactual_pairs.png", dpi=180)
    plt.close(fig)


def plot_budgets(values):
    budgets = (10, 100, 1000)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    x = np.arange(len(CONDITIONS))
    width = .25
    for ax, family in zip(axes, FAMILIES, strict=True):
        for budget_index, budget in enumerate(budgets):
            successes = [
                sum(value <= budget for value in values[family][condition].values())
                for condition in CONDITIONS
            ]
            ax.bar(x + (budget_index - 1) * width, successes, width, label=f"≤{budget:,}")
        ax.set_xticks(x, [CONDITION_LABELS[condition] for condition in CONDITIONS], rotation=28, ha="right")
        ax.set_ylim(0, 12.8)
        ax.set_title(FAMILY_LABELS[family])
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Pairs flipped out of 12")
    axes[0].legend(frameon=False, title="Activation budget")
    fig.suptitle("Sparse reliability under location restrictions")
    fig.tight_layout()
    fig.savefig(ROOT / "restricted_counterfactual_budgets.png", dpi=180)
    plt.close(fig)


def image_uri(path: Path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def strict_sparse_positions(summary):
    metadata = json.loads((BASE_ROOT / "metadata.json").read_text(encoding="utf-8"))
    arrays = np.load(ROOT / "crossover_indices.npz")
    width = 4864
    output = {}
    for family in FAMILIES:
        incidence = Counter()
        activation_counts = Counter()
        kept_pairs = 0
        for pair in metadata["selected_pairs"]:
            if pair["family"] != family:
                continue
            pair_id = pair["pair_id"]
            value = summary["families"][family]["exclude_final_four_layers_and_position"]["pair_values"][pair_id]
            if value > 1000:
                continue
            key = f"{pair_id}__exclude_final_four_layers_and_position__top1"
            positions = [int(index) // width % pair["sequence_length"] for index in arrays[key]]
            incidence.update(set(positions))
            activation_counts.update(positions)
            kept_pairs += 1
        output[family] = {
            "pairs_at_or_below_1000": kept_pairs,
            "pair_incidence_by_position": dict(sorted(incidence.items())),
            "activation_count_by_position": dict(sorted(activation_counts.items())),
        }
    return output


def main():
    base_rows = load_rows(BASE_ROOT / "evaluations.jsonl")
    restricted_rows = load_rows(ROOT / "evaluations.jsonl")
    values = collect(base_rows, restricted_rows)
    plot_medians(values)
    plot_pairs(values)
    plot_budgets(values)

    summary = {"families": {}}
    for family in FAMILIES:
        baseline = values[family]["unrestricted"]
        summary["families"][family] = {}
        for condition in CONDITIONS:
            condition_values = values[family][condition]
            ratios = [condition_values[pair_id] / baseline[pair_id] for pair_id in baseline]
            summary["families"][family][condition] = {
                "successes": len(condition_values),
                "median": statistics.median(condition_values.values()),
                "range": [min(condition_values.values()), max(condition_values.values())],
                "median_ratio_to_unrestricted": statistics.median(ratios),
                "pairs_at_or_below_10": sum(value <= 10 for value in condition_values.values()),
                "pairs_at_or_below_100": sum(value <= 100 for value in condition_values.values()),
                "pairs_at_or_below_1000": sum(value <= 1000 for value in condition_values.values()),
                "pair_values": dict(sorted(condition_values.items())),
            }
    summary["strict_sparse_positions"] = strict_sparse_positions(summary)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Location-restricted MLP counterfactual patching", "",
        "Each condition filters the same signed gradient × donor-change ranking. Excluded activations remain at their base-run values. Counts are the first tested and integer-refined points where the donor answer becomes the full-vocabulary top-1 token.", "",
        "| Task | Unrestricted | No layer 23 | No layers 20–23 | No final position | No late layers or final position |", "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        cells = []
        for condition in CONDITIONS:
            node = summary["families"][family][condition]
            cells.append(f"{node['median']:g} ({node['successes']}/12)")
        lines.append(f"| {FAMILY_LABELS[family]} | " + " | ".join(cells) + " |")
    lines += [
        "", "All 12 pairs eventually flip in every condition, but some restricted crossings require hundreds of thousands or millions of activations. The success count alone therefore does not establish sparse control.", "",
        "## Main findings", "",
        "- Removing only layer 23 has little effect: medians change from 6.5 to 6.5 for IOI, 8.5 to 10.5 for agreement, and 3.5 to 4 for brackets.",
        "- Removing layers 20–23 leaves the IOI median unchanged at 6.5, but raises agreement to 21 and brackets to 6.5. One difficult IOI reverse-direction pair becomes a million-activation outlier.",
        "- Removing the final token position raises IOI from 6.5 to 54.5 activations, agreement from 8.5 to 19, and brackets from 3.5 to 6.5.",
        "- With both late layers and the final position forbidden, the medians are 59 for IOI, 19 for agreement, and 6.5 for brackets. Under a 100-activation budget, 8/12 IOI, 9/12 agreement, and 11/12 bracket pairs still flip.",
        "- Under the strictest restriction, every IOI pair that stays below 1,000 activations uses the repeated-subject position; every sparse agreement pair uses the grammatical-subject position; and every sparse bracket pair uses the unmatched-opener position.", "",
        "## Interpretation", "",
        "The original result is not a final-layer artifact. It is partly a final-position steering result: that shortcut is especially important for IOI. Even after removing it, most pairs retain sparse control through earlier token positions, which supports a nontrivial source-position mechanism. The extreme outliers show that this claim is not uniform across prompts or counterfactual directions.",
    ]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for family in FAMILIES:
        cells = "".join(
            f"<td>{summary['families'][family][condition]['median']:g} ({summary['families'][family][condition]['successes']}/12)</td>"
            for condition in CONDITIONS
        )
        rows.append(f"<tr><td>{FAMILY_LABELS[family]}</td>{cells}</tr>")
    figures = "".join(
        f'<figure><img src="{image_uri(ROOT / filename)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
        for filename, caption in (
            ("restricted_counterfactual_medians.png", "Median strict crossover under each restriction"),
            ("restricted_counterfactual_pairs.png", "Per-pair crossovers reveal hard outliers"),
            ("restricted_counterfactual_budgets.png", "Reliability at fixed sparse budgets"),
        )
    )
    headers = "".join(f"<th>{CONDITION_LABELS[c].replace(chr(10), ' ')}</th>" for c in CONDITIONS)
    html = f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Restricted counterfactual patching</title><style>body{{font:16px/1.5 system-ui;margin:0;background:#f8fafc;color:#172033}}main{{max-width:1000px;margin:auto;padding:22px}}.note{{background:#eff6ff;border-left:4px solid #2563eb;padding:12px 16px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:780px;background:white}}th,td{{padding:8px;border-bottom:1px solid #dbe3ee;text-align:right}}th:first-child,td:first-child{{text-align:left}}figure{{margin:28px 0;background:white;border:1px solid #dbe3ee;padding:10px}}img{{width:100%;height:auto}}figcaption{{text-align:center;color:#475569}}@media(max-width:600px){{main{{padding:14px}}table{{font-size:12px}}}}</style><main><h1>Are sparse counterfactual neurons just late-stage shortcuts?</h1><p>The original targeted ranking was rerun after excluding late layers, the final token position, or both.</p><div class="note"><strong>Answer:</strong> not a final-layer artifact, but partly a final-position steering effect. Most prompts still have sparse earlier-position controls; a few become extremely difficult.</div><div class="scroll"><table><tr><th>Task</th>{headers}</tr>{''.join(rows)}</table></div>{figures}<h2>Interpretation</h2><p>Layer 23 by itself is not carrying the result. Removing the last four layers modestly affects typical IOI behavior and more strongly affects agreement and brackets. Removing the final token position is the sharpest test: typical IOI crossover grows roughly ninefold, yet 8/12 IOI pairs still flip with at most 100 earlier-position activations.</p><p>The million-activation outliers matter. They show that sparse source-position control is common but not universal. These remain answer-aware control interventions rather than certified minimal semantic circuits.</p></main>'''
    (ROOT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
