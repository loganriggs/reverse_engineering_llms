#!/usr/bin/env python3
"""Build one combined MLP-and-attention counterfactual report."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MLP_ROOT = Path("artifacts/counterfactual_patching")
ATTENTION_ROOT = Path("artifacts/attention_counterfactuals")
OUTPUT = Path("artifacts/combined_counterfactuals")
FAMILIES = ("ioi", "agreement", "brackets")
FAMILY_LABELS = {"ioi": "IOI", "agreement": "Agreement", "brackets": "Brackets"}
UNITS = ("mlp", "query", "key", "value", "pattern", "message", "head_output")
UNIT_LABELS = {
    "mlp": "MLP scalar",
    "query": "Query",
    "key": "Key",
    "value": "Value",
    "pattern": "Attention row",
    "message": "Message",
    "head_output": "Head output",
}
COLORS = {
    "mlp": "#64748b",
    "routing": "#8b5cf6",
    "content": "#2563eb",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def value(summary, mlp_summary, family, unit):
    if unit == "mlp":
        node = mlp_summary["families"][family]["strict_top1_crossover"]["instance"]["counterfactual_gradient"]
        return node["median"], node["n"]
    node = summary["families"][family][unit]
    return node["median"], node["successes"]


def plot_overview(attention, mlp):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharey=True)
    colors = [COLORS["mlp"], COLORS["routing"], COLORS["routing"], COLORS["content"], COLORS["routing"], COLORS["content"], COLORS["content"]]
    for ax, family in zip(axes, FAMILIES, strict=True):
        medians, successes = zip(*(value(attention, mlp, family, unit) for unit in UNITS), strict=True)
        y = np.arange(len(UNITS))
        plot_values = [median if median is not None else 0.72 for median in medians]
        bars = ax.barh(y, plot_values, color=colors)
        for index, (bar, median, success) in enumerate(zip(bars, medians, successes, strict=True)):
            if median is None:
                ax.text(0.82, index, f"no flip · {success}/12", va="center", fontsize=9)
            else:
                ax.text(median * 1.12, index, f"{median:g} · {success}/12", va="center", fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(.6, 420)
        ax.set_yticks(y, [UNIT_LABELS[unit] for unit in UNITS])
        ax.invert_yaxis()
        ax.set_title(FAMILY_LABELS[family])
        ax.set_xlabel("Median targeted units at strict flip")
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Counterfactual control separates routing from transported content")
    fig.tight_layout()
    fig.savefig(OUTPUT / "combined_counterfactual_overview.png", dpi=180)
    plt.close(fig)


def plot_success_matrix(attention, mlp):
    values = []
    annotations = []
    for family in FAMILIES:
        row_values, row_annotations = [], []
        for unit in UNITS:
            median, successes = value(attention, mlp, family, unit)
            row_values.append(successes)
            row_annotations.append(f"{successes}/12\n" + (f"med {median:g}" if median is not None else "no flip"))
        values.append(row_values)
        annotations.append(row_annotations)
    matrix = np.array(values)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(matrix, vmin=0, vmax=12, cmap="Blues", aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] >= 8 else "#172033"
            ax.text(column, row, annotations[row][column], ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(np.arange(len(UNITS)), [UNIT_LABELS[unit] for unit in UNITS], rotation=22, ha="right")
    ax.set_yticks(np.arange(len(FAMILIES)), [FAMILY_LABELS[family] for family in FAMILIES])
    ax.set_title("Reliability and crossover size across all 36 counterfactual pairs")
    fig.colorbar(im, ax=ax, label="Pairs flipped out of 12", fraction=.03, pad=.03)
    fig.tight_layout()
    fig.savefig(OUTPUT / "combined_success_matrix.png", dpi=180)
    plt.close(fig)


def plot_controls(attention, mlp):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    units = ("mlp", "value", "message", "head_output")
    for ax, family in zip(axes, FAMILIES, strict=True):
        targeted, controls = [], []
        for unit in units:
            if unit == "mlp":
                node = mlp["families"][family]["strict_top1_crossover"]["instance"]
                targeted.append(node["counterfactual_gradient"]["median"])
                controls.append(node["random"]["median"])
            else:
                node = attention["families"][family][unit]
                targeted.append(node["median"])
                controls.append(node["random_median"])
        x = np.arange(len(units))
        ax.bar(x - .18, targeted, .36, color="#2563eb", label="Targeted")
        ax.bar(x + .18, controls, .36, color="#94a3b8", label="Random")
        ax.set_yscale("log")
        ax.set_xticks(x, [UNIT_LABELS[unit] for unit in units], rotation=18, ha="right")
        ax.set_title(FAMILY_LABELS[family])
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Median units at strict flip")
    axes[0].legend(frameon=False)
    fig.suptitle("Targeted counterfactual control is orders of magnitude sparser than random patching")
    fig.tight_layout()
    fig.savefig(OUTPUT / "combined_random_controls.png", dpi=180)
    plt.close(fig)


def image_uri(path: Path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    attention = load(ATTENTION_ROOT / "summary.json")
    mlp = load(MLP_ROOT / "summary.json")
    plot_overview(attention, mlp)
    plot_success_matrix(attention, mlp)
    plot_controls(attention, mlp)

    combined = {
        "model": attention["model"],
        "pair_count": attention["architecture"]["pair_count"],
        "pairs_per_family": mlp["pairs_per_family"],
        "unit_note": "MLP entries are scalar activation instances; attention entries are 64-dimensional vectors or messages, so counts are not dimension-equivalent.",
        "families": {},
        "examples": attention["examples"],
    }
    for family in FAMILIES:
        combined["families"][family] = {
            unit: {"median": value(attention, mlp, family, unit)[0], "successes": value(attention, mlp, family, unit)[1]}
            for unit in UNITS
        }
        combined["families"][family]["top_recurring_mlp_neurons"] = mlp["recurrence"][family]["neurons"][:3]
    (OUTPUT / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")

    lines = [
        "# Combined counterfactual circuits", "",
        "The same 36 counterfactual pairs were tested with scalar MLP activation patches and six attention intervention types. Strict crossover means that the donor answer became the full-vocabulary top-1 token.", "",
        "| Task | MLP scalar | Query | Key | Value | Attention row | Message | Head output |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        cells = []
        for unit in UNITS:
            median, successes = value(attention, mlp, family, unit)
            cells.append(f"{median:g} ({successes}/12)" if median is not None else f"— ({successes}/12)")
        lines.append(f"| {FAMILY_LABELS[family]} | " + " | ".join(cells) + " |")
    lines += [
        "", "Counts are targeted medians with successful pairs in parentheses. MLP units are scalar activation instances. Attention query, key, value, message, and head-output units are 64-dimensional vectors; an attention row is a routing distribution. Counts are therefore causal intervention sizes, not equal-dimensional circuit sizes.", "",
        "## Interpretation", "",
        "- **IOI:** both routing and transported content can redirect the answer. The John/Mary example flips with five query rows, five attention rows, two messages, two head outputs, or six MLP scalar activations.",
        "- **Agreement:** the result is carried by content. Query, key, attention-row, and joint Q+K patches never flip a pair, while one value, one message, or one head output flips every pair.",
        "- **Brackets:** content patches are reliable; routing-only patches are rare and usually large. One head output, two values, or 2.5 messages is the typical targeted crossover.", "",
        "## Representative routes", "",
        "- **IOI:** MLP L15/N796 appears in 12/12 pairs. In John/Mary, attention messages L20/H9 repeated `John` → final `to` and L23/H9 initial `Mary` → final `to` jointly flip the answer.",
        "- **Agreement:** MLP L20/N4662 appears in 12/12 pairs. In book/pictures, L19/H4 sends the subject `book` to the final prediction position.",
        "- **Brackets:** MLP L22/N1943 appears in 9/12 pairs. In the shown example, L20/H4 and L20/H5 send the unmatched opener to the final arrow.",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = []
    for family in FAMILIES:
        cells = []
        for unit in UNITS:
            median, successes = value(attention, mlp, family, unit)
            cells.append(f"<td>{median:g} ({successes}/12)</td>" if median is not None else f"<td>— ({successes}/12)</td>")
        rows.append(f"<tr><td>{FAMILY_LABELS[family]}</td>{''.join(cells)}</tr>")
    figures = "".join(
        f'<figure><img src="{image_uri(OUTPUT / filename)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
        for filename, caption in (
            ("combined_counterfactual_overview.png", "Targeted crossover size and success rate"),
            ("combined_success_matrix.png", "Reliability across intervention types"),
            ("combined_random_controls.png", "Targeted versus random intervention sizes"),
        )
    )
    html = f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Combined counterfactual circuits</title>
<style>body{{font:16px/1.5 system-ui;margin:0;background:#f8fafc;color:#172033}}main{{max-width:1000px;margin:auto;padding:22px}}.lead{{font-size:1.1rem}}.note{{background:#eff6ff;border-left:4px solid #2563eb;padding:12px 16px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:820px;background:white}}th,td{{padding:8px;border-bottom:1px solid #dbe3ee;text-align:right}}th:first-child,td:first-child{{text-align:left}}figure{{margin:28px 0;background:white;border:1px solid #dbe3ee;padding:10px}}img{{width:100%;height:auto}}figcaption{{text-align:center;color:#475569}}@media(max-width:600px){{main{{padding:14px}}table{{font-size:12px}}th,td{{padding:5px}}}}</style>
<main><h1>Counterfactual control across MLPs and attention</h1><p class="lead">The same 36 minimal prompt changes were used to patch scalar MLP activations and separate attention routing and content signals.</p><div class="note"><strong>Result:</strong> IOI can be redirected through routing or content. Agreement and bracket closing are dominated by values, messages, and head outputs.</div><h2>Strict top-1 crossover</h2><div class="scroll"><table><tr><th>Task</th>{''.join(f'<th>{UNIT_LABELS[u]}</th>' for u in UNITS)}</tr>{''.join(rows)}</table></div>{figures}<h2>How to read the counts</h2><p>Each cell reports the median targeted units followed by the number of counterfactual pairs that flipped. An MLP unit is one scalar activation instance. Most attention units are 64-dimensional vectors, so the numbers compare causal handles, not equal-dimensional circuit sizes.</p><h2>Representative mechanisms</h2><p><strong>IOI:</strong> recurrent MLP L15/N796 plus two concrete John/Mary message routes. <strong>Agreement:</strong> recurrent MLP L20/N4662 plus a single L19/H4 subject-to-answer message. <strong>Brackets:</strong> recurrent MLP L22/N1943 plus L20/H4 and H5 opener-to-answer messages.</p></main>'''
    (OUTPUT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
