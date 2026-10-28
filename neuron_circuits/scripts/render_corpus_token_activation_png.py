#!/usr/bin/env python3
"""Render real WikiText contexts colored by sparse-neuron activation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle


SOURCE = Path("artifacts/corpus_neuron_examples/examples.json")
OUTPUT = Path("artifacts/corpus_neuron_examples/natural_text_activations.png")
COLS = 10


def visible(text: str) -> str:
    if not text.strip():
        return "[space]"
    return text.replace("\n", " ↵").replace("@-@", "-").replace("@,@", ",").replace("@.@", ".")


def choose(data):
    lookup = {unit["unit"]: unit for unit in data["units"]}
    output = []
    for unit_name, label in (("L15/P6/N1606", "Arithmetic-circuit neuron"), ("L4/P16/N3818", "Bracket-circuit neuron")):
        unit = lookup[unit_name]
        output.extend(((f"{label} — maximum positive corpus example", unit, unit["top_positive"][0]), (f"{label} — maximum negative corpus example", unit, unit["top_negative"][0])))
    return output


def draw(ax, title, unit, sample):
    values = sample["activations"]
    rows = (len(values) + COLS - 1) // COLS
    scale = max(abs(value) for value in values)
    norm = colors.TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    cmap = plt.get_cmap("RdBu")
    ax.set_xlim(0, COLS)
    ax.set_ylim(rows, 0)
    ax.axis("off")
    ax.set_title(
        f"{title}\n{unit['unit']}  •  highlighted token: {sample['target_token']!r}  •  activation {sample['activation']:+.3f}",
        loc="left", fontsize=12.5, fontweight="bold", pad=9,
    )
    for index, (text, value) in enumerate(zip(sample["tokens"], values)):
        row, col = divmod(index, COLS)
        face = cmap(norm(value))
        focus = index == sample["target_offset"]
        ax.add_patch(Rectangle(
            (col + 0.04, row + 0.05), 0.92, 0.86,
            facecolor=face, edgecolor="black" if focus else "#777777",
            linewidth=3 if focus else 0.7,
        ))
        luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
        color = "black" if luminance > 0.58 else "white"
        ax.text(col + 0.5, row + 0.37, visible(text), ha="center", va="center", fontsize=9.2, color=color, fontweight="bold")
        ax.text(col + 0.5, row + 0.70, f"{value:+.2f}", ha="center", va="center", fontsize=8.2, color=color)


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), constrained_layout=True)
    fig.suptitle(
        "Natural WikiText-103 examples: raw MLP activation by token\nBlue = positive, red = negative; black outline = extremal token; 14 tokens of context on each side",
        fontsize=16, fontweight="bold",
    )
    for axis, selection in zip(axes, choose(data)):
        draw(axis, *selection)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
