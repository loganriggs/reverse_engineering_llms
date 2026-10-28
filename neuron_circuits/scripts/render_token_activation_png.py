#!/usr/bin/env python3
"""Render four natural token-activation examples to a static PNG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle


SOURCE = Path("artifacts/token_activation_samples/data.json")
OUTPUT = Path("artifacts/token_activation_samples/token_activation_examples.png")
COLS = 10


def visible(text: str) -> str:
    if not text.strip():
        return "[newline]" if "\n" in text else "[space]"
    return text.replace("\n", " ↵")


def choose_samples(data):
    chosen = []
    for family, label in (("arithmetic", "Arithmetic"), ("brackets", "Brackets")):
        neuron = data["families"][family][0]
        positive = next(row for row in neuron["samples"] if row["polarity"] == "positive")
        negative = next(row for row in neuron["samples"] if row["polarity"] == "negative")
        chosen.extend(((f"{label} — maximum positive example", neuron, positive), (f"{label} — maximum negative example", neuron, negative)))
    return chosen


def draw_sample(ax, title, neuron, sample):
    tokens = sample["tokens"]
    rows = (len(tokens) + COLS - 1) // COLS
    scale = max(abs(token["activation"]) for token in tokens)
    norm = colors.TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale)
    cmap = plt.get_cmap("RdBu")
    ax.set_xlim(0, COLS)
    ax.set_ylim(rows, 0)
    ax.axis("off")
    ax.set_title(
        f"{title}\n{neuron['unit']} at token position {neuron['position']}  •  expected answer: {sample['answer']}",
        loc="left", fontsize=13, fontweight="bold", pad=10,
    )
    first_generated = next((index for index, token in enumerate(tokens) if token["generated"]), None)
    for index, token in enumerate(tokens):
        row, col = divmod(index, COLS)
        value = token["activation"]
        face = cmap(norm(value))
        focus = token["index"] == sample["circuit_position"]
        patch = Rectangle(
            (col + 0.04, row + 0.05), 0.92, 0.86,
            facecolor=face, edgecolor="black" if focus else "#777777",
            linewidth=3 if focus else 0.7,
        )
        ax.add_patch(patch)
        luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
        text_color = "black" if luminance > 0.58 else "white"
        ax.text(col + 0.5, row + 0.38, visible(token["text"]), ha="center", va="center", fontsize=10, color=text_color, fontweight="bold")
        ax.text(col + 0.5, row + 0.70, f"{value:+.2f}", ha="center", va="center", fontsize=8.5, color=text_color)
        if first_generated == index:
            ax.text(col + 0.04, row + 0.02, "continuation ↓", ha="left", va="bottom", fontsize=7.5, color="#333333")


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    selected = choose_samples(data)
    fig, axes = plt.subplots(4, 1, figsize=(15, 15), constrained_layout=True)
    fig.suptitle(
        "Raw post-gating MLP activation on intact text\nBlue = positive, red = negative; black outline = sparse-circuit token",
        fontsize=17, fontweight="bold",
    )
    for ax, selection in zip(axes, selected):
        draw_sample(ax, *selection)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
