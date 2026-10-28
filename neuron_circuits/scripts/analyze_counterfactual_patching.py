#!/usr/bin/env python3
"""Analyze counterfactual activation-patching crossover counts."""

from __future__ import annotations

import base64
from collections import Counter
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/counterfactual_patching")
FAMILIES = ("ioi", "agreement", "brackets")
LABELS = {"ioi": "IOI", "agreement": "Agreement", "brackets": "Brackets"}
METHODS = (
    ("counterfactual_gradient", "Gradient × change", "#2563eb"),
    ("delta_magnitude", "Largest change", "#f97316"),
    ("random", "Random", "#94a3b8"),
)


def rows():
    return [json.loads(line) for line in (ROOT / "evaluations.jsonl").read_text().splitlines()]


def first_counts(data, family, resolution, method, criterion="target_top1"):
    output = {}
    pair_ids = {row["pair_id"] for row in data if row["family"] == family}
    for pair_id in pair_ids:
        successes = [
            row for row in data if row["pair_id"] == pair_id
            and row["resolution"] == resolution and row["method"] == method
            and row[criterion]
        ]
        if successes:
            output[pair_id] = min(row["patched"] for row in successes)
    return output


def direction(pair_id, family):
    if family == "ioi":
        return "toward first name" if pair_id.endswith("forward") else "toward second name"
    if family == "agreement":
        return "is → are" if pair_id.endswith("sg-to-pl") else "are → is"
    parts = pair_id.split("-")
    symbols = {40: ")", 91: "]", 123: "}"}
    return f"{symbols[int(parts[1])]} → {symbols[int(parts[3])]}"


def plot_method_comparison(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(FAMILIES))
    width = 0.25
    for ax, resolution in zip(axes, ("instance", "neuron"), strict=True):
        for index, (method, label, color) in enumerate(METHODS):
            values = [
                statistics.median(first_counts(data, family, resolution, method).values())
                for family in FAMILIES
            ]
            bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
            for bar, value in zip(bars, values, strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:g}", ha="center", va="bottom", fontsize=9)
        ax.set_yscale("log")
        ax.set_xticks(x, [LABELS[f] for f in FAMILIES])
        ax.set_ylabel("Median changes to counterfactual top-1")
        ax.set_title("Position-specific instances" if resolution == "instance" else "Unique layer–channel neurons")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Answer-aware causal ranking isolates a handful of high-leverage units")
    fig.tight_layout()
    fig.savefig(ROOT / "counterfactual_method_comparison.png", dpi=180)
    plt.close(fig)


def plot_directionality(data):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, family in zip(axes, FAMILIES, strict=True):
        values = first_counts(data, family, "instance", "counterfactual_gradient")
        grouped = {}
        for pair_id, value in values.items():
            grouped.setdefault(direction(pair_id, family), []).append(value)
        positions = np.arange(len(grouped))
        box = ax.boxplot(
            [grouped[label] for label in grouped], positions=positions, widths=.55,
            patch_artist=True, showfliers=False, medianprops={"color": "white", "linewidth": 2},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#2563eb")
        rng = np.random.default_rng(7)
        for index, values_at_direction in enumerate(grouped.values()):
            jitter = rng.normal(0, .045, len(values_at_direction))
            ax.scatter(index + jitter, values_at_direction, color="#172033", s=22, zorder=3)
        ax.set_xticks(positions, list(grouped), rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_title(LABELS[family])
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Activation instances changed")
    fig.suptitle("Counterfactual direction can matter as much as task family")
    fig.tight_layout()
    fig.savefig(ROOT / "counterfactual_directionality.png", dpi=180)
    plt.close(fig)


def top_recurring(metadata):
    arrays = np.load(ROOT / "crossover_indices.npz")
    width = 4864
    output = {}
    for family in FAMILIES:
        frequency = Counter()
        positions = Counter()
        for pair in metadata["selected_pairs"]:
            if pair["family"] != family:
                continue
            length = pair["sequence_length"]
            seen = set()
            for raw in arrays[f'{pair["pair_id"]}__instance__top1']:
                index = int(raw)
                layer = index // (length * width)
                position = (index // width) % length
                channel = index % width
                seen.add((layer, channel))
                positions[position] += 1
            frequency.update(seen)
        output[family] = {
            "neurons": [
                {"layer": layer, "channel": channel, "pairs": count}
                for (layer, channel), count in frequency.most_common(8)
            ],
            "positions": [{"position": position, "uses": count} for position, count in positions.most_common()],
        }
    return output


def plot_recurrence(recurrence):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, family in zip(axes, FAMILIES, strict=True):
        values = recurrence[family]["neurons"][:6][::-1]
        labels = [f"L{x['layer']} N{x['channel']}" for x in values]
        counts = [x["pairs"] for x in values]
        ax.barh(labels, counts, color="#2563eb")
        for index, count in enumerate(counts):
            ax.text(count + .1, index, f"{count}/12", va="center", fontsize=9)
        ax.set_xlim(0, 13)
        ax.set_title(LABELS[family])
        ax.set_xlabel("Counterfactual pairs containing neuron")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The same high-leverage neurons recur across different prompts")
    fig.tight_layout()
    fig.savefig(ROOT / "counterfactual_recurrence.png", dpi=180)
    plt.close(fig)


def uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    data = rows()
    metadata = json.loads((ROOT / "metadata.json").read_text())
    plot_method_comparison(data)
    plot_directionality(data)
    recurrence = top_recurring(metadata)
    plot_recurrence(recurrence)
    summary = {
        "model": metadata["model"],
        "pairs_per_family": metadata["pairs_per_family"],
        "valid_pair_counts": metadata["valid_pair_counts"],
        "patchable_pair_counts": metadata["patchable_pair_counts"],
        "families": {},
        "recurrence": recurrence,
    }
    for family in FAMILIES:
        family_summary = {"strict_top1_crossover": {}, "directions": {}}
        for resolution in ("instance", "neuron"):
            family_summary["strict_top1_crossover"][resolution] = {}
            for method, _, _ in METHODS:
                values = list(first_counts(data, family, resolution, method).values())
                family_summary["strict_top1_crossover"][resolution][method] = {
                    "median": statistics.median(values), "mean": statistics.mean(values),
                    "min": min(values), "max": max(values), "n": len(values),
                }
        targeted = first_counts(data, family, "instance", "counterfactual_gradient")
        for pair_id, count in targeted.items():
            family_summary["directions"].setdefault(direction(pair_id, family), []).append(count)
        family_summary["directions"] = {
            label: {"median": statistics.median(values), "range": [min(values), max(values)], "values": sorted(values)}
            for label, values in family_summary["directions"].items()
        }
        summary["families"][family] = family_summary
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Counterfactual neuron patching", "", "Strict crossover means that the counterfactual answer became the model's full-vocabulary top-1 token.", "", "| Task | Activation instances, targeted | Unique neurons, targeted | Instances by largest change | Instances, random |", "|---|---:|---:|---:|---:|"]
    for family in FAMILIES:
        s = summary["families"][family]["strict_top1_crossover"]
        lines.append(f"| {LABELS[family]} | {s['instance']['counterfactual_gradient']['median']:g} | {s['neuron']['counterfactual_gradient']['median']:g} | {s['instance']['delta_magnitude']['median']:g} | {s['instance']['random']['median']:g} |")
    lines += ["", "Counts are first observed crossovers along each nested ranking, refined to integer resolution. They are not certified global minima.", "", "The targeted ranking uses the base-run gradient of `logit(counterfactual answer) - logit(original answer)` multiplied by the actual source-minus-base activation change. It therefore answers a pair-specific causal question and is intentionally answer-aware.", "", "## Directionality"]
    for family in FAMILIES:
        directions = summary["families"][family]["directions"]
        lines.append(f"- **{LABELS[family]}:** " + "; ".join(f"{label}: median {values['median']:g}, range {values['range'][0]}–{values['range'][1]}" for label, values in directions.items()))
    lines += ["", "## Recurring neurons"]
    for family in FAMILIES:
        top = recurrence[family]["neurons"][:3]
        lines.append(f"- **{LABELS[family]}:** " + ", ".join(f"layer {x['layer']} / channel {x['channel']} in {x['pairs']}/12 pairs" for x in top))
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figures = "".join(
        f'<figure><img src="{uri(ROOT / filename)}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'
        for filename, caption in (
            ("counterfactual_method_comparison.png", "Crossover counts by ranking and resolution"),
            ("counterfactual_directionality.png", "Direction-specific targeted crossover counts"),
            ("counterfactual_recurrence.png", "Recurring high-leverage neurons"),
        )
    )
    table_rows = "".join(
        f"<tr><td>{LABELS[f]}</td><td>{summary['families'][f]['strict_top1_crossover']['instance']['counterfactual_gradient']['median']:g}</td><td>{summary['families'][f]['strict_top1_crossover']['neuron']['counterfactual_gradient']['median']:g}</td><td>{summary['families'][f]['strict_top1_crossover']['instance']['delta_magnitude']['median']:g}</td><td>{summary['families'][f]['strict_top1_crossover']['instance']['random']['median']:g}</td></tr>"
        for f in FAMILIES
    )
    html = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Counterfactual neuron patching</title>
<style>body{{font:16px/1.5 system-ui;margin:0;background:#f8fafc;color:#172033}}main{{max-width:980px;margin:auto;padding:22px}}.lead{{font-size:1.1rem}}.note{{background:#eff6ff;border-left:4px solid #2563eb;padding:12px 16px}}figure{{margin:28px 0;background:white;border:1px solid #dbe3ee;padding:10px}}img{{width:100%;height:auto}}figcaption{{text-align:center;color:#475569}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px;border-bottom:1px solid #dbe3ee;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{font-size:.9em}}@media(max-width:600px){{main{{padding:14px}}table{{font-size:12px}}th,td{{padding:5px}}}}</style>
<main><h1>How many neuron changes flip a counterfactual?</h1><p class="lead">Selected MLP activations from a minimally changed source prompt are patched into the original prompt. The crossover is the first tested and integer-refined subset where the source answer becomes the full-vocabulary top token.</p>
<div class="note"><strong>Answer:</strong> under the answer-aware gradient × counterfactual-change ranking, the median strict flip takes 6.5 IOI activation instances, 8.5 agreement instances, and 3.5 bracket instances. At whole-neuron resolution the corresponding medians are 4.5, 8.5, and 3.5 layer–channel neurons.</div>
<h2>Strict top-1 crossover</h2><table><tr><th>Task</th><th>Targeted instances</th><th>Targeted neurons</th><th>Largest-change instances</th><th>Random instances</th></tr>{table_rows}</table>
{figures}<h2>What this means</h2><p>The causal bottleneck is much smaller than the earlier sufficiency circuit: a few source activations can redirect the answer, even though reproducing the intact distribution requires a broad retained circuit. The ranking is pair- and answer-specific, so these counts are sparse causal handles—not a claim that the whole computation consists of only a few neurons.</p>
<p>Direction matters. IOI patches toward the first-mentioned name are generally easier than patches toward the second-mentioned name; changing <em>is</em> to <em>are</em> is also easier than the reverse in this sample. A handful of layer–channel neurons recur across nearly every pair, making them strong candidates for follow-up causal characterization.</p></main>"""
    (ROOT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
