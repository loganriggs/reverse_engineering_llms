#!/usr/bin/env python3
"""Post-hoc axis stability analysis for the layer-head scaffold."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_competent_feature_head_scaffold.json")
CHECKPOINT = TENSOR_ROOT / "bilin18_competent_feature_head_scaffold.partial.json"
OUTPUT = TENSOR_ROOT / "bilin18_competent_feature_head_axes.json"
REPORT = TENSOR_ROOT / "BILIN18_COMPETENT_FEATURE_HEAD_AXES.md"
FIGURE = TENSOR_ROOT / "bilin18_competent_feature_head_axes.png"


def pairwise_correlations(rows):
    values = [float(np.corrcoef(rows[first], rows[second])[0, 1]) for first in range(len(rows)) for second in range(first + 1, len(rows))]
    return {"median": float(np.median(values)), "minimum": float(np.min(values)), "maximum": float(np.max(values)), "values": values}


def main():
    config = json.loads(CONFIG.read_text()); checkpoint = json.loads(CHECKPOINT.read_text()); panels = list(config["discovery_panels"]); summaries = {}
    for source in config["source_layers"]:
        offsets = []; heads = []
        for panel in panels:
            scores = checkpoint["discovery_scores"][f"{source}_{panel}"]; offset = np.zeros(config["attention_width"]); head = np.zeros(9)
            for key, value in scores.items():
                block, head_index = map(int, key.split("_")); offset[block - source - 1] += value; head[head_index] += value
            offsets.append(offset / 9); heads.append(head / config["attention_width"])
        offset_mean = np.mean(offsets, axis=0); head_mean = np.mean(heads, axis=0)
        summaries[str(source)] = {"offset_correlation": pairwise_correlations(offsets), "head_correlation": pairwise_correlations(heads), "offset_mean_scores": offset_mean.tolist(), "head_mean_scores": head_mean.tolist(), "offset_order": (np.argsort(offset_mean)[::-1] + 1).tolist(), "head_order": np.argsort(head_mean)[::-1].tolist(), "offset_one_two_both_top_two_fraction": float(np.mean([set((np.argsort(row)[::-1][:2] + 1).tolist()) == {1, 2} for row in offsets]))}
    OUTPUT.write_text(json.dumps({"status": "post-hoc-after-four-panel-head-scaffold", "summaries": summaries}, indent=2, allow_nan=False))
    lines = ["# Relative-depth versus head-identity stability", "", "This post-hoc analysis marginalizes the four independent discovery-panel edge scores over either heads or relative block offsets. It asks which axis supplies the stable part of the weak native head scaffold.", "", "| Source | Offset correlation median/min | Head correlation median/min | Offset order | Head order | Panels with offsets {1,2} top-two |", "|---:|---:|---:|---|---|---:|"]
    for source in config["source_layers"]:
        row = summaries[str(source)]; lines.append(f"| {source} | {row['offset_correlation']['median']:.3f}/{row['offset_correlation']['minimum']:.3f} | {row['head_correlation']['median']:.3f}/{row['head_correlation']['minimum']:.3f} | {', '.join(map(str, row['offset_order']))} | {', '.join(map(str, row['head_order']))} | {row['offset_one_two_both_top_two_fraction']:.2f} |")
    lines += ["", "## Interpretation", "", "Relative depth is substantially more reproducible than head identity at source Layers 0--2; the first two downstream blocks dominate the averaged ordering. At source Layer 3, head-axis stability rises, but the frozen edge advantage over random half-head programs remains small. This supports a coarse near-source routing grammar rather than a universal list of semantic heads.", "", "This marginal analysis does not establish additive credit: leave-one-out scores can move under redundancy, and averaging hides feature-specific heads. Its role is to identify the shortest stable coordinate system for subsequent causal tests."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for source in config["source_layers"]: axes[0].plot(range(1, config["attention_width"] + 1), summaries[str(source)]["offset_mean_scores"], marker="o", label=f"source {source}")
    axes[0].set(xlabel="Relative downstream block", ylabel="Mean leave-one-out score", title="Stable relative-depth profile"); axes[0].legend(); axes[0].grid(alpha=.2)
    for source in config["source_layers"]: axes[1].plot(range(9), summaries[str(source)]["head_mean_scores"], marker="o", label=f"source {source}")
    axes[1].set(xlabel="Native head index", ylabel="Mean leave-one-out score", title="Head identity is less stable early"); axes[1].legend(); axes[1].grid(alpha=.2); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(summaries, indent=2)); print(REPORT)


if __name__ == "__main__": main()
