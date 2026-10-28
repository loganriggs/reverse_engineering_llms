#!/usr/bin/env python3
"""Plot the quotient, identifiability, and endpoint results in one figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/bilin18_tensor_decomposition")
OUTPUT = ROOT / "bilin18_layer17_tangent_bundle_summary.png"


def load(name):
    return json.loads((ROOT / name).read_text())


def main():
    symmetric = load("bilin18_layer17_backward_mlp_symmetric_tensor.json")
    quotient = load("bilin18_layer17_tangent_bundle_quotient.json")
    split = load("bilin18_layer17_tangent_bundle_split_redteam.json")
    core = load("bilin18_layer17_tangent_bundle_core_stability.json")
    behavior = load("bilin18_layer17_tangent_bundle_behavior_core.json")
    corpora = ("fineweb", "wikitext", "tinystories")
    components = ("prediction", "target", "margin")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    colors = {"symmetric": "#a8a8a8", "role": "#2878b5", "tucker": "#54a24b"}

    labels, old, role, tucker = [], [], [], []
    for corpus in corpora:
        for component in components:
            labels.append(f"{corpus[:2].upper()}\n{component[:4]}")
            old.append(symmetric["results"][corpus]["components"][component]["projected_direct_fidelity"])
            role.append(quotient["results"][corpus]["conditions"]["role_separated"]["64"][component])
            tucker.append(core["split_results"]["head"]["evaluation"][corpus]["16"][component])
    x = np.arange(len(labels)); width = .25
    axes[0].bar(x - width, old, width, label="shared symmetric PCA-64", color=colors["symmetric"])
    axes[0].bar(x, role, width, label="role-separated 64³", color=colors["role"])
    axes[0].bar(x + width, tucker, width, label="Tucker-16", color=colors["tucker"])
    axes[0].axhline(0, color="#333", linewidth=.8)
    axes[0].axhline(.9, color="#777", linewidth=.8, linestyle=":")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylim(-.9, 1.05)
    axes[0].set_ylabel("native-path fidelity")
    axes[0].set_title("A. Tensor roles rescue score paths")
    axes[0].legend(fontsize=8, loc="lower right")

    split_names = ("tail", "interleaved", "random")
    roles = ("output", "tangent", "context")
    x = np.arange(len(roles)); width = .22
    for index, name in enumerate(split_names):
        values = [core["physical_core_overlap"][name][role]["16"] for role in roles]
        axes[1].bar(x + (index - 1) * width, values, width, label=name)
    axes[1].axhline(.7, color="#777", linewidth=.8, linestyle=":")
    axes[1].set_xticks(x, roles)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("projector overlap with head split")
    axes[1].set_title("B. Output core is more identifiable")
    axes[1].legend(fontsize=8)
    axes[1].text(
        .02, .03,
        "External role-64 ranges:\nscore ≤ .015; margin ≤ .060",
        transform=axes[1].transAxes, fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#bbb", "alpha": .9})

    baseline_score = behavior["baseline_external_scores"]
    baseline_margin = behavior["baseline_external_margin"]
    score_seed = behavior["best_validation_seed"]["scores"]
    all_seed = behavior["best_validation_seed"]["all"]
    eval_names = corpora
    score_row = behavior["runs"]["scores"][score_seed]
    all_row = behavior["runs"]["all"][all_seed]
    def mean_score(row):
        return np.mean([
            row["evaluation"][corpus][component]
            for corpus in eval_names for component in ("prediction", "target")])
    def mean_margin(row):
        return np.mean([row["evaluation"][corpus]["margin"] for corpus in eval_names])
    groups = ("HOSVD-16", "score-optimized", "margin-aware")
    scores = (baseline_score, mean_score(score_row), mean_score(all_row))
    margins = (baseline_margin, mean_margin(score_row), mean_margin(all_row))
    x = np.arange(3); width = .34
    axes[2].bar(x - width / 2, scores, width, label="mean score", color="#2878b5")
    axes[2].bar(x + width / 2, margins, width, label="mean margin", color="#e45756")
    axes[2].set_xticks(x, groups, rotation=18, ha="right")
    axes[2].set_ylim(0, 1.0)
    axes[2].set_ylabel("external fidelity")
    axes[2].set_title("C. Endpoint-relative optimization")
    axes[2].legend(fontsize=8)
    overlap = behavior["cross_objective_overlap"]
    axes[2].text(
        .02, .03,
        "score/all overlap O/T/C:\n"
        + "/".join(f"{overlap[role]:.2f}" for role in roles),
        transform=axes[2].transAxes, fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#bbb", "alpha": .9})

    for axis in axes:
        axis.grid(axis="y", alpha=.18)
    fig.suptitle(
        "Bilin18 final-MLP backward path: compact program, endpoint-dependent precision, non-unique coordinates",
        fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
