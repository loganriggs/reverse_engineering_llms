#!/usr/bin/env python3
"""Plot and summarize the causal-synergy depth sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/causal_synergy")
COLORS = {"lookup only": "#2774ae", "lookup after parity training": "#2a9d8f", "parity-2": "#d65f32"}


def causal_metric(row, field="median_of_example_medians"):
    return row["causal"]["summary"]["targeted_removal"][field]


def random_metric(row):
    return row["causal"]["summary"]["random_removal"]["median_of_example_medians"]


def memory_groups(rows):
    return {
        "lookup only": [row for row in rows if row["family"] == "memory" and row["task"] == "lookup" and row["group"] == "lookup"],
        "lookup after parity training": [row for row in rows if row["family"] == "memory" and row["task"] == "lookup_parity2" and row["group"] == "lookup"],
        "parity-2": [row for row in rows if row["family"] == "memory" and row["task"] == "lookup_parity2" and row["group"] == "parity2"],
    }


def plot_seed_means(axis, groups, getter, title, ylabel, ylim=None):
    depths = np.arange(1, 5)
    for name, current in groups.items():
        means = []
        for depth in depths:
            values = [getter(row) for row in current if row["depth"] == depth]
            values = [value for value in values if value is not None and np.isfinite(value)]
            means.append(np.mean(values) if values else np.nan)
            axis.scatter([depth] * len(values), values, color=COLORS[name], s=30, alpha=.75)
        axis.plot(depths, means, marker="o", color=COLORS[name], linewidth=2, label=name)
    axis.set(title=title, xlabel="layers", ylabel=ylabel, xticks=depths)
    if ylim: axis.set_ylim(*ylim)
    axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)


def grammar_condition(row):
    if row["rho"] == 0: return "structured"
    if row["rho"] == .5 and row["group"] == "exception": return "mixed exceptions"
    if row["rho"] == 1: return "arbitrary"
    return None


def plot_grammar(axis, rows, architecture, getter, title, ylabel, ylim=None):
    colors = {"structured": "#2774ae", "mixed exceptions": "#e9a23b", "arbitrary": "#d65f32"}
    depths = np.arange(1, 5)
    for condition, color in colors.items():
        current = [row for row in rows if row["family"] == "grammar" and row["architecture"] == architecture
                   and grammar_condition(row) == condition]
        means = []
        for depth in depths:
            row = next(item for item in current if item["depth"] == depth)
            value = getter(row); means.append(value)
            axis.scatter(depth, value, s=45, facecolors=color if row["group_accuracy"] >= .9 else "white",
                         edgecolors=color, linewidth=1.5, zorder=3)
        axis.plot(depths, means, marker="o", color=color, linewidth=2, label=condition)
    axis.set(title=f"{architecture}: {title}", xlabel="layers", ylabel=ylabel, xticks=depths)
    if ylim: axis.set_ylim(*ylim)
    axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False)


def matching_row(rows, **values):
    return next(row for row in rows if all(row.get(key) == value for key, value in values.items()))


def plot_examples(rows):
    examples = [
        ("One-layer lookup\n(additive negative control)", matching_row(rows, family="memory", task="lookup", depth=1, seed=7, group="lookup")),
        ("Four-layer parity\n(compositional interaction)", matching_row(rows, family="memory", task="lookup_parity2", depth=4, seed=7, group="parity2")),
        ("Three-layer arbitrary grammar\n(context-sensitive interaction)", matching_row(rows, family="grammar", architecture="swiglu", depth=3, rho=1.0, group="arbitrary")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, (title, row) in zip(axes, examples):
        example = row["causal"]["largest_interaction"]
        raw = np.asarray(example["margins_11_10_01_00"]).reshape(2, 2)
        centered = raw - raw[1, 1]
        scale = max(np.abs(centered).max(), 1e-8)
        axis.imshow(centered / scale, cmap="coolwarm", vmin=-1, vmax=1)
        for i in range(2):
            for j in range(2): axis.text(j, i, f"{raw[i,j]:.2f}", ha="center", va="center", fontsize=10)
        axis.set(title=f"{title}\nJ={example['interaction']:.3g}, fraction={example['interaction_fraction']:.3f}",
                 xticks=(0, 1), xticklabels=("B on", "B off"), yticks=(0, 1), yticklabels=("A on", "A off"))
    fig.suptitle("Exact 2×2 neuron interventions; cells show correct-answer margin", fontsize=14)
    fig.savefig(ROOT / "factorial_examples.png", dpi=180); plt.close(fig)


def mean(rows, getter):
    values = [getter(row) for row in rows]
    values = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def main():
    payload = json.loads((ROOT / "results.json").read_text()); rows = payload["rows"]
    groups = memory_groups(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_seed_means(axes[0, 0], groups, causal_metric, "Exact pair interaction", "normalized |J|")
    plot_seed_means(axes[0, 1], groups,
                    lambda row: row["causal"]["summary"]["removal_targeted_minus_random"],
                    "Targeted minus random pairs", "interaction-fraction advantage")
    plot_seed_means(axes[1, 0], groups,
                    lambda row: row["information"]["median_adjusted_complementary_bits"],
                    "Joint activation adds answer information", "adjusted complementary bits")
    plot_seed_means(axes[1, 1], groups,
                    lambda row: row["causal"]["global_pair_sign_consistency"]["answer_conditioned_median"],
                    "Interaction sign stability", "answer-conditioned consistency", (.45, 1.03))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fixed self-bilinear memory tasks across depth", fontsize=15)
    fig.savefig(ROOT / "memory_depth_synergy.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    for column, architecture in enumerate(("swiglu", "bilinear")):
        plot_grammar(axes[0, column], rows, architecture, causal_metric,
                     "exact pair interaction", "normalized |J|")
        plot_grammar(axes[1, column], rows, architecture,
                     lambda row: row["causal"]["summary"]["targeted_removal"]["mean_sign_flip_fraction"],
                     "within-example causal sign flips", "fraction of targeted pairs", (0, .11))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Grammar depth sweep: filled points have ≥90% group accuracy", fontsize=15, y=.985)
    fig.subplots_adjust(left=.09, right=.98, bottom=.08, top=.91, hspace=.42, wspace=.27)
    fig.savefig(ROOT / "grammar_depth_synergy.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    markers = {"grammar": "o", "memory": "s"}
    for family in ("grammar", "memory"):
        current = [row for row in rows if row["family"] == family]
        axis.scatter([causal_metric(row) for row in current],
                     [row["information"]["median_adjusted_complementary_bits"] for row in current],
                     marker=markers[family], alpha=.7, label=family)
    axis.axvline(0, color="black", linewidth=.8); axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="exact normalized pair interaction", ylabel="adjusted complementary information (bits)",
             title="Joint information is not the same as causal interaction")
    axis.grid(alpha=.2); axis.spines[["top", "right"]].set_visible(False); axis.legend(frameon=False)
    fig.savefig(ROOT / "information_vs_causal.png", dpi=180); plt.close(fig)
    plot_examples(rows)

    lines = ["# Neurons as context-dependent discriminators", "",
             "## Short answer", "",
             "Joint neuron codes are clearly present, but the strongest proposed definition of polysemanticity—one neuron's causal effect reversing when a partner is toggled—is uncommon in these tasks. Depth creates non-additive interactions; most of the memory interactions remain answer-conditioned and sign-stable, which is more consistent with compositional verbs than arbitrary multiplexing.", "",
             "## Self-bilinear memory: fixed tasks across 1–4 layers", "",
             "All 24 models converged. Values are means over three seeds. `Interaction` is the median exact removal-context interaction fraction for top causal pairs. `Complement` is joint answer information beyond the better individual neuron, after a permutation-null correction.", "",
             "| task | layers | accuracy | interaction | random-pair interaction | complement bits | within-example sign flips | answer-conditioned sign stability |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, current in groups.items():
        for depth in range(1, 5):
            depth_rows = [row for row in current if row["depth"] == depth]
            lines.append(
                f"| {name} | {depth} | {mean(depth_rows, lambda r:r['group_accuracy']):.3f} | "
                f"{mean(depth_rows, causal_metric):.4f} | {mean(depth_rows, random_metric):.4f} | "
                f"{mean(depth_rows, lambda r:r['information']['median_adjusted_complementary_bits']):.3f} | "
                f"{mean(depth_rows, lambda r:r['causal']['summary']['targeted_removal']['mean_sign_flip_fraction']):.3f} | "
                f"{mean(depth_rows, lambda r:r['causal']['global_pair_sign_consistency']['answer_conditioned_median']):.3f} |"
            )
    lines += ["", "The one-layer interaction is approximately `10^-6`, validating the additive negative control. Nevertheless, one-layer activations contain complementary joint information. Therefore a distributed joint code does not by itself imply context-dependent causal meaning.", "",
              "## Grammar transformers", "",
              "Open graph markers have less than 90% group accuracy and are not used as strong mechanistic evidence. The one-layer bilinear cells are especially underfit.", "",
              "| MLP | layers | structured interaction | mixed-exception interaction | arbitrary interaction | structured sign flips | arbitrary sign flips |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for architecture in ("swiglu", "bilinear"):
        for depth in range(1, 5):
            def row(rho, group):
                return matching_row(rows, family="grammar", architecture=architecture, depth=depth, rho=rho, group=group)
            structured, exception, arbitrary = row(0.0, "structured"), row(.5, "exception"), row(1.0, "arbitrary")
            lines.append(
                f"| {architecture} | {depth} | {causal_metric(structured):.3f} | {causal_metric(exception):.3f} | "
                f"{causal_metric(arbitrary):.3f} | {structured['causal']['summary']['targeted_removal']['mean_sign_flip_fraction']:.3f} | "
                f"{arbitrary['causal']['summary']['targeted_removal']['mean_sign_flip_fraction']:.3f} |"
            )
    lines += ["", "## Verdicts", "",
              "1. **Validated:** the exact factorial metric detects zero interaction in the architecture where additivity is guaranteed, then detects increasing interaction when downstream nonlinear layers are added.",
              "2. **Validated:** multiple neurons jointly add answer information even when their causal writes are additive. This is distributed coding, not sufficient evidence of polysemanticity.",
              "3. **Supported:** depth creates context-dependent composition. Targeted pairs interact more than random pairs, and the gap generally grows with depth in self-bilinear memory.",
              "4. **Not supported as a general rule:** arbitrary facts do not consistently cause more pair interaction than structured rules. The ordering varies with depth and architecture.",
              "5. **Mostly falsified in its strongest form:** partner-dependent causal sign reversal is rare—zero in the memory panels and usually below 5% in grammar, with the largest reliable exploratory value around 7.7% for depth-3 arbitrary SwiGLU.",
              "6. **More plausible interpretation:** most measured neurons behave like stable discriminating verbs composed into joint codes. Multilayer models add interaction and cancellation, but the current evidence for widespread contextual multiplexing of individual neurons is weak.", "",
              "## Red-team limits", "",
              "- Grammar depths use one data/model seed; depth-4 has earlier replications, but the new interaction-depth trend is exploratory.",
              "- The audit examines the top eight first-order neurons per example and all 28 pairs among them. Higher-order interactions can exist without strong pair effects.",
              "- Interaction is defined on an answer margin and depends on the chosen neuron basis. Random-pair controls address selection, not rotation dependence.",
              "- Intact margins grow strongly with depth in the self-bilinear models, so raw `J` values are not comparable across depth. The primary depth plot uses a dimensionless interaction fraction; even that normalization should not be read as a complete scale correction.",
              "- Top neurons and their pair effects are measured on the same causal panel. The exact interactions are valid, but the targeted-versus-random advantage is exploratory until pair selection and evaluation use separate examples.",
              "- The sign-flip threshold is 0.05 raw margin units. This makes cross-depth flip rates conservative at shallow depth and easier to exceed at large-margin deep models; the zero memory flip result remains notable despite this bias.",
              "- Complementary information is observational. The exact factorial intervention is the causal test.",
              "- Answer-conditioned sign stability has small groups for the eight-class grammar and should be replicated with larger panels.", "",
              "## Figures", "",
              "- `memory_depth_synergy.png`: replicated fixed-task memory depth sweep.",
              "- `grammar_depth_synergy.png`: structured versus arbitrary grammar interactions.",
              "- `information_vs_causal.png`: why joint information and causal context dependence differ.",
              "- `factorial_examples.png`: three concrete 2×2 neuron interventions."]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(ROOT / "summary.md")


if __name__ == "__main__":
    main()
