#!/usr/bin/env python3
"""Summarize activation sparsity and layer1->2 interference experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/bilin18_activation_interference")
CORPORA = ("fineweb", "wikitext103", "tinystories")
KS = (1, 2, 4, 8, 16, 32)


def select(result, mode, k, amplitude=None, route="both", field="noise_to_joint"):
    return np.asarray([
        row["routes"][route][field] for row in result["interference_records"]
        if row["mode"] == mode and row["k"] == k and
        (amplitude is None or row["amplitude_z"] == amplitude)
    ], dtype=np.float64)


def bootstrap_median(values, seed=20261019):
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))]
    medians = np.median(samples, axis=1)
    return [float(np.quantile(medians, .025)), float(np.quantile(medians, .975))]


def main():
    bf16 = json.loads((ROOT / "results.json").read_text())
    fp32 = json.loads((ROOT / "results_float32.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    thresholds = (1, 2, 3)
    width = .25
    x = np.arange(len(thresholds))
    for offset, corpus in enumerate(CORPORA):
        values = [bf16["sparsity"][corpus]["fraction_abs_z_above"][str(z)]
                  for z in thresholds]
        axes[0, 0].bar(x + (offset - 1) * width, values, width,
                       label=corpus.replace("103", "-103"))
    axes[0, 0].set_xticks(x, [f"|z| > {z}" for z in thresholds])
    axes[0, 0].set_ylabel("fraction of 4,608 channels")
    axes[0, 0].set_title("Natural activation is heavy-tailed")
    axes[0, 0].legend(frameon=False)

    energy_ks = (8, 32, 128, 512, 2048)
    for corpus in CORPORA:
        axes[0, 1].plot(energy_ks,
                        [bf16["sparsity"][corpus]["write_energy_topk_fraction"][str(k)]
                         for k in energy_ks], marker="o", label=corpus)
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xlabel("largest write-weighted channels per token")
    axes[0, 1].set_ylabel("fraction of squared write magnitude")
    axes[0, 1].set_title("Per-token energy concentration")
    axes[0, 1].legend(frameon=False)

    for result, label, style in ((bf16, "bf16 execution", "--"),
                                 (fp32, "float32 diagnostic", "-")):
        axes[1, 0].plot(KS, [np.median(select(result, "natural_remove", k)) for k in KS],
                        marker="o", ls=style, label=label)
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xlabel("strongest natural channels jointly removed")
    axes[1, 0].set_ylabel("interaction noise / joint effect")
    axes[1, 0].set_title("Natural active-set nonadditivity")
    axes[1, 0].legend(frameon=False)

    colors = {1.0: "#4c78a8", 2.0: "#f58518", 3.0: "#e45756"}
    for amplitude in (1.0, 2.0, 3.0):
        axes[1, 1].plot(KS[1:], [np.median(select(fp32, "extra_on", k, amplitude)) for k in KS[1:]],
                        marker="o", color=colors[amplitude], label=f"{amplitude:.0f}σ float32")
        axes[1, 1].plot(KS[1:], [np.median(select(bf16, "extra_on", k, amplitude)) for k in KS[1:]],
                        ls="--", color=colors[amplitude], alpha=.7, label=f"{amplitude:.0f}σ bf16")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("previously quiet channels turned on")
    axes[1, 1].set_ylabel("interaction noise / joint effect")
    axes[1, 1].set_title("More simultaneous features increase interference")
    axes[1, 1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "activation_interference_summary.png", dpi=180)
    plt.close(fig)

    natural_bf = select(bf16, "natural_remove", 32)
    natural_fp = select(fp32, "natural_remove", 32)
    extra_bf = select(bf16, "extra_on", 32, 2.0)
    extra_fp = select(fp32, "extra_on", 32, 2.0)
    sparsity = {
        threshold: np.mean([bf16["sparsity"][c]["fraction_abs_z_above"][str(threshold)]
                            for c in CORPORA]) for threshold in thresholds}
    participation = np.mean([bf16["sparsity"][c]["write_effective_participation"]
                             for c in CORPORA])
    report = f"""# Sparse activity and interference on the layer-1 → layer-2 path

## Bottom line

The current model uses a **sparse-per-token but broadly reused population code**. On an average token, {sparsity[2]:.1%} of the 4,608 layer-1 channels—about {sparsity[2] * 4608:.0f}—are more than two positional standard deviations from their mean, and {sparsity[3]:.1%}—about {sparsity[3] * 4608:.0f}—exceed three. Write-weighted effective participation is about {participation:.0f} channels per token. Yet the identity of those channels changes substantially across tokens: top-32 sets have median pairwise Jaccard zero, while usage entropy spans roughly 2,800–3,700 distinct effective neurons across each corpus.

Turning on more computations does increase algebraic interference, but much less than the native bf16 measurements initially suggested. In a float32 diagnostic, adding 32 previously quiet channels at `2σ` produces interaction noise equal to only {np.median(extra_fp):.1%} of their joint layer-2 effect. Under checkpoint-native bf16 execution, the same statistic is {np.median(extra_bf):.0%}. The amplitude sweep shows that most of that gap is finite-precision nonadditivity rather than bilinear feature collision.

![Activation and interference summary](activation_interference_summary.png)

## Operational definition

For a set of changed layer-1 channels `S`, let `Δh₂(S)` be the standardized change in all 4,608 layer-2 MLP products. Interaction noise is

```text
n(S) = Δh₂(S) − Σᵢ Δh₂({{i}}).
```

The main statistic is `||n(S)|| / ||Δh₂(S)||`. Zero means singleton effects add perfectly. One means the nonadditive component is as large as the observed joint effect. This is a local superposition test, not task-performance loss or random background activation.

The calculation was repeated for:

- the direct `MLP1 → MLP2` path, freezing attention-2's write;
- `MLP1 → attention2 → MLP2`, freezing the direct residual input;
- both paths together.

## How sparse is natural activity?

| Corpus | `|z|>1` | `|z|>2` | `|z|>3` | Effective write participants | Top-32 write energy | Top-128 write energy |
|---|---:|---:|---:|---:|---:|---:|
"""
    for corpus in CORPORA:
        row = bf16["sparsity"][corpus]
        report += (f"| {corpus} | {row['fraction_abs_z_above']['1']:.1%} | "
                   f"{row['fraction_abs_z_above']['2']:.1%} | "
                   f"{row['fraction_abs_z_above']['3']:.1%} | "
                   f"{row['write_effective_participation']:.0f} | "
                   f"{row['write_energy_topk_fraction']['32']:.1%} | "
                   f"{row['write_energy_topk_fraction']['128']:.1%} |\n")
    report += f"""

The top 32 channels carry about 30% of write-weighted energy; 128 carry about 52%; 512 carry about 79%. This is neither a one-hot code nor a dense uniform code.

Top-set diversity is high:

| Corpus | Effective neurons used by top-32 selections | Mean pairwise top-32 Jaccard |
|---|---:|---:|
"""
    for corpus in CORPORA:
        diversity = bf16["sparsity"][corpus]["top_set_diversity"]["32"]
        report += (f"| {corpus} | {diversity['effective_neurons_used']:.0f} | "
                   f"{diversity['random_pair_jaccard_mean']:.3f} |\n")
    report += f"""

This is the expected qualitative pattern for superposition: only a minority of channels dominate any token, but many different channels participate across the dataset.

## Naturally active channels

We jointly mean-ablated each site's strongest write-weighted channels and compared the result with the sum of their singleton ablations.

| Removed channels | bf16 noise/joint | float32 noise/joint |
|---:|---:|---:|
"""
    for k in KS:
        report += (f"| {k} | {np.median(select(bf16, 'natural_remove', k)):.3f} | "
                   f"{np.median(select(fp32, 'natural_remove', k)):.3f} |\n")
    report += f"""

At 32 channels, the bf16 median is {np.median(natural_bf):.3f} with bootstrap 95% interval {bootstrap_median(natural_bf)[0]:.3f}–{bootstrap_median(natural_bf)[1]:.3f}. The float32 diagnostic is {np.median(natural_fp):.3f}, interval {bootstrap_median(natural_fp)[0]:.3f}–{bootstrap_median(natural_fp)[1]:.3f}. The float32 result is consistent across FineWeb, WikiText, and TinyStories: approximately 0.19–0.20.

Thus strong natural channels do interact, but their joint effect remains mostly predicted by summing singleton effects: at 32 channels the float32 linear-versus-joint cosine is about {np.median(select(fp32, 'natural_remove', 32, field='linear_joint_cosine')):.3f}.

## Counterfactually turning quiet channels on

Channels with `|z|<0.25` were selected before intervention, assigned fixed random signs, and set to one, two, or three standard deviations. The same nested channel order was used as `K` increased.

### Float32 algebraic interaction

| Added channels | `1σ` | `2σ` | `3σ` |
|---:|---:|---:|---:|
"""
    for k in KS:
        report += (f"| {k} | {np.median(select(fp32, 'extra_on', k, 1.0)):.3f} | "
                   f"{np.median(select(fp32, 'extra_on', k, 2.0)):.3f} | "
                   f"{np.median(select(fp32, 'extra_on', k, 3.0)):.3f} |\n")
    report += f"""

At fixed `K=32`, absolute interaction norm scales approximately quadratically with amplitude: 0.117 at `1σ`, 0.460 at `2σ`, and 1.038 at `3σ`. This is the expected signature of bilinear cross-terms. Relative noise grows roughly linearly with amplitude because the first-order signal grows linearly while pair interactions grow quadratically.

### Native bf16 execution

| Added channels | `1σ` | `2σ` | `3σ` |
|---:|---:|---:|---:|
"""
    for k in KS:
        report += (f"| {k} | {np.median(select(bf16, 'extra_on', k, 1.0)):.3f} | "
                   f"{np.median(select(bf16, 'extra_on', k, 2.0)):.3f} | "
                   f"{np.median(select(bf16, 'extra_on', k, 3.0)):.3f} |\n")
    report += f"""

In bf16, absolute interaction norm at `K=32` is nearly amplitude-independent: approximately 12.3, 13.0, and 13.4. Consequently the relative statistic is worst for the smallest injected signals. This is a rounding/noise floor, not the scaling law expected from representational collisions.

## Direct and attention-mediated paths

In the float32 diagnostic at `K=32`, `2σ` extra activation gives:

- direct path noise/joint: {np.median(select(fp32, 'extra_on', 32, 2.0, route='direct')):.3f};
- attention-only path: {np.median(select(fp32, 'extra_on', 32, 2.0, route='attention')):.3f};
- both paths: {np.median(extra_fp):.3f}.

The direct route has about 0.96–0.98 times the combined-effect norm. The attention route has about 0.28–0.32 times the combined norm and points partly outside the direct direction. The direct×attention cross-term reaches about 9.7% for removing the 32 strongest natural channels, but only 2.5% for adding 32 random quiet channels at `2σ`.

Attention therefore adds a distinct route, but it is not the main source of mathematical interference in this layer transition.

## What this says about superposition

The evidence supports a qualified version of the hypothesis:

1. **Activity is sparse relative to width.** Roughly 100–250 channels are unusually active per token, depending on threshold, and effective write participation is about 190–200 of 4,608.
2. **The active identity is highly multiplexed.** Thousands of different channels enter top sets across tokens.
3. **Algebraic interference grows with both active count and amplitude.** This matches bilinear cross-term predictions.
4. **The current regime is not near an immediate algebraic capacity wall.** Thirty-two added `2σ` features produce only 6.3% float32 interaction relative to their joint effect.
5. **Native bf16 arithmetic introduces a much larger nonadditivity floor.** Whether the trained model treats this as harmful noise, useful discretization, or compensated error requires task-level tests; it must not be mislabeled as representational superposition.

The strongest current picture is therefore not “features must be nearly one-hot or they catastrophically collide.” It is a heavy-tailed population code where tens to hundreds of features coexist, pairwise bilinear interactions grow predictably, and the model has substantial algebraic headroom. Finite precision is a larger measured source of local nonadditivity than feature collision for the random quiet-channel intervention.

## Limitations

- The extra-on intervention is deliberately synthetic and may leave the natural activation manifold.
- “On” is threshold-dependent because bilinear products are signed and rarely exactly zero.
- Float32 is a diagnostic evaluation of bf16-trained weights, not the checkpoint's native execution.
- Interaction norm is not automatically harmful task noise; some interactions may implement useful conjunctions.
- The experiment ends at MLP2 products and intentionally ignores later residual-stream consequences.
- Natural removals target the largest write-weighted channels and therefore differ from random quiet-channel additions.

## Reproduction

```bash
uv run python scripts/run_bilin18_activation_interference.py
uv run python scripts/run_bilin18_activation_interference.py --float32
MPLBACKEND=Agg uv run python scripts/report_bilin18_activation_interference.py
```
"""
    (ROOT / "REPORT.md").write_text(report)
    print(f"wrote {ROOT / 'REPORT.md'} and summary figure")


if __name__ == "__main__":
    main()
