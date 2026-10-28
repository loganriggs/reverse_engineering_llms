#!/usr/bin/env python3
"""Render the weight-first circuit discovery summary and report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/bilin18_weight_only_circuits")


def fmt(value):
    return "n/a" if not np.isfinite(value) else f"{value:.3f}"


def main():
    discovery = json.loads((ROOT / "discovery.json").read_text())
    validation = json.loads((ROOT / "validation.json").read_text())
    names = validation["names"]
    pretty = [name.replace("_", " ") for name in names]
    panels = ("fineweb", "wikitext103", "tinystories")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    aucs = np.asarray([[validation["semantic"][panel]["auc_matrix"][name][name]
                       for panel in panels] for name in names])
    image = axes[0].imshow(aucs, vmin=.5, vmax=1, cmap="viridis")
    axes[0].set_xticks(range(3), ["FineWeb", "WikiText", "TinyStories"], rotation=25)
    axes[0].set_yticks(range(len(names)), pretty)
    axes[0].set_title("Frozen decoder AUROC")
    for i in range(len(names)):
        for j in range(3):
            axes[0].text(j, i, fmt(aucs[i, j]), ha="center", va="center",
                         color="white" if np.isfinite(aucs[i, j]) and aucs[i, j] < .78 else "black",
                         fontsize=8)
    fig.colorbar(image, ax=axes[0], fraction=.046)

    ratios = [validation["causal"][name]["diagonal_ratio"] for name in names]
    axes[1].barh(range(len(names)), ratios, color="#4c78a8")
    axes[1].set_yticks(range(len(names)), pretty)
    axes[1].invert_yaxis()
    axes[1].axvline(1, color="black", lw=1, ls="--")
    axes[1].set_xlabel("matching-group KL / other-group KL")
    axes[1].set_title("Causal specificity on FineWeb")
    for i, value in enumerate(ratios):
        axes[1].text(value + .25, i, f"{value:.1f}×", va="center", fontsize=8)

    ks = [32, 128, 512, 2048]
    colors = {"weight": "#e45756", "mismatched": "#72b7b2",
              "random": "#bab0ac", "oracle": "#f2cf5b"}
    circuit = "suffix_ing"
    for kind in colors:
        values = [validation["path"][circuit]["median_recovery"][kind][str(k)] for k in ks]
        axes[2].plot(ks, values, marker="o", label=kind, color=colors[kind])
    axes[2].axhline(0, color="black", lw=.8)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(ks, [str(k) for k in ks])
    axes[2].set_xlabel("restored layer-2 channels")
    axes[2].set_ylabel("median recovered logit-delta fraction")
    axes[2].set_title("“-ing” downstream path is distributed")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "weight_first_circuit_summary.png", dpi=180)
    plt.close(fig)

    rows = []
    for name in names:
        source = next(row for row in discovery["categories"] if row["name"] == name)
        semantic = [validation["semantic"][panel]["auc_matrix"][name][name] for panel in panels]
        causal = validation["causal"][name]
        path = validation["path"][name]
        rows.append(
            f"| {name} | {', '.join(map(str, source['units']))} | "
            f"{source['heldout_auc']:.3f} | {' / '.join(fmt(x) for x in semantic)} | "
            f"{causal['diagonal_ratio']:.1f}× | "
            f"{path['mean_weight_enrichment']['32']:.1f}× | "
            f"{path['median_recovery']['weight']['32']:.1%} / "
            f"{path['median_recovery']['weight']['2048']:.1%} |"
        )

    ing_source = next(row for row in discovery["categories"] if row["name"] == "suffix_ing")
    ing_causal = validation["causal"]["suffix_ing"]
    ing_path = validation["path"]["suffix_ing"]
    example = ing_path["sites"][0]
    indiv = ing_causal["individual_neurons"]
    unit_lines = "\n".join(
        f"- neuron {unit}: isolated-token standardized gap {gap:+.2f}; "
        f"mean one-neuron ablation KL {indiv['mean_kl'][str(unit)]:.4g}"
        for unit, gap in zip(ing_source["units"], ing_source["unit_discovery_gaps"])
    )
    report = f"""# Weight-first discovery of six layer-1 circuits

## Bottom line

Yes: a useful local circuit can be proposed from isolated-token neuron responses and weight matrices, then validated on text that played no role in discovery. The clearest example is an **“-ing” detector in layer 1**. Four frozen neurons recognize it across FineWeb, WikiText-103, and TinyStories; their joint ablation is {ing_causal['diagonal_ratio']:.1f}× more disruptive on selected “-ing” sites than the average ablation of the other five groups; and the layer-1→layer-2 weights predict which downstream neurons change by {ing_path['mean_weight_enrichment']['32']:.1f}× over the layer-2 average.

It is not a small end-to-end circuit. Restoring the 32 strongest weight-predicted layer-2 targets recovers a median {ing_path['median_recovery']['weight']['32']:.1%} of the logit effect; even 2,048 of 4,608 targets recover only {ing_path['median_recovery']['weight']['2048']:.1%}. The result is therefore a **clean source predicate feeding a distributed downstream computation**, not a four-neuron semantic module with a few output wires.

![Summary](weight_first_circuit_summary.png)

## What “weight-first” means here

No natural-language corpus was opened during discovery. Every tokenizer item was run as an isolated one-token sequence. Even token IDs selected channels; odd token IDs tested them. Category names came from predeclared tokenizer-string predicates, so this is *not* unsupervised semantics from raw matrices alone. It is a strict “weights plus vocabulary inventory” search. The source groups and downstream layer-2 rankings were frozen before FineWeb, WikiText, or TinyStories validation.

Layer-2 targets were ranked using only the local path magnitude

`layer1 product channel → layer1 Down direction → layer2 Left/Right factors`.

## Six frozen hypotheses

| circuit | layer-1 neurons | isolated held-out AUROC | natural AUROC: FW / WT / TS | causal specificity | top-32 delta enrichment | median recovery: 32 / 2048 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

TinyStories contains no positive non-ASCII examples in this panel, so its AUROC is undefined. “Word continuation” and “-ly” are the weakest transfers; they remain in the table because hypotheses were frozen rather than replaced after seeing natural data.

## Concrete “-ing” circuit

Source code:

{unit_lines}

The signs matter: neuron 2780 becomes quieter for “-ing”, while 896, 3712, and 1846 become larger. The four-channel decoder therefore uses a signed pattern, not merely “four neurons turn on.” Each source neuron is independently causal, although 896 and 2780 have the largest average effects. The sum of four separate ablation-induced logit changes has median cosine {indiv['median_linear_sum_cosine']:.3f} with the simultaneous four-neuron ablation and norm ratio {indiv['median_linear_sum_norm_ratio']:.3f}; the circuit is only moderately linear once propagated through 16 later layers.

One held-out FineWeb site was:

> …{example['context']}

The current token is `{example['token']}`. At this site the 32 weight-ranked layer-2 targets changed {example['weight_target_enrichment']['32']:.1f}× more than an average layer-2 channel. The particular example is also a warning: restoring those targets did not repair the final output for this site. Across sites, recovery becomes reliable only for much larger target sets.

## Red-team results

1. **Semantic transfer:** “-ed” and “-ing” are strong on all three corpora. All-caps is also strong. Word-continuation degrades to 0.625–0.705; “-ly” falls to 0.558 on TinyStories. Token-inventory success alone would have overstated those circuits.
2. **Cross-circuit ablations:** all six matching groups beat the average other group, with ratios from {min(ratios):.1f}× to {max(ratios):.1f}×. This controls for generic damage from ablating four layer-1 channels, though selection of high-scoring positive sites favors the matching group.
3. **Individual-neuron concentration:** the “-ing” result is not carried by one neuron. Mean one-neuron KLs range from {min(indiv['mean_kl'].values()):.4g} to {max(indiv['mean_kl'].values()):.4g}; the simultaneous group effect is {indiv['mean_group_over_sum_individual_kl']:.2f}× the sum of individual KLs.
4. **Weight-path prediction:** for “-ing”, the top 32 predicted targets have {ing_path['mean_weight_enrichment']['32']:.1f}× the average standardized activation change. The advantage declines as the set expands, as it should: {ing_path['mean_weight_enrichment']['2048']:.2f}× at 2,048 targets.
5. **Sufficiency fails at small scale:** 32-target restoration is weak and sometimes negative. Negative recovery means the partial repair perturbed compensating downstream interactions more than it helped. Even the post-hoc oracle ranking is not monotonic, demonstrating nonlinear interactions and making “percentage stored in these neurons” an invalid interpretation.
6. **Control rankings:** for “-ing”, 2,048 weight-ranked targets recover a median {ing_path['median_recovery']['weight']['2048']:.1%}, versus {ing_path['median_recovery']['mismatched']['2048']:.1%} for another circuit's ranking and {ing_path['median_recovery']['random']['2048']:.1%} for random targets. The weights contain circuit-specific information, but much of the path is broad.

## Interpretation

This is the coexistence regime we were looking for. The source representation is fairly neuron-aligned and verb-like: a small signed set discriminates a morphological condition. Causation after that point is dense. Dense connectivity does not make the source neurons meaningless, and clean source neurons do not imply a compact downstream circuit.

The strongest warranted claim is local: **we can discover a semantic detector and a statistically predictive downstream route from weights and the vocabulary, then validate both causally.** We cannot yet recover a faithful sparse end-to-end mechanism from weights alone. The next useful step is to trace “-ing” through layer 2 using low-rank subspaces or groups of downstream neurons rather than expecting individual target neurons to be sufficient.

## Reproduction

```bash
uv run python scripts/discover_bilin18_weight_only_circuits.py
uv run python scripts/validate_bilin18_weight_only_circuits.py
MPLBACKEND=Agg uv run python scripts/report_bilin18_weight_only_circuits.py
```

Raw results: `discovery.json` and `validation.json` in this directory.
"""
    (ROOT / "REPORT.md").write_text(report)
    print(f"wrote {ROOT / 'REPORT.md'} and summary figure")


if __name__ == "__main__":
    main()
