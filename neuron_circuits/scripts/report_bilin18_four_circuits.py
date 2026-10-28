#!/usr/bin/env python3
"""Write the four-circuit/subspace report and summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/bilin18_four_circuits")
NAMES = ["boundary_reset", "capital_word_start", "number_format", "determiner_syntax"]
SHORT = ["Boundary", "Capital", "Number", "Determiner"]


def main():
    analysis = json.loads((ROOT / "analysis.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    qkov = json.loads((ROOT / "qkov" / "analysis.json").read_text())
    semantic = analysis["semantic_validation"]["panels"]
    cross = analysis["circuit_causal_tests"]["positive_cross_ablation"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    panels = ["fineweb_test", "wikitext103", "tinystories"]
    x = np.arange(len(NAMES))
    for offset, panel, label in zip((-.24, 0, .24), panels,
                                    ("FineWeb held out", "WikiText-103", "TinyStories")):
        values = [semantic[panel]["decoder_auc_matrix"][name][name] for name in NAMES]
        ax.bar(x + offset, values, width=.23, label=label)
    ax.set_xticks(x, SHORT)
    ax.set_ylim(.8, 1.01)
    ax.set_ylabel("AUROC")
    ax.set_title("A. Frozen four-neuron semantic decoders transfer OOD")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    matrix = np.asarray([[cross[row][column]["mean_kl"] for column in NAMES]
                         for row in NAMES])
    image = ax.imshow(matrix * 1000, cmap="Blues")
    for row in range(4):
        for column in range(4):
            ax.text(column, row, f"{1000*matrix[row, column]:.2f}",
                    ha="center", va="center",
                    color="white" if matrix[row, column] > matrix.max() * .58 else "black")
    ax.set_xticks(range(4), SHORT, rotation=25, ha="right")
    ax.set_yticks(range(4), SHORT)
    ax.set_xlabel("Ablated neuron group")
    ax.set_ylabel("Current-token concept")
    ax.set_title("B. Causal isolation (mean KL × 1000)")
    fig.colorbar(image, ax=ax, shrink=.8)

    ax = axes[1, 0]
    for layer, marker in ((2, "o"), (3, "s")):
        rows = [row for row in analysis["subspaces"]["per_source"] if row["layer"] == layer]
        values = [np.mean([row["k"][str(k)]["projection_mean"] for row in rows]) for k in (1, 2, 4, 8, 16, 32)]
        ax.plot((1, 2, 4, 8, 16, 32), np.asarray(values) * 100,
                marker=marker, label=f"Layer {layer}")
    ax.set_xscale("log", base=2)
    ax.set_xticks((1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32))
    ax.set_xlabel("Source-specific subspace dimensions")
    ax.set_ylabel("Held-out propagation energy captured (%)")
    ax.set_title("C. Ordinary-site propagation remains high-dimensional")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    matched, mismatch = [], []
    for circuit in NAMES:
        rows = [row["k"]["16"] for row in analysis["subspaces"]["leave_source_out"]
                if row["layer"] == 2 and row["circuit"] == circuit]
        matched.append(np.mean([row["matched_mean"] for row in rows]) * 100)
        mismatch.append(np.mean([row["mismatched_mean"] for row in rows]) * 100)
    ax.bar(x - .18, matched, width=.36, label="same circuit, source held out")
    ax.bar(x + .18, mismatch, width=.36, label="other circuit")
    ax.set_xticks(x, SHORT)
    ax.set_ylabel("Layer-2 propagation energy captured (%)")
    ax.set_title("D. Only capitalization shares a clear circuit subspace")
    ax.legend(fontsize=8)

    figure = ROOT / "four_circuit_summary.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    auc_rows = {}
    for name in NAMES:
        auc_rows[name] = [semantic[panel]["decoder_auc_matrix"][name][name]
                          for panel in panels]
    ratios = {}
    for name in NAMES:
        diagonal = cross[name][name]["mean_kl"]
        other = np.mean([value["mean_kl"] for key, value in cross[name].items() if key != name])
        ratios[name] = diagonal / other

    source_k16 = {}
    source_k32 = {}
    for layer in (2, 3):
        rows = [row for row in analysis["subspaces"]["per_source"] if row["layer"] == layer]
        source_k16[layer] = np.mean([row["k"]["16"]["projection_mean"] for row in rows])
        source_k32[layer] = np.mean([row["k"]["32"]["projection_mean"] for row in rows])
    lso = {}
    for name in NAMES:
        rows = [row["k"]["16"] for row in analysis["subspaces"]["leave_source_out"]
                if row["layer"] == 2 and row["circuit"] == name]
        lso[name] = (
            np.mean([row["matched_mean"] for row in rows]),
            np.mean([row["mismatched_mean"] for row in rows]),
            np.mean([row["matched_site_win_fraction"] for row in rows]),
        )
    patches = analysis["causal_subspace_summary"]
    qtransfer = qkov["splits"]["transfer"]

    lines = [
        "# Four candidate circuits and ordinary-site propagation subspaces", "",
        "## Verdict", "",
        "The four frozen neuron groups are excellent and OOD-stable **input discriminators**, "
        "but only three are causally isolated by the present test, and only one—capitalization—"
        "shares a detectable propagation subspace across different source neurons. This sharply "
        "separates knowing what activates a neuron from knowing the reusable computation it "
        "participates in.", "",
        f"![Four-circuit summary]({figure.name})", "",
        "The strongest supported picture is:", "",
        "- neuron-level token predicates can be remarkably clean;",
        "- attention routes into those neurons are partly predictable from QK×OV weights;",
        "- each neuron produces a reproducible but high-dimensional downstream fingerprint;",
        "- neurons with the same apparent semantic label usually do **not** share the same "
        "downstream implementation;",
        "- semantic patching therefore needs context-conditioned directions and coefficients, "
        "not just a list of neurons with similar activation tails.", "",
        "## Experimental scale", "",
        f"The experiment freezes 16 layer-1 neurons in four groups. It collects {len(manifest['sites']):,} "
        "finite interventions: 64 discovery and 64 document-held-out sites per neuron. Every "
        "activation is an ordinary 0.5–1.5 positional-SD deviation, balanced by sign; no tail "
        "examples are used to fit the subspaces. Layer-2 and layer-3 product changes contain all "
        "4,608 native channels.", "",
        "Semantic decoders are trained on 102,400 FineWeb positions and evaluated on disjoint "
        "FineWeb, WikiText-103, and TinyStories panels of 102,400 positions each. Causal circuit "
        "tests use 64 concept-positive and 64 concept-negative sites per circuit.", "",
        "## Circuit-by-circuit results", "",
        "|Candidate|Frozen layer-1 neurons|FineWeb/WikiText/TinyStories AUROC|Matching-group KL advantage|Verdict|",
        "|---|---|---:|---:|---|",
        f"|Boundary/reset|1825, 291, 2590, 4574|{auc_rows['boundary_reset'][0]:.3f} / {auc_rows['boundary_reset'][1]:.3f} / {auc_rows['boundary_reset'][2]:.3f}|{ratios['boundary_reset']:.2f}×|Strong input and causal isolation|",
        f"|Capital/word start|1690, 2976, 175, 210|{auc_rows['capital_word_start'][0]:.3f} / {auc_rows['capital_word_start'][1]:.3f} / {auc_rows['capital_word_start'][2]:.3f}|{ratios['capital_word_start']:.2f}×|Input/causal isolation and shared subspace|",
        f"|Number format|1535, 3448, 4562, 2812|{auc_rows['number_format'][0]:.3f} / {auc_rows['number_format'][1]:.3f} / {auc_rows['number_format'][2]:.3f}|{ratios['number_format']:.2f}×|Strong input and causal isolation; output role unresolved|",
        f"|Determiner syntax|4361, 1775, 3691, 3996|{auc_rows['determiner_syntax'][0]:.3f} / {auc_rows['determiner_syntax'][1]:.3f} / {auc_rows['determiner_syntax'][2]:.3f}|{ratios['determiner_syntax']:.2f}×|Clean detector, but not an isolated causal circuit|", "",
        "The KL advantage compares ablating the matching four-neuron group with the average of "
        "ablating the other three groups on the exact same concept-positive examples. Boundary, "
        "capitalization, and number formatting have the largest diagonal effect. For determiners, "
        "the capitalization group is slightly more disruptive than the putative determiner group. "
        "Calling the fourth group a distinct circuit would therefore overstate the evidence.", "",
        "All four matching ablations have larger mean KL on concept-positive than concept-negative "
        "sites, but their absolute effects are small (roughly 0.0005–0.0015 KL) and almost never "
        "change top-1. These are real contributors, not complete task circuits.", "",
        "## What the neuron interpretations do and do not establish", "",
        "The lexical interpretations survive two genuine distribution shifts. Several individual "
        "neurons are also strong detectors by themselves, sometimes with reversed sign. Thus the "
        "four-neuron results are not merely an overfit linear probe.", "",
        "Their output roles are much less clean. Boundary/reset ablation most clearly removes "
        "support for common new-span word starts. Determiner neurons weakly support adjective-like "
        "continuations. The number detectors do not simply promote number tokens, and capitalization "
        "writes are heterogeneous. A clean input predicate can therefore feed a conditional or "
        "distributed output computation rather than have one fixed semantic write.", "",
        "The boundary result is also the easiest case: its label is the literal GPT-2 end-of-text "
        "token. It validates the machinery but should not be treated as evidence for a subtle "
        "abstract boundary concept by itself.", "",
        "## Ordinary-site subspaces", "",
        f"A 16-dimensional source-specific subspace captures only {100*source_k16[2]:.2f}% of "
        f"held-out layer-2 propagation energy and {100*source_k16[3]:.2f}% at layer 3. Even 32 "
        f"dimensions capture {100*source_k32[2]:.2f}% and {100*source_k32[3]:.2f}%. This is far "
        "below the earlier extreme-tail result and shows that tail interventions made propagation "
        "look more low-dimensional than ordinary operation really is.", "",
        "Holding out the source neuron within each semantic family is stricter:", "",
        "|Family, layer 2|Matching 16D span|Other-family span|Per-site wins|",
        "|---|---:|---:|---:|",
    ]
    for name, label in zip(NAMES, SHORT):
        matched, mismatched, wins = lso[name]
        lines.append(f"|{label}|{100*matched:.2f}%|{100*mismatched:.2f}%|{wins:.1%}|")
    lines.extend(["",
        "Only capitalization shows a clear shared implementation: the same-family span captures "
        "about five times as much energy as other-family spans and wins on 97.7% of sites. Boundary "
        "and number neurons agree semantically at their inputs but have neuron-specific downstream "
        "fingerprints. Determiner propagation is likewise indistinguishable from the mismatch control.", "",
        "## Finite causal validation of the subspaces", "",
        f"On 128 held-out sites, the 16D source-specific projection contains "
        f"{100*patches['source_specific']['projection_fraction_mean']:.2f}% of layer-2 propagation "
        f"energy. Replaying it produces a mean output-delta cosine of "
        f"{patches['source_specific']['replay_cosine_mean']:.3f}, compared with "
        f"{patches['mismatched_circuit']['replay_cosine_mean']:.3f} for a mismatched circuit span.", "",
        f"But removing that projected component from the ablated run recovers only "
        f"{100*patches['source_specific']['restoration_recovered_norm_fraction_median']:.1f}% of "
        "logit-change norm at the median, and the mean recovery is negative. The leave-source-out "
        "circuit span has essentially zero restoration benefit. These PCA-like spans describe a "
        "repeatable part of internal motion; they are not yet sufficient causal modules.", "",
        "A likely reason the replay cosine remains moderately positive even for mismatched spans is "
        "that downstream normalization and the output map contain broadly shared sensitive "
        "directions. Directional logit agreement alone is therefore too permissive; restoration "
        "and matched-span controls are essential.", "",
        "## Attention-route check", "",
        "One representative neuron per candidate circuit was separately tested with finite "
        "attention-edge removal. On the transfer panel, QK×OV prediction has "
        f"r={qtransfer['layers']['1']['prediction_pearson']:.3f} and "
        f"{qtransfer['layers']['1']['prediction_sign_agreement']:.1%} sign agreement. It beats a "
        f"uniform random edge on {qtransfer['paired_controls']['uniform_random']['qkov_wins_fraction']:.1%} "
        "of sites. The routing-only comparison is weaker—62.5% wins, with a neuron-cluster interval "
        "that includes zero—so the incremental benefit of OV alignment is not established for these "
        "four representatives alone.", "",
        "## What is still missing", "",
        "1. **Coefficient prediction.** The causal patch uses the recipient's observed finite "
        "propagation delta and projects it. A real reverse-engineered mechanism must predict the "
        "subspace coefficients from the prompt and weights before performing the intervention.",
        "2. **Semantic counterfactuals.** Current-token labels verify discrimination, not that a "
        "capitalization or number change transfers correctly between matched prompts.",
        "3. **Output-role explanations.** Three groups are causally specific, but only boundary/reset "
        "has a readily recognizable average vocabulary effect.",
        "4. **Later-layer routing.** Source identity decays sharply by layer 3; attention and residual "
        "paths beyond the immediate layer remain unresolved.",
        "5. **A fourth isolated circuit.** Determiner syntax is a valid detector family but failed the "
        "cross-ablation isolation criterion. Searching for a replacement after seeing this result "
        "would be exploratory and should use fresh data.", "",
        "## Best next experiment", "",
        "Train a context-conditioned low-rank predictor from the receiving residual state, source "
        "activation change, and QK×OV route features to the layer-2 propagation delta. Freeze it, "
        "then evaluate reconstruction, finite restoration, and semantic patch transfer on new "
        "documents and entirely held-out source neurons. In parallel, predeclare a fresh candidate "
        "for the fourth circuit—such as subword continuation—and evaluate it on untouched corpus "
        "rows rather than replacing the failed determiner result post hoc.", "",
        "## Artifacts", "",
        "- `manifest.json`: all 2,048 ordinary sites and output effects",
        "- `ordinary_propagation.npz`: complete layer-2/3 propagation vectors",
        "- `semantic_panels.npz`: four 102,400-position activation panels",
        "- `analysis.json`: semantic, subspace, finite-patch, and cross-ablation results",
        "- `qkov/`: representative incoming-attention intervention results",
    ])
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "REPORT.md")
    print(figure)


if __name__ == "__main__":
    main()
