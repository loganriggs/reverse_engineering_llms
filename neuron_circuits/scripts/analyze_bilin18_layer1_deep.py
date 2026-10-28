#!/usr/bin/env python3
"""Summarize the three-iteration Bilin18 layer-1 deep dive."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/bilin18_layer1_deep")


def at_k(rows):
    return {row["k"]: row for row in rows}


def threshold(rows, limit):
    passing = [row["k"] for row in rows if row["ce_delta"] <= limit]
    return min(passing) if passing else None


def main():
    iteration1 = json.loads((ROOT / "iteration1_channels.json").read_text())
    iteration2 = json.loads((ROOT / "iteration2_redteam.json").read_text())
    qkov = json.loads((ROOT / "iteration3_qkov" / "analysis.json").read_text())
    propagation = json.loads((ROOT / "iteration3_propagation.json").read_text())

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    for panel, color in (("fineweb", "C0"), ("wikitext103", "C1"),
                         ("tinystories", "C2")):
        ranked = iteration2["ranked_subset_curves"][panel]["sufficiency"]
        random = iteration2["random_subset_summary"][panel]["sufficiency"]
        ax.plot([x["k"] for x in ranked], [x["ce_delta"] for x in ranked],
                marker="o", color=color, label=f"{panel} ranked")
        if panel == "fineweb":
            ax.plot([x["k"] for x in random], [x["ce_delta_mean"] for x in random],
                    marker="x", linestyle="--", color="0.35", label="FineWeb random mean")
    ax.axhline(.1, color="0.5", linewidth=1, linestyle=":")
    ax.set_yscale("symlog", linthresh=.05)
    ax.set_xlabel("Contextual layer-1 channels retained")
    ax.set_ylabel("Cross-entropy increase (nats)")
    ax.set_title("A. Sparse sufficiency fails; ranking still helps")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for panel, color in (("fineweb", "C0"), ("wikitext103", "C1"),
                         ("tinystories", "C2")):
        conditions = iteration2["context_controls"][panel]["conditions"]
        rows = sorted((float(name.split("_")[1]), value["ce_delta"])
                      for name, value in conditions.items() if name.startswith("scale_"))
        ax.plot([x for x, _ in rows], [y for _, y in rows], marker="o",
                color=color, label=panel)
    ax.axvline(.5, color="0.5", linewidth=1, linestyle=":")
    ax.set_yscale("symlog", linthresh=.05)
    ax.set_xlabel("Fraction of native contextual variation")
    ax.set_ylabel("Cross-entropy increase (nats)")
    ax.set_title("B. Behavior collapses below ~40–50% variation")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    conditions = ["qkov", "route_only", "same_head_random_source",
                  "same_source_random_head", "uniform_random"]
    labels = ["QK×OV", "routing only", "same head", "same source", "uniform"]
    x = np.arange(len(conditions))
    for offset, split, color in ((-.18, "selection", "C0"), (.18, "transfer", "C1")):
        values = [qkov["splits"][split]["layers"]["1"]["median_abs_product_delta"]
                  if condition == "qkov" else
                  np.median([abs(row["product_delta"]) for row in
                             json.loads((ROOT / "iteration3_qkov" / "results.json").read_text())["records"]
                             if row["split"] == split and row["condition"] == condition])
                  for condition in conditions]
        ax.bar(x + offset, values, width=.36, color=color, label=split)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Median |target-neuron product change|")
    ax.set_title("C. Weight-based attention edges beat controls")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    matched = [propagation["source_specific_subspaces"][str(layer)]["matched_projection_mean"]
               for layer in (2, 3)]
    mismatch = [propagation["source_specific_subspaces"][str(layer)]["mismatched_projection_mean"]
                for layer in (2, 3)]
    x = np.arange(2)
    ax.bar(x - .18, np.asarray(matched) * 100, width=.36, label="matching source")
    ax.bar(x + .18, np.asarray(mismatch) * 100, width=.36, label="other sources")
    ax.set_xticks(x, ["Layer 2", "Layer 3"])
    ax.set_ylabel("Held-out propagation energy in 2D span (%)")
    ax.set_title("D. Distributed effects retain source fingerprints")
    ax.legend(fontsize=8)

    figure = ROOT / "three_iteration_summary.png"
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    fineweb_suff = iteration1["curves"]["fineweb"]["sufficiency"]
    ood = iteration1["ood_rank_stability"]
    qtransfer = qkov["splits"]["transfer"]
    layer_stats = {}
    for layer in (2, 3):
        records = [row["layers"][str(layer)] for row in propagation["records"]]
        layer_stats[layer] = {
            "support": float(np.median([row["standardized_delta_effective_support"] for row in records])),
            "weight_r": float(np.median([row["weight_to_finite_spearman"] for row in records])),
            "top64": float(np.median([
                row["selection_transfer_top64_overlap"] for row in propagation["transfer"]
                if row["target_layer"] == layer])),
            "restore": float(np.median([row["logit_l2_recovered_fraction"] for row in records])),
        }

    lines = [
        "# Bilin18 layer-1 deep dive: three iterations", "",
        "## Bottom line", "",
        "Layer 1 is not well described as a small set of individually essential neurons. "
        "Its computation is spread across a broad native basis, but that basis is not "
        "structureless: input-conditioned QK×OV calculations identify causal attention "
        "routes, and individual source neurons leave reproducible distributed fingerprints "
        "in later layers. The right explanatory objects are therefore conditional edges and "
        "small propagation subspaces, not isolated downstream neurons.", "",
        f"![Three-iteration summary]({figure.name})", "",
        "## Iteration 1 — Is layer 1 carried by a sparse neuron set?", "",
        "**Result: the strong sparse version is falsified.** Channels were frozen-ranked on "
        "256 FineWeb examples by `|gradient × (activation − positional mean)|`, then tested "
        "with full downstream recomputation on 256 disjoint examples from each corpus.", "",
        f"- Retaining the best 1,024/4,608 channels still costs +{at_k(fineweb_suff)[1024]['ce_delta']:.2f} "
        "FineWeb CE; 2,048 channels reduce this to "
        f"+{at_k(fineweb_suff)[2048]['ce_delta']:.2f}.",
        f"- Removing the best 2,048 channels also costs only "
        f"+{at_k(iteration1['curves']['fineweb']['necessity'])[2048]['ce_delta']:.2f} CE. "
        "Thus two complementary large subsets can each sustain most behavior.",
        f"- The coarse first panel put 2,048 channels just above 0.1 nat "
        f"(+{at_k(fineweb_suff)[2048]['ce_delta']:.2f}); the independent refined panel put "
        f"the same width at +{at_k(iteration2['ranked_subset_curves']['fineweb']['sufficiency'])[2048]['ce_delta']:.2f}. "
        "Thus roughly half the layer, not a few dozen neurons, is the stable scale for this "
        "engineering threshold.",
        f"- FineWeb importance ranks transfer moderately to WikiText (Spearman "
        f"{ood['wikitext103']['comparison']['spearman']:.2f}) and TinyStories "
        f"({ood['tinystories']['comparison']['spearman']:.2f}).", "",
        "Individual top-channel ablations are often neutral or slightly beneficial for the "
        "sampled next tokens. Absolute local sensitivity is useful for ordering a large set, "
        "but it does not imply that the leading neuron is independently necessary.", "",
        "## Iteration 2 — Is the result only a mean-replacement artifact?", "",
        "**Result: mean replacement exaggerates the failure, but cannot explain it away.**", "",
        "- Ranked subsets consistently outperform random subsets below roughly 2,500 "
        "channels, so the ranking contains real information.",
        "- Either ranking becomes adequate only after retaining a large fraction of the layer; "
        "there is no hidden 32–128-neuron solution in this native basis.",
        f"- Replacing the complete layer by its mean costs "
        f"{iteration2['context_controls']['fineweb']['conditions']['scale_0']['ce_delta']:+.2f} CE. "
        "Swapping a real joint product vector from another example costs "
        f"{iteration2['context_controls']['fineweb']['conditions']['row_shuffle']['ce_delta']:+.2f}, "
        "and assigning real vectors to the wrong positions costs "
        f"{iteration2['context_controls']['fineweb']['conditions']['position_shuffle']['ce_delta']:+.2f}.",
        "- Attenuating `(activation − mean)` shows a nonlinear operating regime: half-strength "
        "is mostly tolerated, while one-quarter strength is not. Layer 1 depends on aggregate "
        "contextual variation and downstream normalization, not merely a list of symbolic bits.", "",
        "The shuffled states are marginally real but conditionally mismatched to their receiving "
        "residual streams, so they are a stronger control—not a perfect on-manifold intervention.", "",
        "## Iteration 3 — Can weight structure recover causal routes and reusable downstream form?", "",
        "### Incoming attention", "",
        "For eight high-sensitivity neurons, three positive and three negative tail sites were "
        "tested on both discovery and transfer panels. Removing the QK×OV-predicted attention "
        "edge gave, on transfer:", "",
        f"- prediction/finite-effect correlation {qtransfer['layers']['1']['prediction_pearson']:.2f};",
        f"- sign agreement {qtransfer['layers']['1']['prediction_sign_agreement']:.1%};",
        f"- the predicted edge beat the routing-only edge on "
        f"{qtransfer['paired_controls']['route_only']['qkov_wins_fraction']:.1%} of sites;",
        f"- it beat a uniform random edge on "
        f"{qtransfer['paired_controls']['uniform_random']['qkov_wins_fraction']:.1%} of sites.", "",
        "This validates a conditional weight-based explanation: the relevant edge changes with "
        "the input, but the same QK×OV formula predicts it on held-out examples. Only about 27% "
        "of the edge's additive logit effect is mediated by the selected neuron, because an "
        "attention edge writes a full residual vector read by many channels.", "",
        "### Propagation into layers 2–3", "",
        f"A one-channel layer-1 intervention affects about {layer_stats[2]['support']:.0f} "
        f"layer-2 and {layer_stats[3]['support']:.0f} layer-3 channels by effective support. "
        "The direct weight contraction predicts the immediate layer-2 pattern moderately "
        f"(median Spearman {layer_stats[2]['weight_r']:.2f}) but degrades by layer 3 "
        f"({layer_stats[3]['weight_r']:.2f}).", "",
        f"Exact downstream neuron identities are only weakly stable: median discovery/transfer "
        f"top-64 overlap is {layer_stats[2]['top64']:.1%} in layer 2 and "
        f"{layer_stats[3]['top64']:.1%} in layer 3. Restoring the top 32 affected channels "
        f"recovers only {layer_stats[2]['restore']:.1%} and {layer_stats[3]['restore']:.1%} "
        "of logit-change norm at the median.", "",
        "However, the distributed pattern retains source identity. A two-dimensional span fit "
        "from each source neuron's two discovery interventions captures, on average:", "",
        f"- {100*propagation['source_specific_subspaces']['2']['matched_projection_mean']:.1f}% "
        "of its own held-out layer-2 propagation energy versus "
        f"{100*propagation['source_specific_subspaces']['2']['mismatched_projection_mean']:.2f}% "
        "for other sources;",
        f"- {100*propagation['source_specific_subspaces']['3']['matched_projection_mean']:.1f}% "
        "in layer 3 versus "
        f"{100*propagation['source_specific_subspaces']['3']['mismatched_projection_mean']:.2f}% "
        "for other sources.", "",
        "The matching source wins for all eight tested neurons in both layers. This is preliminary "
        "because each span has only two discovery and two transfer sites, but it is the clearest "
        "positive result: a neuron can act like a reusable verb while its downstream realization "
        "is a broad, context-varying population fingerprint.", "",
        "## Revised model of the regime", "",
        "1. **Native neuron level:** some activation conditions are semantically legible, but "
        "single neurons are rarely necessary.",
        "2. **Conditional edge level:** QK×OV weight structure predicts which attention message "
        "drives a chosen discriminator on a given input.",
        "3. **Population level:** the resulting effect fans out through thousands of channels, "
        "while a small source-specific subspace remains detectable across examples.",
        "4. **Output level:** many of these changes cancel or are normalized away, explaining why "
        "large internal motion can coexist with a small CE effect.", "",
        "This is neither a clean one-concept-per-neuron model nor an uninterpretable random code. "
        "It is closer to a distributed implementation of reusable operations whose identity is "
        "more stable as a conditional direction/subspace than as a fixed list of neurons.", "",
        "## Next highest-value experiment", "",
        "Fit propagation subspaces from hundreds of ordinary—not only extreme-tail—sites, hold "
        "out both documents and source neurons, and test whether those subspaces support finite "
        "patch transfer. Compare native coordinates with learned sparse/rotated dictionaries. "
        "That will distinguish a genuinely reusable low-dimensional operation from an artifact "
        "of repeatedly intervening on the same source neuron.", "",
        "## Files", "",
        "- `iteration1_channels.json`: rankings and finite necessity/sufficiency curves",
        "- `iteration2_redteam.json`: random-subset, scaling, and shuffled-state controls",
        "- `iteration3_qkov/`: finite incoming-edge tests and paired-control analysis",
        "- `iteration3_propagation.json`: layer-2/3 propagation, mediation, and subspaces",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(ROOT / "REPORT.md")
    print(figure)


if __name__ == "__main__":
    main()
