#!/usr/bin/env python3
"""Test whether pooled unrelated-edit H6 discovery survives pair disaggregation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import select_records
from run_bilin18_boundary_style_H2_weight_outlier import spectral_atlas
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_terminal_edit_discovery_generalization import collect, rank_of


CONFIG = Path("configs/bilin18_terminal_edit_pair_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_terminal_edit_pair_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_TERMINAL_EDIT_PAIR_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_terminal_edit_pair_redteam.png"


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    pairs = token_pairs(tokenizer, parent["control_pairs"])

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    atlas, bases = spectral_atlas(model)
    spectral_scores = {row["head"]: row["score"] for row in atlas if row["layer"] == 17}

    results = []
    for pair_index, ((first, second), pair) in enumerate(zip(parent["control_pairs"], pairs)):
        name = f"{first.strip()}↔{second.strip()}"
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(
                raw, boundary_flags, [pair], counts, 127,
                config["examples_per_label_per_split"],
                config["seed"] + 10 * pair_index + split,
            )
            print("terminal pair", pair_index, split, name, len(records), flush=True)
            results.append({
                "pair_index": pair_index,
                "pair": name,
                "split": split,
                **collect(model, raw, records, bases, spectral_scores, [], config),
            })

    lookup = {(row["pair_index"], row["split"]): row for row in results}
    pairs_summary = []
    for pair_index, pair in enumerate(parent["control_pairs"]):
        discovery = lookup[(pair_index, 0)]
        evaluation = lookup[(pair_index, 1)]
        context_head = max(discovery["heads"], key=lambda row: row["rank1_write_rms"])["head"]
        quotient_head = max(discovery["heads"], key=lambda row: row["direct_quotient_mean"])["head"]
        h6 = next(row for row in evaluation["heads"] if row["head"] == 6)
        pairs_summary.append({
            "pair_index": pair_index,
            "pair": discovery["pair"],
            "discovery_examples": discovery["examples"],
            "evaluation_examples": evaluation["examples"],
            "endpoint_energy_discovery": discovery["endpoint_fisher_energy_mean"],
            "endpoint_energy_evaluation": evaluation["endpoint_fisher_energy_mean"],
            "context_head": context_head,
            "context_heldout_rank": rank_of(evaluation["heads"], "finite_rank1_mean", context_head),
            "quotient_head": quotient_head,
            "quotient_heldout_rank": rank_of(evaluation["heads"], "finite_rank1_mean", quotient_head),
            "h6_heldout_rank": rank_of(evaluation["heads"], "finite_rank1_mean", 6),
            "h6_heldout_effect": h6["finite_rank1_mean"],
            "heldout_winner": evaluation["strongest_finite_head"],
            "direct_finite_correlation": evaluation["direct_finite_correlation"],
            "leave_winner_out_correlation": evaluation["leave_winner_out_correlation"],
        })

    threshold = config["minimum_pairs_for_three_quarters_gate"]
    p0 = all(
        row["label_counts"][0] == row["label_counts"][1]
        and row["examples"] >= config["minimum_examples_per_cell"]
        for row in results
    )
    h6_top3 = sum(row["h6_heldout_rank"] <= 3 for row in pairs_summary)
    h6_positive = sum(row["h6_heldout_effect"] > 0 for row in pairs_summary)
    h6_context = sum(row["context_head"] == 6 for row in pairs_summary)
    context_transfer = sum(row["context_heldout_rank"] <= 3 for row in pairs_summary)
    quotient_transfer = sum(row["quotient_heldout_rank"] == 1 for row in pairs_summary)
    p1 = h6_top3 >= threshold
    p2 = h6_positive >= threshold
    p3 = h6_context >= threshold
    p4 = context_transfer >= threshold
    p5 = quotient_transfer >= threshold
    hypotheses = {
        "P0_pair_panels_are_balanced_and_powered": {"supported": p0},
        "P1_pooled_H6_recurs_across_pairs": {"supported": p1, "pairs": h6_top3},
        "P2_H6_effect_is_directionally_consistent": {"supported": p2, "pairs": h6_positive},
        "P3_H6_is_the_common_context_only_selection": {"supported": p3, "pairs": h6_context},
        "P4_pair_specific_context_selection_transfers": {"supported": p4, "pairs": context_transfer},
        "P5_pair_specific_quotient_selection_transfers": {"supported": p5, "pairs": quotient_transfer},
        "P6_pooled_H6_is_not_an_averaging_artifact": {"supported": all((p0, p1, p2, p3))},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "pairs": pairs_summary, "results": results}, indent=2))

    lines = [
        "# Pairwise red team of the pooled unrelated-edit H6 result",
        "",
        "The sixteen substitutions used in the pooled unrelated panel are now analyzed separately. Each split-0 head choice is frozen before evaluation on split 1. The previously pooled H6 choice is also evaluated without reselection on every pair.",
        "",
        "## Locked results",
        "",
    ]
    for name, row in hypotheses.items():
        suffix = f" ({row['pairs']}/16 pairs)" if "pairs" in row else ""
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**{suffix}")
    lines += ["", "| Pair | N d/e | context H/rank | quotient H/rank | H6 rank/effect | held-out winner |", "|---|---:|---:|---:|---:|---:|"]
    for row in pairs_summary:
        lines.append(
            f"| {row['pair']} | {row['discovery_examples']}/{row['evaluation_examples']} | H{row['context_head']}/{row['context_heldout_rank']} | H{row['quotient_head']}/{row['quotient_heldout_rank']} | {row['h6_heldout_rank']}/{row['h6_heldout_effect']:+.3g} | H{row['heldout_winner']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Pairwise stability would support H6 as a reusable terminal operation. Failure would show that pooling creates a useful average predictor but erases conditional mechanism diversity. Context-only selection tests whether occupied tensor magnitude is enough; quotient selection tests whether the model's changed output distribution is required. Neither assigns a human semantic name to a head.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    x = np.arange(len(pairs_summary))
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    axes[0].bar(x, [row["h6_heldout_rank"] for row in pairs_summary]); axes[0].invert_yaxis(); axes[0].set(ylabel="Held-out H6 rank", title="Does pooled H6 recur per pair?")
    axes[1].bar(x - .18, [row["context_heldout_rank"] for row in pairs_summary], .36, label="context selection")
    axes[1].bar(x + .18, [row["quotient_heldout_rank"] for row in pairs_summary], .36, label="quotient selection")
    axes[1].invert_yaxis(); axes[1].set(xticks=x, xticklabels=[row["pair"] for row in pairs_summary], ylabel="Held-out rank", title="Prospective pair-specific selection transfer"); axes[1].tick_params(axis="x", rotation=45); axes[1].legend()
    for axis in axes: axis.grid(alpha=.2)
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
