#!/usr/bin/env python3
"""Fresh-split red team of signed cancellation in final-message source edges."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_final_destination_source_program import evaluate_condition
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_source_cancellation_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_source_cancellation_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_SOURCE_CANCELLATION_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_source_cancellation_redteam.png"


def main() -> None:
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    evaluation = {}
    for corpus_index, (corpus, spec) in enumerate(config["evaluation"].items()):
        rows = load_rows(spec); evaluation[corpus] = {}
        for position in config["edit_positions"]:
            print("evaluating", corpus, position, flush=True); evaluation[corpus][str(position)] = evaluate_condition(model, tokenizer, rows, position, config, config["seed"] + 1000 * corpus_index + position)
    def vals(method, budget, metric="fisher_fidelity"):
        return [evaluation[c][str(p)]["metrics"][str(layer)][f"{method}/{budget}"][metric] for c in evaluation for p in config["edit_positions"] for layer in config["layers"]]
    diagnostics = [(corpus, row) for corpus in evaluation for position in config["edit_positions"] for row in evaluation[corpus][str(position)]["diagnostics"].values()]
    cancellation_by_corpus = {}
    full_projection_by_corpus = {}
    for corpus in evaluation:
        cancellations = []; projections = []
        for label, row in diagnostics:
            if label != corpus: continue
            for positive, negative, net, target in zip(row["positive_score"], row["negative_score"], row["net_score"], row["target_energy"]):
                cancellations.append((positive + abs(negative)) / max(abs(net), 1e-30)); projections.append(net / max(target, 1e-30))
        cancellation_by_corpus[corpus] = float(np.median(cancellations)); full_projection_by_corpus[corpus] = float(np.median(projections))
    all_participation = [value for _, row in diagnostics for value in row["participation_values"]]
    edited = [evaluation[c][str(p)]["metrics"][str(layer)]["edited_only"]["fisher_fidelity"] for c in evaluation for p in config["edit_positions"] for layer in config["layers"]]
    global_counts = np.sum([np.asarray(row["absolute_top16_counts"]) for _, row in diagnostics], axis=0); skeleton_fraction = float(np.sort(global_counts)[-16:].sum() / max(global_counts.sum(), 1))
    absolute16 = vals("absolute_fisher", 16); signed16 = vals("signed_fisher", 16); balanced16 = vals("balanced_fisher", 16); state16 = vals("state_norm", 16)
    hypotheses = {
        "C1_full_local_score_predicts_the_finite_target": {"supported": abs(float(np.median(list(full_projection_by_corpus.values()))) - 1) <= .15, "corpus_medians": full_projection_by_corpus},
        "C2_positive_and_negative_source_evidence_both_matter": {"supported": float(np.median(list(cancellation_by_corpus.values()))) >= 1.5, "corpus_medians": cancellation_by_corpus},
        "C3_two_sided_selection_beats_one_sided_selection": {"supported": np.mean(absolute16) - np.mean(signed16) >= .20, "mean_advantage": float(np.mean(absolute16) - np.mean(signed16))},
        "C4_sixtyfour_two_sided_sources_are_robust": {"supported": min(vals("absolute_fisher", 64)) >= .75, "minimum": float(min(vals("absolute_fisher", 64)))},
        "C5_state_magnitude_is_a_competitive_proxy": {"supported": abs(float(np.mean(state16) - np.mean(absolute16))) <= .05, "mean_difference": float(np.mean(state16) - np.mean(absolute16))},
        "C6_participation_understates_required_support": {"supported": np.median(all_participation) <= 16 and min(absolute16) < .50, "participation_median": float(np.median(all_participation)), "top16_minimum": float(min(absolute16))},
        "C7_balancing_signs_is_not_enough": {"supported": np.mean(absolute16) - np.mean(balanced16) >= .05, "mean_gap": float(np.mean(absolute16) - np.mean(balanced16))},
        "C8_cancellation_strength_transfers_across_corpora": {"supported": all(abs(cancellation_by_corpus[c] - cancellation_by_corpus["fineweb"]) <= .30 for c in ("wikitext", "tinystories"))},
        "C9_the_edit_edge_is_not_the_message": {"supported": np.mean(edited) < .10, "mean_fidelity": float(np.mean(edited))},
        "C10_no_fixed_sixteen_position_skeleton_dominates": {"supported": skeleton_fraction < .50, "fraction": skeleton_fraction, "top_positions": np.argsort(global_counts)[-16:][::-1].tolist()},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "evaluation": evaluation, "global_absolute_top16_counts": global_counts.tolist()}, indent=2, allow_nan=False))
    budgets = config["budgets"]; fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for method in ("signed_fisher", "absolute_fisher", "balanced_fisher", "state_norm"):
        axes[0].plot(budgets, [np.mean(vals(method, budget)) for budget in budgets], marker="o", label=method); axes[1].plot(budgets, [min(vals(method, budget)) for budget in budgets], marker="o", label=method)
    axes[0].set_xscale("log", base=2); axes[1].set_xscale("log", base=2); axes[0].set_title("Fresh-split mean fidelity"); axes[1].set_title("Fresh-split minimum fidelity")
    names = list(cancellation_by_corpus); x = np.arange(len(names)); axes[2].bar(x - .18, [cancellation_by_corpus[c] for c in names], .36, label="cancellation ratio"); axes[2].bar(x + .18, [full_projection_by_corpus[c] for c in names], .36, label="full local projection"); axes[2].set_xticks(x, names); axes[2].set_title("Signed-score diagnostics")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=7)
    axes[0].set_xlabel("source budget"); axes[1].set_xlabel("source budget"); axes[0].set_ylabel("Fisher fidelity"); axes[1].set_ylabel("Fisher fidelity")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    lines = ["# Fresh-split source-cancellation red team", "", "The first source-edge audit found a small absolute-score participation ratio but poor one-sided subset fidelity. This preregistered replication uses fresh examples, adds an explicitly sign-balanced ranking, measures positive/negative local Fisher score cancellation, and corrects the recomposition-logit diagnostic.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Cancellation red team](bilin12_source_cancellation_redteam.png)", "", "## Fresh-split frontier", "", "| Budget | Signed mean/min | Absolute mean/min | Balanced mean/min | State mean/min |", "|---:|---:|---:|---:|---:|"]
    for budget in budgets: lines.append(f"| {budget} | {np.mean(vals('signed_fisher', budget)):.3f}/{min(vals('signed_fisher', budget)):.3f} | {np.mean(vals('absolute_fisher', budget)):.3f}/{min(vals('absolute_fisher', budget)):.3f} | {np.mean(vals('balanced_fisher', budget)):.3f}/{min(vals('balanced_fisher', budget)):.3f} | {np.mean(vals('state_norm', budget)):.3f}/{min(vals('state_norm', budget)):.3f} |")
    lines += ["", "| Corpus | Median cancellation ratio | Median full local projection |", "|---|---:|---:|"]
    for corpus in names: lines.append(f"| {corpus} | {cancellation_by_corpus[corpus]:.3f} | {full_projection_by_corpus[corpus]:.3f} |")
    lines += ["", f"Median participation is **{np.median(all_participation):.1f}**, yet top-16 absolute selection has a worst-condition fidelity of **{min(absolute16):.3f}**. The recurrent absolute-position top-16 accounts for **{skeleton_fraction:.3f}** of selections. These quantities answer different questions: participation measures concentration of signed local scores; finite subset fidelity measures whether omitted positive and negative edges still cancel correctly."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
