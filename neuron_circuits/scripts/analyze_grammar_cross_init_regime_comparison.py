#!/usr/bin/env python3
"""Compare pure and mixed cross-initialization recurrence with chance correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("artifacts/grammar_regimes")
OUTPUT = ROOT / "grammar_cross_init_regime_comparison.json"
REPORT = ROOT / "GRAMMAR_CROSS_INIT_REGIME_COMPARISON.md"


def load(name):
    return json.loads((ROOT / name).read_text())


def passing(payload):
    return [row for row in payload["results"] if row["gate_passed"]]


def summarize(name, native, linear):
    native_rows = passing(native); linear_rows = passing(linear)
    discovery = native["common_sites"] // 2
    rank = 64; ambient = discovery * 8; chance = rank / ambient
    overlaps = [row["subspace_overlap"][str(rank)] for row in native_rows]
    adjusted = [(value - chance) / (1 - chance) for value in overlaps]
    return {
        "regime": name,
        "gate_passing_candidates": len(native_rows),
        "discovery_sites": discovery,
        "top64_random_subspace_expectation": chance,
        "top64_overlap_mean": float(np.mean(overlaps)),
        "top64_chance_adjusted_mean": float(np.mean(adjusted)),
        "complete_effect_cosine_mean": float(np.mean([row["complete_effect_median_cosine"] for row in native_rows])),
        "matched_channel_evaluation_cosine_mean": float(np.mean([row["matched_channel_evaluation_median_cosine"] for row in native_rows])),
        "mapped_site_top32_overlap_mean": float(np.mean([row["mapped_site_top32_overlap"] for row in native_rows])),
        "same_template_residual_r2_mean_by_boundary": [float(np.mean([row["boundaries"][i]["same_template_ridge_r2"] for row in linear_rows])) for i in range(5)],
        "heldout_template_residual_r2_mean_by_boundary": [float(np.mean([row["boundaries"][i]["ridge_r2"] for row in linear_rows])) for i in range(5)],
        "heldout_template_gate_r2_mean_by_layer": [float(np.mean([row["gate_layers"][i]["ridge_r2"] for row in linear_rows])) for i in range(4)],
        "causal_jacobian_cosine_mean_by_boundary": [float(np.mean([row["boundaries"][i]["causal_jacobian_median_cosine"] for row in linear_rows])) for i in range(5)],
    }


def main():
    pure = summarize("pure rule", load("grammar_pure_gate_cross_init.json"),
                     load("grammar_pure_cross_init_linear_conjugacy.json"))
    mixed = summarize("50% arbitrary", load("grammar_mixed_gate_cross_init.json"),
                      load("grammar_cross_init_linear_conjugacy.json"))
    OUTPUT.write_text(json.dumps({"analysis_status": "posthoc-regime-comparison-of-locked-assays",
                                  "pure": pure, "mixed": mixed}, indent=2, allow_nan=False))
    lines = ["# Cross-initialization recurrence: pure rules versus mixed memory", "",
             "This is a post-hoc comparison of separately locked assays, not a new preregistered hypothesis test. Only candidates passing 0.97 trained-template recall and 0.95 unseen-template transfer are included (two candidate initializations per regime, each compared with the same reference).", "",
             "| Quantity | Pure rule | 50% arbitrary |", "|---|---:|---:|",
             f"| Top-64 effect-subspace overlap | {pure['top64_overlap_mean']:.3f} | {mixed['top64_overlap_mean']:.3f} |",
             f"| Random-subspace expectation | {pure['top64_random_subspace_expectation']:.3f} | {mixed['top64_random_subspace_expectation']:.3f} |",
             f"| Chance-adjusted top-64 overlap | {pure['top64_chance_adjusted_mean']:.3f} | {mixed['top64_chance_adjusted_mean']:.3f} |",
             f"| Complete natural-effect cosine | {pure['complete_effect_cosine_mean']:.3f} | {mixed['complete_effect_cosine_mean']:.3f} |",
             f"| Matched native-channel held-out cosine | {pure['matched_channel_evaluation_cosine_mean']:.3f} | {mixed['matched_channel_evaluation_cosine_mean']:.3f} |",
             f"| Mapped per-site top-32 overlap | {pure['mapped_site_top32_overlap_mean']:.3f} | {mixed['mapped_site_top32_overlap_mean']:.3f} |",
             f"| Late gate-map R2 (Layer 3) | {pure['heldout_template_gate_r2_mean_by_layer'][3]:.3f} | {mixed['heldout_template_gate_r2_mean_by_layer'][3]:.3f} |",
             f"| Late full-Jacobian cosine (boundary 4) | {pure['causal_jacobian_cosine_mean_by_boundary'][4]:.3f} | {mixed['causal_jacobian_cosine_mean_by_boundary'][4]:.3f} |", "",
             "## Interpretation", "",
             "Pure reuse produces more cross-run recurrence, especially in late distributed gate states, but not canonical one-neuron correspondences. Mixed memory largely destroys even that transferable gate coordinate map. Yet neither regime admits a global linear conjugacy of its complete local answer Jacobian. This is compatible with the earlier positive result that selected pure-rule semantic edit vectors transfer causally after alignment: the recurring object is a task-conditioned finite program, not every possible local perturbation.", "",
             "## Red-team boundary", "",
             "The pure and mixed discovery panels have different sizes because mixed panels balance selected and unselected facts. The subspace comparison therefore reports the analytic random-subspace expectation and a chance-adjusted value. There are only two gate-passing candidate initializations per regime, and checkpoint training durations differ. Treat the magnitude as exploratory; the sign agrees with the independently established within-model semantic-transfer gap. Negative full-Jacobian alignment does not show that the models compute different answers or that no nonlinear/contextual alignment exists."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT)


if __name__ == "__main__":
    main()
