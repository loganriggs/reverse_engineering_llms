#!/usr/bin/env python3
"""Calibrate low mask-selection regret against response-surface flatness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_mask_selection_calibration.json")
OUTPUT = TENSOR_ROOT / "layer8_mask_selection_calibration.json"
REPORT = TENSOR_ROOT / "LAYER8_MASK_SELECTION_CALIBRATION.md"


def main():
    config = json.loads(CONFIG.read_text())
    atlas = json.loads(Path(config["atlas"]).read_text())
    grammar = json.loads(Path(config["grammar"]).read_text())
    folds = grammar["config"]["folds"]
    seed = grammar["config"]["seed"]
    results = {}
    for corpus_index, corpus in enumerate(atlas["results"]):
        rng = np.random.default_rng(seed + corpus_index * 1000)
        permutation = rng.permutation(84)
        rows = []
        grammar_lookup = {(row["query"], row["row"]): row
                          for row in grammar["results"][corpus]["records"]}
        for site in atlas["results"][corpus]["records"]:
            values = np.asarray([row["full_fidelity"] for row in site["masks"]])
            random_regrets = []
            for fold in range(folds):
                valid = permutation[np.arange(len(permutation)) % folds == fold]
                random_regrets.append(float(values[valid].max() - values[valid].mean()))
            pairwise = grammar_lookup[(site["query"], site["row"])]["full_fidelity"]["pairwise"]
            rows.append({
                "query": site["query"], "row": site["row"],
                "near_optimal_fraction": float(np.mean(values >= values.max() - config["near_optimal_tolerance"])),
                "random_expected_regret": float(np.median(random_regrets)),
                "pairwise_regret": pairwise["median_regret"],
            })
        summary = {
            "near_optimal_fraction_median": float(np.median([row["near_optimal_fraction"] for row in rows])),
            "near_optimal_fraction_q10": float(np.quantile([row["near_optimal_fraction"] for row in rows], .10)),
            "random_regret_median": float(np.median([row["random_expected_regret"] for row in rows])),
            "random_regret_q90": float(np.quantile([row["random_expected_regret"] for row in rows], .90)),
            "pairwise_regret_median": float(np.median([row["pairwise_regret"] for row in rows])),
            "pairwise_regret_q90": float(np.quantile([row["pairwise_regret"] for row in rows], .90)),
        }
        results[corpus] = {"summary": summary, "records": rows}
    flat = all(results[c]["summary"]["near_optimal_fraction_median"] >= .25 for c in results)
    skill = all(
        results[c]["summary"]["pairwise_regret_median"] <= .5 * results[c]["summary"]["random_regret_median"]
        and results[c]["summary"]["pairwise_regret_q90"] <= .75 * results[c]["summary"]["random_regret_q90"]
        for c in results
    )
    hypotheses = {
        "F0_broad_flat_top": {"supported": bool(flat)},
        "F1_pairwise_has_selection_skill": {"supported": bool(skill)},
        "F2_low_regret_is_only_flatness": {"supported": bool(flat and not skill)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2))
    lines = ["# Calibration of suffix-mask selection regret", "",
             "Pairwise held-out mask selection is compared with the exact expected regret of choosing a held-out mask uniformly at random. Surface flatness is the fraction of all 84 masks within 0.01 fidelity of the sitewise best.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Near-optimal fraction median/q10 | Pairwise regret median/q90 | Random regret median/q90 |",
              "|---|---:|---:|---:|"]
    for corpus, result in results.items():
        row = result["summary"]
        lines.append(f"| {corpus} | {row['near_optimal_fraction_median']:.3f}/{row['near_optimal_fraction_q10']:.3f} | {row['pairwise_regret_median']:.3f}/{row['pairwise_regret_q90']:.3f} | {row['random_regret_median']:.3f}/{row['random_regret_q90']:.3f} |")
    lines += ["", "Low regret only counts as routing information if it materially beats the random-choice baseline. This calibration does not rescue the pairwise model's modest held-out R2."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
