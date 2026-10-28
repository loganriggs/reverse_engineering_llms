#!/usr/bin/env python3
"""Pair-clustered uncertainty audit for prospective semantic routes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


SOURCE = TENSOR_ROOT / "layer8_semantic_route_prospective.json"
OUTPUT = TENSOR_ROOT / "layer8_semantic_route_prospective_uncertainty.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ROUTE_PROSPECTIVE_UNCERTAINTY.md"


def main():
    source = json.loads(SOURCE.read_text())
    rng = np.random.default_rng(20261112)
    result = {}
    advantages = {}
    for corpus, rows in source["results"].items():
        grouped = {}
        for row in rows["records"]:
            grouped.setdefault(row["pair"], []).append(row["advantage"])
        pair_values = {pair: float(np.median(values)) for pair, values in grouped.items()}
        values = np.asarray(list(pair_values.values()))
        draws = np.median(rng.choice(values, size=(20_000, len(values)), replace=True), axis=1)
        result[corpus] = {
            "pair_median_advantage": float(np.median(values)),
            "pair_bootstrap_95": [float(x) for x in np.quantile(draws, [.025, .975])],
            "positive_pairs": int((values > 0).sum()), "pairs": len(values),
        }
        advantages[corpus] = pair_values
    pairs = list(advantages["fineweb"])
    fine = np.asarray([advantages["fineweb"][pair] for pair in pairs])
    tiny = np.asarray([advantages["tinystories"][pair] for pair in pairs])
    result["cross_corpus"] = {
        "pearson": float(np.corrcoef(fine, tiny)[0, 1]),
        "positive_both": int(((fine > 0) & (tiny > 0)).sum()), "pairs": len(pairs),
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    lines = ["# Pair-clustered uncertainty for prospective semantic routes", "",
             "This post-result audit clusters the three contexts for each predeclared new lexical pair. It does not change the locked verdicts.", "",
             "| Corpus | Pair-median advantage (bootstrap 95%) | Positive pairs |", "|---|---:|---:|"]
    for corpus in ("fineweb", "tinystories"):
        row = result[corpus]
        lines.append(f"| {corpus} | {row['pair_median_advantage']:.3f} [{row['pair_bootstrap_95'][0]:.3f}, {row['pair_bootstrap_95'][1]:.3f}] | {row['positive_pairs']}/{row['pairs']} |")
    cross = result["cross_corpus"]
    lines += ["", f"Cross-corpus pair-advantage Pearson: `{cross['pearson']:.3f}`; positive on both: `{cross['positive_both']}/{cross['pairs']}`.", "",
              "A small family modulation remains plausible, especially OOD, but the effect is below the prospective gate and the global route is sufficient. This does not rescue a clean family-specific verb claim."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT)


if __name__ == "__main__":
    main()
