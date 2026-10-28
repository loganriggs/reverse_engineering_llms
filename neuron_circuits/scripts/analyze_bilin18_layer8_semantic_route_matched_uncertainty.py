#!/usr/bin/env python3
"""Post-result paired uncertainty audit for the matched semantic routes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


SOURCE = TENSOR_ROOT / "layer8_semantic_route_matched_redteam.json"
OUTPUT = TENSOR_ROOT / "layer8_semantic_route_matched_uncertainty.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_ROUTE_MATCHED_UNCERTAINTY.md"


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def main():
    source = json.loads(SOURCE.read_text())
    rng = np.random.default_rng(20261110)
    results = {}
    by_corpus = {}
    for corpus, result in source["results"].items():
        advantages = np.asarray([row["advantage"] for row in result["records"]])
        draws = rng.choice(advantages, size=(20_000, len(advantages)), replace=True)
        medians = np.median(draws, axis=1)
        wins = int((advantages > 0).sum())
        sign_p = sum(math.comb(len(advantages), k)
                     for k in range(wins, len(advantages) + 1)) / 2 ** len(advantages)
        results[corpus] = {
            "count": len(advantages), "median": float(np.median(advantages)),
            "bootstrap_median_95": [float(x) for x in np.quantile(medians, [.025, .975])],
            "positive_pairs": wins, "one_sided_sign_p": sign_p,
        }
        by_corpus[corpus] = {row["pair"]: row["advantage"] for row in result["records"]}
    pairs = list(by_corpus["fineweb"])
    fine = np.asarray([by_corpus["fineweb"][pair] for pair in pairs])
    tiny = np.asarray([by_corpus["tinystories"][pair] for pair in pairs])
    results["cross_corpus"] = {
        "positive_on_both": int(((fine > 0) & (tiny > 0)).sum()),
        "pairs": len(pairs),
        "advantage_pearson": float(np.corrcoef(fine, tiny)[0, 1]),
        "advantage_spearman": float(np.corrcoef(rankdata(fine), rankdata(tiny))[0, 1]),
    }
    OUTPUT.write_text(json.dumps(results, indent=2))
    cross = results["cross_corpus"]
    lines = ["# Uncertainty audit for data-matched semantic routes", "",
             "This is a post-result uncertainty analysis; it does not change the locked verdicts or thresholds.", "",
             "| Corpus | Median advantage (bootstrap 95%) | Positive pairs | One-sided sign p |",
             "|---|---:|---:|---:|"]
    for corpus in ("fineweb", "tinystories"):
        row = results[corpus]
        lines.append(f"| {corpus} | {row['median']:.3f} [{row['bootstrap_median_95'][0]:.3f}, {row['bootstrap_median_95'][1]:.3f}] | {row['positive_pairs']}/{row['count']} | {row['one_sided_sign_p']:.4f} |")
    lines += ["", f"Positive on both corpora: `{cross['positive_on_both']}/{cross['pairs']}`; cross-corpus Pearson/Spearman of pair advantages: `{cross['advantage_pearson']:.3f}/{cross['advantage_spearman']:.3f}`.", "",
              "TinyStories supplies evidence for a small positive family advantage, while FineWeb remains compatible with zero and pairwise advantage magnitudes do not replicate strongly. The correct label is weak/exploratory family modulation, not a validated semantic verb."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT)


if __name__ == "__main__":
    main()
