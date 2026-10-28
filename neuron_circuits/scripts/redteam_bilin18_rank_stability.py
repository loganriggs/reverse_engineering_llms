#!/usr/bin/env python3
"""Red-team whether downstream-ranked inherited CP terms are stable objects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evaluate_bilin18_causal_cp import load_cp
from explore_bilin18_downstream_rank_selection import downstream_ranking
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT


CONFIG = Path("configs/bilin18_rank_stability_redteam.json")
BASE = ROOT / "downstream_rank_selection_results.json"
OUTPUT = ROOT / "rank_stability_redteam.json"
REPORT = ROOT / "RANK_STABILITY_REDTEAM.md"
OOD = Path("artifacts/bilin18_ood_corpora")


def rank_values(scores):
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks.astype(np.float64)


def compare(base_scores, other_scores, counts):
    base_order = np.argsort(base_scores)[::-1]
    other_order = np.argsort(other_scores)[::-1]
    rb, ro = rank_values(base_scores), rank_values(other_scores)
    spearman = float(np.corrcoef(rb, ro)[0, 1])
    cosine = float(np.dot(base_scores, other_scores) /
                   max(np.linalg.norm(base_scores) * np.linalg.norm(other_scores), 1e-30))
    jaccard = {}
    for count in counts:
        a, b = set(base_order[:count]), set(other_order[:count])
        jaccard[str(count)] = len(a & b) / len(a | b)
    return {"score_spearman": spearman, "score_cosine": cosine,
            "top_jaccard": jaccard}


def main():
    config = json.loads(CONFIG.read_text())
    base_payload = json.loads(BASE.read_text())
    base_scores = np.asarray(base_payload["scores"])
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.requires_grad_(False)
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    direct = load_cp(ROOT / "cp_r2048.pt", "cuda")
    sources = {
        "fineweb": np.load(DEFAULT_CORPUS, mmap_mode="r"),
        "wikitext103": np.load(OOD / "wikitext103_1200x129.npy", mmap_mode="r"),
        "tinystories": np.load(OOD / "tinystories_1200x129.npy", mmap_mode="r"),
    }
    ranking_config = {
        "blend_lambda": 0.2,
        "seed": config["seed"],
        "selection_interventions": {
            "gaussian_rms_0.10": {"kind": "gaussian", "scale": 0.10},
            "natural_mix_0.25": {"kind": "natural_mix", "scale": 0.25},
        },
    }
    comparisons = {}
    for name, spec in config["splits"].items():
        ranking_config["selection_rows"] = {
            "start": spec["start"], "rows": config["rows_per_split"]}
        _, scores, _, _ = downstream_ranking(
            model, language, direct, sources[spec["corpus"]], ranking_config)
        comparisons[name] = compare(base_scores, scores.numpy(), config["top_counts"])
    within = comparisons["fineweb_b"]
    h1 = within["score_spearman"] >= .5 and within["top_jaccard"]["512"] >= .5
    h2 = all(row["score_spearman"] >= .5 and row["top_jaccard"]["512"] >= .5
             for name, row in comparisons.items() if name != "fineweb_b")
    result = {"config": config, "comparisons_to_fineweb_a": comparisons,
              "hypotheses": {
                  "H1_within_corpus_stability": {"supported": h1},
                  "H2_cross_corpus_stability": {"supported": h2},
              }}
    OUTPUT.write_text(json.dumps(result, indent=2))
    lines = ["# Downstream rank stability red-team", "", "## Hypotheses", "",
             f"- **H1 within-corpus stability: {'supported' if h1 else 'not supported'}.**",
             f"- **H2 cross-corpus stability: {'supported' if h2 else 'not supported'}.**", "",
             "| Comparison to FineWeb A | Score Spearman | Score cosine | Jaccard@128 | Jaccard@512 | Jaccard@1024 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in comparisons.items():
        lines.append(f"| {name} | {row['score_spearman']:.3f} | {row['score_cosine']:.3f} "
                     f"| {row['top_jaccard']['128']:.3f} | {row['top_jaccard']['512']:.3f} "
                     f"| {row['top_jaccard']['1024']:.3f} |")
    lines += ["", "A stable aggregate rank curve does not imply stable individual terms. These comparisons hold the inherited CP basis fixed and vary only the examples used to estimate downstream sensitivity.", "",
              "Raw results: `rank_stability_redteam.json`."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
