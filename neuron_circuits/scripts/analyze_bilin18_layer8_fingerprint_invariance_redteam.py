#!/usr/bin/env python3
"""Red-team corpus-scale confounding in the downstream fingerprint router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_bilin18_layer8_suffix_mask_grammar import design, ridge_fit
from run_bilin18_layer8_downstream_fingerprint_router import evaluate, routed_rows
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_fingerprint_invariance_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_fingerprint_invariance_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_FINGERPRINT_INVARIANCE_REDTEAM.md"


def transform(values, kind):
    values = np.asarray(values, dtype=np.float64).copy()
    norms, cosines = values[:, :9], values[:, 9:]
    if kind == "relative_norm":
        norms = (norms - norms.mean(1, keepdims=True)) / np.maximum(norms.std(1, keepdims=True), 1e-12)
    elif kind == "rank_norm":
        order = np.argsort(np.argsort(norms, axis=1), axis=1)
        norms = 2 * order / 8 - 1
    elif kind == "cosine_only":
        norms = np.zeros_like(norms)
    else:
        raise ValueError(kind)
    return np.concatenate([norms, cosines], axis=1)


def fit_variant(name, fine, wiki, mask_features, targets, fine_sites, wiki_sites, config):
    grid = []
    for alpha in config["ridge_alphas"]:
        records = []
        for fold in range(config["query_folds"]):
            valid = np.asarray([i for i, site in enumerate(fine_sites)
                                if site["query"] % config["query_folds"] == fold])
            train = np.setdiff1d(np.arange(len(fine_sites)), valid)
            router = ridge_fit(routed_rows(mask_features, fine[train]), targets[train].reshape(-1), alpha)
            records.extend(evaluate(router, mask_features, fine[valid], [fine_sites[i] for i in valid]))
        grid.append({"alpha": alpha,
                     "summary": summarize([row["full_fidelity"] for row in records], config["seed"]),
                     "records": records})
    chosen = max(grid, key=lambda row: (row["summary"]["q10"], row["summary"]["median"], row["alpha"]))
    router = ridge_fit(routed_rows(mask_features, fine), targets.reshape(-1), chosen["alpha"])
    wiki_records = evaluate(router, mask_features, wiki, wiki_sites)
    return {"name": name, "grid": grid, "alpha": chosen["alpha"],
            "summary": {"fineweb_cv": chosen["summary"],
                        "wikitext": summarize([row["full_fidelity"] for row in wiki_records], config["seed"])},
            "wikitext_records": wiki_records}


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent"]).read_text())
    atlas = json.loads(Path(config["atlas"]).read_text())
    raw = {key: np.asarray(value) for key, value in parent["fingerprints"].items()}
    fine_sites = atlas["results"]["fineweb"]["records"]
    wiki_sites = atlas["results"]["wikitext"]["records"]
    masks = [tuple(row["mask"]) for row in fine_sites[0]["masks"]]
    mask_features = design(masks, "pairwise")
    targets = np.asarray([[row["full_fidelity"] for row in site["masks"]] for site in fine_sites])
    results = {}
    for kind in ("relative_norm", "rank_norm", "cosine_only"):
        results[kind] = fit_variant(
            kind, transform(raw["fineweb"], kind), transform(raw["wikitext"], kind),
            mask_features, targets, fine_sites, wiki_sites, config,
        )
    passes = {kind: all(result["summary"][split]["median"] >= .95
                        and result["summary"][split]["q10"] >= .80
                        for split in ("fineweb_cv", "wikitext"))
              for kind, result in results.items()}
    raw_q10 = parent["final"]["wikitext"]["q10"]
    scale = any(passes[kind] and results[kind]["summary"]["wikitext"]["q10"] - raw_q10 >= .50
                for kind in results)
    hypotheses = {
        "I0_relative_encoding_validates": {"supported": bool(any(passes.values()))},
        "I1_absolute_scale_caused_ood_failure": {"supported": bool(scale)},
        "I2_singleton_pair_fingerprint_still_insufficient": {"supported": bool(not any(passes.values()))},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "raw_wikitext_q10": raw_q10, "results": results}, indent=2))
    lines = ["# Invariance red team of the downstream fingerprint", "",
             "Absolute singleton-response norms are replaced by within-site z-scores, ranks, or zeros (cosine-only). Causal labels, folds, masks, and evaluation sites are unchanged.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Encoding | FineWeb median/q10 | WikiText median/q10 | WikiText minimum |",
              "|---|---:|---:|---:|"]
    for kind, result in results.items():
        fine = result["summary"]["fineweb_cv"]; wiki = result["summary"]["wikitext"]
        lines.append(f"| {kind} | {fine['median']:.3f}/{fine['q10']:.3f} | {wiki['median']:.3f}/{wiki['q10']:.3f} | {wiki['minimum']:.3f} |")
    lines += ["", f"The raw fingerprint WikiText 10th percentile was {raw_q10:.3f}. Failure after invariant encodings is stronger evidence against singleton norms/cosines than the original corpus-shifted result."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
