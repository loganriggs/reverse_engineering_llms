#!/usr/bin/env python3
"""Grouped-CV data/rank red team for the Layer-8 state router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_bilin18_layer8_suffix_mask_grammar import design, ridge_fit
from run_bilin18_layer8_state_router import (
    evaluate_router,
    pca_apply,
    pca_fit,
    routed_design,
)
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_state_router_scaling.json")
OUTPUT = TENSOR_ROOT / "layer8_state_router_scaling.json"
REPORT = TENSOR_ROOT / "LAYER8_STATE_ROUTER_SCALING.md"


def collect_states(model, parent, source_config, tensor_basis, directions, values, vectors):
    result = {}
    for corpus, corpus_config in parent["evaluation"].items():
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], parent["evaluation_sites_per_query"])
        result[corpus] = np.asarray([
            prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)["x"].cpu().numpy()
            for query, _, raw in rows
        ])
    return result


def main():
    config = json.loads(CONFIG.read_text())
    first_config = json.loads(Path(config["parent_config"]).read_text())
    atlas_config = json.loads(Path(first_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": parent["evaluation_sites_per_query"],
                     "budget": parent["budget"]}
    atlas = json.loads(Path(first_config["atlas"]).read_text())
    first = json.loads((TENSOR_ROOT / "layer8_state_router.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    states = collect_states(model, parent, source_config, tensor_basis, directions, values, vectors)
    fine_sites = atlas["results"]["fineweb"]["records"]
    wiki_sites = atlas["results"]["wikitext"]["records"]
    masks = [tuple(row["mask"]) for row in fine_sites[0]["masks"]]
    mask_features = design(masks, "pairwise")
    targets = np.asarray([[row["full_fidelity"] for row in site["masks"]] for site in fine_sites])
    grid = []
    for rank in config["state_ranks"]:
        for alpha in config["ridge_alphas"]:
            records = []
            for fold in range(config["query_folds"]):
                valid = np.asarray([i for i, site in enumerate(fine_sites)
                                    if site["query"] % config["query_folds"] == fold])
                train = np.setdiff1d(np.arange(len(fine_sites)), valid)
                pca = pca_fit(states["fineweb"][train], rank)
                train_features = pca_apply(pca, states["fineweb"][train])
                valid_features = pca_apply(pca, states["fineweb"][valid])
                router = ridge_fit(routed_design(mask_features, train_features), targets[train].reshape(-1), alpha)
                records.extend(evaluate_router(router, mask_features, valid_features, [fine_sites[i] for i in valid]))
            stats = summarize([row["full_fidelity"] for row in records], config["seed"])
            grid.append({"rank": rank, "alpha": alpha, "summary": stats, "records": records})
    chosen = max(grid, key=lambda row: (row["summary"]["q10"], row["summary"]["median"],
                                        -row["rank"], row["alpha"]))
    pca = pca_fit(states["fineweb"], chosen["rank"])
    train_features = pca_apply(pca, states["fineweb"])
    wiki_features = pca_apply(pca, states["wikitext"])
    router = ridge_fit(routed_design(mask_features, train_features), targets.reshape(-1), chosen["alpha"])
    train_records = evaluate_router(router, mask_features, train_features, fine_sites)
    wiki_records = evaluate_router(router, mask_features, wiki_features, wiki_sites)
    final = {
        "fineweb_train": summarize([row["full_fidelity"] for row in train_records], config["seed"]),
        "fineweb_cv": chosen["summary"],
        "wikitext": summarize([row["full_fidelity"] for row in wiki_records], config["seed"]),
    }
    old_q10 = first["results"]["state"]["summary"]["wikitext"]["q10"]
    data_limited = final["wikitext"]["q10"] >= .80 and final["wikitext"]["q10"] - old_q10 >= .10
    validates = all(final[c]["median"] >= .95 and final[c]["q10"] >= .80
                    for c in ("fineweb_cv", "wikitext"))
    hypotheses = {
        "D0_previous_failure_was_data_limited": {"supported": bool(data_limited)},
        "D1_linear_state_router_validates": {"supported": bool(validates)},
        "D2_linear_state_address_plateaus": {"supported": bool(not validates)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "old_wikitext_q10": old_q10, "grid": grid,
                                  "chosen": {"rank": chosen["rank"], "alpha": chosen["alpha"]},
                                  "final": final, "train_records": train_records,
                                  "wikitext_records": wiki_records}, indent=2))
    lines = ["# Data/rank red team of the Layer-8 state router", "",
             "State rank and ridge strength are selected by query-grouped FineWeb cross-validation, then frozen for WikiText. Both FineWeb sites per query are used only after model selection for the final OOD fit.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Selected rank `{chosen['rank']}` and ridge alpha `{chosen['alpha']}`.", "",
              "| Split | Median fidelity | 10th percentile | Minimum |",
              "|---|---:|---:|---:|"]
    for split, row in final.items():
        lines.append(f"| {split} | {row['median']:.3f} | {row['q10']:.3f} | {row['minimum']:.3f} |")
    lines += ["", f"The original 16-site WikiText 10th percentile was {old_q10:.3f}. A plateau here rejects only this linear low-rank state router; nonlinear routers or additional state variables remain live."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
