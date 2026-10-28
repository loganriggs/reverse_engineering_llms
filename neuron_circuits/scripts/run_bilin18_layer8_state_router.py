#!/usr/bin/env python3
"""Learn a state-conditioned pairwise router for local suffix masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_bilin18_layer8_suffix_mask_grammar import design, ridge_fit, ridge_predict
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_state_router.json")
OUTPUT = TENSOR_ROOT / "layer8_state_router.json"
REPORT = TENSOR_ROOT / "LAYER8_STATE_ROUTER.md"


def pca_fit(values, rank):
    mean = values.mean(0)
    _, _, vh = np.linalg.svd(values - mean, full_matrices=False)
    return mean, vh[:rank].T


def pca_apply(model, values):
    mean, basis = model
    return (values - mean) @ basis


def routed_design(mask_features, site_features):
    rows = []
    for site in site_features:
        rows.append(np.concatenate([mask_features] + [mask_features * value for value in site], axis=1))
    return np.concatenate(rows, axis=0)


def choose_alpha(mask_features, site_features, targets, config):
    site_count = len(site_features)
    choices = {}
    for alpha in config["ridge_alphas"]:
        selected = []
        for fold in range(config["validation_folds"]):
            valid_sites = np.arange(site_count)[np.arange(site_count) % config["validation_folds"] == fold]
            train_sites = np.setdiff1d(np.arange(site_count), valid_sites)
            x_train = routed_design(mask_features, site_features[train_sites])
            y_train = targets[train_sites].reshape(-1)
            model = ridge_fit(x_train, y_train, alpha)
            for site_index in valid_sites:
                prediction = ridge_predict(model, routed_design(mask_features, site_features[[site_index]]))
                selected.append(float(targets[site_index, int(np.argmax(prediction))]))
        choices[alpha] = float(np.median(selected))
    return max(config["ridge_alphas"], key=lambda alpha: (choices[alpha], -alpha)), choices


def site_diagnostics(record):
    values = []
    for name in ("singleton_utility", "leaveout_necessity", "self_contained_energy"):
        score = record["scores"][name]
        values.extend(float(score[str(layer)]) for layer in range(9, 18))
    return values


def evaluate_router(model, mask_features, features, atlas_sites):
    records = []
    for site_feature, site in zip(features, atlas_sites):
        prediction = ridge_predict(model, routed_design(mask_features, site_feature[None]))
        chosen = int(np.argmax(prediction))
        records.append({
            "query": site["query"], "row": site["row"], "mask": site["masks"][chosen]["mask"],
            "predicted_score": float(prediction[chosen]),
            "full_fidelity": site["masks"][chosen]["full_fidelity"],
            "objective_efficiency": site["masks"][chosen]["objective_efficiency"],
        })
    return records


def main():
    config = json.loads(CONFIG.read_text())
    atlas = json.loads(Path(config["atlas"]).read_text())
    heuristics = json.loads(Path(config["heuristics"]).read_text())
    atlas_config = json.loads(Path(config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": parent["evaluation_sites_per_query"],
                     "budget": parent["budget"]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])

    states = {}; diagnostics = {}
    for corpus, corpus_config in parent["evaluation"].items():
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], parent["evaluation_sites_per_query"])
        state_rows = []
        for query, row_index, raw in rows:
            site = prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)
            state_rows.append(site["x"].cpu().numpy())
        states[corpus] = np.asarray(state_rows)
        hlookup = {(row["query"], row["row"]): row for row in heuristics["results"][corpus]["records"]}
        diagnostics[corpus] = np.asarray([
            site_diagnostics(hlookup[(site["query"], site["row"])])
            for site in atlas["results"][corpus]["records"]
        ])

    fine_sites = atlas["results"]["fineweb"]["records"]
    train_indices = np.asarray([next(i for i, site in enumerate(fine_sites) if site["query"] == query)
                                for query in range(parent["queries"])])
    test_indices = np.asarray([next(i for i, site in enumerate(fine_sites)
                               if site["query"] == query and i not in train_indices)
                               for query in range(parent["queries"])])
    masks = [tuple(row["mask"]) for row in fine_sites[0]["masks"]]
    mask_features = design(masks, "pairwise")
    targets = np.asarray([[row["full_fidelity"] for row in fine_sites[index]["masks"]]
                          for index in train_indices])
    state_pca = pca_fit(states["fineweb"][train_indices], config["state_rank"])
    diagnostic_pca = pca_fit(diagnostics["fineweb"][train_indices], config["diagnostic_rank"])
    feature_banks = {
        "state": {
            "train": pca_apply(state_pca, states["fineweb"][train_indices]),
            "fineweb": pca_apply(state_pca, states["fineweb"][test_indices]),
            "wikitext": pca_apply(state_pca, states["wikitext"]),
        },
        "diagnostics": {
            "train": pca_apply(diagnostic_pca, diagnostics["fineweb"][train_indices]),
            "fineweb": pca_apply(diagnostic_pca, diagnostics["fineweb"][test_indices]),
            "wikitext": pca_apply(diagnostic_pca, diagnostics["wikitext"]),
        },
    }
    feature_banks["combined"] = {key: np.concatenate([
        feature_banks["state"][key], feature_banks["diagnostics"][key],
    ], axis=1) for key in ("train", "fineweb", "wikitext")}

    results = {}
    for name, bank in feature_banks.items():
        alpha, validation = choose_alpha(mask_features, bank["train"], targets, config)
        router = ridge_fit(routed_design(mask_features, bank["train"]), targets.reshape(-1), alpha)
        train_records = evaluate_router(router, mask_features, bank["train"], [fine_sites[i] for i in train_indices])
        fine_records = evaluate_router(router, mask_features, bank["fineweb"], [fine_sites[i] for i in test_indices])
        wiki_records = evaluate_router(router, mask_features, bank["wikitext"], atlas["results"]["wikitext"]["records"])
        results[name] = {"alpha": alpha, "validation": validation, "records": {
            "train": train_records, "fineweb": fine_records, "wikitext": wiki_records,
        }, "summary": {
            corpus: summarize([row["full_fidelity"] for row in records], config["seed"])
            for corpus, records in (("train", train_records), ("fineweb", fine_records), ("wikitext", wiki_records))
        }}

    passes = {name: all(results[name]["summary"][c]["median"] >= .95
                        and results[name]["summary"][c]["q10"] >= .80
                        for c in ("fineweb", "wikitext")) for name in results}
    combined_gain = passes["combined"] and any(
        results["combined"]["summary"][c]["q10"]
        - max(results["state"]["summary"][c]["q10"], results["diagnostics"]["summary"][c]["q10"]) >= .05
        for c in ("fineweb", "wikitext")
    )
    insufficient = all(any(results[name]["summary"][c]["q10"] < .50
                           for c in ("fineweb", "wikitext")) for name in results)
    hypotheses = {
        "R0_state_routes_mask": {"supported": bool(passes["state"])},
        "R1_diagnostics_route_mask": {"supported": bool(passes["diagnostics"])},
        "R2_combined_router": {"supported": bool(combined_gain)},
        "R3_linear_address_is_insufficient": {"supported": bool(insufficient)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "train_indices": train_indices.tolist(),
                                  "test_indices": test_indices.tolist(), "results": results}, indent=2))
    lines = ["# State-conditioned router for the Layer-8 suffix atlas", "",
             "A bilinear ridge model combines pairwise mask features with either Layer-8 state PCs, per-layer derivative-diagnostic PCs, or both. It is trained on one FineWeb site per tensor query and frozen for another FineWeb site and WikiText.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Router | Corpus | Median fidelity | 10th percentile | Minimum | Ridge alpha |",
              "|---|---|---:|---:|---:|---:|"]
    for name, result in results.items():
        for corpus, row in result["summary"].items():
            lines.append(f"| {name} | {corpus} | {row['median']:.3f} | {row['q10']:.3f} | {row['minimum']:.3f} | {result['alpha']:.3g} |")
    lines += ["", "The router is trained from finite causal labels. Passing would show that the atlas address is predictable from a low-rank state description; it would not make the explanation purely weight-derived. Failure only rejects this bilinear low-data router."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
