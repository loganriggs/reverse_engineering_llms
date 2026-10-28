#!/usr/bin/env python3
"""Route suffix triples using exact pairwise attention-JVP response synergies."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from analyze_bilin18_layer8_suffix_mask_grammar import design, ridge_fit, ridge_predict
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_pair_synergy_router.json")
OUTPUT = TENSOR_ROOT / "layer8_pair_synergy_router.json"
REPORT = TENSOR_ROOT / "LAYER8_PAIR_SYNERGY_ROUTER.md"


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / max(values.std(), 1e-12)


@torch.inference_mode()
def pair_fingerprint(model, site, layers, position):
    logits, direct_tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=False,
    )
    direct = output_fisher_response(logits, direct_tangents)
    corrections = []
    for layer in layers:
        _, tangents, _ = analytic_suffix(
            model, site["x8"], site["x0"], site["value_bus"], position,
            site["tangents"], include_attention=True, attention_layers=(layer,),
        )
        corrections.append(output_fisher_response(logits, tangents) - direct)
    singleton_log_norm = zscore(np.log([
        max(float(correction.norm()), 1e-30) for correction in corrections
    ]))
    synergy_log_norm = []
    pair_log_norm = []
    synergy_cosine = []
    for left, right in itertools.combinations(range(len(layers)), 2):
        _, pair_tangents, _ = analytic_suffix(
            model, site["x8"], site["x0"], site["value_bus"], position,
            site["tangents"], include_attention=True,
            attention_layers=(layers[left], layers[right]),
        )
        pair_correction = output_fisher_response(logits, pair_tangents) - direct
        additive = corrections[left] + corrections[right]
        synergy = pair_correction - additive
        synergy_norm = synergy.norm().clamp_min(1e-30)
        synergy_log_norm.append(np.log(float(synergy_norm)))
        pair_log_norm.append(np.log(max(float(pair_correction.norm()), 1e-30)))
        synergy_cosine.append(float((synergy * additive).sum() /
                                    (synergy_norm * additive.norm().clamp_min(1e-30))))
    return {
        "singleton": singleton_log_norm,
        "synergy_norm": zscore(synergy_log_norm),
        "pair_norm": zscore(pair_log_norm),
        "synergy_cosine": np.asarray(synergy_cosine),
    }


def candidate_design(static, fingerprint):
    bits, pairs = static[:, :9], static[:, 9:]
    return np.concatenate([
        static,
        bits * fingerprint["singleton"][None],
        pairs * fingerprint["synergy_norm"][None],
        pairs * fingerprint["pair_norm"][None],
        pairs * fingerprint["synergy_cosine"][None],
    ], axis=1)


def stack_design(static, fingerprints):
    return np.concatenate([candidate_design(static, fingerprint) for fingerprint in fingerprints], axis=0)


def evaluate(router, static, fingerprints, sites):
    records = []
    for fingerprint, site in zip(fingerprints, sites):
        prediction = ridge_predict(router, candidate_design(static, fingerprint))
        chosen = int(np.argmax(prediction))
        records.append({"query": site["query"], "row": site["row"],
                        "mask": site["masks"][chosen]["mask"],
                        "full_fidelity": site["masks"][chosen]["full_fidelity"],
                        "predicted_score": float(prediction[chosen])})
    return records


def main():
    config = json.loads(CONFIG.read_text())
    singleton_config = json.loads(Path(config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": parent["evaluation_sites_per_query"],
                     "budget": parent["budget"]}
    atlas = json.loads(Path(state_config["atlas"]).read_text())
    invariant = json.loads((TENSOR_ROOT / "layer8_fingerprint_invariance_redteam.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    layers = parent["suffix_attention_layers"]
    fingerprints = {}
    for corpus, corpus_config in parent["evaluation"].items():
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], parent["evaluation_sites_per_query"])
        output = []
        for number, (query, _, raw) in enumerate(rows):
            site = prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)
            output.append(pair_fingerprint(model, site, layers, source_config["causal_position"]))
            print(corpus, "pair fingerprint", number + 1, "of", len(rows), flush=True)
        fingerprints[corpus] = output
    fine_sites = atlas["results"]["fineweb"]["records"]
    wiki_sites = atlas["results"]["wikitext"]["records"]
    masks = [tuple(row["mask"]) for row in fine_sites[0]["masks"]]
    static = design(masks, "pairwise")
    targets = np.asarray([[row["full_fidelity"] for row in site["masks"]] for site in fine_sites])
    grid = []
    for alpha in config["ridge_alphas"]:
        records = []
        for fold in range(config["query_folds"]):
            valid = np.asarray([i for i, site in enumerate(fine_sites)
                                if site["query"] % config["query_folds"] == fold])
            train = np.setdiff1d(np.arange(len(fine_sites)), valid)
            router = ridge_fit(stack_design(static, [fingerprints["fineweb"][i] for i in train]),
                               targets[train].reshape(-1), alpha)
            records.extend(evaluate(router, static, [fingerprints["fineweb"][i] for i in valid],
                                    [fine_sites[i] for i in valid]))
        grid.append({"alpha": alpha, "summary": summarize(
            [row["full_fidelity"] for row in records], config["seed"]), "records": records})
    chosen = max(grid, key=lambda row: (row["summary"]["q10"], row["summary"]["median"], row["alpha"]))
    router = ridge_fit(stack_design(static, fingerprints["fineweb"]), targets.reshape(-1), chosen["alpha"])
    wiki_records = evaluate(router, static, fingerprints["wikitext"], wiki_sites)
    rng = np.random.default_rng(config["seed"])
    shuffled_fingerprints = [fingerprints["wikitext"][i] for i in rng.permutation(len(wiki_sites))]
    shuffled_records = evaluate(router, static, shuffled_fingerprints, wiki_sites)
    final = {"fineweb_cv": chosen["summary"],
             "wikitext": summarize([row["full_fidelity"] for row in wiki_records], config["seed"]),
             "wikitext_shuffled": summarize([row["full_fidelity"] for row in shuffled_records], config["seed"])}
    validates = all(final[c]["median"] >= .95 and final[c]["q10"] >= .80
                    for c in ("fineweb_cv", "wikitext"))
    best_invariant = {corpus: max(
        result["summary"][corpus]["q10"] for result in invariant["results"].values()
    ) for corpus in ("fineweb_cv", "wikitext")}
    adds = all(final[c]["q10"] - best_invariant[c] >= .10 for c in best_invariant)
    used = final["wikitext"]["q10"] - final["wikitext_shuffled"]["q10"] >= .10
    hypotheses = {
        "P0_pair_synergy_router_validates": {"supported": bool(validates)},
        "P1_pair_synergy_adds_value": {"supported": bool(adds)},
        "P2_pair_synergy_features_are_used": {"supported": bool(used)},
        "P3_second_order_path_grammar_is_insufficient": {"supported": bool(not validates)},
    }
    serializable = {corpus: [{key: value.tolist() for key, value in row.items()}
                             for row in rows] for corpus, rows in fingerprints.items()}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "grid": grid, "chosen_alpha": chosen["alpha"],
                                  "best_invariant_q10": best_invariant, "final": final,
                                  "wikitext_records": wiki_records,
                                  "shuffled_records": shuffled_records,
                                  "fingerprints": serializable}, indent=2))
    lines = ["# Pair-synergy router for the Layer-8 suffix atlas", "",
             "The router sees invariant singleton norms plus exact pair-response norms, second-order synergy norms, and signed synergy/additive cosines. No triple or full-suffix response is used as an input.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Selected ridge alpha `{chosen['alpha']}`.", "",
              "| Split | Median fidelity | 10th percentile | Minimum |",
              "|---|---:|---:|---:|"]
    for split, row in final.items():
        lines.append(f"| {split} | {row['median']:.3f} | {row['q10']:.3f} | {row['minimum']:.3f} |")
    lines += ["", "A pass would provide a layer-level second-order path grammar. Failure means that pair synergies alone do not robustly choose triples; it does not rule out head-level decomposition or true third-order propagation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
