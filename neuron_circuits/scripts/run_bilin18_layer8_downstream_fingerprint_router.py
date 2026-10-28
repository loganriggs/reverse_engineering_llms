#!/usr/bin/env python3
"""Route Layer-8 suffix masks from singleton downstream Fisher fingerprints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from analyze_bilin18_layer8_suffix_mask_grammar import design, ridge_fit, ridge_predict
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_state_router import evaluate_router
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_downstream_fingerprint_router.json")
OUTPUT = TENSOR_ROOT / "layer8_downstream_fingerprint_router.json"
REPORT = TENSOR_ROOT / "LAYER8_DOWNSTREAM_FINGERPRINT_ROUTER.md"


@torch.inference_mode()
def fingerprint(model, site, layers, position):
    logits, direct_tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=False,
    )
    direct = output_fisher_response(logits, direct_tangents)
    deltas = []
    for layer in layers:
        singleton_logits, singleton_tangents, _ = analytic_suffix(
            model, site["x8"], site["x0"], site["value_bus"], position,
            site["tangents"], include_attention=True, attention_layers=(layer,),
        )
        if float((singleton_logits - logits).abs().max()) > 1e-6:
            raise AssertionError("singleton mask changed native logits")
        deltas.append(output_fisher_response(singleton_logits, singleton_tangents) - direct)
    norms = torch.stack([delta.norm() for delta in deltas]).clamp_min(1e-30)
    cosines = []
    for left in range(len(layers)):
        for right in range(left + 1, len(layers)):
            cosines.append(float((deltas[left] * deltas[right]).sum() / (norms[left] * norms[right])))
    return np.concatenate([
        np.log(norms.cpu().numpy()), np.asarray(cosines, dtype=np.float64),
    ])


def dynamic_design(mask_features, site_fingerprint):
    return np.concatenate([mask_features, mask_features * site_fingerprint[None]], axis=1)


def routed_rows(mask_features, fingerprints):
    return np.concatenate([dynamic_design(mask_features, row) for row in fingerprints], axis=0)


def evaluate(model, mask_features, fingerprints, sites):
    records = []
    for feature, site in zip(fingerprints, sites):
        prediction = ridge_predict(model, dynamic_design(mask_features, feature))
        chosen = int(np.argmax(prediction))
        records.append({
            "query": site["query"], "row": site["row"],
            "mask": site["masks"][chosen]["mask"],
            "full_fidelity": site["masks"][chosen]["full_fidelity"],
            "predicted_score": float(prediction[chosen]),
        })
    return records


def main():
    config = json.loads(CONFIG.read_text())
    scaling_config = json.loads(Path(config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": parent["evaluation_sites_per_query"],
                     "budget": parent["budget"]}
    atlas = json.loads(Path(state_config["atlas"]).read_text())
    raw_state = json.loads((TENSOR_ROOT / "layer8_state_router_scaling.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    layers = parent["suffix_attention_layers"]
    fingerprints = {}
    for corpus, corpus_config in parent["evaluation"].items():
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], parent["evaluation_sites_per_query"])
        values_out = []
        for number, (query, _, raw) in enumerate(rows):
            site = prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)
            values_out.append(fingerprint(model, site, layers, source_config["causal_position"]))
            print(corpus, "fingerprint", number + 1, "of", len(rows), flush=True)
        fingerprints[corpus] = np.asarray(values_out)

    fine_sites = atlas["results"]["fineweb"]["records"]
    wiki_sites = atlas["results"]["wikitext"]["records"]
    masks = [tuple(row["mask"]) for row in fine_sites[0]["masks"]]
    mask_features = design(masks, "pairwise")
    targets = np.asarray([[row["full_fidelity"] for row in site["masks"]] for site in fine_sites])
    grid = []
    for alpha in config["ridge_alphas"]:
        records = []
        for fold in range(config["query_folds"]):
            valid = np.asarray([i for i, site in enumerate(fine_sites)
                                if site["query"] % config["query_folds"] == fold])
            train = np.setdiff1d(np.arange(len(fine_sites)), valid)
            router = ridge_fit(routed_rows(mask_features, fingerprints["fineweb"][train]),
                               targets[train].reshape(-1), alpha)
            records.extend(evaluate(router, mask_features, fingerprints["fineweb"][valid],
                                    [fine_sites[i] for i in valid]))
        grid.append({"alpha": alpha,
                     "summary": summarize([row["full_fidelity"] for row in records], config["seed"]),
                     "records": records})
    chosen = max(grid, key=lambda row: (row["summary"]["q10"], row["summary"]["median"], row["alpha"]))
    router = ridge_fit(routed_rows(mask_features, fingerprints["fineweb"]), targets.reshape(-1), chosen["alpha"])
    wiki_records = evaluate(router, mask_features, fingerprints["wikitext"], wiki_sites)
    rng = np.random.default_rng(config["seed"])
    shuffled = fingerprints["wikitext"][rng.permutation(len(wiki_sites))]
    shuffled_records = evaluate(router, mask_features, shuffled, wiki_sites)
    final = {
        "fineweb_cv": chosen["summary"],
        "wikitext": summarize([row["full_fidelity"] for row in wiki_records], config["seed"]),
        "wikitext_shuffled": summarize([row["full_fidelity"] for row in shuffled_records], config["seed"]),
    }
    validates = all(final[c]["median"] >= .95 and final[c]["q10"] >= .80
                    for c in ("fineweb_cv", "wikitext"))
    beats = all(final[c]["q10"] - raw_state["final"][c]["q10"] >= .05
                for c in ("fineweb_cv", "wikitext"))
    used = final["wikitext"]["q10"] - final["wikitext_shuffled"]["q10"] >= .10
    hypotheses = {
        "J0_downstream_fingerprint_validates": {"supported": bool(validates)},
        "J1_fingerprint_beats_raw_state": {"supported": bool(beats)},
        "J2_fingerprint_is_used": {"supported": bool(used)},
        "J3_pairwise_downstream_geometry_is_insufficient": {"supported": bool(not validates)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "grid": grid, "chosen_alpha": chosen["alpha"],
                                  "final": final, "wikitext_records": wiki_records,
                                  "shuffled_records": shuffled_records,
                                  "fingerprints": {key: value.tolist() for key, value in fingerprints.items()}}, indent=2))
    lines = ["# Downstream-sensitivity fingerprint router", "",
             "The router sees only singleton attention-JVP response norms and their pairwise signed cosines, plus static mask bits. It never sees the full suffix response or candidate finite effects at evaluation time.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Selected ridge alpha `{chosen['alpha']}`.", "",
              "| Split | Median fidelity | 10th percentile | Minimum |",
              "|---|---:|---:|---:|"]
    for split, row in final.items():
        lines.append(f"| {split} | {row['median']:.3f} | {row['q10']:.3f} | {row['minimum']:.3f} |")
    lines += ["", "The shuffled-site control tests whether dynamic fingerprints matter beyond global mask biases. Failure here rejects only singleton norms/cosines; head-level, factor-level, or higher-order path features remain possible."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
