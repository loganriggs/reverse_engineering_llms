#!/usr/bin/env python3
"""Cosine-kernel atlas router over frozen Layer-8 mask-response surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_bilin18_layer8_state_router_scaling import collect_states
from run_bilin18_layer8_suffix_attention_subset import tensor_program
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_kernel_router.json")
OUTPUT = TENSOR_ROOT / "layer8_kernel_router.json"
REPORT = TENSOR_ROOT / "LAYER8_KERNEL_ROUTER.md"


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-30)


def route(train_states, train_surfaces, target_states, target_sites, neighbors, temperature):
    similarity = normalize(target_states) @ normalize(train_states).T
    records = []
    for index, site in enumerate(target_sites):
        chosen_neighbors = np.argsort(similarity[index])[-neighbors:]
        logits = similarity[index, chosen_neighbors] / temperature
        weights = np.exp(logits - logits.max()); weights /= weights.sum()
        prediction = weights @ train_surfaces[chosen_neighbors]
        selected = int(np.argmax(prediction))
        records.append({
            "query": site["query"], "row": site["row"],
            "mask": site["masks"][selected]["mask"],
            "full_fidelity": site["masks"][selected]["full_fidelity"],
            "neighbor_similarities": similarity[index, chosen_neighbors].tolist(),
            "weights": weights.tolist(),
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
    linear = json.loads((TENSOR_ROOT / "layer8_state_router_scaling.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    states = collect_states(model, parent, source_config, tensor_basis, directions, values, vectors)
    fine_sites = atlas["results"]["fineweb"]["records"]
    wiki_sites = atlas["results"]["wikitext"]["records"]
    surfaces = np.asarray([[row["full_fidelity"] for row in site["masks"]] for site in fine_sites])
    grid = []
    for neighbors in config["neighbors"]:
        for temperature in config["temperatures"]:
            records = []
            for fold in range(config["query_folds"]):
                valid = np.asarray([i for i, site in enumerate(fine_sites)
                                    if site["query"] % config["query_folds"] == fold])
                train = np.setdiff1d(np.arange(len(fine_sites)), valid)
                records.extend(route(states["fineweb"][train], surfaces[train],
                                     states["fineweb"][valid], [fine_sites[i] for i in valid],
                                     min(neighbors, len(train)), temperature))
            grid.append({"neighbors": neighbors, "temperature": temperature,
                         "summary": summarize([row["full_fidelity"] for row in records], config["seed"]),
                         "records": records})
    chosen = max(grid, key=lambda row: (row["summary"]["q10"], row["summary"]["median"],
                                        -row["neighbors"], -row["temperature"]))
    wiki_records = route(states["fineweb"], surfaces, states["wikitext"], wiki_sites,
                         chosen["neighbors"], chosen["temperature"])
    final = {"fineweb_cv": chosen["summary"],
             "wikitext": summarize([row["full_fidelity"] for row in wiki_records], config["seed"])}
    validates = all(final[c]["median"] >= .95 and final[c]["q10"] >= .80 for c in final)
    beats = all(final[c]["q10"] - linear["final"][c]["q10"] >= .05 for c in final)
    hypotheses = {
        "K0_local_state_atlas_validates": {"supported": bool(validates)},
        "K1_kernel_beats_linear_tail": {"supported": bool(beats)},
        "K2_raw_state_neighborhood_is_insufficient": {"supported": bool(not validates)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "grid": grid, "chosen": {"neighbors": chosen["neighbors"],
                                  "temperature": chosen["temperature"]}, "final": final,
                                  "wikitext_records": wiki_records}, indent=2))
    lines = ["# Nonlinear nearest-state router for the Layer-8 suffix atlas", "",
             "A cosine kernel averages complete causal mask-response surfaces from neighboring FineWeb states. Hyperparameters are selected by query-grouped FineWeb validation and frozen for WikiText.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Selected `{chosen['neighbors']}` neighbors and temperature `{chosen['temperature']}`.", "",
              "| Split | Median fidelity | 10th percentile | Minimum |",
              "|---|---:|---:|---:|"]
    for split, row in final.items():
        lines.append(f"| {split} | {row['median']:.3f} | {row['q10']:.3f} | {row['minimum']:.3f} |")
    lines += ["", "Failure means raw Layer-8 cosine neighborhoods do not provide a reliable atlas chart selector at this data scale. It does not rule out learned nonlinear metrics or downstream-sensitivity features."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
