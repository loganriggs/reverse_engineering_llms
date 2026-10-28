#!/usr/bin/env python3
"""Test simple per-layer selectors for the Layer-8 suffix route atlas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_layer8_suffix_attention_subset import (
    gram_objective,
    greedy_from_gram,
    masked_gram,
    prepare_site,
    selected_rows,
    tensor_program,
)
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_suffix_route_heuristics.json")
OUTPUT = TENSOR_ROOT / "layer8_suffix_route_heuristics.json"
REPORT = TENSOR_ROOT / "LAYER8_SUFFIX_ROUTE_HEURISTICS.md"
FIGURE = TENSOR_ROOT / "layer8_suffix_route_heuristics.png"


def choose_top(scores, layers):
    return tuple(sorted(layer for _, layer in sorted(
        ((float(scores[layer]), layer) for layer in layers), reverse=True,
    )[:3]))


def main():
    config = json.loads(CONFIG.read_text())
    atlas_config = json.loads(Path(config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": parent["evaluation_sites_per_query"],
                     "budget": parent["budget"]}
    atlas = json.loads((TENSOR_ROOT / "layer8_suffix_route_atlas.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(
        model, source_config, parent["queries"],
    )
    layers = parent["suffix_attention_layers"]
    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(parent["evaluation"].items()):
        rows = selected_rows(
            model, source_config, corpus_config, tensor_basis, values, vectors,
            parent["queries"], parent["evaluation_sites_per_query"],
        )
        atlas_sites = {(site["query"], site["row"]): site
                       for site in atlas["results"][corpus]["records"]}
        records = []
        for number, (query, row_index, raw) in enumerate(rows):
            site = prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)
            _, _, full_gram = masked_gram(
                model, site, source_config["causal_position"], layers,
            )
            _, _, direct_gram = masked_gram(
                model, site, source_config["causal_position"], (),
            )
            full_indices = greedy_from_gram(full_gram, site["amplitudes"], parent["budget"])
            full_objective = gram_objective(full_gram, site["amplitudes"], full_indices)
            singleton_utility = {}
            singleton_energy = {}
            leaveout_necessity = {}
            for layer in layers:
                _, _, singleton = masked_gram(
                    model, site, source_config["causal_position"], (layer,),
                )
                singleton_indices = greedy_from_gram(
                    singleton, site["amplitudes"], parent["budget"],
                )
                singleton_utility[layer] = gram_objective(
                    full_gram, site["amplitudes"], singleton_indices,
                ) / max(full_objective, 1e-30)
                singleton_energy[layer] = float(
                    (singleton - direct_gram).norm() / direct_gram.norm().clamp_min(1e-30)
                )
                _, _, leaveout = masked_gram(
                    model, site, source_config["causal_position"],
                    tuple(item for item in layers if item != layer),
                )
                leaveout_necessity[layer] = float(
                    (full_gram - leaveout).norm() / full_gram.norm().clamp_min(1e-30)
                )
            masks = {
                "singleton_utility": choose_top(singleton_utility, layers),
                "leaveout_necessity": choose_top(leaveout_necessity, layers),
                "self_contained_energy": choose_top(singleton_energy, layers),
            }
            lookup = {tuple(record["mask"]): record
                      for record in atlas_sites[(query, row_index)]["masks"]}
            records.append({
                "query": query, "row": row_index,
                "masks": {name: list(mask) for name, mask in masks.items()},
                "routes": {name: lookup[mask] for name, mask in masks.items()},
                "scores": {
                    "singleton_utility": singleton_utility,
                    "leaveout_necessity": leaveout_necessity,
                    "self_contained_energy": singleton_energy,
                },
            })
            print(corpus, "site", number + 1, "of", len(rows), flush=True)
        summary = {}
        for selector in ("singleton_utility", "leaveout_necessity", "self_contained_energy"):
            summary[selector] = summarize(
                [record["routes"][selector]["full_fidelity"] for record in records],
                config["seed"] + 10 * corpus_index,
            )
        results[corpus] = {"summary": summary, "records": records}

    threshold = lambda row: row["median"] >= .95 and row["q10"] >= .80
    singleton = all(threshold(results[c]["summary"]["singleton_utility"]) for c in results)
    leaveout = all(threshold(results[c]["summary"]["leaveout_necessity"]) for c in results)
    energy = all(threshold(results[c]["summary"]["self_contained_energy"]) for c in results)
    synergy = (not singleton and not leaveout and not energy and any(
        atlas["summary"][c]["sitewise_by_objective"]["q10"]
        - max(results[c]["summary"][selector]["q10"] for selector in results[c]["summary"]) >= .10
        for c in results
    ))
    hypotheses = {
        "S0_singleton_utility_selects_route": {"supported": bool(singleton)},
        "S1_leaveout_necessity_selects_route": {"supported": bool(leaveout)},
        "S2_self_contained_energy_selects_route": {"supported": bool(energy)},
        "S3_strong_layer_synergy": {"supported": bool(synergy)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    selectors = ["singleton_utility", "leaveout_necessity", "self_contained_energy"]
    labels = ["singleton\nutility", "leaveout\nnecessity", "self-contained\nenergy"]
    for corpus, color in (("fineweb", "#1769aa"), ("wikitext", "#b71c1c")):
        axes[0].plot(range(3), [results[corpus]["summary"][s]["median"] for s in selectors], marker="o", color=color, label=corpus)
        axes[1].plot(range(3), [results[corpus]["summary"][s]["q10"] for s in selectors], marker="o", color=color, label=corpus)
    for axis, title in zip(axes, ("Median fidelity", "10th-percentile fidelity")):
        axis.set_xticks(range(3), labels); axis.set_ylim(-.2, 1.08); axis.axhline(.8, color="#777", ls="--", lw=1); axis.set_title(title); axis.grid(alpha=.2)
    axes[0].legend(); figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)

    lines = ["# Simple selectors for the Layer-8 suffix-route atlas", "",
             "Three per-layer heuristics choose a size-3 mask without inspecting that mask's finite causal answer. Their selected masks are scored using the already frozen exhaustive-atlas interventions.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "![Route heuristics](layer8_suffix_route_heuristics.png)", "",
              "| Corpus | Selector | Median fidelity | 10th percentile | Minimum |",
              "|---|---|---:|---:|---:|"]
    for corpus in results:
        for selector, row in results[corpus]["summary"].items():
            lines.append(f"| {corpus} | {selector} | {row['median']:.3f} | {row['q10']:.3f} | {row['minimum']:.3f} |")
    lines += ["", "The singleton and leaveout selectors use the full Gram diagnostically. Only self-contained energy avoids that target. Failure here rejects these short rules, not all learnable or higher-order routers."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
