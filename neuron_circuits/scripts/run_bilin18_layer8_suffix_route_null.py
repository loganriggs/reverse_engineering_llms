#!/usr/bin/env python3
"""Best-of-many random-program null for the compact suffix-route atlas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_suffix_attention_subset import (
    greedy_from_gram,
    masked_gram,
    prepare_site,
    selected_rows,
    tensor_program,
)
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_suffix_route_null.json")
OUTPUT = TENSOR_ROOT / "layer8_suffix_route_null.json"
REPORT = TENSOR_ROOT / "LAYER8_SUFFIX_ROUTE_NULL.md"
FIGURE = TENSOR_ROOT / "layer8_suffix_route_null.png"


def sample_programs(rng, count, dimension, budget, probabilities=None):
    result = []
    seen = set()
    while len(result) < count:
        indices = tuple(sorted(int(index) for index in rng.choice(
            dimension, size=budget, replace=False, p=probabilities,
        )))
        if indices not in seen:
            seen.add(indices); result.append(indices)
    return result


@torch.inference_mode()
def finite_bank(model, site, programs, full_indices, position, edit_fraction, batch):
    fidelities = []
    for start in range(0, len(programs), batch):
        stop = min(start + batch, len(programs))
        routes = {"full_ref": full_indices}
        routes.update({str(number): torch.as_tensor(
            programs[number], device=full_indices.device, dtype=torch.long,
        ) for number in range(start, stop)})
        native, finite = finite_routes(
            model, site["pre"], site["normalized"], site["x0"], site["value_bus"],
            position, site["mode_directions"], routes, site["output_direction"],
            edit_fraction,
        )
        reference = finite["full_ref"]["effect"]
        for number in range(start, stop):
            fidelity, _, _ = fisher_fidelity(native, reference, finite[str(number)]["effect"])
            fidelities.append(fidelity)
    return fidelities


def advantage(structured, null, median_threshold, q10_threshold):
    return (structured["median"] - null["median"] >= median_threshold
            or structured["q10"] - null["q10"] >= q10_threshold)


def equivalent(structured, null):
    return (structured["median"] - null["median"] <= .01
            and structured["q10"] - null["q10"] <= .05)


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
            full_indices = greedy_from_gram(full_gram, site["amplitudes"], parent["budget"])
            score = (full_gram.diagonal().clamp_min(0)
                     / site["amplitudes"].square().clamp_min(1e-30)).cpu().numpy()
            score = np.maximum(score, np.finfo(np.float64).tiny); score /= score.sum()
            rng = np.random.default_rng(config["seed"] + corpus_index * 100000 + query * 1000 + row_index)
            uniform = sample_programs(
                rng, config["draws"], len(score), parent["budget"], None,
            )
            weighted = sample_programs(
                rng, config["draws"], len(score), parent["budget"], score,
            )
            uniform_fidelity = finite_bank(
                model, site, uniform, full_indices, source_config["causal_position"],
                source_config["edit_norm_fraction"], config["finite_batch"],
            )
            weighted_fidelity = finite_bank(
                model, site, weighted, full_indices, source_config["causal_position"],
                source_config["edit_norm_fraction"], config["finite_batch"],
            )
            records.append({
                "query": query, "row": row_index,
                "structured_best": atlas_sites[(query, row_index)]["by_full"]["full_fidelity"],
                "uniform_best": float(max(uniform_fidelity)),
                "weighted_best": float(max(weighted_fidelity)),
                "uniform_median": float(np.median(uniform_fidelity)),
                "weighted_median": float(np.median(weighted_fidelity)),
                "uniform_best_program": list(uniform[int(np.argmax(uniform_fidelity))]),
                "weighted_best_program": list(weighted[int(np.argmax(weighted_fidelity))]),
            })
            print(corpus, "site", number + 1, "of", len(rows), flush=True)
        summary = {
            key: summarize([record[key] for record in records], config["seed"] + corpus_index)
            for key in ("structured_best", "uniform_best", "weighted_best",
                        "uniform_median", "weighted_median")
        }
        results[corpus] = {"summary": summary, "records": records}

    structured_uniform = all(advantage(
        results[c]["summary"]["structured_best"], results[c]["summary"]["uniform_best"], .05, .10,
    ) for c in results)
    structured_weighted = all(advantage(
        results[c]["summary"]["structured_best"], results[c]["summary"]["weighted_best"], .02, .05,
    ) for c in results)
    weighted_collision = all(equivalent(
        results[c]["summary"]["structured_best"], results[c]["summary"]["weighted_best"],
    ) for c in results)
    uniform_collision = all(equivalent(
        results[c]["summary"]["structured_best"], results[c]["summary"]["uniform_best"],
    ) for c in results)
    hypotheses = {
        "N0_structured_beats_uniform": {"supported": bool(structured_uniform)},
        "N1_structured_beats_fisher_weighted": {"supported": bool(structured_weighted)},
        "N2_effect_collision_explains_atlas": {"supported": bool(weighted_collision)},
        "N3_uniform_collision_explains_atlas": {"supported": bool(uniform_collision)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ["structured_best", "uniform_best", "weighted_best"]
    labels = ["attention\ntriples", "uniform\nbest-of-84", "Fisher-weighted\nbest-of-84"]
    for corpus, color in (("fineweb", "#1769aa"), ("wikitext", "#b71c1c")):
        axes[0].plot(range(3), [results[corpus]["summary"][name]["median"] for name in names], marker="o", color=color, label=corpus)
        axes[1].plot(range(3), [results[corpus]["summary"][name]["q10"] for name in names], marker="o", color=color, label=corpus)
    for axis, title in zip(axes, ("Median best fidelity", "10th-percentile best fidelity")):
        axis.set_xticks(range(3), labels); axis.set_ylim(-.2, 1.08); axis.set_title(title); axis.grid(alpha=.2)
    axes[0].legend(); figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)

    lines = ["# Best-of-many null for the compact suffix-route atlas", "",
             "The best of all 84 structured attention triples is compared with equal-size banks of uniform and full-Fisher-weighted random 16-mode programs. Every candidate is evaluated by an exact finite intervention.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "![Route null](layer8_suffix_route_null.png)", "",
              "| Corpus | Bank | Median best fidelity | 10th percentile | Minimum |",
              "|---|---|---:|---:|---:|"]
    for corpus in results:
        for name in names:
            row = results[corpus]["summary"][name]
            lines.append(f"| {corpus} | {name} | {row['median']:.3f} | {row['q10']:.3f} | {row['minimum']:.3f} |")
    lines += ["", "A random-bank match means that compact effect equivalence is common under this search budget; it does not mean random programs reproduce the same internal route. A structured advantage strengthens attention-route localization but still does not provide a reusable selector."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
