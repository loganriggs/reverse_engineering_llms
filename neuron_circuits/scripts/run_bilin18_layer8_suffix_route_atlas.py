#!/usr/bin/env python3
"""Exhaust size-3 suffix attention routes and test query-level recurrence."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_suffix_attention_subset import (
    bootstrap_median_interval,
    gram_objective,
    greedy_from_gram,
    masked_gram,
    prepare_site,
    selected_rows,
    tensor_program,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_suffix_route_atlas.json")
OUTPUT = TENSOR_ROOT / "layer8_suffix_route_atlas.json"
REPORT = TENSOR_ROOT / "LAYER8_SUFFIX_ROUTE_ATLAS.md"
FIGURE = TENSOR_ROOT / "layer8_suffix_route_atlas.png"


def summarize(values, seed):
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "minimum": float(array.min()),
        "ci95_median": bootstrap_median_interval(array, seed),
    }


@torch.inference_mode()
def audit_site(model, site, query, row_index, masks, layers, position, budget,
               edit_norm_fraction, oracle_indices, finite_batch):
    full_logits, _, full_gram = masked_gram(model, site, position, layers)
    full_indices = greedy_from_gram(full_gram, site["amplitudes"], budget)
    full_objective = gram_objective(full_gram, site["amplitudes"], full_indices)
    prior_oracle = torch.as_tensor(
        oracle_indices[(query, row_index)], device=full_indices.device, dtype=torch.long,
    )
    programs = {}
    objective = {}
    for number, mask in enumerate(masks):
        logits, _, gram = masked_gram(model, site, position, mask)
        if float((logits - full_logits).abs().max()) > 1e-6:
            raise AssertionError("derivative mask changed native logits")
        indices = greedy_from_gram(gram, site["amplitudes"], budget)
        programs[number] = indices
        objective[number] = gram_objective(full_gram, site["amplitudes"], indices) / max(full_objective, 1e-30)

    records = []
    reference_kl = None
    for start in range(0, len(masks), finite_batch):
        stop = min(start + finite_batch, len(masks))
        names = {"full_ref": full_indices, "oracle_ref": prior_oracle}
        names.update({str(number): programs[number] for number in range(start, stop)})
        native, finite = finite_routes(
            model, site["pre"], site["normalized"], site["x0"], site["value_bus"],
            position, site["mode_directions"], names, site["output_direction"],
            edit_norm_fraction,
        )
        full_effect = finite["full_ref"]["effect"]
        oracle_effect = finite["oracle_ref"]["effect"]
        if reference_kl is None:
            reference_kl = (finite["full_ref"]["native_kl"], finite["oracle_ref"]["native_kl"])
        for number in range(start, stop):
            full_fidelity, full_cosine, _ = fisher_fidelity(
                native, full_effect, finite[str(number)]["effect"],
            )
            oracle_fidelity, oracle_cosine, _ = fisher_fidelity(
                native, oracle_effect, finite[str(number)]["effect"],
            )
            records.append({
                "mask": list(masks[number]),
                "objective_efficiency": objective[number],
                "full_fidelity": full_fidelity,
                "full_cosine": full_cosine,
                "oracle_fidelity": oracle_fidelity,
                "oracle_cosine": oracle_cosine,
                "support_recall": len(set(programs[number].cpu().tolist()) & set(full_indices.cpu().tolist())) / budget,
            })
    by_objective = max(records, key=lambda record: (record["objective_efficiency"], tuple(-x for x in record["mask"])))
    by_full = max(records, key=lambda record: (record["full_fidelity"], tuple(-x for x in record["mask"])))
    by_oracle = max(records, key=lambda record: (record["oracle_fidelity"], tuple(-x for x in record["mask"])))
    return {
        "query": query,
        "row": row_index,
        "full_reference_kl": reference_kl[0],
        "oracle_reference_kl": reference_kl[1],
        "by_objective": by_objective,
        "by_full": by_full,
        "by_oracle": by_oracle,
        "masks": records,
    }


def mask_lookup(site):
    return {tuple(record["mask"]): record for record in site["masks"]}


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": parent["evaluation_sites_per_query"],
                     "budget": parent["budget"]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(
        model, source_config, parent["queries"],
    )
    layers = parent["suffix_attention_layers"]
    masks = list(itertools.combinations(layers, config["compact_size"]))
    frozen_oracle = json.loads(Path(parent["oracle_results"]).read_text())
    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(parent["evaluation"].items()):
        rows = selected_rows(
            model, source_config, corpus_config, tensor_basis, values, vectors,
            parent["queries"], parent["evaluation_sites_per_query"],
        )
        oracle_indices = {
            (int(record["query"]), int(record["row"])): record["routes"]["oracle"]["indices"]
            for record in frozen_oracle["results"][corpus]["records"]
        }
        records = []
        for number, (query, row_index, raw) in enumerate(rows):
            site = prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)
            records.append(audit_site(
                model, site, query, row_index, masks, layers,
                source_config["causal_position"], parent["budget"],
                source_config["edit_norm_fraction"], oracle_indices,
                config["finite_batch"],
            ))
            print(corpus, "site", number + 1, "of", len(rows), flush=True)
        results[corpus] = {"records": records}

    # First FineWeb site per query is discovery; no evaluated row chooses its own query mask.
    fineweb_by_query = {}
    for query in range(parent["queries"]):
        sites = [site for site in results["fineweb"]["records"] if site["query"] == query]
        fineweb_by_query[query] = {
            selector: tuple(sites[0][selector]["mask"])
            for selector in ("by_objective", "by_full", "by_oracle")
        }

    evaluated = {"fineweb": [], "wikitext": []}
    for corpus, result in results.items():
        for site in result["records"]:
            same_query = [row for row in result["records"] if row["query"] == site["query"]]
            if corpus == "fineweb" and site is same_query[0]:
                continue
            lookup = mask_lookup(site)
            row = {"query": site["query"], "row": site["row"]}
            for selector, mask in fineweb_by_query[site["query"]].items():
                row[selector] = lookup[mask]
            evaluated[corpus].append(row)

    summary = {}
    for corpus_index, corpus in enumerate(results):
        result = results[corpus]["records"]
        summary[corpus] = {
            "sitewise_by_full": summarize([site["by_full"]["full_fidelity"] for site in result], config["seed"] + corpus_index),
            "sitewise_by_objective": summarize([site["by_objective"]["full_fidelity"] for site in result], config["seed"] + 10 + corpus_index),
            "sitewise_by_oracle": summarize([site["by_oracle"]["oracle_fidelity"] for site in result], config["seed"] + 20 + corpus_index),
        }
        for selector in ("by_objective", "by_full", "by_oracle"):
            summary[corpus]["query_frozen_" + selector] = summarize(
                [row[selector]["full_fidelity"] for row in evaluated[corpus]],
                config["seed"] + 100 + corpus_index,
            )

    sitewise = all(summary[c]["sitewise_by_full"]["median"] >= 0.98
                   and summary[c]["sitewise_by_full"]["q10"] >= 0.90 for c in summary)
    recurrent = all(summary[c]["query_frozen_by_full"]["median"] >= 0.95
                    and summary[c]["query_frozen_by_full"]["q10"] >= 0.80 for c in summary)
    broad = any(summary[c]["sitewise_by_full"]["q10"] < 0.80 for c in summary)
    objective = all(
        summary[c]["sitewise_by_full"]["median"] - summary[c]["sitewise_by_objective"]["median"] <= 0.05
        for c in summary
    )
    hypotheses = {
        "A0_sitewise_compact_route_exists": {"supported": bool(sitewise)},
        "A1_query_route_repeats": {"supported": bool(recurrent)},
        "A2_local_route_atlas": {"supported": bool(sitewise and not recurrent)},
        "A3_some_sites_need_broad_suffix": {"supported": bool(broad)},
        "A4_weight_objective_finds_route": {"supported": bool(objective)},
    }
    payload = {"config": config, "hypotheses": hypotheses,
               "query_masks": {str(k): {s: list(m) for s, m in v.items()} for k, v in fineweb_by_query.items()},
               "summary": summary, "evaluated": evaluated, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = ["global\n[9,10,11]", "query-frozen", "sitewise\nhindsight", "sitewise\nobjective"]
    parent_result = json.loads((TENSOR_ROOT / "layer8_suffix_attention_subset.json").read_text())
    for corpus, color in (("fineweb", "#1769aa"), ("wikitext", "#b71c1c")):
        global_values = [row["routes"]["size3"]["effect_fidelity"] for row in parent_result["results"][corpus]["records"]]
        values = [
            global_values,
            [row["by_full"]["full_fidelity"] for row in evaluated[corpus]],
            [site["by_full"]["full_fidelity"] for site in results[corpus]["records"]],
            [site["by_objective"]["full_fidelity"] for site in results[corpus]["records"]],
        ]
        med = [np.median(value) for value in values]
        q10 = [np.quantile(value, .1) for value in values]
        axes[0].plot(range(4), med, marker="o", color=color, label=corpus)
        axes[1].plot(range(4), q10, marker="o", color=color, label=corpus)
    for axis, title in zip(axes, ("Median fidelity", "10th-percentile fidelity")):
        axis.set_xticks(range(4), labels); axis.set_ylim(-.1, 1.08); axis.axhline(.9, color="#777", ls="--", lw=1)
        axis.set_title(title); axis.grid(alpha=.2)
    axes[0].legend(); figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)

    lines = ["# Layer-8 compact suffix-route atlas", "",
             "All 84 three-layer attention-JVP masks were evaluated at every site. Query masks were selected on the first FineWeb site and frozen for the second FineWeb site and both WikiText sites. Sitewise hindsight is an existence ceiling, not a deployable interpreter.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "![Route atlas](layer8_suffix_route_atlas.png)", "",
              "| Corpus | Method | Median fidelity | 10th percentile | Minimum |",
              "|---|---|---:|---:|---:|"]
    for corpus in summary:
        for name, row in summary[corpus].items():
            lines.append(f"| {corpus} | {name} | {row['median']:.3f} | {row['q10']:.3f} | {row['minimum']:.3f} |")
    lines += ["", "## Guardrails", "",
              "A sitewise result uses the answer at that same site and cannot demonstrate a reusable router. Query-frozen transfer is the reusable-mechanism test. Failure of the weight-objective selector can mean finite-scale or selection-objective mismatch; it does not prove no compact mask exists."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
