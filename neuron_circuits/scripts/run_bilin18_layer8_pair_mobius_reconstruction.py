#!/usr/bin/env python3
"""Reconstruct three-layer suffix responses from exact singleton/pair tensors."""

from __future__ import annotations

import itertools
import json
import gc
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_suffix_attention_subset import (
    gram_objective,
    greedy_from_gram,
    prepare_site,
    selected_rows,
    tensor_program,
)
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_pair_mobius_reconstruction.json")
OUTPUT = TENSOR_ROOT / "layer8_pair_mobius_reconstruction.json"
REPORT = TENSOR_ROOT / "LAYER8_PAIR_MOBIUS_RECONSTRUCTION.md"
CHECKPOINT = TENSOR_ROOT / "layer8_pair_mobius_reconstruction.partial.json"


def response(model, site, position, mask):
    logits, tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=bool(mask), attention_layers=mask,
    )
    return logits, output_fisher_response(logits, tangents)


@torch.inference_mode()
def audit_site(model, site, layers, position, budget, edit_fraction, batch_size):
    logits, empty = response(model, site, position, ())
    single = {}
    for layer in layers:
        singleton_logits, single[layer] = response(model, site, position, (layer,))
        if float((singleton_logits - logits).abs().max()) > 1e-6:
            raise AssertionError("mask changed native logits")
    pair = {}
    for left, right in itertools.combinations(layers, 2):
        _, pair[(left, right)] = response(model, site, position, (left, right))
    triples = list(itertools.combinations(layers, 3))
    records = []
    exact_programs = []
    reconstructed_programs = []
    for triple in triples:
        _, exact = response(model, site, position, triple)
        i, j, k = triple
        reconstructed = (pair[(i, j)] + pair[(i, k)] + pair[(j, k)]
                         - single[i] - single[j] - single[k] + empty)
        exact_norm = exact.norm().clamp_min(1e-30)
        reconstructed_norm = reconstructed.norm().clamp_min(1e-30)
        cosine = float((exact * reconstructed).sum() / (exact_norm * reconstructed_norm))
        relative_error = float((exact - reconstructed).norm() / exact_norm)
        exact_gram = exact.T @ exact
        reconstructed_gram = reconstructed.T @ reconstructed
        amplitude = site["amplitudes"].abs()
        floor = float(amplitude.median()) * 1e-6
        scale = amplitude.clamp_min(max(floor, 1e-30))
        outer = scale[:, None] * scale[None]
        whitened_exact = exact_gram / outer
        whitened_reconstructed = reconstructed_gram / outer
        exact_indices = greedy_from_gram(exact_gram, site["amplitudes"], budget)
        reconstructed_indices = greedy_from_gram(
            reconstructed_gram, site["amplitudes"], budget,
        )
        exact_objective = gram_objective(exact_gram, site["amplitudes"], exact_indices)
        objective_efficiency = gram_objective(
            exact_gram, site["amplitudes"], reconstructed_indices,
        ) / max(exact_objective, 1e-30)
        support = len(set(exact_indices.cpu().tolist()) &
                      set(reconstructed_indices.cpu().tolist())) / budget
        records.append({"mask": list(triple), "response_cosine": cosine,
                        "response_relative_error": relative_error,
                        "whitened_gram_error": float(
                            (whitened_exact - whitened_reconstructed).norm()
                            / whitened_exact.norm().clamp_min(1e-30)
                        ),
                        "objective_efficiency": objective_efficiency,
                        "support_recall": support})
        exact_programs.append(exact_indices)
        reconstructed_programs.append(reconstructed_indices)
        del exact, reconstructed, exact_gram, reconstructed_gram
    for start in range(0, len(triples), batch_size):
        stop = min(start + batch_size, len(triples))
        routes = {}
        for number in range(start, stop):
            routes[f"exact_{number}"] = exact_programs[number]
            routes[f"reconstructed_{number}"] = reconstructed_programs[number]
        native, finite = finite_routes(
            model, site["pre"], site["normalized"], site["x0"], site["value_bus"],
            position, site["mode_directions"], routes, site["output_direction"],
            edit_fraction,
        )
        for number in range(start, stop):
            fidelity, cosine, _ = fisher_fidelity(
                native, finite[f"exact_{number}"]["effect"],
                finite[f"reconstructed_{number}"]["effect"],
            )
            records[number]["finite_effect_fidelity"] = fidelity
            records[number]["finite_effect_cosine"] = cosine
    return records


def aggregate(records, seed):
    result = {}
    for metric in ("response_cosine", "response_relative_error", "objective_efficiency",
                   "support_recall", "finite_effect_fidelity", "finite_effect_cosine"):
        values = [row[metric] for site in records for row in site["triples"]]
        result[metric] = summarize(values, seed)
    return result


def main():
    config = json.loads(CONFIG.read_text())
    synergy_config = json.loads(Path(config["parent_config"]).read_text())
    singleton_config = json.loads(Path(synergy_config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": config["sites_per_query"], "budget": parent["budget"]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    layers = parent["suffix_attention_layers"]
    partial = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {"results": {}}
    results = partial["results"]
    for corpus_index, (corpus, corpus_config) in enumerate(parent["evaluation"].items()):
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], config["sites_per_query"])
        records = results.get(corpus, {}).get("records", [])
        for number, (query, row_index, raw) in enumerate(rows[len(records):], start=len(records)):
            site = prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)
            triples = audit_site(
                model, site, layers, source_config["causal_position"], parent["budget"],
                source_config["edit_norm_fraction"], config["finite_triples_per_batch"],
            )
            records.append({"query": query, "row": row_index, "triples": triples})
            results[corpus] = {"records": records}
            CHECKPOINT.write_text(json.dumps({"results": results}))
            del site, triples
            gc.collect()
            torch.cuda.empty_cache()
            print(corpus, "site", number + 1, "of", len(rows), flush=True)
        results[corpus] = {"summary": aggregate(records, config["seed"] + corpus_index),
                           "records": records}
    response_pass = all(results[c]["summary"]["response_cosine"]["median"] >= .95
                        and results[c]["summary"]["response_cosine"]["q10"] >= .80 for c in results)
    finite_pass = all(results[c]["summary"]["finite_effect_fidelity"]["median"] >= .95
                      and results[c]["summary"]["finite_effect_fidelity"]["q10"] >= .80 for c in results)
    support_pass = all(results[c]["summary"]["support_recall"]["median"] >= .75 for c in results)
    third = any(results[c]["summary"]["finite_effect_fidelity"]["median"] < .80
                or results[c]["summary"]["finite_effect_fidelity"]["q10"] < .50 for c in results)
    hypotheses = {
        "M0_pair_mobius_reconstructs_response": {"supported": bool(response_pass)},
        "M1_pair_mobius_selects_equivalent_program": {"supported": bool(finite_pass)},
        "M2_pair_mobius_recovers_support": {"supported": bool(support_pass)},
        "M3_true_third_order_propagation_is_required": {"supported": bool(third)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2))
    CHECKPOINT.unlink(missing_ok=True)
    lines = ["# Pairwise Mobius reconstruction of three-layer suffix routes", "",
             "Complete singleton and pair softmax-Fisher response tensors reconstruct each triple by second-order inclusion-exclusion. Exact and reconstructed responses independently select 16-mode programs, which are compared by finite full-model interventions.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Response cosine median/q10 | Relative error median | Finite fidelity median/q10 | Support recall median | Objective efficiency median |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, result in results.items():
        row = result["summary"]
        lines.append(f"| {corpus} | {row['response_cosine']['median']:.3f}/{row['response_cosine']['q10']:.3f} | {row['response_relative_error']['median']:.3f} | {row['finite_effect_fidelity']['median']:.3f}/{row['finite_effect_fidelity']['q10']:.3f} | {row['support_recall']['median']:.3f} | {row['objective_efficiency']['median']:.3f} |")
    lines += ["", "This tests how a chosen route composes, not how the model or an interpreter chooses that route. Pairwise reconstruction can pass while the address problem remains unsolved."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
