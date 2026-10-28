#!/usr/bin/env python3
"""Localize the suffix attention JVPs needed to select Layer-8 programs."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_downstream_fisher_oracle import mode_write_tangents
from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_router_causal_validation import prefix_to_layer, select_sites
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_suffix_attention_subset.json")
OUTPUT = TENSOR_ROOT / "layer8_suffix_attention_subset.json"
REPORT = TENSOR_ROOT / "LAYER8_SUFFIX_ATTENTION_SUBSET.md"
FIGURE = TENSOR_ROOT / "layer8_suffix_attention_subset.png"


def mask_name(mask):
    return "none" if not mask else "L" + "_".join(str(layer) for layer in mask)


def greedy_from_gram(gram, amplitudes, budget):
    """Equivalent to greedy_normalized_joint without retaining the response."""
    selected = []
    available = torch.ones(gram.shape[0], dtype=torch.bool, device=gram.device)
    indicator = torch.zeros(gram.shape[0], device=gram.device)
    denominator = torch.zeros((), device=gram.device)
    diagonal = gram.diagonal()
    for _ in range(budget):
        cross = gram @ indicator
        current = indicator @ cross
        score = (current + 2 * cross + diagonal) / (
            denominator + amplitudes.square()
        ).clamp_min(1e-30)
        score = score.masked_fill(~available, -torch.inf)
        chosen = int(score.argmax())
        selected.append(chosen)
        indicator[chosen] = 1
        available[chosen] = False
        denominator = denominator + amplitudes[chosen].square()
    return torch.as_tensor(selected, device=gram.device, dtype=torch.long)


def gram_objective(gram, amplitudes, indices):
    indices = torch.as_tensor(indices, device=gram.device, dtype=torch.long)
    return float(
        gram[indices][:, indices].sum()
        / amplitudes[indices].square().sum().clamp_min(1e-30)
    )


def bootstrap_median_interval(values, seed, draws=2000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def tensor_program(model, source_config, queries):
    tensor_config = {**source_config, "positions": source_config["discovery_positions"]}
    discovery_x, discovery_write = collect(
        model, 8, source_config["tensor_discovery"], tensor_config,
    )
    _, _, tensor_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    tensor_basis = tensor_vh.T[:, :source_config["dimension"]]
    centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
    _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
    directions = write_vh.T[:, :queries]
    block = model.blocks[8]
    left = block.mlp.Left.weight.float() @ tensor_basis
    right = block.mlp.Right.weight.float() @ tensor_basis
    coefficients = block.mlp.Down.weight.float().T @ directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    values, vectors = ordered_eigensystem(matrices)
    return tensor_basis, directions, values, vectors


def selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                  queries, sites_per_query):
    scan = {**source_config, "positions": [source_config["causal_position"]]}
    states, _ = collect(model, 8, corpus_config, scan)
    sites = select_sites(states, tensor_basis, values, vectors, sites_per_query)
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
    return [(query, int(row_index), rows[row_index])
            for query in range(queries) for row_index in sites[query]]


@torch.inference_mode()
def prepare_site(model, raw_row, query, source_config, tensor_basis, directions, vectors):
    position = source_config["causal_position"]
    ids = torch.as_tensor(
        np.asarray(raw_row[:position + 1], dtype=np.int64), device="cuda",
    )[None]
    x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
    block = model.blocks[8]
    native_write = block.mlp.Down(
        block.mlp.Left(normalized) * block.mlp.Right(normalized)
    ) + block.mlp.Down_bias
    x8 = pre + native_write
    x = normalized[0, position].float()
    mode_directions = tensor_basis @ vectors[query]
    tangents, amplitudes = mode_write_tangents(block, x, mode_directions)
    return {
        "x0": x0, "value_bus": value_bus, "pre": pre, "normalized": normalized,
        "x8": x8, "x": x, "mode_directions": mode_directions,
        "tangents": tangents, "amplitudes": amplitudes,
        "output_direction": directions[:, query],
    }


@torch.inference_mode()
def masked_gram(model, site, position, mask):
    logits, tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=True, attention_layers=mask,
    )
    response = output_fisher_response(logits, tangents)
    return logits, tangents, response.T @ response


def discover_masks(model, sites, layers, position, maximum_size, budget):
    masks = [tuple(combo) for size in range(maximum_size + 1)
             for combo in itertools.combinations(layers, size)]
    scores = {mask: [] for mask in masks}
    endpoint_errors = []
    weak_checks = []
    for site_number, site in enumerate(sites):
        full_logits, full_tangents, full_gram = masked_gram(model, site, position, layers)
        direct_logits, direct_tangents, _ = analytic_suffix(
            model, site["x8"], site["x0"], site["value_bus"], position,
            site["tangents"], include_attention=False,
        )
        empty_logits, empty_tangents, _ = analytic_suffix(
            model, site["x8"], site["x0"], site["value_bus"], position,
            site["tangents"], include_attention=True, attention_layers=(),
        )
        legacy_full_logits, legacy_full_tangents, _ = analytic_suffix(
            model, site["x8"], site["x0"], site["value_bus"], position,
            site["tangents"], include_attention=True,
        )
        endpoint_errors.append({
            "native_logits": float((full_logits - direct_logits).abs().max()),
            "empty": float((empty_tangents - direct_tangents).abs().max()),
            "full": float((full_tangents - legacy_full_tangents).abs().max()),
        })
        delta = full_tangents - direct_tangents
        weak = direct_tangents + 0.1 * delta
        weak_checks.append({
            "nonzero": float((weak - direct_tangents).norm()),
            "relative_distance": float(
                (weak - direct_tangents).norm() / delta.norm().clamp_min(1e-30)
            ),
        })
        full_indices = greedy_from_gram(full_gram, site["amplitudes"], budget)
        full_objective = gram_objective(full_gram, site["amplitudes"], full_indices)
        for mask in masks:
            if mask == tuple(layers):
                gram = full_gram
            else:
                _, _, gram = masked_gram(model, site, position, mask)
            indices = greedy_from_gram(gram, site["amplitudes"], budget)
            scores[mask].append(
                gram_objective(full_gram, site["amplitudes"], indices)
                / max(full_objective, 1e-30)
            )
        print("discovery site", site_number + 1, "of", len(sites), flush=True)
    best = {}
    for size in range(maximum_size + 1):
        candidates = [mask for mask in masks if len(mask) == size]
        best[size] = max(candidates, key=lambda mask: (np.median(scores[mask]), tuple(-x for x in mask)))
    order = list(best[maximum_size])
    current = set(order)
    while len(current) < len(layers):
        candidates = [tuple(sorted(current | {layer})) for layer in layers if layer not in current]
        candidate_scores = {}
        for mask in candidates:
            per_site = []
            for site in sites:
                _, _, full_gram = masked_gram(model, site, position, layers)
                full_indices = greedy_from_gram(full_gram, site["amplitudes"], budget)
                full_objective = gram_objective(full_gram, site["amplitudes"], full_indices)
                _, _, gram = masked_gram(model, site, position, mask)
                indices = greedy_from_gram(gram, site["amplitudes"], budget)
                per_site.append(gram_objective(full_gram, site["amplitudes"], indices)
                                / max(full_objective, 1e-30))
            candidate_scores[mask] = per_site
        chosen = max(candidates, key=lambda mask: (np.median(candidate_scores[mask]),
                                                    tuple(-x for x in mask)))
        current = set(chosen)
        order.append(next(layer for layer in chosen if layer not in order))
        scores[chosen] = candidate_scores[chosen]
    return {
        "best_by_size": {str(size): list(mask) for size, mask in best.items()},
        "order": order,
        "median_objective": {mask_name(mask): float(np.median(values))
                             for mask, values in scores.items()},
        "endpoint_errors": endpoint_errors,
        "weak_checks": weak_checks,
    }


def random_size_three_masks(layers, selected, count, seed):
    candidates = [mask for mask in itertools.combinations(layers, 3) if tuple(mask) != tuple(selected)]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    return [candidates[int(index)] for index in chosen]


@torch.inference_mode()
def evaluate_corpus(model, selected, corpus_sites, layers, position, budget,
                    edit_norm_fraction, random_masks, seed, oracle_indices):
    order = selected["order"]
    masks = {f"size{size}": tuple(sorted(order[:size])) for size in range(len(layers) + 1)}
    for index, mask in enumerate(random_masks):
        masks[f"random3_{index}"] = tuple(mask)
    records = []
    for site_number, (query, row_index, site) in enumerate(corpus_sites):
        full_logits, full_tangents, full_gram = masked_gram(model, site, position, layers)
        full_indices = greedy_from_gram(full_gram, site["amplitudes"], budget)
        full_objective = gram_objective(full_gram, site["amplitudes"], full_indices)
        route_indices = {"full_reference": full_indices}
        oracle_key = (query, row_index)
        if oracle_key not in oracle_indices:
            raise KeyError(f"missing frozen oracle indices for {oracle_key}")
        route_indices["oracle_backprop"] = torch.as_tensor(
            oracle_indices[oracle_key], device=full_indices.device, dtype=torch.long,
        )
        row = {"query": query, "row": row_index, "routes": {}}
        for name, mask in masks.items():
            logits, _, gram = masked_gram(model, site, position, mask)
            if float((logits - full_logits).abs().max()) > 1e-6:
                raise AssertionError("derivative mask changed native logits")
            indices = greedy_from_gram(gram, site["amplitudes"], budget)
            route_indices[name] = indices
            row["routes"][name] = {
                "mask": list(mask),
                "objective_efficiency": gram_objective(
                    full_gram, site["amplitudes"], indices,
                ) / max(full_objective, 1e-30),
                "support_recall": len(set(indices.cpu().tolist()) & set(full_indices.cpu().tolist()))
                / budget,
            }
        native, finite = finite_routes(
            model, site["pre"], site["normalized"], site["x0"], site["value_bus"],
            position, site["mode_directions"], route_indices, site["output_direction"],
            edit_norm_fraction,
        )
        reference = finite["full_reference"]["effect"]
        oracle_reference = finite["oracle_backprop"]["effect"]
        for name in masks:
            fidelity, cosine, _ = fisher_fidelity(native, reference, finite[name]["effect"])
            oracle_fidelity, oracle_cosine, _ = fisher_fidelity(
                native, oracle_reference, finite[name]["effect"],
            )
            row["routes"][name].update({
                "effect_fidelity": fidelity,
                "effect_cosine": cosine,
                "oracle_effect_fidelity": oracle_fidelity,
                "oracle_effect_cosine": oracle_cosine,
                "native_kl": finite[name]["native_kl"],
                "indices": finite[name]["indices"],
            })
        records.append(row)
        print("evaluation site", site_number + 1, "of", len(corpus_sites), flush=True)
    summary = {}
    for name in masks:
        summary[name] = {}
        for metric in ("effect_fidelity", "effect_cosine", "oracle_effect_fidelity",
                       "oracle_effect_cosine", "objective_efficiency",
                       "support_recall", "native_kl"):
            values = [record["routes"][name][metric] for record in records]
            summary[name][metric] = float(np.median(values))
            summary[name][metric + "_ci95"] = bootstrap_median_interval(
                values, seed + sum(ord(char) for char in name + metric),
            )
    return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    source_config = {**source_config, "queries": config["queries"],
                     "sites_per_query": config["evaluation_sites_per_query"],
                     "budget": config["budget"]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(
        model, source_config, config["queries"],
    )
    discovery_rows = selected_rows(
        model, source_config, config["discovery"], tensor_basis, values, vectors,
        config["discovery_queries"], config["discovery_sites_per_query"],
    )
    discovery_sites = [prepare_site(
        model, raw, query, source_config, tensor_basis, directions, vectors,
    ) for query, _, raw in discovery_rows]
    layers = config["suffix_attention_layers"]
    position = source_config["causal_position"]
    if args.benchmark:
        torch.cuda.synchronize(); start = time.perf_counter()
        masked_gram(model, discovery_sites[0], position, layers[:3])
        torch.cuda.synchronize(); elapsed = time.perf_counter() - start
        print(json.dumps({"one_mask_seconds": elapsed,
                          "estimated_discovery_mask_calls": 130 * len(discovery_sites)}, indent=2))
        return
    discovery = discover_masks(
        model, discovery_sites, layers, position, config["maximum_compact_size"],
        config["budget"],
    )
    selected3 = discovery["best_by_size"]["3"]
    random_masks = random_size_three_masks(
        layers, selected3, config["random_control_count"], config["seed"],
    )
    frozen_oracle = json.loads(Path(config["oracle_results"]).read_text())
    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        rows = selected_rows(
            model, source_config, corpus_config, tensor_basis, values, vectors,
            config["queries"], config["evaluation_sites_per_query"],
        )
        sites = [(query, row_index, prepare_site(
            model, raw, query, source_config, tensor_basis, directions, vectors,
        )) for query, row_index, raw in rows]
        oracle_indices = {
            (int(record["query"]), int(record["row"])): record["routes"]["oracle"]["indices"]
            for record in frozen_oracle["results"][corpus]["records"]
        }
        results[corpus] = evaluate_corpus(
            model, discovery, sites, layers, position, config["budget"],
            source_config["edit_norm_fraction"], random_masks,
            config["seed"] + 10000 * corpus_index, oracle_indices,
        )
    endpoint_ok = max(max(item.values()) for item in discovery["endpoint_errors"]) <= 1e-6
    weak_ok = all(item["nonzero"] > 0 and abs(item["relative_distance"] - 0.1) <= 1e-5
                  for item in discovery["weak_checks"])
    compact = all(results[corpus]["summary"]["size3"][metric] >= 0.90
                  for corpus in results for metric in ("effect_fidelity", "objective_efficiency"))
    full_ok = all(results[corpus]["summary"]["size9"]["effect_fidelity"] >= 0.98
                  for corpus in results)
    crossing = {}
    for corpus in results:
        crossing[corpus] = next((size for size in range(10)
                                 if results[corpus]["summary"][f"size{size}"]["effect_fidelity"] >= 0.90),
                                None)
    broad = (all(results[c]["summary"]["size3"]["effect_fidelity"] < 0.80 for c in results)
             and full_ok and any(value is None or value >= 6 for value in crossing.values()))
    random_wiki = np.median([results["wikitext"]["summary"][f"random3_{i}"]["effect_fidelity"]
                             for i in range(len(random_masks))])
    transfer = results["wikitext"]["summary"]["size3"]["effect_fidelity"] - random_wiki >= 0.05
    robust_compact = all(
        results[corpus]["summary"]["size3"]["oracle_effect_fidelity"] >= 0.90
        and np.quantile(
            [record["routes"]["size3"]["effect_fidelity"]
             for record in results[corpus]["records"]], 0.10,
        ) >= 0.80
        and results[corpus]["summary"]["size3"]["effect_fidelity"] >= np.median([
            results[corpus]["summary"][f"random3_{i}"]["effect_fidelity"]
            for i in range(len(random_masks))
        ])
        for corpus in results
    )
    hypotheses = {
        "C0_endpoints_reproduce": {"supported": bool(endpoint_ok)},
        "C1_compact_reusable_router": {"supported": bool(compact)},
        "C2_broad_suffix_required": {"supported": bool(broad)},
        "C3_selection_transfers_ood": {"supported": bool(transfer)},
        "C4_attention_signal_is_detectable": {"supported": bool(weak_ok)},
        "R1_robust_compact_router_redteam": {"supported": bool(robust_compact)},
    }
    payload = {"config": config, "discovery": discovery, "random_masks": random_masks,
               "crossing_size": crossing, "hypotheses": hypotheses, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for corpus, color in (("fineweb", "#1769aa"), ("wikitext", "#b71c1c")):
        sizes = np.arange(10)
        for axis, metric, ylabel in ((axes[0], "effect_fidelity", "Finite-effect Fisher fidelity"),
                                     (axes[1], "objective_efficiency", "Full-response objective efficiency")):
            y = [results[corpus]["summary"][f"size{s}"][metric] for s in sizes]
            axis.plot(sizes, y, marker="o", label=corpus, color=color)
            axis.axhline(0.9, color="#777", linestyle="--", linewidth=1)
            axis.set(xlabel="Frozen suffix attention-JVP layers", ylabel=ylabel, ylim=(0, 1.08))
            axis.grid(alpha=.2)
    axes[0].legend(); figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)

    lines = [
        "# Layer-8 suffix attention-subset localization", "",
        "The native forward pass is fixed. Only selected attention derivatives in blocks 9–17 are allowed to carry a Layer-8 perturbation. All subsets of up to three blocks were searched on separate FineWeb discovery rows; the masks and wider cumulative order were then frozen before FineWeb and WikiText evaluation.",
        "", "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Best size-3 discovery mask: `{selected3}`. Frozen order: `{discovery['order']}`.",
              "", "![Suffix attention subset curve](layer8_suffix_attention_subset.png)", "",
              "| Corpus | Layers | Full-reference fidelity (95% bootstrap interval) | Backprop-oracle fidelity | Objective efficiency | Support recall |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus in results:
        for size in range(10):
            row = results[corpus]["summary"][f"size{size}"]
            ci = row["effect_fidelity_ci95"]
            lines.append(f"| {corpus} | {size} | {row['effect_fidelity']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {row['oracle_effect_fidelity']:.3f} | {row['objective_efficiency']:.3f} | {row['support_recall']:.3f} |")
    lines += ["", "## Interpretation guardrails", "",
              "This is a derivative-route localization, not an allocation of stored information. A compact mask would show that a few later attention Jacobians are sufficient to choose the same Layer-8 intervention program. A broad curve would instead favor a distributed suffix route. Neither result alone establishes whether the selected modes have semantic meanings.",
              "", "The empty/full endpoint checks, exhaustive compact search, held-out corpora, random masks, weak-signal control, exact finite interventions, and site bootstrap intervals are retained in the JSON output."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
