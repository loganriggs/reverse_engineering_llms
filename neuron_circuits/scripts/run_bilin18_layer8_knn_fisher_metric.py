#!/usr/bin/env python3
"""Retrieve frozen suffix-Fisher metrics from nearby Layer-8 states."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_downstream_fisher_oracle import mode_write_tangents
from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_joint_fisher_program import (
    greedy_normalized_joint,
    normalized_joint_objective,
    weighted_responses,
)
from run_bilin18_layer8_router_causal_validation import prefix_to_layer, select_sites
from run_bilin18_layer8_stratified_fisher_oracle import stratified_fisher_sketch
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_knn_fisher_metric.json")
OUTPUT = TENSOR_ROOT / "layer8_knn_fisher_metric.json"
ARRAYS = TENSOR_ROOT / "layer8_knn_fisher_metric.npz"
REPORT = TENSOR_ROOT / "LAYER8_KNN_FISHER_METRIC.md"
FIGURE = TENSOR_ROOT / "layer8_knn_fisher_metric.png"


def nearest_indices(state, mean, basis, bank_coordinates, count):
    coordinate = (state - mean) @ basis
    coordinate = coordinate / coordinate.norm().clamp_min(1e-30)
    similarity = bank_coordinates @ coordinate
    return similarity.topk(count).indices


def neighbor_responses(gradients, weights, neighbors, mode_tangents):
    selected_gradients = gradients[neighbors]
    selected_weights = weights[neighbors] / len(neighbors)
    weighted = selected_weights.sqrt()[..., None] * selected_gradients
    return weighted.flatten(0, 1) @ mode_tangents.T


def collect_metric_bank(model, config, source_config):
    bank_config = config["metric_discovery"]
    source = np.load(bank_config["path"], mmap_mode="r")
    rows = source[bank_config["start"]:bank_config["start"] + bank_config["rows"]]
    block = model.blocks[8]
    position = source_config["causal_position"]
    states, gradients, weights = [], [], []
    metadata = []
    for index, raw_row in enumerate(rows):
        ids = torch.as_tensor(
            np.asarray(raw_row[:position + 1], dtype=np.int64), device="cuda",
        )[None]
        with torch.no_grad():
            x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
            native_write = block.mlp.Down(
                block.mlp.Left(normalized) * block.mlp.Right(normalized)
            ) + block.mlp.Down_bias
            base_state = (pre + native_write).detach()
        first_g, first_w, _, _, sketch_metadata = stratified_fisher_sketch(
            model, base_state, x0.detach(), value_bus.detach(), position,
            source_config["exact_top_tokens"], source_config["tail_samples_per_split"],
            config["seed"] + index,
        )
        states.append(normalized[0, position].float())
        gradients.append(first_g)
        weights.append(first_w)
        metadata.append(sketch_metadata)
        if (index + 1) % 16 == 0:
            print("metric bank", index + 1, "/", len(rows), flush=True)
    states = torch.stack(states)
    gradients = torch.stack(gradients)
    weights = torch.stack(weights)
    mean = states.mean(0)
    _, _, state_vh = torch.linalg.svd(states - mean, full_matrices=False)
    basis = state_vh.T[:, :config["state_dimension"]]
    coordinates = (states - mean) @ basis
    coordinates = coordinates / coordinates.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return states, gradients, weights, mean, basis, coordinates, metadata


def summarize(records, route_names):
    result = {}
    for name in route_names:
        result[name] = {
            key: float(np.median([record["routes"][name][key] for record in records]))
            for key in (
                "native_kl", "fidelity_to_oracle", "cosine_to_oracle",
                "heldout_objective_efficiency", "oracle_support_recall",
            )
        }
    return result


def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    global_arrays = np.load(config["global_arrays"])
    global_singular = torch.from_numpy(global_arrays["singular"]).cuda()
    global_vh = torch.from_numpy(global_arrays["vh"]).cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    tensor_config = {**source_config, "positions": source_config["discovery_positions"]}
    with torch.no_grad():
        discovery_x, discovery_write = collect(
            model, 8, source_config["tensor_discovery"], tensor_config,
        )
        _, _, tensor_vh = torch.linalg.svd(discovery_x, full_matrices=False)
        tensor_basis = tensor_vh.T[:, :source_config["dimension"]]
        centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
        _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
        directions = write_vh.T[:, :source_config["queries"]]
        block = model.blocks[8]
        left = block.mlp.Left.weight.float() @ tensor_basis
        right = block.mlp.Right.weight.float() @ tensor_basis
        coefficients = block.mlp.Down.weight.float().T @ directions
        matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
        matrices = 0.5 * (matrices + matrices.transpose(1, 2))
        values, vectors = ordered_eigensystem(matrices)
    bank_states, bank_gradients, bank_weights, state_mean, state_basis, bank_coordinates, bank_metadata = collect_metric_bank(
        model, config, source_config,
    )
    np.savez_compressed(
        ARRAYS,
        states=bank_states.cpu().numpy(), gradients=bank_gradients.cpu().numpy(),
        weights=bank_weights.cpu().numpy(), state_mean=state_mean.cpu().numpy(),
        state_basis=state_basis.cpu().numpy(), coordinates=bank_coordinates.cpu().numpy(),
    )

    results = {}
    route_names = ["oracle", "global_full"] + [f"knn_{k}" for k in config["neighbors"]]
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        scan_config = {**source_config, "positions": [source_config["causal_position"]]}
        states, _ = collect(model, 8, corpus_config, scan_config)
        selected_sites = select_sites(
            states, tensor_basis, values, vectors, source_config["sites_per_query"],
        )
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        records = []
        gradient_cache = {}
        for query, sites in enumerate(selected_sites):
            mode_directions = tensor_basis @ vectors[query]
            for site_number, row_index in enumerate(sites):
                ids = torch.as_tensor(
                    np.asarray(rows[row_index, :source_config["causal_position"] + 1], dtype=np.int64),
                    device="cuda",
                )[None]
                with torch.no_grad():
                    x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
                    native_write = block.mlp.Down(
                        block.mlp.Left(normalized) * block.mlp.Right(normalized)
                    ) + block.mlp.Down_bias
                    base_state = (pre + native_write).detach()
                    x = normalized[0, source_config["causal_position"]].float()
                    mode_tangents, amplitudes = mode_write_tangents(block, x, mode_directions)
                if row_index not in gradient_cache:
                    seed = config["seed"] + corpus_index * 100000 + int(row_index)
                    gradient_cache[row_index] = stratified_fisher_sketch(
                        model, base_state, x0.detach(), value_bus.detach(),
                        source_config["causal_position"], source_config["exact_top_tokens"],
                        source_config["tail_samples_per_split"], seed,
                    )[:4]
                first_g, first_w, second_g, second_w = gradient_cache[row_index]
                oracle_response = weighted_responses(first_g, first_w, mode_tangents)
                heldout_response = weighted_responses(second_g, second_w, mode_tangents)
                max_neighbors = max(config["neighbors"])
                ordered_neighbors = nearest_indices(
                    x, state_mean, state_basis, bank_coordinates, max_neighbors,
                )
                tangent_transpose = mode_tangents.T
                responses = {
                    "global_full": global_singular[:, None] * (global_vh @ tangent_transpose),
                }
                for count in config["neighbors"]:
                    responses[f"knn_{count}"] = neighbor_responses(
                        bank_gradients, bank_weights, ordered_neighbors[:count], mode_tangents,
                    )
                route_indices = {
                    "oracle": greedy_normalized_joint(
                        oracle_response, amplitudes, source_config["budget"],
                    ),
                }
                for name, response in responses.items():
                    route_indices[name] = greedy_normalized_joint(
                        response, amplitudes, source_config["budget"],
                    )
                with torch.no_grad():
                    native_logits, route_rows = finite_routes(
                        model, pre, normalized, x0, value_bus,
                        source_config["causal_position"], mode_directions, route_indices,
                        directions[:, query], source_config["edit_norm_fraction"],
                    )
                oracle_effect = route_rows["oracle"]["effect"]
                oracle_objective = normalized_joint_objective(
                    heldout_response, amplitudes, route_indices["oracle"],
                )
                oracle_set = set(route_indices["oracle"].cpu().tolist())
                for name in route_names:
                    fidelity, cosine, _ = fisher_fidelity(
                        native_logits, oracle_effect, route_rows[name]["effect"],
                    )
                    selected = route_indices[name]
                    route_rows[name].update({
                        "fidelity_to_oracle": fidelity,
                        "cosine_to_oracle": cosine,
                        "heldout_objective_efficiency": normalized_joint_objective(
                            heldout_response, amplitudes, selected,
                        ) / max(oracle_objective, 1e-30),
                        "oracle_support_recall": len(set(selected.cpu().tolist()) & oracle_set)
                        / source_config["budget"],
                    })
                    route_rows[name].pop("effect")
                records.append({
                    "query": query, "row": int(row_index),
                    "nearest_similarity": float(
                        bank_coordinates[ordered_neighbors[0]]
                        @ (((x - state_mean) @ state_basis)
                           / ((x - state_mean) @ state_basis).norm().clamp_min(1e-30))
                    ),
                    "routes": route_rows,
                })
            print(corpus, "query", query, "complete", flush=True)
        results[corpus] = {
            "summary": summarize(records, route_names),
            "median_nearest_similarity": float(np.median([r["nearest_similarity"] for r in records])),
            "records": records,
        }

    def route(corpus, name):
        return results[corpus]["summary"][name]

    best_common_k = max(
        config["neighbors"],
        key=lambda k: min(route(c, f"knn_{k}")["fidelity_to_oracle"] for c in results),
    )
    hypotheses = {
        "K1_eight_neighbors_recovers_effect": {
            "supported": all(route(c, "knn_8")["fidelity_to_oracle"] >= 0.60 for c in results),
        },
        "K2_eight_neighbors_beats_global": {
            "supported": all(
                route(c, "knn_8")["fidelity_to_oracle"]
                - route(c, "global_full")["fidelity_to_oracle"] >= 0.15
                for c in results
            ),
        },
        "K3_eight_neighbors_recovers_objective": {
            "supported": all(route(c, "knn_8")["heldout_objective_efficiency"] >= 0.65 for c in results),
        },
        "K4_eight_neighbor_ood_stability": {
            "supported": abs(
                route("fineweb", "knn_8")["fidelity_to_oracle"]
                - route("wikitext", "knn_8")["fidelity_to_oracle"]
            ) <= 0.15,
        },
        "K5_some_frozen_knn_metric_recovers_effect": {
            "supported": any(
                all(route(c, f"knn_{k}")["fidelity_to_oracle"] >= 0.60 for c in results)
                for k in config["neighbors"]
            ),
        },
        "K6_nearest_metric_is_better_than_all_data": {
            "supported": all(
                route(c, "knn_8")["fidelity_to_oracle"] > route(c, "knn_128")["fidelity_to_oracle"]
                for c in results
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "best_common_k_posthoc": best_common_k,
        "median_bank_top_token_mass": float(np.median([
            row["top_probability_mass"] for row in bank_metadata
        ])),
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# State-conditioned KNN suffix-Fisher metric",
        "",
        "A bank of 128 disjoint FineWeb Layer-8 states stores stratified suffix-Fisher gradients. States are centered, projected into a frozen 64D PCA space, and cosine-normalized. At evaluation, the current Layer-8 state retrieves K neighbors; their metrics are pooled and contracted with exact bilinear write tangents. No evaluation-time suffix gradient is used for KNN selection. The per-example gradient program is computed only as the target.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Post-hoc best common K by worst-corpus fidelity: **{best_common_k}**.",
        "",
        "![KNN Fisher metric](layer8_knn_fisher_metric.png)",
        "",
        "| Corpus | Metric | Exact KL | Oracle fidelity/cosine | Objective efficiency | Support recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for name in route_names:
            row = result["summary"][name]
            lines.append(
                f"| {corpus} | {name} | {row['native_kl']:.3e} | "
                f"{row['fidelity_to_oracle']:.3f}/{row['cosine_to_oracle']:.3f} | "
                f"{row['heldout_objective_efficiency']:.3f} | {row['oracle_support_recall']:.3f} |"
            )
        lines.append(
            f"| {corpus} | nearest-state diagnostic | — | cosine {result['median_nearest_similarity']:.3f} | — | — |"
        )
    lines += [
        "",
        "KNN success would show that suffix geometry is locally reusable in Layer-8 state space. Failure would not prove unpredictability: this router uses only Euclidean neighborhood structure in a 64D state PCA and a small FineWeb bank.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for corpus, result in results.items():
        axes[0].plot(config["neighbors"], [route(corpus, f"knn_{k}")["fidelity_to_oracle"] for k in config["neighbors"]], marker="o", label=corpus)
        axes[1].plot(config["neighbors"], [route(corpus, f"knn_{k}")["heldout_objective_efficiency"] for k in config["neighbors"]], marker="o", label=corpus)
    axes[0].set_title("Effect recovery")
    axes[0].set_ylabel("median Fisher fidelity")
    axes[1].set_title("Objective recovery")
    axes[1].set_ylabel("median efficiency")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("neighbors K")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
