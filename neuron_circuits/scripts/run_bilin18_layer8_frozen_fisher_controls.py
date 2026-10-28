#!/usr/bin/env python3
"""Compare the frozen suffix-Fisher metric with generic geometric controls."""

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


CONFIG = Path("configs/bilin18_layer8_frozen_fisher_controls.json")
OUTPUT = TENSOR_ROOT / "layer8_frozen_fisher_controls.json"
REPORT = TENSOR_ROOT / "LAYER8_FROZEN_FISHER_CONTROLS.md"
FIGURE = TENSOR_ROOT / "layer8_frozen_fisher_controls.png"


def summarize(records, routes):
    result = {}
    for route in routes:
        result[route] = {
            key: float(np.median([record["routes"][route][key] for record in records]))
            for key in (
                "native_kl", "fidelity_to_oracle", "cosine_to_oracle",
                "heldout_objective_efficiency", "oracle_support_recall",
            )
        }
    return result


def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    arrays = np.load(config["source_arrays"])
    singular = torch.from_numpy(arrays["singular"]).cuda()
    vh = torch.from_numpy(arrays["vh"]).cuda()
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
        _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
        basis = state_vh.T[:, :source_config["dimension"]]
        centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
        _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
        directions = write_vh.T[:, :source_config["queries"]]
        block = model.blocks[8]
        left = block.mlp.Left.weight.float() @ basis
        right = block.mlp.Right.weight.float() @ basis
        coefficients = block.mlp.Down.weight.float().T @ directions
        matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
        matrices = 0.5 * (matrices + matrices.transpose(1, 2))
        values, vectors = ordered_eigensystem(matrices)
        generator = torch.Generator(device="cuda").manual_seed(config["seed"])
        random_matrix = torch.randn(1152, 128, generator=generator, device="cuda")
        random_basis = torch.linalg.qr(random_matrix, mode="reduced").Q.T
        shuffled = torch.randperm(len(singular), generator=generator, device="cuda")

    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        scan_config = {**source_config, "positions": [source_config["causal_position"]]}
        states, _ = collect(model, 8, corpus_config, scan_config)
        selected_sites = select_sites(
            states, basis, values, vectors, source_config["sites_per_query"],
        )
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        records = []
        gradient_cache = {}
        for query, sites in enumerate(selected_sites):
            mode_directions = basis @ vectors[query]
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
                tangent_transpose = mode_tangents.T
                responses = {
                    "global_full": singular[:, None] * (vh @ tangent_transpose),
                    "identity": tangent_transpose,
                    "global128": singular[:128, None] * (vh[:128] @ tangent_transpose),
                    "projector128": vh[:128] @ tangent_transpose,
                    "random128": random_basis @ tangent_transpose,
                    "shuffled_full": singular[shuffled, None] * (vh @ tangent_transpose),
                }
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
                for name in config["routes"]:
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
                        "oracle_support_recall": len(
                            set(selected.cpu().tolist()) & oracle_set
                        ) / source_config["budget"],
                    })
                    route_rows[name].pop("effect")
                records.append({"query": query, "row": int(row_index), "routes": route_rows})
            print(corpus, "query", query, "complete", flush=True)
        results[corpus] = {"summary": summarize(records, config["routes"]), "records": records}

    def route(corpus, name):
        return results[corpus]["summary"][name]

    hypotheses = {
        "C1_global_fisher_beats_identity": {
            "supported": all(
                route(c, "global_full")["fidelity_to_oracle"]
                - route(c, "identity")["fidelity_to_oracle"] >= 0.10
                for c in results
            ),
        },
        "C2_global128_beats_random128": {
            "supported": all(
                route(c, "global128")["fidelity_to_oracle"]
                - route(c, "random128")["fidelity_to_oracle"] >= 0.10
                for c in results
            ),
        },
        "C3_fisher_eigenvalues_add_value": {
            "supported": all(
                route(c, "global128")["fidelity_to_oracle"]
                - route(c, "projector128")["fidelity_to_oracle"] >= 0.05
                for c in results
            ),
        },
        "C4_fisher_direction_weight_pairing_matters": {
            "supported": all(
                route(c, "global_full")["fidelity_to_oracle"]
                - route(c, "shuffled_full")["fidelity_to_oracle"] >= 0.10
                for c in results
            ),
        },
        "C5_global_metric_is_not_only_generic_impact": {
            "supported": all(
                route(c, "global_full")["fidelity_to_oracle"] > route(c, "identity")["fidelity_to_oracle"]
                and route(c, "global_full")["heldout_objective_efficiency"]
                > route(c, "identity")["heldout_objective_efficiency"]
                for c in results
            ),
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Controls for the frozen suffix-Fisher metric",
        "",
        "The failed global quotient is compared with a full Euclidean write metric, its learned rank-128 metric, the same rank-128 subspace without Fisher eigenvalue weights, a random rank-128 subspace, and a full basis with Fisher eigenvalues assigned to the wrong directions. Every metric selects a joint 16-mode program at the same input budget; the per-example Fisher program is used only as the target.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Frozen Fisher controls](layer8_frozen_fisher_controls.png)",
        "",
        "| Corpus | Metric | Exact KL | Oracle fidelity/cosine | Objective efficiency | Support recall |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for name in config["routes"]:
            row = result["summary"][name]
            lines.append(
                f"| {corpus} | {name} | {row['native_kl']:.3e} | "
                f"{row['fidelity_to_oracle']:.3f}/{row['cosine_to_oracle']:.3f} | "
                f"{row['heldout_objective_efficiency']:.3f} | {row['oracle_support_recall']:.3f} |"
            )
    lines += [
        "",
        "A learned metric only earns mechanistic credit here if it predicts the oracle's effect better than geometry-only controls. Large KL alone can come from a generic high-sensitivity direction with the wrong behavioral sign or token pattern.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = config["routes"][1:]
    x = np.arange(len(names))
    width = 0.36
    for corpus_index, (corpus, result) in enumerate(results.items()):
        offset = (corpus_index - 0.5) * width
        axes[0].bar(x + offset, [result["summary"][n]["fidelity_to_oracle"] for n in names], width, label=corpus)
        axes[1].bar(x + offset, [result["summary"][n]["heldout_objective_efficiency"] for n in names], width, label=corpus)
    axes[0].set_title("Effect recovery")
    axes[0].set_ylabel("median Fisher fidelity")
    axes[1].set_title("Objective recovery")
    axes[1].set_ylabel("median efficiency")
    for axis in axes:
        axis.set_xticks(x, names, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
