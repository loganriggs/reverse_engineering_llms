#!/usr/bin/env python3
"""Add exact squared-bilinear-attention JVPs to the Layer-8 suffix interpreter."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_direct_tensor_jacobian import (
    analytic_suffix,
    output_fisher_response,
    selected_tangent,
)
from run_bilin18_layer8_downstream_fisher_oracle import mode_write_tangents
from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_joint_fisher_program import (
    greedy_normalized_joint,
    normalized_joint_objective,
    weighted_responses,
)
from run_bilin18_layer8_router_causal_validation import prefix_to_layer, select_sites, suffix_from_layer
from run_bilin18_layer8_stratified_fisher_oracle import stratified_fisher_sketch
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_full_tensor_jacobian.json")
OUTPUT = TENSOR_ROOT / "layer8_full_tensor_jacobian.json"
REPORT = TENSOR_ROOT / "LAYER8_FULL_TENSOR_JACOBIAN.md"
FIGURE = TENSOR_ROOT / "layer8_full_tensor_jacobian.png"


def summarize(records, routes):
    result = {}
    for route in routes:
        keys = [
            "native_kl", "fidelity_to_oracle", "cosine_to_oracle",
            "heldout_objective_efficiency", "oracle_support_recall",
        ]
        if route in ("full_tensor", "direct_tensor", "residual_only"):
            keys += ["tangent_to_finite_fidelity", "tangent_to_finite_cosine"]
        if route == "full_tensor":
            keys += ["small_tangent_fidelity", "small_tangent_cosine"]
        result[route] = {
            key: float(np.median([record["routes"][route][key] for record in records]))
            for key in keys
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
        block8 = model.blocks[8]
        left = block8.mlp.Left.weight.float() @ tensor_basis
        right = block8.mlp.Right.weight.float() @ tensor_basis
        coefficients = block8.mlp.Down.weight.float().T @ directions
        matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
        matrices = 0.5 * (matrices + matrices.transpose(1, 2))
        values, vectors = ordered_eigensystem(matrices)

    results = {}
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
                    native_write = block8.mlp.Down(
                        block8.mlp.Left(normalized) * block8.mlp.Right(normalized)
                    ) + block8.mlp.Down_bias
                    x8 = pre + native_write
                    x = normalized[0, source_config["causal_position"]].float()
                    mode_tangents, amplitudes = mode_write_tangents(block8, x, mode_directions)
                    analytic_logits, direct_tangents, residual_tangents = analytic_suffix(
                        model, x8, x0, value_bus, source_config["causal_position"],
                        mode_tangents, include_attention=False,
                    )
                    full_logits, full_tangents, _ = analytic_suffix(
                        model, x8, x0, value_bus, source_config["causal_position"],
                        mode_tangents, include_attention=True,
                    )
                if float((analytic_logits - full_logits).abs().max()) > 1e-6:
                    raise AssertionError("native logits differ between analytic variants")
                with torch.inference_mode(False):
                    assessment_x8 = x8.detach().clone()
                    assessment_x0 = x0.detach().clone()
                    assessment_bus = value_bus.detach().clone()
                if row_index not in gradient_cache:
                    seed = config["seed"] + corpus_index * 100000 + int(row_index)
                    gradient_cache[row_index] = stratified_fisher_sketch(
                        model, assessment_x8, assessment_x0, assessment_bus,
                        source_config["causal_position"], source_config["exact_top_tokens"],
                        source_config["tail_samples_per_split"], seed,
                    )[:4]
                first_g, first_w, second_g, second_w = gradient_cache[row_index]
                oracle_response = weighted_responses(first_g, first_w, mode_tangents)
                heldout_response = weighted_responses(second_g, second_w, mode_tangents)
                responses = {
                    "full_tensor": output_fisher_response(full_logits, full_tangents),
                    "direct_tensor": output_fisher_response(analytic_logits, direct_tangents),
                    "residual_only": output_fisher_response(analytic_logits, residual_tangents),
                    "global_full": global_singular[:, None] * (global_vh @ mode_tangents.T),
                }
                local_contribution = (values[query] * (x @ mode_directions).square()).abs()
                route_indices = {
                    "oracle": greedy_normalized_joint(
                        oracle_response, amplitudes, source_config["budget"],
                    ),
                    "local_query": local_contribution.topk(source_config["budget"]).indices,
                }
                for name, response in responses.items():
                    route_indices[name] = greedy_normalized_joint(
                        response, amplitudes, source_config["budget"],
                    )
                with torch.no_grad():
                    native_single = suffix_from_layer(
                        model, 8, x0, value_bus, x8, source_config["causal_position"],
                    )[0]
                    native_logits, route_rows = finite_routes(
                        model, pre, normalized, x0, value_bus,
                        source_config["causal_position"], mode_directions, route_indices,
                        directions[:, query], source_config["edit_norm_fraction"],
                    )
                    small_native, small_rows = finite_routes(
                        model, pre, normalized, x0, value_bus,
                        source_config["causal_position"], mode_directions,
                        {"full_tensor": route_indices["full_tensor"]},
                        directions[:, query], config["tangent_validation_norm_fraction"],
                    )
                if float((native_single - analytic_logits).abs().max()) > 1e-5:
                    raise AssertionError("full analytic native path does not recompose")
                oracle_effect = route_rows["oracle"]["effect"]
                oracle_objective = normalized_joint_objective(
                    heldout_response, amplitudes, route_indices["oracle"],
                )
                oracle_set = set(route_indices["oracle"].cpu().tolist())
                tangent_map = {
                    "full_tensor": full_tangents,
                    "direct_tensor": direct_tangents,
                    "residual_only": residual_tangents,
                }
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
                        "oracle_support_recall": len(set(selected.cpu().tolist()) & oracle_set)
                        / source_config["budget"],
                    })
                    if name in tangent_map:
                        predicted = selected_tangent(
                            tangent_map[name], amplitudes, selected,
                            source_config["edit_norm_fraction"] * x.norm(),
                        )
                        tangent_fidelity, tangent_cosine, _ = fisher_fidelity(
                            native_logits, route_rows[name]["effect"], predicted,
                        )
                        route_rows[name].update({
                            "tangent_to_finite_fidelity": tangent_fidelity,
                            "tangent_to_finite_cosine": tangent_cosine,
                        })
                    route_rows[name].pop("effect")
                small_predicted = selected_tangent(
                    full_tangents, amplitudes, route_indices["full_tensor"],
                    config["tangent_validation_norm_fraction"] * x.norm(),
                )
                small_fidelity, small_cosine, _ = fisher_fidelity(
                    small_native, small_rows["full_tensor"]["effect"], small_predicted,
                )
                route_rows["full_tensor"].update({
                    "small_tangent_fidelity": small_fidelity,
                    "small_tangent_cosine": small_cosine,
                })
                records.append({"query": query, "row": int(row_index), "routes": route_rows})
            print(corpus, "query", query, "complete", flush=True)
        results[corpus] = {"summary": summarize(records, config["routes"]), "records": records}

    def route(corpus, name):
        return results[corpus]["summary"][name]

    hypotheses = {
        "A0_attention_jacobian_passes_small_finite_check": {
            "supported": all(route(c, "full_tensor")["small_tangent_fidelity"] >= 0.99 for c in results),
        },
        "A1_full_tensor_recovers_oracle_direction": {
            "supported": all(route(c, "full_tensor")["fidelity_to_oracle"] >= 0.90 for c in results),
        },
        "A2_full_tensor_recovers_oracle_magnitude": {
            "supported": all(
                route(c, "full_tensor")["native_kl"] >= 0.90 * route(c, "oracle")["native_kl"]
                for c in results
            ),
        },
        "A3_full_tensor_recovers_oracle_objective": {
            "supported": all(route(c, "full_tensor")["heldout_objective_efficiency"] >= 0.90 for c in results),
        },
        "A4_attention_derivatives_close_ood_gap": {
            "supported": route("wikitext", "full_tensor")["fidelity_to_oracle"]
            - route("wikitext", "direct_tensor")["fidelity_to_oracle"] >= 0.20,
        },
        "A5_full_tangent_predicts_five_percent_edit": {
            "supported": all(route(c, "full_tensor")["tangent_to_finite_fidelity"] >= 0.85 for c in results),
        },
        "A6_full_tensor_ood_stability": {
            "supported": abs(
                route("fineweb", "full_tensor")["fidelity_to_oracle"]
                - route("wikitext", "full_tensor")["fidelity_to_oracle"]
            ) <= 0.10,
        },
        "A7_full_tensor_recovers_support": {
            "supported": all(route(c, "full_tensor")["oracle_support_recall"] >= 0.50 for c in results),
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Full analytic tensor Jacobian through Layers 9–17",
        "",
        "This adds the exact final-position JVP of every squared-bilinear attention block to the residual/RMSNorm/bilinear-MLP interpreter. It differentiates QK-normalized `QK × Q2K2` routing, the current-layer value path, the fixed first-layer value bus, and the attention output projection. Causality keeps a final-token perturbation at the final position, so no earlier-position tangent state is required. The final unembedding Fisher metric remains exact over all 50,304 tokens.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Full tensor Jacobian](layer8_full_tensor_jacobian.png)",
        "",
        "| Corpus | Route | Exact KL | KL/oracle | Oracle fidelity/cosine | Objective efficiency | Support recall | Tangent→5% finite | Tangent→0.1% finite |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        oracle = result["summary"]["oracle"]
        for name in config["routes"]:
            row = result["summary"][name]
            tangent = (
                f"{row['tangent_to_finite_fidelity']:.3f}/{row['tangent_to_finite_cosine']:.3f}"
                if name in ("full_tensor", "direct_tensor", "residual_only") else "—"
            )
            small = (
                f"{row['small_tangent_fidelity']:.3f}/{row['small_tangent_cosine']:.3f}"
                if name == "full_tensor" else "—"
            )
            lines.append(
                f"| {corpus} | {name} | {row['native_kl']:.3e} | "
                f"{row['native_kl'] / max(oracle['native_kl'], 1e-30):.3f} | "
                f"{row['fidelity_to_oracle']:.3f}/{row['cosine_to_oracle']:.3f} | "
                f"{row['heldout_objective_efficiency']:.3f} | {row['oracle_support_recall']:.3f} | "
                f"{tangent} | {small} |"
            )
    lines += [
        "",
        "The 0.1% check is a numerical validation of the analytic JVP. The 5% comparison additionally tests whether the tangent-selected program remains faithful under the finite intervention scale used throughout the causal experiments.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
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
        axis.set_xticks(x, names, rotation=20)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
