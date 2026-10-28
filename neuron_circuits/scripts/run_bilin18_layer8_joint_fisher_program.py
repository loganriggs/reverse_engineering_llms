#!/usr/bin/env python3
"""Select a coherent 16-mode program using the full downstream Fisher Gram."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_downstream_fisher_oracle import mode_write_tangents
from run_bilin18_layer8_router_causal_validation import (
    exact_kl,
    prefix_to_layer,
    repeat_batch,
    select_sites,
    suffix_from_layer,
)
from run_bilin18_layer8_stratified_fisher_oracle import stratified_fisher_sketch
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_joint_fisher_program.json")
OUTPUT = TENSOR_ROOT / "layer8_joint_fisher_program.json"
REPORT = TENSOR_ROOT / "LAYER8_JOINT_FISHER_PROGRAM.md"
FIGURE = TENSOR_ROOT / "layer8_joint_fisher_program.png"


def weighted_responses(gradients, weights, mode_write_tangents):
    return weights.sqrt()[:, None] * (gradients @ mode_write_tangents.T)


def normalized_joint_objective(responses, amplitudes, indices):
    indices = torch.as_tensor(indices, device=responses.device, dtype=torch.long)
    numerator = responses[:, indices].sum(-1).square().sum()
    denominator = amplitudes[indices].square().sum().clamp_min(1e-30)
    return float(numerator / denominator)


def coherence_ratio(responses, indices):
    indices = torch.as_tensor(indices, device=responses.device, dtype=torch.long)
    chosen = responses[:, indices]
    return float(chosen.sum(-1).square().sum() / chosen.square().sum().clamp_min(1e-30))


def cosine(first, second):
    return float(
        (first * second).sum()
        / (first.norm() * second.norm()).clamp_min(1e-30)
    )


def greedy_normalized_joint(responses, amplitudes, budget):
    selected = []
    available = torch.ones(responses.shape[1], dtype=torch.bool, device=responses.device)
    current = torch.zeros(responses.shape[0], device=responses.device)
    denominator = torch.zeros((), device=responses.device)
    column_energy = responses.square().sum(0)
    for _ in range(budget):
        numerator = (
            current.square().sum()
            + 2 * (current[:, None] * responses).sum(0)
            + column_energy
        )
        score = numerator / (denominator + amplitudes.square()).clamp_min(1e-30)
        score = score.masked_fill(~available, -torch.inf)
        chosen = int(score.argmax())
        selected.append(chosen)
        available[chosen] = False
        current = current + responses[:, chosen]
        denominator = denominator + amplitudes[chosen].square()
    return torch.as_tensor(selected, device=responses.device, dtype=torch.long)


@torch.inference_mode()
def finite_routes(model, pre, normalized, x0, value_bus, position, mode_directions,
                  indices, output_direction, edit_norm_fraction):
    names = list(indices)
    block = model.blocks[8]
    x = normalized[0, position].float()
    variants = normalized.repeat(len(names) + 1, 1, 1)
    scale_factors = {}
    raw_norm_fractions = {}
    for variant, name in enumerate(names, start=1):
        directions = mode_directions[:, indices[name]]
        raw_delta = -(x @ directions) @ directions.T
        target = edit_norm_fraction * x.norm()
        scale = target / raw_delta.norm().clamp_min(1e-20)
        variants[variant, position] = (x + scale * raw_delta).to(variants.dtype)
        scale_factors[name] = float(scale)
        raw_norm_fractions[name] = float(raw_delta.norm() / x.norm())
    products = block.mlp.Left(variants) * block.mlp.Right(variants)
    writes = block.mlp.Down(products) + block.mlp.Down_bias
    outputs = repeat_batch(pre, len(variants)) + writes
    logits = suffix_from_layer(
        model, 8, repeat_batch(x0, len(variants)), repeat_batch(value_bus, len(variants)),
        outputs, position,
    )
    native_logits = logits[0]
    native_write = writes[0, position].float()
    result = {}
    for variant, name in enumerate(names, start=1):
        result[name] = {
            "native_kl": exact_kl(native_logits, logits[variant]),
            "absolute_direct_query_effect": abs(float(
                (writes[variant, position].float() - native_write) @ output_direction
            )),
            "scale_factor": scale_factors[name],
            "raw_projection_norm_fraction": raw_norm_fractions[name],
            "indices": indices[name].cpu().tolist(),
        }
    return result


def summarize(records, route_names):
    routes = {}
    for name in route_names:
        routes[name] = {
            key: float(np.median([record["routes"][name][key] for record in records]))
            for key in (
                "native_kl", "absolute_direct_query_effect", "scale_factor",
                "raw_projection_norm_fraction", "heldout_normalized_objective",
                "heldout_coherence_ratio",
                "local_support_recall",
            )
        }
    return {
        "routes": routes,
        "median_greedy_support_recall": float(np.median([
            record["greedy_support_recall"] for record in records
        ])),
        "median_greedy_heldout_efficiency": float(np.median([
            record["greedy_heldout_efficiency"] for record in records
        ])),
        "median_greedy_effect_cosine": float(np.median([
            record["greedy_effect_cosine"] for record in records
        ])),
        "median_greedy_input_cosine": float(np.median([
            record["greedy_input_cosine"] for record in records
        ])),
    }


def evaluate_corpus(model, config, corpus_config, basis, directions, values, vectors, corpus_index):
    scan_config = {**config, "positions": [config["causal_position"]]}
    states, _ = collect(model, 8, corpus_config, scan_config)
    selected_sites = select_sites(states, basis, values, vectors, config["sites_per_query"])
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
    block = model.blocks[8]
    position = config["causal_position"]
    records = []
    for query, sites in enumerate(selected_sites):
        mode_directions = basis @ vectors[query]
        for site_number, row_index in enumerate(sites):
            ids = torch.as_tensor(
                np.asarray(rows[row_index, :position + 1], dtype=np.int64), device="cuda",
            )[None]
            with torch.no_grad():
                x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
                native_write = block.mlp.Down(
                    block.mlp.Left(normalized) * block.mlp.Right(normalized)
                ) + block.mlp.Down_bias
                base_state = (pre + native_write).detach()
                x = normalized[0, position].float()
                mode_tangents, amplitudes = mode_write_tangents(block, x, mode_directions)
            seed = config["seed"] + corpus_index * 100000 + query * 100 + site_number
            first_g, first_w, second_g, second_w, _ = stratified_fisher_sketch(
                model, base_state, x0.detach(), value_bus.detach(), position,
                config["exact_top_tokens"], config["tail_samples_per_split"], seed,
            )
            first_response = weighted_responses(first_g, first_w, mode_tangents)
            second_response = weighted_responses(second_g, second_w, mode_tangents)
            first_energy = first_response.square().sum(0)
            local_contribution = (values[query] * (x @ mode_directions).square()).abs()
            indices = {
                "local_oracle": local_contribution.topk(config["budget"]).indices,
                "fisher_diagonal": first_energy.topk(config["budget"]).indices,
                "fisher_unit": (first_energy / amplitudes.square().clamp_min(1e-30)).topk(config["budget"]).indices,
                "fisher_joint_greedy": greedy_normalized_joint(first_response, amplitudes, config["budget"]),
            }
            second_greedy = greedy_normalized_joint(second_response, amplitudes, config["budget"])
            with torch.no_grad():
                route_rows = finite_routes(
                    model, pre, normalized, x0, value_bus, position, mode_directions,
                    indices, directions[:, query], config["edit_norm_fraction"],
                )
            for name in config["routes"]:
                route_rows[name]["heldout_normalized_objective"] = normalized_joint_objective(
                    second_response, amplitudes, indices[name],
                )
                route_rows[name]["heldout_coherence_ratio"] = coherence_ratio(
                    second_response, indices[name],
                )
                route_rows[name]["local_support_recall"] = len(
                    set(indices[name].cpu().tolist())
                    & set(indices["local_oracle"].cpu().tolist())
                ) / config["budget"]
            first_set = set(indices["fisher_joint_greedy"].cpu().tolist())
            second_set = set(second_greedy.cpu().tolist())
            heldout = route_rows["fisher_joint_greedy"]["heldout_normalized_objective"]
            optimum = normalized_joint_objective(second_response, amplitudes, second_greedy)
            first_effect = second_response[:, indices["fisher_joint_greedy"]].sum(-1)
            second_effect = second_response[:, second_greedy].sum(-1)
            first_input = (
                mode_directions[:, indices["fisher_joint_greedy"]]
                * amplitudes[indices["fisher_joint_greedy"]][None]
            ).sum(-1)
            second_input = (
                mode_directions[:, second_greedy] * amplitudes[second_greedy][None]
            ).sum(-1)
            records.append({
                "query": query,
                "row": int(row_index),
                "greedy_support_recall": len(first_set & second_set) / config["budget"],
                "greedy_heldout_efficiency": heldout / max(optimum, 1e-30),
                "greedy_effect_cosine": cosine(first_effect, second_effect),
                "greedy_input_cosine": cosine(first_input, second_input),
                "routes": route_rows,
            })
            del first_g, second_g
        print("query", query, "complete", flush=True)
    return {"summary": summarize(records, config["routes"]), "records": records}


def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    discovery_config = {**config, "positions": config["discovery_positions"]}
    with torch.no_grad():
        discovery_x, discovery_write = collect(model, 8, config["discovery"], discovery_config)
        _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
        basis = state_vh.T[:, :config["dimension"]]
        centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
        _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
        directions = write_vh.T[:, :config["queries"]]
        block = model.blocks[8]
        left = block.mlp.Left.weight.float() @ basis
        right = block.mlp.Right.weight.float() @ basis
        coefficients = block.mlp.Down.weight.float().T @ directions
        matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
        matrices = 0.5 * (matrices + matrices.transpose(1, 2))
        values, vectors = ordered_eigensystem(matrices)
    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        print("joint Fisher", corpus, flush=True)
        results[corpus] = evaluate_corpus(
            model, config, corpus_config, basis, directions, values, vectors, corpus_index,
        )

    def route(corpus, name):
        return results[corpus]["summary"]["routes"][name]

    corpora = list(results)
    hypotheses = {
        "J1_joint_program_transfers_across_fisher_sketches": {
            "supported": all(results[c]["summary"]["median_greedy_heldout_efficiency"] >= 0.80 for c in corpora),
        },
        "J2_joint_program_is_constructive": {
            "supported": all(route(c, "fisher_joint_greedy")["heldout_coherence_ratio"] >= 0.75 for c in corpora),
        },
        "J3_joint_program_beats_diagonal_finite_effect": {
            "supported": all(
                route(c, "fisher_joint_greedy")["native_kl"] >= 2 * route(c, "fisher_diagonal")["native_kl"]
                for c in corpora
            ),
        },
        "J4_joint_program_beats_local_finite_effect": {
            "supported": all(
                route(c, "fisher_joint_greedy")["native_kl"] >= route(c, "local_oracle")["native_kl"]
                for c in corpora
            ),
        },
        "J5_local_program_retains_query_advantage": {
            "supported": all(
                route(c, "local_oracle")["absolute_direct_query_effect"]
                >= 2 * route(c, "fisher_joint_greedy")["absolute_direct_query_effect"]
                for c in corpora
            ),
        },
        "J6_joint_support_is_recurrent": {
            "supported": all(results[c]["summary"]["median_greedy_support_recall"] >= 0.50 for c in corpora),
        },
        "J7_joint_program_does_not_require_extreme_amplification": {
            "supported": all(route(c, "fisher_joint_greedy")["scale_factor"] <= 10 for c in corpora),
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Joint downstream-Fisher 16-mode program",
        "",
        "The diagonal Fisher ranking ignores signed interactions and selected modes that cancel. This experiment greedily maximizes the full sketch-space Fisher energy of the signed joint mode effect per squared input-component norm. It compares that program with local query modes, natural-component Fisher diagonals, and per-unit-input Fisher diagonals. Selection uses one stratified sketch; all reported objectives use an independent sketch, and final KL is an exact finite forward pass at matched 5% input norm.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Joint Fisher program](layer8_joint_fisher_program.png)",
        "",
        "| Corpus | Route | Exact KL | |query effect| | Held-out normalized Fisher objective | Held-out coherence | Local support | Raw projection norm | Applied scale |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for name in config["routes"]:
            row = result["summary"]["routes"][name]
            lines.append(
                f"| {corpus} | {name} | {row['native_kl']:.3e} | "
                f"{row['absolute_direct_query_effect']:.3e} | {row['heldout_normalized_objective']:.3e} | "
                f"{row['heldout_coherence_ratio']:.3f} | {row['local_support_recall']:.3f} | "
                f"{row['raw_projection_norm_fraction']:.3e} | "
                f"{row['scale_factor']:.2f} |"
            )
        lines.append(
            f"| {corpus} | greedy recurrence | — | — | efficiency "
            f"{result['summary']['median_greedy_heldout_efficiency']:.3f} | support recall "
            f"{result['summary']['median_greedy_support_recall']:.3f}; effect cosine "
            f"{result['summary']['median_greedy_effect_cosine']:.3f}; input cosine "
            f"{result['summary']['median_greedy_input_cosine']:.3f} | — | — | — |"
        )
    lines += [
        "",
        "The greedy program is an intervention-budget optimizer, not a semantic decomposition. A successful joint Fisher program can be useful for finding behaviorally coherent causal directions while remaining a poor explanation of the specified local tensor query.",
        "",
        "Support recurrence, input-direction recurrence, and downstream-effect recurrence answer different identifiability questions. High held-out objective efficiency with low support overlap means several mode sets may implement comparably strong downstream interventions; it does not identify a unique native circuit.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(corpora))
    width = 0.2
    for index, name in enumerate(config["routes"]):
        offset = (index - 1.5) * width
        axes[0].bar(x + offset, [route(c, name)["native_kl"] for c in corpora], width, label=name)
        axes[1].bar(x + offset, [route(c, name)["heldout_coherence_ratio"] for c in corpora], width, label=name)
    axes[0].set_title("Exact matched-norm intervention")
    axes[0].set_ylabel("median native KL")
    axes[1].set_title("Signed held-out coherence")
    axes[1].set_ylabel("joint energy / sum individual energy")
    for axis in axes:
        axis.set_xticks(x, corpora)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
