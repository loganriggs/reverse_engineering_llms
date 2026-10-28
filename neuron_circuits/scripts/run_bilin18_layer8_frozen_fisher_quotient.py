#!/usr/bin/env python3
"""Learn a frozen low-rank suffix-Fisher quotient for Layer-8 tensor modes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_downstream_fisher_oracle import mode_write_tangents
from run_bilin18_layer8_joint_fisher_program import (
    greedy_normalized_joint,
    normalized_joint_objective,
    weighted_responses,
)
from run_bilin18_layer8_router_causal_validation import (
    exact_kl,
    prefix_to_layer,
    repeat_batch,
    select_sites,
    suffix_from_layer,
)
from run_bilin18_layer8_stratified_fisher_oracle import stratified_fisher_sketch
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_frozen_fisher_quotient.json")
OUTPUT = TENSOR_ROOT / "layer8_frozen_fisher_quotient.json"
ARRAYS = TENSOR_ROOT / "layer8_frozen_fisher_quotient.npz"
REPORT = TENSOR_ROOT / "LAYER8_FROZEN_FISHER_QUOTIENT.md"
FIGURE = TENSOR_ROOT / "layer8_frozen_fisher_quotient.png"


def capture_fraction(weighted_gradients, basis):
    total = weighted_gradients.square().sum().clamp_min(1e-30)
    captured = (weighted_gradients @ basis.T).square().sum()
    return float(captured / total)


@torch.inference_mode()
def finite_routes(model, pre, normalized, x0, value_bus, position, mode_directions,
                  route_indices, output_direction, edit_norm_fraction):
    names = list(route_indices)
    block = model.blocks[8]
    x = normalized[0, position].float()
    variants = normalized.repeat(len(names) + 1, 1, 1)
    for variant, name in enumerate(names, start=1):
        directions = mode_directions[:, route_indices[name]]
        raw_delta = -(x @ directions) @ directions.T
        delta = raw_delta * (
            edit_norm_fraction * x.norm() / raw_delta.norm().clamp_min(1e-20)
        )
        variants[variant, position] = (x + delta).to(variants.dtype)
    products = block.mlp.Left(variants) * block.mlp.Right(variants)
    writes = block.mlp.Down(products) + block.mlp.Down_bias
    outputs = repeat_batch(pre, len(variants)) + writes
    logits = suffix_from_layer(
        model, 8, repeat_batch(x0, len(variants)), repeat_batch(value_bus, len(variants)),
        outputs, position,
    )
    native = logits[0]
    native_write = writes[0, position].float()
    result = {}
    for variant, name in enumerate(names, start=1):
        result[name] = {
            "native_kl": exact_kl(native, logits[variant]),
            "absolute_direct_query_effect": abs(float(
                (writes[variant, position].float() - native_write) @ output_direction
            )),
            "effect": logits[variant] - native,
            "indices": route_indices[name].cpu().tolist(),
        }
    return native, result


def summarize(records, route_names, ranks):
    routes = {}
    for name in route_names:
        routes[name] = {
            key: float(np.median([record["routes"][name][key] for record in records]))
            for key in (
                "native_kl", "absolute_direct_query_effect",
                "fidelity_to_oracle", "cosine_to_oracle",
                "heldout_objective_efficiency", "oracle_support_recall",
            )
        }
    capture = {
        str(rank): float(np.median([record["capture_fraction"][str(rank)] for record in records]))
        for rank in ranks
    }
    return {"routes": routes, "capture_fraction": capture}


def collect_frozen_metric(model, config):
    source_config = config["fisher_discovery"]
    source = np.load(source_config["path"], mmap_mode="r")
    rows = source[source_config["start"]:source_config["start"] + source_config["rows"]]
    block = model.blocks[8]
    position = config["causal_position"]
    weighted_rows = []
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
            config["exact_top_tokens"], config["tail_samples_per_split"],
            config["seed"] + index,
        )
        weighted_rows.append(first_w.sqrt()[:, None] * first_g)
        metadata.append(sketch_metadata)
        if (index + 1) % 8 == 0:
            print("fisher discovery", index + 1, "/", len(rows), flush=True)
    matrix = torch.cat(weighted_rows) / np.sqrt(len(rows))
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    return singular, vh, metadata


def evaluate_corpus(model, config, corpus_config, basis, directions, values, vectors,
                    quotient_singular, quotient_vh, corpus_index):
    scan_config = {**config, "positions": [config["causal_position"]]}
    states, _ = collect(model, 8, corpus_config, scan_config)
    selected_sites = select_sites(states, basis, values, vectors, config["sites_per_query"])
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
    block = model.blocks[8]
    position = config["causal_position"]
    max_rank = quotient_vh.shape[0]
    rank_labels = [str(rank) for rank in config["quotient_ranks"]] + ["full"]
    route_names = ["oracle"] + [f"frozen_{label}" for label in rank_labels]
    records = []
    gradient_cache = {}
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
            cache_key = int(row_index)
            if cache_key not in gradient_cache:
                seed = config["seed"] + 1000000 + corpus_index * 100000 + cache_key
                first_g, first_w, second_g, second_w, _ = stratified_fisher_sketch(
                    model, base_state, x0.detach(), value_bus.detach(), position,
                    config["exact_top_tokens"], config["tail_samples_per_split"], seed,
                )
                gradient_cache[cache_key] = (first_g, first_w, second_g, second_w)
            first_g, first_w, second_g, second_w = gradient_cache[cache_key]
            oracle_response = weighted_responses(first_g, first_w, mode_tangents)
            heldout_response = weighted_responses(second_g, second_w, mode_tangents)
            oracle_indices = greedy_normalized_joint(
                oracle_response, amplitudes, config["budget"],
            )
            route_indices = {"oracle": oracle_indices}
            capture = {}
            weighted_second = second_w.sqrt()[:, None] * second_g
            for label in rank_labels:
                rank = max_rank if label == "full" else int(label)
                metric_response = (
                    quotient_singular[:rank, None]
                    * (quotient_vh[:rank] @ mode_tangents.T)
                )
                route_indices[f"frozen_{label}"] = greedy_normalized_joint(
                    metric_response, amplitudes, config["budget"],
                )
                capture[label] = capture_fraction(weighted_second, quotient_vh[:rank])
            with torch.no_grad():
                native_logits, route_rows = finite_routes(
                    model, pre, normalized, x0, value_bus, position, mode_directions,
                    route_indices, directions[:, query], config["edit_norm_fraction"],
                )
            oracle_effect = route_rows["oracle"]["effect"]
            heldout_oracle_objective = normalized_joint_objective(
                heldout_response, amplitudes, oracle_indices,
            )
            oracle_set = set(oracle_indices.cpu().tolist())
            for name in route_names:
                fidelity, cosine, _ = fisher_fidelity(
                    native_logits, oracle_effect, route_rows[name]["effect"],
                )
                selected = torch.as_tensor(route_rows[name]["indices"], device="cuda")
                route_rows[name].update({
                    "fidelity_to_oracle": fidelity,
                    "cosine_to_oracle": cosine,
                    "heldout_objective_efficiency": normalized_joint_objective(
                        heldout_response, amplitudes, selected,
                    ) / max(heldout_oracle_objective, 1e-30),
                    "oracle_support_recall": len(set(selected.cpu().tolist()) & oracle_set)
                    / config["budget"],
                })
                route_rows[name].pop("effect")
            records.append({
                "query": query,
                "row": int(row_index),
                "capture_fraction": capture,
                "routes": route_rows,
            })
        print("query", query, "complete", flush=True)
    return {
        "summary": summarize(records, route_names, rank_labels),
        "records": records,
    }


def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    tensor_config = {**config, "positions": config["discovery_positions"]}
    with torch.no_grad():
        discovery_x, discovery_write = collect(
            model, 8, config["tensor_discovery"], tensor_config,
        )
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
    quotient_singular, quotient_vh, discovery_metadata = collect_frozen_metric(model, config)
    np.savez_compressed(
        ARRAYS,
        singular=quotient_singular.cpu().numpy(),
        vh=quotient_vh.cpu().numpy(),
    )
    spectrum_energy = quotient_singular.square()
    spectrum = {
        str(rank): float(spectrum_energy[:rank].sum() / spectrum_energy.sum())
        for rank in config["quotient_ranks"]
    }
    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        print("frozen quotient", corpus, flush=True)
        results[corpus] = evaluate_corpus(
            model, config, corpus_config, basis, directions, values, vectors,
            quotient_singular, quotient_vh, corpus_index,
        )

    def route(corpus, rank):
        return results[corpus]["summary"]["routes"][f"frozen_{rank}"]

    hypotheses = {
        "Q1_rank64_captures_site_fisher": {
            "supported": all(results[c]["summary"]["capture_fraction"]["64"] >= 0.70 for c in results),
        },
        "Q2_rank64_recovers_half_oracle_effect": {
            "supported": all(
                route(c, "64")["native_kl"]
                >= 0.5 * results[c]["summary"]["routes"]["oracle"]["native_kl"]
                for c in results
            ),
        },
        "Q3_rank64_recovers_oracle_direction": {
            "supported": all(route(c, "64")["fidelity_to_oracle"] >= 0.50 for c in results),
        },
        "Q4_rank128_is_near_full_frozen": {
            "supported": all(
                route(c, "128")["native_kl"] >= 0.8 * route(c, "full")["native_kl"]
                and route(c, "full")["fidelity_to_oracle"] - route(c, "128")["fidelity_to_oracle"] <= 0.10
                for c in results
            ),
        },
        "Q5_full_frozen_recovers_oracle_direction": {
            "supported": all(route(c, "full")["fidelity_to_oracle"] >= 0.70 for c in results),
        },
        "Q6_rank128_ood_fidelity_is_stable": {
            "supported": abs(
                route("fineweb", "128")["fidelity_to_oracle"]
                - route("wikitext", "128")["fidelity_to_oracle"]
            ) <= 0.15,
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "spectrum_energy": spectrum,
        "median_discovery_top_token_mass": float(np.median([
            row["top_probability_mass"] for row in discovery_metadata
        ])),
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# Frozen low-rank suffix-Fisher quotient",
        "",
        "A global suffix metric is learned from 64 disjoint FineWeb final-position states. Each state contributes exact-top-token plus sampled-tail score gradients with probability weights. SVD yields a frozen residual-space Fisher quotient. At evaluation, no suffix gradient is used by the frozen routes: the quotient is contracted with exact Layer-8 bilinear write tangents, and the same joint-Gram greedy selector chooses 16 modes. A per-example gradient oracle is computed only as an assessment target.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Frozen Fisher quotient](layer8_frozen_fisher_quotient.png)",
        "",
        "| Corpus | Rank | Global spectrum energy | Site Fisher capture | Exact KL | KL/oracle | Oracle fidelity/cosine | Objective efficiency | Support recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        oracle = result["summary"]["routes"]["oracle"]
        for label in [str(rank) for rank in config["quotient_ranks"]] + ["full"]:
            row = result["summary"]["routes"][f"frozen_{label}"]
            energy = 1.0 if label == "full" else spectrum[label]
            lines.append(
                f"| {corpus} | {label} | {energy:.3f} | "
                f"{result['summary']['capture_fraction'][label]:.3f} | {row['native_kl']:.3e} | "
                f"{row['native_kl'] / max(oracle['native_kl'], 1e-30):.3f} | "
                f"{row['fidelity_to_oracle']:.3f}/{row['cosine_to_oracle']:.3f} | "
                f"{row['heldout_objective_efficiency']:.3f} | {row['oracle_support_recall']:.3f} |"
            )
        lines.append(
            f"| {corpus} | per-example oracle | — | — | {oracle['native_kl']:.3e} | 1.000 | 1.000/1.000 | 1.000 | 1.000 |"
        )
    lines += [
        "",
        "High exact KL without oracle-effect fidelity would indicate a generic high-impact intervention, not recovery of the same conditional program. Conversely, low native support recall is compatible with success only if objective and finite-effect agreement survive, because the prior experiment showed effect-space equivalence can be more stable than mode identity.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ranks = config["quotient_ranks"] + [quotient_vh.shape[0]]
    labels = [str(rank) for rank in config["quotient_ranks"]] + ["full"]
    for corpus, result in results.items():
        axes[0].plot(ranks, [result["summary"]["capture_fraction"][label] for label in labels], marker="o", label=corpus)
        axes[1].plot(ranks, [result["summary"]["routes"][f"frozen_{label}"]["fidelity_to_oracle"] for label in labels], marker="o", label=corpus)
    axes[0].set_title("Frozen quotient coverage")
    axes[0].set_ylabel("median site Fisher trace captured")
    axes[1].set_title("Conditional-program recovery")
    axes[1].set_ylabel("median Fisher fidelity to oracle effect")
    for axis in axes:
        axis.set_xlabel("quotient rank")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
