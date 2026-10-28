#!/usr/bin/env python3
"""Rank sweep for the exact state-conditioned Layer-8 suffix Fisher Gram."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_downstream_fisher_oracle import mode_write_tangents
from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_joint_fisher_program import (
    greedy_normalized_joint,
    normalized_joint_objective,
)
from run_bilin18_layer8_router_causal_validation import prefix_to_layer, select_sites
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_full_tensor_gram_rank.json")
OUTPUT = TENSOR_ROOT / "layer8_full_tensor_gram_rank.json"
REPORT = TENSOR_ROOT / "LAYER8_FULL_TENSOR_GRAM_RANK.md"
FIGURE = TENSOR_ROOT / "layer8_full_tensor_gram_rank.png"


def gram_eigensystem(response):
    gram = response.T @ response
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    return values[order].clamp_min(0), vectors[:, order]


def low_rank_response(values, vectors, rank):
    return values[:rank].sqrt()[:, None] * vectors[:, :rank].T


def summarize(records, routes, ranks):
    route_summary = {}
    for route in routes:
        route_summary[route] = {
            key: float(np.median([record["routes"][route][key] for record in records]))
            for key in (
                "native_kl", "fidelity_to_full", "cosine_to_full",
                "objective_efficiency", "support_recall",
            )
        }
    energy = {
        str(rank): float(np.median([record["energy_fraction"][str(rank)] for record in records]))
        for rank in ranks
    }
    return {"routes": route_summary, "energy_fraction": energy}


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    tensor_config = {**source_config, "positions": source_config["discovery_positions"]}
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
    for corpus, corpus_config in config["evaluation"].items():
        scan_config = {**source_config, "positions": [source_config["causal_position"]]}
        states, _ = collect(model, 8, corpus_config, scan_config)
        selected_sites = select_sites(
            states, tensor_basis, values, vectors, source_config["sites_per_query"],
        )
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        records = []
        for query, sites in enumerate(selected_sites):
            mode_directions = tensor_basis @ vectors[query]
            for row_index in sites:
                ids = torch.as_tensor(
                    np.asarray(rows[row_index, :source_config["causal_position"] + 1], dtype=np.int64),
                    device="cuda",
                )[None]
                x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
                native_write = block8.mlp.Down(
                    block8.mlp.Left(normalized) * block8.mlp.Right(normalized)
                ) + block8.mlp.Down_bias
                x8 = pre + native_write
                x = normalized[0, source_config["causal_position"]].float()
                mode_tangents, amplitudes = mode_write_tangents(block8, x, mode_directions)
                logits, logit_tangents, _ = analytic_suffix(
                    model, x8, x0, value_bus, source_config["causal_position"],
                    mode_tangents, include_attention=True,
                )
                full_response = output_fisher_response(logits, logit_tangents)
                gram_values, gram_vectors = gram_eigensystem(full_response)
                total_energy = gram_values.sum().clamp_min(1e-30)
                route_indices = {
                    "full": greedy_normalized_joint(
                        full_response, amplitudes, source_config["budget"],
                    ),
                }
                energy_fraction = {}
                for rank in config["ranks"]:
                    rank_response = low_rank_response(gram_values, gram_vectors, rank)
                    route_indices[f"rank{rank}"] = greedy_normalized_joint(
                        rank_response, amplitudes, source_config["budget"],
                    )
                    energy_fraction[str(rank)] = float(gram_values[:rank].sum() / total_energy)
                native_logits, route_rows = finite_routes(
                    model, pre, normalized, x0, value_bus,
                    source_config["causal_position"], mode_directions, route_indices,
                    directions[:, query], source_config["edit_norm_fraction"],
                )
                full_effect = route_rows["full"]["effect"]
                full_objective = normalized_joint_objective(
                    full_response, amplitudes, route_indices["full"],
                )
                full_set = set(route_indices["full"].cpu().tolist())
                for route in config["routes"]:
                    fidelity, cosine, _ = fisher_fidelity(
                        native_logits, full_effect, route_rows[route]["effect"],
                    )
                    selected = route_indices[route]
                    route_rows[route].update({
                        "fidelity_to_full": fidelity,
                        "cosine_to_full": cosine,
                        "objective_efficiency": normalized_joint_objective(
                            full_response, amplitudes, selected,
                        ) / max(full_objective, 1e-30),
                        "support_recall": len(set(selected.cpu().tolist()) & full_set)
                        / source_config["budget"],
                    })
                    route_rows[route].pop("effect")
                records.append({
                    "query": query,
                    "row": int(row_index),
                    "energy_fraction": energy_fraction,
                    "routes": route_rows,
                })
            print(corpus, "query", query, "complete", flush=True)
        results[corpus] = {
            "summary": summarize(records, config["routes"], config["ranks"]),
            "records": records,
        }

    def route(corpus, rank):
        return results[corpus]["summary"]["routes"][f"rank{rank}"]

    hypotheses = {
        "R1_rank16_captures_gram_energy": {
            "supported": all(results[c]["summary"]["energy_fraction"]["16"] >= 0.90 for c in results),
        },
        "R2_rank16_recovers_finite_effect": {
            "supported": all(route(c, 16)["fidelity_to_full"] >= 0.90 for c in results),
        },
        "R3_rank16_recovers_objective": {
            "supported": all(route(c, 16)["objective_efficiency"] >= 0.90 for c in results),
        },
        "R4_rank32_recovers_support": {
            "supported": all(route(c, 32)["support_recall"] >= 0.75 for c in results),
        },
        "R5_rank32_recovers_magnitude": {
            "supported": all(
                route(c, 32)["native_kl"]
                >= 0.90 * results[c]["summary"]["routes"]["full"]["native_kl"]
                for c in results
            ),
        },
        "R6_compact_rank_is_ood_stable": {
            "supported": abs(
                route("fineweb", 32)["fidelity_to_full"]
                - route("wikitext", 32)["fidelity_to_full"]
            ) <= 0.05,
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Rank sweep for the exact state-conditioned Fisher Gram",
        "",
        "For each state and output query, the full analytic tensor Jacobian defines an exact full-vocabulary softmax-Fisher response over 512 Layer-8 modes. Its 512×512 Gram is eigendecomposed, truncated, and used to select a joint 16-mode program. Each truncated program is then evaluated by an exact finite 5%-norm intervention through the full model.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Full tensor Gram rank](layer8_full_tensor_gram_rank.png)",
        "",
        "| Corpus | Rank | Gram energy | Exact KL | KL/full | Effect fidelity/cosine | Objective efficiency | Support recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        full = result["summary"]["routes"]["full"]
        for rank in config["ranks"]:
            row = result["summary"]["routes"][f"rank{rank}"]
            lines.append(
                f"| {corpus} | {rank} | {result['summary']['energy_fraction'][str(rank)]:.3f} | "
                f"{row['native_kl']:.3e} | {row['native_kl'] / max(full['native_kl'], 1e-30):.3f} | "
                f"{row['fidelity_to_full']:.3f}/{row['cosine_to_full']:.3f} | "
                f"{row['objective_efficiency']:.3f} | {row['support_recall']:.3f} |"
            )
        lines.append(
            f"| {corpus} | full | 1.000 | {full['native_kl']:.3e} | 1.000 | 1.000/1.000 | 1.000 | 1.000 |"
        )
    lines += [
        "",
        "Gram energy, selected-program objective, finite behavioral magnitude, effect direction, and native support are different compression targets. A low-rank metric is useful only for the targets it actually preserves.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for corpus, result in results.items():
        axes[0].plot(config["ranks"], [result["summary"]["energy_fraction"][str(r)] for r in config["ranks"]], marker="o", label=corpus)
        axes[1].plot(config["ranks"], [route(corpus, r)["fidelity_to_full"] for r in config["ranks"]], marker="o", label=corpus)
    axes[0].set_title("Exact Gram spectrum")
    axes[0].set_ylabel("cumulative energy")
    axes[1].set_title("Finite program recovery")
    axes[1].set_ylabel("median Fisher fidelity")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Gram rank")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
