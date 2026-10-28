#!/usr/bin/env python3
"""Red-team exact Gram compactness with random sites and output readouts."""

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
from run_bilin18_layer8_full_tensor_gram_rank import gram_eigensystem, low_rank_response
from run_bilin18_layer8_joint_fisher_program import (
    greedy_normalized_joint,
    normalized_joint_objective,
)
from run_bilin18_layer8_router_causal_validation import prefix_to_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_gram_random_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_gram_random_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_GRAM_RANDOM_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer8_gram_random_redteam.png"


def interaction_eigensystems(block, basis, output_directions):
    left = block.mlp.Left.weight.float() @ basis
    right = block.mlp.Right.weight.float() @ basis
    coefficients = block.mlp.Down.weight.float().T @ output_directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    return ordered_eigensystem(matrices)


def summarize(records, ranks):
    result = {}
    for rank in ranks:
        result[str(rank)] = {
            key: float(np.median([record["ranks"][str(rank)][key] for record in records]))
            for key in (
                "energy_fraction", "native_kl", "fidelity_to_full",
                "cosine_to_full", "objective_efficiency", "support_recall",
            )
        }
    result["full_native_kl"] = float(np.median([record["full_native_kl"] for record in records]))
    return result


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    high_score = json.loads(Path(config["high_score_results"]).read_text())
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
    natural_directions = write_vh.T[:, :config["queries_per_family"]]
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_matrix = torch.randn(
        model.config.n_embd, config["queries_per_family"], generator=generator, device="cuda",
    )
    random_directions = torch.linalg.qr(random_matrix, mode="reduced").Q
    block8 = model.blocks[8]
    family_data = {}
    for name, directions in (("natural", natural_directions), ("random", random_directions)):
        values, vectors = interaction_eigensystems(block8, tensor_basis, directions)
        family_data[name] = {"directions": directions, "values": values, "vectors": vectors}

    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(config["seed"] + corpus_index)
        site_grid = rng.integers(
            0, len(rows), size=(config["queries_per_family"], config["sites_per_query"]),
        )
        family_results = {}
        for family, data in family_data.items():
            records = []
            for query in range(config["queries_per_family"]):
                mode_directions = tensor_basis @ data["vectors"][query]
                for row_index in site_grid[query]:
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
                    response = output_fisher_response(logits, logit_tangents)
                    gram_values, gram_vectors = gram_eigensystem(response)
                    total = gram_values.sum().clamp_min(1e-30)
                    route_indices = {
                        "full": greedy_normalized_joint(
                            response, amplitudes, source_config["budget"],
                        ),
                    }
                    for rank in config["ranks"]:
                        route_indices[f"rank{rank}"] = greedy_normalized_joint(
                            low_rank_response(gram_values, gram_vectors, rank),
                            amplitudes, source_config["budget"],
                        )
                    native_logits, route_rows = finite_routes(
                        model, pre, normalized, x0, value_bus,
                        source_config["causal_position"], mode_directions, route_indices,
                        data["directions"][:, query], source_config["edit_norm_fraction"],
                    )
                    full_effect = route_rows["full"]["effect"]
                    full_objective = normalized_joint_objective(
                        response, amplitudes, route_indices["full"],
                    )
                    full_set = set(route_indices["full"].cpu().tolist())
                    rank_rows = {}
                    for rank in config["ranks"]:
                        route = f"rank{rank}"
                        fidelity, cosine, _ = fisher_fidelity(
                            native_logits, full_effect, route_rows[route]["effect"],
                        )
                        selected = route_indices[route]
                        rank_rows[str(rank)] = {
                            "energy_fraction": float(gram_values[:rank].sum() / total),
                            "native_kl": route_rows[route]["native_kl"],
                            "fidelity_to_full": fidelity,
                            "cosine_to_full": cosine,
                            "objective_efficiency": normalized_joint_objective(
                                response, amplitudes, selected,
                            ) / max(full_objective, 1e-30),
                            "support_recall": len(set(selected.cpu().tolist()) & full_set)
                            / source_config["budget"],
                        }
                    records.append({
                        "query": query,
                        "row": int(row_index),
                        "full_native_kl": route_rows["full"]["native_kl"],
                        "ranks": rank_rows,
                    })
                print(corpus, family, "query", query, "complete", flush=True)
            family_results[family] = {"summary": summarize(records, config["ranks"]), "records": records}
        results[corpus] = family_results

    hypotheses = {
        "B1_rank4_survives_random_sites": {
            "supported": all(results[c]["natural"]["summary"]["4"]["fidelity_to_full"] >= 0.95 for c in results),
        },
        "B2_rank4_survives_random_readouts": {
            "supported": all(results[c]["random"]["summary"]["4"]["fidelity_to_full"] >= 0.95 for c in results),
        },
        "B3_rank16_is_universally_near_exact": {
            "supported": all(
                results[c][family]["summary"]["16"]["fidelity_to_full"] >= 0.99
                for c in results for family in ("natural", "random")
            ),
        },
        "B4_high_score_selection_is_not_required": {
            "supported": all(
                abs(
                    results[c]["natural"]["summary"]["4"]["fidelity_to_full"]
                    - high_score["results"][c]["summary"]["routes"]["rank4"]["fidelity_to_full"]
                ) <= 0.05 for c in results
            ),
        },
        "B5_random_readouts_have_similar_gram_compactness": {
            "supported": all(
                abs(
                    results[c]["natural"]["summary"]["4"]["energy_fraction"]
                    - results[c]["random"]["summary"]["4"]["energy_fraction"]
                ) <= 0.10 for c in results
            ),
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Random-site and random-readout red-team of Gram compactness",
        "",
        "The exact full tensor Jacobian is recomputed on uniformly sampled evaluation rows rather than high-query-score sites. Eight leading natural-write readouts are compared with eight seeded random orthonormal output readouts. Each readout defines its own Layer-8 interaction eigenmodes, but program selection and finite evaluation use the same full-vocabulary Fisher construction.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Gram random red-team](layer8_gram_random_redteam.png)",
        "",
        "| Corpus | Readout family | Rank | Gram energy | Exact KL | Effect fidelity/cosine | Objective efficiency | Support recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_result in results.items():
        for family in ("natural", "random"):
            for rank in config["ranks"]:
                row = corpus_result[family]["summary"][str(rank)]
                lines.append(
                    f"| {corpus} | {family} | {rank} | {row['energy_fraction']:.3f} | "
                    f"{row['native_kl']:.3e} | {row['fidelity_to_full']:.3f}/{row['cosine_to_full']:.3f} | "
                    f"{row['objective_efficiency']:.3f} | {row['support_recall']:.3f} |"
                )
    lines += [
        "",
        "If random readouts are equally compact, rank 4 is best interpreted as a generic state-conditioned behavioral bottleneck. Semantic evidence would then need stable token/context labels or selective transfer, not spectral compactness itself.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels, fidelity, energy = [], [], []
    for corpus in results:
        for family in ("natural", "random"):
            labels.append(f"{corpus}/{family}")
            fidelity.append(results[corpus][family]["summary"]["4"]["fidelity_to_full"])
            energy.append(results[corpus][family]["summary"]["4"]["energy_fraction"])
    x = np.arange(len(labels))
    axes[0].bar(x, fidelity)
    axes[1].bar(x, energy)
    axes[0].set_title("Rank-4 finite recovery")
    axes[0].set_ylabel("median Fisher fidelity")
    axes[1].set_title("Rank-4 Gram energy")
    axes[1].set_ylabel("median energy fraction")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
