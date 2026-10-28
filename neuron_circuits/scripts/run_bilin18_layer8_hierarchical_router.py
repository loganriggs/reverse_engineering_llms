#!/usr/bin/env python3
"""Hierarchical 64D shortlist / 256D refine router for layer-8 modes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_quadratic_router import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_query_conditioned_interpreter import score_metrics
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_hierarchical_router.json")
FLAT_OUTPUT = TENSOR_ROOT / "layer8_router_dimension_sweep.json"
OUTPUT = TENSOR_ROOT / "layer8_hierarchical_router.json"
REPORT = TENSOR_ROOT / "LAYER8_HIERARCHICAL_ROUTER.md"
FIGURE = TENSOR_ROOT / "layer8_hierarchical_router.png"


def projection_multiplies(queries: int, modes: int, coarse: int, refine: int, shortlist: int) -> int:
    return queries * (modes * coarse + shortlist * (refine - coarse))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    flat = json.loads(FLAT_OUTPUT.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    layer = config["layer"]
    discovery_x, discovery_write = collect(model, layer, config["discovery"], config)
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = state_vh.T[:, : config["dimension"]]
    centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
    _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
    directions = write_vh.T[:, : config["queries"]]
    block = model.blocks[layer]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float() @ basis
    right = block.mlp.Right.weight.float() @ basis
    coefficients = down.T @ directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    values, vectors = ordered_eigensystem(matrices)
    budget = config["budget"]
    compute = {}
    flat_compute = config["queries"] * config["modes"] * config["refine_dimension"] if "modes" in config else config["queries"] * config["dimension"] * config["refine_dimension"]
    for shortlist in config["shortlist_sizes"]:
        count = projection_multiplies(
            config["queries"], config["dimension"], config["coarse_dimension"],
            config["refine_dimension"], shortlist,
        )
        compute[str(shortlist)] = {
            "candidate_projection_multiplies": count,
            "fraction_of_flat256": count / flat_compute,
            "flat256_multiplies": flat_compute,
        }

    results = {}
    for corpus, corpus_config in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        states, _ = collect(model, layer, corpus_config, config)
        z = states @ basis
        full_projection = torch.einsum("nd,qdk->nqk", z, vectors)
        full_contributions = full_projection.square() * values[None]
        full_score = full_contributions.sum(-1)
        oracle_indices = full_contributions.abs().topk(budget, dim=-1).indices
        oracle_score = torch.gather(full_contributions, -1, oracle_indices).sum(-1)
        oracle_corr, oracle_r2 = score_metrics(full_score.cpu().numpy(), oracle_score.cpu().numpy())
        coarse_projection = torch.einsum(
            "nd,qdk->nqk", z[:, : config["coarse_dimension"]], vectors[:, : config["coarse_dimension"]]
        )
        coarse_contributions = coarse_projection.square() * values[None]
        refine_projection = torch.einsum(
            "nd,qdk->nqk", z[:, : config["refine_dimension"]], vectors[:, : config["refine_dimension"]]
        )
        refine_contributions = refine_projection.square() * values[None]
        rows = {}
        for shortlist in config["shortlist_sizes"]:
            shortlist_indices = coarse_contributions.abs().topk(shortlist, dim=-1).indices
            shortlist_refined = torch.gather(refine_contributions, -1, shortlist_indices)
            local_selected = shortlist_refined.abs().topk(budget, dim=-1).indices
            selected = torch.gather(shortlist_indices, -1, local_selected)
            exact_routed = torch.gather(full_contributions, -1, selected).sum(-1)
            approximate_routed = torch.gather(refine_contributions, -1, selected).sum(-1)
            rows[str(shortlist)] = summarize(
                full_score, exact_routed, approximate_routed, selected,
                oracle_indices, full_contributions,
            )
        results[corpus] = {
            "oracle": {
                "median_correlation": float(np.median(oracle_corr)),
                "median_centered_r2": float(np.median(oracle_r2)),
            },
            "flat256": flat["results"][corpus]["dimensions"]["256"],
            "shortlists": rows,
        }

    def row(corpus: str, shortlist: str = "64") -> dict:
        return results[corpus]["shortlists"][shortlist]

    hypotheses = {
        "H1_shortlist64_matches_flat256": {
            "supported": all(
                results[corpus]["flat256"]["exact_routed"]["median_correlation"]
                - row(corpus)["exact_routed"]["median_correlation"] <= 0.02
                for corpus in results
            )
        },
        "H2_shortlist64_is_predictive": {
            "supported": all(row(corpus)["exact_routed"]["median_correlation"] >= 0.87 for corpus in results)
        },
        "H3_shortlist64_recovers_support": {
            "supported": all(row(corpus)["median_oracle_support_recall"] >= 0.55 for corpus in results)
        },
        "H4_shortlist64_reduces_router_compute": {
            "supported": compute["64"]["fraction_of_flat256"] <= 0.40,
            "fraction": compute["64"]["fraction_of_flat256"],
        },
        "H5_shortlist128_nearly_equals_flat": {
            "supported": all(
                results[corpus]["flat256"]["exact_routed"]["median_correlation"]
                - row(corpus, "128")["exact_routed"]["median_correlation"] <= 0.01
                for corpus in results
            )
        },
        "H6_shortlist64_reduced_score_is_predictive": {
            "supported": all(row(corpus)["approximate_routed"]["median_correlation"] >= 0.80 for corpus in results)
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "compute": compute,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# Hierarchical layer-8 address router",
        "",
        "A 64D coarse scan ranks all 512 candidate modes per query. Only the shortlisted modes receive the remaining 192 coordinates needed for a 256D refinement; 16 refined modes are selected. Exact routed scores evaluate those selected modes in the full 512D form. Candidate-projection multiply counts reuse the coarse partial dot products and exclude common state projection, top-k selection, and final selected-mode evaluation.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Hierarchical router](layer8_hierarchical_router.png)",
        "",
        "| Corpus | Shortlist | Exact corr./R² | Reduced corr./R² | Flat-256 corr. | Oracle corr. | Support recall | Router compute fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_row in results.items():
        for shortlist in config["shortlist_sizes"]:
            item = corpus_row["shortlists"][str(shortlist)]
            lines.append(
                f"| {corpus} | {shortlist} | {item['exact_routed']['median_correlation']:.3f}/{item['exact_routed']['median_centered_r2']:.3f} | "
                f"{item['approximate_routed']['median_correlation']:.3f}/{item['approximate_routed']['median_centered_r2']:.3f} | "
                f"{corpus_row['flat256']['exact_routed']['median_correlation']:.3f} | {corpus_row['oracle']['median_correlation']:.3f} | "
                f"{item['median_oracle_support_recall']:.3f} | {compute[str(shortlist)]['fraction_of_flat256']:.3f} |"
            )
    lines += [
        "",
        "The compute fraction covers candidate projections, the part changed by hierarchical routing. It is not an end-to-end layer speedup: these 16 queries do not reconstruct the full MLP write, and state projection, top-k operations, selected-mode evaluation, memory traffic, and later transformer blocks remain.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for corpus, corpus_row in results.items():
        shortlists = config["shortlist_sizes"]
        axes[0].plot(shortlists, [corpus_row["shortlists"][str(s)]["exact_routed"]["median_correlation"] for s in shortlists], marker="o", label=corpus)
    axes[0].set_xlabel("shortlist size")
    axes[0].set_ylabel("median exact-routed correlation")
    axes[0].set_title("Fidelity")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(config["shortlist_sizes"], [compute[str(s)]["fraction_of_flat256"] for s in config["shortlist_sizes"]], marker="o")
    axes[1].set_xlabel("shortlist size")
    axes[1].set_ylabel("fraction of flat-256 candidate multiplies")
    axes[1].set_title("Router candidate cost")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
