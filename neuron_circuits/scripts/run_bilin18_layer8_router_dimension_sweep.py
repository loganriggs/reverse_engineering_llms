#!/usr/bin/env python3
"""Large-dimension sweep for the layer-8 quadratic support router."""

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


CONFIG = Path("configs/bilin18_layer8_router_dimension_sweep.json")
OUTPUT = TENSOR_ROOT / "layer8_router_dimension_sweep.json"
REPORT = TENSOR_ROOT / "LAYER8_ROUTER_DIMENSION_SWEEP.md"
FIGURE = TENSOR_ROOT / "layer8_router_dimension_sweep.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
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
    results = {}
    for corpus, corpus_config in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        states, _ = collect(model, layer, corpus_config, config)
        z = states @ basis
        full_projections = torch.einsum("nd,qdk->nqk", z, vectors)
        full_contributions = full_projections.square() * values[None]
        full_score = full_contributions.sum(-1)
        oracle_indices = full_contributions.abs().topk(budget, dim=-1).indices
        oracle_score = torch.gather(full_contributions, -1, oracle_indices).sum(-1)
        oracle_corr, oracle_r2 = score_metrics(full_score.cpu().numpy(), oracle_score.cpu().numpy())
        rows = {}
        for dimension in config["router_dimensions"]:
            approximate_projections = torch.einsum(
                "nd,qdk->nqk", z[:, :dimension], vectors[:, :dimension]
            )
            approximate_contributions = approximate_projections.square() * values[None]
            selected = approximate_contributions.abs().topk(budget, dim=-1).indices
            exact_routed = torch.gather(full_contributions, -1, selected).sum(-1)
            approximate_routed = torch.gather(approximate_contributions, -1, selected).sum(-1)
            rows[str(dimension)] = summarize(
                full_score,
                exact_routed,
                approximate_routed,
                selected,
                oracle_indices,
                full_contributions,
            )
        results[corpus] = {
            "oracle": {
                "median_correlation": float(np.median(oracle_corr)),
                "median_centered_r2": float(np.median(oracle_r2)),
            },
            "dimensions": rows,
        }

    def row(corpus: str, dimension: str) -> dict:
        return results[corpus]["dimensions"][dimension]

    below_192 = [dimension for dimension in config["router_dimensions"] if dimension < 192]
    hypotheses = {
        "D1_dimension256_is_predictive": {
            "supported": all(row(corpus, "256")["exact_routed"]["median_correlation"] >= 0.85 for corpus in results)
        },
        "D2_dimension256_is_near_oracle": {
            "supported": all(
                results[corpus]["oracle"]["median_correlation"]
                - row(corpus, "256")["exact_routed"]["median_correlation"] <= 0.05
                for corpus in results
            )
        },
        "D3_at_least_192_dimensions_are_needed": {
            "supported": all(
                any(row(corpus, str(dimension))["exact_routed"]["median_correlation"] < 0.85 for corpus in results)
                for dimension in below_192
            )
        },
        "D4_dimension384_closes_oracle_gap": {
            "supported": all(
                results[corpus]["oracle"]["median_correlation"]
                - row(corpus, "384")["exact_routed"]["median_correlation"] <= 0.02
                for corpus in results
            )
        },
        "D5_dimension256_reduced_score_is_predictive": {
            "supported": all(row(corpus, "256")["approximate_routed"]["median_correlation"] >= 0.80 for corpus in results)
        },
    }
    first_predictive = None
    first_near_oracle = None
    for dimension in config["router_dimensions"]:
        if first_predictive is None and all(
            row(corpus, str(dimension))["exact_routed"]["median_correlation"] >= 0.85 for corpus in results
        ):
            first_predictive = dimension
        if first_near_oracle is None and all(
            results[corpus]["oracle"]["median_correlation"]
            - row(corpus, str(dimension))["exact_routed"]["median_correlation"] <= 0.05
            for corpus in results
        ):
            first_near_oracle = dimension
    OUTPUT.write_text(json.dumps({
        "config": config,
        "first_predictive_dimension": first_predictive,
        "first_near_oracle_dimension": first_near_oracle,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# Large-dimension sweep for the layer-8 quadratic router",
        "",
        "The strongest frozen router estimates each mode's contribution from a prefix of the 512D state basis, chooses 16 modes, and then either evaluates those modes exactly (`exact routed`) or uses the reduced-context contributions (`approximate`). Dimension 512 equals oracle selection by construction and is included as the endpoint, not an independent success.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"First dimension predictive on every corpus: **{first_predictive}**. First within 0.05 of oracle: **{first_near_oracle}**.",
        "",
        "![Router dimension sweep](layer8_router_dimension_sweep.png)",
        "",
        "| Corpus | Router dim | Exact corr./R² | Approx. corr./R² | Oracle corr. | Support recall | Abs. mass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_row in results.items():
        for dimension in config["router_dimensions"]:
            item = corpus_row["dimensions"][str(dimension)]
            lines.append(
                f"| {corpus} | {dimension} | {item['exact_routed']['median_correlation']:.3f}/{item['exact_routed']['median_centered_r2']:.3f} | "
                f"{item['approximate_routed']['median_correlation']:.3f}/{item['approximate_routed']['median_centered_r2']:.3f} | "
                f"{corpus_row['oracle']['median_correlation']:.3f} | {item['median_oracle_support_recall']:.3f} | "
                f"{item['median_absolute_mass_coverage']:.3f} |"
            )
    lines += [
        "",
        "The endpoint shows the interpolation to oracle; the scientifically relevant quantity is how many state dimensions are needed before held-out and OOD gates pass. Exact routed scores isolate support information, while approximate scores additionally require contribution magnitudes to be represented in the reduced state.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for corpus, corpus_row in results.items():
        dims = config["router_dimensions"]
        axes[0].plot(dims, [corpus_row["dimensions"][str(d)]["exact_routed"]["median_correlation"] for d in dims], marker="o", label=corpus)
        axes[1].plot(dims, [corpus_row["dimensions"][str(d)]["approximate_routed"]["median_correlation"] for d in dims], marker="o", label=corpus)
    axes[0].set_title("Support information")
    axes[1].set_title("Reduced-context score")
    for axis in axes:
        axis.set_xlabel("state dimensions")
        axis.set_ylabel("median correlation")
        axis.set_ylim(0.6, 1.01)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
