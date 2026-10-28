#!/usr/bin/env python3
"""Route layer-8 modes from reduced-context quadratic contribution estimates."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_query_conditioned_interpreter import score_metrics
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_quadratic_router.json")
CLUSTER_OUTPUT = TENSOR_ROOT / "layer8_mode_router.json"
OUTPUT = TENSOR_ROOT / "layer8_quadratic_router.json"
REPORT = TENSOR_ROOT / "LAYER8_QUADRATIC_ROUTER.md"
FIGURE = TENSOR_ROOT / "layer8_quadratic_router.png"


def summarize(
    full: torch.Tensor,
    actual: torch.Tensor,
    approximate: torch.Tensor,
    selected: torch.Tensor,
    oracle_indices: torch.Tensor,
    full_contributions: torch.Tensor,
) -> dict:
    full_np = full.cpu().numpy()
    result = {}
    for name, prediction in (("exact_routed", actual), ("approximate_routed", approximate)):
        corr, r2 = score_metrics(full_np, prediction.cpu().numpy())
        result[name] = {
            "median_correlation": float(np.median(corr)),
            "median_centered_r2": float(np.median(r2)),
        }
    recall = (selected[..., :, None] == oracle_indices[..., None, :]).any(-1).float().mean(-1)
    coverage = torch.gather(full_contributions.abs(), -1, selected).sum(-1) / full_contributions.abs().sum(-1).clamp_min(1e-30)
    result["median_oracle_support_recall"] = float(recall.median())
    result["median_absolute_mass_coverage"] = float(coverage.median())
    return result


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    cluster_result = json.loads(CLUSTER_OUTPUT.read_text())
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
        dimension_rows = {}
        for dimension in config["router_dimensions"]:
            approximate_projections = torch.einsum(
                "nd,qdk->nqk", z[:, :dimension], vectors[:, :dimension]
            )
            approximate_contributions = approximate_projections.square() * values[None]
            selected = approximate_contributions.abs().topk(budget, dim=-1).indices
            exact_routed = torch.gather(full_contributions, -1, selected).sum(-1)
            approximate_routed = torch.gather(approximate_contributions, -1, selected).sum(-1)
            dimension_rows[str(dimension)] = summarize(
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
            "prior_cluster64": cluster_result["results"][corpus]["clusters"]["64"][str(budget)]["routed"],
            "router_dimensions": dimension_rows,
        }

    def row(corpus: str, dimension: str = "64") -> dict:
        return results[corpus]["router_dimensions"][dimension]

    hypotheses = {
        "P1_quadratic64_is_predictive": {
            "supported": all(row(corpus)["exact_routed"]["median_correlation"] >= 0.85 for corpus in results)
        },
        "P2_quadratic64_beats_cluster64": {
            "supported": all(
                row(corpus)["exact_routed"]["median_correlation"]
                - results[corpus]["prior_cluster64"]["median_correlation"] >= 0.10
                for corpus in results
            )
        },
        "P3_quadratic128_approaches_oracle": {
            "supported": all(
                results[corpus]["oracle"]["median_correlation"]
                - row(corpus, "128")["exact_routed"]["median_correlation"] <= 0.05
                for corpus in results
            )
        },
        "P4_quadratic64_recovers_support": {
            "supported": all(row(corpus)["median_oracle_support_recall"] >= 0.40 for corpus in results)
        },
        "P5_quadratic64_is_executable": {
            "supported": all(row(corpus)["approximate_routed"]["median_correlation"] >= 0.80 for corpus in results)
        },
        "P6_router_improves_monotonically": {
            "supported": all(
                all(
                    results[corpus]["router_dimensions"][str(first)]["exact_routed"]["median_correlation"]
                    <= results[corpus]["router_dimensions"][str(second)]["exact_routed"]["median_correlation"] + 1e-8
                    for first, second in zip(config["router_dimensions"][:-1], config["router_dimensions"][1:])
                )
                for corpus in results
            )
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Reduced-context quadratic router for layer-8 modes",
        "",
        "For each exact 512D query eigenmode, the router estimates its realized magnitude from only the first `p` state-SVD coordinates using the same squared projection implied by the bilinear form. It selects 16 modes from those estimates. `exact routed` then evaluates the selected modes with the full 512D state and isolates support quality; `approximate routed` uses only reduced-context contributions and is the executable approximation.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Quadratic router](layer8_quadratic_router.png)",
        "",
        "| Corpus | Router dim | Exact routed corr./R² | Approx. corr./R² | Oracle corr. | Support recall | Abs. mass | Prior cluster-64 corr. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_row in results.items():
        for dimension in config["router_dimensions"]:
            item = corpus_row["router_dimensions"][str(dimension)]
            lines.append(
                f"| {corpus} | {dimension} | {item['exact_routed']['median_correlation']:.3f}/{item['exact_routed']['median_centered_r2']:.3f} | "
                f"{item['approximate_routed']['median_correlation']:.3f}/{item['approximate_routed']['median_centered_r2']:.3f} | "
                f"{corpus_row['oracle']['median_correlation']:.3f} | {item['median_oracle_support_recall']:.3f} | "
                f"{item['median_absolute_mass_coverage']:.3f} | {corpus_row['prior_cluster64']['median_correlation']:.3f} |"
            )
    lines += [
        "",
        "This router is architecture-matched but still scans all 512 candidate modes to select 16. Success would establish that mode identity is predictable from a small state quotient; an additional candidate-index or hierarchical factorization would be required for a true sparse speedup.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for corpus, corpus_row in results.items():
        dims = config["router_dimensions"]
        axes[0].plot(dims, [corpus_row["router_dimensions"][str(d)]["exact_routed"]["median_correlation"] for d in dims], marker="o", label=corpus)
        axes[1].plot(dims, [corpus_row["router_dimensions"][str(d)]["approximate_routed"]["median_correlation"] for d in dims], marker="o", label=corpus)
    axes[0].set_title("Support quality with exact selected modes")
    axes[1].set_title("Fully reduced-context score")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("router state dimension")
        axis.set_ylabel("median correlation")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
