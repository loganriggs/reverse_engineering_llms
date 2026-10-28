#!/usr/bin/env python3
"""Nearest-context router for layer-8 interaction-mode support."""

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


CONFIG = Path("configs/bilin18_layer8_knn_router.json")
QUADRATIC_OUTPUT = TENSOR_ROOT / "layer8_quadratic_router.json"
OUTPUT = TENSOR_ROOT / "layer8_knn_router.json"
REPORT = TENSOR_ROOT / "LAYER8_KNN_ROUTER.md"
FIGURE = TENSOR_ROOT / "layer8_knn_router.png"


def route_grid(
    discovery_features: torch.Tensor,
    discovery_absolute: torch.Tensor,
    evaluation_features: torch.Tensor,
    evaluation_contributions: torch.Tensor,
    neighbors: list[int],
    budget: int,
    batch_size: int,
) -> dict:
    maximum = max(neighbors)
    full = evaluation_contributions.sum(-1)
    oracle_indices = evaluation_contributions.abs().topk(budget, dim=-1).indices
    oracle = torch.gather(evaluation_contributions, -1, oracle_indices).sum(-1)
    routed_chunks = {count: [] for count in neighbors}
    recall_chunks = {count: [] for count in neighbors}
    coverage_chunks = {count: [] for count in neighbors}
    discovery_norm = discovery_features.square().sum(-1)
    for start in range(0, len(evaluation_features), batch_size):
        points = evaluation_features[start : start + batch_size]
        distances = (
            points.square().sum(-1, keepdim=True)
            + discovery_norm[None]
            - 2 * points @ discovery_features.T
        )
        nearest = distances.topk(maximum, largest=False, dim=-1).indices
        neighbor_values = discovery_absolute[nearest]
        cumulative = neighbor_values.cumsum(1)
        actual = evaluation_contributions[start : start + len(points)]
        oracle_batch = oracle_indices[start : start + len(points)]
        for count in neighbors:
            predicted_absolute = cumulative[:, count - 1] / count
            selected = predicted_absolute.topk(budget, dim=-1).indices
            routed_chunks[count].append(torch.gather(actual, -1, selected).sum(-1).cpu())
            recall = (selected[..., :, None] == oracle_batch[..., None, :]).any(-1).float().mean(-1)
            coverage = torch.gather(actual.abs(), -1, selected).sum(-1) / actual.abs().sum(-1).clamp_min(1e-30)
            recall_chunks[count].append(recall.cpu())
            coverage_chunks[count].append(coverage.cpu())
    full_np = full.cpu().numpy()
    oracle_corr, oracle_r2 = score_metrics(full_np, oracle.cpu().numpy())
    results = {
        "oracle": {
            "median_correlation": float(np.median(oracle_corr)),
            "median_centered_r2": float(np.median(oracle_r2)),
        },
        "neighbors": {},
    }
    for count in neighbors:
        routed = torch.cat(routed_chunks[count]).numpy()
        corr, r2 = score_metrics(full_np, routed)
        results["neighbors"][str(count)] = {
            "median_correlation": float(np.median(corr)),
            "median_centered_r2": float(np.median(r2)),
            "median_oracle_support_recall": float(torch.cat(recall_chunks[count]).median()),
            "median_absolute_mass_coverage": float(torch.cat(coverage_chunks[count]).median()),
        }
    return results


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    quadratic = json.loads(QUADRATIC_OUTPUT.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    layer = config["layer"]
    discovery_x, discovery_write = collect(model, layer, config["discovery"], config)
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = state_vh.T[:, : config["dimension"]]
    discovery_z = discovery_x @ basis
    feature_mean = discovery_z.mean(0)
    feature_std = discovery_z.std(0).clamp_min(1e-6)
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
    discovery_projections = torch.einsum("nd,qdk->nqk", discovery_z, vectors)
    discovery_absolute = (discovery_projections.square() * values[None]).abs()

    results = {}
    for corpus, corpus_config in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        states, _ = collect(model, layer, corpus_config, config)
        z = states @ basis
        projections = torch.einsum("nd,qdk->nqk", z, vectors)
        contributions = projections.square() * values[None]
        dimensions = {}
        for dimension in config["router_dimensions"]:
            discovery_features = (discovery_z[:, :dimension] - feature_mean[:dimension]) / feature_std[:dimension]
            evaluation_features = (z[:, :dimension] - feature_mean[:dimension]) / feature_std[:dimension]
            dimensions[str(dimension)] = route_grid(
                discovery_features,
                discovery_absolute,
                evaluation_features,
                contributions,
                config["neighbors"],
                config["budget"],
                config["router_batch_size"],
            )
        results[corpus] = {
            "dimensions": dimensions,
            "quadratic128": quadratic["results"][corpus]["router_dimensions"]["128"]["exact_routed"],
        }

    primary_dimension = str(config["primary_router_dimension"])
    primary_neighbors = str(config["primary_neighbors"])

    def primary(corpus: str) -> dict:
        return results[corpus]["dimensions"][primary_dimension]["neighbors"][primary_neighbors]

    hypotheses = {
        "K1_primary_knn_is_predictive": {
            "supported": all(primary(corpus)["median_correlation"] >= 0.85 for corpus in results)
        },
        "K2_primary_knn_beats_quadratic": {
            "supported": all(
                primary(corpus)["median_correlation"]
                - results[corpus]["quadratic128"]["median_correlation"] >= 0.03
                for corpus in results
            )
        },
        "K3_primary_knn_approaches_oracle": {
            "supported": all(
                results[corpus]["dimensions"][primary_dimension]["oracle"]["median_correlation"]
                - primary(corpus)["median_correlation"] <= 0.05
                for corpus in results
            )
        },
        "K4_primary_knn_recovers_support": {
            "supported": all(primary(corpus)["median_oracle_support_recall"] >= 0.50 for corpus in results)
        },
        "K5_primary_knn_transfers_ood": {
            "supported": all(
                primary("fineweb")["median_correlation"] - primary(corpus)["median_correlation"] <= 0.05
                for corpus in ("wikitext", "tinystories")
            )
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Nearest-context router for layer-8 modes",
        "",
        "FineWeb discovery examples form a nonparametric route memory. For each evaluation state, nearest neighbors in standardized state-SVD coordinates vote by mean absolute mode contribution; the resulting top 16 modes are then evaluated exactly. The preregistered router uses 128 coordinates and 8 neighbors. Other dimensions and neighbor counts are exploratory diagnostics.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![KNN router](layer8_knn_router.png)",
        "",
        "| Corpus | State dim | Neighbors | Routed corr./R² | Oracle corr. | Support recall | Abs. mass | Quadratic-128 corr. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_row in results.items():
        for dimension in config["router_dimensions"]:
            dimension_row = corpus_row["dimensions"][str(dimension)]
            for neighbors in config["neighbors"]:
                item = dimension_row["neighbors"][str(neighbors)]
                lines.append(
                    f"| {corpus} | {dimension} | {neighbors} | {item['median_correlation']:.3f}/{item['median_centered_r2']:.3f} | "
                    f"{dimension_row['oracle']['median_correlation']:.3f} | {item['median_oracle_support_recall']:.3f} | "
                    f"{item['median_absolute_mass_coverage']:.3f} | {corpus_row['quadratic128']['median_correlation']:.3f} |"
                )
    lines += [
        "",
        "Nearest-context routing is a predictability diagnostic, not a compact mechanism: it stores the discovery contribution table and performs retrieval. The preregistered router fails every fidelity/support gate except OOD stability, and the exploratory grid remains well below the oracle. Coarse or local similarity in up to 128 leading state coordinates therefore does not recover the adaptive route in this test.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for corpus, corpus_row in results.items():
        for axis, dimension in zip(axes, config["router_dimensions"]):
            values = corpus_row["dimensions"][str(dimension)]["neighbors"]
            axis.plot(config["neighbors"], [values[str(k)]["median_correlation"] for k in config["neighbors"]], marker="o", label=corpus)
            axis.set_xscale("log", base=2)
            axis.set_ylim(0, 1)
            axis.set_title(f"Router dimension {dimension}")
            axis.set_xlabel("nearest neighbors")
            axis.set_ylabel("median routed correlation")
            axis.grid(alpha=0.25)
            axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
