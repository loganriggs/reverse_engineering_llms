#!/usr/bin/env python3
"""Turn layer-8 oracle mode sparsity into a frozen context-cluster router."""

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


CONFIG = Path("configs/bilin18_layer8_mode_router.json")
OUTPUT = TENSOR_ROOT / "layer8_mode_router.json"
REPORT = TENSOR_ROOT / "LAYER8_MODE_ROUTER.md"
FIGURE = TENSOR_ROOT / "layer8_mode_router.png"


def kmeans(points: torch.Tensor, clusters: int, iterations: int, seed: int):
    generator = torch.Generator(device=points.device).manual_seed(seed + clusters)
    initial = torch.randperm(len(points), generator=generator, device=points.device)[:clusters]
    centroids = points[initial].clone()
    assignments = torch.zeros(len(points), dtype=torch.long, device=points.device)
    for _ in range(iterations):
        distances = (
            points.square().sum(-1, keepdim=True)
            + centroids.square().sum(-1)[None]
            - 2 * points @ centroids.T
        )
        new_assignments = distances.argmin(-1)
        sums = torch.zeros_like(centroids)
        sums.index_add_(0, new_assignments, points)
        counts = torch.bincount(new_assignments, minlength=clusters)
        nonempty = counts > 0
        new_centroids = centroids.clone()
        new_centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        changed = not torch.equal(new_assignments, assignments)
        assignments = new_assignments
        centroids = new_centroids
        if not changed:
            break
    return centroids, assignments


def assign_clusters(points: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    distances = (
        points.square().sum(-1, keepdim=True)
        + centroids.square().sum(-1)[None]
        - 2 * points @ centroids.T
    )
    return distances.argmin(-1)


def make_routes(
    absolute_contributions: torch.Tensor,
    assignments: torch.Tensor,
    clusters: int,
    budget: int,
) -> torch.Tensor:
    flat = absolute_contributions.flatten(1)
    sums = torch.zeros(clusters, flat.shape[1], device=flat.device)
    sums.index_add_(0, assignments, flat)
    counts = torch.bincount(assignments, minlength=clusters).clamp_min(1)
    means = (sums / counts[:, None]).reshape(clusters, absolute_contributions.shape[1], -1)
    return means.topk(budget, dim=-1).indices


def routed_metrics(
    contributions: torch.Tensor,
    route_indices: torch.Tensor,
    assignments: torch.Tensor,
    budget: int,
) -> dict:
    full = contributions.sum(-1)
    chosen = route_indices[assignments]
    routed = torch.gather(contributions, -1, chosen).sum(-1)
    oracle_indices = contributions.abs().topk(budget, dim=-1).indices
    oracle = torch.gather(contributions, -1, oracle_indices).sum(-1)
    fixed = contributions[..., :budget].sum(-1)
    correlations = {}
    full_np = full.cpu().numpy()
    for name, prediction in (("fixed", fixed), ("routed", routed), ("oracle", oracle)):
        corr, r2 = score_metrics(full_np, prediction.cpu().numpy())
        correlations[name] = {
            "median_correlation": float(np.median(corr)),
            "median_centered_r2": float(np.median(r2)),
        }
    recall = (chosen[..., :, None] == oracle_indices[..., None, :]).any(-1).float().mean(-1)
    selected_mass = torch.gather(contributions.abs(), -1, chosen).sum(-1) / contributions.abs().sum(-1).clamp_min(1e-30)
    return {
        **correlations,
        "median_oracle_support_recall": float(recall.median()),
        "median_absolute_mass_coverage": float(selected_mass.median()),
        "used_clusters": int(assignments.unique().numel()),
    }


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
    discovery_z = discovery_x @ basis
    feature_mean = discovery_z[:, : config["router_dimension"]].mean(0)
    feature_std = discovery_z[:, : config["router_dimension"]].std(0).clamp_min(1e-6)
    discovery_features = (discovery_z[:, : config["router_dimension"]] - feature_mean) / feature_std
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
    discovery_contributions = discovery_projections.square() * values[None]

    routers = {}
    for clusters in config["cluster_counts"]:
        print("fit", clusters, "clusters", flush=True)
        centroids, assignments = kmeans(
            discovery_features, clusters, config["kmeans_iterations"], config["seed"]
        )
        routes = {
            str(budget): make_routes(discovery_contributions.abs(), assignments, clusters, budget)
            for budget in config["budgets"]
        }
        routers[str(clusters)] = {
            "centroids": centroids,
            "routes": routes,
            "training_cluster_counts": torch.bincount(assignments, minlength=clusters),
        }
    global_routes = {
        str(budget): discovery_contributions.abs().mean(0).topk(budget, dim=-1).indices[None]
        for budget in config["budgets"]
    }

    results = {}
    for corpus, corpus_config in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        states, _ = collect(model, layer, corpus_config, config)
        z = states @ basis
        features = (z[:, : config["router_dimension"]] - feature_mean) / feature_std
        projections = torch.einsum("nd,qdk->nqk", z, vectors)
        contributions = projections.square() * values[None]
        corpus_rows = {"clusters": {}, "global_average": {}}
        for budget in config["budgets"]:
            assignments = torch.zeros(len(z), dtype=torch.long, device="cuda")
            corpus_rows["global_average"][str(budget)] = routed_metrics(
                contributions, global_routes[str(budget)], assignments, budget
            )
        for clusters, router in routers.items():
            assignments = assign_clusters(features, router["centroids"])
            cluster_rows = {}
            for budget in config["budgets"]:
                item = routed_metrics(
                    contributions, router["routes"][str(budget)], assignments, budget
                )
                routes = router["routes"][str(budget)]
                unions = []
                for query in range(config["queries"]):
                    unions.append(int(routes[:, query].unique().numel()))
                item["median_route_union_modes"] = float(np.median(unions))
                cluster_rows[str(budget)] = item
            corpus_rows["clusters"][clusters] = cluster_rows
        results[corpus] = corpus_rows

    def routed(corpus: str, clusters: str = "64", budget: str = "16") -> dict:
        return results[corpus]["clusters"][clusters][budget]

    hypotheses = {
        "R1_cluster64_routes_are_predictive": {
            "supported": all(routed(corpus)["routed"]["median_correlation"] >= 0.85 for corpus in results)
        },
        "R2_routing_beats_fixed_spectrum": {
            "supported": all(
                routed(corpus)["routed"]["median_correlation"]
                - routed(corpus)["fixed"]["median_correlation"] >= 0.15
                for corpus in results
            )
        },
        "R3_cluster256_approaches_oracle": {
            "supported": all(
                routed(corpus, "256")["oracle"]["median_correlation"]
                - routed(corpus, "256")["routed"]["median_correlation"] <= 0.05
                for corpus in results
            )
        },
        "R4_router_transfers_ood": {
            "supported": all(
                routed("fineweb")["routed"]["median_correlation"]
                - routed(corpus)["routed"]["median_correlation"] <= 0.05
                for corpus in ("wikitext", "tinystories")
            )
        },
        "R5_router_predicts_support": {
            "supported": all(routed(corpus)["median_oracle_support_recall"] >= 0.40 for corpus in results)
        },
        "R6_routes_use_a_broad_dictionary": {
            "supported": routed("fineweb")["median_route_union_modes"] >= 128,
            "fineweb_union": routed("fineweb")["median_route_union_modes"],
        },
    }
    serializable_routers = {
        clusters: {
            "training_cluster_counts": router["training_cluster_counts"].cpu().tolist(),
            "route_union_by_budget": {
                budget: [int(router["routes"][budget][:, query].unique().numel()) for query in range(config["queries"])]
                for budget in router["routes"]
            },
        }
        for clusters, router in routers.items()
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "routers": serializable_routers,
        "results": results,
    }, indent=2))

    lines = [
        "# Frozen context router for layer-8 interaction modes",
        "",
        "A router is fit only on FineWeb discovery states. Standardized state-SVD-64 coordinates are clustered; each cluster stores the modes with highest mean absolute contribution for each of 16 natural output queries. At evaluation time, cluster assignment chooses a frozen route without observing realized mode contributions. Scores then use the selected exact modes. Fixed spectral, corpus-average, and oracle per-token subsets are matched-budget controls.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-8 mode router](layer8_mode_router.png)",
        "",
        "### Budget 16",
        "",
        "| Corpus | Clusters | Fixed corr. | Global-average corr. | Routed corr./R² | Oracle corr. | Support recall | Abs. mass | Route union |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_row in results.items():
        global_row = corpus_row["global_average"]["16"]
        for clusters in ("16", "64", "256"):
            item = corpus_row["clusters"][clusters]["16"]
            lines.append(
                f"| {corpus} | {clusters} | {item['fixed']['median_correlation']:.3f} | "
                f"{global_row['routed']['median_correlation']:.3f} | "
                f"{item['routed']['median_correlation']:.3f}/{item['routed']['median_centered_r2']:.3f} | "
                f"{item['oracle']['median_correlation']:.3f} | {item['median_oracle_support_recall']:.3f} | "
                f"{item['median_absolute_mass_coverage']:.3f} | {item['median_route_union_modes']:.0f} |"
            )
    lines += [
        "",
        "The routed score uses exact selected mode projections, so this tests support prediction rather than total execution cost. A practical compiled router must include state projection, nearest-centroid lookup, and sparse mode evaluation, and still needs finite-intervention validation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    corpora = list(results)
    x = np.arange(len(corpora))
    width = 0.22
    for index, clusters in enumerate(("16", "64", "256")):
        axes[0].bar(x + (index - 1) * width, [routed(corpus, clusters)["routed"]["median_correlation"] for corpus in corpora], width, label=f"{clusters} clusters")
        axes[1].bar(x + (index - 1) * width, [routed(corpus, clusters)["median_oracle_support_recall"] for corpus in corpora], width, label=f"{clusters} clusters")
    for axis, title, ylabel in (
        (axes[0], "Routed signed-score prediction", "median correlation"),
        (axes[1], "Oracle support recovery", "median recall"),
    ):
        axis.set_xticks(x, corpora)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
