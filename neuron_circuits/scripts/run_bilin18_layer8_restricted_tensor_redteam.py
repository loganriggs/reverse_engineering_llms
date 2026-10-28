#!/usr/bin/env python3
"""Red-team Layer-8 restricted programs across rank, side, and metric."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_conditional_subspace_quantization import grassmann_similarity, state_centroids
from run_bilin18_layer8_restricted_tensor_program import (
    gram_singular_basis,
    prediction_r2,
    restricted_tensor,
    ridge_covariance_transforms,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_restricted_tensor_redteam.json")
SOURCE = TENSOR_ROOT / "layer8_multimetric_quantization.npz"
BASE_RESULT = TENSOR_ROOT / "layer8_restricted_tensor_program.json"
OUTPUT = TENSOR_ROOT / "layer8_restricted_tensor_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_RESTRICTED_TENSOR_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer8_restricted_tensor_redteam.png"


def bases_for_metric(tensor, square):
    weighted = torch.einsum("oaj,jk->oak", tensor, square)
    context_matrix = weighted.reshape(-1, weighted.shape[-1])
    context_singular, context_basis = gram_singular_basis(context_matrix, "right")
    output_matrix = weighted.reshape(weighted.shape[0], -1)
    output_singular, output_basis = gram_singular_basis(output_matrix, "left")
    return weighted, context_singular, context_basis, output_singular, output_basis


def predictions(x, inverse, weighted, context_basis, output_basis, rank):
    context = context_basis[:, :rank]
    output = output_basis[:, :rank]
    coordinates = (x @ inverse) @ context
    core = torch.einsum("oaj,jr->oar", weighted, context)
    context_only = torch.einsum("nr,oar->nao", coordinates, core)
    output_only = torch.einsum(
        "nao,os,ps->nap",
        torch.einsum("oaj,nj->nao", weighted, x @ inverse),
        output,
        output,
    )
    joint = torch.einsum("nao,os,ps->nap", context_only, output, output)
    return context_only, output_only, joint


def smallest_rank(rows, threshold=.9):
    for row in rows:
        if row["joint_test_r2"] >= threshold:
            return row["rank"]
    return None


def tucker_parameters(rank, output=1152, context=1152, edit=4):
    return output * rank + context * rank + edit * rank * rank


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE)
    base = json.loads(BASE_RESULT.read_text())
    active = [int(value) for value in base["active_prototypes"]]
    basis = torch.from_numpy(source["basis"]).cuda().float()
    axes = torch.from_numpy(source["axes_direct_mlp"]).cuda().float()
    prototypes = torch.from_numpy(source["prototypes_direct_mlp"]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    discovery_states, _ = collect(model, 8, config["prototype_discovery"], {**config, "positions": [config["causal_position"]]})
    discovery_features = discovery_states @ basis
    discovery_features /= discovery_features.norm(dim=1, keepdim=True).clamp_min(1e-30)
    discovery_labels = grassmann_similarity(axes, prototypes).argmax(1)
    centroids = state_centroids(discovery_features, discovery_labels, config["clusters"])
    states, _ = collect(model, 8, config["natural_states"], {**config, "positions": [config["causal_position"]]})
    features = states @ basis
    features /= features.norm(dim=1, keepdim=True).clamp_min(1e-30)
    labels = (features @ centroids.T).argmax(1)
    row_ids = torch.arange(len(labels), device="cuda")
    train_mask = row_ids % 2 == 0
    test_mask = ~train_mask
    global_square, global_inverse, global_covariance = ridge_covariance_transforms(
        states[train_mask], config["primary_ridge_fraction"]
    )
    block = model.blocks[8]
    down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
    results = {}
    ridge_summary = {str(value): [] for value in config["ridge_fractions"]}
    for cluster in active:
        route_train = (labels == cluster) & train_mask
        route_test = (labels == cluster) & test_mask
        train_x = states[route_train]
        test_x = states[route_test]
        print("red-team prototype", cluster, len(train_x), len(test_x), flush=True)
        edit = basis @ prototypes[cluster]
        tensor = restricted_tensor(down, left, right, edit)
        target_train = torch.einsum("oaj,nj->nao", tensor, train_x)
        target_test = torch.einsum("oaj,nj->nao", tensor, test_x)
        metric_rows = {}
        for metric, square, inverse, covariance in (
            ("route", *ridge_covariance_transforms(train_x, config["primary_ridge_fraction"])),
            ("global", global_square, global_inverse, global_covariance),
        ):
            weighted, context_singular, context_basis, output_singular, output_basis = bases_for_metric(tensor, square)
            rows = []
            for rank in config["ranks"]:
                train_predictions = predictions(train_x, inverse, weighted, context_basis, output_basis, rank)
                test_predictions = predictions(test_x, inverse, weighted, context_basis, output_basis, rank)
                rows.append({
                    "rank": rank,
                    "context_train_r2": prediction_r2(target_train, train_predictions[0]),
                    "output_train_r2": prediction_r2(target_train, train_predictions[1]),
                    "joint_train_r2": prediction_r2(target_train, train_predictions[2]),
                    "context_test_r2": prediction_r2(target_test, test_predictions[0]),
                    "output_test_r2": prediction_r2(target_test, test_predictions[1]),
                    "joint_test_r2": prediction_r2(target_test, test_predictions[2]),
                    "context_energy": float(context_singular[:rank].square().sum() / context_singular.square().sum().clamp_min(1e-30)),
                    "output_energy": float(output_singular[:rank].square().sum() / output_singular.square().sum().clamp_min(1e-30)),
                })
            metric_rows[metric] = {"covariance": covariance, "rows": rows, "rank90": smallest_rank(rows)}
        for ridge in config["ridge_fractions"]:
            square, inverse, _ = ridge_covariance_transforms(train_x, ridge)
            weighted, _, context_basis, _, output_basis = bases_for_metric(tensor, square)
            _, _, joint = predictions(test_x, inverse, weighted, context_basis, output_basis, 128)
            ridge_summary[str(ridge)].append(prediction_r2(target_test, joint))
        results[str(cluster)] = {
            "train_sites": int(len(train_x)),
            "test_sites": int(len(test_x)),
            "metrics": metric_rows,
        }
    full = [row["metrics"]["route"]["rows"][-1]["joint_test_r2"] for row in results.values()]
    rank128 = [next(item for item in row["metrics"]["route"]["rows"] if item["rank"] == 128) for row in results.values()]
    rank512 = [next(item for item in row["metrics"]["route"]["rows"] if item["rank"] == 512) for row in results.values()]
    route128 = np.asarray([row["joint_test_r2"] for row in rank128])
    global128 = np.asarray([next(item for item in row["metrics"]["global"]["rows"] if item["rank"] == 128)["joint_test_r2"] for row in results.values()])
    ridge_medians = {key: float(np.median(value)) for key, value in ridge_summary.items()}
    ranks90 = [row["metrics"]["route"]["rank90"] for row in results.values()]
    observed_ranks = [rank for rank in ranks90 if rank is not None]
    dense = 1152 * 4 * 1152
    median_parameters = float(np.median([tucker_parameters(rank) for rank in observed_ranks])) if len(observed_ranks) == len(ranks90) else None
    hypotheses = {
        "R1_full_rank_reconstructs": {"supported": min(full) >= .999, "minimum_r2": min(full)},
        "R2_both_sides_limit_rank128": {"supported": sum(min(row["context_test_r2"], row["output_test_r2"]) < .9 for row in rank128) >= 6},
        "R3_rank512_is_usually_sufficient": {"supported": sum(row["joint_test_r2"] >= .9 for row in rank512) >= len(rank512) / 2, "passing": sum(row["joint_test_r2"] >= .9 for row in rank512)},
        "R4_route_metric_beats_global": {"supported": float(np.median(route128)) > float(np.median(global128)), "route_median": float(np.median(route128)), "global_median": float(np.median(global128))},
        "R5_ridge_choice_is_not_decisive": {"supported": max(ridge_medians.values()) - min(ridge_medians.values()) <= .1, "medians": ridge_medians},
        "R6_no_small_equal_rank_program": {"supported": not observed_ranks or float(np.median(observed_ranks)) > 128, "ranks90": ranks90},
        "R7_rank90_program_is_parameter_compressive": {"supported": median_parameters is not None and median_parameters < dense, "median_parameters": median_parameters, "dense_parameters": dense},
    }
    payload = {"config": config, "active_prototypes": active, "hypotheses": hypotheses, "ridge_rank128": ridge_summary, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = [
        "# Rank and metric red-team for Layer-8 restricted tensor programs", "",
        "The initial equal-rank Tucker failure is extended to full rank, split into context-only and output-only approximations, compared under route-conditioned and global FineWeb covariance metrics, and repeated across three explicit ridge priors.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Restricted tensor red-team](layer8_restricted_tensor_redteam.png)", "", "| Prototype | Route K90 | Joint R² @128 / 512 / full | Context/output R² @128 | Global joint @128 |", "|---:|---:|---:|---:|---:|"]
    for cluster, row in results.items():
        route = row["metrics"]["route"]
        selected = {item["rank"]: item for item in route["rows"]}
        global_row = next(item for item in row["metrics"]["global"]["rows"] if item["rank"] == 128)
        lines.append(
            f"| {cluster} | {route['rank90']} | {selected[128]['joint_test_r2']:.3f} / {selected[512]['joint_test_r2']:.3f} / {selected[1152]['joint_test_r2']:.4f} | {selected[128]['context_test_r2']:.3f} / {selected[128]['output_test_r2']:.3f} | {global_row['joint_test_r2']:.3f} |"
        )
    lines += ["", "Rank-128 ridge-sensitivity medians: " + ", ".join(f"`{key}` → **{value:.3f}**" for key, value in ridge_medians.items()) + ".", "", "A full-rank reconstruction is a numerical identity check, not compression. Context-only and output-only curves distinguish whether failure comes from reading the current state, writing the derivative, or needing their joint product."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.2))
    ranks = config["ranks"]
    for cluster, row in results.items():
        route = row["metrics"]["route"]["rows"]
        plots[0].plot(ranks, [item["joint_test_r2"] for item in route], marker="o", label=cluster)
        plots[1].plot(ranks, [item["context_test_r2"] for item in route], marker="o", label=cluster)
        plots[2].plot(ranks, [item["output_test_r2"] for item in route], marker="o", label=cluster)
    for plot, title in zip(plots, ("Joint Tucker", "Context only", "Output only")):
        plot.set_xscale("log", base=2); plot.set_xticks(ranks, ranks, rotation=45); plot.set_ylim(0, 1.02); plot.set_title(title + " held-out R²"); plot.grid(alpha=.2)
    plots[0].legend(fontsize=7, ncol=2)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
