#!/usr/bin/env python3
"""Separate exact native pruning from calibrated directional compression."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_channel_shapley_redteam import shapley_scores
from run_bilin18_layer8_conditional_subspace_quantization import grassmann_similarity, state_centroids
from run_bilin18_layer8_native_channel_program import channel_target, coefficients, first_rank
from run_bilin18_layer8_restricted_tensor_program import prediction_r2
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_channel_calibration_redteam.json")
SOURCE = TENSOR_ROOT / "layer8_multimetric_quantization.npz"
BASE_RESULT = TENSOR_ROOT / "layer8_native_channel_program.json"
OUTPUT = TENSOR_ROOT / "layer8_channel_calibration_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_CHANNEL_CALIBRATION_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer8_channel_calibration_redteam.png"


def selected_prediction(coefficients, down, indices):
    return channel_target(coefficients[:, :, indices], down[:, indices])


def fit_scalar(prediction, target):
    return (prediction * target).sum() / prediction.square().sum().clamp_min(1e-30)


def squared_cosine(prediction, target):
    dot = (prediction * target).sum()
    return float(dot.square() / (prediction.square().sum() * target.square().sum()).clamp_min(1e-30))


def fit_coordinate_map(prediction, target):
    # Each (state, output-coordinate) pair supplies one four-dimensional row.
    design = prediction.permute(0, 2, 1).reshape(-1, prediction.shape[1])
    response = target.permute(0, 2, 1).reshape(-1, target.shape[1])
    return torch.linalg.lstsq(design, response).solution


def apply_coordinate_map(prediction, mapping):
    return torch.einsum("nao,ab->nbo", prediction, mapping)


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
    train_mask = row_ids % 2 == 0; test_mask = ~train_mask
    block = model.blocks[8]
    down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
    results = {}
    for cluster in active:
        train_x = states[(labels == cluster) & train_mask]
        test_x = states[(labels == cluster) & test_mask]
        print("channel calibration", cluster, len(train_x), len(test_x), flush=True)
        edit = basis @ prototypes[cluster]
        train_coefficients = coefficients(left, right, edit, train_x)
        test_coefficients = coefficients(left, right, edit, test_x)
        train_target = channel_target(train_coefficients, down)
        test_target = channel_target(test_coefficients, down)
        order = shapley_scores(train_coefficients, down, train_target).sum(0).argsort(descending=True)
        rows = []
        for rank in config["ranks"]:
            indices = order[:rank]
            train_prediction = selected_prediction(train_coefficients, down, indices)
            test_prediction = selected_prediction(test_coefficients, down, indices)
            scalar = fit_scalar(train_prediction, train_target)
            mapping = fit_coordinate_map(train_prediction, train_target)
            scalar_prediction = scalar * test_prediction
            coordinate_prediction = apply_coordinate_map(test_prediction, mapping)
            rows.append({
                "rank": rank,
                "unscaled_test_r2": prediction_r2(test_target, test_prediction),
                "squared_cosine": squared_cosine(test_prediction, test_target),
                "scalar": float(scalar),
                "scalar_train_r2": prediction_r2(train_target, scalar * train_prediction),
                "scalar_test_r2": prediction_r2(test_target, scalar_prediction),
                "coordinate_train_r2": prediction_r2(train_target, apply_coordinate_map(train_prediction, mapping)),
                "coordinate_test_r2": prediction_r2(test_target, coordinate_prediction),
                "coordinate_map": mapping.cpu().tolist(),
                "coordinate_off_diagonal_fraction": float((mapping - torch.diag(torch.diagonal(mapping))).square().sum() / mapping.square().sum().clamp_min(1e-30)),
            })
        results[str(cluster)] = {"train_sites": int(len(train_x)), "test_sites": int(len(test_x)), "rows": rows, "scalar_rank90": first_rank([{"rank": row["rank"], "r2": row["scalar_test_r2"]} for row in rows]), "coordinate_rank90": first_rank([{"rank": row["rank"], "r2": row["coordinate_test_r2"]} for row in rows])}
    rank256 = [next(item for item in row["rows"] if item["rank"] == 256) for row in results.values()]
    compact_ranks = [min(value for value in (row["scalar_rank90"], row["coordinate_rank90"]) if value is not None) if row["scalar_rank90"] is not None or row["coordinate_rank90"] is not None else None for row in results.values()]
    observed = [rank for rank in compact_ranks if rank is not None]
    hypotheses = {
        "A1_scalar256_is_sufficient": {"supported": sum(row["scalar_test_r2"] >= .9 for row in rank256) >= len(rank256) / 2, "passing": sum(row["scalar_test_r2"] >= .9 for row in rank256)},
        "A2_coordinate256_is_sufficient": {"supported": sum(row["coordinate_test_r2"] >= .9 for row in rank256) >= len(rank256) / 2, "passing": sum(row["coordinate_test_r2"] >= .9 for row in rank256)},
        "A3_scalar_calibration_transfers": {"supported": all(row["scalar_test_r2"] - row["unscaled_test_r2"] >= .1 for row in rank256), "minimum_gain": min(row["scalar_test_r2"] - row["unscaled_test_r2"] for row in rank256)},
        "A4_coordinate_map_adds_value": {"supported": float(np.median([row["coordinate_test_r2"] - row["scalar_test_r2"] for row in rank256])) >= .02, "median_gain": float(np.median([row["coordinate_test_r2"] - row["scalar_test_r2"] for row in rank256]))},
        "A5_rank256_direction_is_high_fidelity": {"supported": all(row["squared_cosine"] >= .9 for row in rank256), "minimum_squared_cosine": min(row["squared_cosine"] for row in rank256)},
        "A6_calibrated_description_is_compact": {"supported": len(observed) == len(compact_ranks) and float(np.median(observed)) <= 256, "ranks90": compact_ranks},
    }
    OUTPUT.write_text(json.dumps({"config": config, "active_prototypes": active, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Calibration red-team for Layer-8 native-channel prefixes", "",
        "Fixed Shapley-ranked prefixes are evaluated three ways: unchanged native amplitudes; one scalar gain fitted on training contexts; and a 4x4 linear map between the four address coordinates fitted on training contexts. The latter is invariant to choosing a different basis inside the same four-dimensional address, but neither calibrated object is a literal native-channel ablation.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Channel calibration red-team](layer8_channel_calibration_redteam.png)", "", "| Address | Scalar/coordinate K90 | R²@256 native / scalar / coordinate | Squared cosine@256 | Scalar@256 | Coordinate off-diagonal |", "|---:|---:|---:|---:|---:|---:|"]
    for cluster, row in results.items():
        selected = next(item for item in row["rows"] if item["rank"] == 256)
        lines.append(f"| {cluster} | {row['scalar_rank90']} / {row['coordinate_rank90']} | {selected['unscaled_test_r2']:.3f} / {selected['scalar_test_r2']:.3f} / {selected['coordinate_test_r2']:.3f} | {selected['squared_cosine']:.3f} | {selected['scalar']:.3f} | {selected['coordinate_off_diagonal_fraction']:.3f} |")
    lines += ["", "Native-prefix sufficiency, directional fidelity, and calibrated surrogate fidelity are separate definitions. A scalar success would mean the long tail mainly supplies gain; a 4x4-only success would mean it also rotates or rescales coordinates inside an otherwise stable address-level operation."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.2))
    for cluster, row in results.items():
        ranks = [item["rank"] for item in row["rows"]]
        plots[0].plot(ranks, [item["unscaled_test_r2"] for item in row["rows"]], marker="o", label=cluster)
        plots[1].plot(ranks, [item["scalar_test_r2"] for item in row["rows"]], marker="o", label=cluster)
        plots[2].plot(ranks, [item["coordinate_test_r2"] for item in row["rows"]], marker="o", label=cluster)
    for plot, title in zip(plots, ("Native prefix", "Scalar calibrated", "4x4 coordinate calibrated")):
        plot.set_xscale("log", base=2); plot.set_ylim(0, 1.02); plot.set_title(title); plot.set_xlabel("channels"); plot.set_ylabel("held-out derivative R²"); plot.grid(alpha=.2)
    plots[0].legend(fontsize=7, ncol=2)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
