#!/usr/bin/env python3
"""Test covariance-cleaned tensor energy against empirical fourth moments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer17_tail_cleaning_sweep import trace_clean
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_cleaning_fourth_moment_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_cleaning_fourth_moment_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_CLEANING_FOURTH_MOMENT_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_cleaning_fourth_moment_redteam.png"


def score_metrics(target, prediction):
    target = target.double(); prediction = prediction.double()
    centered_target = target - target.mean(); centered_prediction = prediction - prediction.mean()
    correlation = centered_target @ centered_prediction / (centered_target.norm() * centered_prediction.norm()).clamp_min(1e-30)
    r2 = 1 - (centered_target - centered_prediction).square().sum() / centered_target.square().sum().clamp_min(1e-30)
    return float(correlation), float(r2)


def rank_scores(coordinates, matrix, ranks):
    clean, isotropic = trace_clean(matrix)
    values, vectors = torch.linalg.eigh(clean)
    order = values.abs().argsort(descending=True)
    values = values[order]; vectors = vectors[:, order]
    projection = coordinates @ vectors
    contributions = projection.square() * values[None]
    isotropic_score = isotropic * coordinates.square().sum(1)
    full = isotropic_score + contributions.sum(1)
    rows = []
    for rank in ranks:
        prediction = isotropic_score + contributions[:, :rank].sum(1)
        correlation, r2 = score_metrics(full, prediction)
        residual = full - prediction
        rows.append({
            "rank": rank, "correlation": correlation, "r2": r2,
            "residual_variance_fraction": float((residual - residual.mean()).square().sum() / (full - full.mean()).square().sum().clamp_min(1e-30)),
        })
    energy = values.square()
    return rows, [float(energy[:rank].sum() / energy.sum().clamp_min(1e-30)) for rank in ranks]


def first_rank(rows, key, threshold):
    for row in rows:
        if row[key] >= threshold:
            return row["rank"]
    return None


def moment_matched_gaussian(reference, samples, generator):
    # reference has uncentered second moment approximately I after whitening.
    mean = reference.mean(0)
    norm = mean.norm().clamp_max(.999999)
    gaussian = torch.randn(samples, reference.shape[1], generator=generator, device=reference.device)
    if norm > 1e-8:
        direction = mean / norm
        coefficient = gaussian @ direction
        gaussian += ((1 - norm.square()).sqrt() - 1) * coefficient[:, None] * direction[None]
    return gaussian + mean


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    results = {}
    for layer in config["layers"]:
        print("fourth-moment layer", layer, flush=True)
        discovery_x, _ = collect_states_writes(model, layer, config["discovery"], config["positions"], config["batch_size"])
        evaluation_x, _ = collect_states_writes(model, layer, config["evaluation"], config["positions"], config["batch_size"])
        square, inverse, covariance = ridge_covariance_transforms(discovery_x, config["covariance_ridge_fraction"])
        discovery_z = discovery_x @ inverse
        evaluation_z = evaluation_x @ inverse
        gaussian_z = moment_matched_gaussian(discovery_z, config["gaussian_samples"], generator)
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
        _, output_basis = output_map_eigensystem(down, left, right)
        components = []
        for component in range(config["components"]):
            coefficients = down.T @ output_basis[:, component]
            raw = left.T @ (coefficients[:, None] * right)
            raw = 0.5 * (raw + raw.T)
            weighted = square @ raw @ square
            discovery_rows, matrix_energy = rank_scores(discovery_z, weighted, config["ranks"])
            natural_rows, _ = rank_scores(evaluation_z, weighted, config["ranks"])
            gaussian_rows, _ = rank_scores(gaussian_z, weighted, config["ranks"])
            rank4_index = config["ranks"].index(4)
            tail_energy = 1 - matrix_energy[rank4_index]
            components.append({
                "component": component + 1,
                "matrix_energy": matrix_energy,
                "discovery": discovery_rows, "natural": natural_rows, "gaussian": gaussian_rows,
                "natural_rank_corr90": first_rank(natural_rows, "correlation", .9),
                "natural_rank_r280": first_rank(natural_rows, "r2", .8),
                "gaussian_rank_corr90": first_rank(gaussian_rows, "correlation", .9),
                "tail_matrix_energy4": tail_energy,
                "natural_tail_amplification4": natural_rows[rank4_index]["residual_variance_fraction"] / max(tail_energy, 1e-30),
            })
        results[str(layer)] = {"covariance": covariance, "components": components}
    layers = [str(value) for value in config["layers"]]
    rank4 = config["ranks"].index(4); full = len(config["ranks"]) - 1
    hypotheses = {
        "Q1_gaussian_rank4_matches_matrix_energy": {"supported": all(abs(row["gaussian"][rank4]["r2"] - row["matrix_energy"][rank4]) <= .1 for layer in layers for row in results[layer]["components"])},
        "Q2_natural_rank4_is_worse": {"supported": all(float(np.median([row["gaussian"][rank4]["r2"] - row["natural"][rank4]["r2"] for row in results[layer]["components"]])) >= .2 for layer in layers)},
        "Q3_natural_scores_need_more_modes": {"supported": all(float(np.median([row["natural_rank_corr90"] for row in results[layer]["components"]])) > 4 for layer in layers)},
        "Q4_low_energy_tail_is_naturally_amplified": {"supported": all(float(np.median([row["natural_tail_amplification4"] for row in results[layer]["components"]])) >= 2 for layer in layers)},
        "Q5_full_rank_reconstructs_both": {"supported": all(row[distribution][full]["r2"] >= .999 for layer in layers for row in results[layer]["components"] for distribution in ("natural", "gaussian"))},
        "Q6_fourth_moment_gap_is_depth_general": {"supported": all(row["natural"][rank4]["r2"] < row["gaussian"][rank4]["r2"] for layer in layers for row in results[layer]["components"])},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Fourth-moment red-team of covariance-cleaned spectra", "",
        "Each raw outlier form is expressed in a FineWeb second-moment-whitened coordinate system. The same frozen spectral truncations are scored on disjoint FineWeb states and on Gaussian states matched to the whitened discovery mean and second moment. Under a Gaussian model, trace-cleaned Frobenius energy should predict quadratic-score variance; a natural-data gap diagnoses higher-order structure rather than an algebra or transform error.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Fourth-moment red-team](bilin12_cleaning_fourth_moment_redteam.png)", "", "| Layer | Median matrix energy@4 | Natural / Gaussian R²@4 | Natural corr@4 | Natural rank corr≥.90 | Tail amplification |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in layers:
        rows = results[layer]["components"]
        lines.append(f"| {layer} | {np.median([row['matrix_energy'][rank4] for row in rows]):.3f} | {np.median([row['natural'][rank4]['r2'] for row in rows]):.3f} / {np.median([row['gaussian'][rank4]['r2'] for row in rows]):.3f} | {np.median([row['natural'][rank4]['correlation'] for row in rows]):.3f} | {np.median([row['natural_rank_corr90'] for row in rows]):.0f} | {np.median([row['natural_tail_amplification4'] for row in rows]):.1f}× |")
    lines += ["", "Covariance-cleaned Frobenius compactness is a Gaussian/second-moment statement. Empirical quadratic-score compactness is a fourth-moment statement. They are distinct explanatory objects even when the whitening transform is exact."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.2))
    for layer in layers:
        rows = results[layer]["components"]
        ranks = config["ranks"]
        plots[0].plot(ranks, np.median([[row["natural"][i]["r2"] for i in range(len(ranks))] for row in rows], 0), marker="o", label=f"L{layer}")
        plots[1].plot(ranks, np.median([[row["gaussian"][i]["r2"] for i in range(len(ranks))] for row in rows], 0), marker="o", label=f"L{layer}")
    plots[2].bar(layers, [np.median([row["natural_tail_amplification4"] for row in results[layer]["components"]]) for layer in layers])
    for plot, title in zip(plots[:2], ("Natural FineWeb", "Moment-matched Gaussian")):
        plot.set_xscale("log", base=2); plot.set_ylim(-1, 1.05); plot.set_title(title + " score R²"); plot.set_xlabel("spectral rank"); plot.grid(alpha=.2); plot.legend()
    plots[2].set_title("Rank-4 tail variance amplification"); plots[2].set_xlabel("layer"); plots[2].set_ylabel("natural score variance / matrix energy"); plots[2].grid(alpha=.2)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
