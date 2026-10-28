#!/usr/bin/env python3
"""Red-team cleaned spectra across covariance sample size and ridge strength."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_cleaning_fourth_moment_redteam import moment_matched_gaussian, rank_scores
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_cleaning_stability_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_cleaning_stability_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_CLEANING_STABILITY_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_cleaning_stability_redteam.png"


def correlation(first, second):
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    first -= first.mean(); second -= second.mean()
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    results = {}
    for layer in config["layers"]:
        print("cleaning stability layer", layer, flush=True)
        maximum_x, _ = collect_states_writes(model, layer, config["discovery"], config["positions"], config["batch_size"])
        evaluation_x, _ = collect_states_writes(model, layer, config["evaluation"], config["positions"], config["batch_size"])
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
        _, output_basis = output_map_eigensystem(down, left, right)
        raw_forms = []
        for component in range(config["components"]):
            coefficients = down.T @ output_basis[:, component]
            raw = left.T @ (coefficients[:, None] * right)
            raw_forms.append(0.5 * (raw + raw.T))
        settings = {}
        for state_count in config["discovery_state_counts"]:
            discovery_x = maximum_x[:state_count]
            for ridge in config["ridge_fractions"]:
                key = f"n{state_count}_r{ridge:g}"
                square, inverse, covariance = ridge_covariance_transforms(discovery_x, ridge)
                discovery_z = discovery_x @ inverse
                evaluation_z = evaluation_x @ inverse
                gaussian_z = moment_matched_gaussian(discovery_z, config["gaussian_samples"], generator)
                evaluation_second = evaluation_z.T @ evaluation_z / len(evaluation_z)
                second_values = torch.linalg.eigvalsh(0.5 * (evaluation_second + evaluation_second.T))
                mismatch = float((evaluation_second - torch.eye(len(evaluation_second), device="cuda")).norm() / len(evaluation_second) ** .5)
                components = []
                for raw in raw_forms:
                    weighted = square @ raw @ square
                    natural, energy = rank_scores(evaluation_z, weighted, [config["rank"]])
                    gaussian, _ = rank_scores(gaussian_z, weighted, [config["rank"]])
                    components.append({"matrix_energy": energy[0], "natural_correlation": natural[0]["correlation"], "natural_r2": natural[0]["r2"], "gaussian_correlation": gaussian[0]["correlation"], "gaussian_r2": gaussian[0]["r2"], "natural_residual_variance": natural[0]["residual_variance_fraction"]})
                settings[key] = {
                    "state_count": state_count, "ridge_fraction": ridge, "covariance": covariance,
                    "evaluation_second_moment_mismatch": mismatch,
                    "evaluation_second_eigen_min": float(second_values.min()),
                    "evaluation_second_eigen_max": float(second_values.max()),
                    "components": components,
                    "median_natural_correlation": float(np.median([row["natural_correlation"] for row in components])),
                    "median_natural_r2": float(np.median([row["natural_r2"] for row in components])),
                    "median_gaussian_r2": float(np.median([row["gaussian_r2"] for row in components])),
                }
        results[str(layer)] = {"settings": settings}
    layers = [str(value) for value in config["layers"]]
    def setting(layer, count, ridge): return results[layer]["settings"][f"n{count}_r{ridge:g}"]
    all_settings = [(layer, row) for layer in layers for row in results[layer]["settings"].values()]
    mismatches = [row["evaluation_second_moment_mismatch"] for _, row in all_settings]
    failures = [-row["median_natural_r2"] for _, row in all_settings]
    hypotheses = {
        "S1_more_discovery_data_improves_natural_fit": {"supported": all(setting(layer, 8192, 1e-4)["median_natural_r2"] > setting(layer, 1024, 1e-4)["median_natural_r2"] for layer in layers)},
        "S2_ridge_stabilizes_small_sample_fit": {"supported": all(setting(layer, 1024, .01)["median_natural_r2"] > setting(layer, 1024, 1e-4)["median_natural_r2"] for layer in layers)},
        "S3_stable_cleaning_reaches_high_correlation": {"supported": all(max(setting(layer, 8192, ridge)["median_natural_correlation"] for ridge in config["ridge_fractions"]) >= .9 for layer in layers)},
        "S4_natural_gaussian_gap_survives": {"supported": all(setting(layer, 8192, .01)["median_gaussian_r2"] - setting(layer, 8192, .01)["median_natural_r2"] >= .2 for layer in layers)},
        "S5_no_universal_rank4_setting": {"supported": all(not all(all(component["natural_correlation"] >= .9 for component in setting(layer, count, ridge)["components"]) for layer in layers) for count in config["discovery_state_counts"] for ridge in config["ridge_fractions"])},
        "S6_second_moment_mismatch_predicts_failure": {"supported": correlation(mismatches, failures) >= .5, "correlation": correlation(mismatches, failures)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Sample-size and ridge red-team of covariance cleaning", "",
        "The independent checkpoint's cleaned-form anomaly is repeated over 1,024, 4,096, and 8,192 FineWeb discovery states and ridge fractions from `1e-4` to `0.1`. Held-out whitened second-moment mismatch is measured explicitly, so covariance-estimation failure can be separated from residual higher-order effects.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Cleaning stability](bilin12_cleaning_stability_redteam.png)", "", "| Layer | States | Ridge | Second-moment mismatch | Natural corr / R² | Gaussian R² |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in layers:
        for key, row in results[layer]["settings"].items():
            lines.append(f"| {layer} | {row['state_count']} | {row['ridge_fraction']:.4g} | {row['evaluation_second_moment_mismatch']:.2f} | {row['median_natural_correlation']:.3f} / {row['median_natural_r2']:.3f} | {row['median_gaussian_r2']:.3f} |")
    lines += ["", "A stable covariance metric requires both enough discovery states and adequate regularization. Any remaining natural-versus-Gaussian gap after second-moment stabilization is evidence about empirical fourth moments; the unstable small-sample gap is not."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.2))
    for layer in layers:
        for ridge in config["ridge_fractions"]:
            rows = [setting(layer, count, ridge) for count in config["discovery_state_counts"]]
            plots[0].plot(config["discovery_state_counts"], [row["median_natural_correlation"] for row in rows], marker="o", label=f"L{layer} r{ridge:g}")
            plots[1].plot(config["discovery_state_counts"], [row["median_natural_r2"] for row in rows], marker="o", label=f"L{layer} r{ridge:g}")
    plots[2].scatter(mismatches, failures)
    for plot, title in zip(plots[:2], ("Natural rank-4 correlation", "Natural rank-4 R²")):
        plot.set_xscale("log", base=2); plot.set_xlabel("discovery states"); plot.set_title(title); plot.grid(alpha=.2)
    plots[0].legend(fontsize=6, ncol=2); plots[2].set_xlabel("held-out second-moment mismatch"); plots[2].set_ylabel("negative natural R²"); plots[2].set_title("Covariance error vs score failure"); plots[2].grid(alpha=.2)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
