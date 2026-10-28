#!/usr/bin/env python3
"""Compare centered Gaussian-moment tensor spectra with empirical MLP writes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_spectral_redteam import symmetric_channel_gram
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_depth_natural_spectrum.json")
OUTPUT = TENSOR_ROOT / "depth_natural_spectrum.json"
REPORT = TENSOR_ROOT / "DEPTH_NATURAL_SPECTRUM.md"
FIGURE = TENSOR_ROOT / "depth_natural_spectrum.png"


def eigensystem(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = torch.linalg.eigh(matrix)
    order = torch.argsort(values, descending=True)
    return values[order].clamp_min(0), vectors[:, order]


def covariance_summary(values: torch.Tensor, top: int) -> dict:
    total = values.sum().clamp_min(1e-30)
    return {
        "top4_fraction": float(values[:top].sum() / total),
        "effective_rank": float(total.square() / values.square().sum().clamp_min(1e-30)),
        "top8_values": [float(value) for value in values[:8]],
    }


def matrix_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        (first * second).sum()
        / (first.norm() * second.norm()).clamp_min(1e-30)
    )


def gaussian_output_covariance(
    down: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    mean: torch.Tensor,
    covariance_factor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Output covariance if centered whitened coordinates were standard Gaussian.

    For x=mean+Bz, each channel has a quadratic z^T A z and a linear g^T z
    term. Gaussian odd moments vanish, so Cov = 2<A,A'> + <g,g'>.
    """
    left = left_weight @ covariance_factor
    right = right_weight @ covariance_factor
    quadratic_gram = symmetric_channel_gram(left, right)
    left_mean = left_weight @ mean
    right_mean = right_weight @ mean
    linear = left_mean[:, None] * right + right_mean[:, None] * left
    linear_gram = linear @ linear.T
    quadratic_covariance = down @ (2 * quadratic_gram @ down.T)
    full_covariance = quadratic_covariance + down @ (linear_gram @ down.T)
    return 0.5 * (full_covariance + full_covariance.T), 0.5 * (
        quadratic_covariance + quadratic_covariance.T
    )


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    results: dict[str, dict] = {}

    for layer in config["layers"]:
        print("collecting layer", layer, flush=True)
        discovery_x, discovery_write = collect(model, layer, config["discovery"], config)
        evaluation_x, evaluation_write = collect(model, layer, config["evaluation"], config)
        mean = discovery_x.mean(0)
        centered_x = discovery_x - mean
        _, state_singular, state_vh = torch.linalg.svd(centered_x, full_matrices=False)
        state_v = state_vh.T
        state_std = state_singular / np.sqrt(len(centered_x))

        discovery_centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
        evaluation_centered_write = evaluation_write - evaluation_write.mean(0, keepdim=True)
        discovery_covariance = discovery_centered_write.T @ discovery_centered_write / len(discovery_centered_write)
        evaluation_covariance = evaluation_centered_write.T @ evaluation_centered_write / len(evaluation_centered_write)
        discovery_values, discovery_vectors = eigensystem(discovery_covariance)
        evaluation_values, evaluation_vectors = eigensystem(evaluation_covariance)
        discovery_summary = covariance_summary(discovery_values, config["top_components"])
        evaluation_summary = covariance_summary(evaluation_values, config["top_components"])
        top_discovery = discovery_vectors[:, : config["top_components"]]
        captured_evaluation = torch.trace(top_discovery.T @ evaluation_covariance @ top_discovery)
        optimal_evaluation = evaluation_values[: config["top_components"]].sum()

        block = model.blocks[layer]
        down = block.mlp.Down.weight.float()
        left_weight = block.mlp.Left.weight.float()
        right_weight = block.mlp.Right.weight.float()
        dimensions = {}
        for dimension in config["dimensions"]:
            print(" layer", layer, "dimension", dimension, flush=True)
            covariance_factor = state_v[:, :dimension] * state_std[:dimension]
            predicted_covariance, quadratic_only = gaussian_output_covariance(
                down, left_weight, right_weight, mean, covariance_factor
            )
            predicted_values, predicted_vectors = eigensystem(predicted_covariance)
            quadratic_values, _ = eigensystem(quadratic_only)
            canonical = torch.linalg.svdvals(
                predicted_vectors[:, : config["top_components"]].T @ top_discovery
            )
            dimensions[str(dimension)] = {
                **covariance_summary(predicted_values, config["top_components"]),
                "quadratic_only_top4_fraction": covariance_summary(
                    quadratic_values, config["top_components"]
                )["top4_fraction"],
                "covariance_cosine": matrix_cosine(predicted_covariance, discovery_covariance),
                "quadratic_only_covariance_cosine": matrix_cosine(quadratic_only, discovery_covariance),
                "top4_canonical_cosines": [float(value) for value in canonical],
                "mean_top4_canonical_cosine": float(canonical.mean()),
            }

        results[str(layer)] = {
            "discovery_empirical": discovery_summary,
            "evaluation_empirical": evaluation_summary,
            "discovery_to_evaluation_top4_efficiency": float(captured_evaluation / optimal_evaluation),
            "discovery_evaluation_top4_canonical_cosines": [
                float(value)
                for value in torch.linalg.svdvals(
                    top_discovery.T @ evaluation_vectors[:, : config["top_components"]]
                )
            ],
            "mean_state_norm": float(mean.norm()),
            "centered_state_effective_rank": float(
                state_std.square().sum().square() / state_std.pow(4).sum()
            ),
            "dimensions": dimensions,
        }
        del discovery_x, discovery_write, evaluation_x, evaluation_write

    full = str(config["dimensions"][-1])
    hypotheses = {
        "C1_late_empirical_concentration": {
            "supported": all(
                results["17"][split]["top4_fraction"]
                - max(results["0"][split]["top4_fraction"], results["8"][split]["top4_fraction"])
                >= 0.10
                for split in ("discovery_empirical", "evaluation_empirical")
            )
        },
        "C2_late_gaussian_concentration": {
            "supported": results["17"]["dimensions"][full]["top4_fraction"]
            - max(results["0"]["dimensions"][full]["top4_fraction"],
                  results["8"]["dimensions"][full]["top4_fraction"]) >= 0.10
        },
        "C3_gaussian_subspace_alignment": {
            "supported": all(
                results[layer]["dimensions"][full]["mean_top4_canonical_cosine"] >= 0.70
                for layer in results
            )
        },
        "C4_gaussian_covariance_prediction": {
            "supported": all(
                results[layer]["dimensions"][full]["covariance_cosine"] >= 0.80
                for layer in results
            )
        },
        "C5_empirical_subspace_transfer": {
            "supported": all(
                results[layer]["discovery_to_evaluation_top4_efficiency"] >= 0.90
                for layer in results
            )
        },
        "C6_mean_cross_term_matters": {
            "supported": any(
                results[layer]["dimensions"][full]["covariance_cosine"]
                - results[layer]["dimensions"][full]["quadratic_only_covariance_cosine"] >= 0.10
                for layer in results
            )
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "results": results, "hypotheses": hypotheses}, indent=2))

    lines = [
        "# Centered natural-write spectrum across depth",
        "",
        "This replaces uncentered activation weighting with a centered calculation. For `x = mean + Bz`, it includes both the quadratic bilinear term and the mean-state linear cross term, computes the output covariance implied by Gaussian `z`, and compares it with actual centered MLP-write covariance.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Natural spectrum](depth_natural_spectrum.png)",
        "",
        "| Layer | Empirical top-4 discovery / eval | Empirical eff. rank discovery / eval | Transfer efficiency | Centered-state eff. rank |",
        "|---:|---:|---:|---:|---:|",
    ]
    for layer, row in results.items():
        lines.append(
            f"| {layer} | {row['discovery_empirical']['top4_fraction']:.3f} / {row['evaluation_empirical']['top4_fraction']:.3f} | "
            f"{row['discovery_empirical']['effective_rank']:.1f} / {row['evaluation_empirical']['effective_rank']:.1f} | "
            f"{row['discovery_to_evaluation_top4_efficiency']:.3f} | {row['centered_state_effective_rank']:.1f} |"
        )
    lines += [
        "",
        "| Layer | Dimension | Gaussian top-4 | Gaussian eff. rank | Covariance cosine | Quadratic-only cosine | Top-4 subspace cosine |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, row in results.items():
        for dimension in config["dimensions"]:
            item = row["dimensions"][str(dimension)]
            lines.append(
                f"| {layer} | {dimension} | {item['top4_fraction']:.3f} | {item['effective_rank']:.1f} | "
                f"{item['covariance_cosine']:.3f} | {item['quadratic_only_covariance_cosine']:.3f} | "
                f"{item['mean_top4_canonical_cosine']:.3f} |"
            )
    lines += [
        "",
        "The Gaussian calculation is a moment-matched model, not a claim that residual states are Gaussian. Its failures localize non-Gaussian or higher-order state structure that a cleaned weight spectrum omits; empirical write PCA is the authoritative natural-distribution comparison.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = list(results)
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, [results[l]["discovery_empirical"]["top4_fraction"] for l in labels], 0.36, label="discovery")
    axes[0].bar(x + 0.18, [results[l]["evaluation_empirical"]["top4_fraction"] for l in labels], 0.36, label="evaluation")
    axes[0].set_xticks(x, [f"L{layer}" for layer in labels])
    axes[0].set_ylabel("top-four variance fraction")
    axes[0].set_title("Empirical natural writes")
    axes[0].legend()
    for layer, row in results.items():
        dimensions = config["dimensions"]
        axes[1].plot(dimensions, [row["dimensions"][str(d)]["top4_fraction"] for d in dimensions], marker="o", label=f"L{layer}")
        axes[2].plot(dimensions, [row["dimensions"][str(d)]["covariance_cosine"] for d in dimensions], marker="o", label=f"L{layer}")
    for axis, title, ylabel in (
        (axes[1], "Centered Gaussian spectrum", "top-four fraction"),
        (axes[2], "Gaussian vs empirical", "covariance cosine"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("centered state dimension")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
