#!/usr/bin/env python3
"""Red-team depth spectra with full-width and activation-weighted operators."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_depth_spectral_redteam.json")
OUTPUT = TENSOR_ROOT / "depth_spectral_redteam.json"
REPORT = TENSOR_ROOT / "DEPTH_SPECTRAL_REDTEAM.md"
FIGURE = TENSOR_ROOT / "depth_spectral_redteam.png"


def symmetric_channel_gram(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Gram of symmetrized channel forms without materializing d x d forms."""
    ll = left @ left.T
    rr = right @ right.T
    lr = left @ right.T
    gram = 0.5 * (ll * rr + lr * lr.T)
    del ll, rr, lr
    return gram


def output_map_eigensystem(
    down: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return output-side singular values/vectors of the quadratic tensor map."""
    channel_gram = symmetric_channel_gram(left, right)
    output_gram = down @ (channel_gram @ down.T)
    output_gram = 0.5 * (output_gram + output_gram.T)
    eigenvalues, vectors = torch.linalg.eigh(output_gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    vectors = vectors[:, order]
    return eigenvalues.sqrt(), vectors


def centered_r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = target - target.mean(0, keepdim=True)
    prediction = prediction - prediction.mean(0, keepdim=True)
    return float(1 - (target - prediction).square().sum() / target.square().sum().clamp_min(1e-30))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    results: dict[str, dict] = {}

    for layer in config["layers"]:
        print("collecting layer", layer, flush=True)
        discovery_x, _ = collect(model, layer, config["discovery"], config)
        evaluation_x, evaluation_write = collect(model, layer, config["evaluation"], config)
        _, discovery_singular, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
        state_v = state_vh.T
        state_rms = discovery_singular / np.sqrt(len(discovery_x))
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float()
        left_weight = block.mlp.Left.weight.float()
        right_weight = block.mlp.Right.weight.float()
        dimension_rows = {}
        raw_full_vectors = None
        weighted_full_vectors = None

        for dimension in config["dimensions"]:
            print(" layer", layer, "dimension", dimension, flush=True)
            raw_basis = state_v[:, :dimension]
            weighted_basis = raw_basis * state_rms[:dimension]
            raw_singular, raw_vectors = output_map_eigensystem(
                down, left_weight @ raw_basis, right_weight @ raw_basis
            )
            weighted_singular, weighted_vectors = output_map_eigensystem(
                down, left_weight @ weighted_basis, right_weight @ weighted_basis
            )
            raw_energy = raw_singular.square()
            weighted_energy = weighted_singular.square()
            projected_x = (evaluation_x @ raw_basis) @ raw_basis.T
            projected_write = block.mlp.Down(
                block.mlp.Left(projected_x) * block.mlp.Right(projected_x)
            )
            row = {
                "dimension": dimension,
                "write_centered_r2": centered_r2(evaluation_write, projected_write),
                "raw_top4_fraction": float(raw_energy[: config["top_components"]].sum() / raw_energy.sum()),
                "weighted_top4_fraction": float(
                    weighted_energy[: config["top_components"]].sum() / weighted_energy.sum()
                ),
                "raw_effective_rank": float(raw_energy.sum().square() / raw_energy.square().sum()),
                "weighted_effective_rank": float(
                    weighted_energy.sum().square() / weighted_energy.square().sum()
                ),
                "raw_top_singular": [float(value) for value in raw_singular[:8]],
                "weighted_top_singular": [float(value) for value in weighted_singular[:8]],
            }
            dimension_rows[str(dimension)] = row
            if dimension == config["dimensions"][-1]:
                raw_full_vectors = raw_vectors[:, : config["top_components"]]
                weighted_full_vectors = weighted_vectors[:, : config["top_components"]]

        overlap = (raw_full_vectors.T @ weighted_full_vectors).abs()
        results[str(layer)] = {
            "dimensions": dimension_rows,
            "full_width_raw_weighted_overlap": overlap.cpu().tolist(),
            "leading_raw_best_weighted_cosine": float(overlap[0].max()),
            "leading_raw_best_weighted_rank": int(overlap[0].argmax()) + 1,
            "state_rms_top8": [float(value) for value in state_rms[:8]],
            "state_rms_effective_rank": float(
                state_rms.square().sum().square() / state_rms.pow(4).sum()
            ),
        }
        del discovery_x, evaluation_x, evaluation_write

    last_dimension = str(config["dimensions"][-1])
    order_raw = sorted(
        results,
        key=lambda layer: results[layer]["dimensions"][last_dimension]["raw_top4_fraction"],
        reverse=True,
    )
    order_weighted = sorted(
        results,
        key=lambda layer: results[layer]["dimensions"][last_dimension]["weighted_top4_fraction"],
        reverse=True,
    )
    hypotheses = {
        "R1_late_raw_concentration": {
            "supported": all(
                results["17"]["dimensions"][str(d)]["raw_top4_fraction"]
                > max(results["0"]["dimensions"][str(d)]["raw_top4_fraction"],
                      results["8"]["dimensions"][str(d)]["raw_top4_fraction"])
                for d in config["dimensions"]
            )
        },
        "R2_late_natural_concentration": {
            "supported": all(
                results["17"]["dimensions"][str(d)]["weighted_top4_fraction"]
                > max(results["0"]["dimensions"][str(d)]["weighted_top4_fraction"],
                      results["8"]["dimensions"][str(d)]["weighted_top4_fraction"])
                for d in config["dimensions"]
            )
        },
        "R3_middle_requires_more_state_dimensions": {
            "supported": results["17"]["dimensions"]["512"]["write_centered_r2"]
            - results["8"]["dimensions"]["512"]["write_centered_r2"] >= 0.10
        },
        "R4_fixed64_understates_early_fidelity": {
            "supported": all(
                results[layer]["dimensions"]["512"]["write_centered_r2"]
                - results[layer]["dimensions"]["64"]["write_centered_r2"] >= 0.20
                for layer in ("0", "8")
            )
        },
        "R5_raw_and_natural_rankings_agree": {
            "supported": order_raw == order_weighted,
            "raw_order": order_raw,
            "weighted_order": order_weighted,
        },
        "R6_top_weight_component_is_naturally_relevant": {
            "supported": all(results[layer]["leading_raw_best_weighted_cosine"] >= 0.70 for layer in results),
            "per_layer": {layer: results[layer]["leading_raw_best_weighted_cosine"] for layer in results},
        },
    }
    payload = {"config": config, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Depth spectral basis-and-scale red-team",
        "",
        "This reruns the layer comparison across state-subspace dimensions and computes the output-side tensor spectrum through an exact channel Gram matrix. `raw` uses orthonormal state-SVD directions; `weighted` scales each direction by its natural RMS coordinate, so its spectrum prioritizes operations encountered on the discovery distribution.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Depth spectral red-team](depth_spectral_redteam.png)",
        "",
        "| Layer | Dimension | Write R² | Raw top-4 | Weighted top-4 | Raw eff. rank | Weighted eff. rank |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, layer_row in results.items():
        for dimension in config["dimensions"]:
            row = layer_row["dimensions"][str(dimension)]
            lines.append(
                f"| {layer} | {dimension} | {row['write_centered_r2']:.3f} | "
                f"{row['raw_top4_fraction']:.3f} | {row['weighted_top4_fraction']:.3f} | "
                f"{row['raw_effective_rank']:.1f} | {row['weighted_effective_rank']:.1f} |"
            )
    lines += [
        "",
        "Full-width raw-versus-weighted output-component overlap:",
        "",
        "| Layer | Best cosine for leading raw component | Weighted rank | State effective rank |",
        "|---:|---:|---:|---:|",
    ]
    for layer, row in results.items():
        lines.append(
            f"| {layer} | {row['leading_raw_best_weighted_cosine']:.3f} | "
            f"{row['leading_raw_best_weighted_rank']} | {row['state_rms_effective_rank']:.1f} |"
        )
    lines += [
        "",
        "The weighted spectrum is not a distribution-free RMT object: it combines weights with the FineWeb state distribution. The raw spectrum is weight-local but still conditioned on the chosen input subspace. Agreement between them identifies high-weight operations that are also naturally exercised.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for layer, row in results.items():
        dimensions = config["dimensions"]
        axes[0].plot(dimensions, [row["dimensions"][str(d)]["write_centered_r2"] for d in dimensions], marker="o", label=f"L{layer}")
        axes[1].plot(dimensions, [row["dimensions"][str(d)]["raw_top4_fraction"] for d in dimensions], marker="o", label=f"L{layer}")
        axes[2].plot(dimensions, [row["dimensions"][str(d)]["weighted_top4_fraction"] for d in dimensions], marker="o", label=f"L{layer}")
    for axis, title, ylabel in zip(
        axes,
        ("Natural write reconstruction", "Raw tensor spectrum", "Activation-weighted spectrum"),
        ("centered R²", "top-four energy fraction", "top-four energy fraction"),
    ):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("state-SVD dimension")
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
