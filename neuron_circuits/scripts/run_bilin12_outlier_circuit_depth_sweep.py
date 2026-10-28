#!/usr/bin/env python3
"""Test whether bilinear tensor outliers predict compact native circuits across depth."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_layer5_address_channel_decomposition import channel_decomposition
from run_bilin12_layer5_weight_conditioned_addresses import (
    derive_cleaned_addresses,
    first_energy_rank,
    output_conditioned_form,
    restricted_operator,
)
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_outlier_circuit_depth_sweep.json")
OUTPUT = TENSOR_ROOT / "bilin12_outlier_circuit_depth_sweep.json"
REPORT = TENSOR_ROOT / "BILIN12_OUTLIER_CIRCUIT_DEPTH_SWEEP.md"
FIGURE = TENSOR_ROOT / "bilin12_outlier_circuit_depth_sweep.png"


def spearman(first, second) -> float:
    first = np.asarray(first)
    second = np.asarray(second)
    rank_first = np.empty(len(first), dtype=float)
    rank_second = np.empty(len(second), dtype=float)
    rank_first[np.argsort(first)] = np.arange(len(first))
    rank_second[np.argsort(second)] = np.arange(len(second))
    return float(np.corrcoef(rank_first, rank_second)[0, 1])


def powerlaw_fit(singular: torch.Tensor, start: int, stop: int) -> dict:
    values = singular.detach().cpu().double().numpy()
    ranks = np.arange(1, len(values) + 1)
    selected = slice(start - 1, stop)
    slope, intercept = np.polyfit(np.log(ranks[selected]), np.log(values[selected]), 1)
    prediction = intercept + slope * np.log(ranks[selected])
    target = np.log(values[selected])
    r2 = 1 - np.square(target - prediction).sum() / np.square(target - target.mean()).sum()
    leading_prediction = intercept + slope * np.log(1.0)
    return {
        "alpha": float(-slope),
        "r_squared": float(r2),
        "leading_log_excess": float(np.log(values[0]) - leading_prediction),
    }


def pairwise_jaccard(sets: list[set[int]]) -> float:
    rows = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    return float(np.mean(rows))


def address_metrics(edit, origin_output, down, left, right, square) -> dict:
    operator = restricted_operator(down, left, right, edit)
    weighted = operator @ square
    singular = torch.linalg.svdvals(weighted)
    context_rows = (
        (left @ edit)[:, None] * right + (right @ edit)[:, None] * left
    ) @ square
    decomposition = channel_decomposition(down, context_rows, [8, 32, 128])
    energy = weighted.square().sum()
    output_role = (origin_output @ weighted).square().sum() / energy.clamp_min(1e-30)
    return {
        "operator_k90": first_energy_rank(singular, 0.90),
        "operator_effective_rank": float(singular.square().sum().square() / singular.pow(4).sum().clamp_min(1e-30)),
        "shapley_k90": decomposition["shapley_k90"],
        "shapley_effective_support": decomposition["absolute_shapley_effective_support"],
        "largest_shapley_share": decomposition["largest_absolute_shapley_share"],
        "individual_to_total_energy": decomposition["individual_to_total_energy_ratio"],
        "output_role_fraction": float(output_role),
        "top8_channels": decomposition["top_channel_indices"][:8],
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    results = {}
    for layer in config["layers"]:
        print("layer", layer, flush=True)
        states, writes = collect_states_writes(
            model, layer, config["discovery"], config["positions"], config["batch_size"]
        )
        square, _, covariance = ridge_covariance_transforms(
            states, config["covariance_ridge_fraction"]
        )
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float()
        left = block.mlp.Left.weight.float()
        right = block.mlp.Right.weight.float()
        singular, output_vectors = output_map_eigensystem(down, left, right)
        spectrum = powerlaw_fit(singular, config["bulk_rank_start"], config["bulk_rank_stop"])
        output = output_vectors[:, 0]

        centered_write = writes - writes.mean(0, keepdim=True)
        total_variance = centered_write.square().sum()
        raw_variance = (centered_write @ output).square().sum()
        covariance_write = centered_write.T @ centered_write
        optimal_variance = torch.linalg.eigvalsh(covariance_write)[-1]
        natural_capture = raw_variance / total_variance.clamp_min(1e-30)
        natural_efficiency = raw_variance / optimal_variance.clamp_min(1e-30)

        form = output_conditioned_form(down, left, right, output)
        _, _, edits = derive_cleaned_addresses(form, square, config["address_count"])
        clean_rows = [
            address_metrics(edits[:, index], output, down, left, right, square)
            for index in range(config["address_count"])
        ]

        coefficients = down.T @ output
        sign_rows = []
        for _ in range(config["sign_null_count"]):
            signs = torch.where(
                torch.rand(len(coefficients), generator=generator, device="cuda") < 0.5,
                -torch.ones((), device="cuda"), torch.ones((), device="cuda"),
            )
            null_form = left.T @ ((coefficients * signs)[:, None] * right)
            null_form = 0.5 * (null_form + null_form.T)
            _, _, null_edit = derive_cleaned_addresses(null_form, square, 1)
            sign_rows.append(address_metrics(null_edit[:, 0], output, down, left, right, square))

        def median(key, rows):
            return float(np.median([row[key] for row in rows]))

        clean_sets = [set(row["top8_channels"]) for row in clean_rows]
        sign_sets = [set(row["top8_channels"]) for row in sign_rows]
        results[str(layer)] = {
            **spectrum,
            "leading_to_second_singular_ratio": float(singular[0] / singular[1]),
            "natural_rank1_capture": float(natural_capture),
            "natural_rank1_efficiency": float(natural_efficiency),
            "covariance": covariance,
            "clean_summary": {
                "median_operator_k90": median("operator_k90", clean_rows),
                "median_operator_effective_rank": median("operator_effective_rank", clean_rows),
                "median_shapley_k90": median("shapley_k90", clean_rows),
                "median_shapley_effective_support": median("shapley_effective_support", clean_rows),
                "median_largest_shapley_share": median("largest_shapley_share", clean_rows),
                "median_individual_to_total_energy": median("individual_to_total_energy", clean_rows),
                "median_output_role_fraction": median("output_role_fraction", clean_rows),
                "top8_recurrence": pairwise_jaccard(clean_sets),
            },
            "sign_null_summary": {
                "median_operator_k90": median("operator_k90", sign_rows),
                "median_shapley_k90": median("shapley_k90", sign_rows),
                "median_output_role_fraction": median("output_role_fraction", sign_rows),
                "top8_recurrence": pairwise_jaccard(sign_sets),
            },
            "clean_addresses": clean_rows,
            "sign_nulls": sign_rows,
        }
        del states, writes, square, centered_write, covariance_write
        torch.cuda.empty_cache()

    ordered = [results[str(layer)] for layer in config["layers"]]
    excess = [row["leading_log_excess"] for row in ordered]
    efficiency = [row["natural_rank1_efficiency"] for row in ordered]
    operator_k90 = [row["clean_summary"]["median_operator_k90"] for row in ordered]
    channel_k90 = [row["clean_summary"]["median_shapley_k90"] for row in ordered]
    correlations = {
        "excess_vs_natural_efficiency_spearman": spearman(excess, efficiency),
        "excess_vs_operator_k90_spearman": spearman(excess, operator_k90),
        "excess_vs_channel_k90_spearman": spearman(excess, channel_k90),
        "natural_efficiency_vs_channel_k90_spearman": spearman(efficiency, channel_k90),
    }
    strongest = np.argsort(excess)[-3:][::-1].tolist()
    mismatch_layers = [
        layer for layer, row in zip(config["layers"], ordered)
        if row["clean_summary"]["median_operator_k90"] <= 64
        and row["clean_summary"]["median_shapley_k90"]
        >= 4 * row["clean_summary"]["median_operator_k90"]
    ]
    distributed_layers = [
        layer for layer, row in zip(config["layers"], ordered)
        if row["clean_summary"]["median_shapley_k90"]
        >= 4 * row["sign_null_summary"]["median_shapley_k90"]
    ]
    hypotheses = {
        "D1_powerlaw_bulk_holds_at_all_layers": {"supported": all(row["r_squared"] >= 0.90 for row in ordered)},
        "D2_outlier_excess_predicts_natural_use": {"supported": correlations["excess_vs_natural_efficiency_spearman"] >= 0.60},
        "D3_outlier_excess_predicts_operator_compactness": {"supported": correlations["excess_vs_operator_k90_spearman"] <= -0.50},
        "D4_strongest_outliers_are_small_native_circuits": {"supported": all(ordered[layer]["clean_summary"]["median_shapley_k90"] <= 64 for layer in strongest), "strongest_layers": strongest},
        "D5_compact_operator_broad_channel_mismatch_is_common": {"supported": len(mismatch_layers) >= 6, "layers": mismatch_layers},
        "D6_clean_address_recurrence_beats_sign_nulls": {"supported": all(row["clean_summary"]["top8_recurrence"] > row["sign_null_summary"]["top8_recurrence"] for row in ordered)},
        "D7_clean_addresses_are_more_distributed_than_sign_nulls": {"supported": len(distributed_layers) >= 6, "layers": distributed_layers},
        "D8_strong_outliers_are_output_pure": {"supported": all(ordered[layer]["clean_summary"]["median_output_role_fraction"] >= 0.50 for layer in strongest), "strongest_layers": strongest},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses, "correlations": correlations,
        "strongest_layers": strongest, "results": results,
    }, indent=2, allow_nan=False))

    layers = config["layers"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes[0, 0].plot(layers, excess, marker="o", label="log excess")
    axes[0, 0].plot(layers, efficiency, marker="s", label="natural efficiency")
    axes[0, 0].set_title("Outlier strength and natural use"); axes[0, 0].legend()
    axes[0, 1].plot(layers, operator_k90, marker="o", label="operator K90")
    axes[0, 1].plot(layers, channel_k90, marker="s", label="channel K90")
    axes[0, 1].set_yscale("log"); axes[0, 1].set_title("Two kinds of support"); axes[0, 1].legend()
    axes[1, 0].plot(layers, [r["clean_summary"]["top8_recurrence"] for r in ordered], marker="o", label="clean")
    axes[1, 0].plot(layers, [r["sign_null_summary"]["top8_recurrence"] for r in ordered], marker="s", label="sign null")
    axes[1, 0].set_title("Top-eight recurrence"); axes[1, 0].legend()
    axes[1, 1].plot(layers, [r["clean_summary"]["median_output_role_fraction"] for r in ordered], marker="o")
    axes[1, 1].axhline(.5, color="black", ls="--", lw=1); axes[1, 1].set_title("Leading-output role fraction")
    for axis in axes.flat:
        axis.grid(alpha=.25); axis.set_xlabel("layer")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = [
        "# Outlier-to-circuit depth sweep in the independent 12-layer model", "",
        "At every layer, the raw tensor's leading output mode is converted into four covariance-cleaned input addresses. Their exact restricted derivative operators are measured both in singular coordinates and as sums of native rank-one MLP channels. Four channel-sign nulls preserve all channel magnitudes but destroy coordinated signs.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Depth sweep](bilin12_outlier_circuit_depth_sweep.png)", "", f"Cross-layer correlations: `{correlations}`. Strongest leading-excess layers: `{strongest}`.", "",
              "| Layer | Lead excess | Natural eff. | Operator K90 | Channel K90 | Clean/null recurrence | Output role |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for layer, row in zip(layers, ordered):
        clean = row["clean_summary"]; null = row["sign_null_summary"]
        lines.append(f"| {layer} | {row['leading_log_excess']:.3f} | {row['natural_rank1_efficiency']:.3f} | {clean['median_operator_k90']:.1f} | {clean['median_shapley_k90']:.1f} | {clean['top8_recurrence']:.3f}/{null['top8_recurrence']:.3f} | {clean['median_output_role_fraction']:.3f} |")
    lines += ["", "A raw spectral outlier, a naturally used output direction, a compact restricted operator, a localized native-neuron implementation, recurrent channels, and a pure output role are deliberately scored separately. Correlation can prioritize a search; it cannot collapse these definitions into one notion of a clean circuit."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
