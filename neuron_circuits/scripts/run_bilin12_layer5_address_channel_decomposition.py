#!/usr/bin/env python3
"""Decompose Layer-5 restricted address operators into native MLP channels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer5_weight_conditioned_addresses import (
    ARRAYS as ADDRESS_ARRAYS,
    OUTPUT as ADDRESS_OUTPUT,
)
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer5_address_channel_decomposition.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer5_address_channel_decomposition.json"
ARRAYS = TENSOR_ROOT / "bilin12_layer5_address_channel_decomposition.npz"
REPORT = TENSOR_ROOT / "BILIN12_LAYER5_ADDRESS_CHANNEL_DECOMPOSITION.md"
FIGURE = TENSOR_ROOT / "bilin12_layer5_address_channel_decomposition.png"


def effective_support(values: torch.Tensor) -> float:
    values = values.abs()
    return float(values.sum().square() / values.square().sum().clamp_min(1e-30))


def first_fraction_rank(values: torch.Tensor, threshold: float) -> int:
    values = values.abs().sort(descending=True).values
    cumulative = values.cumsum(0) / values.sum().clamp_min(1e-30)
    return int(torch.nonzero(cumulative >= threshold, as_tuple=False)[0]) + 1


def channel_decomposition(
    down: torch.Tensor,
    context_rows: torch.Tensor,
    prefixes: list[int],
) -> dict:
    operator = down @ context_rows
    reconstructed = torch.einsum("oh,hj->oj", down, context_rows)
    total_energy = operator.square().sum()
    individual_energy = down.square().sum(0) * context_rows.square().sum(1)
    shapley = ((down.T @ operator) * context_rows).sum(1)
    order = shapley.abs().argsort(descending=True)
    prefix_rows = []
    for prefix in prefixes:
        selected = order[:prefix]
        approximation = down[:, selected] @ context_rows[selected]
        prefix_rows.append({
            "channels": prefix,
            "operator_r2": float(
                1 - (operator - approximation).square().sum()
                / total_energy.clamp_min(1e-30)
            ),
            "absolute_shapley_fraction": float(
                shapley[selected].abs().sum() / shapley.abs().sum().clamp_min(1e-30)
            ),
        })
    absolute = shapley.abs()
    negative_mass = -shapley[shapley < 0].sum() / absolute.sum().clamp_min(1e-30)
    return {
        "operator_identity_relative_error": float(
            (operator - reconstructed).norm() / operator.norm().clamp_min(1e-30)
        ),
        "shapley_sum_relative_error": float(
            (shapley.sum() - total_energy).abs() / total_energy.clamp_min(1e-30)
        ),
        "shapley_k90": first_fraction_rank(shapley, 0.90),
        "absolute_shapley_effective_support": effective_support(shapley),
        "individual_energy_effective_support": effective_support(individual_energy),
        "largest_absolute_shapley_share": float(absolute.max() / absolute.sum().clamp_min(1e-30)),
        "negative_shapley_mass_fraction": float(negative_mass),
        "individual_to_total_energy_ratio": float(
            individual_energy.sum() / total_energy.clamp_min(1e-30)
        ),
        "top_channel_indices": order[:16].cpu().tolist(),
        "prefix_rows": prefix_rows,
        "shapley": shapley,
        "individual_energy": individual_energy,
        "order": order,
        "operator": operator,
    }


def token_extremes(scores: torch.Tensor, tokenizer, count: int = 5) -> dict[str, list[str]]:
    return {
        "positive": [tokenizer.decode([index]) for index in torch.topk(scores, count).indices.tolist()],
        "negative": [tokenizer.decode([index]) for index in torch.topk(-scores, count).indices.tolist()],
    }


def prefix_r2(row: dict, prefix: int) -> float:
    return next(item["operator_r2"] for item in row["prefix_rows"] if item["channels"] == prefix)


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    address_metadata = json.loads(ADDRESS_OUTPUT.read_text())
    address_arrays = np.load(ADDRESS_ARRAYS)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    square = torch.from_numpy(address_arrays["covariance_square"]).cuda().float()
    embedding = model.embedding.weight.float()

    results = {}
    saved = {}
    for name, metadata in address_metadata["results"].items():
        print("channel decomposition", name, flush=True)
        edit = torch.from_numpy(address_arrays[f"{name}_edit"]).cuda().float()
        left_edit = left @ edit
        right_edit = right @ edit
        context_rows = (
            left_edit[:, None] * right + right_edit[:, None] * left
        ) @ square
        decomposition = channel_decomposition(
            down, context_rows, config["channel_prefixes"]
        )
        top_channel = decomposition["top_channel_indices"][0]
        channel_context = context_rows[top_channel]
        channel_output = down[:, top_channel]
        row = {
            key: value for key, value in decomposition.items()
            if key not in {"shapley", "individual_energy", "order", "operator"}
        }
        row.update({
            "name": name,
            "group": metadata["group"],
            "restricted_singular_k90": metadata["k90"],
            "top_channel_context_tokens": token_extremes(embedding @ channel_context, tokenizer),
            "top_channel_write_tokens": token_extremes(
                model.lm_head(channel_output[None]).squeeze(0), tokenizer
            ),
        })
        results[name] = row
        saved[f"{name}_shapley"] = decomposition["shapley"].cpu().numpy().astype(np.float32)
        saved[f"{name}_individual_energy"] = decomposition["individual_energy"].cpu().numpy().astype(np.float32)
        saved[f"{name}_order"] = decomposition["order"].cpu().numpy().astype(np.int32)

    groups: dict[str, list[dict]] = {}
    for row in results.values():
        groups.setdefault(row["group"], []).append(row)
    leading = groups["leading_clean"]
    raw = groups["leading_raw"]
    sign = groups["sign_null"]
    nonleading = groups["nonleading"]
    hypotheses = {
        "N1_channel_sum_and_shapley_identity_are_exact": {
            "supported": bool(
                max(row["operator_identity_relative_error"] for row in results.values()) <= 1e-5
                and max(row["shapley_sum_relative_error"] for row in results.values()) <= 1e-5
            ),
        },
        "N2_leading_addresses_use_more_channels_than_sign_nulls": {
            "supported": bool(
                np.median([row["shapley_k90"] for row in leading])
                >= 4 * np.median([row["shapley_k90"] for row in sign])
            ),
        },
        "N3_leading_addresses_are_not_single_channel": {
            "supported": bool(
                np.median([row["largest_absolute_shapley_share"] for row in leading]) < 0.25
            ),
        },
        "N4_sign_null_addresses_are_channel_localized": {
            "supported": bool(np.median([row["shapley_k90"] for row in sign]) <= 8),
        },
        "N5_leading_addresses_show_positive_cooperation": {
            "supported": bool(
                np.median([row["individual_to_total_energy_ratio"] for row in leading]) < 1
            ),
        },
        "N6_128_channels_are_sufficient_for_leading_addresses": {
            "supported": bool(all(prefix_r2(row, 128) >= 0.90 for row in leading)),
        },
        "N7_cleaning_increases_channel_distribution": {
            "supported": bool(
                np.median([row["shapley_k90"] for row in leading])
                > np.median([row["shapley_k90"] for row in raw])
            ),
        },
        "N8_nonleading_compactness_is_channel_localized": {
            "supported": bool(np.median([row["shapley_k90"] for row in nonleading]) <= 8),
        },
    }
    group_summary = {
        group: {
            "count": len(rows),
            "median_shapley_k90": float(np.median([row["shapley_k90"] for row in rows])),
            "median_shapley_effective_support": float(np.median([row["absolute_shapley_effective_support"] for row in rows])),
            "median_largest_share": float(np.median([row["largest_absolute_shapley_share"] for row in rows])),
            "median_individual_to_total_energy": float(np.median([row["individual_to_total_energy_ratio"] for row in rows])),
            "median_negative_mass": float(np.median([row["negative_shapley_mass_fraction"] for row in rows])),
        }
        for group, rows in groups.items()
    }
    np.savez_compressed(ARRAYS, **saved)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "group_summary": group_summary,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Native-channel decomposition of Layer-5 spectral addresses",
        "",
        "Each exact address-restricted derivative operator is written as the sum of 3,072 native rank-one channel operators. For squared Frobenius value, channel h has exact Shapley allocation `<C_h, sum_k C_k>`. Static prefixes are ranked by absolute Shapley allocation and tested by reconstructing the full covariance-weighted operator.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Channel decomposition](bilin12_layer5_address_channel_decomposition.png)",
        "",
        "| Address | Singular K90 | Channel Shapley K90 | Eff. channel support | Largest share | Individual/total energy | R² top-128 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in leading + raw + nonleading:
        lines.append(
            f"| {row['name']} | {row['restricted_singular_k90']} | {row['shapley_k90']} | "
            f"{row['absolute_shapley_effective_support']:.1f} | {row['largest_absolute_shapley_share']:.3f} | "
            f"{row['individual_to_total_energy_ratio']:.3f} | {prefix_r2(row, 128):.3f} |"
        )
    lines += [
        "",
        "| Group | Median channel K90 | Effective support | Largest share | Individual/total energy | Negative Shapley mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, row in group_summary.items():
        lines.append(
            f"| {group} | {row['median_shapley_k90']:.1f} | {row['median_shapley_effective_support']:.1f} | "
            f"{row['median_largest_share']:.3f} | {row['median_individual_to_total_energy']:.3f} | {row['median_negative_mass']:.3f} |"
        )
    lines += [
        "",
        "Low singular rank and low channel support are different notions. A rank-two operator can be the coherent sum of many channels, while a high-rank operator can have a few dominant allocations plus a long corrective tail. `individual/total < 1` indicates net positive cooperation; values above one indicate cancellation. Static prefix sufficiency is reported separately from attribution support.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    group_order = ("leading_clean", "leading_raw", "nonleading", "sign_null", "random")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].boxplot(
        [[row["shapley_k90"] for row in groups[group]] for group in group_order],
        tick_labels=group_order,
    )
    axes[0].set_ylabel("absolute-Shapley K90")
    axes[0].set_title("Native channel support")
    axes[1].boxplot(
        [[row["individual_to_total_energy_ratio"] for row in groups[group]] for group in group_order],
        tick_labels=group_order,
    )
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("sum individual / total energy")
    axes[1].set_title("Cooperation versus cancellation")
    for row in leading:
        axes[2].plot(
            config["channel_prefixes"],
            [item["operator_r2"] for item in row["prefix_rows"]],
            marker="o", label=row["name"],
        )
    axes[2].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(config["channel_prefixes"], config["channel_prefixes"], rotation=35)
    axes[2].set_ylabel("operator R²")
    axes[2].set_title("Static channel sufficiency")
    axes[2].legend(fontsize=7)
    for axis in axes[:2]:
        axis.tick_params(axis="x", rotation=30)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
