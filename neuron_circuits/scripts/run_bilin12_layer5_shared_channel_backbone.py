#!/usr/bin/env python3
"""Test whether Layer-5 spectral addresses share a canonical native backbone."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer5_address_channel_decomposition import ARRAYS as CHANNEL_ARRAYS
from run_bilin12_layer5_weight_conditioned_addresses import (
    ARRAYS as ADDRESS_ARRAYS,
    OUTPUT as ADDRESS_OUTPUT,
)
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer5_shared_channel_backbone.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer5_shared_channel_backbone.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER5_SHARED_CHANNEL_BACKBONE.md"
FIGURE = TENSOR_ROOT / "bilin12_layer5_shared_channel_backbone.png"


def aggregate_backbone_order(profiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = profiles.abs() / profiles.abs().sum(1, keepdim=True).clamp_min(1e-30)
    return normalized.mean(0).argsort(descending=True), normalized


def operator_r2(target: torch.Tensor, prediction: torch.Tensor) -> float:
    return float(
        1 - (target - prediction).square().sum()
        / target.square().sum().clamp_min(1e-30)
    )


def singular_k90(operator: torch.Tensor) -> int:
    singular = torch.linalg.svdvals(operator)
    cumulative = singular.square().cumsum(0) / singular.square().sum().clamp_min(1e-30)
    return int(torch.nonzero(cumulative >= 0.90, as_tuple=False)[0]) + 1


def pairwise_jaccard(sets: list[set[int]]) -> list[float]:
    return [
        len(first & second) / len(first | second)
        for first, second in itertools.combinations(sets, 2)
    ]


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    metadata = json.loads(ADDRESS_OUTPUT.read_text())
    address_arrays = np.load(ADDRESS_ARRAYS)
    channel_arrays = np.load(CHANNEL_ARRAYS)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    square = torch.from_numpy(address_arrays["covariance_square"]).cuda().float()
    leading_output = torch.from_numpy(address_arrays["leading_output"]).cuda().float()

    leading_names = [
        name for name, row in metadata["results"].items()
        if row["group"] == "leading_clean"
    ]
    leading_profiles = torch.stack([
        torch.from_numpy(channel_arrays[f"{name}_shapley"]).cuda().float().abs()
        for name in leading_names
    ])
    backbone_order, normalized_profiles = aggregate_backbone_order(leading_profiles)
    profile_singular = torch.linalg.svdvals(normalized_profiles)
    profile_rank1_fraction = float(
        profile_singular[0].square() / profile_singular.square().sum().clamp_min(1e-30)
    )
    top8_sets = [
        set(torch.topk(profile, 8).indices.cpu().tolist())
        for profile in leading_profiles
    ]
    top8_jaccards = pairwise_jaccard(top8_sets)

    results = {}
    for name, address_row in metadata["results"].items():
        edit = torch.from_numpy(address_arrays[f"{name}_edit"]).cuda().float()
        context_rows = (
            (left @ edit)[:, None] * right + (right @ edit)[:, None] * left
        ) @ square
        operator = down @ context_rows
        prefix_rows = []
        for size in config["backbone_sizes"]:
            selected = backbone_order[:size]
            approximation = down[:, selected] @ context_rows[selected]
            full_role = (leading_output @ operator).square().sum()
            approximate_role = (leading_output @ approximation).square().sum()
            prefix_rows.append({
                "channels": size,
                "operator_r2": operator_r2(operator, approximation),
                "leading_output_energy_retention": float(
                    approximate_role / full_role.clamp_min(1e-30)
                ),
            })
        primary = backbone_order[: config["primary_backbone_size"]]
        residual = operator - down[:, primary] @ context_rows[primary]
        results[name] = {
            "name": name,
            "group": address_row["group"],
            "prefix_rows": prefix_rows,
            "residual_singular_k90": singular_k90(residual),
            "full_singular_k90": address_row["k90"],
        }

    groups: dict[str, list[dict]] = {}
    for row in results.values():
        groups.setdefault(row["group"], []).append(row)

    def prefix(row: dict, size: int, field: str = "operator_r2") -> float:
        return next(item[field] for item in row["prefix_rows"] if item["channels"] == size)

    leading = groups["leading_clean"]
    controls = groups["sign_null"] + groups["random"]
    hypotheses = {
        "B1_leading_top8_channels_recur": {
            "supported": bool(np.mean(top8_jaccards) >= 0.50),
            "mean_pairwise_jaccard": float(np.mean(top8_jaccards)),
        },
        "B2_shared_top4_captures_half_operator": {
            "supported": bool(all(prefix(row, 4) >= 0.50 for row in leading)),
        },
        "B3_shared_top16_is_sufficient": {
            "supported": bool(all(prefix(row, 16) >= 0.80 for row in leading)),
        },
        "B4_backbone_is_leading_address_specific": {
            "supported": bool(
                np.median([prefix(row, 16) for row in leading])
                >= 2 * np.median([prefix(row, 16) for row in controls])
            ),
        },
        "B5_shapley_profiles_have_shared_rank1_backbone": {
            "supported": bool(profile_rank1_fraction >= 0.80),
            "rank1_energy_fraction": profile_rank1_fraction,
        },
        "B6_residual_after_backbone_remains_broad": {
            "supported": bool(np.median([row["residual_singular_k90"] for row in leading]) >= 32),
        },
        "B7_backbone_channels_are_not_the_whole_circuit": {
            "supported": bool(sum(prefix(row, 128) < 0.90 for row in leading) >= 3),
        },
        "B8_shared_backbone_preserves_output_role": {
            "supported": bool(all(
                prefix(row, 16, "leading_output_energy_retention") >= 0.50
                for row in leading
            )),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "backbone_channel_order": backbone_order[:128].cpu().tolist(),
        "leading_top8_sets": [sorted(values) for values in top8_sets],
        "leading_top8_pairwise_jaccard": top8_jaccards,
        "profile_singular_values": profile_singular.cpu().tolist(),
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Shared native-channel backbone of Layer-5 spectral addresses",
        "",
        "A shared channel ranking is defined before testing sufficiency by averaging each leading address's normalized absolute Shapley profile. Frozen prefixes are then used to reconstruct every leading address and the sign-null/random controls. This separates recurrence of high-attribution channels from joint causal sufficiency.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Shared backbone](bilin12_layer5_shared_channel_backbone.png)",
        "",
        f"Leading top-eight mean pairwise Jaccard: **{np.mean(top8_jaccards):.3f}**. Normalized-profile first singular energy: **{profile_rank1_fraction:.3f}**. Aggregate backbone begins with channels `{backbone_order[:16].cpu().tolist()}`.",
        "",
        "| Address | R² shared 4 / 16 / 128 | Leading-output retention @16 | Full / residual singular K90 |",
        "|---|---:|---:|---:|",
    ]
    for row in leading:
        lines.append(
            f"| {row['name']} | {prefix(row, 4):.3f} / {prefix(row, 16):.3f} / {prefix(row, 128):.3f} | "
            f"{prefix(row, 16, 'leading_output_energy_retention'):.3f} | "
            f"{row['full_singular_k90']} / {row['residual_singular_k90']} |"
        )
    lines += [
        "",
        "The aggregate ranking is canonical only relative to the declared four addresses and Frobenius/Shapley metric. High top-k overlap establishes a reusable backbone; it does not establish that the backbone is sufficient. A broad residual after removing it is evidence for a backbone-plus-distributed-correction organization.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for row in leading:
        axes[0].plot(
            config["backbone_sizes"],
            [item["operator_r2"] for item in row["prefix_rows"]],
            marker="o", label=row["name"],
        )
    axes[0].axhline(0.9, color="black", linestyle="--", linewidth=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(config["backbone_sizes"], config["backbone_sizes"], rotation=30)
    axes[0].set_title("Shared-backbone sufficiency")
    axes[0].set_ylabel("operator R²")
    axes[0].legend(fontsize=7)
    group_order = ("leading_clean", "sign_null", "random")
    axes[1].boxplot(
        [[prefix(row, 16) for row in groups[group]] for group in group_order],
        tick_labels=group_order,
    )
    axes[1].set_title("Backbone specificity")
    axes[1].set_ylabel("shared-top16 operator R²")
    axes[1].tick_params(axis="x", rotation=25)
    axes[2].bar(np.arange(len(profile_singular)), profile_singular.square().cpu().numpy() / float(profile_singular.square().sum()))
    axes[2].set_xticks(range(len(profile_singular)), range(1, len(profile_singular) + 1))
    axes[2].set_title("Leading-profile singular energy")
    axes[2].set_xlabel("component")
    axes[2].set_ylabel("energy fraction")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
