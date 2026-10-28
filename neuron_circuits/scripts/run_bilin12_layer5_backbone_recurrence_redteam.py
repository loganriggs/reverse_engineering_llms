#!/usr/bin/env python3
"""Red-team apparent recurrence of Layer-5 channel attribution profiles."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin12_layer5_address_channel_decomposition import ARRAYS as CHANNEL_ARRAYS
from run_bilin12_layer5_weight_conditioned_addresses import OUTPUT as ADDRESS_OUTPUT
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer5_backbone_recurrence_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer5_backbone_recurrence_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER5_BACKBONE_RECURRENCE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_layer5_backbone_recurrence_redteam.png"


def profile_statistics(profiles: torch.Tensor, topk: int) -> dict:
    profiles = profiles.abs()
    profiles = profiles / profiles.sum(1, keepdim=True).clamp_min(1e-30)
    singular = torch.linalg.svdvals(profiles)
    cosines = []
    sets = []
    for row in profiles:
        sets.append(set(torch.topk(row, topk).indices.cpu().tolist()))
    for first, second in itertools.combinations(range(len(profiles)), 2):
        cosines.append(float(torch.cosine_similarity(profiles[first], profiles[second], dim=0)))
    jaccards = [
        len(first & second) / len(first | second)
        for first, second in itertools.combinations(sets, 2)
    ]
    common = sorted(set.intersection(*sets)) if sets else []
    return {
        "rank1_energy_fraction": float(
            singular[0].square() / singular.square().sum().clamp_min(1e-30)
        ),
        "mean_pairwise_cosine": float(np.mean(cosines)),
        "mean_topk_jaccard": float(np.mean(jaccards)),
        "common_topk_channels": common,
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    metadata = json.loads(ADDRESS_OUTPUT.read_text())
    arrays = np.load(CHANNEL_ARRAYS)
    groups: dict[str, list[str]] = {}
    for name, row in metadata["results"].items():
        groups.setdefault(row["group"], []).append(name)

    def stack(names: list[str]) -> torch.Tensor:
        return torch.stack([
            torch.from_numpy(arrays[f"{name}_shapley"]).float().abs()
            for name in names
        ])

    leading_profiles = stack(groups["leading_clean"])
    observed = profile_statistics(leading_profiles, config["topk"])
    generator = torch.Generator().manual_seed(config["seed"])
    permutation_rows = []
    for _ in range(config["permutation_nulls"]):
        shuffled = torch.stack([
            row[torch.randperm(len(row), generator=generator)]
            for row in leading_profiles
        ])
        permutation_rows.append(profile_statistics(shuffled, config["topk"]))

    subset_controls = {}
    for group in ("sign_null", "random"):
        profiles = stack(groups[group])
        rows = []
        for subset in itertools.combinations(range(len(profiles)), len(leading_profiles)):
            rows.append(profile_statistics(profiles[list(subset)], config["topk"]))
        subset_controls[group] = rows

    def percentile(rows: list[dict], key: str, q: float = 0.95) -> float:
        return float(np.quantile([row[key] for row in rows], q))

    hypotheses = {
        "R1_rank1_profile_exceeds_coordinate_null": {
            "supported": observed["rank1_energy_fraction"]
            > percentile(permutation_rows, "rank1_energy_fraction"),
        },
        "R2_profile_cosine_exceeds_coordinate_null": {
            "supported": observed["mean_pairwise_cosine"]
            > percentile(permutation_rows, "mean_pairwise_cosine"),
        },
        "R3_top8_overlap_exceeds_coordinate_null": {
            "supported": observed["mean_topk_jaccard"]
            > percentile(permutation_rows, "mean_topk_jaccard"),
        },
        "R4_profile_recurrence_exceeds_sign_controls": {
            "supported": observed["mean_pairwise_cosine"]
            > percentile(subset_controls["sign_null"], "mean_pairwise_cosine"),
        },
        "R5_profile_recurrence_exceeds_random_controls": {
            "supported": observed["mean_pairwise_cosine"]
            > percentile(subset_controls["random"], "mean_pairwise_cosine"),
        },
        "R6_backbone_has_recurrent_channel_core": {
            "supported": len(observed["common_topk_channels"]) >= 2,
            "common_channels": observed["common_topk_channels"],
        },
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    null_summary = {
        "coordinate_permutation": {
            key: {
                "median": float(np.median([row[key] for row in permutation_rows])),
                "p95": percentile(permutation_rows, key),
            }
            for key in ("rank1_energy_fraction", "mean_pairwise_cosine", "mean_topk_jaccard")
        },
        **{
            group: {
                key: {
                    "median": float(np.median([row[key] for row in rows])),
                    "p95": percentile(rows, key),
                }
                for key in ("rank1_energy_fraction", "mean_pairwise_cosine", "mean_topk_jaccard")
            }
            for group, rows in subset_controls.items()
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "observed": observed,
        "null_summary": null_summary,
    }, indent=2, allow_nan=False))

    lines = [
        "# Recurrence red team for the Layer-5 shared channel backbone",
        "",
        "The four leading-address absolute-Shapley profiles are compared with 256 nulls that independently permute channel identities within each profile, preserving sparsity and marginal magnitudes. All four-profile subsets of the eight sign-null and eight random-address profiles provide matched empirical controls.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Backbone recurrence red team](bilin12_layer5_backbone_recurrence_redteam.png)",
        "",
        f"Observed rank-one energy **{observed['rank1_energy_fraction']:.3f}**, pairwise cosine **{observed['mean_pairwise_cosine']:.3f}**, and top-{config['topk']} Jaccard **{observed['mean_topk_jaccard']:.3f}**. Common top-{config['topk']} channels: `{observed['common_topk_channels']}`.",
        "",
        "| Control | Rank-1 median / p95 | Cosine median / p95 | Jaccard median / p95 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in null_summary.items():
        lines.append(
            f"| {name} | {row['rank1_energy_fraction']['median']:.3f} / {row['rank1_energy_fraction']['p95']:.3f} | "
            f"{row['mean_pairwise_cosine']['median']:.3f} / {row['mean_pairwise_cosine']['p95']:.3f} | "
            f"{row['mean_topk_jaccard']['median']:.3f} / {row['mean_topk_jaccard']['p95']:.3f} |"
        )
    lines += [
        "",
        "A high first singular value is not enough because nonnegative sparse profiles share a mean direction. Coordinate permutation tests whether the same channel identities recur; sign/random groups test whether that recurrence is special to the leading weight-conditioned addresses.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    keys = ("rank1_energy_fraction", "mean_pairwise_cosine", "mean_topk_jaccard")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for axis, key, title in zip(axes, keys, ("Rank-1 profile energy", "Pairwise profile cosine", f"Top-{config['topk']} Jaccard")):
        values = [row[key] for row in permutation_rows]
        axis.hist(values, bins=24, alpha=0.7, label="coordinate null")
        axis.axvline(observed[key], color="black", linewidth=2, label="observed")
        axis.axvline(percentile(subset_controls["sign_null"], key), color="tab:orange", linestyle="--", label="sign p95")
        axis.axvline(percentile(subset_controls["random"], key), color="tab:green", linestyle=":", label="random p95")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
