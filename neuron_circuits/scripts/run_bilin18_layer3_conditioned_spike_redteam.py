#!/usr/bin/env python3
"""Stronger nulls and fair random-subspace controls for conditioned spikes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_spectral_redteam import symmetric_channel_gram
from run_bilin18_layer3_conditioned_spectral_spikes import interaction_spectrum
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_conditioned_spike_redteam.json")
OUTPUT = TENSOR_ROOT / "layer3_conditioned_spike_redteam.json"
ARRAYS = TENSOR_ROOT / "layer3_conditioned_spike_redteam.npz"
REPORT = TENSOR_ROOT / "LAYER3_CONDITIONED_SPIKE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer3_conditioned_spike_redteam.png"


def top4_fraction(values: torch.Tensor) -> float:
    energy = values.square()
    return float(energy[:4].sum() / energy.sum().clamp_min(1e-30))


def empirical_p(observed: float, nulls: list[float]) -> float:
    return (1 + sum(value >= observed for value in nulls)) / (len(nulls) + 1)


def canonicalize(
    basis: torch.Tensor, down: torch.Tensor, channel_gram: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = down.T @ basis
    gram = coefficients.T @ (channel_gram @ coefficients)
    gram = 0.5 * (gram + gram.T)
    values, rotation = torch.linalg.eigh(gram)
    order = values.argsort(descending=True)
    return basis @ rotation[:, order], values[order]


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_results"]).read_text())
    saved = np.load(config["source_arrays"])
    families = list(source["results"])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    block = model.blocks[config["layer"]]
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    channel_gram = symmetric_channel_gram(left, right)

    random_subspace_rows = []
    saved_arrays = {}
    for replicate in range(config["random_subspaces"]):
        raw = torch.randn(
            down.shape[0], config["output_rank"], generator=generator, device="cuda",
        )
        random_subspace, _ = torch.linalg.qr(raw, mode="reduced")
        canonical, _ = canonicalize(random_subspace, down, channel_gram)
        fractions = []
        for axis in range(config["output_rank"]):
            values = interaction_spectrum(left, right, down.T @ canonical[:, axis])
            fractions.append(top4_fraction(values))
        random_subspace_rows.append({
            "median_top4": float(np.median(fractions)),
            "tail_top4": float(np.median(fractions[1:])),
            "axis_top4": fractions,
        })
        saved_arrays[f"random_subspace_{replicate}"] = canonical.cpu().numpy()
        print("random_subspace", replicate, flush=True)
    random_medians = [row["median_top4"] for row in random_subspace_rows]
    random_tails = [row["tail_top4"] for row in random_subspace_rows]
    random_median_95 = float(np.quantile(random_medians, .95))
    random_tail_95 = float(np.quantile(random_tails, .95))

    results = {}
    for family in families:
        canonical = torch.from_numpy(saved[f"{family}_canonical_basis"]).cuda()
        axes = []
        for axis in range(config["output_rank"]):
            coefficients = down.T @ canonical[:, axis]
            observed_values = interaction_spectrum(left, right, coefficients)
            observed_leading = float(observed_values[0])
            sign_null = []
            pairing_null = []
            for replicate in range(config["null_replicates"]):
                signs = torch.randint(
                    0, 2, (len(coefficients),), generator=generator, device="cuda",
                ) * 2 - 1
                sign_null.append(float(interaction_spectrum(
                    left, right, coefficients * signs,
                )[0]))
                permutation = torch.randperm(
                    len(coefficients), generator=generator, device="cuda",
                )
                pairing_null.append(float(interaction_spectrum(
                    left, right, coefficients[permutation],
                )[0]))
            previous = source["results"][family]["axes"][axis]
            axes.append({
                "axis": axis,
                "leading": observed_leading,
                "top4_energy": top4_fraction(observed_values),
                "sign_null_max": max(sign_null),
                "pairing_null_max": max(pairing_null),
                "sign_p": empirical_p(observed_leading, sign_null),
                "pairing_p": empirical_p(observed_leading, pairing_null),
                "strict_powerlaw_spike": previous["observed"]["top_log_excess"][0] > 0,
            })
            print("family", family, "axis", axis, flush=True)
        median_top4 = float(np.median([row["top4_energy"] for row in axes]))
        tail_top4 = float(np.median([row["top4_energy"] for row in axes[1:]]))
        results[family] = {
            "axes": axes,
            "median_top4": median_top4,
            "tail_top4": tail_top4,
            "random_subspace_p": empirical_p(median_top4, random_medians),
            "random_tail_p": empirical_p(tail_top4, random_tails),
            "robust_sign_axes": sum(row["sign_p"] <= .05 for row in axes),
            "robust_pairing_axes": sum(row["pairing_p"] <= .05 for row in axes),
            "robust_both_axes": sum(
                row["sign_p"] <= .05 and row["pairing_p"] <= .05 for row in axes
            ),
        }

    strict_count = sum(
        row["strict_powerlaw_spike"]
        for family in families for row in results[family]["axes"]
    )
    hypotheses = {
        "R1_all_initial_sign_outliers_survive": {
            "supported": all(
                row["leading"] > row["sign_null_max"]
                for family in families for row in results[family]["axes"]
            ),
        },
        "R2_families_survive_two_null_types": {
            "supported": sum(results[family]["robust_both_axes"] >= 3 for family in families) >= 3,
        },
        "R3_semantic_subspaces_beat_fair_random_subspaces": {
            "supported": sum(results[family]["median_top4"] > random_median_95 for family in families) >= 3,
            "random_subspace_median_top4_95": random_median_95,
        },
        "R4_enrichment_is_not_only_the_energy_leading_axis": {
            "supported": sum(results[family]["tail_top4"] > random_tail_95 for family in families) >= 3,
            "random_tail_95": random_tail_95,
        },
        "R5_the_strict_powerlaw_rule_remains_low_recall": {
            "supported": strict_count <= 4,
            "strict_axes": strict_count,
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "random_subspaces": random_subspace_rows,
        "results": results,
    }, indent=2))
    np.savez_compressed(ARRAYS, **saved_arrays)

    figure, axes_plot = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(families))
    axes_plot[0].bar(x, [results[f]["median_top4"] for f in families])
    axes_plot[0].axhline(random_median_95, color="tab:red", linestyle="--", label="fair random 95th")
    axes_plot[0].set_xticks(x, families, rotation=25, ha="right")
    axes_plot[0].set_ylabel("Median canonical-axis top-4 energy")
    axes_plot[0].set_title("Semantic vs equally canonicalized random subspaces")
    axes_plot[0].legend()
    width = .36
    axes_plot[1].bar(x - width / 2, [results[f]["robust_sign_axes"] for f in families], width, label="sign")
    axes_plot[1].bar(x + width / 2, [results[f]["robust_pairing_axes"] for f in families], width, label="pairing")
    axes_plot[1].set_xticks(x, families, rotation=25, ha="right")
    axes_plot[1].set_ylabel("Axes with empirical p ≤ 0.05 (of 4)")
    axes_plot[1].set_title("Exceptional directions under stronger nulls")
    axes_plot[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Red-team of output-conditioned Layer-3 spikes",
        "",
        "This follow-up was locked after the initial positive result. It raises each channel-coordination null from eight to 64 draws, adds a coefficient-to-form pairing null, and gives 64 random four-dimensional output subspaces the same tensor-Frobenius canonicalization used for semantic families.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Conditioned spike red-team](layer3_conditioned_spike_redteam.png)",
        "",
        "| Family | Median top-4 | Fair-random p | Tail top-4 | Tail-random p | Robust sign axes | Robust pairing axes | Robust under both |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in families:
        row = results[family]
        lines.append(
            f"| {family} | {row['median_top4']:.3f} | {row['random_subspace_p']:.3f} | "
            f"{row['tail_top4']:.3f} | {row['random_tail_p']:.3f} | "
            f"{row['robust_sign_axes']}/4 | {row['robust_pairing_axes']}/4 | "
            f"{row['robust_both_axes']}/4 |"
        )
    lines += [
        "",
        f"The fair-random 95th percentiles are **{random_median_95:.3f}** for all four axes and "
        f"**{random_tail_95:.3f}** after excluding the tensor-energy-leading axis.",
        "",
        "This distinguishes exceptional learned coordination from low rank caused merely by choosing and canonicalizing a four-dimensional output subspace. Empirical p-values are resolution-limited to 1/65 and apply to the specified null generators, not to every conceivable random-tensor ensemble.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
