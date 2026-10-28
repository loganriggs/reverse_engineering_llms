#!/usr/bin/env python3
"""High-resolution null audit of covariance-cleaned Layer-17 tail forms."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_cleaning_null_redteam.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_cleaning_null_redteam.json"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_CLEANING_NULL_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer17_tail_cleaning_null_redteam.png"


def spectral_summary(matrix: torch.Tensor, rank: int) -> dict[str, float]:
    clean, _ = trace_clean(matrix)
    values = torch.linalg.eigvalsh(clean).abs()
    energy = values.square()
    return {
        "maximum_eigenvalue": float(values.max()),
        "top4_energy": float(torch.topk(energy, rank).values.sum() / energy.sum().clamp_min(1e-30)),
    }


def empirical_p(observed: float, null: list[float]) -> float:
    return float((1 + sum(value >= observed for value in null)) / (len(null) + 1))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE_ARRAYS)
    axes = torch.from_numpy(source["tail_output_basis"][:, : config["components"]]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    x, _ = collect(model, config["layer"], config["discovery"], config)
    square, _, covariance = covariance_transforms(x, config["covariance_floor_fraction"])

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    signs = torch.where(
        torch.rand(config["null_replicates"], down.shape[1], generator=generator, device="cuda") < 0.5,
        -torch.ones((), device="cuda"), torch.ones((), device="cuda"),
    )
    permutations = [
        torch.randperm(down.shape[1], generator=generator, device="cuda")
        for _ in range(config["null_replicates"])
    ]
    rows = []
    for component in range(config["components"]):
        print("axis", component + 1, "/", config["components"], flush=True)
        coefficients = down.T @ axes[:, component]

        def form(current: torch.Tensor) -> torch.Tensor:
            raw = left.T @ (current[:, None] * right)
            return square @ (0.5 * (raw + raw.T)) @ square

        observed = spectral_summary(form(coefficients), config["rank"])
        nulls = {"sign_null": [], "pairing_null": []}
        for replicate in range(config["null_replicates"]):
            nulls["sign_null"].append(spectral_summary(
                form(coefficients * signs[replicate]), config["rank"]
            ))
            nulls["pairing_null"].append(spectral_summary(
                form(coefficients[permutations[replicate]]), config["rank"]
            ))
        row = {"component": component, "observed": observed, "nulls": nulls}
        for family in nulls:
            row[family] = {}
            for metric in ("maximum_eigenvalue", "top4_energy"):
                values = [item[metric] for item in nulls[family]]
                row[family][metric] = {
                    "median": float(np.median(values)),
                    "quantile95": float(np.quantile(values, 0.95)),
                    "empirical_p": empirical_p(observed[metric], values),
                }
        rows.append(row)

    vocab_rotations = []
    observed_rotation_statistic = float(np.mean([row["observed"]["top4_energy"] for row in rows]))
    for replicate in range(config["rotation_replicates"]):
        random = torch.randn(
            config["components"], config["components"], generator=generator, device="cuda"
        )
        rotation = torch.linalg.qr(random).Q
        rotated = axes @ rotation
        energies = []
        for component in range(config["components"]):
            coefficients = down.T @ rotated[:, component]
            raw = left.T @ (coefficients[:, None] * right)
            weighted = square @ (0.5 * (raw + raw.T)) @ square
            energies.append(spectral_summary(weighted, config["rank"])["top4_energy"])
        vocab_rotations.append(float(np.mean(energies)))
    rotation_p = empirical_p(observed_rotation_statistic, vocab_rotations)

    hypotheses = {
        "N1_every_axis_beats_sign_null": {
            "supported": all(row["sign_null"]["maximum_eigenvalue"]["empirical_p"] <= 0.05 for row in rows),
        },
        "N2_every_axis_beats_pairing_null": {
            "supported": all(row["pairing_null"]["maximum_eigenvalue"]["empirical_p"] <= 0.05 for row in rows),
        },
        "N3_concentration_exceeds_sign_null": {
            "supported": all(
                row["observed"]["top4_energy"] > row["sign_null"]["top4_energy"]["quantile95"]
                for row in rows
            ),
        },
        "N4_pca_coordinates_are_spectrally_privileged": {
            "supported": rotation_p <= 0.05,
            "observed": observed_rotation_statistic,
            "null_median": float(np.median(vocab_rotations)),
            "empirical_p": rotation_p,
        },
        "N5_at_least_one_robust_cleaned_spike": {
            "supported": any(
                row["sign_null"]["maximum_eigenvalue"]["empirical_p"] <= 0.05
                and row["pairing_null"]["maximum_eigenvalue"]["empirical_p"] <= 0.05
                for row in rows
            ),
            "components": [
                row["component"] + 1 for row in rows
                if row["sign_null"]["maximum_eigenvalue"]["empirical_p"] <= 0.05
                and row["pairing_null"]["maximum_eigenvalue"]["empirical_p"] <= 0.05
            ],
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "covariance": covariance,
        "axes": rows,
        "rotation_null": {
            "observed_mean_top4_energy": observed_rotation_statistic,
            "replicates": vocab_rotations,
            "median": float(np.median(vocab_rotations)),
            "empirical_p": rotation_p,
        },
    }, indent=2, allow_nan=False))

    lines = [
        "# High-resolution null red-team for cleaned Layer-17 tail forms",
        "",
        "The previous 32-draw sign threshold was too coarse for a borderline result. This rerun uses 128 sign nulls and 128 coefficient-to-input-form pairing nulls per frozen tail axis. A separate rotation null preserves the complete tail-4 output subspace and changes only its coordinate basis.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Tail cleaning nulls](layer17_tail_cleaning_null_redteam.png)",
        "",
        "| Axis | Observed max | Sign median / p | Pairing median / p | Observed top-4 | Sign top-4 95% / p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['component'] + 1} | {row['observed']['maximum_eigenvalue']:.1f} | "
            f"{row['sign_null']['maximum_eigenvalue']['median']:.1f} / {row['sign_null']['maximum_eigenvalue']['empirical_p']:.4f} | "
            f"{row['pairing_null']['maximum_eigenvalue']['median']:.1f} / {row['pairing_null']['maximum_eigenvalue']['empirical_p']:.4f} | "
            f"{row['observed']['top4_energy']:.3f} | {row['sign_null']['top4_energy']['quantile95']:.3f} / "
            f"{row['sign_null']['top4_energy']['empirical_p']:.4f} |"
        )
    lines += [
        "",
        f"PCA-coordinate mean top-four energy **{observed_rotation_statistic:.3f}**; rotation-null median **{np.median(vocab_rotations):.3f}**; empirical `p={rotation_p:.4f}`.",
        "",
        "Sign and pairing nulls answer whether learned channel coordination makes a fixed output form exceptional. The rotation null answers whether PCA chose special axes inside the already selected tail subspace. None corrects for all possible researcher choices, so a null-exceeding result remains a circuit candidate until tied to held-out finite behavior.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, plots = plt.subplots(1, 3, figsize=(14, 4.2))
    indices = np.arange(config["components"])
    observed_max = [row["observed"]["maximum_eigenvalue"] for row in rows]
    sign95 = [row["sign_null"]["maximum_eigenvalue"]["quantile95"] for row in rows]
    pair95 = [row["pairing_null"]["maximum_eigenvalue"]["quantile95"] for row in rows]
    width = 0.25
    plots[0].bar(indices - width, observed_max, width, label="observed")
    plots[0].bar(indices, sign95, width, label="sign 95%")
    plots[0].bar(indices + width, pair95, width, label="pairing 95%")
    plots[0].set_xticks(indices, [f"T{i + 1}" for i in indices])
    plots[0].set_ylabel("maximum |eigenvalue|")
    plots[0].set_title("Null-exceeding strength")
    plots[0].legend(fontsize=8)
    plots[1].bar(indices - width / 2, [row["observed"]["top4_energy"] for row in rows], width, label="observed")
    plots[1].bar(indices + width / 2, [row["sign_null"]["top4_energy"]["quantile95"] for row in rows], width, label="sign 95%")
    plots[1].set_xticks(indices, [f"T{i + 1}" for i in indices])
    plots[1].set_ylabel("top-four energy")
    plots[1].set_title("Concentration null")
    plots[1].legend(fontsize=8)
    plots[2].hist(vocab_rotations, bins=20, alpha=0.75)
    plots[2].axvline(observed_rotation_statistic, color="tab:red", linewidth=2)
    plots[2].set_xlabel("mean top-four energy")
    plots[2].set_title("Tail-axis rotation null")
    for plot in plots:
        plot.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
