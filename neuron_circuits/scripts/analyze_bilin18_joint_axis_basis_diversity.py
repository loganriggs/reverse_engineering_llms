#!/usr/bin/env python3
"""Test whether apparently cleaner rotations merely duplicate output roles."""

from __future__ import annotations

import json
from pathlib import Path

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_joint_axis_basis_diversity.json")
SOURCE = TENSOR_ROOT / "joint_axis_basis_controls.json"
OUTPUT = TENSOR_ROOT / "joint_axis_basis_diversity.json"
REPORT = TENSOR_ROOT / "JOINT_AXIS_BASIS_DIVERSITY.md"


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = json.loads(SOURCE.read_text())
    aggregate = source["aggregate"]
    scores = {}
    for layer, families in aggregate.items():
        scores[layer] = {}
        for family, row in families.items():
            scores[layer][family] = {
                "first_role_energy": row["first_component_energy"],
                "effective_role_diversity": row["first_role_effective_diversity"],
                "pairwise_role_cosine": row["mean_pairwise_first_role_cosine"],
                "diversity_adjusted_cleanliness": row["first_component_energy"]
                * row["first_role_effective_diversity"] / 32,
            }
    hypotheses = {
        "D1_rotation_duplicates_roles": {
            "supported": all(
                scores[layer]["rotated"]["effective_role_diversity"]
                <= 0.8 * scores[layer]["joint"]["effective_role_diversity"]
                for layer in scores
            ),
            "ratios": {
                layer: scores[layer]["rotated"]["effective_role_diversity"]
                / scores[layer]["joint"]["effective_role_diversity"]
                for layer in scores
            },
        },
        "D2_joint_roles_less_correlated": {
            "supported": all(
                scores[layer]["joint"]["pairwise_role_cosine"]
                < scores[layer]["rotated"]["pairwise_role_cosine"]
                for layer in scores
            ),
            "values": {
                layer: {
                    family: scores[layer][family]["pairwise_role_cosine"]
                    for family in ("joint", "rotated")
                }
                for layer in scores
            },
        },
        "D3_diversity_adjusted_cleanliness": {
            "supported": all(
                scores[layer]["joint"]["diversity_adjusted_cleanliness"]
                > scores[layer]["rotated"]["diversity_adjusted_cleanliness"]
                for layer in scores
            ),
            "values": {
                layer: {
                    family: scores[layer][family]["diversity_adjusted_cleanliness"]
                    for family in ("joint", "rotated")
                }
                for layer in scores
            },
        },
    }
    output = {"config": config, "scores": scores, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Diversity red-team of clean contextual roles",
        "",
        "A basis can look clean if every axis has the same dominant output role. This diagnostic therefore pairs within-axis role concentration with between-axis first-role diversity. Effective diversity is the participation rank of the 32 first-role directions; its maximum is 32.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "| Layer | Basis | First-role energy | Effective diversity | Mean pairwise abs cosine | Adjusted cleanliness |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for layer, families in scores.items():
        for family, row in families.items():
            lines.append(
                f"| {layer} | {family} | {row['first_role_energy']:.3f} | "
                f"{row['effective_role_diversity']:.2f} | {row['pairwise_role_cosine']:.3f} | "
                f"{row['diversity_adjusted_cleanliness']:.3f} |"
            )
    lines += [
        "",
        "The adjusted product is a diagnostic, not a canonical objective: it values stable within-axis roles and distinct roles across axes equally. It does not measure completeness or downstream necessity.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
