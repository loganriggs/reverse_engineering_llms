#!/usr/bin/env python3
"""Compare contextual role rank against rotated-subspace and ambient-axis controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_joint_axis_contextual_roles import role_summary
from run_bilin18_joint_axis_residual_roles import random_orthogonal
from run_bilin18_layer0_joint_forms import layer0_input
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_joint_axis_basis_controls.json")
OUTPUT = TENSOR_ROOT / "joint_axis_basis_controls.json"
REPORT = TENSOR_ROOT / "JOINT_AXIS_BASIS_CONTROLS.md"
FIGURE = TENSOR_ROOT / "joint_axis_basis_controls.png"


def summarize(roles: np.ndarray, config: dict) -> dict:
    rows = []
    for axis_index in range(32):
        row = role_summary(
            roles[:, axis_index],
            np.zeros(roles.shape[-1], dtype=np.float32),
            config["top_effect_fraction"],
            config["minimum_top_effect_sites"],
        )
        role_basis = row.pop("role_basis")
        row.pop("role_second_moment")
        row.pop("median_diagonal_absolute_cosine")
        row["first_role"] = role_basis[0].tolist()
        row["axis"] = axis_index
        rows.append(row)
    first_roles = np.asarray([row["first_role"] for row in rows])
    gram = np.abs(first_roles @ first_roles.T)
    off_diagonal = gram[~np.eye(len(gram), dtype=bool)]
    singular_energy = np.linalg.svd(first_roles, compute_uv=False) ** 2
    diversity = float(
        singular_energy.sum() ** 2 / max(np.square(singular_energy).sum(), 1e-30)
    )
    return {
        "first_component_energy": float(np.median([row["first_component_energy"] for row in rows])),
        "four_component_energy": float(np.median([row["four_component_energy"] for row in rows])),
        "rank90": float(np.median([row["rank90"] for row in rows])),
        "first_role_effective_diversity": diversity,
        "mean_pairwise_first_role_cosine": float(off_diagonal.mean()),
        "axes": rows,
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    atlas = np.load(ATLAS_ARRAYS)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    joint = {
        "layer0": torch.from_numpy(atlas["layer0_axes"]).cuda(),
        "layer17": torch.from_numpy(atlas["layer17_axes"]).cuda(),
    }
    bases = {"layer0": {"joint": [joint["layer0"]]}, "layer17": {"joint": [joint["layer17"]]}}
    for name in bases:
        bases[name]["rotated"] = []
        bases[name]["ambient"] = []
        for index in range(config["controls"]):
            rotation = random_orthogonal(32, 32, config["seed"] + 100 * index).cuda()
            bases[name]["rotated"].append(joint[name] @ rotation)
            bases[name]["ambient"].append(
                random_orthogonal(1152, 32, config["seed"] + 1000 + 100 * index).cuda()
            )

    sketch = random_orthogonal(1152, config["sketch_dimension"], config["seed"] + 9999).cuda()
    query = {}
    for name, layer in (("layer0", 0), ("layer17", 17)):
        block = model.blocks[layer]
        query[name] = {
            "coefficients": (block.mlp.Down.weight.float().T @ sketch).T,
            "basis": {},
        }
        for family, family_bases in bases[name].items():
            query[name]["basis"][family] = [
                {
                    "axes": axes,
                    "left_axes": block.mlp.Left.weight.float() @ axes,
                    "right_axes": block.mlp.Right.weight.float() @ axes,
                }
                for axes in family_bases
            ]

    source = np.load(config["corpus"], mmap_mode="r")
    rows = source[config["start"] : config["start"] + config["rows"]]
    total = len(rows) * len(config["positions"])
    collected = {
        name: {
            family: [
                np.empty((total, 32, config["sketch_dimension"]), dtype=np.float32)
                for _ in family_bases
            ]
            for family, family_bases in bases[name].items()
        }
        for name in ("layer0", "layer17")
    }

    offset = 0
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        states = {"layer0": layer0_input(model, ids).float()}
        _, states["layer17"] = prefix_to_layer17(model, ids)
        states["layer17"] = states["layer17"].float()
        count = len(raw) * len(config["positions"])
        for name in ("layer0", "layer17"):
            block = model.blocks[0 if name == "layer0" else 17]
            x = states[name][:, config["positions"]].flatten(0, 1)
            left_state = block.mlp.Left(x)
            right_state = block.mlp.Right(x)
            for family, family_rows in query[name]["basis"].items():
                for control_index, data in enumerate(family_rows):
                    coordinate = x @ data["axes"]
                    product_delta = -coordinate[:, None, :] * (
                        left_state[:, :, None] * data["right_axes"][None, :, :]
                        + right_state[:, :, None] * data["left_axes"][None, :, :]
                    ) + coordinate[:, None, :].square() * (
                        data["left_axes"] * data["right_axes"]
                    )[None, :, :]
                    effects = torch.einsum(
                        "rh,nhk->nkr", query[name]["coefficients"], product_delta
                    )
                    collected[name][family][control_index][offset : offset + count] = effects.cpu().numpy()
        offset += count
        if start % 40 == 0:
            print(min(start + config["batch_size"], len(rows)), "/", len(rows), flush=True)

    results = {}
    aggregate = {}
    for name in ("layer0", "layer17"):
        results[name] = {}
        aggregate[name] = {}
        for family, arrays in collected[name].items():
            family_rows = [summarize(array, config) for array in arrays]
            results[name][family] = family_rows
            aggregate[name][family] = {
                key: float(np.mean([row[key] for row in family_rows]))
                for key in (
                    "first_component_energy",
                    "four_component_energy",
                    "rank90",
                    "first_role_effective_diversity",
                    "mean_pairwise_first_role_cosine",
                )
            }
            aggregate[name][family]["first_component_cv"] = float(
                np.std([row["first_component_energy"] for row in family_rows])
                / max(np.mean([row["first_component_energy"] for row in family_rows]), 1e-20)
            )

    layer17_rank_reduction = 1 - aggregate["layer17"]["joint"]["rank90"] / aggregate["layer17"]["rotated"]["rank90"]
    layer0_rank_reduction = 1 - aggregate["layer0"]["joint"]["rank90"] / aggregate["layer0"]["rotated"]["rank90"]
    ambient_gap = aggregate["layer17"]["ambient"]["first_component_energy"] - aggregate["layer0"]["ambient"]["first_component_energy"]
    hypotheses = {
        "B1_layer17_joint_rank_advantage": {
            "supported": layer17_rank_reduction >= 0.20,
            "rank_reduction": layer17_rank_reduction,
        },
        "B2_layer0_joint_rank_advantage": {
            "supported": layer0_rank_reduction >= 0.20,
            "rank_reduction": layer0_rank_reduction,
        },
        "B3_layer17_joint_concentration": {
            "supported": aggregate["layer17"]["joint"]["first_component_energy"]
            - aggregate["layer17"]["rotated"]["first_component_energy"] >= 0.10,
            "gap": aggregate["layer17"]["joint"]["first_component_energy"]
            - aggregate["layer17"]["rotated"]["first_component_energy"],
        },
        "B4_architectural_depth_gap": {
            "supported": ambient_gap >= 0.20,
            "gap": ambient_gap,
        },
        "B5_control_stability": {
            "supported": all(
                aggregate[name][family]["first_component_cv"] < 0.10
                for name in aggregate for family in ("rotated", "ambient")
            ),
            "values": {
                name: {family: aggregate[name][family]["first_component_cv"] for family in ("rotated", "ambient")}
                for name in aggregate
            },
        },
    }
    output = {
        "config": config,
        "aggregate": aggregate,
        "hypotheses": hypotheses,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Basis controls for contextual residual roles",
        "",
        "The stable joint axes are compared with three random rotations of exactly the same 32D subspace and three random 32D ambient subspaces. Every condition uses the same held-out FineWeb states and random residual-write sketch.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Basis controls](joint_axis_basis_controls.png)",
        "",
        "| Layer | Basis | First role energy | Four-role energy | Rank for 90% | Control CV |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, families in aggregate.items():
        for family, row in families.items():
            lines.append(
                f"| {name} | {family} | {row['first_component_energy']:.3f} | "
                f"{row['four_component_energy']:.3f} | {row['rank90']:.1f} | "
                f"{row['first_component_cv']:.3f} |"
            )
    lines += [
        "",
        "Random rotations control whether the reported cleanliness comes from a particular coordinate choice; ambient controls test whether it is generic to the layer's weights. The residual sketch remains a local MLP-write measure and does not establish which write directions survive the rest of the network.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(10, 4.2))
    for column, name in enumerate(("layer0", "layer17")):
        labels = ["joint", "rotated", "ambient"]
        plots[column].bar(labels, [aggregate[name][label]["first_component_energy"] for label in labels])
        plots[column].set_ylim(0, 1)
        plots[column].set_ylabel("median first-role energy")
        plots[column].set_title(name)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
