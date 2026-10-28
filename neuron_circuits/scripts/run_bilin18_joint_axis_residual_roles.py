#!/usr/bin/env python3
"""Red-team contextual role rank with random sketches of the full residual write."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_joint_axis_contextual_roles import canonical_overlap, role_summary
from run_bilin18_layer0_joint_forms import layer0_input
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_joint_axis_residual_roles.json")
SEMANTIC = TENSOR_ROOT / "joint_axis_contextual_roles.json"
OUTPUT = TENSOR_ROOT / "joint_axis_residual_roles.json"
REPORT = TENSOR_ROOT / "JOINT_AXIS_RESIDUAL_ROLES.md"
FIGURE = TENSOR_ROOT / "joint_axis_residual_roles.png"


def random_orthogonal(dimension: int, columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(dimension, columns, generator=generator)
    return torch.linalg.qr(raw, mode="reduced").Q


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    semantic = json.loads(SEMANTIC.read_text())
    atlas = np.load(ATLAS_ARRAYS)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    axes = {
        "layer0": torch.from_numpy(atlas["layer0_axes"]).cuda(),
        "layer17": torch.from_numpy(atlas["layer17_axes"]).cuda(),
    }
    sketch_dimension = config["sketch_dimension"]
    sketches = []
    for index in range(config["sketches"]):
        sketches.append(random_orthogonal(1152, sketch_dimension, config["seed"] + index))
    combined_sketch = torch.cat(sketches, dim=1).cuda()

    query = {}
    for name, layer in (("layer0", 0), ("layer17", 17)):
        block = model.blocks[layer]
        query[name] = {
            "left_axes": block.mlp.Left.weight.float() @ axes[name],
            "right_axes": block.mlp.Right.weight.float() @ axes[name],
            "sketch_coefficients": (block.mlp.Down.weight.float().T @ combined_sketch).T,
        }

    aggregate = {}
    axes_output = {}
    bases = {}
    for corpus_name, corpus_config in config["corpora"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        total = len(rows) * len(config["positions"])
        aggregate[corpus_name] = {}
        axes_output[corpus_name] = {}
        bases[corpus_name] = {}
        collected = {
            "layer0": np.empty((total, 32, config["sketches"] * sketch_dimension), dtype=np.float32),
            "layer17": np.empty((total, 32, config["sketches"] * sketch_dimension), dtype=np.float32),
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
                coordinate = x @ axes[name]
                data = query[name]
                product_delta = -coordinate[:, None, :] * (
                    left_state[:, :, None] * data["right_axes"][None, :, :]
                    + right_state[:, :, None] * data["left_axes"][None, :, :]
                ) + coordinate[:, None, :].square() * (
                    data["left_axes"] * data["right_axes"]
                )[None, :, :]
                effects = torch.einsum(
                    "rh,nhk->nkr", data["sketch_coefficients"], product_delta
                )
                collected[name][offset : offset + count] = effects.cpu().numpy()
            offset += count
            if start % 100 == 0:
                print(corpus_name, min(start + config["batch_size"], len(rows)), "/", len(rows), flush=True)

        for name in ("layer0", "layer17"):
            axes_output[corpus_name][name] = []
            bases[corpus_name][name] = [[] for _ in range(config["sketches"])]
            per_sketch = []
            for sketch_index in range(config["sketches"]):
                begin = sketch_index * sketch_dimension
                end = begin + sketch_dimension
                sketch_rows = []
                for axis_index in range(32):
                    # The diagonal template is not part of this red-team; a zero vector
                    # keeps role_summary focused on dimensionality and returns zero cosine.
                    row = role_summary(
                        collected[name][:, axis_index, begin:end],
                        np.zeros(sketch_dimension, dtype=np.float32),
                        config["top_effect_fraction"],
                        config["minimum_top_effect_sites"],
                    )
                    bases[corpus_name][name][sketch_index].append(row.pop("role_basis"))
                    row.pop("role_second_moment")
                    row.pop("median_diagonal_absolute_cosine")
                    row["axis"] = axis_index
                    sketch_rows.append(row)
                axes_output[corpus_name][name].append(sketch_rows)
                per_sketch.append({
                    key: float(np.median([row[key] for row in sketch_rows]))
                    for key in ("first_component_energy", "four_component_energy", "rank90")
                })
            bases[corpus_name][name] = np.asarray(bases[corpus_name][name])
            aggregate[corpus_name][name] = {"per_sketch": per_sketch}
            aggregate[corpus_name][name].update({
                key: float(np.mean([row[key] for row in per_sketch]))
                for key in ("first_component_energy", "four_component_energy", "rank90")
            })
            aggregate[corpus_name][name]["first_component_cv"] = float(
                np.std([row["first_component_energy"] for row in per_sketch])
                / max(np.mean([row["first_component_energy"] for row in per_sketch]), 1e-20)
            )
        del collected

    ood_overlap = {}
    for corpus_name in ("wikitext", "tinystories"):
        ood_overlap[corpus_name] = {}
        for name in ("layer0", "layer17"):
            values = []
            for sketch_index in range(config["sketches"]):
                values.extend([
                    canonical_overlap(
                        bases["fineweb"][name][sketch_index, axis],
                        bases[corpus_name][name][sketch_index, axis],
                    )
                    for axis in range(32)
                ])
            ood_overlap[corpus_name][name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
            }

    fine = aggregate["fineweb"]
    differences = {
        name: abs(
            fine[name]["first_component_energy"]
            - semantic["aggregate"]["fineweb"][name]["first_component_energy"]
        )
        for name in ("layer0", "layer17")
    }
    hypotheses = {
        "R1_depth_ordering": {
            "supported": all(
                fine["layer17"]["per_sketch"][index]["first_component_energy"]
                - fine["layer0"]["per_sketch"][index]["first_component_energy"] >= 0.08
                for index in range(config["sketches"])
            ),
            "per_sketch_gap": [
                fine["layer17"]["per_sketch"][index]["first_component_energy"]
                - fine["layer0"]["per_sketch"][index]["first_component_energy"]
                for index in range(config["sketches"])
            ],
        },
        "R2_low_residual_role_rank": {
            "supported": all(
                row["four_component_energy"] >= 0.75
                for name in fine for row in fine[name]["per_sketch"]
            ),
            "per_layer": {name: row["four_component_energy"] for name, row in fine.items()},
        },
        "R3_sketch_stability": {
            "supported": all(row["first_component_cv"] < 0.05 for row in fine.values()),
            "per_layer_cv": {name: row["first_component_cv"] for name, row in fine.items()},
        },
        "R4_ood_residual_role_subspaces": {
            "supported": all(
                row["mean"] >= 0.60 for corpus in ood_overlap.values() for row in corpus.values()
            ),
            "values": ood_overlap,
        },
        "R5_semantic_probe_not_rank_artifact": {
            "supported": all(value <= 0.10 for value in differences.values()),
            "absolute_difference": differences,
        },
    }
    output = {
        "config": config,
        "aggregate": aggregate,
        "ood_overlap": ood_overlap,
        "hypotheses": hypotheses,
        "axes": axes_output,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Random-residual-sketch red-team of contextual roles",
        "",
        "The prior contextual-role result used 14 semantic queries. This red-team instead projects each exact 1,152-dimensional MLP write change into three independent 96-dimensional random orthogonal sketches. A consistent result is evidence about residual-write geometry rather than the hand-built semantic vocabulary.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Residual role ranks](joint_axis_residual_roles.png)",
        "",
        "| Corpus | Layer | First role energy | Four-role energy | Rank for 90% | Sketch CV |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus_name, layers in aggregate.items():
        for name, row in layers.items():
            lines.append(
                f"| {corpus_name} | {name} | {row['first_component_energy']:.3f} | "
                f"{row['four_component_energy']:.3f} | {row['rank90']:.1f} | "
                f"{row['first_component_cv']:.3f} |"
            )
    lines += [
        "",
        "| OOD corpus | Layer 0 overlap | Layer 17 overlap |",
        "|---|---:|---:|",
    ]
    for corpus_name, layers in ood_overlap.items():
        lines.append(
            f"| {corpus_name} | {layers['layer0']['mean']:.3f} | {layers['layer17']['mean']:.3f} |"
        )
    lines += [
        "",
        "Random sketches approximately preserve residual-write geometry but are not exact decompositions of the full 1,152-dimensional space. Agreement across three sketches is the principal safeguard.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(10, 4.1))
    for column, name in enumerate(("layer0", "layer17")):
        for sketch_index in range(config["sketches"]):
            plots[column].scatter(
                range(32),
                [axes_output["fineweb"][name][sketch_index][axis]["first_component_energy"] for axis in range(32)],
                s=16,
                alpha=0.65,
                label=f"sketch {sketch_index + 1}",
            )
        plots[column].set_ylim(0, 1.02)
        plots[column].set_xlabel("stable joint axis")
        plots[column].set_ylabel("first role-direction energy")
        plots[column].set_title(name)
        plots[column].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
