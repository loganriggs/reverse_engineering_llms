#!/usr/bin/env python3
"""Measure how a recovered joint axis changes semantic role across contexts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS, CONFIG as ATLAS_CONFIG
from run_bilin18_layer0_joint_forms import layer0_input
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_joint_axis_contextual_roles.json")
OUTPUT = TENSOR_ROOT / "joint_axis_contextual_roles.json"
REPORT = TENSOR_ROOT / "JOINT_AXIS_CONTEXTUAL_ROLES.md"
FIGURE = TENSOR_ROOT / "joint_axis_contextual_roles.png"


def role_summary(roles: np.ndarray, diagonal: np.ndarray, fraction: float, minimum: int) -> dict:
    """Summarize directional variability for one axis's [site, concept] effects."""
    norms = np.linalg.norm(roles, axis=1)
    count = min(len(norms), max(minimum, int(round(fraction * len(norms)))))
    selected = np.argpartition(norms, -count)[-count:]
    normalized = roles[selected] / np.maximum(norms[selected, None], 1e-20)
    _, singular, vh = np.linalg.svd(normalized, full_matrices=False)
    energy = singular ** 2
    fractions = energy / max(energy.sum(), 1e-30)
    cumulative = np.cumsum(fractions)
    rank90 = int(np.searchsorted(cumulative, 0.9) + 1)
    diagonal_norm = np.linalg.norm(diagonal)
    diagonal_cosines = np.abs(normalized @ diagonal / max(diagonal_norm, 1e-20))
    mean_direction = normalized.T @ normalized
    return {
        "sites": int(count),
        "effect_norm_threshold": float(norms[selected].min()),
        "median_effect_norm": float(np.median(norms[selected])),
        "first_component_energy": float(fractions[0]),
        "four_component_energy": float(fractions[:4].sum()),
        "rank90": rank90,
        "median_diagonal_absolute_cosine": float(np.median(diagonal_cosines)),
        "role_basis": vh[:4],
        "role_second_moment": mean_direction / len(normalized),
    }


def canonical_overlap(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.svd(first @ second.T, compute_uv=False).mean())


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    atlas_config = json.loads(ATLAS_CONFIG.read_text())
    concepts = atlas_config["concepts"]
    atlas = np.load(ATLAS_ARRAYS)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    axes = {
        "layer0": torch.from_numpy(atlas["layer0_axes"]).cuda(),
        "layer17": torch.from_numpy(atlas["layer17_axes"]).cuda(),
    }

    query = {}
    for name, layer in (("layer0", 0), ("layer17", 17)):
        block = model.blocks[layer]
        left = block.mlp.Left.weight.float()
        right = block.mlp.Right.weight.float()
        down = block.mlp.Down.weight.float()
        vocabulary = (
            model.embedding.weight.float()[: tokenizer.vocab_size]
            if name == "layer0"
            else model.lm_head.weight.float()[: tokenizer.vocab_size]
        )
        coefficients = []
        for concept in concepts:
            positive = torch.as_tensor(
                flags[:, CATEGORIES.index(concept)].astype(bool), device="cuda"
            )
            direction = vocabulary[positive].mean(0) - vocabulary[~positive].mean(0)
            direction /= direction.norm().clamp_min(1e-20)
            coefficients.append(down.T @ direction)
        coefficients = torch.stack(coefficients)
        left_axes = left @ axes[name]
        right_axes = right @ axes[name]
        diagonal = -torch.einsum(
            "ch,hk,hk->kc", coefficients, left_axes, right_axes
        )
        query[name] = {
            "coefficients": coefficients,
            "left_axes": left_axes,
            "right_axes": right_axes,
            "diagonal": diagonal.cpu().numpy(),
        }

    role_arrays: dict[str, dict[str, np.ndarray]] = {}
    for corpus_name, corpus_config in config["corpora"].items():
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        total = len(rows) * len(config["positions"])
        collected = {
            "layer0": np.empty((total, 32, len(concepts)), dtype=np.float32),
            "layer17": np.empty((total, 32, len(concepts)), dtype=np.float32),
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
                # Product change after zeroing each coordinate k:
                # -a_k(l*q_k + r*p_k) + a_k^2 p_k*q_k.
                product_delta = -coordinate[:, None, :] * (
                    left_state[:, :, None] * data["right_axes"][None, :, :]
                    + right_state[:, :, None] * data["left_axes"][None, :, :]
                ) + coordinate[:, None, :].square() * (
                    data["left_axes"] * data["right_axes"]
                )[None, :, :]
                effects = torch.einsum(
                    "ch,nhk->nkc", data["coefficients"], product_delta
                )
                collected[name][offset : offset + count] = effects.cpu().numpy()
            offset += count
            if start % 200 == 0:
                print(corpus_name, min(start + config["batch_size"], len(rows)), "/", len(rows), flush=True)
        role_arrays[corpus_name] = collected

    axes_output = {}
    aggregate = {}
    bases = {}
    for corpus_name, layers in role_arrays.items():
        axes_output[corpus_name] = {}
        aggregate[corpus_name] = {}
        bases[corpus_name] = {}
        for name, roles in layers.items():
            rows = []
            layer_bases = []
            for axis_index in range(32):
                row = role_summary(
                    roles[:, axis_index],
                    query[name]["diagonal"][axis_index],
                    config["top_effect_fraction"],
                    config["minimum_top_effect_sites"],
                )
                layer_bases.append(row.pop("role_basis"))
                row.pop("role_second_moment")
                row["axis"] = axis_index
                rows.append(row)
            bases[corpus_name][name] = np.stack(layer_bases)
            axes_output[corpus_name][name] = rows
            aggregate[corpus_name][name] = {
                key: float(np.median([row[key] for row in rows]))
                for key in (
                    "first_component_energy",
                    "four_component_energy",
                    "rank90",
                    "median_diagonal_absolute_cosine",
                )
            }

    ood_overlap = {}
    for corpus_name in ("wikitext", "tinystories"):
        ood_overlap[corpus_name] = {}
        for name in ("layer0", "layer17"):
            values = [
                canonical_overlap(bases["fineweb"][name][axis], bases[corpus_name][name][axis])
                for axis in range(32)
            ]
            ood_overlap[corpus_name][name] = {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "per_axis": values,
            }

    fine = aggregate["fineweb"]
    hypotheses = {
        "C1_layer0_rank1_roles": {
            "supported": fine["layer0"]["first_component_energy"] >= 0.70,
            "value": fine["layer0"]["first_component_energy"],
        },
        "C2_depth_role_rank_gap": {
            "supported": fine["layer0"]["first_component_energy"]
            - fine["layer17"]["first_component_energy"] >= 0.08,
            "layer0": fine["layer0"]["first_component_energy"],
            "layer17": fine["layer17"]["first_component_energy"],
            "gap": fine["layer0"]["first_component_energy"]
            - fine["layer17"]["first_component_energy"],
        },
        "C3_low_dimensional_roles": {
            "supported": all(fine[name]["four_component_energy"] >= 0.90 for name in fine),
            "per_layer": {name: row["four_component_energy"] for name, row in fine.items()},
        },
        "C4_ood_role_subspaces": {
            "supported": all(
                ood_overlap[corpus][name]["mean"] >= 0.65
                for corpus in ood_overlap for name in ood_overlap[corpus]
            ),
            "values": {
                corpus: {name: row["mean"] for name, row in layers.items()}
                for corpus, layers in ood_overlap.items()
            },
        },
        "C5_diagonal_role_at_high_effect": {
            "supported": fine["layer0"]["median_diagonal_absolute_cosine"] >= 0.85
            and fine["layer17"]["median_diagonal_absolute_cosine"] >= 0.70,
            "per_layer": {
                name: row["median_diagonal_absolute_cosine"] for name, row in fine.items()
            },
        },
    }
    output = {
        "config": config,
        "concepts": concepts,
        "aggregate": aggregate,
        "ood_overlap": ood_overlap,
        "hypotheses": hypotheses,
        "axes": axes_output,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Contextual roles of stable joint axes",
        "",
        "For each recovered axis, this analysis exactly computes the 14-dimensional semantic-readout change caused by zeroing that coordinate. It then asks how many role directions are required across the highest-effect 10% of contexts. Sign reversals along one direction still count as one role; extra rank means the axis has genuinely different effects with different partners.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Contextual role ranks](joint_axis_contextual_roles.png)",
        "",
        "| Corpus | Layer | First role energy | Four-role energy | Rank for 90% | Diagonal-role cosine |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus_name, layers in aggregate.items():
        for name, row in layers.items():
            lines.append(
                f"| {corpus_name} | {name} | {row['first_component_energy']:.3f} | "
                f"{row['four_component_energy']:.3f} | {row['rank90']:.1f} | "
                f"{row['median_diagonal_absolute_cosine']:.3f} |"
            )
    lines += [
        "",
        "## OOD role-subspace overlap",
        "",
        "| Corpus | Layer 0 | Layer 17 |",
        "|---|---:|---:|",
    ]
    for corpus_name, layers in ood_overlap.items():
        lines.append(
            f"| {corpus_name} | {layers['layer0']['mean']:.3f} | {layers['layer17']['mean']:.3f} |"
        )
    lines += [
        "",
        "The role basis is computed from normalized finite-effect vectors, so frequent large effects do not automatically dominate the dimensionality estimate. The vocabulary of 14 token predicates is still only a restricted semantic probe, not a complete account of each axis.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    for column, name in enumerate(("layer0", "layer17")):
        positions = np.arange(32)
        plots[column].bar(
            positions - 0.18,
            [axes_output["fineweb"][name][axis]["first_component_energy"] for axis in range(32)],
            width=0.36,
            label="first role",
        )
        plots[column].bar(
            positions + 0.18,
            [axes_output["fineweb"][name][axis]["four_component_energy"] for axis in range(32)],
            width=0.36,
            label="first four",
        )
        plots[column].set_ylim(0, 1.02)
        plots[column].set_xlabel("stable joint axis")
        plots[column].set_ylabel("directional energy")
        plots[column].set_title(name)
        plots[column].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
