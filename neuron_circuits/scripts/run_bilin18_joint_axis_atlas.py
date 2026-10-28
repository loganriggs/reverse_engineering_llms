#!/usr/bin/env python3
"""Large-corpus semantic atlas for stable joint-diagonal axes at layers 0 and 17."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_joint_diagonalization import ARRAYS as ROTATION_ARRAYS
from run_bilin18_layer0_joint_forms import ARRAYS as LAYER0_ARRAYS, layer0_input
from run_bilin18_layer17_joint_forms import ARRAYS as LAYER17_ARRAYS
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_joint_axis_atlas.json")
OUTPUT = TENSOR_ROOT / "joint_axis_atlas.json"
ARRAYS = TENSOR_ROOT / "joint_axis_atlas.npz"
REPORT = TENSOR_ROOT / "JOINT_AXIS_ATLAS.md"
FIGURE = TENSOR_ROOT / "joint_axis_atlas.png"


def effective_support(values: np.ndarray) -> float:
    values = np.abs(np.asarray(values, dtype=np.float64))
    return float(values.sum() ** 2 / max(np.square(values).sum(), 1e-30))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    concepts = config["concepts"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    flags = np.stack([token_flags(text) for text in texts])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()

    layer0_data = np.load(LAYER0_ARRAYS)
    layer17_data = np.load(LAYER17_ARRAYS)
    rotations = np.load(ROTATION_ARRAYS)
    axes = {
        "layer0": torch.from_numpy(layer0_data["embedding_shared_vectors"][:, :32]).cuda()
        @ torch.from_numpy(rotations["layer0_rotation"]).cuda(),
        "layer17": torch.from_numpy(layer17_data["shared_vectors"][:, :32]).cuda()
        @ torch.from_numpy(rotations["layer17_rotation"]).cuda(),
    }
    vocabulary = {
        "layer0": model.embedding.weight.float()[: tokenizer.vocab_size],
        "layer17": model.lm_head.weight.float()[: tokenizer.vocab_size],
    }
    coefficient_profiles = {}
    native_alignment = {}
    for name, layer in (("layer0", 0), ("layer17", 17)):
        block = model.blocks[layer]
        left = block.mlp.Left.weight.float()
        right = block.mlp.Right.weight.float()
        down = block.mlp.Down.weight.float()
        projected_left = left @ axes[name]
        projected_right = right @ axes[name]
        rows = []
        for concept in concepts:
            positive = torch.as_tensor(
                flags[:, CATEGORIES.index(concept)].astype(bool), device="cuda"
            )
            direction = vocabulary[name][positive].mean(0) - vocabulary[name][~positive].mean(0)
            direction /= direction.norm().clamp_min(1e-20)
            channel_coefficients = down.T @ direction
            diagonal = channel_coefficients @ (projected_left * projected_right)
            rows.append(diagonal)
        coefficients = torch.stack(rows)
        coefficients /= coefficients.square().sum(1, keepdim=True).sqrt().clamp_min(1e-20)
        coefficient_profiles[name] = coefficients.cpu().numpy()

        normalized_axes = axes[name] / axes[name].norm(dim=0, keepdim=True).clamp_min(1e-20)
        normalized_left = left / left.norm(dim=1, keepdim=True).clamp_min(1e-20)
        normalized_right = right / right.norm(dim=1, keepdim=True).clamp_min(1e-20)
        left_overlap = (normalized_left @ normalized_axes).abs()
        right_overlap = (normalized_right @ normalized_axes).abs()
        combined = torch.maximum(left_overlap.max(0).values, right_overlap.max(0).values)
        native_alignment[name] = {
            "maximum_cosine": combined.cpu().numpy(),
            "left_maximum": left_overlap.max(0).values.cpu().numpy(),
            "left_unit": left_overlap.argmax(0).cpu().numpy(),
            "right_maximum": right_overlap.max(0).values.cpu().numpy(),
            "right_unit": right_overlap.argmax(0).cpu().numpy(),
        }

    all_rows = np.load(config["corpus"], mmap_mode="r")
    rows = all_rows[config["start"] : config["start"] + config["rows"]]
    positions = config["positions"]
    position_tensor = torch.as_tensor(positions, device="cuda")
    total = len(rows) * len(positions)
    activations = {
        "layer0": np.empty((total, 32), dtype=np.float32),
        "layer17": np.empty((total, 32), dtype=np.float32),
    }
    current_ids = np.empty(total, dtype=np.int64)
    next_ids = np.empty(total, dtype=np.int64)
    site_rows = np.empty(total, dtype=np.int32)
    site_positions = np.empty(total, dtype=np.int16)
    offset = 0
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0 = layer0_input(model, ids)[:, position_tensor].float()
        _, x17_all = prefix_to_layer17(model, ids)
        x17 = x17_all[:, position_tensor].float()
        count = raw.shape[0] * len(positions)
        activations["layer0"][offset : offset + count] = (
            (x0 @ axes["layer0"]).square().flatten(0, 1).cpu().numpy()
        )
        activations["layer17"][offset : offset + count] = (
            (x17 @ axes["layer17"]).square().flatten(0, 1).cpu().numpy()
        )
        current_ids[offset : offset + count] = raw[:, positions].reshape(-1)
        next_ids[offset : offset + count] = raw[:, np.asarray(positions) + 1].reshape(-1)
        site_rows[offset : offset + count] = np.repeat(np.arange(start, start + raw.shape[0]), len(positions))
        site_positions[offset : offset + count] = np.tile(positions, raw.shape[0])
        offset += count
        if start % 200 == 0:
            print("axis atlas", min(start + config["batch_size"], len(rows)), "/", len(rows), flush=True)

    atlas = {}
    summary = {}
    for name, token_ids in (("layer0", current_ids), ("layer17", next_ids)):
        labels = np.stack([
            flags[token_ids, CATEGORIES.index(concept)].astype(bool)
            for concept in concepts
        ], axis=1)
        coefficient = coefficient_profiles[name]
        axis_rows = []
        semantic_count = 0
        agreement_count = 0
        supports = []
        for axis_index in range(32):
            scores = activations[name][:, axis_index]
            aurocs = np.asarray([auc(labels[:, index], scores) for index in range(len(concepts))])
            best_activation_index = int(np.nanargmax(aurocs))
            best_coefficient_index = int(np.argmax(np.abs(coefficient[:, axis_index])))
            if aurocs[best_activation_index] >= (0.70 if name == "layer0" else 0.65):
                semantic_count += 1
            if best_activation_index == best_coefficient_index:
                agreement_count += 1
            support = effective_support(coefficient[:, axis_index])
            supports.append(support)
            top_indices = np.argpartition(scores, -config["top_examples"])[-config["top_examples"] :]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            examples = []
            for index in top_indices:
                row_index = int(site_rows[index])
                position = int(site_positions[index])
                context_ids = np.asarray(
                    rows[row_index, max(0, position - 10) : min(129, position + 2)],
                    dtype=np.int64,
                )
                examples.append({
                    "row": row_index,
                    "position": position,
                    "score": float(scores[index]),
                    "current_token": texts[int(current_ids[index])],
                    "next_token": texts[int(next_ids[index])],
                    "context": tokenizer.decode(context_ids.tolist()),
                })
            axis_rows.append({
                "axis": axis_index,
                "best_activation_concept": concepts[best_activation_index],
                "best_activation_auc": float(aurocs[best_activation_index]),
                "activation_aurocs": {concept: float(aurocs[index]) for index, concept in enumerate(concepts)},
                "strongest_coefficient_concept": concepts[best_coefficient_index],
                "strongest_normalized_coefficient": float(coefficient[best_coefficient_index, axis_index]),
                "normalized_coefficients": {concept: float(coefficient[index, axis_index]) for index, concept in enumerate(concepts)},
                "effective_concept_support": support,
                "native_alignment": {
                    key: (int(value[axis_index]) if "unit" in key else float(value[axis_index]))
                    for key, value in native_alignment[name].items()
                },
                "activation_quantiles": {
                    str(q): float(np.quantile(scores, q)) for q in (0.5, 0.9, 0.99, 0.999)
                },
                "top_examples": examples,
            })
        atlas[name] = axis_rows
        summary[name] = {
            "semantic_axes": semantic_count,
            "coefficient_activation_agreement": agreement_count / 32,
            "median_effective_concept_support": float(np.median(supports)),
            "median_maximum_native_factor_cosine": float(
                np.median(native_alignment[name]["maximum_cosine"])
            ),
            "median_best_activation_auc": float(
                np.median([row["best_activation_auc"] for row in axis_rows])
            ),
        }

    cross_depth_overlap = (axes["layer0"].T @ axes["layer17"]).abs().cpu().numpy()
    hypotheses = {
        "V1_layer0_axis_semantics": {
            "supported": summary["layer0"]["semantic_axes"] >= 16,
            "axes": summary["layer0"]["semantic_axes"],
        },
        "V2_layer17_axis_semantics": {
            "supported": summary["layer17"]["semantic_axes"] >= 8,
            "axes": summary["layer17"]["semantic_axes"],
        },
        "V3_coefficient_concentration": {
            "supported": all(row["median_effective_concept_support"] <= 7 for row in summary.values()),
            "per_layer": {name: row["median_effective_concept_support"] for name, row in summary.items()},
        },
        "V4_non_native_axes": {
            "supported": all(row["median_maximum_native_factor_cosine"] < 0.50 for row in summary.values()),
            "per_layer": {name: row["median_maximum_native_factor_cosine"] for name, row in summary.items()},
        },
        "V5_coefficient_activation_agreement": {
            "supported": all(row["coefficient_activation_agreement"] >= 0.25 for row in summary.values()),
            "per_layer": {name: row["coefficient_activation_agreement"] for name, row in summary.items()},
        },
    }
    output = {
        "config": config,
        "summary": summary,
        "hypotheses": hypotheses,
        "cross_depth": {
            "mean_best_axis_overlap": float(cross_depth_overlap.max(1).mean()),
            "median_best_axis_overlap": float(np.median(cross_depth_overlap.max(1))),
            "matrix": cross_depth_overlap.tolist(),
        },
        "axes": atlas,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        layer0_activations=activations["layer0"],
        layer17_activations=activations["layer17"],
        current_ids=current_ids,
        next_ids=next_ids,
        site_rows=site_rows,
        site_positions=site_positions,
        layer0_axes=axes["layer0"].cpu().numpy(),
        layer17_axes=axes["layer17"].cpu().numpy(),
        layer0_coefficients=coefficient_profiles["layer0"],
        layer17_coefficients=coefficient_profiles["layer17"],
    )

    lines = [
        "# Stable joint-axis semantic atlas",
        "",
        f"The 32 stable axes at each layer are scanned over {total:,} FineWeb positions. Axis activation is squared coordinate magnitude, matching the diagonalized quadratic representation. Coefficients cover 14 frozen token predicates; activation labels use the current token at layer 0 and next token at layer 17.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Axis atlas](joint_axis_atlas.png)",
        "",
        "| Layer | Semantic axes | Median best AUROC | Effective concept support | Native-factor cosine | Coefficient/activation agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.items():
        lines.append(
            f"| {name} | {row['semantic_axes']}/32 | {row['median_best_activation_auc']:.3f} | "
            f"{row['median_effective_concept_support']:.2f} | "
            f"{row['median_maximum_native_factor_cosine']:.3f} | "
            f"{row['coefficient_activation_agreement']:.1%} |"
        )
    lines += [
        "",
        f"Cross-depth mean best-axis overlap is **{output['cross_depth']['mean_best_axis_overlap']:.3f}**. Full coefficient profiles, AUROCs, native-factor matches, and top contexts are in `joint_axis_atlas.json`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes_plot = plt.subplots(2, 2, figsize=(14, 8))
    for column, name in enumerate(("layer0", "layer17")):
        image = axes_plot[0, column].imshow(coefficient_profiles[name], aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        axes_plot[0, column].set_yticks(range(len(concepts)), concepts)
        axes_plot[0, column].set_xlabel("joint axis")
        axes_plot[0, column].set_title(f"{name} normalized coefficients")
        fig.colorbar(image, ax=axes_plot[0, column], fraction=0.025)
        axes_plot[1, column].hist([row["best_activation_auc"] for row in atlas[name]], bins=12)
        axes_plot[1, column].set_xlabel("best token-predicate AUROC")
        axes_plot[1, column].set_ylabel("axes")
        axes_plot[1, column].set_title(f"{name} activation semantics")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
