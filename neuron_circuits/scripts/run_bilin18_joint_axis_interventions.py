#!/usr/bin/env python3
"""Finite coordinate-zeroing validation of stable joint axes."""

from __future__ import annotations

import hashlib
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


CONFIG = Path("configs/bilin18_joint_axis_interventions.json")
OUTPUT = TENSOR_ROOT / "joint_axis_interventions.json"
REPORT = TENSOR_ROOT / "JOINT_AXIS_INTERVENTIONS.md"
FIGURE = TENSOR_ROOT / "joint_axis_interventions.png"


def stable_rng(seed: int, key: str) -> np.random.Generator:
    offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed + offset)


def vector_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict:
    target = target.float()
    prediction = prediction.float()
    energy = target.square().sum().clamp_min(1e-30)
    cosine = float(
        (target * prediction).sum()
        / torch.sqrt(energy * prediction.square().sum()).clamp_min(1e-30)
    )
    fidelity = float(1 - (prediction - target).square().sum() / energy)
    return {"cosine": cosine, "fidelity": fidelity, "target_energy": float(energy)}


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    atlas_config = json.loads(ATLAS_CONFIG.read_text())
    atlas = np.load(ATLAS_ARRAYS)
    concepts = atlas_config["concepts"]
    original_indices = torch.as_tensor(
        [concepts.index(concept) for concept in config["original_concepts"]],
        device="cuda",
        dtype=torch.long,
    )
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
    activation_arrays = {
        "layer0": atlas["layer0_activations"],
        "layer17": atlas["layer17_activations"],
    }
    site_rows = atlas["site_rows"]
    site_positions = atlas["site_positions"]
    all_rows = np.load(atlas_config["corpus"], mmap_mode="r")
    rows = all_rows[
        atlas_config["start"] : atlas_config["start"] + atlas_config["rows"]
    ]

    query_data = {}
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
        channel_rows = []
        for concept in concepts:
            positive = torch.as_tensor(
                flags[:, CATEGORIES.index(concept)].astype(bool), device="cuda"
            )
            direction = vocabulary[positive].mean(0) - vocabulary[~positive].mean(0)
            direction /= direction.norm().clamp_min(1e-20)
            channel_rows.append(down.T @ direction)
        channel_coefficients = torch.stack(channel_rows)
        projected_left = left @ axes[name]
        projected_right = right @ axes[name]
        restricted = torch.einsum(
            "ck,ki,kj->cij", channel_coefficients, projected_left, projected_right
        )
        restricted = 0.5 * (restricted + restricted.transpose(1, 2))
        query_data[name] = {
            "left": left,
            "right": right,
            "channel_coefficients": channel_coefficients,
            "restricted": restricted,
        }

    requests = {"layer0": [], "layer17": []}
    size = config["sites_per_axis_per_stratum"]
    for name in requests:
        for axis_index in range(32):
            scores = activation_arrays[name][:, axis_index]
            top = np.argpartition(scores, -size)[-size:]
            top = top[np.argsort(scores[top])[::-1]]
            excluded = np.ones(len(scores), dtype=bool)
            excluded[top] = False
            pool = np.flatnonzero(excluded)
            rng = stable_rng(config["seed"], f"{name}/{axis_index}")
            random_sites = rng.choice(pool, size=size, replace=False)
            for stratum, indices in (("top", top), ("random", random_sites)):
                for index in indices:
                    requests[name].append({
                        "axis": axis_index,
                        "stratum": stratum,
                        "site_index": int(index),
                        "row": int(site_rows[index]),
                        "position": int(site_positions[index]),
                    })

    needed_rows = sorted(set(
        request["row"] for layer_requests in requests.values() for request in layer_requests
    ))
    states = {"layer0": {}, "layer17": {}}
    for start in range(0, len(needed_rows), 4):
        indices = needed_rows[start : start + 4]
        raw = np.asarray(rows[indices, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0 = layer0_input(model, ids).float()
        _, x17 = prefix_to_layer17(model, ids)
        x17 = x17.float()
        for batch_index, row_index in enumerate(indices):
            states["layer0"][row_index] = x0[batch_index]
            states["layer17"][row_index] = x17[batch_index]

    records = []
    for name in ("layer0", "layer17"):
        block = model.blocks[0 if name == "layer0" else 17]
        data = query_data[name]
        for request in requests[name]:
            x = states[name][request["row"]][request["position"]]
            axis_index = request["axis"]
            coordinate = x @ axes[name][:, axis_index]
            changed_x = x - coordinate * axes[name][:, axis_index]
            native_product = block.mlp.Left(x) * block.mlp.Right(x)
            changed_product = block.mlp.Left(changed_x) * block.mlp.Right(changed_x)
            actual = data["channel_coefficients"] @ (changed_product - native_product)
            z = x @ axes[name]
            changed_z = z.clone()
            changed_z[axis_index] = 0
            form = data["restricted"]
            full_prediction = torch.einsum("ci,i->c", torch.einsum("cij,j->ci", form, changed_z), changed_z) - torch.einsum("ci,i->c", torch.einsum("cij,j->ci", form, z), z)
            diagonal_prediction = -form[:, axis_index, axis_index] * coordinate.square()
            records.append({
                "layer": name,
                **request,
                "coordinate_squared": float(coordinate.square()),
                "original_diagonal": vector_metrics(
                    actual[original_indices], diagonal_prediction[original_indices]
                ),
                "original_full_subspace": vector_metrics(
                    actual[original_indices], full_prediction[original_indices]
                ),
                "all_diagonal": vector_metrics(actual, diagonal_prediction),
                "all_full_subspace": vector_metrics(actual, full_prediction),
            })

    summary = {}
    for name in ("layer0", "layer17"):
        summary[name] = {}
        for stratum in config["strata"]:
            selected = [
                row for row in records if row["layer"] == name and row["stratum"] == stratum
            ]
            summary[name][stratum] = {
                method: {
                    metric: float(np.median([row[method][metric] for row in selected]))
                    for metric in ("cosine", "fidelity")
                }
                for method in (
                    "original_diagonal",
                    "original_full_subspace",
                    "all_diagonal",
                    "all_full_subspace",
                )
            }

    aggregate = {
        name: {
            method: {
                metric: float(np.median([
                    row[method][metric] for row in records if row["layer"] == name
                ]))
                for metric in ("cosine", "fidelity")
            }
            for method in (
                "original_diagonal",
                "original_full_subspace",
                "all_diagonal",
                "all_full_subspace",
            )
        }
        for name in ("layer0", "layer17")
    }
    hypotheses = {
        "I1_layer0_diagonal_execution": {
            "supported": aggregate["layer0"]["original_diagonal"]["cosine"] >= 0.95
            and aggregate["layer0"]["original_diagonal"]["fidelity"] >= 0.80,
            **aggregate["layer0"]["original_diagonal"],
        },
        "I2_layer17_diagonal_execution": {
            "supported": aggregate["layer17"]["original_diagonal"]["cosine"] >= 0.90
            and aggregate["layer17"]["original_diagonal"]["fidelity"] >= 0.60,
            **aggregate["layer17"]["original_diagonal"],
        },
        "I3_open_vocabulary_execution": {
            "supported": all(aggregate[name]["all_diagonal"]["cosine"] >= 0.80 for name in aggregate),
            "per_layer": {name: row["all_diagonal"]["cosine"] for name, row in aggregate.items()},
        },
        "I4_interaction_gain": {
            "supported": aggregate["layer17"]["all_full_subspace"]["fidelity"]
            - aggregate["layer17"]["all_diagonal"]["fidelity"] >= 0.15,
            "diagonal": aggregate["layer17"]["all_diagonal"]["fidelity"],
            "full_subspace": aggregate["layer17"]["all_full_subspace"]["fidelity"],
            "gain": aggregate["layer17"]["all_full_subspace"]["fidelity"]
            - aggregate["layer17"]["all_diagonal"]["fidelity"],
        },
        "I5_activation_regime_stability": {
            "supported": all(
                abs(summary[name]["top"]["original_diagonal"]["cosine"]
                    - summary[name]["random"]["original_diagonal"]["cosine"]) <= 0.10
                for name in summary
            ),
            "per_layer_gap": {
                name: summary[name]["top"]["original_diagonal"]["cosine"]
                - summary[name]["random"]["original_diagonal"]["cosine"]
                for name in summary
            },
        },
    }
    output = {
        "config": config,
        "aggregate": aggregate,
        "summary": summary,
        "hypotheses": hypotheses,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Finite joint-axis intervention validation",
        "",
        "Each stable coordinate is zeroed at eight top-activation and eight random FineWeb sites. Actual changes in 14 exact output-conditioned MLP scalar forms are compared with the diagonal-axis prediction and with the complete 32D interaction-matrix prediction.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Axis interventions](joint_axis_interventions.png)",
        "",
        "| Layer | Stratum | Original diagonal | Original full matrix | All-14 diagonal | All-14 full matrix |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in summary:
        for stratum in config["strata"]:
            row = summary[name][stratum]
            lines.append(
                f"| {name} | {stratum} | {row['original_diagonal']['cosine']:.3f} / {row['original_diagonal']['fidelity']:.3f} | "
                f"{row['original_full_subspace']['cosine']:.3f} / {row['original_full_subspace']['fidelity']:.3f} | "
                f"{row['all_diagonal']['cosine']:.3f} / {row['all_diagonal']['fidelity']:.3f} | "
                f"{row['all_full_subspace']['cosine']:.3f} / {row['all_full_subspace']['fidelity']:.3f} |"
            )
    lines += ["", "Cells show median cosine / fidelity. Per-axis finite interventions are in `joint_axis_interventions.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    methods = ["original_diagonal", "original_full_subspace", "all_diagonal", "all_full_subspace"]
    fig, axes_plot = plt.subplots(1, 2, figsize=(11, 4.3))
    for column, name in enumerate(("layer0", "layer17")):
        for stratum in config["strata"]:
            axes_plot[column].plot(
                range(len(methods)),
                [summary[name][stratum][method]["fidelity"] for method in methods],
                marker="o",
                label=stratum,
            )
        axes_plot[column].set_xticks(range(len(methods)), methods, rotation=35, ha="right")
        axes_plot[column].set_ylabel("median finite-effect fidelity")
        axes_plot[column].set_title(name)
        axes_plot[column].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
