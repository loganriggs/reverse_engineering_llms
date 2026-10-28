#!/usr/bin/env python3
"""Propagate selected layer-0 joint-axis interventions through the full model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS, OUTPUT as ATLAS_OUTPUT
from run_bilin18_joint_axis_residual_roles import random_orthogonal
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer0_downstream_roles.json")
OUTPUT = TENSOR_ROOT / "layer0_downstream_roles.json"
REPORT = TENSOR_ROOT / "LAYER0_DOWNSTREAM_ROLES.md"
FIGURE = TENSOR_ROOT / "layer0_downstream_roles.png"


def directional_summary(vectors: np.ndarray) -> dict:
    norms = np.linalg.norm(vectors, axis=1)
    normalized = vectors / np.maximum(norms[:, None], 1e-20)
    _, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    energy = singular ** 2
    fractions = energy / max(energy.sum(), 1e-30)
    return {
        "first_component_energy": float(fractions[0]),
        "four_component_energy": float(fractions[:4].sum()),
        "rank90": int(np.searchsorted(np.cumsum(fractions), 0.9) + 1),
        "median_effect_norm": float(np.median(norms)),
    }


def repeat_batch(value: torch.Tensor | None, count: int) -> torch.Tensor | None:
    if value is None:
        return None
    return value.repeat((count,) + (1,) * (value.ndim - 1))


@torch.inference_mode()
def layer0_prefix(model, ids: torch.Tensor):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    block = model.blocks[0]
    mixed = block.lambdas[0] * x0 + block.lambdas[1] * x0
    attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), None)
    pre = mixed + attention
    mlp_input = F.rms_norm(pre, (dimension,))
    product = block.mlp.Left(mlp_input) * block.mlp.Right(mlp_input)
    write = block.mlp.Down(product) + block.mlp.Down_bias
    return x0, value_bus, pre, mlp_input, product, write, pre + write


@torch.inference_mode()
def propagate(model, x: torch.Tensor, value_bus: torch.Tensor | None, x0: torch.Tensor):
    dimension = model.config.n_embd
    for block in model.blocks[1:]:
        x, value_bus = block(x, value_bus, x0)
    return F.rms_norm(x, (dimension,))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    atlas_arrays = np.load(ATLAS_ARRAYS)
    atlas = json.loads(ATLAS_OUTPUT.read_text())
    selected_axes = [axis for group in config["axis_groups"].values() for axis in group]
    axes = torch.from_numpy(atlas_arrays["layer0_axes"]).cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block0 = model.blocks[0]
    source = np.load(config["corpus"], mmap_mode="r")
    rows = source[config["start"] : config["start"] + config["scan_rows"]]
    sketch = random_orthogonal(1152, 64, config["seed"]).cuda()
    sketch_coefficients = (block0.mlp.Down.weight.float().T @ sketch).T
    left_axes = block0.mlp.Left.weight.float() @ axes[:, selected_axes]
    right_axes = block0.mlp.Right.weight.float() @ axes[:, selected_axes]

    scores = np.empty((len(rows), len(selected_axes)), dtype=np.float32)
    for start in range(0, len(rows), config["scan_batch_size"]):
        raw = np.asarray(rows[start : start + config["scan_batch_size"], : config["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, _, _, mlp_input, _, _, _ = layer0_prefix(model, ids)
        x = mlp_input[:, config["position"]]
        left_state = block0.mlp.Left(x)
        right_state = block0.mlp.Right(x)
        coordinate = x @ axes[:, selected_axes]
        product_delta = -coordinate[:, None, :] * (
            left_state[:, :, None] * right_axes[None, :, :]
            + right_state[:, :, None] * left_axes[None, :, :]
        ) + coordinate[:, None, :].square() * (left_axes * right_axes)[None, :, :]
        effect_sketch = torch.einsum("rh,nhk->nkr", sketch_coefficients, product_delta)
        scores[start : start + len(raw)] = effect_sketch.square().sum(-1).cpu().numpy()
        if start % 100 == 0:
            print("scan", min(start + config["scan_batch_size"], len(rows)), "/", len(rows), flush=True)

    selected_sites = {}
    requests_by_row: dict[int, list[int]] = {}
    for column, axis in enumerate(selected_axes):
        indices = np.argpartition(scores[:, column], -config["sites_per_axis"])[-config["sites_per_axis"] :]
        indices = indices[np.argsort(scores[indices, column])[::-1]]
        selected_sites[axis] = indices.tolist()
        for index in indices:
            requests_by_row.setdefault(int(index), []).append(axis)

    local_effects = {axis: [] for axis in selected_axes}
    final_effects = {axis: [] for axis in selected_axes}
    records = []
    for progress, (row_index, requested_axes) in enumerate(sorted(requests_by_row.items())):
        raw = np.asarray(rows[row_index : row_index + 1, : config["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0, value_bus, pre, mlp_input, product, write, native_x = layer0_prefix(model, ids)
        position = config["position"]
        variants = [native_x]
        local_deltas = []
        for axis in requested_axes:
            direction = axes[:, axis]
            x = mlp_input[0, position]
            changed_input = x - (x @ direction) * direction
            changed_product = block0.mlp.Left(changed_input) * block0.mlp.Right(changed_input)
            changed_write = block0.mlp.Down(changed_product) + block0.mlp.Down_bias
            changed_x = native_x.clone()
            changed_x[0, position] = pre[0, position] + changed_write
            variants.append(changed_x)
            local_deltas.append(changed_write - write[0, position])
        batch_count = len(variants)
        start_x = torch.cat(variants, dim=0)
        final = propagate(
            model,
            start_x,
            repeat_batch(value_bus, batch_count),
            repeat_batch(x0, batch_count),
        )[:, position]
        logits = 30 * torch.tanh(model.lm_head(final).float() / 30)
        native_log_prob = logits[0].log_softmax(-1)
        native_probability = native_log_prob.exp()
        target = int(rows[row_index, config["sequence_length"]])
        for variant_index, (axis, local_delta) in enumerate(zip(requested_axes, local_deltas), start=1):
            final_delta = final[variant_index] - final[0]
            changed_log_prob = logits[variant_index].log_softmax(-1)
            kl = float((native_probability * (native_log_prob - changed_log_prob)).sum())
            target_change = float(logits[variant_index, target] - logits[0, target])
            local_effects[axis].append(local_delta.cpu().numpy())
            final_effects[axis].append(final_delta.cpu().numpy())
            records.append({
                "row": row_index,
                "axis": axis,
                "local_effect_norm": float(local_delta.norm()),
                "final_effect_norm": float(final_delta.norm()),
                "next_token_kl": kl,
                "correct_token_logit_change": target_change,
            })
        if progress % 50 == 0:
            print("propagate", progress + 1, "/", len(requests_by_row), flush=True)

    axis_rows = []
    for axis in selected_axes:
        local = directional_summary(np.asarray(local_effects[axis]))
        final = directional_summary(np.asarray(final_effects[axis]))
        atlas_row = atlas["axes"]["layer0"][axis]
        axis_rows.append({
            "axis": axis,
            "group": next(group for group, members in config["axis_groups"].items() if axis in members),
            "activation_concept": atlas_row["best_activation_concept"],
            "activation_auc": atlas_row["best_activation_auc"],
            "local": local,
            "final": final,
        })

    group_summary = {}
    for group in config["axis_groups"]:
        group_rows = [row for row in axis_rows if row["group"] == group]
        group_summary[group] = {
            stage: {
                key: float(np.median([row[stage][key] for row in group_rows]))
                for key in ("first_component_energy", "four_component_energy", "rank90")
            }
            for stage in ("local", "final")
        }
    aggregate = {
        stage: {
            key: float(np.median([row[stage][key] for row in axis_rows]))
            for key in ("first_component_energy", "four_component_energy", "rank90")
        }
        for stage in ("local", "final")
    }
    kls = np.asarray([row["next_token_kl"] for row in records])
    target_changes = np.abs([row["correct_token_logit_change"] for row in records])
    rank_ratio = aggregate["final"]["rank90"] / aggregate["local"]["rank90"]
    hypotheses = {
        "P1_downstream_rank_collapse": {
            "supported": rank_ratio <= 0.75,
            "local": aggregate["local"]["rank90"],
            "final": aggregate["final"]["rank90"],
            "ratio": rank_ratio,
        },
        "P2_low_dimensional_final_roles": {
            "supported": aggregate["final"]["four_component_energy"] >= 0.75,
            "value": aggregate["final"]["four_component_energy"],
        },
        "P3_depth_concentrates_each_group": {
            "supported": all(
                row["final"]["first_component_energy"] > row["local"]["first_component_energy"]
                for row in group_summary.values()
            ),
            "groups": {
                group: {
                    "local": row["local"]["first_component_energy"],
                    "final": row["final"]["first_component_energy"],
                }
                for group, row in group_summary.items()
            },
        },
        "P4_effects_reach_predictions": {
            "supported": float(np.median(kls)) >= 1e-6 and float(np.median(target_changes)) >= 0.01,
            "median_kl": float(np.median(kls)),
            "median_absolute_correct_logit_change": float(np.median(target_changes)),
        },
        "P5_not_every_axis_is_rank1": {
            "supported": sum(row["final"]["rank90"] >= 4 for row in axis_rows) >= 3,
            "axes_rank_at_least_4": sum(row["final"]["rank90"] >= 4 for row in axis_rows),
        },
    }
    output = {
        "config": config,
        "aggregate": aggregate,
        "group_summary": group_summary,
        "hypotheses": hypotheses,
        "axes": axis_rows,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Do layer-0 contextual roles survive downstream?",
        "",
        "Twelve predeclared layer-0 joint axes are zeroed at 32 high-local-effect final-token sites each. The changed layer-0 MLP write is propagated through blocks 1–17. Directional rank is compared for the immediate 1,152D write and the final normalized residual state.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Downstream roles](layer0_downstream_roles.png)",
        "",
        "| Group | Local first / four | Local rank90 | Final first / four | Final rank90 |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, row in group_summary.items():
        lines.append(
            f"| {group} | {row['local']['first_component_energy']:.3f} / {row['local']['four_component_energy']:.3f} | "
            f"{row['local']['rank90']:.1f} | {row['final']['first_component_energy']:.3f} / "
            f"{row['final']['four_component_energy']:.3f} | {row['final']['rank90']:.1f} |"
        )
    lines += [
        "",
        f"Across {len(records)} finite interventions, median next-token KL is {np.median(kls):.3g} and median absolute correct-token logit change is {np.median(target_changes):.3f}.",
        "",
        "This is a selected high-effect panel at the final sequence position. It tests downstream survival of local role complexity, not average axis use or effects transmitted from earlier source positions through attention.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = list(config["axis_groups"])
    width = 0.35
    positions = np.arange(len(labels))
    plots[0].bar(positions - width / 2, [group_summary[g]["local"]["first_component_energy"] for g in labels], width, label="local")
    plots[0].bar(positions + width / 2, [group_summary[g]["final"]["first_component_energy"] for g in labels], width, label="final")
    plots[0].set_xticks(positions, labels, rotation=20, ha="right")
    plots[0].set_ylabel("first role-direction energy")
    plots[0].legend()
    plots[1].bar(positions - width / 2, [group_summary[g]["local"]["rank90"] for g in labels], width, label="local")
    plots[1].bar(positions + width / 2, [group_summary[g]["final"]["rank90"] for g in labels], width, label="final")
    plots[1].set_xticks(positions, labels, rotation=20, ha="right")
    plots[1].set_ylabel("directions for 90% energy")
    plots[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
