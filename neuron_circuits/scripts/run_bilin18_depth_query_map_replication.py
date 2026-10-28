#!/usr/bin/env python3
"""Standardized query-map and tangent replication at layers 0, 8, and 17."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_query_conditioned_interpreter import score_metrics
from run_bilin18_query_map_tangent_interpreter import metrics
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_depth_query_map_replication.json")
OUTPUT = TENSOR_ROOT / "depth_query_map_replication.json"
REPORT = TENSOR_ROOT / "DEPTH_QUERY_MAP_REPLICATION.md"
FIGURE = TENSOR_ROOT / "depth_query_map_replication.png"


@torch.inference_mode()
def target_input(model, ids: torch.Tensor, target_layer: int):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x = x0
    value_bus = None
    for block in model.blocks[:target_layer]:
        x, value_bus = block(x, value_bus, x0)
    block = model.blocks[target_layer]
    mixed = block.lambdas[0] * x + block.lambdas[1] * x0
    attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    pre = mixed + attention
    return F.rms_norm(pre, (dimension,))


@torch.inference_mode()
def collect(model, layer: int, corpus_config: dict, config: dict):
    source = np.load(corpus_config["path"], mmap_mode="r")
    rows = source[corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]]
    total = len(rows) * len(config["positions"])
    states = np.empty((total, 1152), dtype=np.float32)
    writes = np.empty((total, 1152), dtype=np.float32)
    offset = 0
    block = model.blocks[layer]
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x = target_input(model, ids, layer)[:, config["positions"]].float().flatten(0, 1)
        write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x))
        count = len(x)
        states[offset : offset + count] = x.cpu().numpy()
        writes[offset : offset + count] = write.cpu().numpy()
        offset += count
    return torch.from_numpy(states).cuda(), torch.from_numpy(writes).cuda()


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    results = {}
    for layer in config["layers"]:
        block = model.blocks[layer]
        discovery_x, _ = collect(model, layer, config["discovery"], config)
        _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
        state_v = state_vh.T
        basis64 = state_v[:, : config["basis_dimension"]]
        basis128 = state_v[:, : config["context_dimension"]]
        left_basis = block.mlp.Left.weight.float() @ basis64
        right_basis = block.mlp.Right.weight.float() @ basis64
        tensor = torch.einsum("dh,hi,hj->dij", block.mlp.Down.weight.float(), left_basis, right_basis)
        tensor = 0.5 * (tensor + tensor.transpose(1, 2))
        tensor_flat = tensor.flatten(1)
        output_u, singular, matrix_vh = torch.linalg.svd(tensor_flat, full_matrices=False)
        spectrum_energy = singular.square()
        top4_map_energy = float(spectrum_energy[:4].sum() / spectrum_energy.sum())

        x, write = collect(model, layer, config["evaluation"], config)
        z = x @ basis64
        phi = torch.einsum("ni,nj->nij", z, z).flatten(1)
        projected_write = phi @ tensor_flat.T
        centered_write = write - write.mean(0, keepdim=True)
        centered_projected = projected_write - projected_write.mean(0, keepdim=True)
        whole_write_r2 = float(
            1 - (centered_projected - centered_write).square().sum()
            / centered_write.square().sum().clamp_min(1e-30)
        )

        component_target = write @ output_u[:, : config["top_components"]]
        component_prediction = (
            phi @ matrix_vh[: config["top_components"]].T
        ) * singular[: config["top_components"]]
        correlations, r2 = score_metrics(
            component_target.cpu().numpy(), component_prediction.cpu().numpy()
        )
        form_energy = []
        tangent64 = []
        tangent128 = []
        component_rows = []
        for component in range(config["top_components"]):
            restricted = singular[component] * matrix_vh[component].reshape(64, 64)
            values, vectors = torch.linalg.eigh(restricted)
            order = torch.argsort(values.abs(), descending=True)
            values, vectors = values[order], vectors[:, order]
            energy = values.square()
            rank_energy = float(energy[: config["eigenspace_rank"]].sum() / energy.sum())
            form_energy.append(rank_energy)
            input_vectors = basis64 @ vectors[:, : config["eigenspace_rank"]]
            scores = component_target[:, component].cpu().numpy()
            centered = scores - np.median(scores)
            indices = np.argpartition(np.abs(centered), -config["sites_per_component"])[-config["sites_per_component"] :]
            selected_x = x[torch.as_tensor(indices, device="cuda")]
            delta = -(selected_x @ input_vectors) @ input_vectors.T
            delta *= (
                config["edit_norm_fraction"] * selected_x.norm(dim=1, keepdim=True)
                / delta.norm(dim=1, keepdim=True).clamp_min(1e-20)
            )
            output_direction = output_u[:, component]
            coefficients = block.mlp.Down.weight.float().T @ output_direction
            full_matrix = block.mlp.Left.weight.float().T @ (
                coefficients[:, None] * block.mlp.Right.weight.float()
            )
            full_matrix = 0.5 * (full_matrix + full_matrix.T)
            actual = 2 * torch.einsum("ni,ij,nj->n", delta, full_matrix, selected_x) + torch.einsum("ni,ij,nj->n", delta, full_matrix, delta)
            predictions = []
            for context_basis in (basis64, basis128):
                approximate_x = (selected_x @ context_basis) @ context_basis.T
                predictions.append(
                    2 * torch.einsum("ni,ij,nj->n", delta, full_matrix, approximate_x)
                    + torch.einsum("ni,ij,nj->n", delta, full_matrix, delta)
                )
            metric64 = metrics(actual.cpu().numpy(), predictions[0].cpu().numpy())
            metric128 = metrics(actual.cpu().numpy(), predictions[1].cpu().numpy())
            tangent64.append(metric64["correlation"])
            tangent128.append(metric128["correlation"])
            component_rows.append({
                "component": component,
                "singular_value": float(singular[component]),
                "score_correlation": float(correlations[component]),
                "score_centered_r2": float(r2[component]),
                "top4_form_energy": rank_energy,
                "tangent64": metric64,
                "tangent128": metric128,
            })

        results[str(layer)] = {
            "whole_write_centered_r2": whole_write_r2,
            "top4_map_energy": top4_map_energy,
            "top4_score_correlation": float(np.median(correlations)),
            "top4_form_energy": float(np.median(form_energy)),
            "tangent64_correlation": float(np.median(tangent64)),
            "tangent128_correlation": float(np.median(tangent128)),
            "components": component_rows,
        }
        print("layer", layer, "complete", flush=True)

    hypotheses = {
        "M1_state64_write_fidelity": {
            "supported": all(row["whole_write_centered_r2"] >= 0.85 for row in results.values()),
            "per_layer": {layer: row["whole_write_centered_r2"] for layer, row in results.items()},
        },
        "M2_top_component_score_fidelity": {
            "supported": all(row["top4_score_correlation"] >= 0.85 for row in results.values()),
            "per_layer": {layer: row["top4_score_correlation"] for layer, row in results.items()},
        },
        "M3_low_rank_component_forms": {
            "supported": all(row["top4_form_energy"] >= 0.70 for row in results.values()),
            "per_layer": {layer: row["top4_form_energy"] for layer, row in results.items()},
        },
        "M4_context128_tangent_fidelity": {
            "supported": all(row["tangent128_correlation"] >= 0.95 for row in results.values()),
            "per_layer": {layer: row["tangent128_correlation"] for layer, row in results.items()},
        },
        "M5_late_spectral_concentration": {
            "supported": results["17"]["top4_map_energy"] - results["0"]["top4_map_energy"] >= 0.10,
            "layer0": results["0"]["top4_map_energy"],
            "layer17": results["17"]["top4_map_energy"],
            "gap": results["17"]["top4_map_energy"] - results["0"]["top4_map_energy"],
        },
    }
    output = {"config": config, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Standardized depth replication of query-map structure",
        "",
        "Layers 0, 8, and 17 are analyzed with the same state-SVD-64 input basis, output×64×64 query map, leading-component score test, four-mode input-form audit, and state-SVD-128 tangent context. This avoids changing the explanatory object between depths.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Depth replication](depth_query_map_replication.png)",
        "",
        "| Layer | Write R² | Top-four map energy | Top-four score corr. | Four-mode form energy | Tangent 64 / 128 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, row in results.items():
        lines.append(
            f"| {layer} | {row['whole_write_centered_r2']:.3f} | {row['top4_map_energy']:.3f} | "
            f"{row['top4_score_correlation']:.3f} | {row['top4_form_energy']:.3f} | "
            f"{row['tangent64_correlation']:.3f} / {row['tangent128_correlation']:.3f} |"
        )
    lines += [
        "",
        "This is a layer-local replication. It does not test how middle-layer writes propagate to final logits or whether output singular vectors have the same human semantics across depth.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(10, 4.2))
    layers = [str(layer) for layer in config["layers"]]
    plots[0].plot(layers, [results[layer]["whole_write_centered_r2"] for layer in layers], marker="o", label="write R²")
    plots[0].plot(layers, [results[layer]["top4_score_correlation"] for layer in layers], marker="o", label="top score corr")
    plots[0].set_ylim(0, 1.02)
    plots[0].set_xlabel("layer")
    plots[0].legend()
    plots[1].plot(layers, [results[layer]["top4_map_energy"] for layer in layers], marker="o", label="map top4 energy")
    plots[1].plot(layers, [results[layer]["top4_form_energy"] for layer in layers], marker="o", label="form top4 energy")
    plots[1].set_ylim(0, 1.02)
    plots[1].set_xlabel("layer")
    plots[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
