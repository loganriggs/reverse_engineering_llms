#!/usr/bin/env python3
"""Test fixed and context-selected native CP channels for Layer-8 addresses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_conditional_subspace_quantization import grassmann_similarity, state_centroids
from run_bilin18_layer8_restricted_tensor_program import prediction_r2
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_native_channel_program.json")
SOURCE = TENSOR_ROOT / "layer8_multimetric_quantization.npz"
BASE_RESULT = TENSOR_ROOT / "layer8_restricted_tensor_program.json"
OUTPUT = TENSOR_ROOT / "layer8_native_channel_program.json"
REPORT = TENSOR_ROOT / "LAYER8_NATIVE_CHANNEL_PROGRAM.md"
FIGURE = TENSOR_ROOT / "layer8_native_channel_program.png"


def coefficients(left, right, edit, states):
    left_edit = (left @ edit).T
    right_edit = (right @ edit).T
    left_state = states @ left.T
    right_state = states @ right.T
    return left_edit[None] * right_state[:, None] + right_edit[None] * left_state[:, None]


def channel_target(coefficients, down):
    return torch.einsum("nah,oh->nao", coefficients, down)


def ordered_curve(coefficients, down, target, order, ranks):
    prediction = torch.zeros_like(target)
    rows = []
    previous = 0
    for rank in ranks:
        selected = order[previous:rank]
        if len(selected):
            prediction += torch.einsum("nah,oh->nao", coefficients[:, :, selected], down[:, selected])
        rows.append({"rank": rank, "r2": prediction_r2(target, prediction)})
        previous = rank
    return rows


def oracle_prediction(coefficients, down, order, rank, batch_size=8):
    outputs = []
    for start in range(0, len(coefficients), batch_size):
        stop = min(start + batch_size, len(coefficients))
        indices = order[start:stop, :rank]
        gathered_coefficients = torch.gather(
            coefficients[start:stop], 2,
            indices[:, None].expand(-1, coefficients.shape[1], -1),
        )
        gathered_down = down.T[indices]
        outputs.append(torch.einsum("bak,bko->bao", gathered_coefficients, gathered_down))
    return torch.cat(outputs)


def first_rank(rows, threshold=.9):
    for row in rows:
        if row["r2"] >= threshold:
            return row["rank"]
    return None


def effective_support(energy, dimension=-1):
    return energy.sum(dimension).square() / energy.square().sum(dimension).clamp_min(1e-30)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE)
    base = json.loads(BASE_RESULT.read_text())
    active = [int(value) for value in base["active_prototypes"]]
    basis = torch.from_numpy(source["basis"]).cuda().float()
    axes = torch.from_numpy(source["axes_direct_mlp"]).cuda().float()
    prototypes = torch.from_numpy(source["prototypes_direct_mlp"]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    discovery_states, _ = collect(model, 8, config["prototype_discovery"], {**config, "positions": [config["causal_position"]]})
    discovery_features = discovery_states @ basis
    discovery_features /= discovery_features.norm(dim=1, keepdim=True).clamp_min(1e-30)
    discovery_labels = grassmann_similarity(axes, prototypes).argmax(1)
    centroids = state_centroids(discovery_features, discovery_labels, config["clusters"])
    states, _ = collect(model, 8, config["natural_states"], {**config, "positions": [config["causal_position"]]})
    features = states @ basis
    features /= features.norm(dim=1, keepdim=True).clamp_min(1e-30)
    labels = (features @ centroids.T).argmax(1)
    row_ids = torch.arange(len(labels), device="cuda")
    train_mask = row_ids % 2 == 0
    test_mask = ~train_mask
    block = model.blocks[8]
    down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
    down_energy = down.square().sum(0)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    results = {}
    for cluster in active:
        train_x = states[(labels == cluster) & train_mask]
        test_x = states[(labels == cluster) & test_mask]
        print("native channels", cluster, len(train_x), len(test_x), flush=True)
        edit = basis @ prototypes[cluster]
        train_coefficients = coefficients(left, right, edit, train_x)
        test_coefficients = coefficients(left, right, edit, test_x)
        train_target = channel_target(train_coefficients, down)
        test_target = channel_target(test_coefficients, down)
        train_channel_energy = train_coefficients.square().sum((0, 1)) * down_energy
        static_order = train_channel_energy.argsort(descending=True)
        static_rows = ordered_curve(test_coefficients, down, test_target, static_order, config["static_ranks"])
        site_channel_energy = test_coefficients.square().sum(1) * down_energy[None]
        oracle_order = site_channel_energy.argsort(1, descending=True)
        oracle_rows = []
        for rank in config["oracle_ranks"]:
            prediction = oracle_prediction(test_coefficients, down, oracle_order, rank, config["batch_size"])
            oracle_rows.append({"rank": rank, "r2": prediction_r2(test_target, prediction)})
        random_512 = []
        for _ in range(config["random_replicates"]):
            order = torch.randperm(down.shape[1], generator=generator, device="cuda")
            prediction = channel_target(test_coefficients[:, :, order[:512]], down[:, order[:512]])
            random_512.append(prediction_r2(test_target, prediction))
        diagonal_energy = float((test_coefficients.square().sum((0, 1)) * down_energy).sum())
        total_energy = float(test_target.square().sum())
        support = effective_support(site_channel_energy)
        results[str(cluster)] = {
            "train_sites": int(len(train_x)), "test_sites": int(len(test_x)),
            "static_rows": static_rows, "static_rank90": first_rank(static_rows),
            "oracle_rows": oracle_rows, "oracle_rank90": first_rank(oracle_rows),
            "random_512": random_512,
            "interference_ratio": diagonal_energy / max(total_energy, 1e-30),
            "normalized_cross_term": (total_energy - diagonal_energy) / max(total_energy, 1e-30),
            "median_site_effective_support": float(support.median()),
            "site_effective_support_quantiles": [float(value) for value in torch.quantile(support, torch.tensor([.1, .5, .9], device="cuda"))],
        }
    full = [row["static_rows"][-1]["r2"] for row in results.values()]
    static512 = [next(item for item in row["static_rows"] if item["rank"] == 512)["r2"] for row in results.values()]
    oracle256 = [next(item for item in row["oracle_rows"] if item["rank"] == 256)["r2"] for row in results.values()]
    static256 = [next(item for item in row["static_rows"] if item["rank"] == 256)["r2"] for row in results.values()]
    supports = [row["median_site_effective_support"] for row in results.values()]
    interference = [row["interference_ratio"] for row in results.values()]
    hypotheses = {
        "C1_native_channel_sum_is_exact": {"supported": min(full) >= .999, "minimum_r2": min(full)},
        "C2_static_512_is_sufficient": {"supported": sum(value >= .9 for value in static512) >= len(static512) / 2, "passing": sum(value >= .9 for value in static512)},
        "C3_oracle_256_is_sufficient": {"supported": sum(value >= .9 for value in oracle256) >= len(oracle256) / 2, "passing": sum(value >= .9 for value in oracle256)},
        "C4_context_selection_matters": {"supported": float(np.median(np.asarray(oracle256) - np.asarray(static256))) >= .1, "median_gap": float(np.median(np.asarray(oracle256) - np.asarray(static256)))},
        "C5_native_ranking_beats_random": {"supported": all(static512[index] > np.quantile(row["random_512"], .95) for index, row in enumerate(results.values()))},
        "C6_channel_program_is_sparse": {"supported": float(np.median(supports)) <= 512, "median_support": float(np.median(supports))},
        "C7_channels_interfere": {"supported": not .8 <= float(np.median(interference)) <= 1.2, "median_interference_ratio": float(np.median(interference))},
    }
    OUTPUT.write_text(json.dumps({"config": config, "active_prototypes": active, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Native-channel programs for routed Layer-8 addresses", "",
        "Each fixed four-dimensional address is expanded in the exact native bilinear CP terms. A fixed channel ordering is learned only from training contexts; a per-example oracle selects channels by their individual derivative-write energy. The latter diagnoses conditional sparsity and is not a deployable router. Summed individual energy divided by total write energy measures cooperation (<1) or cancellation (>1).", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Native channel programs](layer8_native_channel_program.png)", "", "| Address | Static K90 | Static R² @256 / 512 | Oracle R² @256 | Effective support | Individual/total energy | Random-512 p95 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for cluster, row in results.items():
        static = {item["rank"]: item["r2"] for item in row["static_rows"]}
        oracle = {item["rank"]: item["r2"] for item in row["oracle_rows"]}
        lines.append(f"| {cluster} | {row['static_rank90']} | {static[256]:.3f} / {static[512]:.3f} | {oracle[256]:.3f} | {row['median_site_effective_support']:.1f} | {row['interference_ratio']:.3f} | {np.quantile(row['random_512'], .95):.3f} |")
    lines += ["", "A compact static program would support native neuron-aligned modules. A compact oracle but broad static program would instead support context-dependent computation in superposition. If both remain broad, the native CP basis is not a short description of the complete derivative write."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.2))
    for cluster, row in results.items():
        plots[0].plot([item["rank"] for item in row["static_rows"]], [item["r2"] for item in row["static_rows"]], marker="o", label=cluster)
        plots[1].plot([item["rank"] for item in row["oracle_rows"]], [item["r2"] for item in row["oracle_rows"]], marker="o", label=cluster)
    plots[2].scatter(supports, interference)
    plots[0].set_title("Fixed native channels"); plots[1].set_title("Per-example oracle channels"); plots[2].set_title("Support vs interference")
    for plot in plots[:2]:
        plot.set_xscale("log", base=2); plot.set_ylim(-.1, 1.05); plot.grid(alpha=.2); plot.set_xlabel("channels"); plot.set_ylabel("held-out derivative R²")
    plots[2].set_xlabel("median effective support"); plots[2].set_ylabel("individual / total energy"); plots[2].grid(alpha=.2)
    plots[0].legend(fontsize=7, ncol=2)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
