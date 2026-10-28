#!/usr/bin/env python3
"""Predict each bilinear MLP operator from prior state plus its attention write."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_early_operator_context_router import (
    ridge_model,
    ridge_predict,
    select_ridge,
    state_from_coordinates,
    weighted_r2,
)
from run_bilin18_early_operator_tensor_vocabulary import centered_r2, operator_probe_vectors
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_prospective_operator_router.json")
OUTPUT = TENSOR_ROOT / "bilin18_prospective_operator_router.json"
REPORT = TENSOR_ROOT / "BILIN18_PROSPECTIVE_OPERATOR_ROUTER.md"
FIGURE = TENSOR_ROOT / "bilin18_prospective_operator_router.png"
PROJECTIONS = TENSOR_ROOT / "bilin18_prospective_operator_router_bases.pt"


@torch.inference_mode()
def collect_components(model, corpus, start, rows, config):
    """Collect same-position inputs without using the target MLP's output."""
    source = np.load(corpus, mmap_mode="r")
    panel = source[start:start + rows]
    positions = np.asarray(config["positions"])
    total = len(panel) * len(positions)
    dimension = model.config.n_embd
    names = ("prior_residual", "mixed", "attention_input", "attention_write", "target_state")
    values = {
        layer: {name: np.empty((total, dimension), dtype=np.float32) for name in names}
        for layer in config["layers"]
    }
    offset = 0
    for row_start in range(0, len(panel), config["batch_size"]):
        raw = np.asarray(
            panel[row_start:row_start + config["batch_size"], :config["sequence_length"]],
            dtype=np.int64,
        )
        ids = torch.as_tensor(raw, device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,))
        x, value_bus = x0, None
        for layer, block in enumerate(model.blocks[:max(config["layers"]) + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            attention_input = F.rms_norm(mixed, (dimension,))
            attention, value_bus = block.attn(attention_input, value_bus)
            pre = mixed + attention
            target = F.rms_norm(pre, (dimension,))
            if layer in values:
                selected = {
                    "prior_residual": F.rms_norm(x, (dimension,))[:, positions],
                    "mixed": mixed[:, positions],
                    "attention_input": attention_input[:, positions],
                    "attention_write": attention[:, positions],
                    "target_state": target[:, positions],
                }
                count = len(raw) * len(positions)
                for name, tensor in selected.items():
                    values[layer][name][offset:offset + count] = tensor.flatten(0, 1).float().cpu().numpy()
            product = block.mlp.Left(target) * block.mlp.Right(target)
            x = pre + block.mlp.Down(product) + block.mlp.Down_bias
        offset += len(raw) * len(positions)
    return values


def fit_pca(train, maximum_rank):
    mean = train.mean(0)
    centered = train - mean
    covariance = centered.T @ centered / len(centered)
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    order = values.argsort(descending=True)[:maximum_rank]
    return mean, vectors[:, order]


def pca_coordinates(train, test, rank):
    mean, vectors = fit_pca(train, rank)
    return (train - mean) @ vectors, (test - mean) @ vectors


def pca_projection(train, test, rank):
    mean, vectors = fit_pca(train, rank)
    return mean, vectors, (train - mean) @ vectors, (test - mean) @ vectors


def pca_reconstruct(mean, vectors, coordinates, rank):
    return mean + coordinates[:, :rank] @ vectors[:, :rank].T


def operator_coordinates(state, basis, rank):
    mean = basis["mean"].cuda()
    vectors = basis["state_vectors"].cuda()
    rms = basis["state_rms"].cuda()
    modes = basis["operator_modes_first"].cuda()[:, :rank]
    return ((state - mean) @ vectors / rms[None]) @ modes


def fit_router(train_x, train_y, test_x, singular, row_index, config):
    training = row_index < config["rows_per_split"] // 2
    validation = ~training
    alpha, scores = select_ridge(
        train_x[training], train_y[training], train_x[validation], train_y[validation],
        singular, config["ridge_alphas"],
    )
    model = ridge_model(train_x, train_y, alpha)
    return ridge_predict(model, test_x), alpha, scores


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    bases = torch.load(config["bases"], map_location="cpu", weights_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    print("collect prospective-router discovery", flush=True)
    discovery = collect_components(
        model, config["corpus"], config["discovery_start"], config["rows_per_split"], config,
    )
    print("collect prospective-router evaluation", flush=True)
    evaluation = collect_components(
        model, config["corpus"], config["evaluation_start"], config["rows_per_split"], config,
    )
    positions = len(config["positions"])
    row_index = torch.arange(config["rows_per_split"], device="cuda").repeat_interleave(positions)
    maximum = max(config["source_dimensions"])
    results = {}
    saved_projections = {}
    for layer in config["layers"]:
        print("fit prospective router layer", layer, flush=True)
        basis = bases[str(layer)]
        train_target_state = torch.from_numpy(discovery[layer]["target_state"]).cuda()
        test_target_state = torch.from_numpy(evaluation[layer]["target_state"]).cuda()
        train_target = operator_coordinates(train_target_state, basis, config["rank"])
        test_target = operator_coordinates(test_target_state, basis, config["rank"])
        singular = basis["operator_singular_first"].cuda()[:config["rank"]]
        projected = {}
        projections = {}
        for name in ("prior_residual", "mixed", "attention_input", "attention_write"):
            train = torch.from_numpy(discovery[layer][name]).cuda()
            test = torch.from_numpy(evaluation[layer][name]).cuda()
            projection = pca_projection(train, test, maximum)
            projections[name] = projection
            projected[name] = projection[2:]
        features = {}
        for dimension in config["source_dimensions"]:
            prior_train, prior_test = (value[:, :dimension] for value in projected["prior_residual"])
            input_train, input_test = (value[:, :dimension] for value in projected["attention_input"])
            write_train, write_test = (value[:, :dimension] for value in projected["attention_write"])
            features[f"prior_{dimension}"] = (prior_train, prior_test)
            features[f"attention_input_{dimension}"] = (input_train, input_test)
            features[f"attention_write_{dimension}"] = (write_train, write_test)
            features[f"prior_plus_attention_{dimension}"] = (
                torch.cat((prior_train, write_train), 1), torch.cat((prior_test, write_test), 1),
            )
        if layer > 0:
            previous_basis = bases[str(layer - 1)]
            previous_train = operator_coordinates(
                torch.from_numpy(discovery[layer - 1]["target_state"]).cuda(), previous_basis, config["rank"],
            )
            previous_test = operator_coordinates(
                torch.from_numpy(evaluation[layer - 1]["target_state"]).cuda(), previous_basis, config["rank"],
            )
            features["previous_operator_64"] = (previous_train, previous_test)
        predictions = {"mean": torch.zeros_like(test_target), "oracle_top64": test_target}
        fitting = {}
        for name, (train_x, test_x) in features.items():
            prediction, alpha, scores = fit_router(
                train_x, train_target, test_x, singular, row_index, config,
            )
            predictions[name] = prediction
            fitting[name] = {"alpha": alpha, "validation": scores}
        mixed_mean, mixed_vectors, _, mixed_test = projections["mixed"]
        write_mean, write_vectors, _, write_test = projections["attention_write"]
        saved_projections[str(layer)] = {
            "mixed_mean": mixed_mean.cpu(),
            "mixed_vectors": mixed_vectors.cpu(),
            "attention_mean": write_mean.cpu(),
            "attention_vectors": write_vectors.cpu(),
        }
        for dimension in config["source_dimensions"]:
            reconstructed_mixed = pca_reconstruct(mixed_mean, mixed_vectors, mixed_test, dimension)
            reconstructed_write = pca_reconstruct(write_mean, write_vectors, write_test, dimension)
            direct_state = F.rms_norm(reconstructed_mixed + reconstructed_write, (reconstructed_mixed.shape[1],))
            predictions[f"direct_composition_{dimension}"] = operator_coordinates(
                direct_state, basis, config["rank"],
            )
            # Flattening is row-major over positions, so shifting by one full row
            # preserves position while pairing with another sequence's attention.
            shuffled_write = torch.roll(reconstructed_write, positions, 0)
            shuffled_state = F.rms_norm(reconstructed_mixed + shuffled_write, (reconstructed_mixed.shape[1],))
            predictions[f"shuffled_attention_direct_{dimension}"] = operator_coordinates(
                shuffled_state, basis, config["rank"],
            )
        exact_mixed = torch.from_numpy(evaluation[layer]["mixed"]).cuda()
        exact_write = torch.from_numpy(evaluation[layer]["attention_write"]).cuda()
        exact_direct_state = F.rms_norm(exact_mixed + exact_write, (exact_mixed.shape[1],))
        predictions["direct_composition_full"] = operator_coordinates(
            exact_direct_state, basis, config["rank"],
        )
        probe, _, _ = operator_probe_vectors(
            model.blocks[layer], config["operator_eval_probes"], config["seed"] + layer,
        )
        exact_operator = test_target_state @ probe.T
        metrics = {}
        for name, prediction in predictions.items():
            reconstructed = state_from_coordinates(prediction, basis, config["rank"])
            operator_prediction = reconstructed @ probe.T
            metrics[name] = {
                "coordinate_weighted_r2": weighted_r2(test_target, prediction, singular),
                "total_operator_r2": centered_r2(exact_operator, operator_prediction),
            }
        oracle_gain = metrics["oracle_top64"]["total_operator_r2"] - metrics["mean"]["total_operator_r2"]
        for row in metrics.values():
            row["oracle_gain_fraction"] = (
                (row["total_operator_r2"] - metrics["mean"]["total_operator_r2"])
                / max(oracle_gain, 1e-30)
            )
        results[str(layer)] = {"metrics": metrics, "fitting": fitting}
    attention_gain = {
        str(layer): results[str(layer)]["metrics"]["prior_plus_attention_64"]["coordinate_weighted_r2"]
        - results[str(layer)]["metrics"]["prior_64"]["coordinate_weighted_r2"]
        for layer in config["layers"]
    }
    hypotheses = {
        "P0_previous_operator_is_markovian": {
            "supported": all(results[str(layer)]["metrics"]["previous_operator_64"]["coordinate_weighted_r2"] >= .50 for layer in config["layers"] if layer > 0),
        },
        "P1_prior_residual_is_a_compact_router": {
            "supported": all(results[str(layer)]["metrics"]["prior_64"]["coordinate_weighted_r2"] >= .60 for layer in config["layers"]),
        },
        "P2_attention_adds_contextual_routing": {
            "supported": all(attention_gain[str(layer)] >= .05 for layer in config["layers"] if layer > 0),
        },
        "P3_rank128_composed_router_recovers_half_oracle": {
            "supported": all(results[str(layer)]["metrics"]["prior_plus_attention_64"]["oracle_gain_fraction"] >= .50 for layer in config["layers"]),
        },
        "P4_rank512_composed_router_is_near_oracle": {
            "supported": all(results[str(layer)]["metrics"]["prior_plus_attention_256"]["oracle_gain_fraction"] >= .90 for layer in config["layers"]),
        },
        "P5_attention_routing_grows_with_depth": {
            "supported": bool(np.mean([attention_gain[str(layer)] for layer in (4, 5)]) - np.mean([attention_gain[str(layer)] for layer in (0, 1)]) >= .03),
        },
        "P6_direct_rank512_composition_recovers_most_oracle": {
            "supported": all(results[str(layer)]["metrics"]["direct_composition_256"]["oracle_gain_fraction"] >= .80 for layer in config["layers"]),
        },
        "P7_attention_pairing_is_specific": {
            "supported": all(
                results[str(layer)]["metrics"]["direct_composition_64"]["coordinate_weighted_r2"]
                - results[str(layer)]["metrics"]["shuffled_attention_direct_64"]["coordinate_weighted_r2"] >= .10
                for layer in config["layers"]
            ),
        },
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses, "attention_gain": attention_gain, "results": results,
    }, indent=2, allow_nan=False))
    torch.save(saved_projections, PROJECTIONS)
    lines = [
        "# Prospective cross-layer operator router", "",
        "Each layer's frozen rank-64 bilinear-operator coordinates are predicted without observing that layer's normalized pre-MLP state. Candidate routers use a compressed prior residual, the same-position attention input, the context-routed attention write, or prior-residual plus attention-write summaries. PCA and ridge fits use discovery FineWeb; evaluation is disjoint FineWeb and a fresh weight-derived operator sketch.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Prospective router](bilin18_prospective_operator_router.png)", "",
              "Each cell is weighted coordinate R2 / total operator R2; parenthesized values are fractions of the rank-64 oracle gain.", "",
              "| Layer | Prev operator64 | Prior64 | Attn input64 | Prior+attn64 | Direct64 | Shuffled direct64 | Direct256 | Oracle64 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        metrics = results[str(layer)]["metrics"]
        def cell(name):
            if name not in metrics: return "—"
            row = metrics[name]
            return f"{row['coordinate_weighted_r2']:.3f}/{row['total_operator_r2']:.3f} ({row['oracle_gain_fraction']:.2f})"
        lines.append(
            f"| {layer} | {cell('previous_operator_64')} | {cell('prior_64')} | "
            f"{cell('attention_input_64')} | {cell('prior_plus_attention_64')} | "
            f"{cell('direct_composition_64')} | {cell('shuffled_attention_direct_64')} | "
            f"{cell('direct_composition_256')} | {cell('oracle_top64')} |"
        )
    lines += ["", "Attention increment in weighted coordinate R2: `" + json.dumps(attention_gain, sort_keys=True) + "`.", "",
              "## Guardrails", "",
              "The target-fitted ridge routers and the direct algebraic composition answer different questions. Direct64/256 has no fitted map to the target: it reconstructs mixed residual and attention write in discovery-PCA bases, adds them, and applies native RMSNorm. The shuffled control preserves position and both marginal distributions but breaks the sequence-specific pairing. Attention still contains a context aggregation computed by the native attention mechanism, so success localizes the routing interface rather than explaining attention itself. The rank-64 oracle remains a truncation ceiling rather than the full MLP operator."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    layers = config["layers"]
    methods = ("prior_64", "prior_plus_attention_64", "direct_composition_64", "shuffled_attention_direct_64", "direct_composition_256", "oracle_top64")
    for name in methods:
        axes[0].plot(layers, [results[str(layer)]["metrics"][name]["coordinate_weighted_r2"] for layer in layers], marker="o", label=name)
        axes[1].plot(layers, [results[str(layer)]["metrics"][name]["total_operator_r2"] for layer in layers], marker="o", label=name)
    axes[0].set(xlabel="Layer", ylabel="Weighted coordinate R2", title="Predicting the next local operator")
    axes[1].set(xlabel="Layer", ylabel="Independent-sketch total R2", title="Weight-derived operator fidelity")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=8, ncol=2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "attention_gain": attention_gain}, indent=2))
    print(REPORT)


if __name__ == "__main__": main()
