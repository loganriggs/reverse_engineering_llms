#!/usr/bin/env python3
"""Predict early operator-state coordinates from lexical and contextual routers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_early_operator_tensor_vocabulary import (
    centered_r2,
    collect_states,
    operator_probe_vectors,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_early_operator_context_router.json")
OUTPUT = TENSOR_ROOT / "bilin18_early_operator_context_router.json"
REPORT = TENSOR_ROOT / "BILIN18_EARLY_OPERATOR_CONTEXT_ROUTER.md"
FIGURE = TENSOR_ROOT / "bilin18_early_operator_context_router.png"


def token_metadata(corpus, start, rows, positions, sequence_length):
    source = np.load(corpus, mmap_mode="r")
    panel = np.asarray(source[start:start + rows, :sequence_length], dtype=np.int64)
    current = panel[:, positions].reshape(-1)
    previous = panel[:, np.asarray(positions) - 1].reshape(-1)
    windows = np.stack([panel[:, np.asarray(positions) - offset] for offset in range(1, 9)], axis=2)
    previous_eight = windows.reshape(-1, 8)
    position = np.tile(np.arange(len(positions), dtype=np.int64), rows)
    row = np.repeat(np.arange(rows, dtype=np.int64), len(positions))
    return {"current": current, "previous": previous, "previous_eight": previous_eight, "position": position, "row": row}


def weighted_r2(target, prediction, weight):
    difference = (target - prediction) * weight[None]
    target = target * weight[None]
    return float(1 - difference.square().sum() / target.square().sum().clamp_min(1e-30))


def ridge_model(train_x, train_y, alpha):
    x_mean = train_x.mean(0); x_std = train_x.std(0).clamp_min(1e-4); y_mean = train_y.mean(0)
    x = (train_x - x_mean) / x_std; y = train_y - y_mean
    gram = x.T @ x; scale = float(gram.trace() / gram.shape[0])
    weight = torch.linalg.solve(gram + alpha * scale * torch.eye(gram.shape[0], device=x.device), x.T @ y)
    return x_mean, x_std, y_mean, weight


def ridge_predict(model, x):
    x_mean, x_std, y_mean, weight = model
    return ((x - x_mean) / x_std) @ weight + y_mean


def select_ridge(train_x, train_y, validation_x, validation_y, metric_weight, alphas):
    rows = []
    for alpha in alphas:
        model = ridge_model(train_x, train_y, alpha)
        rows.append((weighted_r2(validation_y, ridge_predict(model, validation_x), metric_weight), alpha))
    return max(rows)[1], rows


def fit_lookup(keys, target, size, prior):
    counts = torch.bincount(keys, minlength=size).float()
    sums = torch.zeros(size, target.shape[1], device=target.device)
    sums.index_add_(0, keys, target)
    mean = target.mean(0)
    table = (sums + prior * mean[None]) / (counts[:, None] + prior)
    return table, counts, mean


def lookup_predict(keys, table, counts, mean):
    prediction = table[keys].clone()
    unseen = counts[keys] == 0
    prediction[unseen] = mean
    return prediction, float((~unseen).float().mean())


def state_from_coordinates(coordinates, basis, rank):
    mean = basis["mean"].cuda(); vectors = basis["state_vectors"].cuda(); rms = basis["state_rms"].cuda(); modes = basis["operator_modes_first"].cuda()[:, :rank]
    return mean + ((coordinates @ modes.T) * rms[None]) @ vectors.T


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    bases = torch.load(config["bases"], map_location="cpu", weights_only=True)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    dummy_flags = np.zeros((model.config.vocab_size, 1), dtype=np.uint8)
    print("collect router discovery", flush=True)
    discovery, _, _ = collect_states(model, config["corpus"], config["discovery_start"], config["rows_per_split"], config, dummy_flags)
    print("collect router evaluation", flush=True)
    evaluation, _, _ = collect_states(model, config["corpus"], config["evaluation_start"], config["rows_per_split"], config, dummy_flags)
    train_meta = token_metadata(config["corpus"], config["discovery_start"], config["rows_per_split"], config["positions"], config["sequence_length"])
    test_meta = token_metadata(config["corpus"], config["evaluation_start"], config["rows_per_split"], config["positions"], config["sequence_length"])
    embedding = F.rms_norm(model.embedding.weight.float(), (model.config.n_embd,))
    results = {}
    for layer in config["layers"]:
        print("operator router layer", layer, flush=True)
        basis = bases[str(layer)]; mean = basis["mean"].cuda(); vectors = basis["state_vectors"].cuda(); rms = basis["state_rms"].cuda(); modes = basis["operator_modes_first"].cuda()[:, :config["rank"]]; singular = basis["operator_singular_first"].cuda()[:config["rank"]]
        train_state = torch.from_numpy(discovery[layer]).cuda(); test_state = torch.from_numpy(evaluation[layer]).cuda()
        train_white = ((train_state - mean) @ vectors) / rms[None]; test_white = ((test_state - mean) @ vectors) / rms[None]
        train_target = train_white @ modes; test_target = test_white @ modes
        state_projection = vectors[:, :config["embedding_dimension"]]
        train_current = embedding[torch.as_tensor(train_meta["current"], device="cuda")] @ state_projection
        test_current = embedding[torch.as_tensor(test_meta["current"], device="cuda")] @ state_projection
        train_previous = embedding[torch.as_tensor(train_meta["previous"], device="cuda")] @ state_projection
        test_previous = embedding[torch.as_tensor(test_meta["previous"], device="cuda")] @ state_projection
        train_mean8 = embedding[torch.as_tensor(train_meta["previous_eight"], device="cuda")].mean(1) @ state_projection
        test_mean8 = embedding[torch.as_tensor(test_meta["previous_eight"], device="cuda")].mean(1) @ state_projection
        train_context = torch.cat((train_current, train_previous, train_mean8), 1)
        test_context = torch.cat((test_current, test_previous, test_mean8), 1)
        train_position = F.one_hot(torch.as_tensor(train_meta["position"], device="cuda"), len(config["positions"])).float()
        test_position = F.one_hot(torch.as_tensor(test_meta["position"], device="cuda"), len(config["positions"])).float()
        train_features = {
            "embedding": train_current,
            "short_context": train_context,
            "embedding_position": torch.cat((train_current, train_position), 1),
            **{f"state_pca_{dimension}": train_white[:, :dimension] for dimension in config["state_dimensions"]},
        }
        test_features = {
            "embedding": test_current,
            "short_context": test_context,
            "embedding_position": torch.cat((test_current, test_position), 1),
            **{f"state_pca_{dimension}": test_white[:, :dimension] for dimension in config["state_dimensions"]},
        }
        train_rows = torch.as_tensor(train_meta["row"], device="cuda") < config["rows_per_split"] // 2
        validation_rows = ~train_rows
        predictions, selected_alpha = {}, {}
        for name in train_features:
            alpha, _ = select_ridge(train_features[name][train_rows], train_target[train_rows], train_features[name][validation_rows], train_target[validation_rows], singular, config["ridge_alphas"])
            fitted = ridge_model(train_features[name], train_target, alpha)
            predictions[name] = ridge_predict(fitted, test_features[name])
            selected_alpha[name] = alpha
        current_keys = torch.as_tensor(train_meta["current"], device="cuda")
        test_current_keys = torch.as_tensor(test_meta["current"], device="cuda")
        token_table, token_counts, token_mean = fit_lookup(current_keys, train_target, model.config.vocab_size, config["lookup_prior"])
        predictions["token"], token_coverage = lookup_predict(test_current_keys, token_table, token_counts, token_mean)
        position_keys = torch.as_tensor(train_meta["position"], device="cuda")
        test_position_keys = torch.as_tensor(test_meta["position"], device="cuda")
        position_table, position_counts, position_mean = fit_lookup(position_keys, train_target, len(config["positions"]), config["lookup_prior"])
        predictions["position"], _ = lookup_predict(test_position_keys, position_table, position_counts, position_mean)
        predictions["mean"] = torch.zeros_like(test_target)
        predictions["oracle_top64"] = test_target
        probe, _, _ = operator_probe_vectors(model.blocks[layer], config["operator_eval_probes"], config["seed"] + layer)
        exact_operator = test_state @ probe.T
        metrics = {}
        for name, prediction in predictions.items():
            reconstructed = state_from_coordinates(prediction, basis, config["rank"])
            operator_prediction = reconstructed @ probe.T
            metrics[name] = {
                "coordinate_weighted_r2": weighted_r2(test_target, prediction, singular),
                "total_operator_r2": centered_r2(exact_operator, operator_prediction),
            }
        oracle_gain = metrics["oracle_top64"]["total_operator_r2"] - metrics["mean"]["total_operator_r2"]
        for name in metrics:
            metrics[name]["oracle_gain_fraction"] = (metrics[name]["total_operator_r2"] - metrics["mean"]["total_operator_r2"]) / max(oracle_gain, 1e-30)
        results[str(layer)] = {"metrics": metrics, "selected_alpha": selected_alpha, "token_coverage": token_coverage}
    context_gain = {layer: results[str(layer)]["metrics"]["short_context"]["coordinate_weighted_r2"] - results[str(layer)]["metrics"]["embedding"]["coordinate_weighted_r2"] for layer in config["layers"]}
    hypotheses = {
        "R0_current_token_routes_operator_state": {"supported": all(results[str(layer)]["metrics"]["token"]["coordinate_weighted_r2"] >= .20 for layer in config["layers"])},
        "R1_embedding_geometry_generalizes_router": {"supported": all(results[str(layer)]["metrics"]["embedding"]["coordinate_weighted_r2"] >= results[str(layer)]["metrics"]["token"]["coordinate_weighted_r2"] - .05 for layer in config["layers"])},
        "R2_short_context_improves_routing": {"supported": all(context_gain[layer] >= .05 for layer in (4, 5))},
        "R3_contextual_routing_strengthens_with_depth": {"supported": bool(np.mean([context_gain[layer] for layer in (4, 5)]) - np.mean([context_gain[layer] for layer in (0, 1)]) >= .03)},
        "R4_state64_is_a_compact_router": {"supported": all(results[str(layer)]["metrics"]["state_pca_64"]["coordinate_weighted_r2"] >= .70 for layer in config["layers"])},
        "R5_short_context_recovers_half_oracle_operator_gain": {"supported": all(results[str(layer)]["metrics"]["short_context"]["oracle_gain_fraction"] >= .50 for layer in config["layers"])},
        "R6_position_is_not_the_primary_router": {"supported": all(results[str(layer)]["metrics"]["position"]["coordinate_weighted_r2"] <= results[str(layer)]["metrics"]["short_context"]["coordinate_weighted_r2"] - .05 for layer in config["layers"])},
    }
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"context_gain":context_gain,"results":results},indent=2,allow_nan=False))
    lines=["# Early operator-state context router","","The converged top-64 operator subspace is frozen at each layer. Routers predict its whitened state coefficients from exact current-token lookup, position, a 128D current embedding projection, the current plus previous-token and previous-eight-token embedding summary, or 64/256 natural-state coordinates. Ridge penalties are selected on held-out discovery rows. Evaluation uses disjoint FineWeb and an independent weight-derived operator sketch.","","## Locked hypotheses",""]
    for name,row in hypotheses.items():lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["","![Context router](bilin18_early_operator_context_router.png)","","Each cell is weighted coordinate R2 / total operator R2. Total R2 includes the rank-64 truncation ceiling.","","| Layer | Mean | Position | Token (coverage) | Embedding | Short context | State64 | State256 | Oracle64 |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row=results[str(layer)];m=row["metrics"];fmt=lambda name:f"{m[name]['coordinate_weighted_r2']:.3f}/{m[name]['total_operator_r2']:.3f}"
        lines.append(f"| {layer} | {fmt('mean')} | {fmt('position')} | {fmt('token')} ({row['token_coverage']:.2f}) | {fmt('embedding')} | {fmt('short_context')} | {fmt('state_pca_64')} | {fmt('state_pca_256')} | {fmt('oracle_top64')} |")
    lines += ["","Short-context minus current-embedding weighted R2 by layer: `"+json.dumps(context_gain,sort_keys=True)+"`.","","## Guardrails","","The embedding features are projected into each layer's first 128 natural-state directions, so failure can reflect an insufficient lexical feature budget rather than absence of lexical routing. Conversely, state-PCA routers directly observe the hidden state and are compression ceilings, not prospective input-only explanations. Total operator R2 is measured with a fresh random tensor sketch and cannot exceed the frozen rank-64 oracle ceiling."]
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout="constrained");layers=config["layers"]
    for name in ("position","token","embedding","short_context","state_pca_64","state_pca_256","oracle_top64"):
        axes[0].plot(layers,[results[str(layer)]["metrics"][name]["coordinate_weighted_r2"] for layer in layers],marker="o",label=name)
        axes[1].plot(layers,[results[str(layer)]["metrics"][name]["total_operator_r2"] for layer in layers],marker="o",label=name)
    axes[0].set(xlabel="Layer",ylabel="Weighted coordinate R2",title="Predicting operator-state coordinates");axes[1].set(xlabel="Layer",ylabel="Independent-sketch total R2",title="End-to-end local operator fidelity")
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=8,ncol=2)
    fig.savefig(FIGURE,dpi=180);plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"context_gain":context_gain,"results":results},indent=2));print(REPORT)


if __name__=="__main__":main()
