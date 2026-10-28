#!/usr/bin/env python3
"""Finite suffix test of task-agnostic route representations on fresh FineWeb."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unseeded_route_importance_coverage import fit_basis, pearson


CONFIG = Path("configs/bilin18_unseeded_route_end_to_end_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_unseeded_route_end_to_end_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSEEDED_ROUTE_END_TO_END_CAUSALITY.md"


def reconstruct_pca(routes, basis):
    return (routes @ basis.T) @ basis


def reconstruct_centroid(routes, centers):
    norm = routes.norm(dim=-1, keepdim=True)
    unit = routes / norm.clamp_min(1e-30)
    return norm * centers[(unit @ centers.T).argmax(-1)]


def reconstruct_gated(routes, centers, threshold):
    norm = routes.norm(dim=-1, keepdim=True)
    unit = routes / norm.clamp_min(1e-30)
    similarity = unit @ centers.T
    assignment = similarity.argmax(-1)
    selected = similarity.max(-1).values >= threshold
    return torch.where(selected[..., None], norm * centers[assignment], 0)


def empty_metric():
    return {
        "target_energy": 0.0, "error_energy": 0.0, "dot": 0.0, "approx_energy": 0.0,
        "ce_sum": 0.0, "native_agree": 0, "examples": 0, "kl_sum": 0.0,
        "ce_target_energy": 0.0, "ce_error_energy": 0.0, "ce_dot": 0.0,
        "ce_approx_energy": 0.0, "ce_sign_agree": 0,
    }


def update_metric(metric, native, deleted, approximation, targets):
    target = native - deleted; effect = approximation - deleted
    metric["target_energy"] += float(target.square().sum())
    metric["error_energy"] += float((effect - target).square().sum())
    metric["dot"] += float((target * effect).sum())
    metric["approx_energy"] += float(effect.square().sum())
    metric["ce_sum"] += float(F.cross_entropy(approximation, targets, reduction="sum"))
    metric["native_agree"] += int((approximation.argmax(-1) == native.argmax(-1)).sum())
    native_logp = F.log_softmax(native, -1); approximate_logp = F.log_softmax(approximation, -1)
    metric["kl_sum"] += float((native_logp.exp() * (native_logp - approximate_logp)).sum())
    native_loss = F.cross_entropy(native, targets, reduction="none")
    deleted_loss = F.cross_entropy(deleted, targets, reduction="none")
    approximation_loss = F.cross_entropy(approximation, targets, reduction="none")
    ce_target = deleted_loss - native_loss
    ce_effect = deleted_loss - approximation_loss
    metric["ce_target_energy"] += float(ce_target.square().sum())
    metric["ce_error_energy"] += float((ce_effect - ce_target).square().sum())
    metric["ce_dot"] += float((ce_target * ce_effect).sum())
    metric["ce_approx_energy"] += float(ce_effect.square().sum())
    metric["ce_sign_agree"] += int(((ce_target >= 0) == (ce_effect >= 0)).sum())
    metric["examples"] += len(targets)


def finish(metric):
    target = metric["target_energy"]; approximation = metric["approx_energy"]
    ce_target = metric["ce_target_energy"]; ce_approximation = metric["ce_approx_energy"]
    return {
        "causal_fidelity": 1 - metric["error_energy"] / max(target, 1e-30),
        "causal_cosine": metric["dot"] / max(np.sqrt(target * approximation), 1e-30),
        "effect_norm_ratio": np.sqrt(approximation / max(target, 1e-30)),
        "cross_entropy": metric["ce_sum"] / max(metric["examples"], 1),
        "native_top1_agreement": metric["native_agree"] / max(metric["examples"], 1),
        "kl_native_to_approximation": metric["kl_sum"] / max(metric["examples"], 1),
        "ce_effect_fidelity": 1 - metric["ce_error_energy"] / max(ce_target, 1e-30),
        "ce_effect_cosine": metric["ce_dot"] / max(np.sqrt(ce_target * ce_approximation), 1e-30),
        "ce_effect_norm_ratio": np.sqrt(ce_approximation / max(ce_target, 1e-30)),
        "ce_effect_sign_agreement": metric["ce_sign_agree"] / max(metric["examples"], 1),
        "examples": metric["examples"],
    }


@torch.inference_mode()
def full_logits(model, ids):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,)); x0 = x; bus = None
    for block in model.blocks:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        bus = details["value_bus"]
        pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def intervene(model, ids, target_layer, routes_by_condition):
    """Run one layer natively to its routes, then propagate all replacements."""
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    destination = ids.shape[1] - 1
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        bus = details["value_bus"]
        if layer != target_layer:
            pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
            continue

        projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
        source_writes = torch.einsum("bshd,hdk->bshk", details["mixed_value"].float(), projection)
        names = list(routes_by_condition)
        writes = []
        for name in names:
            route = routes_by_condition[name]
            destination_write = torch.einsum("bsh,bshk->bk", route, source_writes[:, :destination + 1])
            bias = getattr(block.attn.c_proj, "bias", None)
            if bias is not None:
                destination_write = destination_write + bias.float()
            write = details["write"].repeat(len(names), 1, 1) if False else details["write"].clone()
            write[:, destination] = destination_write.to(write.dtype)
            writes.append(write)
        count = len(names); mixed = mixed.repeat(count, 1, 1); x0 = x0.repeat(count, 1, 1)
        bus = bus.repeat(count, 1, 1, 1) if bus.ndim == 4 else bus.repeat(count, 1, 1)
        pre = mixed + torch.cat(writes); normalized = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        for later in model.blocks[layer + 1:]:
            mixed = later.lambdas[0] * x + later.lambdas[1] * x0
            later_details = attention_details(later.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = later_details["value_bus"]
            pre = mixed + later_details["write"]; normalized = F.rms_norm(pre, (dimension,))
            x = pre + later.mlp.Down(later.mlp.Left(normalized) * later.mlp.Right(normalized)) + later.mlp.Down_bias
        raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
        logits = 30 * torch.tanh(raw / 30)
        return names, logits.reshape(count, len(ids), -1)
    raise ValueError(target_layer)


@torch.inference_mode()
def native_routes(model, ids, target_layer):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    destination = ids.shape[1] - 1
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
        if layer == target_layer:
            return details["pattern"][:, :, destination, :destination + 1].float().permute(0, 2, 1)
        pre = mixed + details["write"]; normalized = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    raise ValueError(target_layer)


def main():
    config = json.loads(CONFIG.read_text()); geometry = json.loads(Path(config["geometry_source"]).read_text())
    arrays = np.load(config["geometry_arrays"]); semantic = json.loads(Path(config["semantic_source"]).read_text())
    validation = json.loads(Path(config["semantic_validation"]).read_text()); local = json.loads(Path(config["local_coverage_source"]).read_text())
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    semantic_centers = {layer: [] for layer in config["layers"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in semantic_centers:
            semantic_centers[row["layer"]].append(row["center"])
    assert all(semantic_centers.values())

    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    bases = {layer: torch.as_tensor(fit_basis(arrays[f"discovery_routes_L{layer}"], config["continuous_rank"]), device="cuda") for layer in config["layers"]}
    centroids = {layer: torch.as_tensor([row["center"] for row in geometry["clusters"][str(layer)]], device="cuda") for layer in config["layers"]}
    semantics = {layer: torch.as_tensor(semantic_centers[layer], device="cuda") for layer in config["layers"]}
    rng = np.random.default_rng(config["seed"])
    nulls = {
        layer: [
            torch.as_tensor(
                np.stack([center[rng.permutation(model.config.n_head)] for center in np.asarray(semantic_centers[layer], dtype=np.float32)]),
                device="cuda", dtype=torch.float32,
            )
            for _ in range(config["null_replicates"])
        ]
        for layer in config["layers"]
    }

    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    method_names = ["native", "deleted", "pca4", "centroid4", "semantic_modes"] + [f"permuted_{index}" for index in range(config["null_replicates"])]
    metrics = {layer: {name: empty_metric() for name in method_names} for layer in config["layers"]}
    maximum_native_error = 0.0
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda"); targets = torch.as_tensor(np.asarray(corpus[start:start + len(raw), length], dtype=np.int64), device="cuda")
            reference = full_logits(model, ids)
            for layer in config["layers"]:
                routes = native_routes(model, ids, layer)
                conditions = {
                    "native": routes,
                    "deleted": torch.zeros_like(routes),
                    "pca4": reconstruct_pca(routes, bases[layer]),
                    "centroid4": reconstruct_centroid(routes, centroids[layer]),
                    "semantic_modes": reconstruct_gated(routes, semantics[layer], config["mode_cosine_threshold"]),
                }
                for index, centers in enumerate(nulls[layer]):
                    conditions[f"permuted_{index}"] = reconstruct_gated(routes, centers, config["mode_cosine_threshold"])
                names, logits = intervene(model, ids, layer, conditions)
                output = {name: logits[index] for index, name in enumerate(names)}
                maximum_native_error = max(maximum_native_error, float((output["native"] - reference).abs().max()))
                for name in method_names:
                    update_metric(metrics[layer][name], output["native"], output["deleted"], output[name], targets)
            if start % 32 == 0:
                print(f"end-to-end d={destination} rows {start}/{len(corpus)}", flush=True)

    results = {}
    for layer in config["layers"]:
        results[str(layer)] = {name: finish(row) for name, row in metrics[layer].items()}
        native_ce = results[str(layer)]["native"]["cross_entropy"]; deleted_ce = results[str(layer)]["deleted"]["cross_entropy"]
        denominator = deleted_ce - native_ce
        for name, row in results[str(layer)].items():
            row["ce_recovery_vs_deleted"] = (deleted_ce - row["cross_entropy"]) / denominator if abs(denominator) > 1e-9 else None

    pca_wins = [layer for layer in config["layers"] if results[str(layer)]["pca4"]["causal_fidelity"] > results[str(layer)]["centroid4"]["causal_fidelity"]]
    semantic_fidelity = [results[str(layer)]["semantic_modes"]["causal_fidelity"] for layer in config["layers"]]
    real_mean = float(np.mean(semantic_fidelity))
    null_means = [float(np.mean([results[str(layer)][f"permuted_{index}"]["causal_fidelity"] for layer in config["layers"]])) for index in range(config["null_replicates"])]
    local_by_layer = {row["layer"]: row for row in local["layer_results"]}
    local_values, suffix_values = [], []
    for layer in config["layers"]:
        for method in ("pca4", "centroid4", "semantic_modes"):
            local_values.append(local_by_layer[layer][method]["edge_fidelity"])
            suffix_values.append(results[str(layer)][method]["causal_fidelity"])
    correlation = pearson(local_values, suffix_values)
    ce_improvements = [layer for layer in config["layers"] if results[str(layer)]["semantic_modes"]["cross_entropy"] < results[str(layer)]["deleted"]["cross_entropy"]]
    hypotheses = {
        "E0_native_path_is_exact": {"supported": maximum_native_error <= .02, "maximum_logit_error": maximum_native_error},
        "E1_continuous_beats_discrete_end_to_end": {"supported": len(pca_wins) >= 3, "layers": pca_wins},
        "E2_semantic_modes_have_finite_causal_coverage": {"supported": sum(value > 0 for value in semantic_fidelity) >= 3 and real_mean >= .05, "by_layer": dict(zip(map(str, config["layers"]), semantic_fidelity)), "mean": real_mean},
        "E3_semantic_binding_beats_permuted_modes": {"supported": real_mean > max(null_means), "observed": real_mean, "null_means": null_means},
        "E4_local_write_fidelity_predicts_suffix_fidelity": {"supported": correlation >= .50, "correlation": correlation, "cells": len(local_values)},
        "E5_logit_coverage_is_not_only_noise": {"supported": len(ce_improvements) >= 3, "layers": ce_improvements},
    }
    if "E6_semantic_modes_recover_signed_examplewise_ce_effects" in config["hypotheses"]:
        semantic_ce = [results[str(layer)]["semantic_modes"]["ce_effect_fidelity"] for layer in config["layers"]]
        null_ce_means = [float(np.mean([results[str(layer)][f"permuted_{index}"]["ce_effect_fidelity"] for layer in config["layers"]])) for index in range(config["null_replicates"])]
        hypotheses["E6_semantic_modes_recover_signed_examplewise_ce_effects"] = {
            "supported": sum(value > 0 for value in semantic_ce) >= 3 and float(np.mean(semantic_ce)) >= .05,
            "by_layer": dict(zip(map(str, config["layers"]), semantic_ce)), "mean": float(np.mean(semantic_ce)),
        }
        hypotheses["E7_ce_effect_recovery_beats_permuted_modes"] = {
            "supported": float(np.mean(semantic_ce)) > float(np.quantile(null_ce_means, .95)),
            "observed": float(np.mean(semantic_ce)), "null_mean": float(np.mean(null_ce_means)),
            "null_p95": float(np.quantile(null_ce_means, .95)), "null_means": null_ce_means,
        }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# End-to-end causality of task-agnostic route representations", "", "Fresh FineWeb sequences are run to four prospectively selected layers. Only the destination's realized nine-head source routes are replaced; the target-layer MLP and all later layers are recomputed. Fidelity is measured relative to the complete finite effect of deleting those destination routes.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | Native/deleted CE | PCA fidelity/CE | Centroid fidelity/CE | Semantic fidelity/CE | Semantic CE recovery | Best permuted fidelity |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; best_null = max(row[f"permuted_{index}"]["causal_fidelity"] for index in range(config["null_replicates"]))
        lines.append(f"| {layer} | {row['native']['cross_entropy']:.3f}/{row['deleted']['cross_entropy']:.3f} | {row['pca4']['causal_fidelity']:.3f}/{row['pca4']['cross_entropy']:.3f} | {row['centroid4']['causal_fidelity']:.3f}/{row['centroid4']['cross_entropy']:.3f} | {row['semantic_modes']['causal_fidelity']:.3f}/{row['semantic_modes']['cross_entropy']:.3f} | {row['semantic_modes']['ce_recovery_vs_deleted']:.3f} | {best_null:.3f} |")
    lines += ["", "### Per-example finite CE effects", "", "| Layer | PCA fidelity | Centroid fidelity | Semantic fidelity/cosine/sign |", "|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]
        lines.append(f"| {layer} | {row['pca4']['ce_effect_fidelity']:.3f} | {row['centroid4']['ce_effect_fidelity']:.3f} | {row['semantic_modes']['ce_effect_fidelity']:.3f}/{row['semantic_modes']['ce_effect_cosine']:.3f}/{row['semantic_modes']['ce_effect_sign_agreement']:.3f} |")
    lines += ["", "## Guardrails", "", "This is finite end-to-end evidence for one destination-route intervention, not a fraction of total model computation. Full-logit fidelity weights all vocabulary coordinates; CE tests the observed next token and can disagree when the removed native route was locally harmful. The four layers were chosen after the local coverage audit, so conclusions about all layers require another predeclared replication."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
