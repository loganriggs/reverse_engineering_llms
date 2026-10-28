#!/usr/bin/env python3
"""Trace the contextual L9H8 boundary value back through Layers 0--8."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_upstream_trace.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_upstream_trace.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_UPSTREAM_TRACE.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_upstream_trace.png"


def product(values):
    result = 1.0
    for value in values: result = result * value
    return result


def residual_components(model, delta_x0, attention_deltas, mlp_deltas, target_layer=9):
    """Exact additive telescope of the mixed-state change at target_layer."""
    carry = [model.blocks[layer].lambdas[0].float() for layer in range(target_layer + 1)]
    inject = [model.blocks[layer].lambdas[1].float() for layer in range(target_layer + 1)]
    components = {"initial_embedding": carry[target_layer] * product(carry[:target_layer]) * delta_x0}
    for layer in range(target_layer):
        scale = carry[target_layer] * product(carry[layer + 1:target_layer])
        components[f"repeat_embedding_l{layer}"] = scale * inject[layer] * delta_x0
        components[f"attention_l{layer}"] = scale * attention_deltas[layer]
        components[f"mlp_l{layer}"] = scale * mlp_deltas[layer]
    components[f"repeat_embedding_l{target_layer}"] = inject[target_layer] * delta_x0
    return components


def rmsnorm_jvp(state, directions):
    radius = (state.square().mean(-1, keepdim=True) + torch.finfo(state.dtype).eps).sqrt()
    dot = (directions * state).mean(-1, keepdim=True)
    return directions / radius - state * dot / radius.pow(3)


@torch.inference_mode()
def integrated_value_components(block, base, target, components, head, points):
    names = list(components); batch, dimension = base.shape
    directions = torch.stack([components[name] for name in names])
    nodes, weights = np.polynomial.legendre.leggauss(points)
    values = torch.zeros_like(directions)
    delta = target - base
    for node, weight in zip((nodes + 1) / 2, weights / 2):
        state = base + float(node) * delta
        expanded = state[None].expand(len(names), -1, -1)
        normalized = rmsnorm_jvp(expanded, directions)
        projected = F.linear(normalized.reshape(-1, dimension), block.attn.c_v.weight.float(), bias=None)
        values += float(weight) * projected.reshape(len(names), batch, dimension)
    head_dim = dimension // block.attn.n_head
    sl = slice(head * head_dim, (head + 1) * head_dim)
    result = {name: values[index, :, sl] for index, name in enumerate(names)}
    base_value = block.attn.c_v(F.rms_norm(base, (dimension,))).float()[:, sl]
    target_value = block.attn.c_v(F.rms_norm(target, (dimension,))).float()[:, sl]
    return result, target_value - base_value


def empty_stats(names):
    return {name: {"sum": 0.0, "wins": 0, "count": 0} for name in names}


@torch.inference_mode()
def collect(model, raw, records, contrasts, config):
    layer, head, destination = 9, 8, 127
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    projection = model.blocks[layer].attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
    variant_names = ["full", "attention_all", "mlp_all"] + [f"layer_{i}" for i in range(layer)] + [f"prefix_{i}" for i in range(layer)]
    endpoint = empty_stats(variant_names); local_inner = {name: 0.0 for name in [f"layer_{i}" for i in range(layer)] + ["attention_all", "mlp_all"]}
    local_energy = 0.0; embedding_energy = mixed_energy = 0.0
    telescope_error = value_error = 0.0
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]; batch_size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch]); swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate([original, swapped]), device="cuda")
        source = torch.as_tensor([row[1] for row in batch], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        attention_deltas, mlp_deltas = [], []
        for block_index, block in enumerate(model.blocks[:layer + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus,
                                        model.config, model.tt)
            pre = mixed + details["write"]
            state = F.rms_norm(pre, (dimension,))
            mlp_write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            if block_index < layer:
                paired = torch.arange(batch_size, device="cuda")
                attention_deltas.append(details["write"][paired + batch_size, source].float() - details["write"][paired, source].float())
                mlp_deltas.append(mlp_write[paired + batch_size, source].float() - mlp_write[paired, source].float())
                x = pre + mlp_write; bus = details["value_bus"]; continue

            paired = torch.arange(batch_size, device="cuda")
            base_mixed = mixed[paired, source].float(); target_mixed = mixed[paired + batch_size, source].float()
            delta_x0 = x0[paired + batch_size, source].float() - x0[paired, source].float()
            components = residual_components(model, delta_x0, attention_deltas, mlp_deltas, layer)
            mixed_target = target_mixed - base_mixed; reconstructed = sum(components.values())
            telescope_error = max(telescope_error, float((reconstructed - mixed_target).norm() / mixed_target.norm().clamp_min(1e-30)))
            mixed_energy += float(mixed_target.square().sum())
            embedding_energy += float(sum(value for name, value in components.items() if "embedding" in name).square().sum())
            value_components, value_target = integrated_value_components(block, base_mixed, target_mixed, components,
                                                                          head, config["quadrature_points"])
            value_error = max(value_error, float((sum(value_components.values()) - value_target).norm()
                                                 / value_target.norm().clamp_min(1e-30)))
            layer_values = {f"layer_{i}": value_components[f"attention_l{i}"] + value_components[f"mlp_l{i}"]
                            for i in range(layer)}
            attention_all = sum(value_components[f"attention_l{i}"] for i in range(layer))
            mlp_all = sum(value_components[f"mlp_l{i}"] for i in range(layer))
            grouped = {**layer_values, "attention_all": attention_all, "mlp_all": mlp_all}
            local_energy += float(value_target.double().square().sum())
            for name, value in grouped.items(): local_inner[name] += float((value.double() * value_target.double()).sum())

            prefix = torch.zeros_like(value_target); variant_values = {"full": value_target,
                                                                      "attention_all": attention_all,
                                                                      "mlp_all": mlp_all, **layer_values}
            for index in range(layer):
                prefix = prefix + layer_values[f"layer_{index}"]
                variant_values[f"prefix_{index}"] = prefix.clone()

            pattern = (details["score1"][paired, head, destination, source]
                       * details["score2"][paired, head, destination, source]).float()
            base_attention = details["write"][:batch_size].float()
            attention_variants = []
            for name in variant_names:
                edge_delta = pattern[:, None] * (1 - block.attn.lamb.float()) * variant_values[name] @ projection
                modified = base_attention.clone(); modified[torch.arange(batch_size, device="cuda"), destination] += edge_delta
                attention_variants.append(modified)
            attention = torch.cat(attention_variants); repeats = len(variant_names)
            mixed0 = mixed[:batch_size].float().repeat(repeats, 1, 1); x00 = x0[:batch_size].float().repeat(repeats, 1, 1)
            bus0 = details["value_bus"][:batch_size].float().repeat(repeats, 1, 1, 1)
            state0 = F.rms_norm(mixed0 + attention, (dimension,))
            suffix_x = mixed0 + attention + block.mlp.Down(block.mlp.Left(state0) * block.mlp.Right(state0)) + block.mlp.Down_bias
            suffix_bus = bus0
            for suffix in model.blocks[layer + 1:]: suffix_x, suffix_bus = suffix(suffix_x, suffix_bus, x00)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(suffix_x, (dimension,))[:, destination]).float() / 30)
            scores = (logits @ contrasts[:, :1]).reshape(repeats, batch_size)
            base_score = scores[0] * 0  # only the common origin matters; variants are ordered below
            # Re-run the unmodified original edge endpoint once by subtracting the
            # first variant's delta through a zero slot computed from native logits.
            native_state = F.rms_norm(mixed[:batch_size] + details["write"][:batch_size].float(), (dimension,))
            native_x = mixed[:batch_size].float() + details["write"][:batch_size].float() + block.mlp.Down(block.mlp.Left(native_state) * block.mlp.Right(native_state)) + block.mlp.Down_bias
            native_bus = details["value_bus"][:batch_size].float(); native_x0 = x0[:batch_size].float()
            for suffix in model.blocks[layer + 1:]: native_x, native_bus = suffix(native_x, native_bus, native_x0)
            native_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(native_x, (dimension,))[:, destination]).float() / 30)
            native_score = native_logits @ contrasts[:, 0]
            direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
            for index, name in enumerate(variant_names):
                directed = direction * (scores[index] - native_score)
                endpoint[name]["sum"] += float(directed.sum()); endpoint[name]["wins"] += int((directed > 0).sum()); endpoint[name]["count"] += batch_size
            break
    endpoint = {name: {"mean": row["sum"] / row["count"], "win": row["wins"] / row["count"]} for name, row in endpoint.items()}
    return {
        "telescope_error": telescope_error, "value_recomposition_error": value_error,
        "embedding_energy_ratio": embedding_energy / max(mixed_energy, 1e-30),
        "local_projection_share": {name: value / max(local_energy, 1e-30) for name, value in local_inner.items()},
        "endpoint": endpoint,
    }


def main():
    config = json.loads(CONFIG.read_text()); base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    pairs = [(tokenizer.encode(us, add_special_tokens=False)[0], tokenizer.encode(uk, add_special_tokens=False)[0]) for us, uk in base["spelling_pairs"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = []
    for fold in range(config["folds"]):
        target_indices = set(range(fold * config["target_pairs_per_fold"], (fold + 1) * config["target_pairs_per_fold"]))
        target_pairs = [pair for index, pair in enumerate(pairs) if index in target_indices]
        cues = [pair for index, pair in enumerate(pairs) if index not in target_indices]
        contrasts = make_contrasts(model.lm_head.weight.shape[0], target_pairs, 0, config["seed"] + fold, "cuda")
        for split, (corpus_start, corpus_end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[corpus_start:corpus_end, :128], dtype=np.int64)
            records = select_records(raw, boundary, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + fold * 10 + split)
            print("trace", fold, split, len(records), flush=True)
            results.append({"fold": fold, "split": split, **collect(model, raw, records, contrasts, config)})
    max_error = max(max(row["telescope_error"], row["value_recomposition_error"]) for row in results)
    u0 = max_error <= 1e-5; u1 = all(row["embedding_energy_ratio"] <= 1e-8 for row in results)
    max_layer = [max(row["local_projection_share"][f"layer_{i}"] for i in range(9)) for row in results]
    u2 = all(value <= .75 for value in max_layer)
    material = sum(abs(row["endpoint"][name]["mean"]) >= .1 * abs(row["endpoint"]["full"]["mean"])
                   for row in results for name in ("attention_all", "mlp_all"))
    material_cells = sum(all(abs(row["endpoint"][name]["mean"]) >= .1 * abs(row["endpoint"]["full"]["mean"])
                             for name in ("attention_all", "mlp_all")) for row in results)
    u3 = material_cells >= 5
    u4 = all(row["endpoint"]["prefix_2"]["mean"] >= .75 * row["endpoint"]["full"]["mean"] for row in results)
    u5 = all(row["endpoint"]["prefix_6"]["mean"] >= .75 * row["endpoint"]["full"]["mean"] for row in results)
    u6 = all(row["endpoint"]["full"]["win"] >= .80 for row in results)
    hypotheses = {
        "U0_telescope_and_value_decomposition_recompose": {"supported": u0, "max_error": max_error},
        "U1_repeated_embedding_is_negative_control": {"supported": u1},
        "U2_no_single_layer_creates_the_value": {"supported": u2, "max_layer_shares": max_layer},
        "U3_attention_and_mlp_are_both_material": {"supported": u3, "cells": material_cells},
        "U4_style_value_is_compiled_by_layer2": {"supported": u4},
        "U5_style_value_is_compiled_by_layer6": {"supported": u5},
        "U6_full_trace_preserves_direct_transport": {"supported": u6},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Upstream construction of the L9H8 boundary style value", "",
             "The cue swap leaves the boundary token unchanged. The Layer-9 mixed-state difference at that boundary is telescoped exactly into initial/repeated embeddings and every earlier attention/MLP write. A 32-point integrated RMSNorm/value-map decomposition then assigns the finite L9H8 current-value change to those additive sources. Each assembled value is inserted into the single boundary edge and propagated through the native suffix.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Fold/split | Full win/mean | Attention mean | MLP mean | Prefix L2/L6 mean | strongest layer/share |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        strongest = max(range(9), key=lambda i: row["local_projection_share"][f"layer_{i}"])
        lines.append(f"| {row['fold']}/{row['split']} | {row['endpoint']['full']['win']:.3f}/{row['endpoint']['full']['mean']:+.4g} | {row['endpoint']['attention_all']['mean']:+.4g} | {row['endpoint']['mlp_all']['mean']:+.4g} | {row['endpoint']['prefix_2']['mean']:+.4g}/{row['endpoint']['prefix_6']['mean']:+.4g} | L{strongest}/{row['local_projection_share'][f'layer_{strongest}']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "This localizes which earlier writes compose the contextual value available at Layer 9. It does not yet decompose an earlier attention write into its own source positions or QK/value factors. Integrated contributions are path-dependent allocations along the straight original-to-counterfactual state segment; the exact telescope and finite suffix interventions are path-independent endpoints."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(9); fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for row in results: axes[0].plot(x, [row["local_projection_share"][f"layer_{i}"] for i in x], marker="o", alpha=.65, label=f"f{row['fold']}s{row['split']}")
    axes[0].axhline(0, color="black", linewidth=.8); axes[0].set(xlabel="Source layer", ylabel="Projection share onto L9 value change", title="Where the contextual value is constructed"); axes[0].legend(fontsize=7)
    for row in results: axes[1].plot(x, [row["endpoint"][f"prefix_{i}"]["mean"] / max(abs(row["endpoint"]["full"]["mean"]), 1e-30) for i in x], marker="o", alpha=.65)
    axes[1].axhline(.75, color="black", linestyle="--", linewidth=.8); axes[1].set(xlabel="Cumulative through layer", ylabel="Finite mean / full mean", title="Cumulative causal compilation")
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
