#!/usr/bin/env python3
"""Exact Q/K/current-value/value-bus and head audit for squared attention."""

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
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_attention_qkov_depth_audit.json")
OUTPUT = TENSOR_ROOT / "bilin12_attention_qkov_depth_audit.json"
REPORT = TENSOR_ROOT / "BILIN12_ATTENTION_QKOV_DEPTH_AUDIT.md"
FIGURE = TENSOR_ROOT / "bilin12_attention_qkov_depth_audit.png"
FACTORS = ("query", "key", "current_value", "value_bus")


def vector_shapley(worlds):
    count = len(FACTORS); result = {}
    for index, factor in enumerate(FACTORS):
        bit = 1 << index; effect = torch.zeros_like(worlds[0])
        for mask in range(1 << count):
            if mask & bit: continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            effect += weight * (worlds[mask | bit] - worlds[mask])
        result[factor] = effect
    return result


@torch.inference_mode()
def attention_details(model, block, normalized, value_bus):
    batch, length, dimension = normalized.shape
    heads, head_dimension = block.attn.n_head, block.attn.head_dim
    query = block.attn.c_q(normalized).view(batch, length, heads, head_dimension)
    key = block.attn.c_k(normalized).view(batch, length, heads, head_dimension)
    current_value = block.attn.c_v(normalized).view(batch, length, heads, head_dimension)
    first_value = current_value if value_bus is None else value_bus.view_as(current_value)
    cosine, sine = block.attn.rotary(query)
    query = model.tt.apply_rotary_emb(F.rms_norm(query, (head_dimension,)), cosine, sine)
    key = model.tt.apply_rotary_emb(F.rms_norm(key, (head_dimension,)), cosine, sine)
    scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dimension
    causal = torch.tril(torch.ones(length, length, device=normalized.device, dtype=torch.bool))
    pattern = scores.square().masked_fill(~causal, 0.0)
    pattern = pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
    mixed_value = (1 - block.attn.lamb) * current_value + block.attn.lamb * first_value
    attended = torch.einsum("bhqk,bkhd->bhqd", pattern, mixed_value)
    flattened = attended.transpose(1, 2).contiguous().view(batch, length, dimension)
    write = block.attn.c_proj(flattened)
    return {
        "query": query, "key": key, "current_value": current_value,
        "value_bus": first_value, "pattern": pattern, "attended": attended, "write": write,
    }


@torch.inference_mode()
def trace(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; value_bus = None; rows = []
    maximum_native_error = 0.0
    for block in model.blocks:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (dimension,))
        details = attention_details(model, block, normalized, value_bus)
        native_write, next_bus = block.attn(normalized, value_bus)
        maximum_native_error = max(maximum_native_error, float((details["write"] - native_write).abs().max()))
        pre = mixed + details["write"]
        state = F.rms_norm(pre, (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + mlp
        rows.append({"x0": x0, "mixed": mixed, "details": details, "value_bus": next_bus, "output": x})
        value_bus = next_bus
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return rows, 30 * torch.tanh(raw / 30), maximum_native_error


def attention_world_writes(block, base, changed):
    batch, length, heads, head_dimension = base["query"].shape
    dimension = heads * head_dimension
    worlds, attended_worlds = [], []
    for mask in range(16):
        query = changed["query"] if mask & 1 else base["query"]
        key = changed["key"] if mask & 2 else base["key"]
        current = changed["current_value"] if mask & 4 else base["current_value"]
        bus = changed["value_bus"] if mask & 8 else base["value_bus"]
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dimension
        causal = torch.tril(torch.ones(length, length, device=query.device, dtype=torch.bool))
        pattern = scores.square().masked_fill(~causal, 0.0)
        pattern = pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)
        mixed = (1 - block.attn.lamb) * current + block.attn.lamb * bus
        attended = torch.einsum("bhqk,bkhd->bhqd", pattern, mixed)
        flattened = attended.transpose(1, 2).contiguous().view(batch, length, dimension)
        worlds.append(block.attn.c_proj(flattened)); attended_worlds.append(attended)
    return torch.stack(worlds), torch.stack(attended_worlds)


@torch.inference_mode()
def logits_from_attention_writes(model, layer, context, writes):
    worlds, batch, length, dimension = writes.shape
    mixed = context["mixed"].repeat(worlds, 1, 1)
    flattened_writes = writes.flatten(0, 1)
    pre = mixed + flattened_writes
    block = model.blocks[layer]
    state = F.rms_norm(pre, (dimension,))
    mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    x = pre + mlp
    x0 = context["x0"].repeat(worlds, 1, 1)
    value_bus = context["value_bus"].repeat(worlds, 1, 1, 1)
    for suffix_block in model.blocks[layer + 1:]: x, value_bus = suffix_block(x, value_bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return (30 * torch.tanh(raw / 30)).reshape(worlds, batch, -1)


def projected_shapley(worlds, reference_logits=None):
    target = worlds[-1] - worlds[0]
    effects = vector_shapley(worlds)
    if reference_logits is None:
        denominator = target.square().sum().clamp_min(1e-30)
        return {factor: float((effect * target).sum() / denominator) for factor, effect in effects.items()}
    energy, _, _, _ = fisher_terms(reference_logits, target, target)
    denominator = energy.sum().clamp_min(1e-30)
    result = {}
    for factor, effect in effects.items():
        _, _, cross, _ = fisher_terms(reference_logits, target, effect)
        result[factor] = float(cross.sum() / denominator)
    return result


def head_writes(block, base_attended, changed_attended, position):
    delta = changed_attended[:, :, position] - base_attended[:, :, position]
    heads, head_dimension = block.attn.n_head, block.attn.head_dim
    writes = []
    for head in range(heads):
        weight = block.attn.c_proj.weight[:, head * head_dimension:(head + 1) * head_dimension]
        writes.append(delta[:, head] @ weight.T)
    return torch.stack(writes)


def empty_accumulator(heads):
    return {
        "examples": 0, "native_error": 0.0, "factor_write_error": 0.0,
        "local_den": 0.0, "local_cross": {factor: 0.0 for factor in FACTORS},
        "insert_den": 0.0, "insert_cross": {factor: 0.0 for factor in FACTORS},
        "remove_den": 0.0, "remove_cross": {factor: 0.0 for factor in FACTORS},
        "insert_kl": 0.0, "full_kl": 0.0,
        "head_local_cross": [0.0] * heads, "head_insert_cross": [0.0] * heads,
        "head_insert_energy": 0.0, "head_insert_sum_error": 0.0,
    }


def update(acc, block, base, changed, world_writes, attended, insert_logits, remove_logits, head_logits, native_base, native_changed, native_errors, position):
    local_target = world_writes[15, :, position] - world_writes[0, :, position]
    local_den = local_target.square().sum().clamp_min(1e-30)
    local_effects = vector_shapley(world_writes[:, :, position])
    acc["local_den"] += float(local_den)
    for factor, effect in local_effects.items(): acc["local_cross"][factor] += float((effect * local_target).sum())
    for name, logits, key in (("insert", insert_logits, "insert"), ("remove", remove_logits, "remove")):
        target = logits[15] - logits[0]; reference = logits[0]
        energy, _, _, _ = fisher_terms(reference, target, target); denominator = energy.sum()
        acc[f"{key}_den"] += float(denominator)
        for factor, effect in vector_shapley(logits).items():
            _, _, cross, _ = fisher_terms(reference, target, effect)
            acc[f"{key}_cross"][factor] += float(cross.sum())
    insertion_target = insert_logits[15] - insert_logits[0]
    insertion_energy, _, _, _ = fisher_terms(insert_logits[0], insertion_target, insertion_target)
    head_effects = head_logits[1:] - head_logits[0]
    head_sum = head_effects.sum(0)
    _, _, _, sum_error = fisher_terms(insert_logits[0], insertion_target, head_sum)
    acc["head_insert_energy"] += float(insertion_energy.sum())
    acc["head_insert_sum_error"] += float(sum_error.sum())
    native_head_writes = head_writes(block, base["attended"], changed["attended"], position)
    for head in range(block.attn.n_head):
        acc["head_local_cross"][head] += float((native_head_writes[head] * local_target).sum())
        _, _, cross, _ = fisher_terms(insert_logits[0], insertion_target, head_effects[head])
        acc["head_insert_cross"][head] += float(cross.sum())
    acc["examples"] += len(native_base)
    acc["native_error"] = max(acc["native_error"], *native_errors)
    acc["factor_write_error"] = max(acc["factor_write_error"], float((world_writes[15] - changed["write"]).abs().max()), float((world_writes[0] - base["write"]).abs().max()))
    acc["insert_kl"] += float(exact_kl(insert_logits[0], insert_logits[15]).sum())
    acc["full_kl"] += float(exact_kl(native_base, native_changed).sum())


def finalize(acc):
    local = {factor: acc["local_cross"][factor] / max(acc["local_den"], 1e-30) for factor in FACTORS}
    insertion = {factor: acc["insert_cross"][factor] / max(acc["insert_den"], 1e-30) for factor in FACTORS}
    removal = {factor: acc["remove_cross"][factor] / max(acc["remove_den"], 1e-30) for factor in FACTORS}
    local_heads = [value / max(acc["local_den"], 1e-30) for value in acc["head_local_cross"]]
    insertion_heads = [value / max(acc["insert_den"], 1e-30) for value in acc["head_insert_cross"]]
    absolute = sum(abs(value) for value in local_heads)
    return {
        "examples": acc["examples"], "local_write_shapley": local,
        "insertion_causal_shapley": insertion, "removal_causal_shapley": removal,
        "local_head_projection": local_heads, "insertion_head_projection": insertion_heads,
        "top_local_head": int(np.argmax(np.abs(local_heads))),
        "top_local_head_absolute_share": max(abs(value) for value in local_heads) / max(absolute, 1e-30),
        "isolated_head_sum_fidelity": 1 - acc["head_insert_sum_error"] / max(acc["head_insert_energy"], 1e-30),
        "attention_insertion_to_full_kl": acc["insert_kl"] / max(acc["full_kl"], 1e-30),
        "maximum_native_write_error": acc["native_error"],
        "maximum_factor_write_error": acc["factor_write_error"],
    }


@torch.inference_mode()
def evaluate(model, tokenizer, spec, config):
    rows = load_rows(spec); newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    accumulators = {layer: empty_accumulator(model.config.n_head) for layer in config["layers"]}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        base_ids = torch.as_tensor(raw, device="cuda"); keep = base_ids[:, config["position"]] != newline
        if not bool(keep.any()): continue
        base_ids = base_ids[keep]; changed_ids = base_ids.clone(); changed_ids[:, config["position"]] = newline
        base_rows, native_base, base_error = trace(model, base_ids)
        changed_rows, native_changed, changed_error = trace(model, changed_ids)
        for layer in config["layers"]:
            block = model.blocks[layer]; base = base_rows[layer]; changed = changed_rows[layer]
            writes, attended = attention_world_writes(block, base["details"], changed["details"])
            insert_logits = logits_from_attention_writes(model, layer, base, writes)
            remove_logits = logits_from_attention_writes(model, layer, changed, writes)
            native_head_delta = head_writes(block, base["details"]["attended"], changed["details"]["attended"], config["position"])
            head_write_worlds = [base["details"]["write"]]
            for head in range(model.config.n_head):
                value = base["details"]["write"].clone(); value[:, config["position"]] += native_head_delta[head]
                head_write_worlds.append(value)
            head_logits = logits_from_attention_writes(model, layer, base, torch.stack(head_write_worlds))
            update(accumulators[layer], block, base["details"], changed["details"], writes, attended, insert_logits, remove_logits, head_logits, native_base, native_changed, (base_error, changed_error), config["position"])
    return {str(layer): finalize(accumulators[layer]) for layer in config["layers"]}


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    results = {}
    for corpus, spec in config["evaluation"].items(): print("evaluating", corpus, flush=True); results[corpus] = evaluate(model, tokenizer, spec, config)
    rows = [results[c][str(layer)] for c in results for layer in config["layers"]]
    hypotheses = {
        "Q1_manual_attention_and_factor_recomposition_are_exact": {"supported": all(max(row["maximum_native_write_error"], row["maximum_factor_write_error"]) <= 1e-4 for row in rows)},
        "Q2_query_routing_dominates_self_key_routing": {"supported": all(abs(row["local_write_shapley"]["query"]) >= abs(row["local_write_shapley"]["key"]) for row in rows)},
        "Q3_routing_dominates_values": {"supported": all(sum(abs(results[c][str(layer)]["local_write_shapley"]["query"]) + abs(results[c][str(layer)]["local_write_shapley"]["key"]) >= abs(results[c][str(layer)]["local_write_shapley"]["current_value"]) + abs(results[c][str(layer)]["local_write_shapley"]["value_bus"]) for layer in config["layers"]) >= 8 for c in results)},
        "Q4_persistent_bus_is_not_the_main_value_source": {"supported": all(abs(row["local_write_shapley"]["current_value"]) >= abs(row["local_write_shapley"]["value_bus"]) for row in rows)},
        "Q5_insertion_and_removal_agree_on_top_factor": {"supported": all(max(FACTORS, key=lambda factor: abs(row["insertion_causal_shapley"][factor])) == max(FACTORS, key=lambda factor: abs(row["removal_causal_shapley"][factor])) for row in rows)},
        "Q6_attention_effect_is_head_concentrated": {"supported": all(max(results[c][str(layer)]["top_local_head_absolute_share"] for layer in config["layers"]) >= .50 for c in results)},
        "Q7_isolated_heads_add_approximately": {"supported": all(row["isolated_head_sum_fidelity"] >= .80 for row in rows)},
        "Q8_attention_only_behavior_is_small": {"supported": all(row["attention_insertion_to_full_kl"] <= .10 for row in rows)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))

    layers = np.asarray(config["layers"]); colors = dict(zip(FACTORS, ("#4c78a8", "#f58518", "#54a24b", "#e45756"))); fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for factor in FACTORS:
        values = np.asarray([[results[c][str(layer)]["local_write_shapley"][factor] for layer in layers] for c in results]); axes[0].plot(layers, values.mean(0), marker="o", color=colors[factor], label=factor); axes[0].fill_between(layers, values.min(0), values.max(0), color=colors[factor], alpha=.12)
    axes[0].set_title("Exact attention-write allocation")
    for corpus in results: axes[1].plot(layers, [results[corpus][str(layer)]["top_local_head_absolute_share"] for layer in layers], marker="o", label=corpus)
    axes[1].axhline(.5, color="black", ls="--", lw=.8); axes[1].set_title("Largest head absolute share")
    for corpus in results: axes[2].plot(layers, [results[corpus][str(layer)]["attention_insertion_to_full_kl"] for layer in layers], marker="o", label=corpus)
    axes[2].axhline(.1, color="black", ls="--", lw=.8); axes[2].set_title("Attention-only KL / full token KL")
    for axis in axes: axis.set_xlabel("layer"); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = ["# Squared-attention QK/OV audit across depth", "", "For the final-token-to-newline edit, each native squared-attention write is reconstructed from normalized rotary queries and keys, row-normalized squared scores, the current layer's value projection, and the persistent Layer-0 value bus. All 16 Q/K/current-value/bus hybrids are evaluated exactly. Their local write allocation is separated from insertion into a base block and removal inside a changed block. Native per-head OV writes provide a second decomposition.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![QKOV depth audit](bilin12_attention_qkov_depth_audit.png)", "", "| Corpus | Layer | Q / K / current-V / bus local | Q / K / current-V / bus insert | Top head/share | Head-sum fidelity | Attention/full KL |", "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus in results:
        for layer in config["layers"]:
            row = results[corpus][str(layer)]; local = " / ".join(f"{row['local_write_shapley'][factor]:.3f}" for factor in FACTORS); insertion = " / ".join(f"{row['insertion_causal_shapley'][factor]:.3f}" for factor in FACTORS)
            lines.append(f"| {corpus} | {layer} | {local} | {insertion} | {row['top_local_head']} / {row['top_local_head_absolute_share']:.3f} | {row['isolated_head_sum_fidelity']:.3f} | {row['attention_insertion_to_full_kl']:.4f} |")
    lines += ["", "The local allocation explains the attention operator itself. Insertion and removal ask whether the same factorization survives MLP recomputation and the remaining nonlinear suffix. A factor can dominate the local write yet have little end-to-end importance if attention is not the bottleneck for this edit."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
