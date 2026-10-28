#!/usr/bin/env python3
"""Factor the sharp his->her layer-1 path into Q/K/current-V/value-bus edges."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import fisher_terms
from run_bilin12_semantic_source_jvp_predictor import _rotary_constants, projected_source_writes
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_his_her_layer1_qkov_path.json")
OUTPUT = TENSOR_ROOT / "bilin12_his_her_layer1_qkov_path.json"
REPORT = TENSOR_ROOT / "BILIN12_HIS_HER_LAYER1_QKOV_PATH.md"
FIGURE = TENSOR_ROOT / "bilin12_his_her_layer1_qkov_path.png"
FACTORS = ("query_routing", "key_routing", "current_value", "value_bus")


def trace_qkov(model, x0, target_layer, destination):
    x = x0
    value_bus = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (mixed.shape[-1],))
        batch, length, dimension = normalized.shape
        heads, head_dimension = block.attn.n_head, block.attn.head_dim
        raw_query = block.attn.c_q(normalized).view(batch, length, heads, head_dimension)
        raw_key = block.attn.c_k(normalized).view(batch, length, heads, head_dimension)
        query = raw_query
        key = raw_key
        current = block.attn.c_v(normalized).view(batch, length, heads, head_dimension)
        first = current if value_bus is None else value_bus
        cosine, sine = _rotary_constants(block, length, normalized.device, normalized.dtype)
        query = model.tt.apply_rotary_emb(F.rms_norm(query, (head_dimension,)), cosine, sine)
        key = model.tt.apply_rotary_emb(F.rms_norm(key, (head_dimension,)), cosine, sine)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dimension
        causal = torch.tril(torch.ones(length, length, device=x0.device, dtype=torch.bool))
        unnormalized = scores.square().masked_fill(~causal, 0.0)
        pattern = unnormalized / unnormalized.sum(-1, keepdim=True).clamp_min(1e-9)
        value = (1 - block.attn.lamb) * current + block.attn.lamb * first
        attended = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
        write = block.attn.c_proj(attended.transpose(1, 2).contiguous().view(batch, length, dimension))
        source = projected_source_writes(block, pattern, value, destination)
        if layer == target_layer:
            return raw_query[:, destination], raw_key, query[:, destination], key, current, first, pattern[:, :, destination], value, scores[:, :, destination], source
        pre = mixed + write
        state = F.rms_norm(pre, (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + mlp
        value_bus = first
    raise ValueError(target_layer)


def project_heads(block, per_head_source):
    pieces = []
    for head in range(block.attn.n_head):
        start = head * block.attn.head_dim
        stop = (head + 1) * block.attn.head_dim
        pieces.append(per_head_source[:, head] @ block.attn.c_proj.weight[:, start:stop].T)
    return torch.stack(pieces, dim=1)


def factor_row(model, base_x0, tangent, location, layer, destination):
    point = base_x0 + location * tangent
    function = lambda value: trace_qkov(model, value, layer, destination)
    primal, derivative = torch.func.jvp(function, (point,), (tangent,))
    raw_query, raw_key, query, key, current, bus, pattern, mixed_value, scores, _ = primal
    draw_query, draw_key, dquery, dkey, dcurrent, dbus, dpattern, _, dscores, dsource = derivative
    head_dimension = model.blocks[layer].attn.head_dim
    score = scores
    query_score = torch.einsum("bhd,bshd->bhs", dquery, key) / head_dimension
    key_score = torch.einsum("bhd,bshd->bhs", query, dkey) / head_dimension
    unnormalized = score.square()
    denominator = unnormalized.sum(-1, keepdim=True).clamp_min(1e-9)

    def pattern_part(score_part):
        dunnormalized = 2 * score * score_part
        return (dunnormalized * denominator - unnormalized * dunnormalized.sum(-1, keepdim=True)) / denominator.square()

    query_pattern = pattern_part(query_score)
    key_pattern = pattern_part(key_score)
    block = model.blocks[layer]
    factors_per_head = {
        "query_routing": query_pattern[:, :, :, None] * mixed_value.permute(0, 2, 1, 3),
        "key_routing": key_pattern[:, :, :, None] * mixed_value.permute(0, 2, 1, 3),
        "current_value": pattern[:, :, :, None] * ((1 - block.attn.lamb) * dcurrent).permute(0, 2, 1, 3),
        "value_bus": pattern[:, :, :, None] * (block.attn.lamb * dbus).permute(0, 2, 1, 3),
    }
    factors = {name: project_heads(block, value).float() for name, value in factors_per_head.items()}
    return {
        "factors": factors,
        "source": dsource.float(),
        "scores": score.float(),
        "score_derivative": dscores.float(),
        "pattern": pattern.float(),
        "pattern_derivative": dpattern.float(),
        "raw_query": raw_query.float(),
        "raw_key": raw_key.float(),
        "raw_query_derivative": draw_query.float(),
        "raw_key_derivative": draw_key.float(),
        "query_derivative": dquery.float(),
        "key_derivative": dkey.float(),
    }


def causal_stats(base_logits, target_delta, prediction_delta):
    target, _, cross, error = fisher_terms(base_logits, target_delta, prediction_delta)
    denominator = target.sum().clamp_min(1e-30)
    return {"fidelity": float(1 - error.sum() / denominator), "projection": float(cross.sum() / denominator)}


def main():
    red = json.loads(CONFIG.read_text())
    config = json.loads(Path(red["source_config"]).read_text())
    prior = json.loads(Path(red["source_output"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    sources = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"] : spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    records = []
    locations = np.linspace(0, 1, red["dense_points"])
    weights = np.ones(red["dense_points"]) / (red["dense_points"] - 1)
    weights[[0, -1]] *= 0.5
    maximum_recomposition_error = 0.0

    for prior_record in prior["records"]:
        if prior_record["pair"] != red["pair"]:
            continue
        corpus = prior_record["corpus"]
        row_index, start = prior_record["row"], prior_record["start"]
        _, target_token = (int(value) for value in red["pair"].split("->"))
        raw = np.asarray(sources[corpus][row_index, start : start + config["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy()
        changed[config["edit_position"]] = target_token
        base_ids = torch.as_tensor(raw, device="cuda")[None]
        changed_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            base_rows, _, _ = trace(model, base_ids)
            changed_rows, _, _ = trace(model, changed_ids)
            base_x0 = F.rms_norm(model.embedding(base_ids), (model.config.n_embd,))
            changed_x0 = F.rms_norm(model.embedding(changed_ids), (model.config.n_embd,))
            tangent = changed_x0 - base_x0
        with torch.enable_grad():
            path = [factor_row(model, base_x0, tangent, float(location), red["layer"], config["prediction_position"]) for location in locations]

        source_norms = np.asarray([float(row["source"].norm()) for row in path])
        peak_index = int(source_norms.argmax())
        peak = path[peak_index]
        for row in path:
            recomposed = sum(row["factors"].values()).sum(1)
            maximum_recomposition_error = max(maximum_recomposition_error, float((recomposed - row["source"]).norm() / row["source"].norm().clamp_min(1e-30)))
        integrated = {factor: sum(float(weight) * row["factors"][factor] for row, weight in zip(path, weights)) for factor in FACTORS}
        integrated_sum = sum(integrated.values())
        block = model.blocks[red["layer"]]
        base = base_rows[red["layer"]]
        changed_row = changed_rows[red["layer"]]
        full_write = base["details"]["write"].clone()
        full_write[:, config["prediction_position"]] = changed_row["details"]["write"][:, config["prediction_position"]]
        base_logits, target_logits = logits_from_attention_writes(model, red["layer"], base, torch.stack((base["details"]["write"], full_write)))
        target_delta = target_logits - base_logits
        worlds = []
        names = list(FACTORS) + ["sum"]
        for name in names:
            delta = integrated[name] if name in integrated else integrated_sum
            write = base["details"]["write"].clone()
            write[:, config["prediction_position"]] += delta.sum((1, 2)).to(write.dtype) if delta.ndim == 4 else delta.sum(1).to(write.dtype)
            worlds.append(write)
        logits = logits_from_attention_writes(model, red["layer"], base, torch.stack(worlds))
        causal = {name: causal_stats(base_logits, target_delta, logits[index] - base_logits) for index, name in enumerate(names)}

        routing_heads = peak["factors"]["query_routing"] + peak["factors"]["key_routing"]
        head_norms = routing_heads.flatten(2).norm(dim=-1)[0]
        dominant_head = int(head_norms.argmax())
        source_norms_in_head = routing_heads[0, dominant_head].norm(dim=-1)
        dominant_source = int(source_norms_in_head.argmax())
        dominant_head_share = float(head_norms.max() / head_norms.sum().clamp_min(1e-30))
        dominant_source_share = float(source_norms_in_head.max() / source_norms_in_head.sum().clamp_min(1e-30))
        query_norm = float(peak["factors"]["query_routing"].norm())
        key_norm = float(peak["factors"]["key_routing"].norm())
        current_norm = float(peak["factors"]["current_value"].norm())
        bus_norm = float(peak["factors"]["value_bus"].norm())
        score_path = np.asarray([float(row["scores"][0, dominant_head, dominant_source]) for row in path])
        signs = np.sign(score_path)
        crossings = np.where(signs[:-1] * signs[1:] <= 0)[0]
        crossing_locations = [float((locations[index] + locations[index + 1]) / 2) for index in crossings]
        near_crossing = min((abs(location - locations[peak_index]) for location in crossing_locations), default=None)
        factor_norm_paths = {factor: [float(row["factors"][factor].norm()) for row in path] for factor in FACTORS}
        records.append({
            "corpus": corpus,
            "peak_location": float(locations[peak_index]),
            "query_routing_share": query_norm / max(query_norm + key_norm, 1e-30),
            "key_routing_share": key_norm / max(query_norm + key_norm, 1e-30),
            "current_value_share": current_norm / max(current_norm + bus_norm, 1e-30),
            "value_bus_share": bus_norm / max(current_norm + bus_norm, 1e-30),
            "dominant_head": dominant_head,
            "dominant_head_share": dominant_head_share,
            "dominant_source": dominant_source,
            "dominant_source_share": dominant_source_share,
            "dominant_score_crossings": crossing_locations,
            "nearest_crossing_to_peak": near_crossing,
            "causal": causal,
            "paths": {"locations": locations.tolist(), "source": source_norms.tolist(), **factor_norm_paths, "dominant_score": score_path.tolist()},
        })
        print(corpus, flush=True)

    hypotheses = {
        "Q1_qk_and_value_factors_recompose_the_source_derivative": {"supported": bool(maximum_recomposition_error <= 1e-5), "maximum_relative_error": maximum_recomposition_error},
        "Q2_query_routing_dominates_the_sharp_peak": {"supported": all(row["query_routing_share"] >= 0.60 for row in records), "shares": {row["corpus"]: row["query_routing_share"] for row in records}},
        "Q3_one_native_head_dominates_peak_routing": {"supported": all(row["dominant_head_share"] >= 0.50 for row in records), "shares": {row["corpus"]: row["dominant_head_share"] for row in records}},
        "Q4_one_source_edge_dominates_within_the_top_head": {"supported": all(row["dominant_source_share"] >= 0.30 for row in records), "shares": {row["corpus"]: row["dominant_source_share"] for row in records}},
        "Q5_the_same_head_recurs_across_corpora": {"supported": len({row["dominant_head"] for row in records}) == 1, "heads": {row["corpus"]: row["dominant_head"] for row in records}},
        "Q6_the_same_source_position_recurs_across_corpora": {"supported": len({row["dominant_source"] for row in records}) == 1, "sources": {row["corpus"]: row["dominant_source"] for row in records}},
        "Q7_persistent_bus_values_dominate_current_values": {"supported": all(row["value_bus_share"] > row["current_value_share"] for row in records), "bus_shares": {row["corpus"]: row["value_bus_share"] for row in records}},
        "Q8_the_dominant_qk_score_crosses_zero_near_the_peak": {"supported": all(row["nearest_crossing_to_peak"] is not None and row["nearest_crossing_to_peak"] <= 0.10 for row in records), "distances": {row["corpus"]: row["nearest_crossing_to_peak"] for row in records}},
        "Q9_query_routing_has_a_large_integrated_causal_role": {"supported": all(abs(row["causal"]["query_routing"]["projection"]) >= 0.50 for row in records), "projections": {row["corpus"]: row["causal"]["query_routing"]["projection"] for row in records}},
        "Q10_dense_qkov_factors_recover_the_finite_attention_effect": {"supported": all(row["causal"]["sum"]["fidelity"] >= 0.999 for row in records), "fidelities": {row["corpus"]: row["causal"]["sum"]["fidelity"] for row in records}},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "records": records}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for row in records:
        axes[0].plot(row["paths"]["locations"], row["paths"]["source"], label=row["corpus"])
    axes[0].set_title("Layer-1 source-derivative norm")
    axes[0].set_xlabel("path location")
    for factor in FACTORS:
        axes[1].plot(records[0]["paths"]["locations"], records[0]["paths"][factor], label=factor)
    axes[1].set_title(f"FineWeb factor norms")
    axes[1].set_xlabel("path location")
    for row in records:
        axes[2].plot(row["paths"]["locations"], row["paths"]["dominant_score"], label=f"{row['corpus']} H{row['dominant_head']} S{row['dominant_source']}")
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_title("Dominant raw QK score")
    axes[2].set_xlabel("path location")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# QKOV path factorization of the layer-1 his→her transition",
        "",
        "The sharp layer-1 source derivative is decomposed analytically into final-query routing, per-source key routing, current-layer values, and the persistent layer-0 value bus. Squared-score normalization derivatives are included exactly. Each factor is integrated over the dense token path and inserted into the native layer-1 attention write before the nonlinear suffix is recomputed.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-1 QKOV path](bilin12_his_her_layer1_qkov_path.png)",
        "",
        "| Corpus | Peak | Q/K routing share | Current/bus value share | Dominant head/share | Dominant source/share | Q / K / current / bus projection | Sum fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        projections = " / ".join(f"{row['causal'][factor]['projection']:.3f}" for factor in FACTORS)
        lines.append(f"| {row['corpus']} | {row['peak_location']:.3f} | {row['query_routing_share']:.3f}/{row['key_routing_share']:.3f} | {row['current_value_share']:.3f}/{row['value_bus_share']:.3f} | H{row['dominant_head']} / {row['dominant_head_share']:.3f} | S{row['dominant_source']} / {row['dominant_source_share']:.3f} | {projections} | {row['causal']['sum']['fidelity']:.3f} |")
    lines += [
        "",
        "The decomposition is path-local: it explains how this finite replacement traverses native QKOV factors. Stable head recurrence would support a reusable operator; changing dominant sources would instead support context-dependent instantiation of that operator.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
