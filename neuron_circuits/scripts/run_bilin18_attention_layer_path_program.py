#!/usr/bin/env python3
"""Exact layer-subset paths for the 546M model's two-QK attention response."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_mlp_path_symmetry_redteam import GROUPS as MLP_GROUPS, collapse
from run_bilin12_mlp_path_type_program import (
    aggregate_gram, boundary, greedy_order, norm, oracle_fidelity, path_fidelity,
    path_matrices, rms_jvp,
)
from run_bilin12_router_finite_transport import fidelity
from run_bilin18_mlp_path_attention_boundary import dynamic_finite
from run_bilin18_mlp_path_symmetry_replication import build_axes, load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_attention_layer_path_program.json")
OUTPUT = TENSOR_ROOT / "bilin18_attention_layer_path_program.json"
REPORT = TENSOR_ROOT / "BILIN18_ATTENTION_LAYER_PATH_PROGRAM.md"
ATTENTION_PATHS = tuple(itertools.product(("I", "A"), repeat=5))


def sequence_rms_jvp(x, tangents):
    root = x.square().mean(-1, keepdim=True).sqrt()
    coupling = (x[:, None] * tangents).mean(-1, keepdim=True)
    return tangents / root[:, None] - x[:, None] * coupling / root[:, None].pow(3)


def rotate(values, cosine, sine):
    half = values.shape[-1] // 2
    first, second = values[..., :half], values[..., half:]
    return torch.cat([first * cosine + second * sine,
                      first * (-sine) + second * cosine], -1)


@torch.inference_mode()
def two_qk_attention_jvp(attention, state, bus, tangents, return_components=False):
    """Full-sequence exact JVP; the persistent Layer-0 value bus is fixed."""
    batch, length, dimension = state.shape; heads, head = attention.n_head, attention.head_dim
    q1_raw = F.linear(state, attention.c_q.weight.float()).view(batch, length, heads, head)
    k1_raw = F.linear(state, attention.c_k.weight.float()).view(batch, length, heads, head)
    q2_raw = F.linear(state, attention.c_q2.weight.float()).view(batch, length, heads, head)
    k2_raw = F.linear(state, attention.c_k2.weight.float()).view(batch, length, heads, head)
    current_v = F.linear(state, attention.c_v.weight.float()).view(batch, length, heads, head)
    value = (1 - attention.lamb.float()) * current_v + attention.lamb.float() * bus.view_as(current_v).float()
    cosine, sine = attention.rotary(q1_raw)
    cosine = cosine.float(); sine = sine.float()
    q1 = rotate(F.rms_norm(q1_raw, (head,)), cosine, sine)
    k1 = rotate(F.rms_norm(k1_raw, (head,)), cosine, sine)
    q2 = rotate(F.rms_norm(q2_raw, (head,)), cosine, sine)
    k2 = rotate(F.rms_norm(k2_raw, (head,)), cosine, sine)

    dq1_raw = F.linear(tangents, attention.c_q.weight.float()).view(batch, -1, length, heads, head)
    dk1_raw = F.linear(tangents, attention.c_k.weight.float()).view_as(dq1_raw)
    dq2_raw = F.linear(tangents, attention.c_q2.weight.float()).view_as(dq1_raw)
    dk2_raw = F.linear(tangents, attention.c_k2.weight.float()).view_as(dq1_raw)
    dv = (1 - attention.lamb.float()) * F.linear(
        tangents, attention.c_v.weight.float()
    ).view_as(dq1_raw)
    dq1 = rotate(sequence_rms_jvp(q1_raw, dq1_raw), cosine[:, None], sine[:, None])
    dk1 = rotate(sequence_rms_jvp(k1_raw, dk1_raw), cosine[:, None], sine[:, None])
    dq2 = rotate(sequence_rms_jvp(q2_raw, dq2_raw), cosine[:, None], sine[:, None])
    dk2 = rotate(sequence_rms_jvp(k2_raw, dk2_raw), cosine[:, None], sine[:, None])

    score1 = torch.einsum("bqhd,bkhd->bhqk", q1, k1)
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2)
    dscore1 = (torch.einsum("bnqhd,bkhd->bnhqk", dq1, k1)
               + torch.einsum("bqhd,bnkhd->bnhqk", q1, dk1))
    dscore2 = (torch.einsum("bnqhd,bkhd->bnhqk", dq2, k2)
               + torch.einsum("bqhd,bnkhd->bnhqk", q2, dk2))
    mask = torch.tril(torch.ones(length, length, device=state.device, dtype=torch.bool))
    pattern = (score1 * score2 / (head ** 2)).masked_fill(~mask[None, None], 0)
    dpattern = ((dscore1 * score2[:, None] + score1[:, None] * dscore2) / (head ** 2))
    dpattern = dpattern.masked_fill(~mask[None, None, None], 0)
    native_heads = torch.einsum("bhqk,bkhd->bqhd", pattern, value)
    # Apply the causal mask independently to the factor components.
    qk1_heads = torch.einsum(
        "bnhqk,bkhd->bnqhd",
        (dscore1 * score2[:, None] / (head ** 2)).masked_fill(~mask[None, None, None], 0), value)
    qk2_heads = torch.einsum(
        "bnhqk,bkhd->bnqhd",
        (score1[:, None] * dscore2 / (head ** 2)).masked_fill(~mask[None, None, None], 0), value)
    value_heads = torch.einsum("bhqk,bnkhd->bnqhd", pattern, dv)
    tangent_heads = qk1_heads + qk2_heads + value_heads
    native = F.linear(native_heads.reshape(batch, length, dimension), attention.c_proj.weight.float())
    tangent = F.linear(tangent_heads.reshape(batch, -1, length, dimension), attention.c_proj.weight.float())
    reference, _ = attention(state, bus)
    forward_error = float((native - reference.float()).norm() / reference.float().norm().clamp_min(1e-30))
    if not return_components:
        return tangent, forward_error
    project = lambda heads_value: F.linear(
        heads_value.reshape(batch, -1, length, dimension), attention.c_proj.weight.float())
    return tangent, forward_error, {
        "qk1": project(qk1_heads), "qk2": project(qk2_heads),
        "qk": project(qk1_heads + qk2_heads), "value": project(value_heads),
    }


def mlp_post_jvp(block, pre, state, tangents):
    normalized = sequence_rms_jvp(pre, tangents)
    left = F.linear(normalized, block.mlp.Left.weight.float())
    right = F.linear(normalized, block.mlp.Right.weight.float())
    native_left = block.mlp.Left(state).float()
    native_right = block.mlp.Right(state).float()
    dwrite = F.linear(left * native_right[:, None] + native_left[:, None] * right,
                      block.mlp.Down.weight.float())
    return tangents + dwrite


@torch.inference_mode()
def native_cache(model, ids, config):
    x, x0, bus = boundary(model, ids, config["layer"]); current = x; rows = []
    for index in range(config["layer"] + 1, config["final_layer"] + 1):
        block = model.blocks[index]; mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        attention_input = norm(mixed); bus_before = bus
        attention, bus = block.attn(attention_input, bus)
        pre = mixed + attention; state = norm(pre)
        rows.append({"index": index, "mixed": mixed, "attention_input": attention_input,
                     "bus": bus_before, "pre": pre, "state": state})
        if index < config["final_layer"]:
            current = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    return x, rows


@torch.inference_mode()
def attention_paths(model, ids, position, config, input_axes, output_axes):
    x, cache = native_cache(model, ids, config); batch, length, dimension = x.shape
    rank_in = input_axes.shape[1]; rank_out = output_axes.shape[1]; chunks = []
    maximum_forward_error = 0.0; maximum_unexpanded_error = 0.0
    for first in range(0, rank_in, config["tangent_chunk"]):
        axes = input_axes[:, first:first + config["tangent_chunk"]]
        tangent = torch.zeros(batch, axes.shape[1], length, dimension, device="cuda")
        tangent[:, :, position] = (config["scale"] * x[:, position].square().mean(-1).sqrt()[:, None, None]
                                    * axes.T[None])
        paths = tangent[:, None]
        unexpanded = tangent
        for row in cache:
            index = row["index"]; block = model.blocks[index]
            # Expanded paths: flatten path and input-axis dimensions for one exact local operator call.
            flat = paths.reshape(batch, -1, length, dimension)
            mixed = block.lambdas[0].float() * flat
            attention_input_tangent = sequence_rms_jvp(row["mixed"].float(), mixed)
            attention_tangent, error = two_qk_attention_jvp(
                block.attn, row["attention_input"].float(), row["bus"], attention_input_tangent
            )
            maximum_forward_error = max(maximum_forward_error, error)
            if index < config["final_layer"]:
                base = mlp_post_jvp(block, row["pre"].float(), row["state"].float(), mixed)
                routed = mlp_post_jvp(block, row["pre"].float(), row["state"].float(), attention_tangent)
                base = base.reshape(batch, paths.shape[1], axes.shape[1], length, dimension)
                routed = routed.reshape_as(base)
                paths = torch.stack([base, routed], 2).reshape(
                    batch, paths.shape[1] * 2, axes.shape[1], length, dimension
                )
            else:
                base = sequence_rms_jvp(row["pre"].float(), mixed)[:, :, position] @ output_axes
                routed = sequence_rms_jvp(row["pre"].float(), attention_tangent)[:, :, position] @ output_axes
                base = base.reshape(batch, paths.shape[1], axes.shape[1], rank_out)
                routed = routed.reshape_as(base)
                paths = torch.stack([base, routed], 2).reshape(
                    batch, paths.shape[1] * 2, axes.shape[1], rank_out
                )

            # Independent unexpanded recurrence.
            umixed = block.lambdas[0].float() * unexpanded
            uinput = sequence_rms_jvp(row["mixed"].float(), umixed)
            uattention, error = two_qk_attention_jvp(
                block.attn, row["attention_input"].float(), row["bus"], uinput
            )
            maximum_forward_error = max(maximum_forward_error, error)
            upre = umixed + uattention
            if index < config["final_layer"]:
                unexpanded = mlp_post_jvp(block, row["pre"].float(), row["state"].float(), upre)
            else:
                unexpanded = sequence_rms_jvp(row["pre"].float(), upre)[:, :, position] @ output_axes
        total = paths.sum(1)
        relative = ((total - unexpanded).flatten(1).norm(dim=1)
                    / unexpanded.flatten(1).norm(dim=1).clamp_min(1e-30))
        maximum_unexpanded_error = max(maximum_unexpanded_error, float(relative.max()))
        chunks.append(paths.cpu())
    return torch.cat(chunks, 2), maximum_forward_error, maximum_unexpanded_error


def identity_random_summary(paths, budget, replicates, generator):
    values = []
    for _ in range(replicates):
        if budget == 1:
            selected = [0]
        else:
            random = torch.randperm(len(ATTENTION_PATHS) - 1, generator=generator)[:budget - 1] + 1
            selected = [0] + random.tolist()
        values.append(path_fidelity(paths, selected))
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "maximum": max(values)}


@torch.inference_mode()
def collect(model, input_axes, output_axes, spec, config, ordering=None, mlp_order=None, finite=False):
    source = load_rows(spec); path_rows = []; finite_rows = []; mlp4_rows = []
    maximum_forward_error = 0.0; maximum_path_error = 0.0
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        for position in config["positions"]:
            paths, forward_error, path_error = attention_paths(
                model, ids, position, config, input_axes, output_axes
            )
            path_rows.append(paths); maximum_forward_error = max(maximum_forward_error, forward_error)
            maximum_path_error = max(maximum_path_error, path_error)
            if mlp_order is not None:
                raw_mlp = path_matrices(model, ids, position, config, input_axes, output_axes)
                grouped = collapse(raw_mlp.cpu())
                mlp4_rows.append(grouped[:, mlp_order[:4]].sum(1))
            if finite:
                finite_rows.append(dynamic_finite(
                    model, ids, position, config, input_axes, output_axes
                ).cpu())
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    return {"paths": torch.cat(path_rows),
            "finite": torch.cat(finite_rows) if finite_rows else None,
            "mlp4": torch.cat(mlp4_rows) if mlp4_rows else None,
            "maximum_forward_error": maximum_forward_error,
            "maximum_path_error": maximum_path_error}


@torch.inference_mode(False)
def autograd_jvp_check(model, ids, config, input_axes):
    # One real block, one natural tangent; independent torch.func.jvp validation.
    _, cache = native_cache(model, ids, config); row = cache[0]; block = model.blocks[row["index"]]
    state = row["attention_input"].detach().clone()
    bus = row["bus"].detach().clone()
    tangent = torch.zeros_like(state)
    tangent[:, config["positions"][0]] = input_axes[:, 0]
    manual, _ = two_qk_attention_jvp(
        block.attn, state.float(), bus, tangent[:, None]
    )
    # The native Rotary module caches cos/sin tensors. The manual inference call
    # above creates inference tensors, so clear that observational cache before
    # the independent autograd trace constructs ordinary differentiable ones.
    block.attn.rotary.seq_len_cached = None
    block.attn.rotary.cos_cached = None
    block.attn.rotary.sin_cached = None
    def function(value):
        return block.attn(value, bus)[0]
    _, expected = torch.func.jvp(function, (state.float(),), (tangent,))
    return float((manual[:, 0] - expected).norm() / expected.norm().clamp_min(1e-30))


@torch.inference_mode(False)
def main():
    raw_config = json.loads(CONFIG.read_text()); base = json.loads(Path(raw_config["source_config"]).read_text())
    config = {**base, **{key: value for key, value in raw_config.items()
                        if key not in ("source_config", "source_artifact", "hypotheses")},
              "hypotheses": raw_config["hypotheses"]}
    mlp_artifact = json.loads(Path(raw_config["source_artifact"]).read_text())
    mlp_order = [MLP_GROUPS.index(tuple(label)) for label in mlp_artifact["group_order"]]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    input_axes, output_axes = build_axes(model, config)
    source = load_rows(config["path_discovery"])
    check_ids = torch.as_tensor(np.asarray(source[:1, :128], dtype=np.int64), device="cuda")
    autograd_error = autograd_jvp_check(model, check_ids, config, input_axes)
    print("collect discovery", flush=True)
    discovery = collect(model, input_axes, output_axes, config["path_discovery"], config)
    ordering = greedy_order(*aggregate_gram(discovery["paths"]))
    generator = torch.Generator().manual_seed(config["seed"])
    maximum_forward_error = discovery["maximum_forward_error"]
    maximum_path_error = discovery["maximum_path_error"]; results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = collect(model, input_axes, output_axes, spec, config, ordering, mlp_order, finite=True)
        maximum_forward_error = max(maximum_forward_error, data["maximum_forward_error"])
        maximum_path_error = max(maximum_path_error, data["maximum_path_error"])
        paths = data["paths"]; finite_target = data["finite"]; rows = {}
        for budget in config["budgets"]:
            selected = ordering[:budget]; analytic = paths[:, selected].sum(1)
            nonidentity = [index for index in selected if index != 0]
            combined = data["mlp4"] + (paths[:, nonidentity].sum(1) if nonidentity else 0)
            rows[str(budget)] = {
                "frozen_fidelity": path_fidelity(paths, selected),
                "oracle_fidelity": oracle_fidelity(paths, budget),
                "random": identity_random_summary(paths, budget, config["random_replicates"], generator),
                "analytic_to_finite_fidelity": fidelity(finite_target, analytic),
                "combined_mlp4_to_finite_fidelity": fidelity(finite_target, combined),
            }
        results[corpus] = {"sites": len(paths), "budgets": rows,
                           "identity_path_fidelity": path_fidelity(paths, [0]),
                           "full_analytic_to_finite_fidelity": fidelity(finite_target, paths.sum(1))}

    fineweb4 = results["fineweb"]["budgets"]["4"]["frozen_fidelity"]
    low_order_top4 = sum(ATTENTION_PATHS[index].count("A") <= 1 for index in ordering[:4])
    exact_error = max(maximum_forward_error, autograd_error)
    supported = {
        "T0_manual_two_qk_attention_jvp_is_exact": exact_error <= 1e-5,
        "T1_attention_layer_paths_recompose_exactly": maximum_path_error <= 1e-5,
        "T2_full_analytic_jvp_predicts_finite_effect": all(row["full_analytic_to_finite_fidelity"] >= .99 for row in results.values()),
        "T3_mlp_identity_path_explains_most_effect": all(row["identity_path_fidelity"] >= .94 for row in results.values()),
        "T4_four_attention_paths_are_high_fidelity": all(row["budgets"]["4"]["frozen_fidelity"] >= .98 for row in results.values()),
        "T5_compact_combined_program_predicts_native_effect": all(row["budgets"]["4"]["combined_mlp4_to_finite_fidelity"] >= .94 for row in results.values()),
        "T6_attention_paths_beat_random": all(row["budgets"]["4"]["frozen_fidelity"] >= row["budgets"]["4"]["random"]["p95"] + .01 for row in results.values()),
        "T7_attention_program_transfers_ood": all(results[name]["budgets"]["4"]["frozen_fidelity"] >= fineweb4 - .03 for name in ("wikitext", "tinystories")),
        "T8_attention_program_is_low_order": low_order_top4 >= 3,
        "T9_context_specific_attention_selection_is_small": all(row["budgets"]["4"]["oracle_fidelity"] < row["budgets"]["4"]["frozen_fidelity"] + .02 for row in results.values()),
    }
    hypotheses = {name: {"supported": bool(value)} for name, value in supported.items()}
    hypotheses["T0_manual_two_qk_attention_jvp_is_exact"].update(
        maximum_forward_error=maximum_forward_error, autograd_relative_error=autograd_error)
    hypotheses["T1_attention_layer_paths_recompose_exactly"]["maximum_relative_error"] = maximum_path_error
    payload = {"config": raw_config, "hypotheses": hypotheses,
               "path_order": ["".join(ATTENTION_PATHS[index]) for index in ordering],
               "top4_low_order_count": low_order_top4, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Exact two-QK attention layer paths in the 546M model", "",
             "The manual derivative includes both normalized rotary QK factors, their unnormalized product pattern, current-value response, the fixed persistent Layer-0 value bus, OV projection, RMSNorms, and every intervening bilinear MLP. Each of Layers 13–17 contributes either baseline transport `I` or its complete attention response `A`, giving 32 exact layer-subset paths. The top-four MLP program and attention ordering are frozen on separate FineWeb panels.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum native-forward/JVP error: `{exact_error:.2e}`. Maximum path recomposition error: `{maximum_path_error:.2e}`.",
              f"Top-four paths with attention degree ≤1: **{low_order_top4}/4**.", "",
              "Frozen order: `" + "`, `".join(payload["path_order"]) + "`.", "",
              "| Corpus | Identity | K4 analytic | K4 oracle/random-p95 | Full analytic→finite | MLP4 + K4 attention→finite |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        four = row["budgets"]["4"]
        lines.append(f"| {corpus} | {row['identity_path_fidelity']:.3f} | {four['frozen_fidelity']:.3f} | {four['oracle_fidelity']:.3f}/{four['random']['p95']:.3f} | {row['full_analytic_to_finite_fidelity']:.3f} | {four['combined_mlp4_to_finite_fidelity']:.3f} |")
    lines += ["", "An `A` label means the complete two-QK plus value attention derivative at that layer, followed by the native state-conditioned downstream maps. It is not a head, semantic feature, or fixed matrix. Further QK/value and head decomposition is justified only if this layer-level program is exact and transferably compact."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
