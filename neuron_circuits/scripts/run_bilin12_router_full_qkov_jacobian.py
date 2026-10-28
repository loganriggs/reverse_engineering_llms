#!/usr/bin/env python3
"""Manual weight-derived QK/value/MLP Jacobian of the Layer-6 route."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_analytic_mlp_jacobian import boundary
from run_bilin12_router_finite_transport import edge_values, fidelity
from run_bilin12_router_transport_operator_decomposition import base_suffix_cache, batch_worlds
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_full_qkov_jacobian.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_full_qkov_jacobian.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_FULL_QKOV_JACOBIAN.md"
MODES = {"mlp_only": (False, False), "qk_only": (True, False),
         "value_only": (False, True), "full": (True, True)}


def norm(x): return F.rms_norm(x, (x.shape[-1],))


def rms_jvp(x, dx):
    rms = x.square().mean(-1, keepdim=True).sqrt()
    cross = (x[:, None] * dx).mean(-1, keepdim=True)
    return dx / rms[:, None] - x[:, None] * cross / rms[:, None].pow(3)


def rotary(x, cos, sin):
    half = x.shape[-1] // 2; first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos + second * sin, first * (-sin) + second * cos], -1).type_as(x)


@torch.inference_mode()
def attention_jvp(model, block, state, dstate, bus, use_qk, use_value):
    attention = block.attn; batch, length, dimension = state.shape
    heads, head_dimension = attention.n_head, attention.head_dim
    q = attention.c_q(state).view(batch, length, heads, head_dimension)
    k = attention.c_k(state).view(batch, length, heads, head_dimension)
    v_native = attention.c_v(state).view(batch, length, heads, head_dimension)
    dq = torch.einsum("bntd,ed->bnte", dstate, attention.c_q.weight.float()).reshape(batch, -1, length, heads, head_dimension)
    dk = torch.einsum("bntd,ed->bnte", dstate, attention.c_k.weight.float()).reshape_as(dq)
    dv = torch.einsum("bntd,ed->bnte", dstate, attention.c_v.weight.float()).reshape_as(dq)
    if bus is None: raise RuntimeError("Layer-6 suffix must have the persistent Layer-0 value bus")
    value = (1 - attention.lamb) * v_native + attention.lamb * bus.view_as(v_native)
    dvalue = (1 - attention.lamb) * dv
    cos, sin = attention.rotary(q)
    dq = rms_jvp(q, dq); dk = rms_jvp(k, dk)
    q = norm(q); k = norm(k)
    q = rotary(q, cos, sin); k = rotary(k, cos, sin)
    tangent_cos, tangent_sin = cos[:, None], sin[:, None]
    dq = rotary(dq, tangent_cos, tangent_sin); dk = rotary(dk, tangent_cos, tangent_sin)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k)
    dscores = torch.einsum("bnqhd,bkhd->bnhqk", dq, k) + torch.einsum("bqhd,bnkhd->bnhqk", q, dk)
    raw = (scores / head_dimension).square()
    draw = 2 * scores[:, None] * dscores / (head_dimension ** 2)
    mask = torch.tril(torch.ones(length, length, device=state.device, dtype=torch.bool))
    raw = raw.masked_fill(~mask, 0); draw = draw.masked_fill(~mask[None, None], 0)
    denominator = raw.sum(-1, keepdim=True).clamp_min(1e-9)
    ddenominator = draw.sum(-1, keepdim=True)
    pattern = raw / denominator
    dpattern = (draw * denominator[:, None] - raw[:, None] * ddenominator) / denominator[:, None].square()
    z = torch.einsum("bhqk,bkhd->bqhd", pattern, value)
    dz = torch.zeros(batch, dstate.shape[1], length, heads, head_dimension, device=state.device)
    if use_qk: dz += torch.einsum("bnhqk,bkhd->bnqhd", dpattern, value)
    if use_value: dz += torch.einsum("bhqk,bnkhd->bnqhd", pattern, dvalue)
    output = z.reshape(batch, length, dimension) @ attention.c_proj.weight.float().T
    doutput = dz.reshape(batch, dstate.shape[1], length, dimension) @ attention.c_proj.weight.float().T
    native, _ = attention(state, bus)
    return output, doutput, float((output - native).abs().max())


@torch.inference_mode()
def analytic_position(model, ids, position, config, axes, use_qk, use_value):
    x, x0, bus = boundary(model, ids, config["layer"]); current = x
    dx = config["scale"] * x[:, position].square().mean(-1).sqrt()[:, None, None] * axes.T[None]
    dx_sequence = torch.zeros(len(ids), 32, x.shape[1], x.shape[2], device="cuda")
    dx_sequence[:, :, position] = dx; dx = dx_sequence
    maximum_trace_error = 0.0
    for index in range(config["layer"] + 1, 12):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        dmixed = block.lambdas[0] * dx
        state_in = norm(mixed); dstate_in = rms_jvp(mixed, dmixed)
        attention, dattention, error = attention_jvp(model, block, state_in, dstate_in, bus, use_qk, use_value)
        maximum_trace_error = max(maximum_trace_error, error)
        pre = mixed + attention; dpre = dmixed + dattention
        state = norm(pre); dstate = rms_jvp(pre, dpre)
        if index == 11: return dstate[:, :, position] @ axes, state, maximum_trace_error
        left = block.mlp.Left(state); right = block.mlp.Right(state)
        dleft = torch.einsum("bntd,ed->bnte", dstate, block.mlp.Left.weight.float())
        dright = torch.einsum("bntd,ed->bnte", dstate, block.mlp.Right.weight.float())
        dwrite = (dleft * right[:, None] + left[:, None] * dright) @ block.mlp.Down.weight.float().T
        write = block.mlp.Down(left * right) + block.mlp.Down_bias
        current = pre + write; dx = dpre + dwrite
    raise RuntimeError


def quadratic(values, forms):
    flat = values.reshape(-1, 32)
    return torch.einsum("ni,kij,nj->nk", flat, forms, flat).reshape(*values.shape[:-1], -1)


def projection(target, value): return float((target * value).sum() / target.square().sum().clamp_min(1e-30))


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); graph = json.loads(Path(config["selected_graph"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": 32, "discovery": config["basis_discovery"]})
    block = model.blocks[11]; channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    natural, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False); natural_basis = vh[:32].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms, natural_basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (natural_basis @ rotation.detach()).float()
    forms = (rotation.T[None] @ restricted @ rotation[None]).cpu(); pairs = [tuple(pair) for pair in graph["selected_pairs"]]
    results = {}; maximum_trace_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True); stores = {name: [] for name in MODES}; finite = []; finite_mlp = []; bases = []; source = load_rows(spec)
        for start in range(0, len(source), config["batch_size"]):
            ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
            worlds = batch_worlds(model, config["layer"], ids, config["positions"], axes, config["scale"])
            finite.append(worlds["full"].reshape(-1, 64, 32)); finite_mlp.append(worlds["mlp_only"].reshape(-1, 64, 32))
            x, x0, bus = boundary(model, ids, config["layer"]); final, _ = base_suffix_cache(model, config["layer"], x, x0, bus)
            for position in config["positions"]:
                bases.append((final[:, position] @ axes).cpu())
                for name, (use_qk, use_value) in MODES.items():
                    tangent, manual_final, error = analytic_position(model, ids, position, config, axes, use_qk, use_value)
                    maximum_trace_error = max(maximum_trace_error, error, float((manual_final - final).abs().max()))
                    stores[name].append(tangent.cpu())
        target = torch.cat(finite); target_mlp = torch.cat(finite_mlp); base = torch.cat(bases)[:, None].expand(-1, 64, -1)
        predictions = {}
        for name, values in stores.items():
            tangent = torch.cat(values); predictions[name] = torch.cat([tangent, -tangent], 1)
        full = predictions["full"]; mlp_only = predictions["mlp_only"]
        qk_component = .5 * ((predictions["qk_only"] - mlp_only) + (full - predictions["value_only"]))
        value_component = .5 * ((predictions["value_only"] - mlp_only) + (full - predictions["qk_only"]))
        interaction = full - predictions["qk_only"] - predictions["value_only"] + mlp_only
        edge_target = edge_values(base + target, pairs) - edge_values(base, pairs)
        edge_prediction = edge_values(base + full, pairs) - edge_values(base, pairs)
        coefficient_target = quadratic(base + target, forms) - quadratic(base, forms)
        coefficient_prediction = quadratic(base + full, forms) - quadratic(base, forms)
        results[corpus] = {
            "full_vs_native": {"coordinate_fidelity": fidelity(target, full),
                               "edge_change_fidelity": fidelity(edge_target, edge_prediction),
                               "coefficient_change_fidelity": fidelity(coefficient_target, coefficient_prediction)},
            "mlp_analytic_vs_native_coordinate_fidelity": fidelity(target, mlp_only),
            "mlp_analytic_vs_finite_mlp_coordinate_fidelity": fidelity(target_mlp, mlp_only),
            "target_projection": {"mlp_residual": projection(full, mlp_only),
                                  "qk": projection(full, qk_component), "value": projection(full, value_component)},
            "component_norm_ratio": {"mlp_residual": float(mlp_only.norm() / full.norm()),
                                     "qk": float(qk_component.norm() / full.norm()),
                                     "value": float(value_component.norm() / full.norm())},
            "qk_value_interaction_norm_ratio": float(interaction.norm() / full.norm()),
        }
    q0 = maximum_trace_error <= 1e-5
    q1 = all(row["full_vs_native"]["coordinate_fidelity"] >= .99 for row in results.values())
    q2 = all(row["full_vs_native"]["coordinate_fidelity"] >= row["mlp_analytic_vs_native_coordinate_fidelity"] + .01 for row in results.values())
    q3 = all(row["full_vs_native"]["edge_change_fidelity"] >= .99 and row["full_vs_native"]["coefficient_change_fidelity"] >= .99 for row in results.values())
    q4 = all(abs(row["target_projection"]["qk"]) >= .01 for row in results.values())
    q5 = all(abs(row["target_projection"]["value"]) >= .01 for row in results.values())
    q6 = all(row["qk_value_interaction_norm_ratio"] >= .01 for row in results.values())
    fineweb = results["fineweb"]["target_projection"]
    q7 = all(max(abs(row["target_projection"][key] - fineweb[key]) for key in fineweb) <= .05
             for corpus, row in results.items() if corpus != "fineweb")
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "Q0_manual_attention_trace_is_exact": q0, "Q1_full_weight_jacobian_predicts_native_edits": q1,
        "Q2_full_jacobian_closes_the_mlp_only_gap": q2, "Q3_full_jacobian_predicts_sparse_grammar": q3,
        "Q4_qk_response_is_material": q4, "Q5_value_response_is_material": q5,
        "Q6_qk_value_interaction_is_material": q6, "Q7_qkov_profile_transfers_ood": q7}.items()}
    hypotheses["Q0_manual_attention_trace_is_exact"]["maximum_error"] = maximum_trace_error
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Full weight-derived QK/value/MLP Jacobian", "",
             "The normalized squared-attention derivative is implemented explicitly from Q, K, V, rotary, pattern normalization, persistent Layer-0 value mixing, and OV projection. It is composed with the exact RMSNorm and bilinear-MLP derivatives from Layers 7–11. QK-pattern response and value/OV response can be independently enabled, giving an exact two-factor derivative Shapley decomposition.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Full native coord / edge / coeff | MLP-only native | MLP/QK/V projection | QK-V interaction |", "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        full = row["full_vs_native"]; projection_row = row["target_projection"]
        lines.append(f"| {corpus} | {full['coordinate_fidelity']:.4f} / {full['edge_change_fidelity']:.4f} / {full['coefficient_change_fidelity']:.4f} | "
                     f"{row['mlp_analytic_vs_native_coordinate_fidelity']:.4f} | {projection_row['mlp_residual']:.3f}/{projection_row['qk']:.3f}/{projection_row['value']:.3f} | "
                     f"{row['qk_value_interaction_norm_ratio']:.3f} |")
    lines += ["", "The projections sum to one but are signed and may hide cancellation; norm ratios and the explicit interaction are reported alongside them. This closes a local causal route, not the global task-level explanation for why a text context selects a particular Jacobian."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
