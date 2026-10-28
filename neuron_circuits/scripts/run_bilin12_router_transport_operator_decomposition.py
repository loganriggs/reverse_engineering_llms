#!/usr/bin/env python3
"""Exact attention/MLP hybrid decomposition of Layer-6 causal transport."""

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
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_transport_operator_decomposition.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_transport_operator_decomposition.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_TRANSPORT_OPERATOR_DECOMPOSITION.md"
WORLDS = {"residual_only": (False, False), "attention_only": (True, False),
          "mlp_only": (False, True), "full": (True, True)}


def norm(x): return F.rms_norm(x, (x.shape[-1],))


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


@torch.inference_mode()
def boundary(model, ids, layer):
    x0 = norm(model.embedding(ids)); x, bus = x0, None
    for index, block in enumerate(model.blocks):
        x, bus = block(x, bus, x0)
        if index == layer: return x, x0, bus
    raise RuntimeError


@torch.inference_mode()
def base_suffix_cache(model, layer, x, x0, bus):
    caches = []; current = x
    for index in range(layer + 1, 12):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        attention, bus = block.attn(norm(mixed), bus)
        pre = mixed + attention; state = norm(pre)
        if index == 11:
            caches.append((attention, None)); return state, caches
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        caches.append((attention, write)); current = pre + write
    raise RuntimeError


@torch.inference_mode()
def hybrid_suffix(model, layer, x, x0, bus, caches, base_batch, dynamic_attention, dynamic_mlp):
    current = x; events = len(x) // base_batch
    for offset, index in enumerate(range(layer + 1, 12)):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        if dynamic_attention:
            attention, bus = block.attn(norm(mixed), bus)
        else:
            attention = caches[offset][0][:, None].expand(base_batch, events, -1, -1).reshape_as(mixed)
        pre = mixed + attention; state = norm(pre)
        if index == 11: return state
        if dynamic_mlp:
            write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        else:
            write = caches[offset][1][:, None].expand(base_batch, events, -1, -1).reshape_as(pre)
        current = pre + write
    raise RuntimeError


@torch.inference_mode()
def batch_worlds(model, layer, ids, positions, axes, scale):
    x, x0, bus = boundary(model, ids, layer)
    base_final, caches = base_suffix_cache(model, layer, x, x0, bus)
    base_input = norm(x) @ axes; base_output = base_final @ axes
    bsz, dimension = len(ids), len(axes.T); directions = torch.eye(dimension, device="cuda")
    event_directions = torch.cat([directions, -directions]); events = len(event_directions)
    stores = {name: [] for name in WORLDS}; inputs = []
    for position in positions:
        xr = x[:, None].expand(bsz, events, *x.shape[1:]).clone().reshape(bsz * events, *x.shape[1:])
        x0r = x0[:, None].expand(bsz, events, *x0.shape[1:]).reshape(bsz * events, *x0.shape[1:])
        br = bus[:, None].expand(bsz, events, *bus.shape[1:]).reshape(bsz * events, *bus.shape[1:])
        native_direction = (event_directions @ axes.T)[None].expand(bsz, -1, -1).reshape(bsz * events, -1)
        rms = x[:, position].square().mean(-1).sqrt()[:, None].expand(bsz, events).reshape(-1, 1)
        xr[:, position] += scale * rms * native_direction
        input_delta = (norm(xr[:, position]) @ axes).reshape(bsz, events, -1) - base_input[:, position, None]
        inputs.append(input_delta.cpu())
        for name, (attention_on, mlp_on) in WORLDS.items():
            output = hybrid_suffix(model, layer, xr, x0r, br, caches, bsz, attention_on, mlp_on)
            delta = (output[:, position] @ axes).reshape(bsz, events, -1) - base_output[:, position, None]
            stores[name].append(delta.cpu())
    result = {name: torch.cat([value.reshape(-1, dimension) for value in rows]) for name, rows in stores.items()}
    result["input"] = torch.cat([value.reshape(-1, dimension) for value in inputs])
    return result


def central(values):
    shaped = values.reshape(-1, 64, 32)
    return (shaped[:, :32] - shaped[:, 32:]) / 2


def jacobians(input_value, output_value):
    x = input_value.reshape(-1, 64, 32); y = output_value.reshape(-1, 64, 32)
    gram = x.transpose(1, 2) @ x
    penalty = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
    return torch.linalg.solve(gram + penalty * torch.eye(32)[None], x.transpose(1, 2) @ y)


def projection(target, value):
    return float((target * value).sum() / target.square().sum().clamp_min(1e-30))


def finalize(acc):
    values = {key: torch.cat(rows) for key, rows in acc.items()}
    response = {name: central(values[name]) for name in WORLDS}; target = response["full"]
    residual = response["residual_only"]
    attention = .5 * ((response["attention_only"] - residual) + (target - response["mlp_only"]))
    mlp = .5 * ((response["mlp_only"] - residual) + (target - response["attention_only"]))
    interaction = target - response["attention_only"] - response["mlp_only"] + residual
    components = {"residual": residual, "attention": attention, "mlp": mlp}
    reconstruction = residual + attention + mlp
    full_norm = float(target.square().sum().sqrt())
    js = {name: jacobians(values["input"], values[name]) for name in WORLDS}
    variance = {name: float((value - value.mean(0)).square().sum()) for name, value in js.items()}
    return {
        "sites": len(target),
        "maximum_recomposition_error": float((target - reconstruction).abs().max()),
        "target_projection": {name: projection(target, value) for name, value in components.items()},
        "component_norm_ratio": {name: float(value.square().sum().sqrt()) / full_norm for name, value in components.items()},
        "interaction_norm_ratio": float(interaction.square().sum().sqrt()) / full_norm,
        "world_fidelity": {name: 1 - float((target - value).square().sum() / target.square().sum())
                           for name, value in response.items()},
        "jacobian_variance_ratio": {name: value / max(variance["full"], 1e-30) for name, value in variance.items()},
    }


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]; channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    natural, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False); basis = vh[:32].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", basis, forms, basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (basis @ rotation.detach()).float()
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True); acc = {"input": [], **{name: [] for name in WORLDS}}
        source = load_rows(spec)
        for start in range(0, len(source), config["batch_size"]):
            raw_ids = np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64)
            row = batch_worlds(model, config["layer"], torch.as_tensor(raw_ids, device="cuda"),
                               config["positions"], axes, config["scale"])
            for name, value in row.items(): acc[name].append(value)
        results[corpus] = finalize(acc)
    d0 = all(row["maximum_recomposition_error"] <= 1e-5 for row in results.values())
    d1 = all(row["target_projection"]["residual"] >= .50 for row in results.values())
    d2 = all(abs(row["target_projection"]["attention"]) >= .10 for row in results.values())
    d3 = all(abs(row["target_projection"]["mlp"]) >= .10 for row in results.values())
    d4 = all((1 - row["jacobian_variance_ratio"]["mlp_only"]) >=
             (1 - row["jacobian_variance_ratio"]["attention_only"]) + .25 for row in results.values())
    d5 = all(row["interaction_norm_ratio"] >= .10 for row in results.values())
    profile = lambda row: np.array([row["target_projection"][x] for x in ("residual", "attention", "mlp")])
    d6 = all(np.max(np.abs(profile(results[c]) - profile(results["fineweb"]))) <= .10
             for c in ("wikitext", "tinystories"))
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "D0_hybrid_recomposition_is_exact": d0,
        "D1_residual_path_is_major": d1,
        "D2_attention_response_is_material": d2,
        "D3_mlp_response_is_material": d3,
        "D4_attention_dominates_context_variation": d4,
        "D5_operator_interaction_is_material": d5,
        "D6_operator_profile_transfers_ood": d6,
    }.items()}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Which operators create context-dependent transport?", "",
             "For finite Layer-6 coordinate edits, every downstream attention response and every downstream bilinear-MLP response is independently either recomputed or frozen to its unperturbed value. Exact two-factor Shapley vectors divide the final coordinate response into the always-present residual route, attention response, and MLP response. This is a causal hybrid decomposition, not a probe.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Residual/attention/MLP projection | Interaction norm | Freeze-attn/freeze-MLP variance ratio | Residual-only fidelity |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        p = row["target_projection"]; v = row["jacobian_variance_ratio"]
        lines.append(f"| {corpus} | {p['residual']:.3f}/{p['attention']:.3f}/{p['mlp']:.3f} | "
                     f"{row['interaction_norm_ratio']:.3f} | {v['mlp_only']:.3f}/{v['attention_only']:.3f} | "
                     f"{row['world_fidelity']['residual_only']:.3f} |")
    lines += ["", "Here `freeze-attn` is the `mlp_only` world and `freeze-MLP` is the `attention_only` world. Target projections sum to one but may include cancellation. Variance ratios compare across-site Jacobian variability with the native full path; they are descriptive because freezing one operator also changes the inputs received by later operators."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
