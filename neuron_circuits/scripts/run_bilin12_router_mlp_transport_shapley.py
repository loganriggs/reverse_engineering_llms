#!/usr/bin/env python3
"""Exact four-player Shapley decomposition of downstream MLP transport."""

from __future__ import annotations

import json
import math
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


CONFIG = Path("configs/bilin12_router_mlp_transport_shapley.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_mlp_transport_shapley.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_MLP_TRANSPORT_SHAPLEY.md"


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
def cache(model, layer, x, x0, bus):
    rows = []; current = x
    for index in range(layer + 1, 12):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        attention, bus = block.attn(norm(mixed), bus)
        pre = mixed + attention; state = norm(pre)
        if index == 11: rows.append((attention, None)); return state, rows
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        rows.append((attention, write)); current = pre + write
    raise RuntimeError


@torch.inference_mode()
def suffix(model, layer, x, x0, caches, base_batch, mask, mlp_layers):
    current = x; events = len(x) // base_batch
    for offset, index in enumerate(range(layer + 1, 12)):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        attention = caches[offset][0][:, None].expand(base_batch, events, -1, -1).reshape_as(mixed)
        pre = mixed + attention; state = norm(pre)
        if index == 11: return state
        factor = mlp_layers.index(index)
        if mask & (1 << factor):
            write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        else:
            write = caches[offset][1][:, None].expand(base_batch, events, -1, -1).reshape_as(pre)
        current = pre + write
    raise RuntimeError


@torch.inference_mode()
def batch_worlds(model, ids, config, axes):
    layer = config["boundary_layer"]; mlp_layers = config["mlp_layers"]
    x, x0, bus = boundary(model, ids, layer); base_final, caches = cache(model, layer, x, x0, bus)
    base_input = norm(x) @ axes; base_output = base_final @ axes
    bsz = len(ids); directions = torch.eye(32, device="cuda"); event_directions = torch.cat([directions, -directions]); events = 64
    stores = {mask: [] for mask in range(16)}
    for position in config["positions"]:
        xr = x[:, None].expand(bsz, events, *x.shape[1:]).clone().reshape(bsz * events, *x.shape[1:])
        x0r = x0[:, None].expand(bsz, events, *x0.shape[1:]).reshape(bsz * events, *x0.shape[1:])
        native_direction = (event_directions @ axes.T)[None].expand(bsz, -1, -1).reshape(bsz * events, -1)
        rms = x[:, position].square().mean(-1).sqrt()[:, None].expand(bsz, events).reshape(-1, 1)
        xr[:, position] += config["scale"] * rms * native_direction
        for mask in range(16):
            output = suffix(model, layer, xr, x0r, caches, bsz, mask, mlp_layers)
            delta = (output[:, position] @ axes).reshape(bsz, events, -1) - base_output[:, position, None]
            stores[mask].append(delta.cpu())
    return {mask: torch.cat([value.reshape(-1, 32) for value in rows]) for mask, rows in stores.items()}


def central(x):
    x = x.reshape(-1, 64, 32); return (x[:, :32] - x[:, 32:]) / 2


def projection(target, value): return float((target * value).sum() / target.square().sum().clamp_min(1e-30))


def finalize(acc, layers):
    worlds = {mask: central(torch.cat(acc[mask])) for mask in range(16)}; target = worlds[15]
    shapley = []
    for factor in range(4):
        bit = 1 << factor; effect = torch.zeros_like(target)
        for mask in range(16):
            if mask & bit: continue
            size = mask.bit_count(); weight = math.factorial(size) * math.factorial(3 - size) / math.factorial(4)
            effect += weight * (worlds[mask | bit] - worlds[mask])
        shapley.append(effect)
    reconstruction = worlds[0] + sum(shapley)
    energy = target.square().sum().clamp_min(1e-30); target_norm = float(energy.sqrt())
    single_additive = worlds[0] + sum(worlds[1 << i] - worlds[0] for i in range(4))
    cumulative = {}
    for stop in range(1, 5):
        mask = (1 << stop) - 1
        cumulative[str(layers[stop - 1])] = 1 - float((target - worlds[mask]).square().sum() / energy)
    projections = {str(layer): projection(target, shapley[index]) for index, layer in enumerate(layers)}
    return {
        "sites": len(target),
        "maximum_recomposition_error": float((target - reconstruction).abs().max()),
        "residual_projection": projection(target, worlds[0]),
        "layer_target_projection": projections,
        "layer_norm_ratio": {str(layer): float(shapley[index].square().sum().sqrt()) / target_norm
                             for index, layer in enumerate(layers)},
        "absolute_projection_sum": sum(abs(value) for value in projections.values()),
        "additive_interaction_norm_ratio": float((target - single_additive).square().sum().sqrt()) / target_norm,
        "cumulative_fidelity": cumulative,
        "single_layer_fidelity": {str(layer): 1 - float((target - worlds[1 << index]).square().sum() / energy)
                                  for index, layer in enumerate(layers)},
    }


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": 32, "discovery": config["basis_discovery"]})
    block = model.blocks[11]; channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    natural, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False); basis = vh[:32].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", basis, forms, basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (basis @ rotation.detach()).float()
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True); acc = {mask: [] for mask in range(16)}; source = load_rows(spec)
        for start in range(0, len(source), config["batch_size"]):
            raw_ids = np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64)
            row = batch_worlds(model, torch.as_tensor(raw_ids, device="cuda"), config, axes)
            for mask, value in row.items(): acc[mask].append(value)
        results[corpus] = finalize(acc, config["mlp_layers"])
    m0 = all(row["maximum_recomposition_error"] <= 1e-5 for row in results.values())
    m1 = all(max(row["layer_target_projection"].values()) >= .50 for row in results.values())
    m2 = all(sum(row["layer_target_projection"][str(layer)] for layer in (9, 10)) >= .70 for row in results.values())
    m3 = True
    for row in results.values():
        values = [row["cumulative_fidelity"][str(layer)] for layer in config["mlp_layers"]]
        m3 &= all(values[i + 1] >= values[i] - .02 for i in range(3))
    m4 = all(row["additive_interaction_norm_ratio"] >= .10 for row in results.values())
    fineweb = results["fineweb"]["layer_target_projection"]
    m5 = all(max(abs(row["layer_target_projection"][str(layer)] - fineweb[str(layer)])
                 for layer in config["mlp_layers"]) <= .10
             for corpus, row in results.items() if corpus != "fineweb")
    m6 = all(row["absolute_projection_sum"] >= 1.20 for row in results.values())
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "M0_shapley_recomposition_is_exact": m0, "M1_one_mlp_layer_dominates": m1,
        "M2_late_mlp_layers_dominate": m2, "M3_progressive_unfreezing_is_monotonic": m3,
        "M4_mlp_layer_interactions_are_material": m4, "M5_layer_profile_transfers_ood": m5,
        "M6_early_late_cancellation_is_material": m6}.items()}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Exact layerwise Shapley decomposition of MLP transport", "",
             "With all downstream attention responses frozen, the four bilinear MLPs in Layers 7–10 are independently recomputed or frozen across all 16 hybrid worlds. Exact Shapley vectors attribute their finite causal transport while retaining every interaction. The residual response is reported separately.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Residual projection | L7/L8/L9/L10 projections | Additive interaction norm | Cumulative L7/L8/L9/L10 fidelity |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        p = row["layer_target_projection"]; c = row["cumulative_fidelity"]
        lines.append(f"| {corpus} | {row['residual_projection']:.3f} | " + "/".join(f"{p[str(x)]:.3f}" for x in config["mlp_layers"]) +
                     f" | {row['additive_interaction_norm_ratio']:.3f} | " + "/".join(f"{c[str(x)]:.3f}" for x in config["mlp_layers"]) + " |")
    lines += ["", "Shapley target projections describe signed alignment with the complete attention-frozen response; they are not percentages of stored information. The additive-interaction norm compares the full four-MLP response with a sum of isolated one-MLP responses, exposing composition that no single-layer ranking captures."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
