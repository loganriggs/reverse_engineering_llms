#!/usr/bin/env python3
"""Finite causal transport of the frozen late sparse-graph coordinates."""

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


CONFIG = Path("configs/bilin12_router_finite_transport.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_finite_transport.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_FINITE_TRANSPORT.md"


def rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


def normalize(x):
    return F.rms_norm(x, (x.shape[-1],))


@torch.inference_mode()
def trace_to_boundary(model, ids, boundary):
    x0 = normalize(model.embedding(ids)); x, bus = x0, None
    for layer, block in enumerate(model.blocks):
        x, bus = block(x, bus, x0)
        if layer == boundary:
            return x, x0, bus
    raise ValueError(boundary)


@torch.inference_mode()
def suffix_to_final_state(model, boundary, x, x0, bus):
    for layer in range(boundary + 1, 12):
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, bus = block.attn(normalize(mixed), bus)
        pre = mixed + attention
        state = normalize(pre)
        if layer == 11:
            return state
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + write
    raise RuntimeError("suffix did not reach final attention")


@torch.inference_mode()
def native_final_states(model, ids):
    x0 = normalize(model.embedding(ids)); x, bus = x0, None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, bus = block.attn(normalize(mixed), bus)
        pre = mixed + attention; state = normalize(pre)
        if layer == 11:
            return state
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + write
    raise RuntimeError


def edge_values(z, pairs):
    return torch.stack([2 * z[..., i] * z[..., j] for i, j in pairs], -1)


def fidelity(target, prediction):
    return 1 - float((target - prediction).square().sum() / target.square().sum().clamp_min(1e-30))


def cosine_rows(left, right):
    return (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-30)


def fit_map(x, y, ridge):
    gram = x.T @ x
    penalty = ridge * float(torch.trace(gram)) / len(gram)
    return torch.linalg.solve(gram + penalty * torch.eye(len(gram)), x.T @ y)


@torch.inference_mode()
def perturb_batch(model, boundary, ids, positions, axes, directions, scale):
    """Return actual normalized input/final coordinate deltas for +/- edits."""
    x, x0, bus = trace_to_boundary(model, ids, boundary)
    final = suffix_to_final_state(model, boundary, x, x0, bus)
    boundary_z = normalize(x) @ axes; final_z = final @ axes
    bsz, count = len(ids), len(directions)
    outputs = []
    maximum_trace_error = float((final - native_final_states(model, ids)).abs().max())
    for position in positions:
        event_directions = torch.cat([directions, -directions], 0)
        events = len(event_directions)
        xr = x[:, None].expand(bsz, events, *x.shape[1:]).clone().reshape(bsz * events, *x.shape[1:])
        x0r = x0[:, None].expand(bsz, events, *x0.shape[1:]).reshape(bsz * events, *x0.shape[1:])
        br = bus[:, None].expand(bsz, events, *bus.shape[1:]).reshape(bsz * events, *bus.shape[1:])
        native_direction = event_directions @ axes.T
        native_direction = native_direction[None].expand(bsz, events, -1).reshape(bsz * events, -1)
        rms = x[:, position].square().mean(-1).sqrt()
        rms = rms[:, None].expand(bsz, events).reshape(-1, 1)
        xr[:, position] += scale * rms * native_direction
        input_z = (normalize(xr[:, position]) @ axes).reshape(bsz, events, -1)
        output_z = (suffix_to_final_state(model, boundary, xr, x0r, br)[:, position] @ axes).reshape(bsz, events, -1)
        base_input = boundary_z[:, position, None, :]
        base_output = final_z[:, position, None, :]
        base_input_full = normalize(x[:, position])[:, None, :]
        outputs.append({
            "input": (input_z - base_input).cpu(),
            "output": (output_z - base_output).cpu(),
            "base_input": base_input.expand(-1, events, -1).cpu(),
            "base_input_full": base_input_full.expand(-1, events, -1).cpu(),
            "base_output": base_output.expand(-1, events, -1).cpu(),
            "direction": event_directions[None].expand(bsz, -1, -1).cpu(),
            "position": torch.full((bsz, events, 1), int(position), dtype=torch.float32),
            "count": count,
        })
    merged = {key: torch.cat([item[key].reshape(-1, item[key].shape[-1]) for item in outputs])
              for key in ("input", "output", "base_input", "base_input_full", "base_output", "direction", "position")}
    merged["trace_error"] = maximum_trace_error
    merged["direction_count"] = count
    return merged


@torch.inference_mode()
def collect_events(model, boundary, spec, positions, axes, directions, scale, batch_size):
    stores = {key: [] for key in ("input", "output", "base_input", "base_input_full", "base_output", "direction", "position")}
    maximum = 0.0
    source = rows(spec)
    for start in range(0, len(source), batch_size):
        raw = np.asarray(source[start:start + batch_size, :128], dtype=np.int64)
        batch = perturb_batch(model, boundary, torch.as_tensor(raw, device="cuda"), positions,
                              axes, directions, scale)
        maximum = max(maximum, batch["trace_error"])
        for key in stores: stores[key].append(batch[key])
    result = {key: torch.cat(value).float() for key, value in stores.items()}
    result["trace_error"] = maximum
    result["direction_count"] = len(directions)
    return result


def central(values, direction_count):
    # Event order inside each site is +directions followed by -directions.
    shaped = values.reshape(-1, 2 * direction_count, values.shape[-1])
    return (shaped[:, :direction_count] - shaped[:, direction_count:]) / 2


def summarize(events, transport, pairs, hubs, scale_ratio, small=None):
    predicted = events["input"] @ transport
    edge_target = edge_values(events["base_output"] + events["output"], pairs) - edge_values(events["base_output"], pairs)
    edge_prediction = edge_values(events["base_output"] + predicted, pairs) - edge_values(events["base_output"], pairs)
    count = events["direction_count"]
    cin = central(events["input"], count); cout = central(events["output"], count)
    directions = central(events["direction"], count)
    direction_cosine = cosine_rows(cout, directions)
    hub_mask = torch.zeros(count, dtype=torch.bool)
    if count == 32:
        hub_mask[hubs] = True
    result = {
        "coordinate_fidelity": fidelity(events["output"], predicted),
        "coordinate_cosine": float(cosine_rows(events["output"], predicted).median()),
        "edge_change_fidelity": fidelity(edge_target, edge_prediction),
        "edge_change_cosine": float(cosine_rows(edge_target, edge_prediction).median()),
        "central_direction_cosine_median": float(direction_cosine.median()),
        "central_gain_median": float((cout * directions).sum(-1).median()),
    }
    if count == 32:
        result["hub_direction_cosine_median"] = float(direction_cosine[:, hub_mask].median())
        result["nonhub_direction_cosine_median"] = float(direction_cosine[:, ~hub_mask].median())
    if small is not None:
        small_central = central(small["output"], count) * scale_ratio
        result["cross_scale_central_fidelity"] = fidelity(cout, small_central)
        result["cross_scale_central_cosine"] = float(cosine_rows(cout, small_central).median())
    return result


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); graph = json.loads(Path(config["selected_graph"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    states, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(states.float(), full_matrices=False)
    natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, forms, natural_basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (natural_basis @ rotation.detach()).float()
    pairs = [tuple(pair) for pair in graph["selected_pairs"]]
    degree = np.bincount([node for pair in pairs for node in pair], minlength=32)
    hubs = np.flatnonzero(degree >= 3).tolist()
    canonical = torch.eye(32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_directions, _ = torch.linalg.qr(torch.randn(32, 32, generator=generator, device="cuda"))
    results = {}; maps = {}; trace_error = 0.0
    ratio = config["large_scale"] / config["small_scale"]
    for layer in config["layers"]:
        print("fit transport", layer, flush=True)
        train = collect_events(model, layer, config["transport_train"], config["positions"], axes,
                               canonical, config["small_scale"], config["batch_size"])
        transport = fit_map(train["input"], train["output"], config["ridge"])
        maps[str(layer)] = {
            "diagonal_energy_fraction": float(transport.diagonal().square().sum() / transport.square().sum()),
            "identity_cosine": float(torch.trace(transport) / (transport.norm() * np.sqrt(32))),
            "singular_values": torch.linalg.svdvals(transport).tolist(),
        }
        results[str(layer)] = {}
        for corpus, spec in config["evaluation"].items():
            print("evaluate", layer, corpus, flush=True)
            small = collect_events(model, layer, spec, config["positions"], axes, canonical,
                                   config["small_scale"], config["batch_size"])
            large = collect_events(model, layer, spec, config["positions"], axes, canonical,
                                   config["large_scale"], config["batch_size"])
            random_large = collect_events(model, layer, spec, config["positions"], axes, random_directions,
                                          config["large_scale"], config["batch_size"])
            trace_error = max(trace_error, small["trace_error"], large["trace_error"], random_large["trace_error"])
            canonical_row = summarize(large, transport, pairs, hubs, ratio, small)
            random_row = summarize(random_large, transport, pairs, hubs, ratio)
            results[str(layer)][corpus] = {"canonical": canonical_row, "random": random_row}
    f0 = trace_error <= 1e-5
    f1 = all(results["10"][c]["canonical"]["hub_direction_cosine_median"] >= .90 for c in results["10"])
    f2 = all(results["6"][c]["canonical"]["hub_direction_cosine_median"] >= .70 for c in results["6"])
    f3 = all(results[str(layer)][c]["canonical"]["coordinate_fidelity"] >= .80
             for layer in config["layers"] for c in results[str(layer)])
    f4 = all(results[str(layer)][c]["canonical"]["edge_change_fidelity"] >= .70
             for layer in config["layers"] for c in results[str(layer)])
    f5 = all(results[str(layer)][c]["canonical"]["hub_direction_cosine_median"] >=
             results[str(layer)][c]["random"]["central_direction_cosine_median"] + .10
             for layer in config["layers"] for c in results[str(layer)])
    f6 = all(results[str(layer)][c]["canonical"]["cross_scale_central_fidelity"] >= .85
             for layer in config["layers"] for c in results[str(layer)])
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "F0_finite_trace_recomposes": f0,
        "F1_late_hubs_are_causally_persistent": f1,
        "F2_midlayer_hubs_are_causally_persistent": f2,
        "F3_linear_transport_predicts_large_edits": f3,
        "F4_transport_predicts_graph_changes": f4,
        "F5_canonical_hubs_are_privileged_carriers": f5,
        "F6_transport_is_stable_across_edit_size": f6,
    }.items()}
    hypotheses["F0_finite_trace_recomposes"]["maximum_error"] = trace_error
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "hubs": hubs,
                                  "maps": maps, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite causal transport of the late sparse interface", "",
             "The frozen 32-dimensional final-layer basis was perturbed at residual boundaries after Layers 6 and 10. A 32-by-32 transport matrix was fit only on small FineWeb perturbations, then frozen and tested on four-times-larger edits, disjoint FineWeb, two OOD corpora, canonical coordinates, and random rotations of the same subspace.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Corpus | Hub direction cosine | Random direction cosine | Coordinate fidelity | Edge-change fidelity | Cross-scale fidelity |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus, row in results[str(layer)].items():
            c, r = row["canonical"], row["random"]
            lines.append(f"| {layer} | {corpus} | {c['hub_direction_cosine_median']:.3f} | "
                         f"{r['central_direction_cosine_median']:.3f} | {c['coordinate_fidelity']:.3f} | "
                         f"{c['edge_change_fidelity']:.3f} | {c['cross_scale_central_fidelity']:.3f} |")
    lines += ["", "The directional cosine asks whether a finite nudge to one proposed coordinate remains pointed along that coordinate downstream. Coordinate and edge fidelities ask whether one small-edit FineWeb transport map predicts substantially larger unseen edits. Random rotations test whether named graph hubs are special carriers, rather than arbitrary coordinates in a generally persistent subspace."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
