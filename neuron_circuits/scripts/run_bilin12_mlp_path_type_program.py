#!/usr/bin/env python3
"""Discovery-frozen executable path types through the late bilinear MLP suffix."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_analytic_mlp_jacobian import boundary, norm, rms_jvp
from run_bilin12_router_finite_transport import fidelity
from run_bilin12_router_transport_operator_decomposition import batch_worlds, central
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_mlp_path_type_program.json")
OUTPUT = TENSOR_ROOT / "bilin12_mlp_path_type_program.json"
REPORT = TENSOR_ROOT / "BILIN12_MLP_PATH_TYPE_PROGRAM.md"
TERMS = ("I", "L", "R")
PATHS = tuple(itertools.product(TERMS, repeat=4))


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


@torch.inference_mode()
def suffix_cache(model, ids, layer, final_layer=None):
    """Cache native final-position states needed by the attention-frozen MLP JVP."""
    if final_layer is None:
        final_layer = len(model.blocks) - 1
    x, x0, bus = boundary(model, ids, layer)
    current = x
    cache = []
    for index in range(layer + 1, final_layer + 1):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        attention, bus = block.attn(norm(mixed), bus)
        pre = mixed + attention
        state = norm(pre)
        cache.append({"index": index, "pre": pre, "state": state})
        if index < final_layer:
            write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            current = pre + write
    return x, cache


def forward_term(block, pre, state, tangent, term):
    scaled = block.lambdas[0] * tangent
    if term == "I":
        return scaled
    normalized = rms_jvp(pre, scaled)
    if term == "L":
        gate = normalized @ block.mlp.Left.weight.float().T
        gate = gate * block.mlp.Right(state)[:, None]
    elif term == "R":
        gate = normalized @ block.mlp.Right.weight.float().T
        gate = gate * block.mlp.Left(state)[:, None]
    else:
        raise ValueError(term)
    return gate @ block.mlp.Down.weight.float().T


def adjoint_rms(pre, rows):
    # The RMSNorm Jacobian is symmetric, so its row adjoint is the same map.
    return rms_jvp(pre, rows)


def adjoint_term(block, pre, state, rows, term):
    if term == "I":
        return block.lambdas[0] * rows
    gate = rows @ block.mlp.Down.weight.float()
    if term == "L":
        gate = gate * block.mlp.Right(state)[:, None]
        value = gate @ block.mlp.Left.weight.float()
    elif term == "R":
        gate = gate * block.mlp.Left(state)[:, None]
        value = gate @ block.mlp.Right.weight.float()
    else:
        raise ValueError(term)
    return block.lambdas[0] * adjoint_rms(pre, value)


@torch.inference_mode()
def path_matrices(model, ids, position, config, axes, output_axes=None):
    """Return [B,81,32,32] exact path matrices by a 9x9 meet in the middle."""
    if output_axes is None:
        output_axes = axes
    final_layer = config.get("final_layer", len(model.blocks) - 1)
    layers = config.get("path_layers", list(range(config["layer"] + 1, final_layer)))
    if len(layers) != 4 or layers != list(range(layers[0], layers[0] + 4)):
        raise ValueError("path decomposition requires four consecutive MLP layers")
    x, cache = suffix_cache(model, ids, config["layer"], final_layer)
    by_index = {row["index"]: row for row in cache}
    scale = config["scale"] * x[:, position].square().mean(-1).sqrt()
    initial = scale[:, None, None] * axes.T[None]

    prefixes = []
    for first in TERMS:
        one = forward_term(model.blocks[layers[0]], by_index[layers[0]]["pre"][:, position],
                           by_index[layers[0]]["state"][:, position], initial, first)
        for second in TERMS:
            two = forward_term(model.blocks[layers[1]], by_index[layers[1]]["pre"][:, position],
                               by_index[layers[1]]["state"][:, position], one, second)
            prefixes.append(two)
    prefixes = torch.stack(prefixes, 1)  # B,9,input-axis,D

    final_pre = by_index[final_layer]["pre"][:, position]
    suffix = model.blocks[final_layer].lambdas[0] * adjoint_rms(
        final_pre, output_axes.T[None].expand(len(ids), -1, -1)
    )
    suffixes = []
    # Backward order is layer 10 then 9, but output ordering is layer9, layer10.
    for third in TERMS:
        for fourth in TERMS:
            ten = adjoint_term(model.blocks[layers[3]], by_index[layers[3]]["pre"][:, position],
                               by_index[layers[3]]["state"][:, position], suffix, fourth)
            nine = adjoint_term(model.blocks[layers[2]], by_index[layers[2]]["pre"][:, position],
                                by_index[layers[2]]["state"][:, position], ten, third)
            suffixes.append(nine.transpose(-1, -2))
    suffixes = torch.stack(suffixes, 1)  # B,9,D,output-axis
    paths = torch.einsum("bpnd,bqdo->bpqno", prefixes, suffixes)
    return paths.reshape(len(ids), len(PATHS), axes.shape[1], output_axes.shape[1])


def aggregate_gram(paths):
    flat = paths.double().reshape(len(paths), paths.shape[1], -1)
    gram = torch.einsum("npi,nqi->pq", flat, flat)
    target = flat.sum(1)
    dot = torch.einsum("npi,ni->p", flat, target)
    target2 = target.square().sum()
    return gram, dot, target2


def error_from_gram(gram, dot, target2, selected):
    if not selected:
        return float(target2)
    indices = torch.as_tensor(selected, dtype=torch.long)
    return float(target2 - 2 * dot[indices].sum() + gram[indices][:, indices].sum())


def greedy_order(gram, dot, target2):
    selected = []
    remaining = set(range(len(gram)))
    while remaining:
        choice = min(remaining, key=lambda item: error_from_gram(
            gram, dot, target2, selected + [item]
        ))
        selected.append(choice); remaining.remove(choice)
    return selected


def path_fidelity(paths, selected):
    target = paths.sum(1)
    prediction = paths[:, selected].sum(1) if selected else torch.zeros_like(target)
    return fidelity(target, prediction)


def oracle_fidelity(paths, budget):
    numerator = 0.0; denominator = 0.0
    for site in paths:
        gram, dot, target2 = aggregate_gram(site[None])
        selected = greedy_order(gram, dot, target2)[:budget]
        numerator += error_from_gram(gram, dot, target2, selected)
        denominator += float(target2)
    return 1 - numerator / max(denominator, 1e-30)


def random_summary(paths, budget, replicates, generator):
    values = []
    for _ in range(replicates):
        selected = torch.randperm(len(PATHS), generator=generator)[:budget].tolist()
        values.append(path_fidelity(paths, selected))
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "maximum": max(values)}


@torch.inference_mode(False)
def build_axes(model, config):
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": 32,
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(),
                       block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    natural, _ = collect_states_writes(model, 11, config["natural"],
                                       config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False)
    basis = vh[:32].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", basis, forms, basis).detach()
    rotation, _ = train_rotation(restricted, config)
    return (basis @ rotation.detach()).float()


@torch.inference_mode()
def collect_paths(model, axes, spec, config, include_finite=False):
    source = load_rows(spec); collected = []; finite = []
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128],
                                         dtype=np.int64), device="cuda")
        finite_worlds = None
        if include_finite:
            finite_worlds = central(batch_worlds(
                model, config["layer"], ids, config["positions"], axes, config["scale"]
            )["mlp_only"])
        for position_index, position in enumerate(config["positions"]):
            paths = path_matrices(model, ids, position, config, axes)
            collected.append(paths.cpu())
            if finite_worlds is not None:
                # batch_worlds emits position-major chunks, with batch inside each chunk.
                finite.append(finite_worlds.reshape(len(config["positions"]), len(ids), 32, 32)[position_index].cpu())
        print(f"  rows {min(start + config['batch_size'], len(source))}/{len(source)}", flush=True)
    return torch.cat(collected), (torch.cat(finite) if finite else None)


@torch.inference_mode()
def _direct_matrix(model, ids, position, config, axes, output_axes=None):
    """Independent unexpanded attention-frozen JVP used only for exactness tests."""
    if output_axes is None:
        output_axes = axes
    final_layer = config.get("final_layer", len(model.blocks) - 1)
    x, cache = suffix_cache(model, ids, config["layer"], final_layer)
    tangent = config["scale"] * x[:, position].square().mean(-1).sqrt()[:, None, None] * axes.T[None]
    for row in cache:
        index = row["index"]; block = model.blocks[index]
        scaled = block.lambdas[0] * tangent
        normalized = rms_jvp(row["pre"][:, position], scaled)
        if index == final_layer:
            return normalized @ output_axes
        left = normalized @ block.mlp.Left.weight.float().T
        right = normalized @ block.mlp.Right.weight.float().T
        dwrite = (left * block.mlp.Right(row["state"][:, position])[:, None]
                  + block.mlp.Left(row["state"][:, position])[:, None] * right)
        tangent = scaled + dwrite @ block.mlp.Down.weight.float().T
    raise RuntimeError


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    axes = build_axes(model, config)
    print("collect discovery paths", flush=True)
    discovery, _ = collect_paths(model, axes, config["path_discovery"], config)
    gram, dot, target2 = aggregate_gram(discovery)
    ordering = greedy_order(gram, dot, target2)
    generator = torch.Generator().manual_seed(config["seed"])
    results = {}; maximum_relative_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        paths, finite = collect_paths(model, axes, spec, config, include_finite=True)
        # Exactness is checked against an independently implemented unexpanded JVP.
        source = load_rows(spec); direct_rows = []
        for start in range(0, len(source), config["batch_size"]):
            ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
            for position in config["positions"]:
                direct_rows.append(_direct_matrix(model, ids, position, config, axes).cpu())
        direct = torch.cat(direct_rows); total = paths.sum(1)
        site_relative = ((total - direct).flatten(1).norm(dim=1)
                         / direct.flatten(1).norm(dim=1).clamp_min(1e-30))
        maximum_relative_error = max(maximum_relative_error, float(site_relative.max()))
        rows = {}
        for budget in config["budgets"]:
            selected = ordering[:budget]
            tangent = paths[:, selected].sum(1)
            finite_target = finite.reshape_as(tangent)
            rows[str(budget)] = {
                "frozen_fidelity": path_fidelity(paths, selected),
                "oracle_fidelity": oracle_fidelity(paths, budget),
                "random": random_summary(paths, budget, config["random_replicates"], generator),
                "finite_mlp_path_fidelity": fidelity(finite_target, tangent),
            }
        results[corpus] = {"sites": len(paths), "budgets": rows}

    top16 = ordering[:16]
    bilinear_deep = sum(sum(term != "I" for term in PATHS[index]) >= 2 for index in top16)
    fineweb16 = results["fineweb"]["budgets"]["16"]["frozen_fidelity"]
    hypotheses = {
        "P0_path_sum_is_exact": maximum_relative_error <= 1e-5,
        "P1_frozen_16_path_program_is_sufficient": all(row["budgets"]["16"]["frozen_fidelity"] >= .80 for row in results.values()),
        "P2_frozen_32_path_program_is_high_fidelity": all(row["budgets"]["32"]["frozen_fidelity"] >= .90 for row in results.values()),
        "P3_frozen_paths_beat_random": all(row["budgets"]["16"]["frozen_fidelity"] >= row["budgets"]["16"]["random"]["p95"] + .10 for row in results.values()),
        "P4_path_program_transfers_ood": all(results[name]["budgets"]["16"]["frozen_fidelity"] >= fineweb16 - .05 for name in ("wikitext", "tinystories")),
        "P5_one_path_is_not_enough": all(row["budgets"]["1"]["frozen_fidelity"] < .80 for row in results.values()),
        "P6_context_specific_selection_is_material": any(row["budgets"]["16"]["oracle_fidelity"] >= row["budgets"]["16"]["frozen_fidelity"] + .05 for row in results.values()),
        "P7_bilinear_path_types_are_material": bilinear_deep >= 8,
        "P8_program_predicts_finite_effects": all(row["budgets"]["32"]["finite_mlp_path_fidelity"] >= .85 for row in results.values()),
    }
    locked = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    locked["P0_path_sum_is_exact"]["maximum_relative_error"] = maximum_relative_error
    payload = {"config": config, "hypotheses": locked,
               "path_order": ["".join(PATHS[index]) for index in ordering],
               "top16_paths_with_two_or_more_bilinear_arms": bilinear_deep,
               "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = ["# Executable late-MLP path types in the 12-layer FineWeb tensor model", "",
             "With attention responses frozen, each of Layers 7–10 has three exact derivative terms: residual (`I`), left-factor bilinear arm (`L`), and right-factor bilinear arm (`R`). Their Cartesian product gives 81 executable path types. A nested path ordering is fit only to discovery FineWeb Jacobians, frozen, and tested on disjoint FineWeb, WikiText, and TinyStories. Random subsets and per-example greedy oracles use the same path inventory.", "", "## Locked hypotheses", ""]
    for name, row in locked.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum independent-JVP recomposition error: `{maximum_relative_error:.2e}`.",
              f"Top-16 paths with bilinear arms in at least two layers: **{bilinear_deep}/16**.", "",
              "Top 16 frozen path types: `" + "`, `".join(payload["path_order"][:16]) + "`.", "",
              "| Corpus | K | Frozen | Oracle | Random p95 | Finite-effect fidelity |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for budget in config["budgets"]:
            item = row["budgets"][str(budget)]
            lines.append(f"| {corpus} | {budget} | {item['frozen_fidelity']:.3f} | {item['oracle_fidelity']:.3f} | {item['random']['p95']:.3f} | {item['finite_mlp_path_fidelity']:.3f} |")
    lines += ["", "A path label names a recurring algebraic operation sequence, not a fixed neuron set or semantic feature. Each path still uses the current text-conditioned RMS Jacobians and bilinear gates. Frozen success would therefore support a reusable operator grammar over contextual operands; oracle-only success would reject that stronger claim."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
