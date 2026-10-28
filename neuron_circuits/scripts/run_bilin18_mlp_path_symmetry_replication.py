#!/usr/bin/env python3
"""Replicate the invariant late-MLP path program in the 546M FineWeb model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin12_mlp_path_symmetry_redteam import GROUPS, collapse, random_summary
from run_bilin12_mlp_path_type_program import (
    _direct_matrix, aggregate_gram, boundary, greedy_order, norm,
    oracle_fidelity, path_fidelity, path_matrices,
)
from run_bilin12_router_finite_transport import fidelity
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_mlp_path_symmetry_replication.json")
OUTPUT = TENSOR_ROOT / "bilin18_mlp_path_symmetry_replication.json"
REPORT = TENSOR_ROOT / "BILIN18_MLP_PATH_SYMMETRY_REPLICATION.md"


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


@torch.inference_mode()
def trace_suffix(model, ids, config):
    x, x0, bus = boundary(model, ids, config["layer"])
    current = x; attentions = []
    for index in range(config["layer"] + 1, config["final_layer"] + 1):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        attention, bus = block.attn(norm(mixed), bus)
        attentions.append(attention)
        pre = mixed + attention; state = norm(pre)
        if index == config["final_layer"]:
            return x, x0, state, attentions
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        current = pre + write
    raise RuntimeError


@torch.inference_mode()
def build_axes(model, config):
    inputs = []; outputs = []; source = load_rows(config["axis_discovery"])
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        x, _, final, _ = trace_suffix(model, ids, config)
        inputs.append(x[:, config["positions"]].float().flatten(0, 1).cpu())
        outputs.append(final[:, config["positions"]].float().flatten(0, 1).cpu())
    _, _, input_vh = torch.linalg.svd(torch.cat(inputs), full_matrices=False)
    _, _, output_vh = torch.linalg.svd(torch.cat(outputs), full_matrices=False)
    return input_vh[:config["rank"]].T.cuda(), output_vh[:config["rank"]].T.cuda()


@torch.inference_mode()
def frozen_mlp_finite(model, ids, position, config, input_axes, output_axes):
    x, x0, final, attentions = trace_suffix(model, ids, config)
    batch = len(ids); rank = input_axes.shape[1]
    directions = torch.cat([torch.eye(rank, device="cuda"), -torch.eye(rank, device="cuda")])
    events = len(directions)
    xr = x[:, None].expand(batch, events, *x.shape[1:]).clone().reshape(batch * events, *x.shape[1:])
    x0r = x0[:, None].expand(batch, events, *x0.shape[1:]).reshape(batch * events, *x0.shape[1:])
    native_direction = (directions @ input_axes.T)[None].expand(batch, -1, -1).reshape(batch * events, -1)
    rms = x[:, position].square().mean(-1).sqrt()[:, None].expand(batch, events).reshape(-1, 1)
    xr[:, position] += config["scale"] * rms * native_direction
    current = xr
    for offset, index in enumerate(range(config["layer"] + 1, config["final_layer"] + 1)):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0r
        attention = attentions[offset][:, None].expand(batch, events, *attentions[offset].shape[1:]).reshape_as(mixed)
        pre = mixed + attention; state = norm(pre)
        if index == config["final_layer"]:
            changed = (state[:, position].float() @ output_axes).reshape(batch, events, rank)
            base = (final[:, position].float() @ output_axes)[:, None]
            delta = changed - base
            return (delta[:, :rank] - delta[:, rank:]) / 2
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        current = pre + write
    raise RuntimeError


@torch.inference_mode()
def collect(model, input_axes, output_axes, spec, config, finite=False):
    source = load_rows(spec); path_rows = []; finite_rows = []; maximum_error = 0.0
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        for position in config["positions"]:
            raw = path_matrices(model, ids, position, config, input_axes, output_axes)
            grouped = collapse(raw.cpu()); path_rows.append(grouped)
            direct = _direct_matrix(model, ids, position, config, input_axes, output_axes).cpu()
            total = grouped.sum(1)
            relative = ((total - direct).flatten(1).norm(dim=1)
                        / direct.flatten(1).norm(dim=1).clamp_min(1e-30))
            maximum_error = max(maximum_error, float(relative.max()))
            if finite:
                finite_rows.append(frozen_mlp_finite(
                    model, ids, position, config, input_axes, output_axes
                ).cpu())
        print(f"  rows {min(start + config['batch_size'], len(source))}/{len(source)}", flush=True)
    return torch.cat(path_rows), (torch.cat(finite_rows) if finite_rows else None), maximum_error


def k90(paths, ordering):
    for count in range(1, len(ordering) + 1):
        if path_fidelity(paths, ordering[:count]) >= .90:
            return count
    return len(ordering)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    # The checkpoint loader preserves bf16 modules.  A 2%-norm finite edit can
    # fall below bf16 resolution along many coordinates, while the analytic JVP
    # is accumulated in float32.  Match the established causal scripts and run
    # both the native finite world and the derivative trace in float32.
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    parameters = sum(p.numel() for module in (model.embedding, model.blocks, model.lm_head)
                     for p in module.parameters())
    input_axes, output_axes = build_axes(model, config)
    print("collect discovery", flush=True)
    discovery, _, discovery_error = collect(model, input_axes, output_axes,
                                             config["path_discovery"], config)
    ordering = greedy_order(*aggregate_gram(discovery))
    low_order = [i for i, group in enumerate(GROUPS) if group.count("B") <= 1]
    generator = torch.Generator().manual_seed(config["seed"])
    maximum_error = discovery_error; results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        paths, finite, error = collect(model, input_axes, output_axes, spec, config, finite=True)
        maximum_error = max(maximum_error, error); rows = {}
        for budget in config["budgets"]:
            selected = ordering[:budget]; tangent = paths[:, selected].sum(1)
            rows[str(budget)] = {
                "frozen_fidelity": path_fidelity(paths, selected),
                "oracle_fidelity": oracle_fidelity(paths, budget),
                "random": random_summary(paths, budget, config["random_replicates"], generator),
                "finite_mlp_path_fidelity": fidelity(finite.reshape_as(tangent), tangent),
            }
        results[corpus] = {"sites": len(paths), "k90": k90(paths, ordering),
                           "degree_le_one_fidelity": path_fidelity(paths, low_order),
                           "budgets": rows}

    fineweb4 = results["fineweb"]["budgets"]["4"]["frozen_fidelity"]
    gates = {"k4": .85, "k8": .95, "degree_one": .85, "finite_k8": .90,
             "ood_drop": .05, "maximum_k90": 5, "oracle_gap": .05,
             **config.get("gates", {})}
    supported = {
        "X0_checkpoint_is_over_500m": parameters > 500_000_000,
        "X1_grouped_path_sum_is_exact": maximum_error <= 1e-5,
        "X2_frozen_four_group_program_is_compact": all(row["budgets"]["4"]["frozen_fidelity"] >= gates["k4"] for row in results.values()),
        "X3_frozen_eight_group_program_is_high_fidelity": all(row["budgets"]["8"]["frozen_fidelity"] >= gates["k8"] for row in results.values()),
        "X4_degree_one_truncation_is_sufficient": all(row["degree_le_one_fidelity"] >= gates["degree_one"] for row in results.values()),
        "X5_group_program_predicts_finite_effects": all(row["budgets"]["8"]["finite_mlp_path_fidelity"] >= gates["finite_k8"] for row in results.values()),
        "X6_group_program_transfers_ood": all(results[name]["budgets"]["4"]["frozen_fidelity"] >= fineweb4 - gates["ood_drop"] for name in ("wikitext", "tinystories")),
        "X7_native_suffix_is_low_order": all(row["k90"] <= gates["maximum_k90"] for row in results.values()),
        "X8_context_specific_selection_is_small": all(row["budgets"]["4"]["oracle_fidelity"] < row["budgets"]["4"]["frozen_fidelity"] + gates["oracle_gap"] for row in results.values()),
    }
    hypotheses = {name: {"supported": bool(value)} for name, value in supported.items()}
    hypotheses["X0_checkpoint_is_over_500m"]["parameter_count"] = parameters
    hypotheses["X1_grouped_path_sum_is_exact"]["maximum_relative_error"] = maximum_error
    payload = {"config": config, "hypotheses": hypotheses, "parameter_count": parameters,
               "group_order": ["".join(GROUPS[index]) for index in ordering],
               "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    title = ("# High-count invariant late-MLP path replication in the 546M FineWeb model"
             if "source_artifact" in config else
             "# Invariant late-MLP path replication in the 546M FineWeb model")
    lines = [title, "",
             f"The checkpoint has **{parameters:,} parameters**. Natural FineWeb PCA supplies separate 32D input and output charts around the Layer-12→17 suffix. With attention responses frozen, Layers 13–16 are expanded into the same 16 symmetry-invariant `I/B` layer-subset paths used in the independent 190M model. Ordering is learned on discovery FineWeb and frozen for disjoint FineWeb, WikiText, and TinyStories.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum independent-JVP recomposition error: `{maximum_error:.2e}`.", "",
              "Frozen group order: `" + "`, `".join(payload["group_order"]) + "`.", "",
              "| Corpus | K | Frozen | Oracle | Random p95 | Finite effect |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for budget in config["budgets"]:
            item = row["budgets"][str(budget)]
            lines.append(f"| {corpus} | {budget} | {item['frozen_fidelity']:.3f} | {item['oracle_fidelity']:.3f} | {item['random']['p95']:.3f} | {item['finite_mlp_path_fidelity']:.3f} |")
        lines.append(f"| {corpus} | degree≤1 (5), K90={row['k90']} | {row['degree_le_one_fidelity']:.3f} | — | — | — |")
    lines += ["", "This is an exact local account of the declared attention-frozen suffix route, not an explanation of how text selects the input perturbation or how attention carries it. Replication would establish that low interaction order is not peculiar to the smaller checkpoint; failure would make the regime architecture- or scale-dependent."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
