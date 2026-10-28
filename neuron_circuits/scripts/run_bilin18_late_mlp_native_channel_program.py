#!/usr/bin/env python3
"""Test native-channel sparsity inside the compact late bilinear path operator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin12_mlp_path_type_program import norm
from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_late_mlp_native_channel_program.json")
OUTPUT = TENSOR_ROOT / "bilin18_late_mlp_native_channel_program.json"
REPORT = TENSOR_ROOT / "BILIN18_LATE_MLP_NATIVE_CHANNEL_PROGRAM.md"


@torch.inference_mode()
def collect_states(model, spec, config):
    source = load_rows(spec); stores = {layer: [] for layer in config["layers"]}
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        _, cache = native_cache(model, ids, config)
        by_layer = {row["index"]: row for row in cache}
        for layer in config["layers"]:
            stores[layer].append(by_layer[layer]["state"][:, config["positions"]].float().flatten(0, 1).cpu())
    return {layer: torch.cat(rows) for layer, rows in stores.items()}


def build_axes(states, rank):
    return {layer: torch.linalg.svd(value, full_matrices=False).Vh[:rank].T.contiguous()
            for layer, value in states.items()}


@torch.inference_mode()
def derivative_components(block, states, axes, batch_size=16):
    left_weight = block.mlp.Left.weight.float(); right_weight = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float(); left_direction = axes.T.cuda() @ left_weight.T
    right_direction = axes.T.cuda() @ right_weight.T; outputs = []
    for start in range(0, len(states), batch_size):
        x = states[start:start + batch_size].cuda()
        left = x @ left_weight.T; right = x @ right_weight.T
        gate = (right[:, :, None] * left_direction.T[None]
                + left[:, :, None] * right_direction.T[None])
        target = torch.einsum("scn,dc->snd", gate, down)
        outputs.append((gate.cpu(), target.cpu()))
    gates = torch.cat([row[0] for row in outputs]); targets = torch.cat([row[1] for row in outputs])
    return gates, targets, down.cpu()


def ranking(gates, target, down):
    # Exact marginal reduction in squared error when one native contribution is
    # added to the zero predictor: 2<C_j,T>-||C_j||^2.
    tdot = torch.einsum("snd,dc->scn", target.float(), down.float())
    dot = (gates.float() * tdot).sum((0, 2)).double()
    energy = gates.float().square().sum((0, 2)).double() * down.float().square().sum(0).double()
    return torch.argsort(2 * dot - energy, descending=True)


def reconstruct(gates, down, selected, chunk=32):
    predictions = []
    index = torch.as_tensor(selected, dtype=torch.long)
    for start in range(0, len(gates), chunk):
        gate = gates[start:start + chunk, index].cuda()
        predictions.append(torch.einsum("scn,dc->snd", gate, down[:, index].cuda()).cpu())
    return torch.cat(predictions)


def score(target, prediction):
    return 1 - float((target - prediction).double().square().sum()
                     / target.double().square().sum().clamp_min(1e-30))


def random_summary(gates, target, down, budget, replicates, generator):
    values = []
    for _ in range(replicates):
        selected = torch.randperm(gates.shape[1], generator=generator)[:budget]
        values.append(score(target, reconstruct(gates, down, selected)))
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "maximum": max(values)}


def oracle_score(gates, target, down, budget):
    numerator = denominator = 0.0
    down_norm = down.float().square().sum(0)
    for site_gate, site_target in zip(gates, target):
        tdot = torch.einsum("nd,dc->cn", site_target.float(), down.float())
        dot = (site_gate.float() * tdot).sum(1)
        energy = site_gate.float().square().sum(1) * down_norm
        selected = torch.topk(2 * dot - energy, budget).indices
        prediction = torch.einsum("cn,dc->nd", site_gate[selected].cuda(), down[:, selected].cuda()).cpu()
        numerator += float((site_target - prediction).double().square().sum())
        denominator += float(site_target.double().square().sum())
    return 1 - numerator / max(denominator, 1e-30)


@torch.inference_mode(False)
def main():
    raw = json.loads(CONFIG.read_text()); base = json.loads(Path(raw["source_config"]).read_text())
    config = {**base, **{key: value for key, value in raw.items()
                        if key not in ("source_config", "hypotheses")}, "hypotheses": raw["hypotheses"]}
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    axis_states = collect_states(model, config["axis_discovery"], config); axes = build_axes(axis_states, config["rank"])
    discovery_states = collect_states(model, config["bank_discovery"], config)
    orderings = {}; swap_error = 0.0
    for layer in config["layers"]:
        print("rank layer", layer, flush=True)
        gates, target, down = derivative_components(model.blocks[layer], discovery_states[layer], axes[layer])
        orderings[layer] = ranking(gates, target, down)
        # Left/right exchange leaves each gate derivative algebraically unchanged.
        block = model.blocks[layer]; x = discovery_states[layer][:8].cuda(); axis = axes[layer].cuda()
        original = ((x @ block.mlp.Right.weight.float().T)[:, :, None]
                    * (axis.T @ block.mlp.Left.weight.float().T).T[None]
                    + (x @ block.mlp.Left.weight.float().T)[:, :, None]
                    * (axis.T @ block.mlp.Right.weight.float().T).T[None])
        swapped = ((x @ block.mlp.Left.weight.float().T)[:, :, None]
                   * (axis.T @ block.mlp.Right.weight.float().T).T[None]
                   + (x @ block.mlp.Right.weight.float().T)[:, :, None]
                   * (axis.T @ block.mlp.Left.weight.float().T).T[None])
        swap_error = max(swap_error, float((original - swapped).norm() / original.norm().clamp_min(1e-30)))

    evaluation_states = {corpus: collect_states(model, spec, config)
                         for corpus, spec in config["evaluation"].items()}
    generator = torch.Generator().manual_seed(config["seed"]); results = {}
    for corpus, layer_states in evaluation_states.items():
        print("evaluate", corpus, flush=True); layer_rows = {}
        for layer in config["layers"]:
            gates, target, down = derivative_components(model.blocks[layer], layer_states[layer], axes[layer])
            rows = {}
            for budget in config["budgets"]:
                selected = orderings[layer][:budget]
                rows[str(budget)] = {"fidelity": score(target, reconstruct(gates, down, selected))}
                if budget == config["oracle_budget"]:
                    rows[str(budget)]["oracle_fidelity"] = oracle_score(gates, target, down, budget)
                    rows[str(budget)]["random"] = random_summary(
                        gates, target, down, budget, config["random_replicates"], generator)
            k90 = next((budget for budget in config["budgets"] if rows[str(budget)]["fidelity"] >= .90), 4608)
            layer_rows[str(layer)] = {"sites": len(gates), "k90": k90, "budgets": rows}
        results[corpus] = layer_rows

    cells = [(corpus, layer, results[corpus][str(layer)])
             for corpus in results for layer in config["layers"]]
    supported = {
        "N0_native_channel_sum_is_exact": all(row["budgets"]["4608"]["fidelity"] >= .999999 for _, _, row in cells),
        "N1_static_512_channels_are_sufficient": all(row["budgets"]["512"]["fidelity"] >= .80 for _, _, row in cells),
        "N2_static_1024_channels_are_high_fidelity": all(row["budgets"]["1024"]["fidelity"] >= .90 for _, _, row in cells),
        "N3_ranked_channels_beat_random": all(row["budgets"]["512"]["fidelity"] >= row["budgets"]["512"]["random"]["p95"] + .10 for _, _, row in cells),
        "N4_context_specific_channel_selection_is_material": any(row["budgets"]["512"]["oracle_fidelity"] >= row["budgets"]["512"]["fidelity"] + .05 for _, _, row in cells),
        "N5_static_bank_transfers_ood": all(results[name][str(layer)]["budgets"]["512"]["fidelity"] >= results["fineweb"][str(layer)]["budgets"]["512"]["fidelity"] - .05 for name in ("wikitext", "tinystories") for layer in config["layers"]),
        "N6_native_neuron_sparsity_is_sufficient": all(row["k90"] <= 512 for _, _, row in cells),
        "N7_some_local_operators_are_broad": any(row["budgets"]["1024"]["fidelity"] < .90 for _, _, row in cells),
        "N8_factor_swap_is_invariant": swap_error <= 1e-6,
    }
    hypotheses = {name: {"supported": bool(value)} for name, value in supported.items()}
    hypotheses["N8_factor_swap_is_invariant"]["maximum_relative_error"] = swap_error
    payload = {"config": raw, "hypotheses": hypotheses,
               "ranked_channel_prefixes": {str(layer): orderings[layer][:128].tolist() for layer in config["layers"]},
               "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Native-channel sparsity inside the compact 546M late-MLP operator", "",
             "At each of Layers 13–16, the exact local bilinear derivative on a frozen natural rank-32 input chart is expanded into all 4,608 native CP channels. A static bank is ranked only on FineWeb by its exact individual reduction in derivative error, then tested on disjoint FineWeb, WikiText, and TinyStories. The per-example bank is a hindsight ceiling. This is a local operator test, not yet an end-to-end suffix intervention.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Left/right factor-swap relative error: `{swap_error:.2e}`.", "",
              "| Corpus | Layer | Static 128 / 512 / 1024 / 2048 | Oracle512 | Random512 p95 | K90 |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus in results:
        for layer in config["layers"]:
            row = results[corpus][str(layer)]; budgets = row["budgets"]
            values = " / ".join(f"{budgets[str(k)]['fidelity']:.3f}" for k in (128, 512, 1024, 2048))
            lines.append(f"| {corpus} | {layer} | {values} | {budgets['512']['oracle_fidelity']:.3f} | {budgets['512']['random']['p95']:.3f} | {row['k90']} |")
    lines += ["", "A short `I/B` path description can still execute a dense native operator. Path-type sparsity and neuron sparsity answer different questions: the former compresses algebraic interaction order; the latter asks whether the operator itself is localized in the architecture's CP basis."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
