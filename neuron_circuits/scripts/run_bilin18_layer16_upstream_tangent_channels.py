#!/usr/bin/env python3
"""Test native Layer-16 loss breadth on tangents actually propagated from Layer 12."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_mlp_path_type_program import norm
from run_bilin12_router_analytic_mlp_jacobian import boundary
from run_bilin18_attention_layer_path_program import (
    mlp_post_jvp, native_cache, sequence_rms_jvp, two_qk_attention_jvp,
)
from run_bilin18_layer16_ce_aware_mode_selection import ce_gradient, score
from run_bilin18_layer16_ce_native_channels import marginal_ranking
from run_bilin18_layer16_ce_native_greedy_redteam import coordinated_greedy
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_upstream_tangent_channels.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_upstream_tangent_channels.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_UPSTREAM_TANGENT_CHANNELS.md"


@torch.inference_mode()
def build_input_axes(model, config):
    source = load_rows(config["axis_discovery"]); rows = []
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        x, _, _ = boundary(model, ids, config["layer"])
        rows.append(x[:, config["positions"]].float().flatten(0, 1).cpu())
    return torch.linalg.svd(torch.cat(rows), full_matrices=False).Vh[:config["rank"]].T.cuda()


@torch.inference_mode()
def propagate_to_layer16(model, x, cache, position, axes, config):
    outputs = []; maximum_forward_error = 0.0
    for start in range(0, axes.shape[1], config["tangent_chunk"]):
        selected = axes[:, start:start + config["tangent_chunk"]]
        tangent = torch.zeros(len(x), selected.shape[1], x.shape[1], x.shape[2], device="cuda")
        tangent[:, :, position] = selected.T[None]
        for row in cache:
            if row["index"] > 16: break
            block = model.blocks[row["index"]]
            mixed = block.lambdas[0].float() * tangent
            attention_input = sequence_rms_jvp(row["mixed"].float(), mixed)
            attention, error = two_qk_attention_jvp(
                block.attn, row["attention_input"].float(), row["bus"], attention_input)
            maximum_forward_error = max(maximum_forward_error, error)
            pre = mixed + attention
            state = sequence_rms_jvp(row["pre"].float(), pre)
            if row["index"] == 16:
                outputs.append(state[:, :, position].cpu())
                break
            tangent = mlp_post_jvp(block, row["pre"].float(), row["state"].float(), pre)
    return torch.cat(outputs, 1), maximum_forward_error


@torch.inference_mode(False)
def autograd_check(model, ids, axes, config):
    x, cache = native_cache(model, ids, config)
    manual, forward_error = propagate_to_layer16(
        model, x, cache, config["positions"][0], axes, config)
    native_x, x0, bus = boundary(model, ids, config["layer"])
    state = native_x.detach().clone(); fixed_x0 = x0.detach().clone(); fixed_bus = bus.detach().clone()
    tangent = torch.zeros_like(state); tangent[:, config["positions"][0]] = axes[:, 0]
    for index in range(13, 17):
        rotary = model.blocks[index].attn.rotary
        rotary.seq_len_cached = None; rotary.cos_cached = None; rotary.sin_cached = None

    def function(value):
        current = value; current_bus = fixed_bus
        for index in range(13, 17):
            block = model.blocks[index]
            mixed = block.lambdas[0] * current + block.lambdas[1] * fixed_x0
            attention, current_bus = block.attn(norm(mixed), current_bus)
            pre = mixed + attention; normalized = norm(pre)
            if index == 16: return normalized[:, config["positions"][0]]
            current = pre + block.mlp(normalized)
        raise RuntimeError

    _, expected = torch.func.jvp(function, (state,), (tangent,))
    error = float((manual[:, 0].cuda() - expected).double().norm()
                  / expected.double().norm().clamp_min(1e-30))
    return max(error, forward_error), error


@torch.inference_mode()
def channel_path(block, states, tangents, gradients):
    left_weight = block.mlp.Left.weight.float(); right_weight = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float(); x = states.float().cuda(); dx = tangents.float().cuda()
    left = x @ left_weight.T; right = x @ right_weight.T
    dleft = F.linear(dx, left_weight); dright = F.linear(dx, right_weight)
    gate = dleft * right[:, None] + dright * left[:, None]
    direct_write = F.linear(gate, down)
    target = (direct_write * gradients.float().cuda()[:, None]).sum(-1)
    sensitivity = gradients.float().cuda() @ down
    contribution = gate.transpose(1, 2) * sensitivity[:, :, None]
    error = float((contribution.sum(1) - target).double().norm()
                  / target.double().norm().clamp_min(1e-30))
    return contribution.cpu(), target.cpu(), error


@torch.inference_mode(False)
def collect(model, spec, axes, config):
    source = load_rows(spec); contributions = []; targets = []; maximum_forward_error = 0.0
    maximum_sum_error = 0.0
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        x, cache = native_cache(model, ids, config); by_layer = {row["index"]: row for row in cache}
        for position in config["positions"]:
            incoming, error = propagate_to_layer16(model, x, cache, position, axes, config)
            maximum_forward_error = max(maximum_forward_error, error)
            labels = torch.as_tensor(array[:, position + 1].astype(np.int64), device="cuda")
            gradient, _ = ce_gradient(model, by_layer[17], position, labels)
            contribution, target, error = channel_path(
                model.blocks[16], by_layer[16]["state"][:, position].cpu(), incoming, gradient.cpu())
            maximum_sum_error = max(maximum_sum_error, error)
            contributions.append(contribution); targets.append(target)
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    return {"contribution": torch.cat(contributions), "target": torch.cat(targets),
            "maximum_forward_error": maximum_forward_error,
            "maximum_sum_error": maximum_sum_error}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); local = json.loads(Path(config["local_pca_artifact"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    axes = build_input_axes(model, config)
    source = load_rows(config["selection_discovery"])
    ids = torch.as_tensor(np.asarray(source[:1, :128], dtype=np.int64), device="cuda")
    exact_error, autograd_error = autograd_check(model, ids, axes, config)
    print("selection discovery", flush=True)
    discovery = collect(model, config["selection_discovery"], axes, config)
    marginal_order = marginal_ranking(discovery["contribution"], discovery["target"]).tolist()
    greedy_order, discovery_curve = coordinated_greedy(discovery["contribution"], discovery["target"])
    maximum_sum_error = discovery["maximum_sum_error"]
    maximum_forward_error = max(exact_error, discovery["maximum_forward_error"]); results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = collect(model, spec, axes, config)
        maximum_sum_error = max(maximum_sum_error, data["maximum_sum_error"])
        maximum_forward_error = max(maximum_forward_error, data["maximum_forward_error"])
        rows = {}
        for budget in config["budgets"]:
            rows[str(budget)] = {
                "marginal_fidelity": score(data["target"], data["contribution"][:, marginal_order[:budget]].sum(1)),
                "greedy_fidelity": score(data["target"], data["contribution"][:, greedy_order[:budget]].sum(1)),
                "discovery_greedy_fidelity": discovery_curve[budget - 1],
            }
        k90 = next((budget for budget in config["budgets"] if rows[str(budget)]["greedy_fidelity"] >= .90), 4608)
        results[corpus] = {"sites": len(data["target"]), "greedy_k90": k90,
                           "local_pca_greedy_k90": local["results"][corpus]["k90"], "budgets": rows}
    hypotheses = {
        "U0_upstream_dynamic_jvp_is_exact": maximum_forward_error <= 1e-5,
        "U1_native_channel_path_sum_is_exact": maximum_sum_error <= 1e-5,
        "U2_marginal_512_is_sufficient_on_natural_tangents": all(row["budgets"]["512"]["marginal_fidelity"] >= .90 for row in results.values()),
        "U3_greedy_512_is_sufficient_on_natural_tangents": all(row["budgets"]["512"]["greedy_fidelity"] >= .90 for row in results.values()),
        "U4_greedy_natural_tangent_k90_is_at_most_1024": all(row["greedy_k90"] <= 1024 for row in results.values()),
        "U5_natural_tangents_are_materially_sparser_than_local_pca": all(row["greedy_k90"] <= 2048 for row in results.values()),
        "U6_natural_tangent_bank_transfers_ood": all(results[name]["budgets"]["512"]["greedy_fidelity"] >= results["fineweb"]["budgets"]["512"]["greedy_fidelity"] - .05 for name in ("wikitext", "tinystories")),
        "U7_coordination_materially_improves_natural_tangents": all(row["budgets"]["512"]["greedy_fidelity"] >= row["budgets"]["512"]["marginal_fidelity"] + .10 for row in results.values()),
        "U8_greedy_discovery_generalizes": results["fineweb"]["budgets"]["512"]["greedy_fidelity"] >= discovery_curve[511] - .10,
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["U0_upstream_dynamic_jvp_is_exact"].update({
        "autograd_relative_error": autograd_error,
        "maximum_error_including_native_forward": maximum_forward_error})
    hypothesis_rows["U1_native_channel_path_sum_is_exact"]["maximum_relative_error"] = maximum_sum_error
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "marginal_channel_order": marginal_order,
               "greedy_channel_order": greedy_order, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Native Layer-16 channel support on naturally propagated upstream tangents", "",
             "A natural rank-32 Layer-12 boundary basis is propagated through exact dynamic Layers 13--16 JVPs. At each site, the resulting Layer-16 state tangents replace the local PCA axes used previously. The Layer-16 MLP contribution to true-token CE is then decomposed exactly into all 4,608 native channels. Both marginal and coordinated greedy banks are frozen from discovery FineWeb.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Autograd upstream-JVP error: `{autograd_error:.2e}`; maximum channel-sum error: `{maximum_sum_error:.2e}`.", "",
              "| Corpus | K | Discovery greedy | Marginal | Greedy |", "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for budget in config["budgets"]:
            item = row["budgets"][str(budget)]
            lines.append(f"| {corpus} | {budget} | {item['discovery_greedy_fidelity']:.3f} | {item['marginal_fidelity']:.3f} | {item['greedy_fidelity']:.3f} |")
        lines.append(f"| {corpus} | K90 | — | — | {row['greedy_k90']} (local-PCA {row['local_pca_greedy_k90']}) |")
    lines += ["", "This isolates the Layer-16 MLP path, not the complete Layer-12-to-loss derivative. It asks whether the native neurons used by that path become sparse when the perturbation directions are generated by the actual upstream tensor network rather than chosen independently at Layer 16."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
