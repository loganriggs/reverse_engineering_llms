#!/usr/bin/env python3
"""Exact block-Jacobian factorization of grammar-model MLP causal charts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import collect_jacobians, correct


CONFIG = Path("configs/grammar_exact_factor_chain.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_exact_factor_chain.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_EXACT_FACTOR_CHAIN.md")


def rms_jacobian(x, weight, epsilon):
    if epsilon is None:
        epsilon = torch.finfo(x.dtype).eps
    width = x.shape[-1]; rms = (x.square().mean() + epsilon).sqrt()
    return torch.diag(weight / rms) - torch.outer(weight * x, x) / (width * rms ** 3)


@torch.no_grad()
def local_block_components(block, hidden):
    """Return exact QK/value/bilinear components of the local block Jacobian."""
    width = hidden.shape[-1]; final = hidden[-1]
    norm_jacobian = rms_jacobian(final, block.attn_norm.weight, block.attn_norm.eps)
    normalized = block.attn_norm(hidden)
    query, key, value = block.qkv(normalized).chunk(3, dim=-1)
    query = query.view(len(hidden), block.heads, block.head_dim)
    key = key.view(len(hidden), block.heads, block.head_dim)
    value = value.view(len(hidden), block.heads, block.head_dim)
    qkv_weight = block.qkv.weight.view(3, block.heads, block.head_dim, width)
    pattern_derivatives, value_derivatives = [], []
    for head in range(block.heads):
        q = query[-1, head]; keys = key[:, head]; values = value[:, head]
        attention = torch.softmax(keys @ q / math.sqrt(block.head_dim), 0)
        wq, wk, wv = qkv_weight[0, head], qkv_weight[1, head], qkv_weight[2, head]
        score_derivative = keys @ wq / math.sqrt(block.head_dim)
        score_derivative[-1] += q @ wk / math.sqrt(block.head_dim)
        softmax_derivative = torch.diag(attention) - torch.outer(attention, attention)
        pattern_derivative = softmax_derivative @ score_derivative
        pattern_derivatives.append(values.T @ pattern_derivative)
        value_derivatives.append(attention[-1] * wv)
    qk_component = block.attn_out.weight @ torch.cat(pattern_derivatives, 0) @ norm_jacobian
    value_component = block.attn_out.weight @ torch.cat(value_derivatives, 0) @ norm_jacobian
    after_attention_jacobian = torch.eye(width, device=hidden.device) + qk_component + value_component
    # Use the native block to obtain the exact post-attention state, then apply
    # the closed-form bilinear derivative there.
    normalized_all = block.attn_norm(hidden[None])
    q, k, v = block.qkv(normalized_all).chunk(3, dim=-1)
    def heads(tensor):
        return tensor.view(1, len(hidden), block.heads, block.head_dim).transpose(1, 2)
    attended = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True)
    attended = attended.transpose(1, 2).reshape(1, len(hidden), width)
    after_attention = hidden + block.attn_out(attended)[0]
    mlp_input = after_attention[-1]
    mlp_norm_jacobian = rms_jacobian(mlp_input, block.mlp_norm.weight, block.mlp_norm.eps)
    normalized_mlp = block.mlp_norm(mlp_input)
    left, right = block.left(normalized_mlp), block.right(normalized_mlp)
    left_arm = block.down.weight @ (right[:, None] * block.left.weight) @ mlp_norm_jacobian
    right_arm = block.down.weight @ (left[:, None] * block.right.weight) @ mlp_norm_jacobian
    mlp_jacobian = torch.eye(width, device=hidden.device) + left_arm + right_arm
    factor = mlp_jacobian @ after_attention_jacobian
    output, _ = block(hidden[None])
    return output[0], {"A": factor, "B": after_attention_jacobian, "C": mlp_jacobian,
                       "Q": qk_component, "V": value_component,
                       "ML": left_arm, "MR": right_arm,
                       "left": left, "right": right,
                       "mlp_norm_jacobian": mlp_norm_jacobian}


@torch.no_grad()
def local_block_factor(block, hidden):
    """Return d final-position block output / d final-position block input."""
    output, components = local_block_components(block, hidden)
    return output, components["A"]


@torch.no_grad()
def trace_factors(model, prompts, batch_size, device):
    factors = [[] for _ in model.blocks]; readouts = []; maximum_block_error = 0.0
    answer_weight = model.unembedding.weight[list(VOCAB.answers)]
    center = torch.eye(8, device=device) - torch.ones(8, 8, device=device) / 8
    for start in range(0, len(prompts), batch_size):
        tokens = torch.as_tensor(prompts[start:start + batch_size].astype(np.int64), device=device)
        hidden_batch = model.embedding(tokens) + model.position[:tokens.shape[1]]
        batch_outputs = [[] for _ in model.blocks]; batch_readouts = []
        for row in range(len(tokens)):
            hidden = hidden_batch[row]
            for layer, block in enumerate(model.blocks):
                analytic_output, factor = local_block_factor(block, hidden)
                native_output, _ = block(hidden[None])
                maximum_block_error = max(maximum_block_error, float((analytic_output - native_output[0]).abs().max()))
                batch_outputs[layer].append(factor.detach().cpu()); hidden = native_output[0]
            final = hidden[-1]
            output_factor = center @ answer_weight @ rms_jacobian(final, model.final_norm.weight, model.final_norm.eps)
            batch_readouts.append(output_factor.detach().cpu())
        for layer in range(model.layers_count): factors[layer].extend(batch_outputs[layer])
        readouts.extend(batch_readouts)
    return [torch.stack(rows) for rows in factors], torch.stack(readouts), maximum_block_error


def compose(readout, factors, down, source):
    result = down
    for layer in range(source + 1, len(factors)):
        result = factors[layer] @ result
    return readout @ result


def fidelity(actual, predicted):
    return 1 - float((actual - predicted).square().sum() / actual.square().sum().clamp_min(1e-30))


def effective_rank(values):
    flat = values.flatten(1); singular = torch.linalg.svdvals(flat - flat.mean(0)); energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-30)
    return float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))


def audit(spec, rho, indices, prompts, split, config, device):
    root = Path(spec["root"]); model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    jacobians, _, _, forward_error = collect_jacobians(model, prompts, config["batch_size"], device)
    factors, readout, block_error = trace_factors(model, prompts, config["batch_size"], device)
    downs = [block.down.weight.detach().cpu() for block in model.blocks]
    analytic = [torch.stack([compose(readout[row], [factor[row] for factor in factors], downs[source], source)
                             for row in range(len(prompts))]) for source in range(4)]
    exact_fidelity = [fidelity(jacobians[source], analytic[source]) for source in range(4)]
    factor_ranks = {f"A{layer}": effective_rank(factors[layer][:split]) for layer in range(4)}
    factor_ranks["O"] = effective_rank(readout[:split])
    means = [factor[:split].mean(0) for factor in factors]; mean_o = readout[:split].mean(0)
    source_results = []
    for source in range(4):
        actual = analytic[source][split:]; count = len(actual)
        def build(site_specific):
            predictions = []
            for row in range(count):
                chosen = [factors[layer][split + row] if f"A{layer}" in site_specific else means[layer]
                          for layer in range(4)]
                selected_o = readout[split + row] if "O" in site_specific else mean_o
                predictions.append(compose(selected_o, chosen, downs[source], source))
            return torch.stack(predictions)
        relevant = [f"A{layer}" for layer in range(source + 1, 4)] + ["O"]
        all_mean = fidelity(actual, build(set()))
        freeze_one = {factor: fidelity(actual, build(set(relevant) - {factor})) for factor in relevant}
        only_one = {factor: fidelity(actual, build({factor})) for factor in relevant}
        source_results.append({"source_layer": source, "all_mean_fidelity": all_mean,
                               "freeze_one_fidelity": freeze_one, "only_one_fidelity": only_one})
    return {"seed": spec["seed"], "rho": rho, "sites": len(prompts),
            "maximum_forward_error": forward_error, "maximum_block_error": block_error,
            "exact_fidelity": exact_fidelity, "factor_effective_rank": factor_ranks,
            "sources": source_results}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; runs = []
    for spec in config["roots"]:
        root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
        facts = np.asarray(manifest["train_facts"]); rule = labels_for_rho(manifest, "train", 0.0)[0]
        prompt = encode_prompts(manifest, facts, config["template"]); eligible = []
        for rho in config["rhos"]:
            model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
            labels = labels_for_rho(manifest, "train", rho)[0]; eligible.append(correct(model, prompt, labels, device))
        common = np.logical_and.reduce(eligible); rng = np.random.default_rng(config["seed"] + spec["seed"])
        discovery, evaluation = [], []
        for attribute in range(8):
            for answer in range(8):
                candidates = np.flatnonzero(common & (facts[:, 2] == attribute) & (rule == answer)); rng.shuffle(candidates)
                need = config["examples_per_attribute_rule_answer"]
                if len(candidates) < need: raise RuntimeError((spec["seed"], attribute, answer, len(candidates)))
                split = config["discovery_per_cell"]; discovery.extend(candidates[:split]); evaluation.extend(candidates[split:need])
        indices = np.asarray(discovery + evaluation); prompts = prompt[indices]
        for rho in config["rhos"]:
            print("audit", spec["seed"], rho, flush=True)
            runs.append(audit(spec, rho, indices, prompts, len(discovery), config, device))
    def get(seed, rho): return next(row for row in runs if row["seed"] == seed and row["rho"] == rho)
    f0 = all(min(row["exact_fidelity"]) >= .9999 and row["maximum_forward_error"] <= 1e-5
             and row["maximum_block_error"] <= 1e-5 for row in runs)
    f1 = all(get(seed, 0.0)["sources"][0]["all_mean_fidelity"]
             >= get(seed, .5)["sources"][0]["all_mean_fidelity"] + .3 for seed in (0, 1))
    f2 = all(sum(get(seed, .5)["factor_effective_rank"][f"A{layer}"]
                 >= 2 * get(seed, 0.0)["factor_effective_rank"][f"A{layer}"] for layer in (1, 2, 3)) >= 2
             for seed in (0, 1))
    f3 = all(row["sources"][0]["freeze_one_fidelity"]["O"] >= .9 for row in runs)
    f4 = all(min(row["sources"][0]["freeze_one_fidelity"].values()) <= .95
             for row in runs if row["rho"] == .5)
    f5 = all(max(row["sources"][0]["only_one_fidelity"].values()) < .8
             for row in runs if row["rho"] == .5)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (f0, f1, f2, f3, f4, f5))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Exact block-factor chain across rule and mixed regimes", "",
             "For final-position interventions, causality makes the local chain exact: centered answer readout O times downstream final-position block Jacobians A_k times the source MLP down-projection D_l. Every A_k is computed analytically from RMSNorm, softmax QK/value attention, and the bilinear MLP at the natural state.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | rho | Exact min | rank A1/A2/A3/O | Layer-0 all-mean | Freeze O | Worst freeze-one | Best only-one |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in runs:
        source = row["sources"][0]; ranks = row["factor_effective_rank"]
        lines.append(f"| {row['seed']} | {row['rho']} | {min(row['exact_fidelity']):.6f} | "
                     f"{ranks['A1']:.1f}/{ranks['A2']:.1f}/{ranks['A3']:.1f}/{ranks['O']:.1f} | "
                     f"{source['all_mean_fidelity']:.3f} | {source['freeze_one_fidelity']['O']:.3f} | "
                     f"{min(source['freeze_one_fidelity'].values()):.3f} | {max(source['only_one_fidelity'].values()):.3f} |")
    lines += ["", "All-mean asks whether one global chain suffices. Freeze-one asks which state-conditioned factor is necessary given all the others. Only-one asks whether a single varying factor is sufficient. These are exact operator reconstructions, not probes."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
