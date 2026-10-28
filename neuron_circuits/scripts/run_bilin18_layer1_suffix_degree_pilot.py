#!/usr/bin/env python3
"""Exact nonlinear-degree expansion of the Layer-1-to-logit backward program."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import clear_rotary, semantic_gradient
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_degree_pilot.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_degree_pilot.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_DEGREE_PILOT.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_degree_pilot.png"


def rms_vjp(state, adjoint):
    root = state.square().mean(-1, keepdim=True).sqrt()
    coupling = (state[None] * adjoint).mean(-1, keepdim=True)
    return adjoint / root[None] - state[None] * coupling / root[None].pow(3)


def mlp_vjp(block, pre, adjoint):
    state = F.rms_norm(pre, (pre.shape[-1],))
    left = block.mlp.Left(state).float(); right = block.mlp.Right(state).float()
    channel = torch.einsum("kbsd,dc->kbsc", adjoint, block.mlp.Down.weight.float())
    state_adjoint = torch.einsum("kbsc,bsc,cd->kbsd", channel, right, block.mlp.Left.weight.float()) + torch.einsum("kbsc,bsc,cd->kbsd", channel, left, block.mlp.Right.weight.float())
    return rms_vjp(pre.float(), state_adjoint)


def attention_vjp(block, mixed, bus, adjoints):
    clear_rotary(type("Wrapper", (), {"blocks": [block]})(), 0)
    with torch.enable_grad():
        variable = mixed.detach().clone().requires_grad_(True)
        attention, _ = block.attn(F.rms_norm(variable, (variable.shape[-1],)), bus.detach().clone())
        gradient, = torch.autograd.grad(attention, variable, adjoints, is_grads_batched=True)
    return gradient.detach().float()


def final_adjoint(model, final_state, position, contrast):
    with torch.enable_grad():
        variable = final_state.detach().clone().requires_grad_(True)
        raw = model.lm_head(F.rms_norm(variable, (variable.shape[-1],))[:, position]).float()
        logits = 30 * torch.tanh(raw / 30)
        score = logits[0, :contrast.numel()] @ contrast
        gradient, = torch.autograd.grad(score, variable)
    return gradient.detach().float()


def degree_program(model, trace, position, contrast, maximum_degree):
    current = torch.zeros(maximum_degree + 1, *trace[-1]["output"].shape, device="cuda")
    current[0] = final_adjoint(model, trace[-1]["output"], position, contrast)
    for index in range(len(model.blocks) - 1, 1, -1):
        block = model.blocks[index]; previous = trace[index - 1]["output"].float(); x0 = trace[index]["x0"].float(); mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0; bus = trace[index - 1]["value_bus"]
        mlp_rows = mlp_vjp(block, trace[index]["pre"].float(), current)
        attention_inputs = torch.cat((current[:-1], mlp_rows[:-2] if maximum_degree >= 2 else current[:0]), 0)
        attention_rows = attention_vjp(block, mixed, bus, attention_inputs) if len(attention_inputs) else attention_inputs
        attention_plain = attention_rows[:maximum_degree]; attention_mlp = attention_rows[maximum_degree:]
        updated = block.lambdas[0].float() * current
        updated[1:] += block.lambdas[0].float() * mlp_rows[:-1]
        updated[1:] += block.lambdas[0].float() * attention_plain
        if maximum_degree >= 2: updated[2:] += block.lambdas[0].float() * attention_mlp
        current = updated
    down = model.blocks[1].mlp.Down.weight.float()
    return torch.einsum("kd,dc->kc", current[:, 0, position], down)


def cosine(left, right): return float((left @ right) / (left.norm() * right.norm()).clamp_min(1e-30))


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive; contrast_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]; contrast = torch.as_tensor(contrast_np, device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]
    selected = []
    for feature in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature]; selected.extend((next(row for row in rows if row["selection"] == "extreme"), next(row for row in rows if row["selection"] == "uniform")))
    records = []; max_degree = config["degrees"][-1]; k = config["support_rank"]
    for site, row in enumerate(selected):
        feature = FEATURES.index(row["feature"]); vector = contrast[:, feature]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64); recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, 17); recipient_trace = forward_layers_native(model, recipient_ids, 17)
        delta = product_at(model, donor_trace, 1, row["donor_position"]) - product_at(model, recipient_trace, 1, row["recipient_position"])
        exact = semantic_gradient(model, recipient_trace, 1, row["recipient_position"], vector)
        components = degree_program(model, recipient_trace, row["recipient_position"], vector, max_degree)
        exact_contribution = exact * delta; exact_support = exact_contribution.topk(k).indices; exact_leverage = exact_contribution[exact_support].sum()
        metrics = {}
        for degree in config["degrees"]:
            prediction = components[:degree + 1].sum(0); contribution = prediction * delta; support = contribution.topk(k).indices
            metrics[str(degree)] = {"gradient_cosine": cosine(prediction, exact), "gradient_relative_error": float((prediction - exact).norm() / exact.norm().clamp_min(1e-30)), "contribution_cosine": cosine(contribution, exact_contribution), "support_jaccard": len(set(support.cpu().tolist()) & set(exact_support.cpu().tolist())) / len(set(support.cpu().tolist()) | set(exact_support.cpu().tolist())), "true_leverage_ratio": float(exact_contribution[support].sum() / exact_leverage.clamp_min(1e-30))}
        degree_norms = [float(component.norm()) for component in components]
        records.append({"feature": row["feature"], "selection": row["selection"], "metrics": metrics, "degree_norms": degree_norms})
        print("suffix degree site", site + 1, "/", len(selected), row["feature"], row["selection"], flush=True)
    summaries = {}
    for degree in config["degrees"]:
        key = str(degree); summaries[key] = {metric: float(np.median([row["metrics"][key][metric] for row in records])) for metric in ("gradient_cosine", "gradient_relative_error", "contribution_cosine", "support_jaccard", "true_leverage_ratio")}
    per_feature8 = {feature: float(np.median([row["metrics"]["8"]["gradient_cosine"] for row in records if row["feature"] == feature])) for feature in causal["features"]}; full = summaries[str(max_degree)]; degree4 = summaries["4"]; degree8 = summaries["8"]
    hypotheses = {"E0_full_degree_recomposition_is_exact": {"supported": all(row["metrics"][str(max_degree)]["gradient_cosine"] >= .999 and row["metrics"][str(max_degree)]["gradient_relative_error"] <= .02 for row in records)}, "E1_degree4_is_useful": {"supported": degree4["gradient_cosine"] >= .70}, "E2_degree8_is_high_fidelity": {"supported": degree8["gradient_cosine"] >= .90}, "E3_degree8_preserves_contributions": {"supported": degree8["contribution_cosine"] >= .85}, "E4_degree8_preserves_sparse_leverage": {"supported": degree8["true_leverage_ratio"] >= .80}, "E5_degree8_transfers_across_features": {"supported": all(value >= .80 for value in per_feature8.values())}}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "per_feature_degree8": per_feature8, "records": records}, indent=2, allow_nan=False))
    lines = ["# Layer-1 semantic suffix backward-degree pilot", "", "Each Layer-2–17 local backward operator is decomposed exactly into identity, MLP, attention, and attention-after-MLP terms. The exponentially many paths are grouped by total nonlinear interaction degree and accumulated without choosing layers or fitting evaluation effects.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Backward degree pilot](bilin18_layer1_suffix_degree_pilot.png)", "", "| Degree ≤ | Gradient cosine | Relative error | Contribution cosine | Support Jaccard | Leverage ratio |", "|---:|---:|---:|---:|---:|---:|"]
    for degree in config["degrees"]:
        row = summaries[str(degree)]; lines.append(f"| {degree} | {row['gradient_cosine']:.3f} | {row['gradient_relative_error']:.3f} | {row['contribution_cosine']:.3f} | {row['support_jaccard']:.3f} | {row['true_leverage_ratio']:.3f} |")
    lines += ["", "## Guardrail", "", "This pilot uses sixteen sites, one extreme and one uniform pair per feature. Exact degree-32 recomposition is the implementation gate; lower-degree conclusions require a larger held-out replication and finite patching."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained"); degrees = config["degrees"]
    axes[0].plot(degrees, [summaries[str(d)]["gradient_cosine"] for d in degrees], marker="o", label="gradient"); axes[0].plot(degrees, [summaries[str(d)]["contribution_cosine"] for d in degrees], marker="o", label="delta × gradient"); axes[0].set(xlabel="Maximum nonlinear degree", ylabel="Median cosine", title="Backward tensor-program convergence"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(degrees, [summaries[str(d)]["support_jaccard"] for d in degrees], marker="o", label="top-64 Jaccard"); axes[1].plot(degrees, [summaries[str(d)]["true_leverage_ratio"] for d in degrees], marker="o", label="first-order leverage"); axes[1].set(xlabel="Maximum nonlinear degree", ylabel="Median", title="Semantic coalition preservation"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig); print(json.dumps({"hypotheses": hypotheses, "summaries": summaries, "per_feature_degree8": per_feature8}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
