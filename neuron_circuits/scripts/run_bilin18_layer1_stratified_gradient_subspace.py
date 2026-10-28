#!/usr/bin/env python3
"""Position-stratified replication of the semantic suffix-gradient basis test."""

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
from run_bilin12_layer01_frozen_taylor_behavior import differentiable_suffix
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import clear_rotary
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_stratified_gradient_subspace.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_stratified_gradient_subspace.json"
ARRAYS = TENSOR_ROOT / "bilin18_layer1_stratified_gradient_subspace_arrays.npz"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_STRATIFIED_GRADIENT_SUBSPACE.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_stratified_gradient_subspace.png"


def row_cosine(left, right): return np.sum(left * right, axis=1) / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-30)


def batch_gradients(model, trace, layer, position, contrasts):
    decoder = model.blocks[layer].mlp.Down.weight.float(); batch = contrasts.shape[0]
    x0 = trace[layer]["x0"].detach().clone(); bus = trace[layer]["value_bus"].detach().clone(); output = trace[layer]["output"].detach().clone(); clear_rotary(model, layer + 1)
    with torch.enable_grad():
        variable = torch.zeros(batch, decoder.shape[1], device="cuda", requires_grad=True)
        mask = torch.zeros_like(output); mask[:, position] = F.linear(variable, decoder).to(output.dtype)
        logits = differentiable_suffix(model, layer, x0, bus, output + mask, position)[:, :contrasts.shape[1]]
        score = (logits.float() * contrasts).sum()
        gradient = torch.autograd.grad(score, variable)[0]
    return gradient.detach().float()


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive; contrast_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]
    contrast = torch.as_tensor(contrast_np, device="cuda", dtype=torch.float32); feature_indices = [FEATURES.index(name) for name in causal["features"]]
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); start = source["start_row"]; discovery_rows = corpus[start:start + source["discovery_rows"]]; rng = np.random.default_rng(config["seed"]); layer = causal["layer"]
    gradients, gradient_features, gradient_positions = [], [], []
    for position in source["positions"]:
        selected_rows = []
        for feature in feature_indices:
            candidates = np.flatnonzero(~flags[np.asarray(discovery_rows[:, position + 1], dtype=np.int64), feature].astype(bool))
            selected_rows.append(int(rng.choice(candidates)))
        ids = torch.as_tensor(np.asarray(discovery_rows[selected_rows, :source["sequence_length"]], dtype=np.int64), device="cuda")
        with torch.no_grad(): trace = forward_layers_native(model, ids, layer)
        values = batch_gradients(model, trace, layer, position, contrast[:, feature_indices].T)
        gradients.extend(values.cpu().numpy().astype(np.float16)); gradient_features.extend(feature_indices); gradient_positions.extend([position] * len(feature_indices)); print("stratified gradient position", position, flush=True)
    discovery = np.stack(gradients).astype(np.float32); np.savez_compressed(ARRAYS, discovery_gradients=discovery.astype(np.float16), discovery_features=np.asarray(gradient_features), discovery_positions=np.asarray(gradient_positions))
    evaluation_arrays = np.load(config["evaluation_arrays"]); evaluation = evaluation_arrays["evaluation_gradients"].astype(np.float32); deltas = evaluation_arrays["evaluation_deltas"].astype(np.float32)
    _, singular, vh = np.linalg.svd(discovery, full_matrices=False); exact_contribution = evaluation * deltas; k = config["support_rank"]; exact_support = np.argpartition(exact_contribution, -k, axis=1)[:, -k:]; exact_shift = np.take_along_axis(exact_contribution, exact_support, axis=1).sum(1)
    results = {}
    for rank in config["ranks"]:
        basis = vh[:rank].T; prediction = evaluation @ basis @ basis.T; contribution = prediction * deltas; support = np.argpartition(contribution, -k, axis=1)[:, -k:]; selected_true = np.take_along_axis(exact_contribution, support, axis=1).sum(1); jaccard = []
        for left, right in zip(exact_support, support):
            a, b = set(left.tolist()), set(right.tolist()); jaccard.append(len(a & b) / len(a | b))
        results[str(rank)] = {"gradient_cosine_median": float(np.median(row_cosine(evaluation, prediction))), "gradient_cosine_q10": float(np.quantile(row_cosine(evaluation, prediction), .1)), "contribution_cosine_median": float(np.median(row_cosine(exact_contribution, contribution))), "support_jaccard_median": float(np.median(jaccard)), "true_first_order_leverage_ratio_median": float(np.median(selected_true / np.maximum(exact_shift, 1e-30)))}
    rng = np.random.default_rng(config["seed"] + 1); random_basis, _ = np.linalg.qr(rng.standard_normal((evaluation.shape[1], 256)).astype(np.float32)); random_cosine = float(np.median(row_cosine(evaluation, evaluation @ random_basis @ random_basis.T)))
    summary = {"discovery_gradients": len(discovery), "positions": len(set(gradient_positions)), "random_rank256_cosine": random_cosine, "singular_values": singular.tolist(), "results": results}; rank32, rank64, rank128, rank256 = (results[str(rank)] for rank in (32,64,128,256))
    hypotheses = {"K0_rank64_transfer": {"supported": rank64["gradient_cosine_median"] >= .50}, "K1_rank128_transfer": {"supported": rank128["gradient_cosine_median"] >= .70}, "K2_rank256_transfer": {"supported": rank256["gradient_cosine_median"] >= .85}, "K3_rank256_contribution_transfer": {"supported": rank256["contribution_cosine_median"] >= .80}, "K4_rank256_support_transfer": {"supported": rank256["support_jaccard_median"] >= .40}, "K5_rank256_leverage_transfer": {"supported": rank256["true_first_order_leverage_ratio_median"] >= .70}, "K6_stratification_materially_improves_rank32": {"supported": rank32["gradient_cosine_median"] - .2520467936992645 >= .10}}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary}, indent=2, allow_nan=False))
    lines = ["# Position-stratified suffix-gradient subspace", "", "One discovery gradient is collected for each of eight semantic objectives at every one of the 32 evaluated sequence positions. The resulting 256-gradient span is tested on the previously frozen 96 held-out gradients and activation deltas.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Stratified gradient subspace](bilin18_layer1_stratified_gradient_subspace.png)", "", f"Random rank-256 projection cosine: **{random_cosine:.3f}**.", "", "| Rank | Gradient cosine | Q10 | Contribution cosine | Support Jaccard | Leverage ratio |", "|---:|---:|---:|---:|---:|---:|"]
    for rank in config["ranks"]:
        row = results[str(rank)]; lines.append(f"| {rank} | {row['gradient_cosine_median']:.3f} | {row['gradient_cosine_q10']:.3f} | {row['contribution_cosine_median']:.3f} | {row['support_jaccard_median']:.3f} | {row['true_first_order_leverage_ratio_median']:.3f} |")
    lines += ["", "## Guardrail", "", "The evaluation gradient is still projected to obtain coefficients, so this is a basis-existence test. It cannot establish a forward-state router even if the span transfers."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained"); ranks = config["ranks"]
    axes[0].plot(ranks, [results[str(r)]["gradient_cosine_median"] for r in ranks], marker="o", label="gradient"); axes[0].plot(ranks, [results[str(r)]["contribution_cosine_median"] for r in ranks], marker="o", label="delta × gradient"); axes[0].set(xlabel="Discovery basis rank", ylabel="Median cosine", title="Stratified held-out transfer"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(ranks, [results[str(r)]["support_jaccard_median"] for r in ranks], marker="o", label="support Jaccard"); axes[1].plot(ranks, [results[str(r)]["true_first_order_leverage_ratio_median"] for r in ranks], marker="o", label="leverage ratio"); axes[1].set(xlabel="Discovery basis rank", ylabel="Median", title="Dynamic-coalition preservation"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig); print(json.dumps({"hypotheses": hypotheses, "summary": {"random_rank256_cosine": random_cosine, "results": results}}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
