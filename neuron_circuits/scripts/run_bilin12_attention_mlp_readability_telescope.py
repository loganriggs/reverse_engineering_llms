#!/usr/bin/env python3
"""Measure when late-write coefficients become readable across half-blocks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_backbone_router_depth import (
    coefficient_metrics, fit_ridge, pooled_basis, predict,
)
from run_bilin12_boundary_bilinear_router import causal
from run_bilin12_boundary_output_interface import token_id
from run_bilin12_boundary_state_contraction import paired_panel
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_attention_mlp_readability_telescope.json")
OUTPUT = TENSOR_ROOT / "bilin12_attention_mlp_readability_telescope.json"
REPORT = TENSOR_ROOT / "BILIN12_ATTENTION_MLP_READABILITY_TELESCOPE.md"
FIGURE = TENSOR_ROOT / "bilin12_attention_mlp_readability_telescope.png"
STAGES = ("embedding",) + tuple(
    stage for layer in range(12) for stage in
    ((f"L{layer}.attention", f"L{layer}.mlp") if layer < 11 else (f"L{layer}.attention",))
)


def make_pair(base, changed):
    dimension = base.shape[-1]
    base = F.rms_norm(base, (dimension,)).float().cpu()
    changed = F.rms_norm(changed, (dimension,)).float().cpu()
    return torch.cat((base, changed - base), -1)


@torch.inference_mode()
def trace_stages(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    stages = {"embedding": x[:, -1]}
    final_pre = final_write = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        stages[f"L{layer}.attention"] = pre[:, -1]
        state = F.rms_norm(pre, (dimension,))
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + write
        if layer < 11:
            stages[f"L{layer}.mlp"] = x[:, -1]
        else:
            final_pre, final_write = pre[:, -1], write[:, -1]
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return stages, final_pre, final_write, 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def collect(model, spec, target, config):
    rows = load_rows(spec); features = {stage: [] for stage in STAGES}
    final_columns = [[] for _ in range(5)]
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda"); keep = ids[:, -1] != target
        if not bool(keep.any()): continue
        ids = ids[keep]; changed = ids.clone(); changed[:, -1] = target
        base_stages, base_pre, base_write, base_logits = trace_stages(model, ids)
        changed_stages, changed_pre, changed_write, _ = trace_stages(model, changed)
        for stage in STAGES:
            features[stage].append(make_pair(base_stages[stage], changed_stages[stage]))
        for store, value in zip(final_columns, (base_pre, base_write, base_logits,
                                                  changed_pre, changed_write)):
            store.append(value)
    return {stage: torch.cat(values) for stage, values in features.items()}, tuple(
        torch.cat(values) for values in final_columns)


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values)); ranks[order] = np.arange(len(values))
    return ranks


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    target = token_id(tokenizer, config["target"])
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    basis = pooled_basis(model, tokenizer, config)
    train_features, train_panel = collect(model, config["discovery"], target, config)
    train_coefficients = ((train_panel[4] - train_panel[1]).float() @ basis).cpu()
    fitted = {stage: fit_ridge(train_features[stage], train_coefficients,
                               config["ridge_relative_penalty"]) for stage in STAGES}
    results = {}; maximum_trace_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        features, panel = collect(model, spec, target, config)
        target_coefficients = ((panel[4] - panel[1]).float() @ basis).cpu()
        oracle = causal(model, panel, target_coefficients.to("cuda") @ basis.T)
        native = paired_panel(model, spec, 11, target, config["batch_size"])
        maximum_trace_error = max(maximum_trace_error, float((panel[2] - native[2]).abs().max()))
        stages = {}
        for stage in STAGES:
            coefficients = predict(fitted[stage], features[stage])
            metric = causal(model, panel, coefficients.to("cuda") @ basis.T)
            stages[stage] = {**coefficient_metrics(target_coefficients, coefficients), **metric}
        attention_jumps = {}; mlp_jumps = {}
        previous = "embedding"
        for layer in range(12):
            attention = f"L{layer}.attention"
            attention_jumps[str(layer)] = stages[attention]["fidelity"] - stages[previous]["fidelity"]
            if layer < 11:
                mlp = f"L{layer}.mlp"
                mlp_jumps[str(layer)] = stages[mlp]["fidelity"] - stages[attention]["fidelity"]
                previous = mlp
        results[corpus] = {"examples": len(target_coefficients), "oracle": oracle,
                           "stages": stages, "attention_jumps": attention_jumps,
                           "mlp_jumps": mlp_jumps}
    a0 = maximum_trace_error <= 1e-4
    a1 = results["wikitext"]["stages"]["L11.attention"]["fidelity"] >= \
         results["wikitext"]["stages"]["L10.mlp"]["fidelity"] + .10
    a2 = all(sum(value >= .02 for value in row["attention_jumps"].values()) >= 4
             for row in results.values())
    a3 = all(sum(value >= .02 for value in row["mlp_jumps"].values()) >= 4
             for row in results.values())
    a4 = all(max(row["attention_jumps"].values()) >= .05
             and max(row["mlp_jumps"].values()) >= .05 for row in results.values())
    a5 = all(row["stages"]["L11.attention"]["fidelity"] >= .70
             and row["stages"]["L11.attention"]["fidelity"] >= row["oracle"]["fidelity"] - .15
             for row in results.values())
    profiles = {corpus: [row["stages"][stage]["fidelity"] for stage in STAGES]
                for corpus, row in results.items()}
    correlations = {corpus: float(np.corrcoef(rankdata(profiles["fineweb"]),
                                               rankdata(profiles[corpus]))[0, 1])
                    for corpus in ("wikitext", "tinystories")}
    a6 = all(value >= .70 for value in correlations.values())
    hypotheses = {
        "A0_trace_is_valid": {"supported": a0, "maximum_trace_error": maximum_trace_error},
        "A1_final_attention_repairs_wikitext": {"supported": a1},
        "A2_attention_often_increases_readability": {"supported": a2},
        "A3_mlp_often_increases_readability": {"supported": a3},
        "A4_both_operator_types_have_material_jumps": {"supported": a4},
        "A5_final_pre_mlp_state_is_universally_readable": {"supported": a5},
        "A6_half_block_profile_transfers_ood": {"supported": a6, "correlations": correlations},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "results": results}, indent=2, allow_nan=False))
    x = np.arange(len(STAGES)); fig, axes = plt.subplots(1, 2, figsize=(17, 4.8))
    for corpus, row in results.items():
        axes[0].plot(x, [row["stages"][stage]["fidelity"] for stage in STAGES],
                     marker="o", ms=3, label=corpus)
        axes[1].plot(range(12), list(row["attention_jumps"].values()), marker="o",
                     label=f"{corpus} attention")
        axes[1].plot(range(11), list(row["mlp_jumps"].values()), marker="x", ls="--",
                     label=f"{corpus} MLP")
    axes[0].set_xticks(x, STAGES, rotation=70, fontsize=7); axes[0].axhline(.70, color="black", ls="--", lw=.8)
    axes[0].set_title("Frozen pair-state router fidelity")
    axes[1].axhline(0, color="black", lw=.8); axes[1].set_title("Change in readability by half-block")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    lines = ["# Attention/MLP readability telescope", "",
             "A single FineWeb-trained pair-state ridge readout is fit independently at the embedding boundary and after each attention or MLP half-block. Its target is the pooled rank-32 final-MLP write coefficient vector; predictions are scored by finite intervention, including on two OOD corpora.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Half-block telescope](bilin12_attention_mlp_readability_telescope.png)", "",
              "| Corpus | Embedding | L5 attention/MLP | L10 attention/MLP | L11 attention | Oracle | Max attention/MLP jump |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        stage = row["stages"]
        lines.append(f"| {corpus} | {stage['embedding']['fidelity']:.3f} | "
                     f"{stage['L5.attention']['fidelity']:.3f}/{stage['L5.mlp']['fidelity']:.3f} | "
                     f"{stage['L10.attention']['fidelity']:.3f}/{stage['L10.mlp']['fidelity']:.3f} | "
                     f"{stage['L11.attention']['fidelity']:.3f} | {row['oracle']['fidelity']:.3f} | "
                     f"{max(row['attention_jumps'].values()):.3f}/{max(row['mlp_jumps'].values()):.3f} |")
    lines += ["", "These are accessibility changes under separately trained frozen readouts, not additive causal credit. An attention jump means that the post-attention destination state makes the eventual program easier to predict; it does not prove that the newly readable information originated solely in that attention layer."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
