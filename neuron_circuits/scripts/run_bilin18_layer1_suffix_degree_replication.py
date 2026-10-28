#!/usr/bin/env python3
"""Full finite replication of the semantic suffix interaction-degree program."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_degree_pilot import cosine, degree_program
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_degree_replication.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_degree_replication.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_DEGREE_REPLICATION.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_degree_replication.png"


def aggregate(records, method):
    shifts = np.asarray([row["finite_shifts"][method] for row in records]); return {"directional_fraction": float(np.mean(shifts > 0)), "mean_shift": float(shifts.mean()), "median_shift": float(np.median(shifts)), "mean_absolute_shift": float(np.abs(shifts).mean())}


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive; contrast_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]; contrast = torch.as_tensor(contrast_np, device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]; k = config["support_rank"]; max_degree = config["degrees"][-1]; records = []
    for site, row in enumerate(old["records"]):
        feature = FEATURES.index(row["feature"]); vector = contrast[:, feature]; donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64); recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64); donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, 1); recipient_trace = forward_layers_native(model, recipient_ids, 17)
        delta = product_at(model, donor_trace, 1, row["donor_position"]) - product_at(model, recipient_trace, 1, row["recipient_position"]); exact = semantic_gradient(model, recipient_trace, 1, row["recipient_position"], vector); components = degree_program(model, recipient_trace, row["recipient_position"], vector, max_degree); exact_contribution = exact * delta; exact_support = exact_contribution.topk(k).indices; exact_leverage = exact_contribution[exact_support].sum()
        predictions = {str(degree): components[:degree + 1].sum(0) for degree in config["degrees"]}; supports = {str(degree): (prediction * delta).topk(k).indices for degree, prediction in ((degree, predictions[str(degree)]) for degree in config["degrees"])}; supports["exact"] = exact_support
        deltas = {"base": torch.zeros_like(delta), **{f"degree_{degree}": selected_delta(delta, supports[str(degree)]) for degree in config["degrees"]}, "exact": selected_delta(delta, exact_support)}
        with torch.no_grad(): logits = logits_for_deltas(model, 1, recipient_trace[1], deltas, row["recipient_position"])
        base = class_score(logits["base"], vector); finite = {name: class_score(value, vector) - base for name, value in logits.items() if name != "base"}; metrics = {}
        for degree in config["degrees"]:
            prediction = predictions[str(degree)]; contribution = prediction * delta; support = supports[str(degree)]; metrics[str(degree)] = {"gradient_cosine": cosine(prediction, exact), "gradient_relative_error": float((prediction - exact).norm() / exact.norm().clamp_min(1e-30)), "contribution_cosine": cosine(contribution, exact_contribution), "true_leverage_ratio": float(exact_contribution[support].sum() / exact_leverage.clamp_min(1e-30))}
        records.append({"feature": row["feature"], "selection": row["selection"], "metrics": metrics, "finite_shifts": finite})
        if (site + 1) % 8 == 0: print("degree replication", site + 1, "/", len(old["records"]), flush=True)
    summaries = {}
    for degree in config["degrees"]:
        key = str(degree); values = [row["metrics"][key] for row in records]; summaries[key] = {metric + "_median": float(np.median([value[metric] for value in values])) for metric in ("gradient_cosine", "gradient_relative_error", "contribution_cosine", "true_leverage_ratio")}; summaries[key]["gradient_cosine_q10"] = float(np.quantile([value["gradient_cosine"] for value in values], .1)); summaries[key]["finite"] = aggregate(records, f"degree_{degree}")
    exact_finite = aggregate(records, "exact"); per_feature12 = {feature: {"gradient_cosine_median": float(np.median([row["metrics"]["12"]["gradient_cosine"] for row in records if row["feature"] == feature])), "finite_directional_fraction": float(np.mean([row["finite_shifts"]["degree_12"] > 0 for row in records if row["feature"] == feature]))} for feature in causal["features"]}; d8, d12, d16, d24 = (summaries[str(degree)] for degree in (8,12,16,24))
    hypotheses = {"R0_degree24_recomposition_is_exact": {"supported": all(row["metrics"]["24"]["gradient_cosine"] >= .999 and row["metrics"]["24"]["gradient_relative_error"] <= .02 for row in records)}, "R1_degree12_gradient_replication": {"supported": d12["gradient_cosine_median"] >= .90 and d12["gradient_cosine_q10"] >= .80}, "R2_degree12_finite_coalitions_are_directional": {"supported": d12["finite"]["directional_fraction"] >= .90}, "R3_degree12_preserves_causal_strength": {"supported": d12["finite"]["mean_shift"] >= .70 * exact_finite["mean_shift"]}, "R4_degree16_matches_exact_coalitions": {"supported": abs(d16["finite"]["directional_fraction"] - exact_finite["directional_fraction"]) <= .02 and abs(d16["finite"]["mean_shift"] / exact_finite["mean_shift"] - 1) <= .10}, "R5_degree12_transfers_across_features": {"supported": all(value["gradient_cosine_median"] >= .80 and value["finite_directional_fraction"] >= .75 for value in per_feature12.values())}, "R6_degree8_remains_an_insufficient_control": {"supported": d8["gradient_cosine_median"] < .80 or d8["finite"]["directional_fraction"] < .80}}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "exact_finite": exact_finite, "per_feature_degree12": per_feature12, "records": records}, indent=2, allow_nan=False))
    lines = ["# Full replication of the Layer-1 suffix backward-degree program", "", "The exact identity/MLP/attention/attention-after-MLP expansion is evaluated on all 96 frozen semantic donor–recipient pairs. For each degree cutoff, its predicted gradient selects 64 positive-contribution native channels, which are then patched finitely through the complete nonlinear suffix.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Degree replication](bilin18_layer1_suffix_degree_replication.png)", "", f"Exact-gradient top-64 finite direction: **{exact_finite['directional_fraction']:.3f}**; mean shift: **{exact_finite['mean_shift']:+.4f}**.", "", "| Degree ≤ | Grad. cosine / Q10 | Contribution cosine | Leverage | Finite direction | Mean shift |", "|---:|---:|---:|---:|---:|---:|"]
    for degree in config["degrees"]:
        row = summaries[str(degree)]; lines.append(f"| {degree} | {row['gradient_cosine_median']:.3f}/{row['gradient_cosine_q10']:.3f} | {row['contribution_cosine_median']:.3f} | {row['true_leverage_ratio_median']:.3f} | {row['finite']['directional_fraction']:.3f} | {row['finite']['mean_shift']:+.4f} |")
    lines += ["", "## Interpretation guardrail", "", "Interaction degree is program complexity, not neuron count. A degree-12 cutoff still executes the native local tensor maps across sixteen blocks and sums many layer paths; its value is an invariant algebraic truncation, not a claim that twelve neurons or twelve literal circuits suffice."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained"); degrees = config["degrees"]
    axes[0].plot(degrees, [summaries[str(d)]["gradient_cosine_median"] for d in degrees], marker="o", label="gradient"); axes[0].plot(degrees, [summaries[str(d)]["contribution_cosine_median"] for d in degrees], marker="o", label="delta × gradient"); axes[0].set(xlabel="Maximum nonlinear degree", ylabel="Median cosine", title="96-site backward-program convergence"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(degrees, [summaries[str(d)]["finite"]["directional_fraction"] for d in degrees], marker="o", label="directional fraction"); axes[1].plot(degrees, [summaries[str(d)]["finite"]["mean_shift"] / exact_finite["mean_shift"] for d in degrees], marker="o", label="mean-shift ratio"); axes[1].axhline(1, color="gray", linestyle=":"); axes[1].set(xlabel="Maximum nonlinear degree", ylabel="Relative finite behavior", title="Causal validation of selected coalitions"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig); print(json.dumps({"hypotheses": hypotheses, "summaries": summaries, "exact_finite": exact_finite, "per_feature_degree12": per_feature12}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
