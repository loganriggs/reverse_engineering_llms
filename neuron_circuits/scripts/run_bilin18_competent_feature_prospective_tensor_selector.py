#!/usr/bin/env python3
"""Compress the exact context-conditioned selector with a short tensor program."""

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
from run_bilin18_competent_feature_causal_transfer import product_gradients_for_layers
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_competent_feature_prospective_tensor_selector.json")
OUTPUT = TENSOR_ROOT / "bilin18_competent_feature_prospective_tensor_selector.json"
REPORT = TENSOR_ROOT / "BILIN18_COMPETENT_FEATURE_PROSPECTIVE_TENSOR_SELECTOR.md"
FIGURE = TENSOR_ROOT / "bilin18_competent_feature_prospective_tensor_selector.png"
CHECKPOINT = TENSOR_ROOT / "bilin18_competent_feature_prospective_tensor_selector.partial.json"


def program_masks(source_layer, block_count, widths):
    suffix = list(range(source_layer + 1, block_count)); result = {"mlp_only": set(), "all_attention": set(suffix)}
    for width in widths: result[f"near_{width}"] = set(suffix[:width])
    result["late_8"] = set(suffix[-8:])
    return result


def masked_attention_program(model, trace, source_layer, position, contrast, masks, prediction_position=None):
    """Compose a suffix VJP and read it at one source-product position.

    ``prediction_position`` defaults to ``position`` for the original
    same-position assay.  Keeping the two coordinates distinct lets the same
    recurrence serve cross-position and multi-source tests without duplicating
    the tensor program.
    """
    if prediction_position is None:
        prediction_position = position
    names = list(masks); current = final_adjoint(model, trace[-1]["output"], prediction_position, contrast).repeat(len(names), 1, 1, 1)
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]; updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current); enabled = [row for row, name in enumerate(names) if layer in masks[name]]
        if enabled:
            previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float(); mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0; updated[enabled] += attention_vjp(block, mixed, trace[layer - 1]["value_bus"], updated[enabled])
        current = block.lambdas[0].float() * updated
    down = model.blocks[source_layer].mlp.Down.weight.float()
    return {name: current[row, 0, position] @ down for row, name in enumerate(names)}


def summarize(records, layer, method):
    rows = [row for row in records if row["layer"] == layer]; values = [row["metrics"][method] for row in rows]; shifts = np.asarray([row["finite_shifts"][method] for row in rows]); oracle = np.asarray([row["finite_shifts"]["all_attention"] for row in rows])
    return {"gradient_cosine_median": float(np.median([value["gradient_cosine"] for value in values])), "gradient_cosine_q10": float(np.quantile([value["gradient_cosine"] for value in values], .1)), "contribution_cosine_median": float(np.median([value["contribution_cosine"] for value in values])), "support_jaccard_median": float(np.median([value["support_jaccard"] for value in values])), "true_leverage_ratio_median": float(np.median([value["true_leverage_ratio"] for value in values])), "finite_directional_fraction": float(np.mean(shifts > 0)), "finite_mean_shift": float(shifts.mean()), "finite_oracle_effect_ratio": float(shifts.mean() / max(oracle.mean(), 1e-30))}


def selector_metrics(prediction, exact, delta, rank):
    value = metrics(prediction, exact, delta, rank); predicted = set((prediction * delta).topk(rank).indices.cpu().tolist()); target = set((exact * delta).topk(rank).indices.cpu().tolist()); value["support_jaccard"] = len(predicted & target) / len(predicted | target)
    return value


def main():
    config = json.loads(CONFIG.read_text()); source_results = json.loads(Path(config["source_results"]).read_text()); census_config = json.loads(Path(config["census_config"]).read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive; contrasts = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(census_config["corpus"], mmap_mode="r"); heldout_start = census_config["start_row"] + census_config["discovery_rows"]; heldout = corpus[heldout_start:heldout_start + census_config["heldout_rows"]]
    pairs = [{key: row[key] for key in ("label", "feature_index", "selection", "position", "donor_row", "recipient_row", "model_margin", "feature")} for row in source_results["records"] if row["layer"] == config["source_layers"][0]]
    all_widths = sorted(set(config["near_widths"] + config.get("posthoc_near_widths", []))); signature = {"widths": all_widths, "source_layers": config["source_layers"], "pair_count": len(pairs)}
    if CHECKPOINT.exists():
        checkpoint = json.loads(CHECKPOINT.read_text())
        if checkpoint.get("signature") == signature: records = checkpoint.get("records", []); maximum_autograd_error = checkpoint.get("maximum_autograd_error", 0.0)
        else: records = []; maximum_autograd_error = 0.0
    else: records = []; maximum_autograd_error = 0.0
    completed_pairs = len(records) // len(config["source_layers"])
    for pair_index, pair in enumerate(pairs):
        if pair_index < completed_pairs: continue
        donor_ids = torch.as_tensor(np.asarray(heldout[pair["donor_row"], :census_config["sequence_length"]], dtype=np.int64), device="cuda")[None]; recipient_ids = torch.as_tensor(np.asarray(heldout[pair["recipient_row"], :census_config["sequence_length"]], dtype=np.int64), device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"])); recipient_trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        contrast = contrasts[:, pair["feature_index"]]; exact_gradients = product_gradients_for_layers(model, recipient_trace, pair["position"], contrast, config["source_layers"])
        for layer in config["source_layers"]:
            delta = product_at(model, donor_trace, layer, pair["position"]) - product_at(model, recipient_trace, layer, pair["position"]); masks = program_masks(layer, len(model.blocks), all_widths); predictions = masked_attention_program(model, recipient_trace, layer, pair["position"], contrast, masks); exact = exact_gradients[layer]
            if pair_index < config["autograd_check_pairs"]:
                reference = semantic_gradient(model, recipient_trace, layer, pair["position"], contrast); maximum_autograd_error = max(maximum_autograd_error, float((predictions["all_attention"] - reference).norm() / reference.norm().clamp_min(1e-30)))
            patch_deltas = {"base": torch.zeros_like(delta)}
            for name, prediction in predictions.items(): patch_deltas[name] = selected_delta(delta, (prediction * delta).topk(config["support_rank"]).indices)
            with torch.no_grad(): logits = logits_for_deltas(model, layer, recipient_trace[layer], patch_deltas, pair["position"])
            base_score = class_score(logits["base"], contrast); finite = {name: class_score(value, contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({**pair, "layer": layer, "metrics": {name: selector_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in predictions.items()}, "finite_shifts": finite})
        CHECKPOINT.write_text(json.dumps({"signature": signature, "maximum_autograd_error": maximum_autograd_error, "records": records}, allow_nan=False))
        print(f"prospective tensor pair {pair_index + 1}/{len(pairs)} {pair['feature']} {pair['selection']}", flush=True)
    methods = list(records[0]["metrics"]); summaries = {str(layer): {method: summarize(records, layer, method) for method in methods} for layer in config["source_layers"]}
    hypotheses = {"P0_exact_composition_matches_autograd": {"supported": maximum_autograd_error <= 1e-5}, "P1_near8_gradient_fidelity": {"supported": all(summaries[str(layer)]["near_8"]["gradient_cosine_median"] >= .90 for layer in config["source_layers"])}, "P2_near8_causal_direction": {"supported": all(summaries[str(layer)]["near_8"]["finite_directional_fraction"] >= .95 for layer in config["source_layers"])}, "P3_near8_retains_oracle_effect": {"supported": all(summaries[str(layer)]["near_8"]["finite_oracle_effect_ratio"] >= .80 for layer in config["source_layers"])}, "P4_near4_is_insufficient": {"supported": any(summaries[str(layer)]["near_4"]["gradient_cosine_median"] < .90 or summaries[str(layer)]["near_4"]["finite_directional_fraction"] < .90 for layer in config["source_layers"])}, "P5_near_attention_beats_late_attention": {"supported": all(summaries[str(layer)]["near_8"]["gradient_cosine_median"] - summaries[str(layer)]["late_8"]["gradient_cosine_median"] >= .10 for layer in config["source_layers"])}, "P6_mlp_only_is_insufficient": {"supported": all(summaries[str(layer)]["mlp_only"]["gradient_cosine_median"] < .85 for layer in config["source_layers"])}, "P7_near8_recovers_oracle_support": {"supported": all(summaries[str(layer)]["near_8"]["support_jaccard_median"] >= .25 for layer in config["source_layers"])}}
    posthoc_minimum_width = {}
    for layer in config["source_layers"]:
        passing = [width for width in all_widths if summaries[str(layer)][f"near_{width}"]["gradient_cosine_median"] >= .90 and summaries[str(layer)][f"near_{width}"]["gradient_cosine_q10"] >= .80 and summaries[str(layer)][f"near_{width}"]["support_jaccard_median"] >= .50 and summaries[str(layer)][f"near_{width}"]["finite_directional_fraction"] >= .95 and summaries[str(layer)][f"near_{width}"]["finite_oracle_effect_ratio"] >= .90]
        posthoc_minimum_width[str(layer)] = min(passing) if passing else None
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "posthoc_minimum_width": posthoc_minimum_width, "maximum_autograd_error": maximum_autograd_error, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    CHECKPOINT.write_text(json.dumps({"complete": True, "signature": signature, "maximum_autograd_error": maximum_autograd_error, "records": records}, allow_nan=False))
    lines = ["# Competent-feature prospective tensor selector", "", "The exact recipient-specific selector is compressed by retaining native state-conditioned MLP Jacobians throughout the suffix while enabling exact attention Jacobians only in a short window after each source layer. All 32 competent features and the same 64 held-out causal pairs are used; each approximate gradient selects its own top-64 product channels before a finite suffix recomputation.", "", f"Maximum all-attention recurrence error against independent autograd checks: `{maximum_autograd_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Prospective selector](bilin18_competent_feature_prospective_tensor_selector.png)", "", "| Layer | Method | Gradient cosine | Top-64 Jaccard | Leverage | Finite direction | Mean shift | Oracle-effect ratio |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for method in ("mlp_only", "near_2", "near_4", "near_8", "near_12", "late_8", "all_attention"):
            row = summaries[str(layer)][method]; lines.append(f"| {layer} | {method} | {row['gradient_cosine_median']:.3f} | {row['support_jaccard_median']:.3f} | {row['true_leverage_ratio_median']:.3f} | {row['finite_directional_fraction']:.3f} | {row['finite_mean_shift']:.4f} | {row['finite_oracle_effect_ratio']:.3f} |")
    lines += ["", "## Post-hoc minimum near-source window", "", "A width passes only if median gradient cosine is at least 0.90, Q10 at least 0.80, top-64 Jaccard at least 0.50, finite direction at least 0.95, and mean finite effect at least 0.90 of the exact selector.", "", "| Source layer | Minimum tested width |", "|---:|---:|"]
    for layer in config["source_layers"]: lines.append(f"| {layer} | {posthoc_minimum_width[str(layer)]} |")
    lines += ["", "## Guardrails", "", "This is prospective with respect to channel selection but still conditions on the declared semantic objective, donor activation change, and recipient forward states. It is a compressed executable explanation, not an unsupervised ontology detector. Every MLP Jacobian remains exact; the compression tested here concerns how many attention Jacobians must be evaluated. The all-attention program is mathematically the exact local gradient and serves as the endpoint upper control."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["source_layers"])); fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for method in ("mlp_only", "near_4", "near_8", "near_12", "late_8"): axes[0].plot(x, [summaries[str(layer)][method]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=method)
    axes[0].set_xticks(x, config["source_layers"]); axes[0].set(xlabel="Source layer", ylabel="Median gradient cosine", title="Short tensor programs"); axes[0].legend(); axes[0].grid(alpha=.2)
    for method in ("mlp_only", "near_4", "near_8", "near_12", "all_attention"): axes[1].plot(x, [summaries[str(layer)][method]["finite_directional_fraction"] for layer in config["source_layers"]], marker="o", label=method)
    axes[1].set_xticks(x, config["source_layers"]); axes[1].set(xlabel="Source layer", ylabel="Positive top-64 finite fraction", title="Prospective causal selection"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "maximum_autograd_error": maximum_autograd_error, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
