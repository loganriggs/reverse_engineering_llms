#!/usr/bin/env python3
"""Factor the successful near-source selector into bilinear-attention roles."""

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
from run_bilin18_competent_feature_prospective_tensor_selector import selector_metrics
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_suffix_qk_role_pilot import ROLES, attention_role_vjps
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_competent_feature_attention_role_selector.json")
OUTPUT = TENSOR_ROOT / "bilin18_competent_feature_attention_role_selector.json"
REPORT = TENSOR_ROOT / "BILIN18_COMPETENT_FEATURE_ATTENTION_ROLE_SELECTOR.md"
FIGURE = TENSOR_ROOT / "bilin18_competent_feature_attention_role_selector.png"
CHECKPOINT = TENSOR_ROOT / "bilin18_competent_feature_attention_role_selector.partial.json"


def specifications(source_layer, block_count, width):
    near = set(range(source_layer + 1, min(block_count, source_layer + 1 + width))); all_layers = set(range(source_layer + 1, block_count))
    return {"mlp_only": {"layers": set(), "roles": set()}, "near_queries": {"layers": near, "roles": {"q1", "q2"}}, "near_keys": {"layers": near, "roles": {"k1", "k2"}}, "near_pair1": {"layers": near, "roles": {"q1", "k1"}}, "near_pair2": {"layers": near, "roles": {"q2", "k2"}}, "near_qk": {"layers": near, "roles": {"q1", "k1", "q2", "k2"}}, "near_value": {"layers": near, "roles": {"value"}}, "near_full": {"layers": near, "roles": set(ROLES)}, "all_full": {"layers": all_layers, "roles": set(ROLES)}}


def role_program(model, trace, source_layer, position, contrast, specs):
    names = list(specs); current = final_adjoint(model, trace[-1]["output"], position, contrast).repeat(len(names), 1, 1, 1); maximum_error = 0.0
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]; updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current); enabled = [row for row, name in enumerate(names) if layer in specs[name]["layers"] and specs[name]["roles"]]
        if enabled:
            previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float(); mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0; inputs = updated[enabled]; pieces = attention_role_vjps(block, mixed, trace[layer - 1]["value_bus"], inputs); native = attention_vjp(block, mixed, trace[layer - 1]["value_bus"], inputs); maximum_error = max(maximum_error, float((sum(pieces.values()) - native).norm() / native.norm().clamp_min(1e-30)))
            for local, row in enumerate(enabled): updated[row] += sum((pieces[role][local] for role in specs[names[row]]["roles"]), torch.zeros_like(updated[row]))
        current = block.lambdas[0].float() * updated
    down = model.blocks[source_layer].mlp.Down.weight.float(); return {name: current[row, 0, position] @ down for row, name in enumerate(names)}, maximum_error


def summarize(records, layer, method):
    rows = [row for row in records if row["layer"] == layer]; values = [row["metrics"][method] for row in rows]; shifts = np.asarray([row["finite_shifts"][method] for row in rows]); full = np.asarray([row["finite_shifts"]["near_full"] for row in rows])
    return {"gradient_cosine_median": float(np.median([value["gradient_cosine"] for value in values])), "gradient_cosine_q10": float(np.quantile([value["gradient_cosine"] for value in values], .1)), "contribution_cosine_median": float(np.median([value["contribution_cosine"] for value in values])), "support_jaccard_median": float(np.median([value["support_jaccard"] for value in values])), "true_leverage_ratio_median": float(np.median([value["true_leverage_ratio"] for value in values])), "finite_directional_fraction": float(np.mean(shifts > 0)), "finite_mean_shift": float(shifts.mean()), "near_full_effect_ratio": float(shifts.mean() / max(full.mean(), 1e-30))}


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source_results"]).read_text()); census = json.loads(Path(config["census_config"]).read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive; contrasts = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(census["corpus"], mmap_mode="r"); start = census["start_row"] + census["discovery_rows"]; heldout = corpus[start:start + census["heldout_rows"]]; pairs = [{key: row[key] for key in ("label", "feature_index", "selection", "position", "donor_row", "recipient_row", "model_margin", "feature")} for row in source["records"] if row["layer"] == config["source_layers"][0]]
    signature = {"width": config["attention_width"], "pairs": len(pairs), "layers": config["source_layers"]}; checkpoint = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}; records = checkpoint.get("records", []) if checkpoint.get("signature") == signature else []; maximum_error = checkpoint.get("maximum_error", 0.0) if checkpoint.get("signature") == signature else 0.0; completed = len(records) // len(config["source_layers"])
    for pair_index, pair in enumerate(pairs):
        if pair_index < completed: continue
        donor_ids = torch.as_tensor(np.asarray(heldout[pair["donor_row"], :census["sequence_length"]], dtype=np.int64), device="cuda")[None]; recipient_ids = torch.as_tensor(np.asarray(heldout[pair["recipient_row"], :census["sequence_length"]], dtype=np.int64), device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"])); recipient_trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        contrast = contrasts[:, pair["feature_index"]]; exacts = product_gradients_for_layers(model, recipient_trace, pair["position"], contrast, config["source_layers"])
        for layer in config["source_layers"]:
            delta = product_at(model, donor_trace, layer, pair["position"]) - product_at(model, recipient_trace, layer, pair["position"]); predictions, error = role_program(model, recipient_trace, layer, pair["position"], contrast, specifications(layer, len(model.blocks), config["attention_width"])); maximum_error = max(maximum_error, error); exact = exacts[layer]; patches = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, (prediction * delta).topk(config["support_rank"]).indices) for name, prediction in predictions.items()}}
            with torch.no_grad(): logits = logits_for_deltas(model, layer, recipient_trace[layer], patches, pair["position"])
            base = class_score(logits["base"], contrast); finite = {name: class_score(value, contrast) - base for name, value in logits.items() if name != "base"}; records.append({**pair, "layer": layer, "metrics": {name: selector_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in predictions.items()}, "finite_shifts": finite})
        CHECKPOINT.write_text(json.dumps({"signature": signature, "maximum_error": maximum_error, "records": records}, allow_nan=False)); print(f"attention role pair {pair_index + 1}/{len(pairs)} {pair['feature']} {pair['selection']}", flush=True)
    methods = list(records[0]["metrics"]); summaries = {str(layer): {method: summarize(records, layer, method) for method in methods} for layer in config["source_layers"]}
    hypotheses = {"Q0_role_factorization_recomposes_attention": {"supported": maximum_error <= 1e-5}, "Q1_qk_routing_is_sufficient": {"supported": all(summaries[str(layer)]["near_qk"]["gradient_cosine_median"] >= .90 and summaries[str(layer)]["near_qk"]["finite_directional_fraction"] >= .95 for layer in config["source_layers"])}, "Q2_qk_retains_full_program_effect": {"supported": all(summaries[str(layer)]["near_qk"]["near_full_effect_ratio"] >= .85 for layer in config["source_layers"])}, "Q3_value_alone_is_insufficient": {"supported": all(summaries[str(layer)]["near_value"]["gradient_cosine_median"] < .85 for layer in config["source_layers"])}, "Q4_queries_and_keys_are_complementary": {"supported": all(summaries[str(layer)]["near_qk"]["gradient_cosine_median"] - max(summaries[str(layer)]["near_queries"]["gradient_cosine_median"], summaries[str(layer)]["near_keys"]["gradient_cosine_median"]) >= .05 for layer in config["source_layers"])}, "Q5_two_tensor_factors_are_complementary": {"supported": all(summaries[str(layer)]["near_qk"]["gradient_cosine_median"] - max(summaries[str(layer)]["near_pair1"]["gradient_cosine_median"], summaries[str(layer)]["near_pair2"]["gradient_cosine_median"]) >= .05 for layer in config["source_layers"])}, "Q6_qk_recovers_causal_support": {"supported": all(summaries[str(layer)]["near_qk"]["support_jaccard_median"] >= .25 for layer in config["source_layers"])}}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "maximum_local_error": maximum_error, "summaries": summaries, "records": records}, indent=2, allow_nan=False)); CHECKPOINT.write_text(json.dumps({"complete": True, "signature": signature, "maximum_error": maximum_error, "records": records}, allow_nan=False))
    lines = ["# Competent-feature attention-role selector", "", "The validated next-eight attention window is split into the two bilinear QK factors and the current-value/OV path. Exact state-conditioned MLP Jacobians remain active throughout. Each role program prospectively selects its own top-64 native product channels on the same 64 held-out causal pairs.", "", f"Maximum local five-role recomposition error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Attention roles](bilin18_competent_feature_attention_role_selector.png)", "", "| Layer | Method | Gradient cosine | Support Jaccard | Leverage | Finite direction | Mean shift | Full-window ratio |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for method in ("mlp_only", "near_queries", "near_keys", "near_pair1", "near_pair2", "near_qk", "near_value", "near_full", "all_full"):
            row = summaries[str(layer)][method]; lines.append(f"| {layer} | {method} | {row['gradient_cosine_median']:.3f} | {row['support_jaccard_median']:.3f} | {row['true_leverage_ratio_median']:.3f} | {row['finite_directional_fraction']:.3f} | {row['finite_mean_shift']:.4f} | {row['near_full_effect_ratio']:.3f} |")
    lines += ["", "## Guardrails", "", "The five roles are exact derivative terms of this tensor-attention architecture, not learned semantic labels. Sufficiency of QK routing would identify a reusable computational verb for same-position selection; it would not imply that a feature is stored in queries or keys. Cross-position edits remain governed by the separate key/value transport result."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["source_layers"])); fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for method in ("near_queries", "near_keys", "near_pair1", "near_pair2", "near_qk", "near_value", "near_full"): axes[0].plot(x, [summaries[str(layer)][method]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=method.replace("near_", ""))
    axes[0].set_xticks(x, config["source_layers"]); axes[0].set(xlabel="Source layer", ylabel="Median gradient cosine", title="Attention role grammar"); axes[0].legend(ncol=2); axes[0].grid(alpha=.2)
    for method in ("near_qk", "near_value", "near_full"): axes[1].plot(x, [summaries[str(layer)][method]["finite_directional_fraction"] for layer in config["source_layers"]], marker="o", label=method.replace("near_", ""))
    axes[1].set_xticks(x, config["source_layers"]); axes[1].set(xlabel="Source layer", ylabel="Positive top-64 finite fraction", title="Causal role sufficiency"); axes[1].legend(); axes[1].grid(alpha=.2); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "maximum_local_error": maximum_error, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
