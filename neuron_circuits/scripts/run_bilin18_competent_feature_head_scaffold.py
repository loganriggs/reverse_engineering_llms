#!/usr/bin/env python3
"""Discovery-frozen layer-head scaffold for the near-eight causal selector."""

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
from run_bilin18_layer1_cross_position_head_sparsity import attention_head_vjps
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_competent_feature_head_scaffold.json")
OUTPUT = TENSOR_ROOT / "bilin18_competent_feature_head_scaffold.json"
REPORT = TENSOR_ROOT / "BILIN18_COMPETENT_FEATURE_HEAD_SCAFFOLD.md"
FIGURE = TENSOR_ROOT / "bilin18_competent_feature_head_scaffold.png"
CHECKPOINT = TENSOR_ROOT / "bilin18_competent_feature_head_scaffold.partial.json"


def head_program(model, trace, source_layer, position, contrast, programs):
    names = list(programs); current = final_adjoint(model, trace[-1]["output"], position, contrast).repeat(len(names), 1, 1, 1); maximum_error = 0.0
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]; updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current); enabled = [row for row, name in enumerate(names) if any(edge[0] == layer for edge in programs[name])]
        if enabled:
            previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float(); mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0; inputs = updated[enabled]; pieces = attention_head_vjps(block, mixed, trace[layer - 1]["value_bus"], inputs); native = attention_vjp(block, mixed, trace[layer - 1]["value_bus"], inputs); maximum_error = max(maximum_error, float((pieces.sum(0) - native).norm() / native.norm().clamp_min(1e-30)))
            for local, row in enumerate(enabled):
                heads = [head for edge_layer, head in programs[names[row]] if edge_layer == layer]; updated[row] += pieces[heads, local].sum(0)
        current = block.lambdas[0].float() * updated
    down = model.blocks[source_layer].mlp.Down.weight.float(); return {name: current[row, 0, position] @ down for row, name in enumerate(names)}, maximum_error


def pearson(first, second):
    return float(np.corrcoef(first, second)[0, 1])


def jaccard(first, second):
    return len(set(first) & set(second)) / len(set(first) | set(second))


def summarize(rows, method):
    values = [row["metrics"][method] for row in rows]; shifts = np.asarray([row["finite_shifts"][method] for row in rows]); full = np.asarray([row["finite_shifts"]["top_72"] for row in rows])
    return {"gradient_cosine_median": float(np.median([value["gradient_cosine"] for value in values])), "gradient_cosine_q10": float(np.quantile([value["gradient_cosine"] for value in values], .1)), "support_jaccard_median": float(np.median([value["support_jaccard"] for value in values])), "true_leverage_ratio_median": float(np.median([value["true_leverage_ratio"] for value in values])), "finite_directional_fraction": float(np.mean(shifts > 0)), "finite_mean_shift": float(shifts.mean()), "full_effect_ratio": float(shifts.mean() / max(full.mean(), 1e-30))}


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source_results"]).read_text()); census = json.loads(Path(config["census_config"]).read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive; contrasts = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(census["corpus"], mmap_mode="r"); start = census["start_row"] + census["discovery_rows"]; heldout = corpus[start:start + census["heldout_rows"]]; all_pairs = [{key: row[key] for key in ("label", "feature_index", "selection", "position", "donor_row", "recipient_row", "model_margin", "feature")} for row in source["records"] if row["layer"] == config["source_layers"][0]]
    panels = {name: [next(pair for pair in all_pairs if pair["feature"] == feature and pair["selection"] == "extreme") for feature in features] for name, features in config["discovery_panels"].items()}; evaluation = [pair for pair in all_pairs if pair["selection"] == "uniform"]
    signature = {"width": config["attention_width"], "layers": config["source_layers"]}; checkpoint = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}; previous_signature = checkpoint.get("signature", {}); compatible = previous_signature.get("width") == signature["width"] and previous_signature.get("layers") == signature["layers"]; discovery_scores = checkpoint.get("discovery_scores", {}) if compatible else {}; maximum_error = checkpoint.get("maximum_error", 0.0) if compatible else 0.0
    for source_layer in config["source_layers"]:
        edges = [(layer, head) for layer in range(source_layer + 1, source_layer + 1 + config["attention_width"]) for head in range(model.config.n_head)]; full = set(edges)
        for panel_name, pairs in panels.items():
            key = f"{source_layer}_{panel_name}"
            if key in discovery_scores: continue
            edge_values = {f"{layer}_{head}": [] for layer, head in edges}
            for pair_index, pair in enumerate(pairs):
                recipient_ids = torch.as_tensor(np.asarray(heldout[pair["recipient_row"], :census["sequence_length"]], dtype=np.int64), device="cuda")[None]
                with torch.no_grad(): trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
                contrast = contrasts[:, pair["feature_index"]]; exact = product_gradients_for_layers(model, trace, pair["position"], contrast, [source_layer])[source_layer]; programs = {"full": full, **{f"leave_{layer}_{head}": full - {(layer, head)} for layer, head in edges}}; predictions, error = head_program(model, trace, source_layer, pair["position"], contrast, programs); maximum_error = max(maximum_error, error)
                for layer, head in edges:
                    prediction = predictions[f"leave_{layer}_{head}"]; cosine = float((prediction @ exact) / (prediction.norm() * exact.norm()).clamp_min(1e-30)); edge_values[f"{layer}_{head}"].append(1 - cosine)
                print(f"head discovery L{source_layer} {panel_name} {pair_index + 1}/{len(pairs)} {pair['feature']}", flush=True)
            discovery_scores[key] = {edge: float(np.mean(values)) for edge, values in edge_values.items()}; CHECKPOINT.write_text(json.dumps({"signature": signature, "maximum_error": maximum_error, "discovery_scores": discovery_scores}, allow_nan=False))
    rankings = {}; stability = {}; posthoc_panel_stability = {}
    for source_layer in config["source_layers"]:
        edges = [(layer, head) for layer in range(source_layer + 1, source_layer + 1 + config["attention_width"]) for head in range(model.config.n_head)]; labels = [f"{layer}_{head}" for layer, head in edges]; panel_names = list(config["discovery_panels"]); panel_values = {name: np.asarray([discovery_scores[f"{source_layer}_{name}"][label] for label in labels]) for name in panel_names}; first = panel_values["a"]; second = panel_values["b"]; order_a = np.argsort(first)[::-1]; order_b = np.argsort(second)[::-1]; averaged = np.stack(list(panel_values.values())).mean(0); ordering = np.argsort(averaged)[::-1]; rankings[str(source_layer)] = [edges[index] for index in ordering]; stability[str(source_layer)] = {"score_correlation": pearson(first, second), "top32_jaccard": jaccard(order_a[:32], order_b[:32])}
        correlations = []; overlaps = []
        for first_index in range(len(panel_names)):
            for second_index in range(first_index + 1, len(panel_names)):
                left = panel_values[panel_names[first_index]]; right = panel_values[panel_names[second_index]]; correlations.append(pearson(left, right)); overlaps.append(jaccard(np.argsort(left)[::-1][:32], np.argsort(right)[::-1][:32]))
        posthoc_panel_stability[str(source_layer)] = {"pairwise_correlation_median": float(np.median(correlations)), "pairwise_correlation_minimum": float(np.min(correlations)), "top32_jaccard_median": float(np.median(overlaps)), "top32_jaccard_minimum": float(np.min(overlaps))}
    rng = np.random.default_rng(config["seed"]); records = []
    for pair_index, pair in enumerate(evaluation):
        donor_ids = torch.as_tensor(np.asarray(heldout[pair["donor_row"], :census["sequence_length"]], dtype=np.int64), device="cuda")[None]; recipient_ids = torch.as_tensor(np.asarray(heldout[pair["recipient_row"], :census["sequence_length"]], dtype=np.int64), device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"])); recipient_trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        contrast = contrasts[:, pair["feature_index"]]; exacts = product_gradients_for_layers(model, recipient_trace, pair["position"], contrast, config["source_layers"])
        for source_layer in config["source_layers"]:
            ordering = rankings[str(source_layer)]; programs = {f"top_{rank}": set(ordering[:rank]) for rank in config["edge_ranks"]}; edges = ordering[:]
            for rank in config["random_ranks"]:
                for replicate in range(config["random_replicates"]): programs[f"random_{rank}_{replicate}"] = {edges[index] for index in rng.choice(len(edges), rank, replace=False)}
            predictions, error = head_program(model, recipient_trace, source_layer, pair["position"], contrast, programs); maximum_error = max(maximum_error, error); delta = product_at(model, donor_trace, source_layer, pair["position"]) - product_at(model, recipient_trace, source_layer, pair["position"]); exact = exacts[source_layer]; patches = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, (prediction * delta).topk(config["support_rank"]).indices) for name, prediction in predictions.items()}}
            with torch.no_grad(): logits = logits_for_deltas(model, source_layer, recipient_trace[source_layer], patches, pair["position"])
            base = class_score(logits["base"], contrast); finite = {name: class_score(value, contrast) - base for name, value in logits.items() if name != "base"}; records.append({**pair, "layer": source_layer, "metrics": {name: selector_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in predictions.items()}, "finite_shifts": finite})
        print(f"head evaluation {pair_index + 1}/{len(evaluation)} {pair['feature']}", flush=True)
    summaries = {}
    for layer in config["source_layers"]:
        rows = [row for row in records if row["layer"] == layer]; summaries[str(layer)] = {method: summarize(rows, method) for method in rows[0]["metrics"]}
        for rank in config["random_ranks"]:
            medians = [summaries[str(layer)][f"random_{rank}_{replicate}"]["gradient_cosine_median"] for replicate in range(config["random_replicates"])]; summaries[str(layer)][f"random_{rank}_aggregate"] = {"median": float(np.median(medians)), "p95": float(np.quantile(medians, .95)), "maximum": float(np.max(medians))}
    hypotheses = {"H0_head_decomposition_is_exact": {"supported": maximum_error <= 1e-5}, "H1_discovery_rankings_are_stable": {"supported": all(stability[str(layer)]["score_correlation"] >= .5 and stability[str(layer)]["top32_jaccard"] >= .4 for layer in config["source_layers"])}, "H2_top32_is_high_fidelity": {"supported": all(summaries[str(layer)]["top_32"]["gradient_cosine_median"] >= .90 for layer in config["source_layers"])}, "H3_top32_is_causally_directional": {"supported": all(summaries[str(layer)]["top_32"]["finite_directional_fraction"] >= .95 for layer in config["source_layers"])}, "H4_top32_retains_near8_effect": {"supported": all(summaries[str(layer)]["top_32"]["full_effect_ratio"] >= .85 for layer in config["source_layers"])}, "H5_top16_is_not_universally_sufficient": {"supported": any(summaries[str(layer)]["top_16"]["gradient_cosine_median"] < .90 or summaries[str(layer)]["top_16"]["finite_directional_fraction"] < .90 for layer in config["source_layers"])}, "H6_top32_beats_random": {"supported": all(summaries[str(layer)]["top_32"]["gradient_cosine_median"] - summaries[str(layer)]["random_32_aggregate"]["p95"] >= .05 for layer in config["source_layers"])}}
    ranked = {str(layer): [{"rank": index + 1, "block": edge[0], "offset": edge[0] - layer, "head": edge[1], "mean_score": float(np.mean([discovery_scores[f"{layer}_{panel}"][f"{edge[0]}_{edge[1]}"] for panel in config["discovery_panels"]]))} for index, edge in enumerate(rankings[str(layer)])] for layer in config["source_layers"]}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "maximum_error": maximum_error, "stability": stability, "posthoc_panel_stability": posthoc_panel_stability, "ranked_edges": ranked, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Competent-feature layer-head scaffold", "", "Four disjoint eight-feature discovery panels rank the 72 layer-head edges in each source layer's successful next-eight attention window. Their averaged frozen ordering is evaluated on one disjoint uniform pair for every one of the 32 competent features, with matched random edge programs. The locked stability hypothesis still refers to panels A and B; the four-panel aggregation is a post-hoc sampling-noise red team.", "", f"Maximum local nine-head recomposition error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Head scaffold](bilin18_competent_feature_head_scaffold.png)", "", "| Source | Discovery correlation | Top32 Jaccard | Top16 cosine/direction | Top32 cosine/direction/effect | Random32 p95 |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        s = stability[str(layer)]; a = summaries[str(layer)]["top_16"]; b = summaries[str(layer)]["top_32"]; random = summaries[str(layer)]["random_32_aggregate"]; lines.append(f"| {layer} | {s['score_correlation']:.3f} | {s['top32_jaccard']:.3f} | {a['gradient_cosine_median']:.3f}/{a['finite_directional_fraction']:.3f} | {b['gradient_cosine_median']:.3f}/{b['finite_directional_fraction']:.3f}/{b['full_effect_ratio']:.3f} | {random['p95']:.3f} |")
    lines += ["", "## Post-hoc four-panel stability", "", "| Source | Pairwise correlation median/min | Top32 Jaccard median/min |", "|---:|---:|---:|"]
    for layer in config["source_layers"]:
        row = posthoc_panel_stability[str(layer)]; lines.append(f"| {layer} | {row['pairwise_correlation_median']:.3f}/{row['pairwise_correlation_minimum']:.3f} | {row['top32_jaccard_median']:.3f}/{row['top32_jaccard_minimum']:.3f} |")
    lines += ["", "## Top eight frozen edges by source", ""]
    for layer in config["source_layers"]: lines.append(f"- Source {layer}: " + ", ".join(f"B{row['block']}/H{row['head']}" for row in ranked[str(layer)][:8]))
    lines += ["", "## Guardrails", "", "The scaffold is discovery-frozen but source-layer specific. Head VJPs remain state-conditioned, and edge sparsity does not imply a head has one semantic meaning. Leave-one-out scores can underestimate redundant or synergistic heads; the independent-panel stability and random-program controls test, but do not eliminate, that ambiguity."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.asarray(config["edge_ranks"]); fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for layer in config["source_layers"]: axes[0].plot(x, [summaries[str(layer)][f"top_{rank}"]["gradient_cosine_median"] for rank in x], marker="o", label=f"source {layer}")
    axes[0].set(xlabel="Frozen layer-head edges", ylabel="Median exact-gradient cosine", title="Head scaffold frontier"); axes[0].legend(); axes[0].grid(alpha=.2)
    for layer in config["source_layers"]: axes[1].plot(x, [summaries[str(layer)][f"top_{rank}"]["finite_directional_fraction"] for rank in x], marker="o", label=f"source {layer}")
    axes[1].set(xlabel="Frozen layer-head edges", ylabel="Positive top-64 finite fraction", title="Causal scaffold frontier"); axes[1].legend(); axes[1].grid(alpha=.2); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "maximum_error": maximum_error, "stability": stability, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
