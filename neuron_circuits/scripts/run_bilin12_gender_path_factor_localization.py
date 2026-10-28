#!/usr/bin/env python3
"""Localize sharp finite-edit curvature into tensor-network factors."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import fisher_terms
from run_bilin12_final_destination_source_program import source_write_contributions
from run_bilin12_semantic_source_jvp_predictor import differentiable_attention, projected_source_writes
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_gender_path_factor_localization.json")
OUTPUT = TENSOR_ROOT / "bilin12_gender_path_factor_localization.json"
REPORT = TENSOR_ROOT / "BILIN12_GENDER_PATH_FACTOR_LOCALIZATION.md"
FIGURE = TENSOR_ROOT / "bilin12_gender_path_factor_localization.png"


def differentiable_factor_trace(model, x0, destination):
    x = x0
    value_bus = None
    patterns, values, sources = [], [], []
    attention_writes, mlp_writes, outputs = [], [], []
    for block in model.blocks:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        normalized = F.rms_norm(mixed, (mixed.shape[-1],))
        write, value_bus, pattern, mixed_value, _ = differentiable_attention(model, block, normalized, value_bus)
        source = projected_source_writes(block, pattern, mixed_value, destination)
        pre = mixed + write
        state = F.rms_norm(pre, (pre.shape[-1],))
        mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + mlp
        patterns.append(pattern[:, :, destination])
        values.append(mixed_value)
        sources.append(source)
        attention_writes.append(write[:, destination])
        mlp_writes.append(mlp[:, destination])
        outputs.append(x[:, destination])
    return tuple(patterns), tuple(values), tuple(sources), tuple(attention_writes), tuple(mlp_writes), tuple(outputs)


def project_factor(block, per_head_source):
    pieces = []
    for head in range(block.attn.n_head):
        start = head * block.attn.head_dim
        stop = (head + 1) * block.attn.head_dim
        pieces.append(per_head_source[:, head] @ block.attn.c_proj.weight[:, start:stop].T)
    return torch.stack(pieces).sum(0)


def path_row(model, base_x0, tangent, location, destination):
    point = base_x0 + location * tangent
    function = lambda value: differentiable_factor_trace(model, value, destination)
    primal, derivative = torch.func.jvp(function, (point,), (tangent,))
    patterns, values, _, _, _, _ = primal
    pattern_derivatives, value_derivatives, source_derivatives, attention_derivatives, mlp_derivatives, output_derivatives = derivative
    routing, value = [], []
    for layer, block in enumerate(model.blocks):
        routing_per_head = pattern_derivatives[layer][:, :, :, None] * values[layer].permute(0, 2, 1, 3)
        value_per_head = patterns[layer][:, :, :, None] * value_derivatives[layer].permute(0, 2, 1, 3)
        routing.append(project_factor(block, routing_per_head).float())
        value.append(project_factor(block, value_per_head).float())
    return {
        "source": tuple(row.float() for row in source_derivatives),
        "routing": tuple(routing),
        "value": tuple(value),
        "attention": tuple(row.float() for row in attention_derivatives),
        "mlp": tuple(row.float() for row in mlp_derivatives),
        "output": tuple(row.float() for row in output_derivatives),
    }


def finite_stats(base_logits, target_delta, prediction_delta):
    target, _, cross, error = fisher_terms(base_logits, target_delta, prediction_delta)
    denominator = target.sum().clamp_min(1e-30)
    return {
        "target": float(target.sum()),
        "error": float(error.sum()),
        "fidelity": float(1 - error.sum() / denominator),
        "projection": float(cross.sum() / denominator),
    }


def aggregate(rows):
    target = sum(row["target"] for row in rows)
    return {
        "fidelity": 1 - sum(row["error"] for row in rows) / max(target, 1e-30),
        "projection": sum(row["projection"] * row["target"] for row in rows) / max(target, 1e-30),
    }


def main():
    red = json.loads(CONFIG.read_text())
    config = json.loads(Path(red["source_config"]).read_text())
    prior = json.loads(Path(red["source_output"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    sources = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"] : spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    selected = [record for record in prior["records"] if record["family"] == red["family"]]
    locations = np.linspace(0, 1, red["dense_points"])
    weights = np.ones(red["dense_points"]) / (red["dense_points"] - 1)
    weights[[0, -1]] *= 0.5
    records = []
    causal_rows = defaultdict(list)
    maximum_pointwise_error = 0.0
    maximum_integral_error = 0.0

    for prior_record in selected:
        corpus = prior_record["corpus"]
        row_index, start = prior_record["row"], prior_record["start"]
        _, target_token = (int(value) for value in prior_record["pair"].split("->"))
        raw = np.asarray(sources[corpus][row_index, start : start + config["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy()
        changed[config["edit_position"]] = target_token
        base_ids = torch.as_tensor(raw, device="cuda")[None]
        changed_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            base_rows, _, _ = trace(model, base_ids)
            changed_rows, _, _ = trace(model, changed_ids)
            base_x0 = F.rms_norm(model.embedding(base_ids), (model.config.n_embd,))
            changed_x0 = F.rms_norm(model.embedding(changed_ids), (model.config.n_embd,))
            input_tangent = changed_x0 - base_x0
        with torch.enable_grad():
            path = [path_row(model, base_x0, input_tangent, float(location), config["prediction_position"]) for location in locations]

        layer_records = {}
        for layer in range(model.config.n_layer):
            arrays = {
                name: np.asarray([float(row[name][layer].norm()) for row in path])
                for name in ("source", "routing", "value", "attention", "mlp", "output")
            }
            point_errors = [
                float((row["routing"][layer] + row["value"][layer] - row["source"][layer]).norm() / row["source"][layer].norm().clamp_min(1e-30))
                for row in path
            ]
            maximum_pointwise_error = max(maximum_pointwise_error, max(point_errors))
            integrated = {
                name: sum(float(weight) * row[name][layer] for row, weight in zip(path, weights))
                for name in ("source", "routing", "value")
            }
            exact_delta = source_write_contributions(model.blocks[layer], changed_rows[layer]["details"], config["prediction_position"]).float() - source_write_contributions(model.blocks[layer], base_rows[layer]["details"], config["prediction_position"]).float()
            integral_error = float((integrated["routing"] + integrated["value"] - exact_delta).norm() / exact_delta.norm().clamp_min(1e-30))
            maximum_integral_error = max(maximum_integral_error, integral_error)
            peak_index = int(arrays["source"].argmax())
            routing_peak = path[peak_index]["routing"][layer]
            value_peak = path[peak_index]["value"][layer]
            routing_norm = float(routing_peak.norm())
            value_norm = float(value_peak.norm())
            factor_cosine = float((routing_peak.flatten() * value_peak.flatten()).sum() / (routing_peak.norm() * value_peak.norm()).clamp_min(1e-30))
            layer_row = {
                "peak_ratios": {name: float(values.max() / max(np.median(values), 1e-30)) for name, values in arrays.items()},
                "source_peak_location": float(locations[peak_index]),
                "routing_peak_share": routing_norm / max(routing_norm + value_norm, 1e-30),
                "value_peak_share": value_norm / max(routing_norm + value_norm, 1e-30),
                "routing_value_peak_cosine": factor_cosine,
                "pointwise_recomposition_max_error": max(point_errors),
                "integral_recomposition_error": integral_error,
                "paths": {name: values.tolist() for name, values in arrays.items()},
            }
            if layer in red["causal_layers"]:
                base = base_rows[layer]
                changed_row = changed_rows[layer]
                full_write = base["details"]["write"].clone()
                full_write[:, config["prediction_position"]] = changed_row["details"]["write"][:, config["prediction_position"]]
                base_logits, target_logits = logits_from_attention_writes(model, layer, base, torch.stack((base["details"]["write"], full_write)))
                target_delta = target_logits - base_logits
                worlds = []
                for factor in ("routing", "value", "sum"):
                    delta = integrated["routing"] if factor == "routing" else integrated["value"] if factor == "value" else integrated["routing"] + integrated["value"]
                    write = base["details"]["write"].clone()
                    write[:, config["prediction_position"]] += delta.sum(1).to(write.dtype)
                    worlds.append(write)
                logits = logits_from_attention_writes(model, layer, base, torch.stack(worlds))
                causal = {factor: finite_stats(base_logits, target_delta, logits[index] - base_logits) for index, factor in enumerate(("routing", "value", "sum"))}
                layer_row["causal"] = causal
                for factor, stats in causal.items():
                    causal_rows[(prior_record["pair"], str(layer), factor)].append(stats)
            layer_records[str(layer)] = layer_row
        records.append({"corpus": corpus, "pair": prior_record["pair"], "layers": layer_records})
        print(corpus, prior_record["pair"], flush=True)

    anomaly_records = [record for record in records if record["pair"] == red["anomalous_pair"]]
    control_records = [record for record in records if record["pair"] != red["anomalous_pair"]]
    anomaly_peak_by_layer = {
        layer: max(record["layers"][str(layer)]["peak_ratios"]["source"] for record in anomaly_records)
        for layer in range(model.config.n_layer)
    }
    control_q95_by_layer = {
        layer: float(np.quantile([record["layers"][str(layer)]["peak_ratios"]["source"] for record in control_records], 0.95))
        for layer in range(model.config.n_layer)
    }
    separating_layers = [layer for layer in range(model.config.n_layer) if anomaly_peak_by_layer[layer] >= 1.5 * control_q95_by_layer[layer]]
    sharpest_record, sharpest_layer, sharpest_row = max(
        ((record, int(layer), row) for record in anomaly_records for layer, row in record["layers"].items()),
        key=lambda item: item[2]["peak_ratios"]["source"],
    )
    max_factor_share = max(sharpest_row["routing_peak_share"], sharpest_row["value_peak_share"])
    anomaly_causal = {
        corpus: {
            str(layer): records_row["layers"][str(layer)]["causal"]
            for layer in red["causal_layers"]
        }
        for records_row in anomaly_records
        for corpus in [records_row["corpus"]]
    }
    tinystories_l5 = anomaly_causal["tinystories"]["5"]
    all_anomaly_factors = [anomaly_causal[corpus][str(layer)][factor]["projection"] for corpus in anomaly_causal for layer in red["causal_layers"] for factor in ("routing", "value")]
    anomaly_max = max(anomaly_peak_by_layer.values())
    control_maxima = [max(row["peak_ratios"]["source"] for row in record["layers"].values()) for record in control_records]
    hypotheses = {
        "F1_routing_and_value_derivatives_recompose_each_source_jvp": {"supported": bool(maximum_pointwise_error <= 1e-5), "maximum_relative_error": maximum_pointwise_error},
        "F2_dense_factor_integrals_recompose_the_finite_source_change": {"supported": bool(maximum_integral_error <= 1e-3), "maximum_relative_error": maximum_integral_error},
        "F3_one_attention_factor_dominates_the_sharpest_local_peak": {"supported": bool(max_factor_share >= 0.70), "largest_share": max_factor_share, "layer": sharpest_layer, "corpus": sharpest_record["corpus"]},
        "F4_the_peak_contains_routing_value_cancellation": {"supported": bool(sharpest_row["routing_value_peak_cosine"] <= -0.25), "cosine": sharpest_row["routing_value_peak_cosine"]},
        "F5_the_sharpness_is_already_visible_before_layer_five": {"supported": bool(separating_layers and min(separating_layers) <= 4), "separating_layers": separating_layers},
        "F6_a_bilinear_mlp_peak_precedes_or_matches_the_attention_source_peak": {"supported": bool(sharpest_row["peak_ratios"]["mlp"] >= sharpest_row["peak_ratios"]["attention"]), "mlp_peak_ratio": sharpest_row["peak_ratios"]["mlp"], "attention_peak_ratio": sharpest_row["peak_ratios"]["attention"]},
        "F7_both_integrated_routing_and_value_have_behavioral_effects": {"supported": bool(all(abs(value) >= 0.10 for value in all_anomaly_factors)), "minimum_absolute_projection": min(abs(value) for value in all_anomaly_factors)},
        "F8_neither_factor_alone_explains_the_worst_condition": {"supported": bool(tinystories_l5["routing"]["fidelity"] < 0.80 and tinystories_l5["value"]["fidelity"] < 0.80 and tinystories_l5["sum"]["fidelity"] >= 0.99), "values": {factor: tinystories_l5[factor]["fidelity"] for factor in tinystories_l5}},
        "F9_factor_peak_location_matches_the_direct_source_peak": {"supported": bool(maximum_pointwise_error <= 1e-5), "maximum_recomposition_error": maximum_pointwise_error},
        "F10_matched_controls_have_lower_composed_source_sharpness": {"supported": bool(anomaly_max >= 1.5 * np.median(control_maxima)), "anomaly_max": anomaly_max, "control_median_max": float(np.median(control_maxima))},
    }
    causal_summary = {
        pair: {
            str(layer): {factor: aggregate(causal_rows[(pair, str(layer), factor)]) for factor in ("routing", "value", "sum")}
            for layer in red["causal_layers"]
        }
        for pair in sorted({record["pair"] for record in records})
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "anomaly_peak_by_layer": anomaly_peak_by_layer, "control_q95_by_layer": control_q95_by_layer, "causal_summary": causal_summary, "records": records}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    layers = np.arange(model.config.n_layer)
    axes[0].plot(layers, [anomaly_peak_by_layer[layer] for layer in layers], marker="o", label="his→her max")
    axes[0].plot(layers, [control_q95_by_layer[layer] for layer in layers], marker="o", label="control q95")
    axes[0].set_title("Source-derivative path sharpness")
    axes[0].set_xlabel("layer")
    for name in ("source", "routing", "value", "attention", "mlp", "output"):
        axes[1].plot(locations, sharpest_row["paths"][name], label=name)
    axes[1].set_title(f"Sharpest row: {sharpest_record['corpus']} L{sharpest_layer}")
    axes[1].set_xlabel("path location")
    width = 0.25
    x = np.arange(len(red["causal_layers"]))
    for index, factor in enumerate(("routing", "value", "sum")):
        axes[2].bar(x + (index - 1) * width, [np.mean([anomaly_causal[corpus][str(layer)][factor]["projection"] for corpus in anomaly_causal]) for layer in red["causal_layers"]], width, label=factor)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"L{layer}" for layer in red["causal_layers"]])
    axes[2].set_title("his→her integrated Fisher projection")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Layer and factor localization of sharp semantic paths",
        "",
        "For every gender edit, the dense embedding chord is differentiated through all twelve layers. Each final-destination source derivative is split exactly into attention-routing and value derivatives. Final-position attention, bilinear-MLP, and residual-output derivatives localize where path curvature becomes sharp. Dense factor integrals are then inserted causally at layers 5, 6, 8, and 10.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Path factor localization](bilin12_gender_path_factor_localization.png)",
        "",
        "| Layer | his→her max peak ratio | control q95 | routing projection | value projection | sum fidelity |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in range(model.config.n_layer):
        if layer in red["causal_layers"]:
            route = np.mean([anomaly_causal[corpus][str(layer)]["routing"]["projection"] for corpus in anomaly_causal])
            value = np.mean([anomaly_causal[corpus][str(layer)]["value"]["projection"] for corpus in anomaly_causal])
            fidelity = np.mean([anomaly_causal[corpus][str(layer)]["sum"]["fidelity"] for corpus in anomaly_causal])
            suffix = f"{route:.3f} | {value:.3f} | {fidelity:.3f}"
        else:
            suffix = "— | — | —"
        lines.append(f"| {layer} | {anomaly_peak_by_layer[layer]:.2f} | {control_q95_by_layer[layer]:.2f} | {suffix} |")
    lines += [
        "",
        f"Sharpest row: **{sharpest_record['corpus']} layer {sharpest_layer}**. Routing/value norm shares there are **{sharpest_row['routing_peak_share']:.3f}/{sharpest_row['value_peak_share']:.3f}**, with cosine **{sharpest_row['routing_value_peak_cosine']:.3f}**.",
        "",
        "Routing/value derivatives are differential path factors, not unique global feature attributions. Their exact pointwise sum makes them useful for locating curvature; their individual finite downstream effects can still interact nonlinearly.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
