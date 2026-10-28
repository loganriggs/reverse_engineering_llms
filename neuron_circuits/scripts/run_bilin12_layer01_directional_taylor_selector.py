#!/usr/bin/env python3
"""Compile Layer-0/1 coalitions with base-point directional Taylor contractions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import product_from_x0
from run_bilin12_layer01_dynamic_alignment_replication import (
    evaluate_deltas,
    final_product_delta,
    forward_layers_native,
    median,
    pair_cluster_interval,
    selected_delta,
)
from run_bilin12_layer01_conditional_coalition_selector import jaccard, local_scores
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_directional_taylor_selector.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_directional_taylor_selector.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_DIRECTIONAL_TAYLOR_SELECTOR.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_directional_taylor_selector.png"
METHODS = (
    "base_jvp", "quadratic", "cubic_midpoint", "cubic_integral",
    "explicit_midpoint", "endpoint_local",
)


def directional_derivatives(function, base, direction):
    def first(value):
        return torch.func.jvp(function, (value,), (direction,))[1]

    def second(value):
        return torch.func.jvp(first, (value,), (direction,))[1]

    def third(value):
        return torch.func.jvp(second, (value,), (direction,))[1]

    first_value = first(base)
    second_value = second(base)
    third_value = third(base)
    return first_value, second_value, third_value, first, second


def cosine(left, right):
    return float((left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-30))


def numerical_row(analytic, finite):
    return {
        "cosine": cosine(analytic, finite),
        "relative_error": float((analytic - finite).norm() / finite.norm().clamp_min(1e-30)),
    }


def main():
    red = json.loads(CONFIG.read_text())
    config = json.loads(Path(red["source_config"]).read_text())
    pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    layers = config["layers"]
    edit_position, position = config["edit_position"], config["prediction_position"]
    records = {str(layer): [] for layer in layers}

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(red["window_seed"] + corpus_index)
        for family, family_pairs in pairs["new_families"].items():
            for source_token, target_token in family_pairs:
                windows = select_windows(
                    rows, source_token, edit_position, position,
                    red["contexts_per_pair"], rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy()
                    changed = raw.copy(); changed[edit_position] = target_token
                    base_ids = torch.as_tensor(raw, device="cuda")[None]
                    changed_ids = torch.as_tensor(changed, device="cuda")[None]
                    with torch.no_grad():
                        base_trace = forward_layers_native(model, base_ids, max(layers))
                        endpoint_trace = forward_layers_native(model, changed_ids, max(layers))
                        dimension = model.config.n_embd
                        base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                        endpoint_x0 = F.rms_norm(model.embedding(changed_ids), (dimension,)).float()
                    direction = endpoint_x0 - base_x0
                    for layer in layers:
                        function = lambda value, layer=layer: product_from_x0(model, value, layer, position)
                        with torch.enable_grad():
                            first, second, third, first_function, second_function = directional_derivatives(
                                function, base_x0, direction,
                            )
                            explicit_midpoint = first_function(base_x0 + .5 * direction)
                            epsilon = red["finite_difference_epsilon"]
                            positive = base_x0 + epsilon * direction
                            negative = base_x0 - epsilon * direction
                            finite_first = (function(positive) - function(negative)) / (2 * epsilon)
                            finite_second = (first_function(positive) - first_function(negative)) / (2 * epsilon)
                            finite_third = (second_function(positive) - second_function(negative)) / (2 * epsilon)
                        first = first.detach().float(); second = second.detach().float(); third = third.detach().float()
                        explicit_midpoint = explicit_midpoint.detach().float()
                        vectors = {
                            "base_jvp": first,
                            "quadratic": first + .5 * second,
                            "cubic_midpoint": first + .5 * second + .125 * third,
                            "cubic_integral": first + .5 * second + third / 6,
                            "explicit_midpoint": explicit_midpoint,
                        }
                        full = final_product_delta(
                            model, layer, base_trace[layer], endpoint_trace[layer], position,
                        )
                        vectors["endpoint_local"] = full
                        decoder = model.blocks[layer].mlp.Down.weight.float()
                        endpoint_indices = local_scores(full, decoder).topk(red["rank"]).indices
                        indices = {
                            method: local_scores(vector, decoder).topk(red["rank"]).indices
                            for method, vector in vectors.items()
                        }
                        deltas = {
                            method: selected_delta(full, indices[method]) for method in METHODS
                        }
                        with torch.no_grad():
                            effects = evaluate_deltas(
                                model, layer, base_trace[layer], deltas, full, position,
                            )
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "fidelity": {method: effects[method] for method in METHODS},
                            "support_jaccard": {
                                method: jaccard(indices[method].cpu().tolist(), endpoint_indices.cpu().tolist())
                                for method in METHODS
                            },
                            "midpoint_cosine": {
                                method: cosine(vector, explicit_midpoint) for method, vector in vectors.items()
                            },
                            "numerical": {
                                "first": numerical_row(first, finite_first.detach().float()),
                                "second": numerical_row(second, finite_second.detach().float()),
                                "third": numerical_row(third, finite_third.detach().float()),
                            },
                            "derivative_norms": {
                                "first": float(first.norm()),
                                "second": float(second.norm()),
                                "third": float(third.norm()),
                                "midpoint": float(explicit_midpoint.norm()),
                            },
                        })
                print("directional Taylor", corpus, family, source_token, target_token, flush=True)

    summaries = {str(layer): {} for layer in layers}
    numerical_summary = {}
    for layer in layers:
        numerical_summary[str(layer)] = {}
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summary = {
                "fidelity": {method: median([row["fidelity"][method] for row in selected]) for method in METHODS},
                "support_jaccard": {method: median([row["support_jaccard"][method] for row in selected]) for method in METHODS},
                "midpoint_cosine": {method: median([row["midpoint_cosine"][method] for row in selected]) for method in METHODS},
                "derivative_norm_ratio": {
                    order: median([row["derivative_norms"][order] / max(row["derivative_norms"]["midpoint"], 1e-30) for row in selected])
                    for order in ("first", "second", "third")
                },
            }
            for order in ("first", "second", "third"):
                numerical_summary[str(layer)].setdefault(corpus, {})[order] = {
                    "minimum_cosine": min(row["numerical"][order]["cosine"] for row in selected),
                    "maximum_relative_error": max(row["numerical"][order]["relative_error"] for row in selected),
                    "median_relative_error": median([row["numerical"][order]["relative_error"] for row in selected]),
                }
            comparisons = (
                ("quadratic_minus_base", lambda row: row["fidelity"]["quadratic"] - row["fidelity"]["base_jvp"], 0),
                ("cubic_minus_quadratic", lambda row: row["fidelity"]["cubic_midpoint"] - row["fidelity"]["quadratic"], 10000),
                ("midpoint_minus_cubic", lambda row: row["fidelity"]["explicit_midpoint"] - row["fidelity"]["cubic_midpoint"], 20000),
                ("integral_minus_midpoint_coefficient", lambda row: row["fidelity"]["cubic_integral"] - row["fidelity"]["cubic_midpoint"], 30000),
            )
            for name, function, offset in comparisons:
                value, interval = pair_cluster_interval(
                    selected, function, red["bootstrap_samples"],
                    red["bootstrap_seed"] + offset + layer * 100 + corpus_index,
                )
                summary[f"{name}_pair_median"] = value
                summary[f"{name}_pair_ci"] = interval
            summaries[str(layer)][corpus] = summary

    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    numerical_valid = True
    thresholds = {"first": .05, "second": .10, "third": .20}
    for layer in layers:
        for corpus in arrays:
            for order, threshold in thresholds.items():
                row = numerical_summary[str(layer)][corpus][order]
                numerical_valid = numerical_valid and row["minimum_cosine"] >= .99 and row["maximum_relative_error"] <= threshold
    hypotheses = {
        "T0_nested_directional_derivatives_are_numerically_valid": {"supported": numerical_valid},
        "T1_quadratic_taylor_selection_beats_the_base_jvp": {"supported": all(
            cell(layer, corpus)["quadratic_minus_base_pair_median"] >= .10
            and cell(layer, corpus)["quadratic_minus_base_pair_ci"][0] > 0
            for layer in layers for corpus in arrays
        )},
        "T2_cubic_taylor_selection_beats_quadratic_selection": {"supported": all(
            cell(layer, corpus)["cubic_minus_quadratic_pair_median"] >= .05
            and cell(layer, corpus)["cubic_minus_quadratic_pair_ci"][0] > 0
            for layer in layers for corpus in arrays
        )},
        "T3_cubic_taylor_selection_matches_explicit_midpoint_selection": {"supported": all(
            abs(cell(layer, corpus)["fidelity"]["explicit_midpoint"] - cell(layer, corpus)["fidelity"]["cubic_midpoint"]) <= .05
            for layer in layers for corpus in arrays
        )},
        "T4_cubic_taylor_selection_is_high_fidelity": {"supported": all(
            cell(layer, corpus)["fidelity"]["cubic_midpoint"] >= .70
            for layer in layers for corpus in arrays
        )},
        "T5_cubic_taylor_recovers_native_support": {"supported": all(
            cell(layer, corpus)["support_jaccard"]["cubic_midpoint"] >= .50
            for layer in layers for corpus in arrays
        )},
        "T6_integral_coefficient_improves_endpoint_selection": {"supported": all(
            cell(layer, corpus)["integral_minus_midpoint_coefficient_pair_median"] > 0
            for layer in layers for corpus in arrays
        )},
        "T7_cubic_selector_transfers_out_of_distribution": {"supported": all(
            abs(cell(layer, "fineweb")["fidelity"]["cubic_midpoint"] - cell(layer, "tinystories")["fidelity"]["cubic_midpoint"]) <= .10
            for layer in layers
        )},
        "T8_endpoint_local_reference_is_adequate": {"supported": all(
            cell(layer, corpus)["fidelity"]["endpoint_local"] >= .70
            for layer in layers for corpus in arrays
        )},
        "T9_no_midpoint_or_endpoint_outcome_is_used_by_taylor_selectors": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({
        "config": red,
        "hypotheses": hypotheses,
        "numerical": numerical_summary,
        "summaries": summaries,
        "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Directional Taylor compilation of native coalitions", "",
        "At the original normalized embedding state, nested forward-mode contractions compute the first three directional derivatives of the exact prefix-to-MLP-product map along each declared token edit. Rank-256 selectors use `Jd`, `Jd + 1/2 H[d,d]`, and cubic variants before any midpoint or finite endpoint outcome is evaluated. Their selected coordinates are scored causally by inserting only the corresponding exact finite product deltas through the full suffix.",
        "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "![Directional Taylor selector](bilin12_layer01_directional_taylor_selector.png)", "",
        "| Layer | Corpus | Base / quadratic / cubic-mid / cubic-integral / explicit-mid / endpoint fidelity | Cubic support / midpoint cosine | Quadratic−base pair median (95% CI) | Cubic−quadratic pair median (95% CI) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for layer in layers:
        for corpus in arrays:
            row = cell(layer, corpus); f = row["fidelity"]
            qci = row["quadratic_minus_base_pair_ci"]; cci = row["cubic_minus_quadratic_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {f['base_jvp']:.3f} / {f['quadratic']:.3f} / {f['cubic_midpoint']:.3f} / {f['cubic_integral']:.3f} / {f['explicit_midpoint']:.3f} / {f['endpoint_local']:.3f} | "
                f"{row['support_jaccard']['cubic_midpoint']:.3f} / {row['midpoint_cosine']['cubic_midpoint']:.3f} | "
                f"{row['quadratic_minus_base_pair_median']:+.3f} ({qci[0]:+.3f}, {qci[1]:+.3f}) | "
                f"{row['cubic_minus_quadratic_pair_median']:+.3f} ({cci[0]:+.3f}, {cci[1]:+.3f}) |"
            )
    lines += ["", "## Numerical derivative controls", "", "| Layer | Corpus | First min cosine/max error | Second | Third |", "|---:|---|---:|---:|---:|"]
    for layer in layers:
        for corpus in arrays:
            row = numerical_summary[str(layer)][corpus]
            display = lambda order: f"{row[order]['minimum_cosine']:.5f}/{row[order]['maximum_relative_error']:.3f}"
            lines.append(f"| {layer} | {corpus} | {display('first')} | {display('second')} | {display('third')} |")
    lines += [
        "",
        "The Taylor selectors are weight-derived compilers, not semantic probes: they know the exact base state and proposed input edit. Success would establish a short local algebraic description of coalition selection. Failure can arise from a limited convergence radius even when the explicit midpoint contraction succeeds; derivative norms and finite-difference controls are retained to distinguish numerical bugs from genuine high curvature.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    labels = [f"L{layer}\n{corpus}" for layer in layers for corpus in arrays]
    x = np.arange(len(labels)); width = .14
    for index, method in enumerate(METHODS):
        offset = (index - (len(METHODS) - 1) / 2) * width
        axes[0].bar(x + offset, [cell(layer, corpus)["fidelity"][method] for layer in layers for corpus in arrays], width, label=method)
        axes[1].bar(x + offset, [cell(layer, corpus)["support_jaccard"][method] for layer in layers for corpus in arrays], width, label=method)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 1); axes[0].set_ylabel("finite causal fidelity"); axes[0].set_title("Directional-order selector")
    axes[1].set_xticks(x, labels); axes[1].set_ylim(0, 1); axes[1].set_ylabel("endpoint top-256 Jaccard"); axes[1].set_title("Native support recovery")
    for axis in axes: axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
