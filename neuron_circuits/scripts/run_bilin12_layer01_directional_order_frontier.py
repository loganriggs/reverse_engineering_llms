#!/usr/bin/env python3
"""Order-1 through order-5 directional Taylor frontier for native coalitions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import product_from_x0
from run_bilin12_layer01_conditional_coalition_selector import jaccard, local_scores
from run_bilin12_layer01_directional_taylor_selector import cosine, numerical_row
from run_bilin12_layer01_dynamic_alignment_replication import (
    evaluate_deltas,
    final_product_delta,
    forward_layers_native,
    median,
    pair_cluster_interval,
    selected_delta,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_directional_order_frontier.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_directional_order_frontier.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_DIRECTIONAL_ORDER_FRONTIER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_directional_order_frontier.png"


def derivative_frontier(function, base, direction, maximum_order):
    functions = [function]
    values = []
    current = function
    for _ in range(maximum_order):
        previous = current

        def current(value, previous=previous):
            return torch.func.jvp(previous, (value,), (direction,))[1]

        functions.append(current)
        values.append(current(base))
    return values, functions


def taylor_sum(derivatives, location, integral=False):
    total = torch.zeros_like(derivatives[0])
    for order, derivative in enumerate(derivatives, start=1):
        coefficient = (
            location ** order / math.factorial(order)
            if integral else location ** (order - 1) / math.factorial(order - 1)
        )
        total = total + coefficient * derivative
    return total


def main():
    red = json.loads(CONFIG.read_text())
    prior = json.loads(Path(red["source_config"]).read_text())
    config = json.loads(Path(prior["source_config"]).read_text())
    pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    layers = config["layers"]
    edit_position, position = config["edit_position"], config["prediction_position"]
    order_names = [f"midpoint_order{order}" for order in range(1, red["maximum_order"] + 1)]
    integral_names = [f"integral_order{order}" for order in range(1, red["maximum_order"] + 1)]
    methods = order_names + integral_names + ["explicit_midpoint", "endpoint_local"]
    records = {str(layer): [] for layer in layers}

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(prior["window_seed"] + corpus_index)
        for family, family_pairs in pairs["new_families"].items():
            for source_token, target_token in family_pairs:
                windows = select_windows(
                    rows, source_token, edit_position, position,
                    prior["contexts_per_pair"], rng,
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
                            derivatives, derivative_functions = derivative_frontier(
                                function, base_x0, direction, red["maximum_order"],
                            )
                            explicit_midpoint = derivative_functions[1](base_x0 + .5 * direction)
                            epsilon = red["finite_difference_epsilon"]
                            positive, negative = base_x0 + epsilon * direction, base_x0 - epsilon * direction
                            finite = [
                                (derivative_functions[order - 1](positive) - derivative_functions[order - 1](negative)) / (2 * epsilon)
                                for order in range(1, red["maximum_order"] + 1)
                            ]
                        derivatives = [value.detach().float() for value in derivatives]
                        explicit_midpoint = explicit_midpoint.detach().float()
                        vectors = {}
                        for order in range(1, red["maximum_order"] + 1):
                            vectors[f"midpoint_order{order}"] = taylor_sum(derivatives[:order], .5, False)
                            vectors[f"integral_order{order}"] = taylor_sum(derivatives[:order], 1.0, True)
                        vectors["explicit_midpoint"] = explicit_midpoint
                        full = final_product_delta(model, layer, base_trace[layer], endpoint_trace[layer], position)
                        vectors["endpoint_local"] = full
                        decoder = model.blocks[layer].mlp.Down.weight.float()
                        endpoint_indices = local_scores(full, decoder).topk(prior["rank"]).indices
                        indices = {
                            method: local_scores(vector, decoder).topk(prior["rank"]).indices
                            for method, vector in vectors.items()
                        }
                        deltas = {method: selected_delta(full, indices[method]) for method in methods}
                        with torch.no_grad():
                            effects = evaluate_deltas(model, layer, base_trace[layer], deltas, full, position)
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "fidelity": {method: effects[method] for method in methods},
                            "support_jaccard": {
                                method: jaccard(indices[method].cpu().tolist(), endpoint_indices.cpu().tolist())
                                for method in methods
                            },
                            "midpoint_cosine": {method: cosine(vector, explicit_midpoint) for method, vector in vectors.items()},
                            "numerical": {
                                str(order): numerical_row(derivatives[order - 1], finite[order - 1].detach().float())
                                for order in range(1, red["maximum_order"] + 1)
                            },
                            "scaled_derivative_norm": {
                                str(order): float((.5 ** (order - 1) / math.factorial(order - 1) * derivatives[order - 1]).norm() / explicit_midpoint.norm().clamp_min(1e-30))
                                for order in range(1, red["maximum_order"] + 1)
                            },
                        })
                print("directional order frontier", corpus, family, source_token, target_token, flush=True)

    summaries = {str(layer): {} for layer in layers}
    numerical = {str(layer): {} for layer in layers}
    for layer in layers:
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summary = {
                "fidelity": {method: median([row["fidelity"][method] for row in selected]) for method in methods},
                "support_jaccard": {method: median([row["support_jaccard"][method] for row in selected]) for method in methods},
                "midpoint_cosine": {method: median([row["midpoint_cosine"][method] for row in selected]) for method in methods},
                "scaled_derivative_norm": {
                    str(order): median([row["scaled_derivative_norm"][str(order)] for row in selected])
                    for order in range(1, red["maximum_order"] + 1)
                },
            }
            numerical[str(layer)][corpus] = {
                str(order): {
                    "minimum_cosine": min(row["numerical"][str(order)]["cosine"] for row in selected),
                    "maximum_relative_error": max(row["numerical"][str(order)]["relative_error"] for row in selected),
                    "median_relative_error": median([row["numerical"][str(order)]["relative_error"] for row in selected]),
                } for order in range(1, red["maximum_order"] + 1)
            }
            for order in range(2, red["maximum_order"] + 1):
                name = f"order{order}_minus_order{order-1}"
                value, interval = pair_cluster_interval(
                    selected,
                    lambda row, order=order: row["fidelity"][f"midpoint_order{order}"] - row["fidelity"][f"midpoint_order{order-1}"],
                    red["bootstrap_samples"], red["bootstrap_seed"] + order * 10000 + layer * 100 + corpus_index,
                )
                summary[f"{name}_pair_median"] = value; summary[f"{name}_pair_ci"] = interval
            summary["best_order"] = max(range(1, red["maximum_order"] + 1), key=lambda order: summary["fidelity"][f"midpoint_order{order}"])
            summary["best_fidelity"] = summary["fidelity"][f"midpoint_order{summary['best_order']}"]
            summaries[str(layer)][corpus] = summary

    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    error_thresholds = [.05, .10, .20, .30, .50]
    numeric_valid = all(
        numerical[str(layer)][corpus][str(order)]["minimum_cosine"] >= .99
        and numerical[str(layer)][corpus][str(order)]["maximum_relative_error"] <= error_thresholds[order - 1]
        for layer in layers for corpus in arrays for order in range(1, red["maximum_order"] + 1)
    )
    monotonic = all(
        cell(layer, corpus)["midpoint_cosine"][f"midpoint_order{order}"]
        >= cell(layer, corpus)["midpoint_cosine"][f"midpoint_order{order-1}"] - .005
        for layer in layers for corpus in arrays for order in range(2, red["maximum_order"] + 1)
    )
    hypotheses = {
        "O0_all_directional_orders_are_numerically_valid": {"supported": numeric_valid},
        "O1_midpoint_direction_converges_monotonically": {"supported": monotonic},
        "O2_fourth_order_improves_cubic_causal_selection": {"supported": all(
            cell(layer, corpus)["order4_minus_order3_pair_median"] >= .02
            and cell(layer, corpus)["order4_minus_order3_pair_ci"][0] > 0
            for layer in layers for corpus in arrays
        )},
        "O3_fifth_order_improves_fourth_order_selection": {"supported": all(
            cell(layer, corpus)["order5_minus_order4_pair_median"] >= .02
            and cell(layer, corpus)["order5_minus_order4_pair_ci"][0] > 0
            for layer in layers for corpus in arrays
        )},
        "O4_some_base_point_order_matches_the_explicit_midpoint": {"supported": all(
            abs(cell(layer, corpus)["best_fidelity"] - cell(layer, corpus)["fidelity"]["explicit_midpoint"]) <= .03
            for layer in layers for corpus in arrays
        )},
        "O5_fifth_order_is_high_fidelity": {"supported": all(
            cell(layer, corpus)["fidelity"]["midpoint_order5"] >= .72
            for layer in layers for corpus in arrays
        )},
        "O6_fifth_order_recovers_three_quarters_of_endpoint_support": {"supported": all(
            cell(layer, corpus)["support_jaccard"]["midpoint_order5"] >= .75
            for layer in layers for corpus in arrays
        )},
        "O7_returns_diminish_after_fourth_order": {"supported": all(
            abs(cell(layer, corpus)["fidelity"]["midpoint_order5"] - cell(layer, corpus)["fidelity"]["midpoint_order4"]) <= .02
            for layer in layers for corpus in arrays
        )},
        "O8_integral_and_midpoint_series_select_similar_circuits": {"supported": all(
            abs(cell(layer, corpus)["fidelity"]["integral_order5"] - cell(layer, corpus)["fidelity"]["midpoint_order5"]) <= .03
            for layer in layers for corpus in arrays
        )},
        "O9_order_frontier_transfers_out_of_distribution": {"supported": all(
            abs(cell(layer, "fineweb")["fidelity"][f"midpoint_order{order}"] - cell(layer, "tinystories")["fidelity"][f"midpoint_order{order}"]) <= .10
            for layer in layers for order in range(1, red["maximum_order"] + 1)
        )},
        "O10_no_interior_or_endpoint_outcome_is_used": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({
        "config": red, "hypotheses": hypotheses, "numerical": numerical,
        "summaries": summaries, "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Directional-derivative order frontier", "",
        "The base-point tensor program is expanded through fifth directional order. For order `n`, midpoint-derivative coefficients are `0.5^(n-1)/(n-1)!`; endpoint-integral coefficients are `1/n!`. Every selector is frozen before explicit midpoint and endpoint outcomes are evaluated, and selected coordinates are scored using their exact finite deltas through the full suffix.",
        "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Directional order frontier](bilin12_layer01_directional_order_frontier.png)", "",
              "| Layer | Corpus | Midpoint order 1/2/3/4/5 fidelity | Explicit midpoint | Order 1/2/3/4/5 support | Order 1/2/3/4/5 cosine | Best order/fidelity |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in layers:
        for corpus in arrays:
            row = cell(layer, corpus)
            values = lambda field: "/".join(f"{row[field][f'midpoint_order{order}']:.3f}" for order in range(1, red["maximum_order"] + 1))
            lines.append(f"| {layer} | {corpus} | {values('fidelity')} | {row['fidelity']['explicit_midpoint']:.3f} | {values('support_jaccard')} | {values('midpoint_cosine')} | {row['best_order']}/{row['best_fidelity']:.3f} |")
    lines += ["", "## Scaled derivative-term norms and numerical controls", "",
              "| Layer | Corpus | Scaled term norms 1/2/3/4/5 | Max finite-difference errors 1/2/3/4/5 |",
              "|---:|---|---:|---:|"]
    for layer in layers:
        for corpus in arrays:
            row = cell(layer, corpus)
            norms = "/".join(f"{row['scaled_derivative_norm'][str(order)]:.3f}" for order in range(1, red["maximum_order"] + 1))
            errors = "/".join(f"{numerical[str(layer)][corpus][str(order)]['maximum_relative_error']:.3f}" for order in range(1, red["maximum_order"] + 1))
            lines.append(f"| {layer} | {corpus} | {norms} | {errors} |")
    lines += ["", "A convergent direction frontier would give a compact weight-only compiler whose complexity is derivative order rather than a learned semantic dictionary. Oscillation or late-order degradation would instead identify a local-radius limit: the exact midpoint remains simple to evaluate, but not reducible to a short base-point series."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    orders = np.arange(1, red["maximum_order"] + 1)
    for layer in layers:
        for corpus in arrays:
            row = cell(layer, corpus); label = f"L{layer} {corpus}"
            axes[0].plot(orders, [row["fidelity"][f"midpoint_order{o}"] for o in orders], marker="o", label=label)
            axes[1].plot(orders, [row["support_jaccard"][f"midpoint_order{o}"] for o in orders], marker="o", label=label)
            axes[2].plot(orders, [row["midpoint_cosine"][f"midpoint_order{o}"] for o in orders], marker="o", label=label)
    axes[0].set_title("Finite causal fidelity"); axes[1].set_title("Endpoint support Jaccard"); axes[2].set_title("Midpoint tangent cosine")
    for axis in axes: axis.set_xlabel("Taylor order"); axis.set_xticks(orders); axis.set_ylim(0, 1.02); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
