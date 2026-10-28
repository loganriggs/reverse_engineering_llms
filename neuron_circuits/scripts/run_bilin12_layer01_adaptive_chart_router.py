#!/usr/bin/env python3
"""Prospective tensor router over native and rotated Layer-0/1 charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_dynamic_alignment_replication import (
    evaluate_deltas,
    final_product_delta,
    forward_layers_native,
    median,
    pair_cluster_interval,
    selected_delta,
)
from run_bilin18_layer01_native_rotation_tournament import (
    make_rotation,
    rotate,
    selected_rotated,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_adaptive_chart_router.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_adaptive_chart_router.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_ADAPTIVE_CHART_ROUTER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_adaptive_chart_router.png"


def cosine(left, right):
    return float((left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-30))


def write_fidelity(decoder, target_coordinates, approximation_coordinates):
    target = F.linear(target_coordinates, decoder)
    approximation = F.linear(approximation_coordinates, decoder)
    return float(1 - (target - approximation).square().sum() / target.square().sum().clamp_min(1e-30))


def native_density(delta, threshold=1e-4):
    maximum = delta.abs().max().clamp_min(1e-30)
    return float((delta.abs() >= threshold * maximum).float().mean())


def product_from_x0(model, x0, target_layer, position):
    """Differentiable native prefix ending at one MLP product vector."""
    dimension = model.config.n_embd
    x, value_bus = x0, None
    for layer, block in enumerate(model.blocks[:target_layer + 1]):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == target_layer:
            return product[0, position].float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raise AssertionError("unreachable")


def midpoint_product_jvp(model, base_ids, changed_ids, layer, position, path_point):
    dimension = model.config.n_embd
    with torch.no_grad():
        base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
        changed_x0 = F.rms_norm(model.embedding(changed_ids), (dimension,)).float()
    direction = changed_x0 - base_x0
    point = base_x0 + path_point * direction
    function = lambda value: product_from_x0(model, value, layer, position)
    with torch.enable_grad():
        _, tangent = torch.func.jvp(function, (point,), (direction,))
    return tangent.float(), point, direction


def chart_program(full, router, decoder, rank, permutations=None):
    """Rank with router coordinates; insert the exact finite delta after selection."""
    if permutations is None:
        router_coordinates = router
        full_coordinates = full
        chart_decoder = decoder
        router_write = F.linear(router_coordinates, chart_decoder)
        indices = (router_coordinates * (router_write @ chart_decoder)).abs().topk(rank).indices
        selected_router = selected_delta(router_coordinates, indices)
        selected_full = selected_delta(full_coordinates, indices)
        native_patch = selected_full
    else:
        router_coordinates = rotate(router, permutations)
        full_coordinates = rotate(full, permutations)
        chart_decoder = rotate(decoder, permutations)
        router_write = F.linear(router_coordinates, chart_decoder)
        indices = (router_coordinates * (router_write @ chart_decoder)).abs().topk(rank).indices
        selected_router_coordinates = selected_delta(router_coordinates, indices)
        selected_full_coordinates = selected_delta(full_coordinates, indices)
        selected_router = selected_rotated(router, indices, permutations)
        native_patch = selected_rotated(full, indices, permutations)
        # Fidelity is invariant, but calculating in chart coordinates avoids an
        # unnecessary inverse transform in the score used by the router.
        selected_router = selected_router_coordinates
    local = write_fidelity(chart_decoder, router_coordinates, selected_router)
    return native_patch, local, indices


@torch.inference_mode()
def finite_difference(model, point, direction, layer, position, epsilon):
    positive = product_from_x0(model, point + epsilon * direction, layer, position)
    negative = product_from_x0(model, point - epsilon * direction, layer, position)
    return (positive - negative) / (2 * epsilon)


def main():
    red = json.loads(CONFIG.read_text())
    parent = json.loads(Path(red["source_config"]).read_text())
    config = json.loads(Path(parent["source_config"]).read_text())
    evaluation_spec = json.loads(Path(config["evaluation_pair_config"]).read_text())
    seeds = parent["rotation_seeds"]
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for parameter in model.embedding.parameters(): parameter.requires_grad_(False)
    for parameter in model.blocks.parameters(): parameter.requires_grad_(False)
    for parameter in model.lm_head.parameters(): parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    layers = config["layers"]
    edit_position, position = config["edit_position"], config["prediction_position"]
    rank = red["rank"]
    rotations = {layer: {
        seed: make_rotation(
            model.blocks[layer].mlp.Down.weight.shape[1], config["rotation_rounds"], seed + layer, "cuda",
        ) for seed in seeds
    } for layer in layers}
    records = {str(layer): [] for layer in layers}

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family, pairs in evaluation_spec["new_families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, position,
                    config["evaluation_contexts_per_pair"], rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy()
                    changed = raw.copy(); changed[edit_position] = target_token
                    base_ids = torch.as_tensor(raw, device="cuda")[None]
                    changed_ids = torch.as_tensor(changed, device="cuda")[None]
                    with torch.no_grad():
                        base = forward_layers_native(model, base_ids, max(layers))
                        counter = forward_layers_native(model, changed_ids, max(layers))
                    for layer in layers:
                        decoder = model.blocks[layer].mlp.Down.weight.float()
                        full = final_product_delta(model, layer, base[layer], counter[layer], position)
                        tangent, point, direction = midpoint_product_jvp(
                            model, base_ids, changed_ids, layer, position, red["path_point"],
                        )
                        finite = finite_difference(
                            model, point, direction, layer, position, red["finite_difference_epsilon"],
                        )
                        jvp_cosine = cosine(tangent, finite)
                        jvp_error = float((tangent - finite).norm() / finite.norm().clamp_min(1e-30))

                        endpoint_programs = {}
                        midpoint_programs = {}
                        endpoint_local = {}
                        midpoint_local = {}
                        endpoint_indices = {}
                        midpoint_indices = {}
                        patch, local, indices = chart_program(full, full, decoder, rank)
                        endpoint_programs["native"] = patch; endpoint_local["native"] = local; endpoint_indices["native"] = indices
                        patch, local, indices = chart_program(full, tangent, decoder, rank)
                        midpoint_programs["native"] = patch; midpoint_local["native"] = local; midpoint_indices["native"] = indices
                        for seed in seeds:
                            name = str(seed); permutation = rotations[layer][seed]
                            patch, local, indices = chart_program(full, full, decoder, rank, permutation)
                            endpoint_programs[name] = patch; endpoint_local[name] = local; endpoint_indices[name] = indices
                            patch, local, indices = chart_program(full, tangent, decoder, rank, permutation)
                            midpoint_programs[name] = patch; midpoint_local[name] = local; midpoint_indices[name] = indices

                        deltas = {
                            **{f"endpoint_{name}": value for name, value in endpoint_programs.items()},
                            **{f"midpoint_{name}": value for name, value in midpoint_programs.items()},
                        }
                        with torch.no_grad():
                            effects = evaluate_deltas(model, layer, base[layer], deltas, full, position)
                        rotation_names = [str(seed) for seed in seeds]
                        endpoint_choice = max(rotation_names, key=lambda name: endpoint_local[name])
                        midpoint_choice = max(rotation_names, key=lambda name: midpoint_local[name])
                        midpoint_all_choice = max(["native"] + rotation_names, key=lambda name: midpoint_local[name])
                        downstream_oracle_choice = max(rotation_names, key=lambda name: effects[f"endpoint_{name}"])
                        chosen_patch = midpoint_programs[midpoint_choice]
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "jvp_cosine": jvp_cosine,
                            "jvp_relative_error": jvp_error,
                            "endpoint_choice": endpoint_choice,
                            "midpoint_choice": midpoint_choice,
                            "midpoint_all_choice": midpoint_all_choice,
                            "downstream_oracle_choice": downstream_oracle_choice,
                            "endpoint_oracle_agreement": endpoint_choice == downstream_oracle_choice,
                            "midpoint_endpoint_agreement": midpoint_choice == endpoint_choice,
                            "midpoint_oracle_agreement": midpoint_choice == downstream_oracle_choice,
                            "midpoint_selected_native_density": native_density(chosen_patch),
                            "effects": {
                                "native": effects["endpoint_native"],
                                "midpoint_native": effects["midpoint_native"],
                                "fixed_rotation": effects[f"endpoint_{rotation_names[0]}"],
                                "endpoint_router": effects[f"endpoint_{endpoint_choice}"],
                                "midpoint_router": effects[f"midpoint_{midpoint_choice}"],
                                "midpoint_all": effects[f"midpoint_{midpoint_all_choice}"],
                                "downstream_oracle": effects[f"endpoint_{downstream_oracle_choice}"],
                            },
                            "local": {
                                "endpoint_native": endpoint_local["native"],
                                "endpoint_router": endpoint_local[endpoint_choice],
                                "midpoint_native": midpoint_local["native"],
                                "midpoint_router": midpoint_local[midpoint_choice],
                            },
                        })
                print("adaptive router", corpus, family, source_token, target_token, flush=True)

    summaries = {str(layer): {} for layer in layers}
    for layer in layers:
        for corpus_index, corpus in enumerate(config["evaluation"]):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            methods = {name: median([row["effects"][name] for row in selected]) for name in selected[0]["effects"]}
            midpoint_advantage, midpoint_ci = pair_cluster_interval(
                selected, lambda row: row["effects"]["midpoint_router"] - row["effects"]["native"],
                red["bootstrap_samples"], red["bootstrap_seed"] + layer * 100 + corpus_index,
            )
            summaries[str(layer)][corpus] = {
                "methods": methods,
                "midpoint_minus_native_pair_median": midpoint_advantage,
                "midpoint_minus_native_pair_ci": midpoint_ci,
                "jvp_cosine_minimum": min(row["jvp_cosine"] for row in selected),
                "jvp_relative_error_maximum": max(row["jvp_relative_error"] for row in selected),
                "endpoint_oracle_agreement": float(np.mean([row["endpoint_oracle_agreement"] for row in selected])),
                "midpoint_endpoint_agreement": float(np.mean([row["midpoint_endpoint_agreement"] for row in selected])),
                "midpoint_oracle_agreement": float(np.mean([row["midpoint_oracle_agreement"] for row in selected])),
                "midpoint_native_choice_fraction": float(np.mean([row["midpoint_all_choice"] == "native" for row in selected])),
                "rotated_native_density_minimum": min(row["midpoint_selected_native_density"] for row in selected),
                "rotated_native_density_median": median([row["midpoint_selected_native_density"] for row in selected]),
            }

    def cell(layer, corpus): return summaries[str(layer)][corpus]

    jvp_valid = all(
        cell(layer, corpus)["jvp_cosine_minimum"] >= 0.999
        and cell(layer, corpus)["jvp_relative_error_maximum"] <= 0.02
        for layer in layers for corpus in config["evaluation"]
    )
    exact_recovers = all(
        cell(layer, corpus)["methods"]["downstream_oracle"] - cell(layer, corpus)["methods"]["endpoint_router"] <= 0.05
        and cell(layer, corpus)["endpoint_oracle_agreement"] >= 0.30
        for layer in layers for corpus in config["evaluation"]
    )
    midpoint_tracks = all(
        cell(layer, corpus)["methods"]["endpoint_router"] - cell(layer, corpus)["methods"]["midpoint_router"] <= 0.05
        and cell(layer, corpus)["midpoint_endpoint_agreement"] >= 0.30
        for layer in layers for corpus in config["evaluation"]
    )
    midpoint_beats_native = all(
        cell(layer, corpus)["midpoint_minus_native_pair_median"] >= 0.03
        and cell(layer, corpus)["midpoint_minus_native_pair_ci"][0] > 0
        for layer in layers for corpus in config["evaluation"]
    )
    gap_closed = True
    gap_fractions = {}
    for layer in layers:
        gap_fractions[str(layer)] = {}
        for corpus in config["evaluation"]:
            methods = cell(layer, corpus)["methods"]
            fraction = (methods["midpoint_router"] - methods["fixed_rotation"]) / max(
                methods["downstream_oracle"] - methods["fixed_rotation"], 1e-30,
            )
            gap_fractions[str(layer)][corpus] = fraction
            gap_closed = gap_closed and fraction >= 0.50
    native_chosen = all(
        cell(layer, corpus)["midpoint_native_choice_fraction"] >= 0.50
        for layer in layers for corpus in config["evaluation"]
    )
    dense = all(
        cell(layer, corpus)["rotated_native_density_minimum"] >= 0.75
        for layer in layers for corpus in config["evaluation"]
    )
    ood = all(
        abs(
            (cell(layer, "fineweb")["methods"]["midpoint_router"] - cell(layer, "fineweb")["methods"]["native"])
            - (cell(layer, "tinystories")["methods"]["midpoint_router"] - cell(layer, "tinystories")["methods"]["native"])
        ) <= 0.10 for layer in layers
    )
    hypotheses = {
        "A0_midpoint_jvp_is_numerically_correct": {"supported": jvp_valid},
        "A1_exact_local_router_recovers_the_downstream_oracle": {"supported": exact_recovers},
        "A2_midpoint_router_tracks_the_exact_local_router": {"supported": midpoint_tracks},
        "A3_midpoint_adaptive_dictionary_beats_native": {"supported": midpoint_beats_native},
        "A4_midpoint_router_closes_half_the_adaptive_oracle_gap": {"supported": gap_closed, "fractions": gap_fractions},
        "A5_native_wins_when_it_is_a_router_candidate": {"supported": native_chosen},
        "A6_adaptive_rotation_is_not_a_sparse_native_intervention": {"supported": dense},
        "A7_router_behavior_transfers_ood": {"supported": ood},
    }
    OUTPUT.write_text(json.dumps({
        "config": red, "hypotheses": hypotheses, "summaries": summaries, "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Prospective adaptive chart router for Layer-0/1 edits", "",
        "Seven random rotated product bases form an adaptive dictionary. The endpoint-local router chooses a basis using only residual-write reconstruction at the edited MLP boundary. The prospective router instead contracts the exact tensor prefix once at the embedding midpoint, ranks 256 coordinates per basis from that JVP, and chooses the basis with the best tangent-write reconstruction. Neither router sees suffix logits or downstream fidelity. Actual finite product deltas are inserted only after support and basis choice are frozen.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Adaptive router](bilin12_layer01_adaptive_chart_router.png)", "", "| Layer | Corpus | Native / fixed rotation / endpoint router / midpoint router / downstream oracle | Midpoint−native pair median (95% CI) | Endpoint↔oracle / midpoint↔endpoint agreement | Native chosen among all charts | Rotated patch native density |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in layers:
        for corpus in config["evaluation"]:
            row = cell(layer, corpus); methods = row["methods"]; ci = row["midpoint_minus_native_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {methods['native']:.3f}/{methods['fixed_rotation']:.3f}/{methods['endpoint_router']:.3f}/{methods['midpoint_router']:.3f}/{methods['downstream_oracle']:.3f} | "
                f"{row['midpoint_minus_native_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) | "
                f"{row['endpoint_oracle_agreement']:.3f}/{row['midpoint_endpoint_agreement']:.3f} | {row['midpoint_native_choice_fraction']:.3f} | {row['rotated_native_density_median']:.3f} (min {row['rotated_native_density_minimum']:.3f}) |"
            )
    lines += ["", "The rotated rank-256 program is sparse only in its chosen chart. After inverse rotation it generally changes most native product coordinates, so it is an effect-compression device rather than a literal sparse-neuron circuit. The native program remains the appropriate object for independently ablatable neuron interventions; an adaptive chart is useful only if its router predicts held-out effects well enough to justify the added dictionary and routing description."]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    method_names = ["native", "fixed_rotation", "endpoint_router", "midpoint_router", "downstream_oracle"]
    for axis, layer in zip(axes, layers):
        x = np.arange(len(method_names)); width = 0.35
        for offset, corpus in zip((-width / 2, width / 2), config["evaluation"]):
            axis.bar(x + offset, [cell(layer, corpus)["methods"][name] for name in method_names], width, label=corpus)
        axis.set_xticks(x); axis.set_xticklabels(method_names, rotation=25, ha="right"); axis.set_ylim(0, 1.02)
        axis.set_title(f"Layer {layer}"); axis.set_ylabel("median Fisher fidelity"); axis.grid(axis="y", alpha=0.25); axis.legend()
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
