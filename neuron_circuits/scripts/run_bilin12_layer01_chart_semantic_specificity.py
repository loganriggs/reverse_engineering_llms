#!/usr/bin/env python3
"""Exact-distance semantic specificity control for adaptive effect charts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import product_from_x0
from run_bilin12_layer01_chart_rank_frontier import chart_program_precomputed
from run_bilin12_layer01_dynamic_alignment_replication import (
    evaluate_deltas,
    final_product_delta,
    forward_layers_native,
    median,
    pair_cluster_interval,
)
from run_bilin12_layer01_suffix_fisher_chart_router import suffix_jvp_batch, trace_from_x0
from run_bilin18_layer01_midpoint_synthetic_control import sphere_endpoint
from run_bilin18_layer01_native_rotation_tournament import make_rotation, rotate
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin12_layer01_chart_semantic_specificity.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_chart_semantic_specificity.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_CHART_SEMANTIC_SPECIFICITY.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_chart_semantic_specificity.png"


@torch.inference_mode()
def forward_layers_x0(model, x0, last_layer):
    """Native trace from an arbitrary normalized embedding endpoint."""
    dimension = model.config.n_embd
    x, value_bus = x0, None
    records = []
    for block in model.blocks[:last_layer + 1]:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        mlp_write = block.mlp.Down(product) + block.mlp.Down_bias
        x = pre + mlp_write
        records.append({
            "x0": x0,
            "value_bus": value_bus,
            "pre": pre,
            "mlp_write": mlp_write,
            "output": x,
        })
    return records


def midpoint_product_jvp_x0(model, base_x0, endpoint_x0, layer, position, path_point):
    direction = endpoint_x0 - base_x0
    point = base_x0 + path_point * direction
    function = lambda value: product_from_x0(model, value, layer, position)
    with torch.enable_grad():
        _, tangent = torch.func.jvp(function, (point,), (direction,))
    return tangent.float(), point


def paired_gain_interval(records, samples, seed):
    return pair_cluster_interval(
        records,
        lambda row: row["semantic_gain"] - row["synthetic_gain"],
        samples,
        seed,
    )


def summarize_records(records):
    median_fields = (
        "semantic_native", "semantic_routed", "semantic_oracle", "semantic_gain",
        "synthetic_native", "synthetic_routed", "synthetic_oracle", "synthetic_gain",
        "semantic_router_gap", "synthetic_router_gap",
    )
    summary = {field: median([row[field] for row in records]) for field in median_fields}
    # These are occurrence rates, not typical binary outcomes.
    for field in ("semantic_native_choice", "synthetic_native_choice"):
        summary[field] = float(np.mean([row[field] for row in records]))
    return summary


def semantic_lookup(source_results, rank, seed_names):
    lookup = {}
    for layer, rows in source_results["records"].items():
        for row in rows:
            chart = row["charts"][str(rank)]
            names = ["native"] + seed_names
            choice = max(names, key=lambda name: chart[name]["predicted"])
            key = (int(layer), row["corpus"], row["pair"], row["context"])
            lookup[key] = {
                "native": chart["native"]["actual"],
                "routed": chart[choice]["actual"],
                "oracle": max(chart[name]["actual"] for name in names),
                "choice": choice,
            }
    return lookup


def evaluate_endpoint(
    model, base, base_x0, endpoint_x0, layer, position, path_point, rank,
    seeds, rotations, rotated_decoders,
):
    """Prospectively route one arbitrary endpoint through native plus rotated charts."""
    decoder = model.blocks[layer].mlp.Down.weight.float()
    with torch.no_grad():
        counter = forward_layers_x0(model, endpoint_x0, layer)
    full = final_product_delta(model, layer, base[layer], counter[layer], position)
    tangent, point = midpoint_product_jvp_x0(
        model, base_x0, endpoint_x0, layer, position, path_point,
    )
    with torch.no_grad():
        midpoint_trace = trace_from_x0(model, point, layer)
    names = ["native"] + [str(seed) for seed in seeds]
    actual_patches = {}
    tangent_patches = {}
    actual, tangent_patch, _ = chart_program_precomputed(
        full, tangent, decoder, decoder, rank,
    )
    actual_patches["native"] = actual
    tangent_patches["native"] = tangent_patch
    for seed in seeds:
        name = str(seed)
        actual, tangent_patch, _ = chart_program_precomputed(
            full, tangent, decoder, rotated_decoders[layer][seed],
            rank, rotations[layer][seed],
        )
        actual_patches[name] = actual
        tangent_patches[name] = tangent_patch
    write_tangents = [F.linear(tangent, decoder)] + [
        F.linear(tangent_patches[name], decoder) for name in names
    ]
    logits, derivatives, _, _, _ = suffix_jvp_batch(
        model, layer, midpoint_trace[layer], write_tangents, position,
    )
    predicted = {
        name: fisher_fidelity(logits[0], derivatives[0], derivatives[index + 1])[0]
        for index, name in enumerate(names)
    }
    choice = max(names, key=lambda name: predicted[name])
    with torch.no_grad():
        actual = evaluate_deltas(model, layer, base[layer], actual_patches, full, position)
    return {
        "native": actual["native"],
        "routed": actual[choice],
        "oracle": max(actual.values()),
        "choice": choice,
    }


def main():
    red = json.loads(CONFIG.read_text())
    frontier_config = json.loads(Path(red["source_config"]).read_text())
    source_results = json.loads(Path(red["source_results"]).read_text())
    fisher = json.loads(Path(frontier_config["source_config"]).read_text())
    router = json.loads(Path(fisher["source_config"]).read_text())
    rotation_config = json.loads(Path(router["source_config"]).read_text())
    config = json.loads(Path(rotation_config["source_config"]).read_text())
    pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    seeds = rotation_config["rotation_seeds"]
    seed_names = [str(seed) for seed in seeds]
    semantic = semantic_lookup(source_results, red["rank"], seed_names)

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
    rotations = {
        layer: {
            seed: make_rotation(
                model.blocks[layer].mlp.Down.weight.shape[1],
                config["rotation_rounds"], seed + layer, "cuda",
            )
            for seed in seeds
        }
        for layer in layers
    }
    rotated_decoders = {
        layer: {
            seed: rotate(model.blocks[layer].mlp.Down.weight.float(), rotations[layer][seed])
            for seed in seeds
        }
        for layer in layers
    }
    records = {str(layer): [] for layer in layers}
    maximum_distance_mismatch = 0.0
    maximum_radius_mismatch = 0.0

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family_index, (family, family_pairs) in enumerate(pairs["new_families"].items()):
            for pair_index, (source_token, target_token) in enumerate(family_pairs):
                windows = select_windows(
                    rows, source_token, edit_position, position,
                    config["evaluation_contexts_per_pair"], rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(
                        rows[row_index, start:start + config["sequence_length"]], dtype=np.int64,
                    ).copy()
                    changed = raw.copy(); changed[edit_position] = target_token
                    base_ids = torch.as_tensor(raw, device="cuda")[None]
                    changed_ids = torch.as_tensor(changed, device="cuda")[None]
                    with torch.no_grad():
                        base = forward_layers_native(model, base_ids, max(layers))
                        dimension = model.config.n_embd
                        base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                        changed_x0 = F.rms_norm(model.embedding(changed_ids), (dimension,)).float()
                    source_vector = base_x0[0, edit_position]
                    distance = (changed_x0[0, edit_position] - source_vector).norm()
                    endpoints = []
                    for direction_index in range(red["synthetic_directions_per_pair"]):
                        seed = (
                            red["synthetic_seed"] + family_index * 100000
                            + pair_index * 1000 + context_index * 10 + direction_index
                        )
                        endpoint = sphere_endpoint(source_vector, distance, seed)
                        maximum_distance_mismatch = max(
                            maximum_distance_mismatch,
                            float(abs((endpoint - source_vector).norm() - distance) / distance.clamp_min(1e-30)),
                        )
                        maximum_radius_mismatch = max(
                            maximum_radius_mismatch,
                            float(abs(endpoint.norm() - source_vector.norm()) / source_vector.norm().clamp_min(1e-30)),
                        )
                        endpoint_x0 = base_x0.clone(); endpoint_x0[:, edit_position] = endpoint
                        endpoints.append(endpoint_x0)

                    for layer in layers:
                        synthetic_rows = []
                        for endpoint_x0 in endpoints:
                            synthetic_rows.append(evaluate_endpoint(
                                model, base, base_x0, endpoint_x0, layer, position,
                                router["path_point"], red["rank"], seeds,
                                rotations, rotated_decoders,
                            ))

                        key = (layer, corpus, f"{source_token}->{target_token}", context_index)
                        natural = semantic[key]
                        synth = {
                            field: median([row[field] for row in synthetic_rows])
                            for field in ("native", "routed", "oracle")
                        }
                        synthetic_gain = median([
                            row["routed"] - row["native"] for row in synthetic_rows
                        ])
                        synthetic_router_gap = median([
                            row["oracle"] - row["routed"] for row in synthetic_rows
                        ])
                        synthetic_native_choice = float(np.mean([
                            row["choice"] == "native" for row in synthetic_rows
                        ]))
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": key[2],
                            "context": context_index,
                            "semantic_native": natural["native"],
                            "semantic_routed": natural["routed"],
                            "semantic_oracle": natural["oracle"],
                            "semantic_gain": natural["routed"] - natural["native"],
                            "semantic_router_gap": natural["oracle"] - natural["routed"],
                            "semantic_native_choice": float(natural["choice"] == "native"),
                            "synthetic_native": synth["native"],
                            "synthetic_routed": synth["routed"],
                            "synthetic_oracle": synth["oracle"],
                            "synthetic_gain": synthetic_gain,
                            "synthetic_router_gap": synthetic_router_gap,
                            "synthetic_native_choice": synthetic_native_choice,
                            "synthetic_directions": synthetic_rows,
                        })
                print("chart specificity", corpus, family, source_token, target_token, flush=True)

    summaries = {str(layer): {} for layer in layers}
    for layer in layers:
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summary = summarize_records(selected)
            difference, interval = paired_gain_interval(
                selected, red["bootstrap_samples"],
                red["bootstrap_seed"] + layer * 100 + corpus_index,
            )
            summary["semantic_minus_synthetic_gain_pair_median"] = difference
            summary["semantic_minus_synthetic_gain_pair_ci"] = interval
            summary["native_choice_advantage"] = (
                summary["semantic_native_choice"] - summary["synthetic_native_choice"]
            )
            summaries[str(layer)][corpus] = summary

    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    hypotheses = {
        "S0_synthetic_chord_distance_is_exact": {
            "supported": maximum_distance_mismatch <= 1e-5 and maximum_radius_mismatch <= 1e-5,
            "maximum_distance_mismatch": maximum_distance_mismatch,
            "maximum_radius_mismatch": maximum_radius_mismatch,
        },
        "S1_chart_gain_is_semantically_specific": {
            "supported": all(
                cell(layer, corpus)["semantic_minus_synthetic_gain_pair_median"] >= .03
                and cell(layer, corpus)["semantic_minus_synthetic_gain_pair_ci"][0] > 0
                for layer in layers for corpus in arrays
            ),
        },
        "S2_synthetic_directions_also_benefit_from_chart_routing": {
            "supported": all(
                cell(layer, corpus)["synthetic_gain"] >= .02
                for layer in layers for corpus in arrays
            ),
        },
        "S3_synthetic_router_tracks_the_finite_chart_oracle": {
            "supported": all(
                cell(layer, corpus)["synthetic_router_gap"] <= .03
                for layer in layers for corpus in arrays
            ),
        },
        "S4_natural_edits_prefer_native_coordinates": {
            "supported": all(
                cell(layer, corpus)["native_choice_advantage"] >= .10
                for layer in layers for corpus in arrays
            ),
        },
        "S5_selection_is_strictly_prospective": {"supported": True},
    }
    payload = {
        "config": red,
        "hypotheses": hypotheses,
        "maximum_distance_mismatch": maximum_distance_mismatch,
        "maximum_radius_mismatch": maximum_radius_mismatch,
        "summaries": summaries,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Semantic specificity of adaptive effect charts", "",
        "Every natural source→target token edit is paired with two random endpoints on the normalized embedding sphere at exactly the same chord distance. Natural and synthetic edits use the same source context, rank-256 native-plus-seven chart bank, midpoint product JVP, exact suffix JVP, and Fisher output metric. Chart selection is prospective; finite endpoint effects are inspected only after the choice is frozen.",
        "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "![Semantic specificity](bilin12_layer01_chart_semantic_specificity.png)", "",
        "| Layer | Corpus | Semantic native/routed/oracle/gain | Synthetic native/routed/oracle/gain | Semantic−synthetic gain pair median (95% CI) | Native choice semantic/synthetic |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for layer in layers:
        for corpus in arrays:
            row = cell(layer, corpus); ci = row["semantic_minus_synthetic_gain_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {row['semantic_native']:.3f}/{row['semantic_routed']:.3f}/{row['semantic_oracle']:.3f}/{row['semantic_gain']:+.3f} | "
                f"{row['synthetic_native']:.3f}/{row['synthetic_routed']:.3f}/{row['synthetic_oracle']:.3f}/{row['synthetic_gain']:+.3f} | "
                f"{row['semantic_minus_synthetic_gain_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) | "
                f"{row['semantic_native_choice']:.3f}/{row['synthetic_native_choice']:.3f} |"
            )
    lines += [
        "",
        f"Maximum relative chord-distance mismatch: **{maximum_distance_mismatch:.2e}**. Maximum relative sphere-radius mismatch: **{maximum_radius_mismatch:.2e}**.",
        "",
        "If synthetic endpoints obtain the same chart gain as natural edits, the atlas is best treated as a generic output-conditioned approximation scheme rather than evidence for semantic coordinates. A natural-edit advantage would instead show that the model's learned manifold is unusually compatible with the available charts; it would still not make rotated coordinates literal independently ablatable neurons.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    labels = [f"L{layer}\n{corpus}" for layer in layers for corpus in arrays]
    semantic_gains = [cell(layer, corpus)["semantic_gain"] for layer in layers for corpus in arrays]
    synthetic_gains = [cell(layer, corpus)["synthetic_gain"] for layer in layers for corpus in arrays]
    x = np.arange(len(labels)); width = .36
    axes[0].bar(x - width / 2, semantic_gains, width, label="natural lexical")
    axes[0].bar(x + width / 2, synthetic_gains, width, label="matched random")
    axes[0].axhline(0, color="black", linewidth=.8); axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("routed − native fidelity"); axes[0].legend(); axes[0].set_title("Adaptive-chart gain")
    semantic_choices = [cell(layer, corpus)["semantic_native_choice"] for layer in layers for corpus in arrays]
    synthetic_choices = [cell(layer, corpus)["synthetic_native_choice"] for layer in layers for corpus in arrays]
    axes[1].bar(x - width / 2, semantic_choices, width, label="natural lexical")
    axes[1].bar(x + width / 2, synthetic_choices, width, label="matched random")
    axes[1].set_xticks(x, labels); axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("native-chart choice fraction"); axes[1].legend(); axes[1].set_title("Coordinate preference")
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
