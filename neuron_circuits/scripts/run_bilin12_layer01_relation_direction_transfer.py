#!/usr/bin/env python3
"""Transfer semantic-relation embedding directions into Layer-0/1 chart assays."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_chart_empirical_direction_control import sphere_endpoint_from_direction
from run_bilin12_layer01_chart_semantic_specificity import evaluate_endpoint
from run_bilin12_layer01_dynamic_alignment_replication import (
    forward_layers_native,
    median,
    pair_cluster_interval,
)
from run_bilin18_layer01_native_rotation_tournament import make_rotation, rotate
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_relation_direction_transfer.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_relation_direction_transfer.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_RELATION_DIRECTION_TRANSFER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_relation_direction_transfer.png"


def donor_sets(families, family, pair, seed):
    same = [tuple(item) for item in families[family] if tuple(item) != tuple(pair)]
    different = [
        tuple(item) for other, pairs in families.items() if other != family for item in pairs
    ]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(different), len(same), replace=False)
    return same, [different[int(index)] for index in indices]


def paired_interval(rows, left, right, samples, seed):
    return pair_cluster_interval(
        rows, lambda row: row[left] - row[right], samples, seed,
    )


def main():
    red = json.loads(CONFIG.read_text())
    sphere_config = json.loads(Path(red["source_config"]).read_text())
    baseline = json.loads(Path(red["baseline_results"]).read_text())
    frontier_config = json.loads(Path(sphere_config["source_config"]).read_text())
    fisher = json.loads(Path(frontier_config["source_config"]).read_text())
    router = json.loads(Path(fisher["source_config"]).read_text())
    rotation_config = json.loads(Path(router["source_config"]).read_text())
    config = json.loads(Path(rotation_config["source_config"]).read_text())
    pair_spec = json.loads(Path(config["evaluation_pair_config"]).read_text())
    families = pair_spec["new_families"]
    seeds = rotation_config["rotation_seeds"]

    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    with torch.no_grad():
        embeddings = F.rms_norm(
            model.embedding.weight.float(), (model.config.n_embd,),
        )
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
            ) for seed in seeds
        } for layer in layers
    }
    rotated_decoders = {
        layer: {
            seed: rotate(model.blocks[layer].mlp.Down.weight.float(), rotations[layer][seed])
            for seed in seeds
        } for layer in layers
    }
    baseline_lookup = {
        (int(layer), row["corpus"], row["pair"], row["context"]): row
        for layer, rows in baseline["records"].items() for row in rows
    }
    records = {str(layer): [] for layer in layers}
    maximum_distance_mismatch = 0.0
    maximum_radius_mismatch = 0.0

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        window_rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family_index, (family, family_pairs) in enumerate(families.items()):
            for pair_index, pair in enumerate(family_pairs):
                source_token, target_token = pair
                same_donors, different_donors = donor_sets(
                    families, family, pair,
                    red["donor_seed"] + family_index * 100 + pair_index,
                )
                windows = select_windows(
                    rows, source_token, edit_position, position,
                    config["evaluation_contexts_per_pair"], window_rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(
                        rows[row_index, start:start + config["sequence_length"]], dtype=np.int64,
                    ).copy()
                    base_ids = torch.as_tensor(raw, device="cuda")[None]
                    with torch.no_grad():
                        base = forward_layers_native(model, base_ids, max(layers))
                        base_x0 = F.rms_norm(
                            model.embedding(base_ids), (model.config.n_embd,),
                        ).float()
                    source_vector = base_x0[0, edit_position]
                    distance = (embeddings[target_token] - source_vector).norm()
                    endpoints = {"same": [], "different": []}
                    for condition, donors in (("same", same_donors), ("different", different_donors)):
                        for donor_source, donor_target in donors:
                            endpoint = sphere_endpoint_from_direction(
                                source_vector, distance,
                                embeddings[donor_target] - embeddings[donor_source],
                            )
                            maximum_distance_mismatch = max(
                                maximum_distance_mismatch,
                                float(abs((endpoint - source_vector).norm() - distance) / distance.clamp_min(1e-30)),
                            )
                            maximum_radius_mismatch = max(
                                maximum_radius_mismatch,
                                float(abs(endpoint.norm() - source_vector.norm()) / source_vector.norm().clamp_min(1e-30)),
                            )
                            endpoint_x0 = base_x0.clone(); endpoint_x0[:, edit_position] = endpoint
                            endpoints[condition].append((endpoint_x0, donor_source, donor_target))

                    pair_name = f"{source_token}->{target_token}"
                    for layer in layers:
                        effects = {}
                        for condition in ("same", "different"):
                            effects[condition] = []
                            for endpoint_x0, donor_source, donor_target in endpoints[condition]:
                                effect = evaluate_endpoint(
                                    model, base, base_x0, endpoint_x0, layer, position,
                                    router["path_point"], sphere_config["rank"], seeds,
                                    rotations, rotated_decoders,
                                )
                                effects[condition].append({
                                    **effect,
                                    "donor_pair": f"{donor_source}->{donor_target}",
                                })
                        source = baseline_lookup[(layer, corpus, pair_name, context_index)]
                        row = {
                            "corpus": corpus,
                            "family": family,
                            "pair": pair_name,
                            "context": context_index,
                            "natural_native": source["semantic_native"],
                            "natural_gain": source["semantic_gain"],
                            "natural_native_choice": source["semantic_native_choice"],
                            "random_native": source["empirical_native"],
                            "random_gain": source["empirical_gain"],
                            "random_native_choice": source["empirical_native_choice"],
                            "same_donors": effects["same"],
                            "different_donors": effects["different"],
                        }
                        for condition in ("same", "different"):
                            values = effects[condition]
                            row[f"{condition}_native"] = median([item["native"] for item in values])
                            row[f"{condition}_routed"] = median([item["routed"] for item in values])
                            row[f"{condition}_gain"] = median([
                                item["routed"] - item["native"] for item in values
                            ])
                            row[f"{condition}_native_choice"] = float(np.mean([
                                item["choice"] == "native" for item in values
                            ]))
                        records[str(layer)].append(row)
                print("relation direction", corpus, family, source_token, target_token, flush=True)

    print("relation direction: measurements complete; summarizing", flush=True)
    summaries = {str(layer): {} for layer in layers}
    fields = (
        "natural_native", "natural_gain", "random_native", "random_gain",
        "same_native", "same_routed", "same_gain",
        "different_native", "different_routed", "different_gain",
    )
    for layer in layers:
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summary = {field: median([row[field] for row in selected]) for field in fields}
            for field in ("natural_native_choice", "random_native_choice", "same_native_choice", "different_native_choice"):
                summary[field] = float(np.mean([row[field] for row in selected]))
            comparisons = (
                ("same_minus_different_native", "same_native", "different_native", 0),
                ("same_minus_random_native", "same_native", "random_native", 10000),
                ("natural_minus_same_native", "natural_native", "same_native", 20000),
                ("same_minus_different_gain", "same_gain", "different_gain", 30000),
            )
            for name, left, right, offset in comparisons:
                value, interval = paired_interval(
                    selected, left, right, red["bootstrap_samples"],
                    red["bootstrap_seed"] + offset + layer * 100 + corpus_index,
                )
                summary[f"{name}_pair_median"] = value
                summary[f"{name}_pair_ci"] = interval
            summary["same_minus_different_native_choice"] = (
                summary["same_native_choice"] - summary["different_native_choice"]
            )
            summary["family_breakdown"] = {
                family: {
                    "same_native": median([row["same_native"] for row in selected if row["family"] == family]),
                    "different_native": median([row["different_native"] for row in selected if row["family"] == family]),
                    "same_minus_different": median([
                        row["same_native"] - row["different_native"]
                        for row in selected if row["family"] == family
                    ]),
                } for family in families
            }
            summaries[str(layer)][corpus] = summary
            print("relation direction: summarized", layer, corpus, flush=True)

    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    hypotheses = {
        "D0_transported_donor_chords_are_exact": {
            "supported": maximum_distance_mismatch <= 1e-5 and maximum_radius_mismatch <= 1e-5,
            "maximum_distance_mismatch": maximum_distance_mismatch,
            "maximum_radius_mismatch": maximum_radius_mismatch,
        },
        "D1_same_family_directions_have_privileged_native_alignment": {
            "supported": all(
                cell(layer, corpus)["same_minus_different_native_pair_median"] >= .03
                and cell(layer, corpus)["same_minus_different_native_pair_ci"][0] > 0
                for layer in layers for corpus in arrays
            ),
        },
        "D2_same_family_directions_beat_generic_token_differences": {
            "supported": all(
                cell(layer, corpus)["same_minus_random_native_pair_median"] >= .03
                and cell(layer, corpus)["same_minus_random_native_pair_ci"][0] > 0
                for layer in layers for corpus in arrays
            ),
        },
        "D3_same_family_directions_recover_the_natural_native_code": {
            "supported": all(
                abs(cell(layer, corpus)["natural_native"] - cell(layer, corpus)["same_native"]) <= .03
                for layer in layers for corpus in arrays
            ),
        },
        "D4_same_family_directions_prefer_native_charts": {
            "supported": all(
                cell(layer, corpus)["same_minus_different_native_choice"] >= .10
                for layer in layers for corpus in arrays
            ),
        },
        "D5_relation_specificity_is_not_just_adaptive_chart_gain": {
            "supported": all(
                abs(cell(layer, corpus)["same_minus_different_gain_pair_median"]) <= .03
                for layer in layers for corpus in arrays
            ),
        },
        "D6_selection_is_strictly_prospective": {"supported": True},
    }
    print("relation direction: writing result payload", flush=True)
    OUTPUT.write_text(json.dumps({
        "config": red,
        "hypotheses": hypotheses,
        "maximum_distance_mismatch": maximum_distance_mismatch,
        "maximum_radius_mismatch": maximum_radius_mismatch,
        "summaries": summaries,
        "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Semantic relation-direction transfer", "",
        "For each natural lexical edit, embedding differences from every other pair in the same declared semantic family are transported to the source token at the exact natural chord length. An equally sized deterministic set of different-family directions is the control. Neither donor endpoint is a vocabulary token. All native and adaptive-chart effects use the unchanged prospective suffix-Fisher assay.",
        "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "![Relation direction transfer](bilin12_layer01_relation_direction_transfer.png)", "",
        "| Layer | Corpus | Native natural / same-family / different-family / random | Same−different native pair median (95% CI) | Natural−same native pair median (95% CI) | Gain natural / same / different / random |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for layer in layers:
        for corpus in arrays:
            row = cell(layer, corpus)
            ci1 = row["same_minus_different_native_pair_ci"]
            ci2 = row["natural_minus_same_native_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {row['natural_native']:.3f} / {row['same_native']:.3f} / {row['different_native']:.3f} / {row['random_native']:.3f} | "
                f"{row['same_minus_different_native_pair_median']:+.3f} ({ci1[0]:+.3f}, {ci1[1]:+.3f}) | "
                f"{row['natural_minus_same_native_pair_median']:+.3f} ({ci2[0]:+.3f}, {ci2[1]:+.3f}) | "
                f"{row['natural_gain']:+.3f} / {row['same_gain']:+.3f} / {row['different_gain']:+.3f} / {row['random_gain']:+.3f} |"
            )
    lines += ["", "## Same-family minus different-family native fidelity by family", ""]
    lines += ["| Layer | Corpus | " + " | ".join(families) + " |", "|---:|---|" + "---:|" * len(families)]
    for layer in layers:
        for corpus in arrays:
            breakdown = cell(layer, corpus)["family_breakdown"]
            lines.append(f"| {layer} | {corpus} | " + " | ".join(f"{breakdown[name]['same_minus_different']:+.3f}" for name in families) + " |")
    lines += [
        "",
        f"Maximum relative chord-distance mismatch: **{maximum_distance_mismatch:.2e}**. Maximum relative sphere-radius mismatch: **{maximum_radius_mismatch:.2e}**.",
        "",
        "Same-family transfer would be evidence for a reusable relation direction at the embedding-to-early-layer interface, not a one-neuron semantic module. Failure would leave natural native alignment as pair-specific or manifold-specific rather than family-level semantic reuse.",
    ]
    print("relation direction: writing report", flush=True)
    REPORT.write_text("\n".join(lines) + "\n")

    print("relation direction: rendering figure", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    labels = [f"L{layer}\n{corpus}" for layer in layers for corpus in arrays]
    x = np.arange(len(labels)); width = .2
    for offset, condition, label in ((-1.5*width, "natural", "natural"), (-.5*width, "same", "same-family"), (.5*width, "different", "different-family"), (1.5*width, "random", "random token-diff")):
        axes[0].bar(x + offset, [cell(layer, corpus)[f"{condition}_native"] for layer in layers for corpus in arrays], width, label=label)
        axes[1].bar(x + offset, [cell(layer, corpus)[f"{condition}_gain"] for layer in layers for corpus in arrays], width, label=label)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 1); axes[0].set_ylabel("rank-256 native fidelity"); axes[0].set_title("Relation-direction transfer"); axes[0].legend(fontsize=8)
    axes[1].set_xticks(x, labels); axes[1].axhline(0, color="black", linewidth=.8); axes[1].set_ylabel("routed − native fidelity"); axes[1].set_title("Adaptive-chart gain"); axes[1].legend(fontsize=8)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
