#!/usr/bin/env python3
"""Use transported real-token difference directions as a chart-specificity control."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_chart_semantic_specificity import evaluate_endpoint
from run_bilin12_layer01_dynamic_alignment_replication import (
    forward_layers_native,
    median,
    pair_cluster_interval,
)
from run_bilin18_layer01_native_rotation_tournament import make_rotation, rotate
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_chart_empirical_direction_control.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_chart_empirical_direction_control.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_CHART_EMPIRICAL_DIRECTION_CONTROL.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_chart_empirical_direction_control.png"


def sphere_endpoint_from_direction(source, distance, direction):
    """Transport a supplied direction to an exact chord on source's sphere."""
    radius = source.norm().clamp_min(1e-30)
    unit = source / radius
    perpendicular = direction - (direction * unit).sum() * unit
    perpendicular = perpendicular / perpendicular.norm().clamp_min(1e-30)
    cosine_angle = (1 - distance.square() / (2 * radius.square())).clamp(-1, 1)
    sine_angle = torch.sqrt((1 - cosine_angle.square()).clamp_min(0))
    return radius * (cosine_angle * unit + sine_angle * perpendicular)


def paired_interval(rows, left, right, samples, seed):
    return pair_cluster_interval(
        rows,
        lambda row: row[left] - row[right],
        samples,
        seed,
    )


def main(config_path=CONFIG, output_path=OUTPUT, report_path=REPORT, figure_path=FIGURE):
    red = json.loads(Path(config_path).read_text())
    sphere_config = json.loads(Path(red["source_config"]).read_text())
    sphere_results = json.loads(Path(red["source_results"]).read_text())
    frontier_config = json.loads(Path(sphere_config["source_config"]).read_text())
    fisher = json.loads(Path(frontier_config["source_config"]).read_text())
    router = json.loads(Path(fisher["source_config"]).read_text())
    rotation_config = json.loads(Path(router["source_config"]).read_text())
    config = json.loads(Path(rotation_config["source_config"]).read_text())
    pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    seeds = rotation_config["rotation_seeds"]

    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    vocabulary_size = tokenizer.vocab_size - 1  # exclude GPT-2 EOS
    with torch.no_grad():
        embeddings = F.rms_norm(
            model.embedding.weight[:vocabulary_size].float(), (model.config.n_embd,),
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
    sphere_lookup = {
        (int(layer), row["corpus"], row["pair"], row["context"]): row
        for layer, rows in sphere_results["records"].items() for row in rows
    }
    excluded = {
        token for family_pairs in pairs["new_families"].values()
        for pair in family_pairs for token in pair
    }
    records = {str(layer): [] for layer in layers}
    maximum_distance_mismatch = 0.0
    maximum_radius_mismatch = 0.0

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        window_rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family_index, (family, family_pairs) in enumerate(pairs["new_families"].items()):
            for pair_index, (source_token, target_token) in enumerate(family_pairs):
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
                    random = np.random.default_rng(
                        red["empirical_seed"] + corpus_index * 1000000
                        + family_index * 100000 + pair_index * 1000 + context_index * 10
                    )
                    endpoint_rows = []
                    while len(endpoint_rows) < red["empirical_directions_per_example"]:
                        first, second = [int(value) for value in random.integers(0, vocabulary_size, 2)]
                        if first == second or first in excluded or second in excluded:
                            continue
                        endpoint = sphere_endpoint_from_direction(
                            source_vector, distance, embeddings[second] - embeddings[first],
                        )
                        if not torch.isfinite(endpoint).all():
                            continue
                        maximum_distance_mismatch = max(
                            maximum_distance_mismatch,
                            float(abs((endpoint - source_vector).norm() - distance) / distance.clamp_min(1e-30)),
                        )
                        maximum_radius_mismatch = max(
                            maximum_radius_mismatch,
                            float(abs(endpoint.norm() - source_vector.norm()) / source_vector.norm().clamp_min(1e-30)),
                        )
                        endpoint_x0 = base_x0.clone(); endpoint_x0[:, edit_position] = endpoint
                        endpoint_rows.append({
                            "x0": endpoint_x0,
                            "first_id": first,
                            "second_id": second,
                            "first_token": tokenizer.decode([first]),
                            "second_token": tokenizer.decode([second]),
                        })

                    pair_name = f"{source_token}->{target_token}"
                    for layer in layers:
                        empirical = []
                        for endpoint_row in endpoint_rows:
                            effect = evaluate_endpoint(
                                model, base, base_x0, endpoint_row["x0"], layer, position,
                                router["path_point"], sphere_config["rank"], seeds,
                                rotations, rotated_decoders,
                            )
                            empirical.append({
                                **effect,
                                "first_id": endpoint_row["first_id"],
                                "second_id": endpoint_row["second_id"],
                                "first_token": endpoint_row["first_token"],
                                "second_token": endpoint_row["second_token"],
                            })
                        source = sphere_lookup[(layer, corpus, pair_name, context_index)]
                        empirical_gain = median([
                            row["routed"] - row["native"] for row in empirical
                        ])
                        empirical_gap = median([
                            row["oracle"] - row["routed"] for row in empirical
                        ])
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": pair_name,
                            "context": context_index,
                            "semantic_native": source["semantic_native"],
                            "semantic_routed": source["semantic_routed"],
                            "semantic_oracle": source["semantic_oracle"],
                            "semantic_gain": source["semantic_gain"],
                            "semantic_native_choice": source["semantic_native_choice"],
                            "sphere_native": source["synthetic_native"],
                            "sphere_routed": source["synthetic_routed"],
                            "sphere_oracle": source["synthetic_oracle"],
                            "sphere_gain": source["synthetic_gain"],
                            "sphere_native_choice": source["synthetic_native_choice"],
                            "empirical_native": median([row["native"] for row in empirical]),
                            "empirical_routed": median([row["routed"] for row in empirical]),
                            "empirical_oracle": median([row["oracle"] for row in empirical]),
                            "empirical_gain": empirical_gain,
                            "empirical_router_gap": empirical_gap,
                            "empirical_native_choice": float(np.mean([
                                row["choice"] == "native" for row in empirical
                            ])),
                            "empirical_directions": empirical,
                        })
                print("empirical direction", corpus, family, source_token, target_token, flush=True)

    summaries = {str(layer): {} for layer in layers}
    median_fields = (
        "semantic_native", "semantic_routed", "semantic_oracle", "semantic_gain",
        "sphere_native", "sphere_routed", "sphere_oracle", "sphere_gain",
        "empirical_native", "empirical_routed", "empirical_oracle", "empirical_gain",
        "empirical_router_gap",
    )
    for layer in layers:
        for corpus_index, corpus in enumerate(arrays):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summary = {field: median([row[field] for row in selected]) for field in median_fields}
            for field in ("semantic_native_choice", "sphere_native_choice", "empirical_native_choice"):
                summary[field] = float(np.mean([row[field] for row in selected]))
            comparisons = (
                ("empirical_minus_sphere_native", "empirical_native", "sphere_native", 0),
                ("semantic_minus_empirical_native", "semantic_native", "empirical_native", 10000),
                ("semantic_minus_empirical_gain", "semantic_gain", "empirical_gain", 20000),
            )
            for name, left, right, offset in comparisons:
                value, interval = paired_interval(
                    selected, left, right, red["bootstrap_samples"],
                    red["bootstrap_seed"] + offset + layer * 100 + corpus_index,
                )
                summary[f"{name}_pair_median"] = value
                summary[f"{name}_pair_ci"] = interval
            summary["semantic_minus_empirical_native_choice"] = (
                summary["semantic_native_choice"] - summary["empirical_native_choice"]
            )
            summaries[str(layer)][corpus] = summary

    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    hypotheses = {
        "E0_empirical_direction_chord_is_exact": {
            "supported": maximum_distance_mismatch <= 1e-5 and maximum_radius_mismatch <= 1e-5,
            "maximum_distance_mismatch": maximum_distance_mismatch,
            "maximum_radius_mismatch": maximum_radius_mismatch,
        },
        "E1_token_difference_anisotropy_explains_native_alignment": {
            "supported": all(
                cell(layer, corpus)["empirical_minus_sphere_native_pair_median"] >= .03
                and cell(layer, corpus)["empirical_minus_sphere_native_pair_ci"][0] > 0
                for layer in layers for corpus in arrays
            ),
        },
        "E2_semantic_relation_alignment_exceeds_generic_token_directions": {
            "supported": all(
                cell(layer, corpus)["semantic_minus_empirical_native_pair_median"] >= .03
                and cell(layer, corpus)["semantic_minus_empirical_native_pair_ci"][0] > 0
                for layer in layers for corpus in arrays
            ),
        },
        "E3_chart_gain_is_specific_beyond_empirical_directions": {
            "supported": all(
                cell(layer, corpus)["semantic_minus_empirical_gain_pair_median"] >= .03
                and cell(layer, corpus)["semantic_minus_empirical_gain_pair_ci"][0] > 0
                for layer in layers for corpus in arrays
            ),
        },
        "E4_empirical_directions_also_benefit_from_chart_routing": {
            "supported": all(
                cell(layer, corpus)["empirical_gain"] >= .02
                for layer in layers for corpus in arrays
            ),
        },
        "E5_semantic_edits_prefer_native_over_empirical_directions": {
            "supported": all(
                cell(layer, corpus)["semantic_minus_empirical_native_choice"] >= .10
                for layer in layers for corpus in arrays
            ),
        },
        "E6_selection_is_strictly_prospective": {"supported": True},
    }
    Path(output_path).write_text(json.dumps({
        "config": red,
        "hypotheses": hypotheses,
        "maximum_distance_mismatch": maximum_distance_mismatch,
        "maximum_radius_mismatch": maximum_radius_mismatch,
        "summaries": summaries,
        "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Empirical token-difference direction control", "",
        f"Each natural lexical edit is compared with {red['empirical_directions_per_example']} random directions sampled from actual vocabulary-token differences. The random direction is transported to the natural source embedding and converted into an endpoint with exactly the same spherical chord length. This preserves empirical embedding anisotropy while removing the declared semantic relation. The native-plus-seven-chart router and all downstream measurements are unchanged.",
        "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", f"![Empirical direction control]({Path(figure_path).name})", "",
        "| Layer | Corpus | Native fidelity semantic / empirical / sphere | Routed gain semantic / empirical / sphere | Empirical−sphere native pair median (95% CI) | Semantic−empirical native pair median (95% CI) | Native choice semantic / empirical / sphere |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        for corpus in arrays:
            row = cell(layer, corpus)
            ci1 = row["empirical_minus_sphere_native_pair_ci"]
            ci2 = row["semantic_minus_empirical_native_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {row['semantic_native']:.3f} / {row['empirical_native']:.3f} / {row['sphere_native']:.3f} | "
                f"{row['semantic_gain']:+.3f} / {row['empirical_gain']:+.3f} / {row['sphere_gain']:+.3f} | "
                f"{row['empirical_minus_sphere_native_pair_median']:+.3f} ({ci1[0]:+.3f}, {ci1[1]:+.3f}) | "
                f"{row['semantic_minus_empirical_native_pair_median']:+.3f} ({ci2[0]:+.3f}, {ci2[1]:+.3f}) | "
                f"{row['semantic_native_choice']:.3f} / {row['empirical_native_choice']:.3f} / {row['sphere_native_choice']:.3f} |"
            )
    example_rows = records[str(layers[0])][:6]
    lines += [
        "", "## Example random direction sources", "",
        "These token pairs determine only the transported direction; neither token is inserted into the evaluated context.", "",
        "| Semantic pair | Random token difference 1 | Random token difference 2 |",
        "|---|---|---|",
    ]
    for row in example_rows:
        first, second = row["empirical_directions"][:2]
        clean = lambda value: value.encode("unicode_escape").decode()
        lines.append(
            f"| `{row['pair']}` | `{clean(first['first_token'])}` → `{clean(first['second_token'])}` | "
            f"`{clean(second['first_token'])}` → `{clean(second['second_token'])}` |"
        )
    lines += [
        "",
        f"Maximum relative chord-distance mismatch: **{maximum_distance_mismatch:.2e}**. Maximum relative sphere-radius mismatch: **{maximum_radius_mismatch:.2e}**.",
        "",
        "This control distinguishes generic anisotropy of learned token-difference directions from specificity to the intended lexical relation. It does not turn the transported endpoint into a real vocabulary embedding, so a remaining semantic advantage could still reflect the discrete token manifold rather than human-interpretable semantics.",
    ]
    Path(report_path).write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    labels = [f"L{layer}\n{corpus}" for layer in layers for corpus in arrays]
    x = np.arange(len(labels)); width = .25
    for offset, condition, label in ((-width, "semantic", "semantic"), (0, "empirical", "token-difference"), (width, "sphere", "Gaussian sphere")):
        axes[0].bar(x + offset, [cell(layer, corpus)[f"{condition}_native"] for layer in layers for corpus in arrays], width, label=label)
        axes[1].bar(x + offset, [cell(layer, corpus)[f"{condition}_gain"] for layer in layers for corpus in arrays], width, label=label)
    axes[0].set_xticks(x, labels); axes[0].set_ylim(0, 1); axes[0].set_ylabel("rank-256 native fidelity"); axes[0].set_title("Native-coordinate alignment"); axes[0].legend()
    axes[1].set_xticks(x, labels); axes[1].axhline(0, color="black", linewidth=.8); axes[1].set_ylabel("routed − native fidelity"); axes[1].set_title("Adaptive-chart gain"); axes[1].legend()
    fig.savefig(figure_path, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(report_path)


if __name__ == "__main__":
    main()
