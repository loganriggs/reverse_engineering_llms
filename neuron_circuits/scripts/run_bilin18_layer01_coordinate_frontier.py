#!/usr/bin/env python3
"""Native and rotated coordinate sufficiency frontiers for Layer-0/1 semantic edits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer01_native_rotation_tournament import (
    coordinate_scores,
    evaluate_deltas,
    make_rotation,
    rotate,
    selected_native,
    selected_rotated,
    summarize,
)
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_coordinate_frontier.json")
OUTPUT = TENSOR_ROOT / "layer01_coordinate_frontier.json"
REPORT = TENSOR_ROOT / "LAYER01_COORDINATE_FRONTIER.md"
FIGURE = TENSOR_ROOT / "layer01_coordinate_frontier.png"


def threshold_rank(summary: dict, prefix: str, ranks: list[int], threshold: float) -> int | None:
    for rank in ranks:
        if summary[f"{prefix}_{rank}"]["downstream_fidelity"] >= threshold:
            return rank
    return None


def format_threshold_range(values: list[int | None]) -> str:
    finite = [value for value in values if value is not None]
    if not finite:
        return "not reached"
    if len(finite) != len(values):
        return f"{min(finite)}–not reached"
    return str(finite[0]) if min(finite) == max(finite) else f"{min(finite)}–{max(finite)}"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    red_config = json.loads(Path(config["source_config"]).read_text())
    tournament_config = json.loads(Path(red_config["source_config"]).read_text())
    source_config = json.loads(Path(tournament_config["source_config"]).read_text())
    fixed, parent = recover_configs(source_config)
    saved = json.loads(Path(config["source_results"]).read_text())
    ranks = config["ranks"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }
    specs = fixed["evaluation"]

    def reconstruct(record, layer):
        corpus = record["corpus"]
        spec = specs[corpus]
        rows = arrays[corpus][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(
            rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[edit_position] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counterfactual = forward_layers(model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer)
        return base[layer], counterfactual[layer]

    discovery_saved = [
        row for row in saved["semantic_records"]
        if row["corpus"] == config["discovery_corpus"]
        and row["context_index"] == config["discovery_context_index"]
    ]
    query_saved = [
        row for row in saved["semantic_records"]
        if row["context_index"] == config["evaluation_context_index"]
    ]
    family_by_pair = {row["pair"]: row["family"] for row in discovery_saved}
    results = {}

    for layer in config["layers"]:
        decoder = model.blocks[layer].mlp.Down.weight.float()
        dimension = decoder.shape[1]
        rotations = {
            str(seed): make_rotation(dimension, config["rotation_rounds"], seed, "cuda")
            for seed in config["rotation_seeds"]
        }
        rotated_decoders = {seed: rotate(decoder, spec) for seed, spec in rotations.items()}
        native_by_pair = {}
        rotated_by_pair = {seed: {} for seed in rotations}
        for record in discovery_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            native_by_pair[record["pair"]] = coordinate_scores(
                state["delta_product"], decoder, state["target_write"], final_position,
            ).abs().cpu()
            for seed, permutations in rotations.items():
                coordinates = rotate(state["delta_product"], permutations)
                rotated_by_pair[seed][record["pair"]] = coordinate_scores(
                    coordinates, rotated_decoders[seed], state["target_write"], final_position,
                ).abs().cpu()

        def lopo_score(scores, pair):
            family = family_by_pair[pair]
            members = [
                candidate for candidate in scores
                if candidate != pair and family_by_pair[candidate] == family
            ]
            return torch.stack([scores[candidate] for candidate in members]).mean(0)

        native_lopo = {
            pair: {rank: top_indices(lopo_score(native_by_pair, pair), rank).cuda() for rank in ranks}
            for pair in native_by_pair
        }
        rotated_lopo = {
            seed: {
                pair: {rank: top_indices(lopo_score(scores, pair), rank).cuda() for rank in ranks}
                for pair in scores
            }
            for seed, scores in rotated_by_pair.items()
        }
        records = []
        route_names = []
        for record in query_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            native_score = coordinate_scores(
                state["delta_product"], decoder, state["target_write"], final_position,
            ).abs()
            deltas = {}
            for rank in ranks:
                deltas[f"native_oracle_{rank}"] = selected_native(
                    state["delta_product"], top_indices(native_score, rank),
                )
                deltas[f"native_lopo_{rank}"] = selected_native(
                    state["delta_product"], native_lopo[record["pair"]][rank],
                )
                for seed, permutations in rotations.items():
                    coordinates = rotate(state["delta_product"], permutations)
                    rotated_score = coordinate_scores(
                        coordinates, rotated_decoders[seed], state["target_write"], final_position,
                    ).abs()
                    deltas[f"rot{seed}_oracle_{rank}"] = selected_rotated(
                        state["delta_product"], top_indices(rotated_score, rank), permutations,
                    )
                    deltas[f"rot{seed}_lopo_{rank}"] = selected_rotated(
                        state["delta_product"], rotated_lopo[seed][record["pair"]][rank], permutations,
                    )
            if not route_names:
                route_names = list(deltas)
            route_rows = evaluate_deltas(
                model, layer, base, counterfactual, state, deltas, final_position,
            )
            records.append({
                "corpus": record["corpus"], "family": record["family"],
                "pair": record["pair"], "routes": route_rows,
            })
            print("frontier", layer, record["corpus"], record["family"], record["pair"], flush=True)
        summaries = {
            corpus: summarize([row for row in records if row["corpus"] == corpus], route_names)
            for corpus in fixed["evaluation"]
        }
        thresholds = {}
        for corpus, summary in summaries.items():
            thresholds[corpus] = {
                "native_oracle_s50": threshold_rank(summary, "native_oracle", ranks, .5),
                "native_oracle_s90": threshold_rank(summary, "native_oracle", ranks, .9),
                "native_lopo_s50": threshold_rank(summary, "native_lopo", ranks, .5),
                "native_lopo_s90": threshold_rank(summary, "native_lopo", ranks, .9),
            }
            for seed in rotations:
                thresholds[corpus][f"rot{seed}_oracle_s50"] = threshold_rank(summary, f"rot{seed}_oracle", ranks, .5)
                thresholds[corpus][f"rot{seed}_oracle_s90"] = threshold_rank(summary, f"rot{seed}_oracle", ranks, .9)
                thresholds[corpus][f"rot{seed}_lopo_s50"] = threshold_rank(summary, f"rot{seed}_lopo", ranks, .5)
                thresholds[corpus][f"rot{seed}_lopo_s90"] = threshold_rank(summary, f"rot{seed}_lopo", ranks, .9)
        results[str(layer)] = {"summaries": summaries, "thresholds": thresholds, "records": records}

    def value(layer, corpus, route):
        return results[str(layer)]["summaries"][corpus][route]["downstream_fidelity"]

    def rot_median(layer, corpus, kind, rank):
        return median([
            value(layer, corpus, f"rot{seed}_{kind}_{rank}")
            for seed in map(str, config["rotation_seeds"])
        ])

    hypotheses = {
        "F0_full_coordinate_positive_control": {
            "supported": all(
                value(layer, corpus, "native_lopo_4608") >= .999
                and rot_median(layer, corpus, "lopo", 4608) >= .999
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "F1_no_tiny_family_circuit": {
            "supported": all(value(layer, corpus, "native_lopo_32") < .25 for layer in config["layers"] for corpus in fixed["evaluation"]),
        },
        "F2_individual_edits_have_moderate_native_sparsity": {
            "supported": all(value(layer, corpus, "native_oracle_256") >= .5 for layer in config["layers"] for corpus in fixed["evaluation"]),
        },
        "F3_reusable_family_banks_reach_half_fidelity": {
            "supported": all(value(layer, corpus, "native_lopo_1024") >= .5 for layer in config["layers"] for corpus in fixed["evaluation"]),
        },
        "F4_high_fidelity_requires_broad_native_banks": {
            "supported": all(
                all(value(layer, corpus, f"native_lopo_{rank}") < .9 for rank in ranks if rank < 2048)
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "F5_native_advantage_persists_at_512": {
            "supported": all(
                value(layer, corpus, "native_lopo_512") - rot_median(layer, corpus, "lopo", 512) >= .05
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "F6_rotation_converges_only_at_broad_rank": {
            "supported": all(
                all(rot_median(layer, corpus, "lopo", rank) < .5 for rank in ranks if rank < 2048)
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "F7_layer1_family_code_is_more_compact": {
            "supported": all(
                value(1, corpus, "native_lopo_512") - value(0, corpus, "native_lopo_512") >= .03
                for corpus in fixed["evaluation"]
            ),
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "layers": results}, indent=2))

    lines = [
        "# Layer-0/1 native and rotated coordinate sufficiency frontiers", "",
        "Per-example oracle and leave-one-pair-out semantic-family coordinate banks are swept from one coordinate to the complete 4,608-channel layer. Every partial product change is propagated through the nonlinear suffix and scored against the complete MLP-only patch in the native softmax-Fisher metric.", "",
        "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "![Coordinate frontiers](layer01_coordinate_frontier.png)", "",
        "| Layer | Corpus | Native oracle S50/S90 | Native LOPO S50/S90 | Rotated oracle S50/S90 | Rotated LOPO S50/S90 |", "|---:|---|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            threshold = results[str(layer)]["thresholds"][corpus]
            rot_oracle50 = format_threshold_range([threshold[f"rot{s}_oracle_s50"] for s in map(str, config["rotation_seeds"])])
            rot_oracle90 = format_threshold_range([threshold[f"rot{s}_oracle_s90"] for s in map(str, config["rotation_seeds"])])
            rot_lopo50 = format_threshold_range([threshold[f"rot{s}_lopo_s50"] for s in map(str, config["rotation_seeds"])])
            rot_lopo90 = format_threshold_range([threshold[f"rot{s}_lopo_s90"] for s in map(str, config["rotation_seeds"])])
            lines.append(
                f"| {layer} | {corpus} | {threshold['native_oracle_s50']} / {threshold['native_oracle_s90']} | "
                f"{threshold['native_lopo_s50']} / {threshold['native_lopo_s90']} | "
                f"{rot_oracle50} / {rot_oracle90} | {rot_lopo50} / {rot_lopo90} |"
            )
    lines += ["", "S50/S90 are the first tested coordinate budgets reaching 0.50/0.90 median Fisher fidelity. They describe the selected family of semantic edits, not the number of neurons required for all natural language behavior. A native advantage establishes basis privilege; a large native S90 establishes that basis privilege is compatible with a distributed feature."]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, layer in enumerate(config["layers"]):
        for column, corpus in enumerate(("fineweb", "tinystories")):
            axis = axes[row, column]
            axis.plot(ranks, [value(layer, corpus, f"native_oracle_{rank}") for rank in ranks], marker="o", label="native oracle")
            axis.plot(ranks, [value(layer, corpus, f"native_lopo_{rank}") for rank in ranks], marker="o", label="native LOPO")
            axis.plot(ranks, [rot_median(layer, corpus, "oracle", rank) for rank in ranks], marker=".", label="rotated oracle")
            axis.plot(ranks, [rot_median(layer, corpus, "lopo", rank) for rank in ranks], marker=".", label="rotated LOPO")
            axis.axhline(.9, color="black", linewidth=.8, linestyle=":")
            axis.axhline(.5, color="black", linewidth=.8, linestyle="--")
            axis.set_xscale("log", base=2)
            axis.set_ylim(-.05, 1.05)
            axis.set_title(f"Layer {layer} — {corpus}")
            axis.set_xlabel("coordinates retained")
            axis.set_ylabel("Fisher fidelity")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps(hypotheses, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
