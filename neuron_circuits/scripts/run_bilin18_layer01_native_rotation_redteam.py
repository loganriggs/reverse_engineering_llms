#!/usr/bin/env python3
"""LOPO, rotation-seed, and decoder-norm red-team for Layer-0/1 axis privilege."""

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


CONFIG = Path("configs/bilin18_layer01_native_rotation_redteam.json")
OUTPUT = TENSOR_ROOT / "layer01_native_rotation_redteam.json"
REPORT = TENSOR_ROOT / "LAYER01_NATIVE_ROTATION_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer01_native_rotation_redteam.png"


def participation_ratio(vector: torch.Tensor) -> float:
    squared = vector.double().square()
    return float(squared.sum().square() / squared.square().sum().clamp_min(1e-30))


def normalized_score(score: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
    return score / decoder.norm(dim=0).cpu().clamp_min(1e-12)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tournament_config = json.loads(Path(config["source_config"]).read_text())
    source_config = json.loads(Path(tournament_config["source_config"]).read_text())
    fixed, parent = recover_configs(source_config)
    saved = json.loads(Path(config["source_results"]).read_text())
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    ranks = config["ranks"]
    family_names = list(parent["families"])
    wrong_family = {
        family: family_names[(index + 1) % len(family_names)]
        for index, family in enumerate(family_names)
    }
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
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )
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
    results = {}
    mixing = {}

    for layer in config["layers"]:
        block = model.blocks[layer]
        native_decoder = block.mlp.Down.weight.float()
        dimension = native_decoder.shape[1]
        rotations = {
            str(seed): make_rotation(dimension, config["rotation_rounds"], seed, "cuda")
            for seed in config["rotation_seeds"]
        }
        rotated_decoders = {
            seed: rotate(native_decoder, permutations)
            for seed, permutations in rotations.items()
        }
        mixing[str(layer)] = {}
        for seed, permutations in rotations.items():
            indices = torch.linspace(0, dimension - 1, 64, device="cuda").long()
            unit = torch.zeros(64, dimension, device="cuda")
            unit[torch.arange(64, device="cuda"), indices] = 1
            from run_bilin18_layer01_native_rotation_tournament import inverse_rotate
            axes = inverse_rotate(unit, permutations)
            prs = [participation_ratio(axis) for axis in axes]
            mixing[str(layer)][seed] = {
                "median_participation_ratio": float(np.median(prs)),
                "minimum_participation_ratio": float(np.min(prs)),
                "median_max_abs": float(torch.max(axes.abs(), dim=1).values.median()),
            }

        native_by_pair = {}
        rotated_by_pair = {seed: {} for seed in rotations}
        family_by_pair = {}
        for record in discovery_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            pair = record["pair"]
            family_by_pair[pair] = record["family"]
            native_by_pair[pair] = coordinate_scores(
                state["delta_product"], native_decoder, state["target_write"], final_position,
            ).abs().cpu()
            for seed, permutations in rotations.items():
                coordinates = rotate(state["delta_product"], permutations)
                rotated_by_pair[seed][pair] = coordinate_scores(
                    coordinates, rotated_decoders[seed], state["target_write"], final_position,
                ).abs().cpu()
            print("red discovery", layer, record["family"], pair, flush=True)

        def average(scores, pairs):
            return torch.stack([scores[pair] for pair in pairs]).mean(0)

        all_pairs = list(native_by_pair)
        cache = {}
        for pair in all_pairs:
            family = family_by_pair[pair]
            same = [p for p in all_pairs if family_by_pair[p] == family]
            lopo = [p for p in same if p != pair]
            global_lopo = [p for p in all_pairs if p != pair]
            wrong = [p for p in all_pairs if family_by_pair[p] == wrong_family[family]]
            native_rows = {
                "inclusive": average(native_by_pair, same),
                "lopo": average(native_by_pair, lopo),
                "global": average(native_by_pair, global_lopo),
                "wrong": average(native_by_pair, wrong),
            }
            cache[pair] = {"native": {}, "rotated": {}}
            for rank in ranks:
                cache[pair]["native"][rank] = {
                    name: top_indices(score, rank).cuda() for name, score in native_rows.items()
                }
                cache[pair]["native"][rank]["lopo_normed"] = top_indices(
                    normalized_score(native_rows["lopo"], native_decoder), rank,
                ).cuda()
                cache[pair]["native"][rank]["norm"] = top_indices(
                    native_decoder.norm(dim=0).cpu(), rank,
                ).cuda()
            for seed in rotations:
                scores = rotated_by_pair[seed]
                rows = {
                    "inclusive": average(scores, same),
                    "lopo": average(scores, lopo),
                    "global": average(scores, global_lopo),
                    "wrong": average(scores, wrong),
                }
                cache[pair]["rotated"][seed] = {}
                for rank in ranks:
                    cache[pair]["rotated"][seed][rank] = {
                        name: top_indices(score, rank).cuda() for name, score in rows.items()
                    }
                    cache[pair]["rotated"][seed][rank]["lopo_normed"] = top_indices(
                        normalized_score(rows["lopo"], rotated_decoders[seed]), rank,
                    ).cuda()
                    cache[pair]["rotated"][seed][rank]["norm"] = top_indices(
                        rotated_decoders[seed].norm(dim=0).cpu(), rank,
                    ).cuda()

        records = []
        route_names = []
        for record in query_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            pair_cache = cache[record["pair"]]
            deltas = {}
            for rank in ranks:
                for name, indices in pair_cache["native"][rank].items():
                    deltas[f"native_{name}_{rank}"] = selected_native(
                        state["delta_product"], indices,
                    )
                for seed, permutations in rotations.items():
                    for name, indices in pair_cache["rotated"][seed][rank].items():
                        deltas[f"rot{seed}_{name}_{rank}"] = selected_rotated(
                            state["delta_product"], indices, permutations,
                        )
            if not route_names:
                route_names = list(deltas)
            routes = evaluate_deltas(
                model, layer, base, counterfactual, state, deltas, final_position,
            )
            records.append({
                "corpus": record["corpus"], "family": record["family"],
                "pair": record["pair"], "routes": routes,
            })
            print("red evaluation", layer, record["corpus"], record["family"], record["pair"], flush=True)
        summaries = {
            corpus: summarize(
                [row for row in records if row["corpus"] == corpus], route_names,
            )
            for corpus in fixed["evaluation"]
        }
        results[str(layer)] = {"summaries": summaries, "records": records}

    def value(layer, corpus, route):
        return results[str(layer)]["summaries"][corpus][route]["downstream_fidelity"]

    def rotation_values(layer, corpus, name, rank):
        return [
            value(layer, corpus, f"rot{seed}_{name}_{rank}")
            for seed in map(str, config["rotation_seeds"])
        ]

    def rotation_median(layer, corpus, name, rank):
        return median(rotation_values(layer, corpus, name, rank))

    hypotheses = {
        "R0_rotations_are_dense": {
            "supported": all(
                row["median_participation_ratio"] >= 512
                for layer in mixing.values() for row in layer.values()
            ),
            "minimum_median_participation_ratio": min(
                row["median_participation_ratio"]
                for layer in mixing.values() for row in layer.values()
            ),
        },
        "R1_native_lopo_banks_are_family_specific": {
            "supported": all(
                value(layer, corpus, "native_lopo_256")
                - value(layer, corpus, "native_wrong_256") >= 0.03
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "R2_native_lopo_beats_global": {
            "supported": all(
                value(layer, corpus, "native_lopo_256")
                - value(layer, corpus, "native_global_256") >= 0.03
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "R3_native_axis_advantage_survives_lopo": {
            "supported": all(
                value(layer, corpus, "native_lopo_256")
                - rotation_median(layer, corpus, "lopo", 256) >= 0.05
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "R4_axis_advantage_survives_decoder_norm_control": {
            "supported": all(
                value(layer, corpus, "native_lopo_normed_256")
                - rotation_median(layer, corpus, "lopo_normed", 256) >= 0.03
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "R5_weight_norm_alone_is_insufficient": {
            "supported": all(
                value(layer, corpus, "native_lopo_256")
                - value(layer, corpus, "native_norm_256") >= 0.05
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "R6_same_pair_inclusion_materially_helps": {
            "supported": all(any(
                value(layer, corpus, "native_inclusive_256")
                - value(layer, corpus, "native_lopo_256") >= 0.05
                for corpus in fixed["evaluation"]
            ) for layer in config["layers"]),
        },
        "R7_rotation_null_is_stable": {
            "supported": all(
                max(rotation_values(layer, corpus, "lopo", 256))
                - min(rotation_values(layer, corpus, "lopo", 256)) <= 0.10
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
            "ranges": {
                f"{layer}_{corpus}": max(rotation_values(layer, corpus, "lopo", 256))
                - min(rotation_values(layer, corpus, "lopo", 256))
                for layer in config["layers"] for corpus in fixed["evaluation"]
            },
        },
        "R8_layer1_has_more_reusable_native_family_structure": {
            "supported": all(
                value(1, corpus, "native_lopo_256")
                - value(0, corpus, "native_lopo_256") >= 0.05
                for corpus in fixed["evaluation"]
            ),
        },
        "R9_native_lopo_transfers_ood": {
            "supported": all(
                value(layer, "tinystories", "native_lopo_256")
                >= 0.75 * value(layer, "fineweb", "native_lopo_256")
                for layer in config["layers"]
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses, "mixing": mixing,
        "wrong_family_mapping": wrong_family, "layers": results,
    }, indent=2))

    lines = [
        "# Layer-0/1 native-axis red-team: leave-one-pair-out and norm controls",
        "",
        "The first tournament allowed each family bank to see the evaluated lexical pair in a different context. This stricter run excludes that pair, expands the null to eight dense orthogonal rotations, and adds decoder-norm-only and decoder-norm-normalized selections. All banks remain frozen before held-out FineWeb and TinyStories evaluation through the nonlinear suffix.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "![LOPO native-axis red-team](layer01_native_rotation_redteam.png)", "",
        "| Layer | Corpus | Rank | Native inclusive | Native LOPO | Rotated LOPO | Native LOPO normed | Rotated LOPO normed | Native global | Native wrong | Native norm-only |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            for rank in ranks:
                lines.append(
                    f"| {layer} | {corpus} | {rank} | "
                    f"{value(layer, corpus, f'native_inclusive_{rank}'):.3f} | "
                    f"{value(layer, corpus, f'native_lopo_{rank}'):.3f} | "
                    f"{rotation_median(layer, corpus, 'lopo', rank):.3f} | "
                    f"{value(layer, corpus, f'native_lopo_normed_{rank}'):.3f} | "
                    f"{rotation_median(layer, corpus, 'lopo_normed', rank):.3f} | "
                    f"{value(layer, corpus, f'native_global_{rank}'):.3f} | "
                    f"{value(layer, corpus, f'native_wrong_{rank}'):.3f} | "
                    f"{value(layer, corpus, f'native_norm_{rank}'):.3f} |"
                )
    lines += [
        "",
        "LOPO success is evidence for a family-reusable address set; inclusive-only success is merely recurrence of the same lexical edit. A native-over-rotation result after decoder-norm normalization supports architectural coordinate privilege beyond a few unusually large Down columns. It still does not imply one neuron is a complete semantic circuit.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, layer in enumerate(config["layers"]):
        for column, corpus in enumerate(("fineweb", "tinystories")):
            axis = axes[row, column]
            axis.plot(ranks, [value(layer, corpus, f"native_inclusive_{r}") for r in ranks], marker="o", label="native inclusive")
            axis.plot(ranks, [value(layer, corpus, f"native_lopo_{r}") for r in ranks], marker="o", label="native LOPO")
            axis.plot(ranks, [rotation_median(layer, corpus, "lopo", r) for r in ranks], marker="o", label="rotated LOPO")
            axis.plot(ranks, [value(layer, corpus, f"native_norm_{r}") for r in ranks], marker="o", label="norm-only")
            axis.set_title(f"Layer {layer} — {corpus}")
            axis.set_xlabel("coordinate budget")
            axis.set_ylabel("Fisher fidelity to full MLP patch")
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps(hypotheses, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
