#!/usr/bin/env python3
"""Causal native-axis versus orthogonal-rotation tournament for MLP layers 0 and 1."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer3_frozen_channel_transfer import (
    channel_state,
    recover_configs,
    top_indices,
)
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_layer8_semantic_path_decomposition import fisher_inner
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer01_native_rotation_tournament.json")
OUTPUT = TENSOR_ROOT / "layer01_native_rotation_tournament.json"
REPORT = TENSOR_ROOT / "LAYER01_NATIVE_ROTATION_TOURNAMENT.md"
FIGURE = TENSOR_ROOT / "layer01_native_rotation_tournament.png"


def make_rotation(dimension: int, rounds: int, seed: int, device: str) -> list[torch.Tensor]:
    """Build an implicit dense orthogonal transform from shuffled Hadamard pairs."""
    if dimension % 2:
        raise ValueError("rotation dimension must be even")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randperm(dimension, generator=generator).to(device) for _ in range(rounds)]


def rotate(values: torch.Tensor, permutations: list[torch.Tensor]) -> torch.Tensor:
    result = values
    scale = 1 / math.sqrt(2)
    for permutation in permutations:
        result = result[..., permutation]
        paired = result.reshape(*result.shape[:-1], -1, 2)
        left, right = paired[..., 0], paired[..., 1]
        result = torch.stack(((left + right) * scale, (left - right) * scale), dim=-1)
        result = result.reshape(*values.shape)
    return result


def inverse_rotate(values: torch.Tensor, permutations: list[torch.Tensor]) -> torch.Tensor:
    result = values
    scale = 1 / math.sqrt(2)
    for permutation in reversed(permutations):
        paired = result.reshape(*result.shape[:-1], -1, 2)
        left, right = paired[..., 0], paired[..., 1]
        unpaired = torch.stack(((left + right) * scale, (left - right) * scale), dim=-1)
        unpaired = unpaired.reshape(*values.shape)
        restored = torch.empty_like(unpaired)
        restored[..., permutation] = unpaired
        result = restored
    return result


def coordinate_scores(
    coordinates: torch.Tensor,
    decoder: torch.Tensor,
    target_write: torch.Tensor,
    final_position: int,
) -> torch.Tensor:
    target = target_write[0, : final_position + 1]
    denominator = target.square().sum().clamp_min(1e-30)
    return (
        coordinates[0, : final_position + 1]
        * (target @ decoder)
    ).sum(0) / denominator


def selected_native(delta: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    selected = torch.zeros_like(delta)
    selected[..., indices] = delta[..., indices]
    return selected


def selected_rotated(
    delta: torch.Tensor,
    indices: torch.Tensor,
    permutations: list[torch.Tensor],
) -> torch.Tensor:
    coordinates = rotate(delta, permutations)
    selected = torch.zeros_like(coordinates)
    selected[..., indices] = coordinates[..., indices]
    return inverse_rotate(selected, permutations)


@torch.inference_mode()
def evaluate_deltas(model, layer, base, counterfactual, state, deltas, final_position):
    block = model.blocks[layer]
    names = list(deltas)
    target_write = state["target_write"]
    local = {}
    products = []
    for name in names:
        predicted = F.linear(deltas[name], block.mlp.Down.weight.float(), bias=None)
        local[name] = 1 - float(
            (predicted - target_write).square().sum()
            / target_write.square().sum().clamp_min(1e-30)
        )
        products.append(state["base_product"] + deltas[name].to(state["base_product"].dtype))
    products.append(state["counterfactual_product"])

    outputs = [base["output"], counterfactual["output"]]
    x0s = [base["x0"], counterfactual["x0"]]
    buses = [base["value_bus"], counterfactual["value_bus"]]
    for product in products:
        outputs.append(base["pre"] + block.mlp.Down(product) + block.mlp.Down_bias)
        x0s.append(base["x0"])
        buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), final_position,
    )
    base_logits, counterfactual_logits = logits[0], logits[1]
    semantic_target = counterfactual_logits - base_logits
    target_energy = fisher_inner(base_logits, semantic_target, semantic_target)
    full_mlp_patch = logits[-1] - base_logits
    full_mlp_energy = fisher_inner(base_logits, full_mlp_patch, full_mlp_patch)
    rows = {}
    for offset, name in enumerate(names):
        effect = logits[2 + offset] - base_logits
        fidelity, cosine, _ = fisher_fidelity(base_logits, full_mlp_patch, effect)
        rows[name] = {
            "local_fidelity": local[name],
            "downstream_fidelity": fidelity,
            "downstream_cosine": cosine,
            "semantic_target_projection": fisher_inner(base_logits, effect, semantic_target)
            / max(target_energy, 1e-30),
            "energy_ratio": fisher_inner(base_logits, effect, effect)
            / max(full_mlp_energy, 1e-30),
        }
    return rows


def summarize(records, route_names):
    return {
        name: {
            metric: median([record["routes"][name][metric] for record in records])
            for metric in (
                "local_fidelity", "downstream_fidelity", "downstream_cosine",
                "semantic_target_projection", "energy_ratio",
            )
        }
        for name in route_names
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    fixed, parent = recover_configs(source_config)
    saved = json.loads(Path(config["source_results"]).read_text())
    ranks = config["ranks"]
    maximum_rank = max(ranks)
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
    family_names = list(parent["families"])
    wrong_family = {
        family: family_names[(index + 1) % len(family_names)]
        for index, family in enumerate(family_names)
    }

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
        record for record in saved["semantic_records"]
        if record["corpus"] == config["discovery_corpus"]
        and record["context_index"] == config["discovery_context_index"]
    ]
    query_saved = [
        record for record in saved["semantic_records"]
        if record["context_index"] == config["evaluation_context_index"]
    ]
    all_results = {}
    max_reconstruction_error = 0.0

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
        family_native_scores = {family: [] for family in family_names}
        family_rotated_scores = {
            seed: {family: [] for family in family_names} for seed in rotations
        }
        global_native_scores = []
        global_rotated_scores = {seed: [] for seed in rotations}

        for record in discovery_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            native_score = coordinate_scores(
                state["delta_product"], native_decoder, state["target_write"], final_position,
            ).abs().cpu()
            family_native_scores[record["family"]].append(native_score)
            global_native_scores.append(native_score)
            for seed, permutations in rotations.items():
                coordinates = rotate(state["delta_product"], permutations)
                rotated_score = coordinate_scores(
                    coordinates, rotated_decoders[seed], state["target_write"], final_position,
                ).abs().cpu()
                family_rotated_scores[seed][record["family"]].append(rotated_score)
                global_rotated_scores[seed].append(rotated_score)
                reconstructed = inverse_rotate(coordinates, permutations)
                error = float(
                    (reconstructed - state["delta_product"]).norm()
                    / state["delta_product"].norm().clamp_min(1e-30)
                )
                write_error = float(
                    (F.linear(coordinates, rotated_decoders[seed]) - state["target_write"]).norm()
                    / state["target_write"].norm().clamp_min(1e-30)
                )
                max_reconstruction_error = max(max_reconstruction_error, error, write_error)
            print("discovery", layer, record["family"], record["pair"], flush=True)

        native_vectors = {
            family: torch.stack(rows).mean(0) for family, rows in family_native_scores.items()
        }
        native_sets = {
            family: {rank: top_indices(score, rank).cuda() for rank in ranks}
            for family, score in native_vectors.items()
        }
        native_global = {
            rank: top_indices(torch.stack(global_native_scores).mean(0), rank).cuda()
            for rank in ranks
        }
        rotated_vectors = {
            seed: {
                family: torch.stack(rows).mean(0)
                for family, rows in families.items()
            }
            for seed, families in family_rotated_scores.items()
        }
        rotated_sets = {
            seed: {
                family: {rank: top_indices(score, rank).cuda() for rank in ranks}
                for family, score in families.items()
            }
            for seed, families in rotated_vectors.items()
        }
        rotated_global = {
            seed: {
                rank: top_indices(torch.stack(global_rotated_scores[seed]).mean(0), rank).cuda()
                for rank in ranks
            }
            for seed in rotations
        }
        generator = torch.Generator().manual_seed(config["selection_seed"] + layer)
        native_random_order = torch.randperm(dimension, generator=generator)
        rotated_random_order = {
            seed: torch.randperm(
                dimension,
                generator=torch.Generator().manual_seed(config["selection_seed"] + layer + int(seed)),
            )
            for seed in rotations
        }

        records = []
        route_names = []
        for record in query_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            native_score = coordinate_scores(
                state["delta_product"], native_decoder, state["target_write"], final_position,
            ).abs()
            deltas = {}
            for rank in ranks:
                for label, indices in (
                    ("oracle", top_indices(native_score, rank)),
                    ("family", native_sets[record["family"]][rank]),
                    ("global", native_global[rank]),
                    ("random", native_random_order[:rank].cuda()),
                ):
                    deltas[f"native_{label}_{rank}"] = selected_native(
                        state["delta_product"], indices,
                    )
                deltas[f"native_wrong_{rank}"] = selected_native(
                    state["delta_product"], native_sets[wrong_family[record["family"]]][rank],
                )
            for seed, permutations in rotations.items():
                coordinates = rotate(state["delta_product"], permutations)
                score = coordinate_scores(
                    coordinates, rotated_decoders[seed], state["target_write"], final_position,
                ).abs()
                for rank in ranks:
                    for label, indices in (
                        ("oracle", top_indices(score, rank)),
                        ("family", rotated_sets[seed][record["family"]][rank]),
                        ("global", rotated_global[seed][rank]),
                        ("random", rotated_random_order[seed][:rank].cuda()),
                    ):
                        deltas[f"rot{seed}_{label}_{rank}"] = selected_rotated(
                            state["delta_product"], indices, permutations,
                        )
                    deltas[f"rot{seed}_wrong_{rank}"] = selected_rotated(
                        state["delta_product"],
                        rotated_sets[seed][wrong_family[record["family"]]][rank],
                        permutations,
                    )
            if not route_names:
                route_names = list(deltas)
            routes = evaluate_deltas(
                model, layer, base, counterfactual, state, deltas, final_position,
            )
            records.append({
                "corpus": record["corpus"],
                "family": record["family"],
                "pair": record["pair"],
                "routes": routes,
            })
            print("evaluation", layer, record["corpus"], record["family"], record["pair"], flush=True)

        summaries = {
            corpus: summarize(
                [record for record in records if record["corpus"] == corpus], route_names,
            )
            for corpus in fixed["evaluation"]
        }
        all_results[str(layer)] = {
            "summaries": summaries,
            "records": records,
            "route_names": route_names,
        }

    def value(layer, corpus, route):
        return all_results[str(layer)]["summaries"][corpus][route]["downstream_fidelity"]

    def rotation_median(layer, corpus, label, rank):
        return median([
            value(layer, corpus, f"rot{seed}_{label}_{rank}")
            for seed in map(str, config["rotation_seeds"])
        ])

    advantages = {
        str(layer): {
            corpus: value(layer, corpus, "native_family_128")
            - rotation_median(layer, corpus, "family", 128)
            for corpus in fixed["evaluation"]
        }
        for layer in config["layers"]
    }
    hypotheses = {
        "T0_rotation_is_an_exact_change_of_basis": {
            "supported": max_reconstruction_error <= 2e-5,
            "maximum_relative_error": max_reconstruction_error,
        },
        "T1_layer0_native_family_axes_are_privileged": {
            "supported": all(advantages["0"][c] >= 0.05 for c in fixed["evaluation"]),
            "advantages": advantages["0"],
        },
        "T2_native_privilege_weakens_by_layer1": {
            "supported": all(
                advantages["0"][c] - advantages["1"][c] >= 0.03
                for c in fixed["evaluation"]
            ),
            "layer0_minus_layer1": {
                c: advantages["0"][c] - advantages["1"][c] for c in fixed["evaluation"]
            },
        },
        "T3_native_family_banks_are_specific": {
            "supported": all(
                value(layer, corpus, "native_family_256")
                - value(layer, corpus, "native_wrong_256") >= 0.05
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "T4_rotated_family_banks_retain_specificity": {
            "supported": all(
                rotation_median(layer, corpus, "family", 256)
                - rotation_median(layer, corpus, "wrong", 256) >= 0.03
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "T5_family_banks_transfer_ood": {
            "supported": all(
                value(layer, "tinystories", "native_family_256")
                >= 0.75 * value(layer, "fineweb", "native_family_256")
                and rotation_median(layer, "tinystories", "family", 256)
                >= 0.75 * rotation_median(layer, "fineweb", "family", 256)
                for layer in config["layers"]
            ),
        },
        "T6_layer0_native_oracle_is_sparser": {
            "supported": all(
                value(0, corpus, "native_oracle_128")
                - rotation_median(0, corpus, "oracle", 128) >= 0.05
                for corpus in fixed["evaluation"]
            ),
        },
        "T7_layer1_requires_context_routing": {
            "supported": all(
                value(1, corpus, "native_oracle_256")
                - value(1, corpus, "native_family_256") >= 0.10
                for corpus in fixed["evaluation"]
            ),
        },
        "T8_rotated_family_selection_is_not_random": {
            "supported": all(
                rotation_median(layer, corpus, "family", 256)
                - rotation_median(layer, corpus, "random", 256) >= 0.05
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
        "T9_small_native_banks_are_useful": {
            "supported": all(
                value(layer, corpus, "native_family_32") >= 0.20
                for layer in config["layers"] for corpus in fixed["evaluation"]
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "advantages": advantages,
        "wrong_family_mapping": wrong_family,
        "layers": all_results,
    }, indent=2))

    lines = [
        "# Native-axis versus rotation causal tournament in Layers 0–1",
        "",
        "Family-specific coordinate banks are selected on FineWeb context 1 and frozen before testing context 0 on FineWeb and TinyStories. Native product coordinates compete with two independently and densely mixed orthogonal bases. Every route patches the current example's own product change and is propagated through the full nonlinear suffix.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Native versus rotated coordinates](layer01_native_rotation_tournament.png)",
        "",
        "| Layer | Corpus | Rank | Native oracle | Rotated oracle | Native family | Rotated family | Native wrong | Rotated wrong | Native random | Rotated random |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            for rank in ranks:
                lines.append(
                    f"| {layer} | {corpus} | {rank} | "
                    f"{value(layer, corpus, f'native_oracle_{rank}'):.3f} | "
                    f"{rotation_median(layer, corpus, 'oracle', rank):.3f} | "
                    f"{value(layer, corpus, f'native_family_{rank}'):.3f} | "
                    f"{rotation_median(layer, corpus, 'family', rank):.3f} | "
                    f"{value(layer, corpus, f'native_wrong_{rank}'):.3f} | "
                    f"{rotation_median(layer, corpus, 'wrong', rank):.3f} | "
                    f"{value(layer, corpus, f'native_random_{rank}'):.3f} | "
                    f"{rotation_median(layer, corpus, 'random', rank):.3f} |"
                )
    lines += [
        "",
        f"Maximum full-coordinate rotation reconstruction error: **{max_reconstruction_error:.3e}**.",
        "",
        "A native advantage means the architectural product coordinates give a shorter causal description than a matched coordinate change. Rotated-family success still indicates reusable family structure, but locates it at the subspace/distributed-code level. The oracle/frozen gap measures context-dependent routing rather than semantic coherence by itself.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, layer in enumerate(config["layers"]):
        for column, corpus in enumerate(("fineweb", "tinystories")):
            axis = axes[row, column]
            axis.plot(ranks, [value(layer, corpus, f"native_family_{rank}") for rank in ranks], marker="o", label="native family")
            axis.plot(ranks, [rotation_median(layer, corpus, "family", rank) for rank in ranks], marker="o", label="rotated family")
            axis.plot(ranks, [value(layer, corpus, f"native_oracle_{rank}") for rank in ranks], marker="o", linestyle="--", label="native oracle")
            axis.plot(ranks, [rotation_median(layer, corpus, "oracle", rank) for rank in ranks], marker="o", linestyle="--", label="rotated oracle")
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
