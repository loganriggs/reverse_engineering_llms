#!/usr/bin/env python3
"""Test functional and symmetry-aware stability of Layer-11 output routers."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer11_conditional_output_router import (
    CHECKPOINT as PRIMARY_CHECKPOINT,
    CONFIG as PRIMARY_CONFIG,
    SOURCE,
    ConditionalOutputRouter,
    collect_context,
    evaluate,
    fit_program,
    router_features,
)
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer11_router_seed_stability.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer11_router_seed_stability.json"
CHECKPOINT = TENSOR_ROOT / "bilin12_layer11_router_seed_stability.pt"
REPORT = TENSOR_ROOT / "BILIN12_LAYER11_ROUTER_SEED_STABILITY.md"
FIGURE = TENSOR_ROOT / "bilin12_layer11_router_seed_stability.png"


def best_expert_permutation(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    anchor_rank: int,
) -> tuple[tuple[int, ...], float]:
    overlap = torch.empty(len(reference), len(candidate), device=reference.device)
    for first in range(len(reference)):
        for second in range(len(candidate)):
            overlap[first, second] = torch.linalg.svdvals(
                reference[first, :, anchor_rank:].T
                @ candidate[second, :, anchor_rank:]
            ).mean()
    best = None
    for permutation in itertools.permutations(range(len(candidate))):
        score = float(torch.stack([
            overlap[index, permutation[index]] for index in range(len(reference))
        ]).mean())
        if best is None or score > best[1]:
            best = (permutation, score)
    assert best is not None
    return best


def topk_jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    intersection = (
        first[:, :, None] == second[:, None, :]
    ).any(2).sum(1).float()
    union = first.shape[1] + second.shape[1] - intersection
    return float((intersection / union).mean())


def union_basis(bases: torch.Tensor, anchor_rank: int) -> torch.Tensor:
    anchor = bases[0, :, :anchor_rank]
    tails = bases[:, :, anchor_rank:].permute(1, 0, 2).flatten(1)
    return torch.linalg.qr(torch.cat((anchor, tails), 1), mode="reduced").Q


def aligned_route_metrics(
    reference_soft: torch.Tensor,
    candidate_soft: torch.Tensor,
    permutation: tuple[int, ...],
) -> dict[str, float]:
    inverse = torch.empty(len(permutation), device=candidate_soft.device, dtype=torch.long)
    for reference_index, candidate_index in enumerate(permutation):
        inverse[candidate_index] = reference_index
    reference_hard = reference_soft.argmax(1)
    candidate_hard = inverse[candidate_soft.argmax(1)]
    agreement = float((reference_hard == candidate_hard).float().mean())
    reference_top2 = reference_soft.topk(2, 1).indices
    candidate_top2 = inverse[candidate_soft.topk(2, 1).indices]
    return {
        "hard_agreement": agreement,
        "top2_jaccard": topk_jaccard(reference_top2, candidate_top2),
    }


def make_program(
    raw4: torch.Tensor,
    tail: torch.Tensor,
    config: dict,
    seed: int,
) -> ConditionalOutputRouter:
    initial_tail = torch.stack([
        tail[:, expert * config["tail_rank_per_expert"] : (expert + 1) * config["tail_rank_per_expert"]]
        for expert in range(config["experts"])
    ])
    return ConditionalOutputRouter(
        raw4, initial_tail, config["router_rank"], seed
    ).cuda()


def main() -> None:
    config = json.loads(CONFIG.read_text())
    primary_config = json.loads(PRIMARY_CONFIG.read_text())
    source = np.load(SOURCE)
    primary_checkpoint = torch.load(
        PRIMARY_CHECKPOINT, map_location="cuda", weights_only=False
    )
    raw4 = torch.from_numpy(source["raw_outlier_basis"]).cuda().float()
    tail = torch.from_numpy(source["tail_basis"]).cuda().float()
    mean = torch.from_numpy(source["post_bias_write_mean"]).cuda().float()
    fixed_tail16 = torch.cat((raw4, tail[:, :16]), 1)
    fixed_tail64 = torch.cat((raw4, tail[:, :64]), 1)
    transform = (
        primary_checkpoint["router_mean"].cuda(),
        primary_checkpoint["router_vectors"].cuda(),
        primary_checkpoint["router_scale"].cuda(),
    )
    model = load_snapshot_model(Path(primary_config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    print("collecting shared router discovery data", flush=True)
    discovery = collect_context(model, primary_config["discovery"], primary_config)
    features = router_features(discovery["state"], *transform).detach()
    train_indices = torch.nonzero(discovery["row_id"] % 2 == 0, as_tuple=False).flatten()
    all_validation = torch.nonzero(discovery["row_id"] % 2 == 1, as_tuple=False).flatten()
    validation_generator = torch.Generator(device="cuda").manual_seed(primary_config["seed"])
    validation_indices = all_validation[torch.randperm(
        len(all_validation), generator=validation_generator, device="cuda"
    )[: primary_config["validation_examples"]]]

    programs = {}
    histories = {}
    states = {}
    for index, seed in enumerate(config["seeds"]):
        name = f"seed{seed}"
        program = make_program(raw4, tail, primary_config, seed)
        if index == 0:
            print("loading primary seed", seed, flush=True)
            program.load_state_dict(primary_checkpoint["program"])
            histories[name] = [{"source": "primary_checkpoint"}]
            states[name] = program.state_dict()
        else:
            print("training restart", seed, flush=True)
            fit_config = {**primary_config, "seed": seed}
            history, state = fit_program(
                program, discovery, features, mean, train_indices,
                validation_indices, model, fit_config,
            )
            histories[name] = history
            states[name] = state
        program.eval()
        programs[name] = program

    position_routes = {}
    with torch.no_grad():
        for name, program in programs.items():
            hard, _ = program.gates(features, 0.2, "hard")
            means = torch.stack([
                hard[discovery["position"] == position].float().mean(0)
                for position in range(len(primary_config["positions"]))
            ])
            position_routes[name] = F.one_hot(
                means.argmax(1), primary_config["experts"]
            ).float()

    behavioral = {name: {} for name in programs}
    route_alignment = {}
    reference_name = f"seed{config['seeds'][0]}"
    reference_bases = programs[reference_name].expert_bases().detach()
    reference_union = union_basis(reference_bases, config["anchor_rank"])
    expert_alignment = {}
    for name, program in programs.items():
        candidate_bases = program.expert_bases().detach()
        permutation, score = best_expert_permutation(
            reference_bases, candidate_bases, config["anchor_rank"]
        )
        candidate_union = union_basis(candidate_bases, config["anchor_rank"])
        expert_alignment[name] = {
            "permutation_reference_to_candidate": list(permutation),
            "mean_matched_tail_overlap": score,
            "union_mean_canonical_cosine": float(
                torch.linalg.svdvals(reference_union.T @ candidate_union).mean()
            ),
        }
        route_alignment[name] = {}

    for corpus_index, (corpus, spec) in enumerate(primary_config["evaluation"].items()):
        print("seed-stability evaluation", corpus, flush=True)
        context = collect_context(model, spec, primary_config)
        corpus_features = router_features(context["state"], *transform)
        reference_soft, _ = programs[reference_name].gates(corpus_features, 0.2, "soft")
        for name, program in programs.items():
            row = evaluate(
                model, spec, primary_config, program, mean, transform,
                fixed_tail16, fixed_tail64, position_routes[name],
                config["seeds"][0] + corpus_index,
            )
            behavioral[name][corpus] = row
            candidate_soft, _ = program.gates(corpus_features, 0.2, "soft")
            permutation = tuple(expert_alignment[name]["permutation_reference_to_candidate"])
            route_alignment[name][corpus] = aligned_route_metrics(
                reference_soft, candidate_soft, permutation
            )

    performance = {}
    for corpus in primary_config["evaluation"]:
        hard_values = np.asarray([
            behavioral[name][corpus]["behavioral"]["hard"]["mean_exact_kl"]
            for name in programs
        ])
        performance[corpus] = {
            "hard_mean_kl": hard_values.tolist(),
            "hard_kl_coefficient_of_variation": float(hard_values.std() / hard_values.mean()),
        }

    nonprimary = [f"seed{seed}" for seed in config["seeds"][1:]]
    hypotheses = {
        "S1_hard_performance_is_stable": {
            "supported": all(row["hard_kl_coefficient_of_variation"] <= 0.10 for row in performance.values()),
        },
        "S2_expert_union_is_stable": {
            "supported": all(expert_alignment[name]["union_mean_canonical_cosine"] >= 0.90 for name in nonprimary),
        },
        "S3_matched_expert_subspaces_are_stable": {
            "supported": all(expert_alignment[name]["mean_matched_tail_overlap"] >= 0.70 for name in nonprimary),
        },
        "S4_hard_routes_are_stable": {
            "supported": all(route_alignment[name][corpus]["hard_agreement"] >= 0.70 for name in nonprimary for corpus in primary_config["evaluation"]),
        },
        "S5_top2_route_sets_are_stable": {
            "supported": all(route_alignment[name][corpus]["top2_jaccard"] >= 0.70 for name in nonprimary for corpus in primary_config["evaluation"]),
        },
        "S6_contextual_gain_replicates": {
            "supported": all(
                behavioral[name][corpus]["behavioral"]["hard"]["mean_exact_kl"]
                < behavioral[name][corpus]["behavioral"]["fixed_tail16"]["mean_exact_kl"]
                for name in programs for corpus in primary_config["evaluation"]
            ),
        },
        "S7_fixed68_gap_replicates": {
            "supported": all(
                behavioral[name][corpus]["behavioral"]["hard"]["mean_exact_kl"]
                >= 1.10 * behavioral[name][corpus]["behavioral"]["fixed_tail64"]["mean_exact_kl"]
                for name in programs for corpus in primary_config["evaluation"]
            ),
        },
    }
    torch.save({"states": states, "config": config}, CHECKPOINT)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "expert_alignment": expert_alignment,
        "route_alignment": route_alignment,
        "performance": performance,
        "histories": histories,
        "behavioral": behavioral,
    }, indent=2, allow_nan=False))

    lines = [
        "# Seed stability of the independent Layer-11 conditional router",
        "",
        "Three programs use the same data split, frozen router coordinates, raw-four anchor, and 4×16 tail capacity but independent initialization and minibatch sampling. Expert names are aligned by the permutation maximizing mean tail-subspace canonical overlap before route agreement is measured.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Router seed stability](bilin12_layer11_router_seed_stability.png)",
        "",
        "| Seed | Matched tail overlap | Union overlap | FineWeb hard KL | WikiText | TinyStories |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in config["seeds"]:
        name = f"seed{seed}"
        alignment = expert_alignment[name]
        lines.append(
            f"| {seed} | {alignment['mean_matched_tail_overlap']:.3f} | {alignment['union_mean_canonical_cosine']:.3f} | "
            + " | ".join(
                f"{behavioral[name][corpus]['behavioral']['hard']['mean_exact_kl']:.5f}"
                for corpus in ("fineweb", "wikitext", "tinystories")
            ) + " |"
        )
    lines += [
        "",
        "| Restart / corpus | Hard route agreement | Top-2 Jaccard |",
        "|---|---:|---:|",
    ]
    for name in nonprimary:
        for corpus, row in route_alignment[name].items():
            lines.append(f"| {name} / {corpus} | {row['hard_agreement']:.3f} | {row['top2_jaccard']:.3f} |")
    lines += [
        "",
        "Stable performance with unstable expert identities would support a reproducible function class but a noncanonical internal decomposition. Stable aligned routes and subspaces would justify interpreting the individual experts. Failure of both says the original conditional gain is optimization-contingent.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(primary_config["evaluation"])
    seeds = [f"seed{seed}" for seed in config["seeds"]]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(corpora))
    width = 0.24
    for index, name in enumerate(seeds):
        axes[0].bar(
            x + (index - 1) * width,
            [behavioral[name][corpus]["behavioral"]["hard"]["mean_exact_kl"] for corpus in corpora],
            width, label=name,
        )
    axes[0].set_xticks(x, corpora)
    axes[0].set_ylabel("hard-router mean KL")
    axes[0].set_title("Functional recurrence")
    axes[0].legend(fontsize=8)
    labels = [f"{name[-2:]}/{corpus}" for name in nonprimary for corpus in corpora]
    agreement = [route_alignment[name][corpus]["hard_agreement"] for name in nonprimary for corpus in corpora]
    jaccard = [route_alignment[name][corpus]["top2_jaccard"] for name in nonprimary for corpus in corpora]
    positions = np.arange(len(labels))
    axes[1].bar(positions - 0.18, agreement, 0.36, label="hard agreement")
    axes[1].bar(positions + 0.18, jaccard, 0.36, label="top-2 Jaccard")
    axes[1].axhline(0.70, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(positions, labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Alignment-aware route stability")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, CHECKPOINT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
