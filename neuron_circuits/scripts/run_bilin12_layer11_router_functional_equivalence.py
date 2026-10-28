#!/usr/bin/env python3
"""Determine whether unstable routers are functionally gauge-equivalent."""

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
    exact_kl,
    logits_from_write,
    router_features,
)
from run_bilin12_layer11_router_seed_stability import CHECKPOINT as STABILITY_CHECKPOINT
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer11_router_functional_equivalence.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer11_router_functional_equivalence.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER11_ROUTER_FUNCTIONAL_EQUIVALENCE.md"
FIGURE = TENSOR_ROOT / "bilin12_layer11_router_functional_equivalence.png"


def symmetric_kl(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first_logprob = F.log_softmax(first_logits, -1)
    second_logprob = F.log_softmax(second_logits, -1)
    first_probability = first_logprob.exp()
    second_probability = second_logprob.exp()
    forward = (first_probability * (first_logprob - second_logprob)).sum(-1)
    backward = (second_probability * (second_logprob - first_logprob)).sum(-1)
    return 0.5 * (forward + backward)


def projector_overlap(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    rank = min(first.shape[2], second.shape[2])
    return torch.einsum("bdr,bds->brs", first, second).square().sum((1, 2)) / rank


def correlation(first: list[float], second: list[float]) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def make_program(raw4, tail, primary_config, seed, state):
    initial_tail = torch.stack([
        tail[:, expert * primary_config["tail_rank_per_expert"] : (expert + 1) * primary_config["tail_rank_per_expert"]]
        for expert in range(primary_config["experts"])
    ])
    program = ConditionalOutputRouter(
        raw4, initial_tail, primary_config["router_rank"], seed
    ).cuda()
    program.load_state_dict(state)
    program.eval()
    return program


@torch.inference_mode()
def evaluate_corpus(
    model,
    spec: dict,
    primary_config: dict,
    config: dict,
    programs: dict[str, ConditionalOutputRouter],
    mean: torch.Tensor,
    transform: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict:
    data = collect_context(model, spec, primary_config)
    features = router_features(data["state"], *transform)
    pairs = list(itertools.combinations(programs, 2))
    records = {
        pair: {
            key: [] for key in (
                "write_disagreement", "native_error_first", "native_error_second",
                "symmetric_kl", "native_kl_first", "native_kl_second",
                "projector_overlap", "write_cosine", "top1_agreement",
            )
        }
        for pair in pairs
    }
    bases = {name: program.expert_bases() for name, program in programs.items()}
    for start in range(0, len(features), config["batch_size"]):
        stop = start + config["batch_size"]
        selected = slice(start, stop)
        native_write = data["write"][selected]
        centered = native_write - mean
        pre = data["pre"][selected]
        native_logits = logits_from_write(model, pre, native_write)
        rows = {}
        for name, program in programs.items():
            hard, _ = program.gates(features[selected], 0.2, "hard")
            expert_index = hard.argmax(1)
            changed, _, _ = program(
                features[selected], centered, mean, 0.2, mode="hard"
            )
            logits = logits_from_write(model, pre, changed)
            rows[name] = {
                "write": changed,
                "centered": changed - mean,
                "logits": logits,
                "native_kl": exact_kl(native_logits, logits),
                "basis": bases[name][expert_index],
            }
        for pair in pairs:
            first, second = (rows[name] for name in pair)
            row = records[pair]
            row["write_disagreement"].extend((first["write"] - second["write"]).square().sum(1).cpu().tolist())
            row["native_error_first"].extend((first["write"] - native_write).square().sum(1).cpu().tolist())
            row["native_error_second"].extend((second["write"] - native_write).square().sum(1).cpu().tolist())
            row["symmetric_kl"].extend(symmetric_kl(first["logits"], second["logits"]).cpu().tolist())
            row["native_kl_first"].extend(first["native_kl"].cpu().tolist())
            row["native_kl_second"].extend(second["native_kl"].cpu().tolist())
            row["projector_overlap"].extend(projector_overlap(first["basis"], second["basis"]).cpu().tolist())
            row["write_cosine"].extend(F.cosine_similarity(first["centered"], second["centered"], dim=1).cpu().tolist())
            row["top1_agreement"].extend((first["logits"].argmax(1) == second["logits"].argmax(1)).cpu().tolist())
    output = {}
    for pair, row in records.items():
        mean_native_write_error = 0.5 * (
            np.mean(row["native_error_first"]) + np.mean(row["native_error_second"])
        )
        mean_native_kl = 0.5 * (
            np.mean(row["native_kl_first"]) + np.mean(row["native_kl_second"])
        )
        output["__".join(pair)] = {
            "relative_write_disagreement": float(np.mean(row["write_disagreement"]) / max(mean_native_write_error, 1e-30)),
            "mean_write_cosine": float(np.mean(row["write_cosine"])),
            "symmetric_prediction_kl": float(np.mean(row["symmetric_kl"])),
            "symmetric_kl_relative_to_native_error": float(np.mean(row["symmetric_kl"]) / max(mean_native_kl, 1e-30)),
            "mean_selected_projector_overlap": float(np.mean(row["projector_overlap"])),
            "native_kl_correlation": correlation(row["native_kl_first"], row["native_kl_second"]),
            "approximation_top1_agreement": float(np.mean(row["top1_agreement"])),
        }
    return output


def main() -> None:
    config = json.loads(CONFIG.read_text())
    primary_config = json.loads(PRIMARY_CONFIG.read_text())
    source = np.load(SOURCE)
    primary_checkpoint = torch.load(PRIMARY_CHECKPOINT, map_location="cuda", weights_only=False)
    stability_checkpoint = torch.load(STABILITY_CHECKPOINT, map_location="cuda", weights_only=False)
    raw4 = torch.from_numpy(source["raw_outlier_basis"]).cuda().float()
    tail = torch.from_numpy(source["tail_basis"]).cuda().float()
    mean = torch.from_numpy(source["post_bias_write_mean"]).cuda().float()
    transform = (
        primary_checkpoint["router_mean"].cuda(),
        primary_checkpoint["router_vectors"].cuda(),
        primary_checkpoint["router_scale"].cuda(),
    )
    programs = {}
    for seed in config["seeds"]:
        name = f"seed{seed}"
        programs[name] = make_program(
            raw4, tail, primary_config, seed,
            stability_checkpoint["states"][name],
        )
    model = load_snapshot_model(Path(primary_config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    results = {}
    for corpus, spec in primary_config["evaluation"].items():
        print("functional equivalence", corpus, flush=True)
        results[corpus] = evaluate_corpus(
            model, spec, primary_config, config, programs, mean, transform
        )
    all_rows = [row for corpus in results.values() for row in corpus.values()]
    hypotheses = {
        "E1_reconstructed_writes_are_equivalent": {
            "supported": all(row["relative_write_disagreement"] <= 0.20 for row in all_rows),
        },
        "E2_output_distributions_are_equivalent": {
            "supported": all(row["symmetric_kl_relative_to_native_error"] <= 0.10 for row in all_rows),
        },
        "E3_selected_projectors_are_equivalent": {
            "supported": all(row["mean_selected_projector_overlap"] >= 0.80 for row in all_rows),
        },
        "E4_per_example_error_profiles_are_equivalent": {
            "supported": all(row["native_kl_correlation"] >= 0.80 for row in all_rows),
        },
        "E5_approximation_top1_is_equivalent": {
            "supported": all(row["approximation_top1_agreement"] >= 0.95 for row in all_rows),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Functional equivalence of unstable Layer-11 routers",
        "",
        "The three seed fits have nearly identical aggregate native-distribution KL but different experts and route labels. This audit compares the actual hard-selected projectors, reconstructed writes, approximate output distributions, and per-example error profiles without aligning expert names.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Functional equivalence](bilin12_layer11_router_functional_equivalence.png)",
        "",
        "| Corpus / seed pair | Write disagreement / native error | Symmetric KL / native KL | Projector overlap | Error correlation | Top-1 agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, pairs in results.items():
        for pair, row in pairs.items():
            lines.append(
                f"| {corpus} / {pair} | {row['relative_write_disagreement']:.3f} | "
                f"{row['symmetric_kl_relative_to_native_error']:.3f} | {row['mean_selected_projector_overlap']:.3f} | "
                f"{row['native_kl_correlation']:.3f} | {row['approximation_top1_agreement']:.3f} |"
            )
    lines += [
        "",
        "If writes and distributions agree despite projector disagreement, the decompositions are gauge choices for one stable function. If only aggregate error agrees, the training objective admits many inequivalent partial approximations; individual routes, experts, and reconstructed examples then cannot be interpreted as recovered native mechanisms.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    labels, write_values, distribution_values, projector_values = [], [], [], []
    for corpus, pairs in results.items():
        for pair, row in pairs.items():
            labels.append(f"{corpus[:2]}/{pair.split('__')[1][-2:]}")
            write_values.append(row["relative_write_disagreement"])
            distribution_values.append(row["symmetric_kl_relative_to_native_error"])
            projector_values.append(row["mean_selected_projector_overlap"])
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].bar(x - 0.18, write_values, 0.36, label="write disagreement / native error")
    axes[0].bar(x + 0.18, distribution_values, 0.36, label="symmetric KL / native KL")
    axes[0].axhline(0.20, color="black", linestyle="--", linewidth=1, label="write gate")
    axes[0].axhline(0.10, color="gray", linestyle=":", linewidth=1, label="distribution gate")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_title("Pairwise functional disagreement")
    axes[0].legend(fontsize=8)
    axes[1].bar(x, projector_values)
    axes[1].axhline(0.80, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Selected-projector overlap")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
