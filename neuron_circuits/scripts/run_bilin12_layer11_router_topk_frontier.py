#!/usr/bin/env python3
"""Measure the exact top-k execution frontier of the frozen Layer-11 router."""

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
    CHECKPOINT as ROUTER_CHECKPOINT,
    SOURCE,
    ConditionalOutputRouter,
    collect_context,
    exact_kl,
    logits_from_write,
    router_features,
)
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer11_router_topk_frontier.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer11_router_topk_frontier.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER11_ROUTER_TOPK_FRONTIER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer11_router_topk_frontier.png"


def combination_bases(
    expert_bases: torch.Tensor,
    anchor_rank: int,
    k: int,
) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    subsets = list(itertools.combinations(range(len(expert_bases)), k))
    anchor = expert_bases[0, :, :anchor_rank]
    rows = []
    for subset in subsets:
        tails = torch.cat(
            [expert_bases[index, :, anchor_rank:] for index in subset], dim=1
        )
        rows.append(torch.linalg.qr(torch.cat((anchor, tails), dim=1), mode="reduced").Q)
    return subsets, torch.stack(rows)


def routed_combination_ids(
    soft_gates: torch.Tensor,
    subsets: list[tuple[int, ...]],
    k: int,
) -> torch.Tensor:
    selected = soft_gates.topk(k, dim=1).indices.sort(dim=1).values
    powers = 2 ** selected
    masks = powers.sum(1)
    lookup = torch.full(
        (2 ** soft_gates.shape[1],), -1, device=soft_gates.device, dtype=torch.long
    )
    for index, subset in enumerate(subsets):
        lookup[sum(2 ** value for value in subset)] = index
    result = lookup[masks]
    assert bool((result >= 0).all())
    return result


def summarize(records: dict[str, list[float]]) -> dict[str, float]:
    return {
        "mean_exact_kl": float(np.mean(records["kl"])),
        "median_exact_kl": float(np.median(records["kl"])),
        "p90_exact_kl": float(np.quantile(records["kl"], 0.9)),
        "mean_ce_delta": float(np.mean(records["ce_delta"])),
        "top1_agreement": float(np.mean(records["top1"])),
        "relative_write_mse": float(
            np.sum(records["write_error"]) / max(np.sum(records["write_energy"]), 1e-30)
        ),
    }


@torch.inference_mode()
def evaluate(
    model,
    spec: dict,
    config: dict,
    program: ConditionalOutputRouter,
    mean: torch.Tensor,
    transform: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    raw4: torch.Tensor,
    pca_tail: torch.Tensor,
    seed: int,
) -> dict:
    data = collect_context(model, spec, config)
    features = router_features(data["state"], *transform)
    soft_gates, _ = program.gates(features, 0.2, "soft")
    expert_bases = program.expert_bases()
    centered = data["write"] - mean
    generator = torch.Generator(device="cuda").manual_seed(seed)

    structures = {}
    for k in config["topk"]:
        subsets, bases = combination_bases(expert_bases, raw4.shape[1], k)
        router_ids = routed_combination_ids(soft_gates, subsets, k)
        shuffled_ids = router_ids[
            torch.randperm(len(router_ids), generator=generator, device="cuda")
        ]
        structures[k] = {
            "subsets": subsets,
            "bases": bases,
            "router_ids": router_ids,
            "shuffled_ids": shuffled_ids,
            "fixed_basis": torch.cat((raw4, pca_tail[:, : k * 16]), dim=1),
        }

    names = [
        f"{prefix}_top{k}"
        for k in config["topk"]
        for prefix in ("router", "oracle", "shuffled")
    ] + [f"fixed_rank{4 + 16 * k}" for k in config["topk"]]
    records = {
        name: {key: [] for key in ("kl", "ce_delta", "top1", "write_error", "write_energy")}
        for name in names
    }
    batch_size = 64
    for start in range(0, len(centered), batch_size):
        stop = min(start + batch_size, len(centered))
        batch_index = torch.arange(stop - start, device="cuda")
        native_write = data["write"][start:stop]
        local_centered = centered[start:stop]
        pre = data["pre"][start:stop]
        targets = data["target"][start:stop]
        native_logits = logits_from_write(model, pre, native_write)
        native_logprob = F.log_softmax(native_logits, -1)
        native_probability = native_logprob.exp()
        native_ce = -native_logprob.gather(1, targets[:, None]).squeeze(1)
        native_top1 = native_logits.argmax(-1)
        energy = local_centered.square().sum(1)

        for k, structure in structures.items():
            bases = structure["bases"]
            coordinates = torch.einsum("bd,cdr->bcr", local_centered, bases)
            combination_centered = torch.einsum("bcr,cdr->bcd", coordinates, bases)
            combination_logits = torch.stack([
                logits_from_write(model, pre, mean + combination_centered[:, index])
                for index in range(len(bases))
            ], dim=1)
            combination_kl = torch.stack([
                (native_probability * (
                    native_logprob - F.log_softmax(combination_logits[:, index], -1)
                )).sum(-1)
                for index in range(len(bases))
            ], dim=1)
            ids = {
                f"router_top{k}": structure["router_ids"][start:stop],
                f"shuffled_top{k}": structure["shuffled_ids"][start:stop],
                f"oracle_top{k}": combination_kl.argmin(1),
            }
            candidates = {
                name: (
                    mean + combination_centered[batch_index, chosen],
                    combination_logits[batch_index, chosen],
                )
                for name, chosen in ids.items()
            }
            fixed_name = f"fixed_rank{4 + 16 * k}"
            fixed_basis = structure["fixed_basis"]
            fixed_write = mean + (local_centered @ fixed_basis) @ fixed_basis.T
            candidates[fixed_name] = (
                fixed_write,
                logits_from_write(model, pre, fixed_write),
            )
            for name, (changed_write, logits) in candidates.items():
                logprob = F.log_softmax(logits, -1)
                kl = (native_probability * (native_logprob - logprob)).sum(-1)
                ce = -logprob.gather(1, targets[:, None]).squeeze(1)
                records[name]["kl"].extend(kl.cpu().tolist())
                records[name]["ce_delta"].extend((ce - native_ce).cpu().tolist())
                records[name]["top1"].extend((logits.argmax(-1) == native_top1).cpu().tolist())
                records[name]["write_error"].extend((changed_write - native_write).square().sum(1).cpu().tolist())
                records[name]["write_energy"].extend(energy.cpu().tolist())
    behavioral = {name: summarize(row) for name, row in records.items()}
    full = behavioral["router_top4"]["mean_exact_kl"]
    minimum_k = next(
        (k for k in config["topk"] if behavioral[f"router_top{k}"]["mean_exact_kl"] <= 1.1 * full),
        None,
    )
    return {
        "behavioral": behavioral,
        "minimum_routed_k_within_10pct_union": minimum_k,
        "mean_route_confidence": float(soft_gates.max(1).values.mean()),
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE)
    checkpoint = torch.load(ROUTER_CHECKPOINT, map_location="cuda", weights_only=False)
    raw4 = torch.from_numpy(source["raw_outlier_basis"]).cuda().float()
    pca_tail = torch.from_numpy(source["tail_basis"]).cuda().float()
    mean = torch.from_numpy(source["post_bias_write_mean"]).cuda().float()
    initial_tail = torch.stack([
        pca_tail[:, expert * config["tail_rank_per_expert"] : (expert + 1) * config["tail_rank_per_expert"]]
        for expert in range(config["experts"])
    ])
    program = ConditionalOutputRouter(
        raw4, initial_tail, config["router_rank"], config["seed"]
    ).cuda()
    program.load_state_dict(checkpoint["program"])
    program.eval()
    transform = (
        checkpoint["router_mean"].cuda(),
        checkpoint["router_vectors"].cuda(),
        checkpoint["router_scale"].cuda(),
    )
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    results = {}
    for index, (corpus, spec) in enumerate(config["evaluation"].items()):
        print("top-k frontier", corpus, flush=True)
        results[corpus] = evaluate(
            model, spec, config, program, mean, transform, raw4, pca_tail,
            config["seed"] + index,
        )

    def value(corpus: str, name: str) -> float:
        return results[corpus]["behavioral"][name]["mean_exact_kl"]

    hypotheses = {
        "K1_top2_matches_full_union": {
            "supported": all(value(c, "router_top2") <= 1.1 * value(c, "router_top4") for c in results),
        },
        "K2_top2_beats_fixed_rank36": {
            "supported": all(value(c, "router_top2") <= 0.8 * value(c, "fixed_rank36") for c in results),
        },
        "K3_top2_router_closes_oracle_gap": {
            "supported": all(
                (value(c, "fixed_rank36") - value(c, "router_top2"))
                / max(value(c, "fixed_rank36") - value(c, "oracle_top2"), 1e-30)
                >= 0.70 for c in results
            ),
        },
        "K4_oracle_top2_is_near_sufficient": {
            "supported": all(results[c]["behavioral"]["oracle_top2"]["median_exact_kl"] <= 0.02 for c in results),
        },
        "K5_top3_retains_most_union_gain": {
            "supported": all(
                (value(c, "router_top1") - value(c, "router_top3"))
                >= 0.8 * (value(c, "router_top1") - value(c, "router_top4"))
                for c in results
            ),
        },
        "K6_top2_assignment_matters": {
            "supported": all(value(c, "shuffled_top2") >= 1.25 * value(c, "router_top2") for c in results),
        },
        "K7_two_experts_are_enough": {
            "supported": all(results[c]["minimum_routed_k_within_10pct_union"] <= 2 for c in results),
        },
        "K8_more_experts_monotonically_help": {
            "supported": all(
                all(
                    value(c, f"router_top{k + 1}") <= value(c, f"router_top{k}")
                    for k in (1, 2, 3)
                ) for c in results
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Frozen top-k execution frontier for the independent Layer-11 router",
        "",
        "The trained four-expert bank and upstream router are frozen. For each token, the top-k route selects the exact orthogonal union of its k expert tails plus the shared raw four-space, using 20/36/52/68 directions for k=1/2/3/4. Per-example oracle subsets, shuffled assignments, and fixed PCA spaces of the same rank are evaluated on the same states.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Top-k execution frontier](bilin12_layer11_router_topk_frontier.png)",
        "",
        "| Corpus | Routed k=1 / 2 / 3 / 4 | Oracle k=1 / 2 / 3 / 4 | Fixed rank 20 / 36 / 52 / 68 | Minimum k within 10% |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        b = row["behavioral"]
        routed = " / ".join(f"{b[f'router_top{k}']['mean_exact_kl']:.5f}" for k in config["topk"])
        oracle = " / ".join(f"{b[f'oracle_top{k}']['mean_exact_kl']:.5f}" for k in config["topk"])
        fixed = " / ".join(f"{b[f'fixed_rank{4 + 16*k}']['mean_exact_kl']:.5f}" for k in config["topk"])
        lines.append(
            f"| {corpus} | {routed} | {oracle} | {fixed} | {row['minimum_routed_k_within_10pct_union']} |"
        )
    lines += [
        "",
        "A low minimum k with a small router–oracle gap supports sparse conditional execution. A smooth improvement through k=4, especially when oracle subsets also need large k, supports simultaneous distributed tail use. Fixed-rank controls distinguish useful learned expert unions from merely adding dimensions. Oracle KL is not required to be monotone in k: Euclidean projection onto a larger output space can worsen final KL after RMSNorm and unembedding. Here oracle k=3 beats k=4, another direct instance of variance/write geometry disagreeing with the behavioral metric.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(results)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for corpus in corpora:
        axes[0].plot(
            config["topk"],
            [value(corpus, f"router_top{k}") for k in config["topk"]],
            marker="o", label=f"{corpus} routed",
        )
        axes[0].plot(
            config["topk"],
            [value(corpus, f"oracle_top{k}") for k in config["topk"]],
            marker="x", linestyle="--", label=f"{corpus} oracle",
        )
    axes[0].set_xticks(config["topk"])
    axes[0].set_xlabel("experts retained")
    axes[0].set_ylabel("mean exact KL")
    axes[0].set_title("Execution-support frontier")
    axes[0].legend(fontsize=7, ncol=2)
    x = np.arange(len(corpora))
    width = 0.25
    for offset, name in enumerate(("router_top2", "oracle_top2", "shuffled_top2")):
        axes[1].bar(
            x + (offset - 1) * width,
            [value(corpus, name) for corpus in corpora],
            width, label=name,
        )
    axes[1].set_xticks(x, corpora)
    axes[1].set_ylabel("mean exact KL")
    axes[1].set_title("Top-two assignment controls")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
