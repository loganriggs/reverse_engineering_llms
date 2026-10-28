#!/usr/bin/env python3
"""Train a nonlinear 128D router for layer-8 top interaction modes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_query_conditioned_interpreter import score_metrics
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_learned_router.json")
QUADRATIC_OUTPUT = TENSOR_ROOT / "layer8_quadratic_router.json"
OUTPUT = TENSOR_ROOT / "layer8_learned_router.json"
CHECKPOINT = TENSOR_ROOT / "layer8_learned_router.pt"
REPORT = TENSOR_ROOT / "LAYER8_LEARNED_ROUTER.md"
FIGURE = TENSOR_ROOT / "layer8_learned_router.png"


class SupportRouter(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, queries: int, modes: int):
        super().__init__()
        self.queries = queries
        self.modes = modes
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, queries * modes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).reshape(len(inputs), self.queries, self.modes)


def route_metrics(
    router: SupportRouter,
    features: torch.Tensor,
    contributions: torch.Tensor,
    budget: int,
) -> dict:
    logits = router(features)
    selected = logits.topk(budget, dim=-1).indices
    full = contributions.sum(-1)
    routed = torch.gather(contributions, -1, selected).sum(-1)
    oracle_indices = contributions.abs().topk(budget, dim=-1).indices
    oracle = torch.gather(contributions, -1, oracle_indices).sum(-1)
    full_np = full.detach().cpu().numpy()
    result = {}
    for name, prediction in (("routed", routed), ("oracle", oracle)):
        corr, r2 = score_metrics(full_np, prediction.detach().cpu().numpy())
        result[name] = {
            "median_correlation": float(np.median(corr)),
            "median_centered_r2": float(np.median(r2)),
        }
    recall = (selected[..., :, None] == oracle_indices[..., None, :]).any(-1).float().mean(-1)
    coverage = torch.gather(contributions.abs(), -1, selected).sum(-1) / contributions.abs().sum(-1).clamp_min(1e-30)
    result["median_oracle_support_recall"] = float(recall.median())
    result["median_absolute_mass_coverage"] = float(coverage.median())
    return result


@torch.inference_mode()
def build_contributions(states: torch.Tensor, basis: torch.Tensor, vectors: torch.Tensor, values: torch.Tensor):
    z = states @ basis
    projections = torch.einsum("nd,qdk->nqk", z, vectors)
    return z, projections.square() * values[None]


def main() -> None:
    config = json.loads(CONFIG.read_text())
    quadratic = json.loads(QUADRATIC_OUTPUT.read_text())
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    layer = config["layer"]
    discovery_x, discovery_write = collect(model, layer, config["discovery"], config)
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = state_vh.T[:, : config["dimension"]]
    centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
    _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
    directions = write_vh.T[:, : config["queries"]]
    block = model.blocks[layer]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float() @ basis
    right = block.mlp.Right.weight.float() @ basis
    coefficients = down.T @ directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    values, vectors = ordered_eigensystem(matrices)
    discovery_z, discovery_contributions = build_contributions(discovery_x, basis, vectors, values)
    feature_mean = discovery_z[:, : config["router_dimension"]].mean(0)
    feature_std = discovery_z[:, : config["router_dimension"]].std(0).clamp_min(1e-6)
    features = (discovery_z[:, : config["router_dimension"]] - feature_mean) / feature_std
    oracle_indices = discovery_contributions.abs().topk(config["budget"], dim=-1).indices
    targets = torch.zeros_like(discovery_contributions)
    targets.scatter_(-1, oracle_indices, 1)

    permutation = torch.randperm(len(features), device="cuda")
    split = int(config["training_fraction"] * len(features))
    train_indices = permutation[:split]
    validation_indices = permutation[split:]
    router = SupportRouter(
        config["router_dimension"], config["hidden_dimension"], config["queries"], config["modes"]
    ).cuda()
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    positive_weight = torch.tensor(
        (config["modes"] - config["budget"]) / config["budget"], device="cuda"
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 1)
    best = None
    history = []
    router.train()
    for step in range(1, config["steps"] + 1):
        chosen = train_indices[
            torch.randint(0, len(train_indices), (config["router_batch_size"],), generator=generator, device="cuda")
        ]
        logits = router(features[chosen])
        loss = loss_function(logits, targets[chosen])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % config["validation_interval"] == 0 or step == 1:
            router.eval()
            with torch.inference_mode():
                validation = route_metrics(
                    router,
                    features[validation_indices],
                    discovery_contributions[validation_indices],
                    config["budget"],
                )
            row = {
                "step": step,
                "training_loss": float(loss.detach()),
                "validation_correlation": validation["routed"]["median_correlation"],
                "validation_r2": validation["routed"]["median_centered_r2"],
                "validation_support_recall": validation["median_oracle_support_recall"],
            }
            history.append(row)
            print(row, flush=True)
            if best is None or row["validation_correlation"] > best["validation_correlation"]:
                best = {**row, "state": copy.deepcopy(router.state_dict())}
            router.train()
    router.load_state_dict(best.pop("state"))
    router.eval()
    torch.save({
        "config": config,
        "state_dict": router.state_dict(),
        "feature_mean": feature_mean.cpu(),
        "feature_std": feature_std.cpu(),
        "basis": basis.cpu(),
        "directions": directions.cpu(),
        "values": values.cpu(),
        "vectors": vectors.cpu(),
    }, CHECKPOINT)

    results = {}
    for corpus, corpus_config in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        states, _ = collect(model, layer, corpus_config, config)
        z, contributions = build_contributions(states, basis, vectors, values)
        evaluation_features = (z[:, : config["router_dimension"]] - feature_mean) / feature_std
        result = route_metrics(router, evaluation_features, contributions, config["budget"])
        result["quadratic128"] = quadratic["results"][corpus]["router_dimensions"]["128"]["exact_routed"]
        results[corpus] = result

    hypotheses = {
        "L1_learned_router_is_predictive": {
            "supported": all(results[corpus]["routed"]["median_correlation"] >= 0.85 for corpus in results)
        },
        "L2_learned_router_beats_quadratic": {
            "supported": all(
                results[corpus]["routed"]["median_correlation"]
                - results[corpus]["quadratic128"]["median_correlation"] >= 0.03
                for corpus in results
            )
        },
        "L3_learned_router_approaches_oracle": {
            "supported": all(
                results[corpus]["oracle"]["median_correlation"]
                - results[corpus]["routed"]["median_correlation"] <= 0.05
                for corpus in results
            )
        },
        "L4_learned_router_recovers_support": {
            "supported": all(results[corpus]["median_oracle_support_recall"] >= 0.50 for corpus in results)
        },
        "L5_learned_router_transfers_ood": {
            "supported": all(
                results["fineweb"]["routed"]["median_correlation"]
                - results[corpus]["routed"]["median_correlation"] <= 0.05
                for corpus in ("wikitext", "tinystories")
            )
        },
        "L6_validation_predicts_test": {
            "supported": abs(results["fineweb"]["routed"]["median_correlation"] - best["validation_correlation"]) <= 0.05,
            "validation": best["validation_correlation"],
            "fineweb": results["fineweb"]["routed"]["median_correlation"],
        },
    }
    parameter_count = sum(parameter.numel() for parameter in router.parameters())
    OUTPUT.write_text(json.dumps({
        "config": config,
        "parameter_count": parameter_count,
        "best_validation": best,
        "history": history,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# Supervised nonlinear router for layer-8 modes",
        "",
        "A two-layer GELU router maps 128 standardized state coordinates to 16×512 support logits. It is trained on 80% of the FineWeb discovery positions with weighted multilabel BCE, selected only by routed-score correlation on the remaining 20%, then frozen for all evaluation corpora. Selected modes are evaluated exactly, so this is an information/predictability test rather than an efficient replacement.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Router parameters: **{parameter_count:,}**; best validation step: **{best['step']}**; validation correlation: **{best['validation_correlation']:.3f}**.",
        "",
        "![Learned router](layer8_learned_router.png)",
        "",
        "| Corpus | Learned corr./R² | Quadratic-128 corr. | Oracle corr. | Support recall | Abs. mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, item in results.items():
        lines.append(
            f"| {corpus} | {item['routed']['median_correlation']:.3f}/{item['routed']['median_centered_r2']:.3f} | "
            f"{item['quadratic128']['median_correlation']:.3f} | {item['oracle']['median_correlation']:.3f} | "
            f"{item['median_oracle_support_recall']:.3f} | {item['median_absolute_mass_coverage']:.3f} |"
        )
    lines += [
        "",
        "The router is deliberately overpowered as a predictability probe. Its parameter count and dense 16×512 logits must be counted; even a successful result would show that route identity is learnable from 128D context, not that the resulting explanation is shorter or cheaper than the native computation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot([row["step"] for row in history], [row["validation_correlation"] for row in history], marker="o")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("validation routed correlation")
    axes[0].set_title("Checkpoint selection")
    corpora = list(results)
    x = np.arange(len(corpora))
    axes[1].bar(x - 0.25, [results[c]["quadratic128"]["median_correlation"] for c in corpora], 0.25, label="quadratic")
    axes[1].bar(x, [results[c]["routed"]["median_correlation"] for c in corpora], 0.25, label="learned")
    axes[1].bar(x + 0.25, [results[c]["oracle"]["median_correlation"] for c in corpora], 0.25, label="oracle")
    axes[1].set_xticks(x, corpora)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("median correlation")
    axes[1].set_title("Frozen evaluation")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, CHECKPOINT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
