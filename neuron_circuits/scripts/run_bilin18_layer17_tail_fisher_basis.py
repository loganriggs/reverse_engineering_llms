#!/usr/bin/env python3
"""Select a compact Layer-17 tail program using downstream Fisher curvature."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_bottleneck_tail_ontology import corpus_rows
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_tensor_canonicalization import (
    centered_vector_r2,
    common_input,
    predict_scores,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_fisher_basis.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_fisher_basis.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_fisher_basis.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_FISHER_BASIS.md"
FIGURE = TENSOR_ROOT / "layer17_tail_fisher_basis.png"


def fisher_quadratic(error: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bi,bij,bj->b", error, metric, error)


@torch.inference_mode()
def final_logit_fisher(
    model,
    residual: torch.Tensor,
    output_basis: torch.Tensor,
) -> torch.Tensor:
    """Softmax Fisher pulled back through tanh, unembedding, and final RMSNorm."""
    dimension = residual.shape[1]
    rms = residual.square().mean(1, keepdim=True).sqrt()
    dot = residual @ output_basis
    normalized_basis = (
        output_basis[None] / rms[:, None]
        - residual[:, :, None] * dot[:, None, :] / (dimension * rms[:, None].pow(3))
    )
    raw = model.lm_head(residual / rms).float()
    logits = 30 * torch.tanh(raw / 30)
    derivative = 1 - torch.tanh(raw / 30).square()
    logit_jacobian = torch.einsum(
        "vd,bdk->bvk", model.lm_head.weight.float(), normalized_basis
    ) * derivative[:, :, None]
    probability = logits.softmax(-1)
    mean = torch.einsum("bv,bvk->bk", probability, logit_jacobian)
    second = torch.einsum(
        "bv,bvi,bvj->bij", probability, logit_jacobian, logit_jacobian
    )
    metric = second - mean[:, :, None] * mean[:, None, :]
    return 0.5 * (metric + metric.transpose(1, 2))


def objective(
    target: torch.Tensor,
    prediction: torch.Tensor,
    metric: torch.Tensor,
) -> float:
    return float(fisher_quadratic(target - prediction, metric).mean())


def greedy_axis_selection(
    coordinates: torch.Tensor,
    restricted_forms: torch.Tensor,
    target: torch.Tensor,
    metric: torch.Tensor,
    rank: int,
) -> tuple[list[int], list[float]]:
    """Greedily add canonical axes minimizing a fixed quadratic score metric."""
    selected: list[int] = []
    available = set(range(coordinates.shape[1]))
    prediction = torch.zeros_like(target)
    losses = []
    for _ in range(rank):
        best = None
        for candidate in sorted(available):
            zc = coordinates[:, candidate]
            delta = zc[:, None].square() * restricted_forms[:, candidate, candidate][None]
            if selected:
                selected_tensor = torch.as_tensor(selected, device=coordinates.device)
                cross = coordinates[:, selected_tensor] @ restricted_forms[:, selected_tensor, candidate].T
                delta = delta + 2 * zc[:, None] * cross
            loss = objective(target, prediction + delta, metric)
            if best is None or loss < best[0]:
                best = (loss, candidate, delta)
        assert best is not None
        losses.append(best[0])
        selected.append(best[1])
        available.remove(best[1])
        prediction = prediction + best[2]
    return selected, losses


@torch.inference_mode()
def collect_fisher(
    model,
    spec: dict,
    config: dict,
    output_basis: torch.Tensor,
) -> torch.Tensor:
    rows = corpus_rows(spec)
    positions = config["positions"]
    block = model.blocks[config["layer"]]
    total = len(rows) * len(positions)
    metrics = torch.empty(total, config["components"], config["components"], device="cuda")
    offset = 0
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1)
        x = normalized[:, positions].float().flatten(0, 1)
        write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x)).float()
        residual = pre + write + block.mlp.Down_bias.float()
        current = final_logit_fisher(model, residual, output_basis)
        metrics[offset : offset + len(current)] = current
        offset += len(current)
    return metrics


def prediction_for_selection(
    coordinates: torch.Tensor,
    forms: torch.Tensor,
    selection: list[int],
    isotropic: torch.Tensor,
) -> torch.Tensor:
    indices = torch.as_tensor(selection, device=coordinates.device)
    z = coordinates[:, indices]
    restricted = forms[:, indices][:, :, indices]
    return torch.einsum("bi,kij,bj->bk", z, restricted, z) + isotropic


def score_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target_np, prediction_np = target.cpu().numpy(), prediction.cpu().numpy()
    correlations = []
    for index in range(target.shape[1]):
        first = target_np[:, index] - target_np[:, index].mean()
        second = prediction_np[:, index] - prediction_np[:, index].mean()
        correlations.append(float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30)))
    return {
        "centered_vector_r2": centered_vector_r2(target_np, prediction_np),
        "median_axis_correlation": float(np.median(correlations)),
        "minimum_axis_correlation": float(np.min(correlations)),
    }


def summarize_behavior(records: dict[str, list[float]]) -> dict[str, float]:
    return {
        "median_exact_kl": float(np.median(records["exact_kl"])),
        "mean_exact_kl": float(np.mean(records["exact_kl"])),
        "ce_delta": float(np.mean(records["ce_delta"])),
        "top1_agreement": float(np.mean(records["top1"])),
    }


@torch.inference_mode()
def evaluate_behavior(
    model,
    spec: dict,
    config: dict,
    output_basis: torch.Tensor,
    inverse: torch.Tensor,
    candidate_basis: torch.Tensor,
    restricted_forms: torch.Tensor,
    traces: torch.Tensor,
    selections: dict[str, list[int]],
) -> dict[str, dict[str, float]]:
    rows = corpus_rows(spec)
    positions = config["positions"]
    block = model.blocks[config["layer"]]
    dimension = model.config.n_embd
    variants = {
        f"{method}_rank{rank}": axes[:rank]
        for method, axes in selections.items() for rank in config["evaluation_ranks"]
    }
    records = {
        name: {field: [] for field in ("exact_kl", "ce_delta", "top1")}
        for name in variants
    }
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1)
        x = normalized[:, positions].float().flatten(0, 1)
        product = block.mlp.Left(x) * block.mlp.Right(x)
        raw_write = block.mlp.Down(product).float()
        native_write = raw_write + block.mlp.Down_bias.float()
        native_tail_score = raw_write @ output_basis
        z = x @ inverse
        coordinates = z @ candidate_basis
        isotropic = z.square().sum(1, keepdim=True) * traces[None]
        native_logits = 30 * torch.tanh(
            model.lm_head(F.rms_norm(pre + native_write, (dimension,))).float() / 30
        )
        native_logprob = native_logits.log_softmax(-1)
        native_probability = native_logprob.exp()
        targets = torch.as_tensor(
            rows[start : start + len(raw), np.asarray(positions) + 1].reshape(-1),
            device="cuda", dtype=torch.long,
        )
        native_ce = -native_logprob.gather(1, targets[:, None]).squeeze(1)
        native_top1 = native_logits.argmax(-1)
        for name, axes in variants.items():
            prediction = prediction_for_selection(
                coordinates, restricted_forms, axes, isotropic
            )
            changed_write = native_write + (prediction - native_tail_score) @ output_basis.T
            logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(pre + changed_write, (dimension,))).float() / 30
            )
            logprob = logits.log_softmax(-1)
            exact_kl = (native_probability * (native_logprob - logprob)).sum(1)
            ce = -logprob.gather(1, targets[:, None]).squeeze(1)
            records[name]["exact_kl"].extend(exact_kl.cpu().tolist())
            records[name]["ce_delta"].extend((ce - native_ce).cpu().tolist())
            records[name]["top1"].extend((logits.argmax(-1) == native_top1).cpu().tolist())
    return {name: summarize_behavior(row) for name, row in records.items()}


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE_ARRAYS)
    output_basis = torch.from_numpy(source["tail_output_basis"][:, : config["components"]]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()

    print("discovery states and tensor", flush=True)
    discovery_x, discovery_write = collect(
        model, config["layer"], config["discovery"], config
    )
    square, inverse, covariance = covariance_transforms(
        discovery_x, config["covariance_floor_fraction"]
    )
    forms, traces = [], []
    for component in range(config["components"]):
        coefficients = down.T @ output_basis[:, component]
        raw = left.T @ (coefficients[:, None] * right)
        raw = 0.5 * (raw + raw.T)
        clean, trace = trace_clean(square @ raw @ square)
        forms.append(clean)
        traces.append(trace)
    traces_tensor = torch.stack(traces)
    _, common_basis = common_input(forms)
    candidate_basis = common_basis[:, : config["candidate_pool_rank"]]
    forms_tensor = torch.stack(forms)
    restricted_forms = torch.einsum(
        "di,kde,ej->kij", candidate_basis, forms_tensor, candidate_basis
    )
    z = discovery_x @ inverse
    coordinates = z @ candidate_basis
    isotropic = z.square().sum(1, keepdim=True) * traces_tensor[None]
    target = discovery_write @ output_basis - isotropic

    print("discovery final-logit Fisher", flush=True)
    fisher = collect_fisher(model, config["discovery"], config, output_basis)
    identity = torch.eye(config["components"], device="cuda")[None].expand(len(target), -1, -1)
    fisher_selected, fisher_losses = greedy_axis_selection(
        coordinates, restricted_forms, target, fisher, config["selection_rank"]
    )
    mse_selected, mse_losses = greedy_axis_selection(
        coordinates, restricted_forms, target, identity, config["selection_rank"]
    )
    frobenius_selected = list(range(config["selection_rank"]))
    selections = {
        "fisher": fisher_selected,
        "score_mse": mse_selected,
        "frobenius": frobenius_selected,
    }

    discovery_rows = {}
    for method, selection in selections.items():
        discovery_rows[method] = {}
        for rank in config["evaluation_ranks"]:
            clean_prediction = prediction_for_selection(
                coordinates, restricted_forms, selection[:rank], torch.zeros_like(isotropic)
            )
            error = target - clean_prediction
            discovery_rows[method][str(rank)] = {
                "fisher_objective": float(fisher_quadratic(error, fisher).mean()),
                "score_mse": float(error.square().mean()),
                **score_metrics(target + isotropic, clean_prediction + isotropic),
            }

    generator = np.random.default_rng(config["seed"])
    random_objectives = []
    for _ in range(config["random_subsets"]):
        selection = generator.choice(
            config["candidate_pool_rank"], size=8, replace=False
        ).tolist()
        prediction = prediction_for_selection(
            coordinates, restricted_forms, selection, torch.zeros_like(isotropic)
        )
        random_objectives.append(float(fisher_quadratic(target - prediction, fisher).mean()))

    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluation", corpus, flush=True)
        x, write = collect(model, config["layer"], spec, config)
        eval_z = x @ inverse
        eval_coordinates = eval_z @ candidate_basis
        eval_isotropic = eval_z.square().sum(1, keepdim=True) * traces_tensor[None]
        eval_target = write @ output_basis
        score_rows = {}
        for method, selection in selections.items():
            score_rows[method] = {}
            for rank in config["evaluation_ranks"]:
                prediction = prediction_for_selection(
                    eval_coordinates, restricted_forms, selection[:rank], eval_isotropic
                )
                score_rows[method][str(rank)] = score_metrics(eval_target, prediction)
        behavior = evaluate_behavior(
            model, spec, config, output_basis, inverse, candidate_basis,
            restricted_forms, traces_tensor, selections,
        )
        results[corpus] = {"scores": score_rows, "behavioral": behavior}

    fisher_rank8 = discovery_rows["fisher"]["8"]["fisher_objective"]
    frobenius_rank8 = discovery_rows["frobenius"]["8"]["fisher_objective"]
    hypotheses = {
        "F1_fisher_selection_improves_discovery_objective": {
            "supported": fisher_rank8 <= 0.80 * frobenius_rank8,
            "fisher": fisher_rank8,
            "frobenius": frobenius_rank8,
        },
        "F2_fisher_rank8_improves_exact_kl": {
            "supported": all(
                row["behavioral"]["fisher_rank8"]["median_exact_kl"]
                <= 0.80 * row["behavioral"]["frobenius_rank8"]["median_exact_kl"]
                for row in results.values()
            ),
        },
        "F3_fisher_rank16_reaches_behavioral_gate": {
            "supported": all(
                row["behavioral"]["fisher_rank16"]["median_exact_kl"] <= 0.005
                for row in results.values()
            ),
        },
        "F4_fisher_selection_beats_mse_selection": {
            "supported": all(
                row["behavioral"]["fisher_rank8"]["median_exact_kl"]
                < row["behavioral"]["score_mse_rank8"]["median_exact_kl"]
                for row in results.values()
            ),
        },
        "F5_fisher_selection_is_nontrivial": {
            "supported": sum(axis >= 8 for axis in fisher_selected[:8]) >= 2,
            "selected_rank8": fisher_selected[:8],
        },
        "F6_fisher_selection_beats_random_subsets": {
            "supported": fisher_rank8 < min(random_objectives),
            "fisher": fisher_rank8,
            "best_random": min(random_objectives),
        },
    }
    np.savez_compressed(ARRAYS, **{
        "candidate_basis": candidate_basis.cpu().numpy(),
        "restricted_forms": restricted_forms.cpu().numpy(),
        "traces": traces_tensor.cpu().numpy(),
        "fisher_selected": np.asarray(fisher_selected),
        "mse_selected": np.asarray(mse_selected),
        "fisher_losses": np.asarray(fisher_losses),
        "mse_losses": np.asarray(mse_losses),
    })
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "covariance": covariance,
        "selections": selections,
        "selection_losses": {"fisher": fisher_losses, "score_mse": mse_losses},
        "random_rank8_fisher_objectives": random_objectives,
        "discovery": discovery_rows,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Downstream-Fisher selection of the Layer-17 tail basis",
        "",
        "The tensor, covariance transform, four-output tail space, candidate pool, and retained rank are fixed. Only the axis-selection objective changes: cleaned tensor Frobenius order, unweighted tail-score MSE, or the exact local softmax Fisher pulled back through final RMSNorm, unembedding, and logit tanh. Every selected program is evaluated by exact finite replacement of the native tail scores.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Tail Fisher basis](layer17_tail_fisher_basis.png)",
        "",
        f"Fisher rank-8 axes: **{fisher_selected[:8]}**. Score-MSE rank-8 axes: **{mse_selected[:8]}**. Frobenius rank-8 axes: **{frobenius_selected[:8]}**.",
        "",
        "| Corpus | Method | Rank-8 score R² | Rank-8 exact KL | Rank-16 exact KL |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        for method in ("frobenius", "score_mse", "fisher"):
            lines.append(
                f"| {corpus} | {method} | {row['scores'][method]['8']['centered_vector_r2']:.3f} | "
                f"{row['behavioral'][f'{method}_rank8']['median_exact_kl']:.5f} | "
                f"{row['behavioral'][f'{method}_rank16']['median_exact_kl']:.5f} |"
            )
    lines += [
        "",
        "| Method | Discovery rank-8 Fisher objective | Discovery rank-8 score MSE |",
        "|---|---:|---:|",
    ]
    for method in ("frobenius", "score_mse", "fisher"):
        row = discovery_rows[method]["8"]
        lines.append(f"| {method} | {row['fisher_objective']:.4e} | {row['score_mse']:.4e} |")
    lines += [
        "",
        "The Fisher objective is a local quadratic approximation around native final distributions; the reported KL is an exact finite recomputation. Improvement in the former without the latter would falsify local-metric sufficiency. Axis selection is restricted to a frozen 128D tensor-derived candidate pool, so failure would not rule out a continuously optimized subspace.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, plots = plt.subplots(1, 3, figsize=(14, 4.2))
    ranks = config["evaluation_ranks"]
    for method in ("frobenius", "score_mse", "fisher"):
        plots[0].plot(ranks, [discovery_rows[method][str(rank)]["fisher_objective"] for rank in ranks], marker="o", label=method)
    plots[0].set_yscale("log")
    plots[0].set_xlabel("selected rank")
    plots[0].set_ylabel("discovery Fisher score error")
    plots[0].set_title("Selection objective")
    plots[0].legend(fontsize=8)
    width = 0.25
    x_axis = np.arange(len(results))
    for index, method in enumerate(("frobenius", "score_mse", "fisher")):
        plots[1].bar(x_axis + (index - 1) * width, [results[c]["behavioral"][f"{method}_rank8"]["median_exact_kl"] for c in results], width, label=method)
    plots[1].set_xticks(x_axis, list(results), rotation=15)
    plots[1].set_ylabel("rank-8 median exact KL")
    plots[1].set_title("Held-out behavior")
    plots[1].legend(fontsize=8)
    plots[2].hist(random_objectives, bins=16, alpha=0.75, label="random rank-8")
    plots[2].axvline(fisher_rank8, color="tab:red", linewidth=2, label="Fisher greedy")
    plots[2].axvline(frobenius_rank8, color="black", linewidth=2, linestyle="--", label="Frobenius")
    plots[2].set_xlabel("discovery Fisher objective")
    plots[2].set_title("Candidate-pool control")
    plots[2].legend(fontsize=8)
    for plot in plots:
        plot.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
