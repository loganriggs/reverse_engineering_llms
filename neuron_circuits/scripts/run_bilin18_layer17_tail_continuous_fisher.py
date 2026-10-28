#!/usr/bin/env python3
"""Continuously optimize a fixed-rank Layer-17 tail subspace for Fisher loss."""

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
from run_bilin18_layer17_tail_fisher_basis import collect_fisher, fisher_quadratic
from run_bilin18_layer17_tail_tensor_canonicalization import centered_vector_r2, common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_continuous_fisher.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_continuous_fisher.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_fisher.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_CONTINUOUS_FISHER.md"
FIGURE = TENSOR_ROOT / "layer17_tail_continuous_fisher.png"


def continuous_prediction(
    coordinates: torch.Tensor,
    forms: torch.Tensor,
    quotient: torch.Tensor,
) -> torch.Tensor:
    reduced_coordinates = coordinates @ quotient
    reduced_forms = torch.einsum("pa,kpq,qb->kab", quotient, forms, quotient)
    return torch.einsum(
        "bi,kij,bj->bk", reduced_coordinates, reduced_forms, reduced_coordinates
    )


def canonical_overlap(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    values = torch.linalg.svdvals(first.T @ second)
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "values": values.detach().cpu().tolist(),
    }


def metric_objective(
    coordinates: torch.Tensor,
    forms: torch.Tensor,
    target: torch.Tensor,
    metric: torch.Tensor,
    quotient: torch.Tensor,
) -> torch.Tensor:
    prediction = continuous_prediction(coordinates, forms, quotient)
    return fisher_quadratic(target - prediction, metric).mean()


def optimize_subspace(
    coordinates: torch.Tensor,
    forms: torch.Tensor,
    target: torch.Tensor,
    metric: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    config: dict,
    seed: int,
    initialization: str = "frobenius_prefix",
) -> tuple[torch.Tensor, dict]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if initialization == "frobenius_prefix":
        initial = torch.eye(
            config["candidate_pool_rank"], config["rank"], device="cuda"
        ) + 1e-3 * torch.randn(
            config["candidate_pool_rank"], config["rank"], generator=generator, device="cuda"
        )
    elif initialization == "random":
        initial = torch.randn(
            config["candidate_pool_rank"], config["rank"], generator=generator, device="cuda"
        )
    elif initialization == "orthogonal_to_prefix":
        initial = torch.zeros(
            config["candidate_pool_rank"], config["rank"], device="cuda"
        )
        initial[config["rank"] : 2 * config["rank"]] = torch.eye(
            config["rank"], device="cuda"
        )
        initial = initial + 1e-3 * torch.randn(
            initial.shape, generator=generator, device="cuda"
        )
    else:
        raise ValueError(f"unknown initialization: {initialization}")
    parameter = torch.nn.Parameter(initial)
    optimizer = torch.optim.Adam([parameter], lr=config["learning_rate"])
    history = []
    with torch.enable_grad():
        for step in range(config["optimization_steps"]):
            sampled = train_indices[torch.randint(
                len(train_indices),
                (min(config["optimization_batch_size"], len(train_indices)),),
                generator=generator,
                device="cuda",
            )]
            quotient = torch.linalg.qr(parameter, mode="reduced").Q
            loss = metric_objective(
                coordinates[sampled], forms, target[sampled], metric[sampled], quotient
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 50 == 0 or step == config["optimization_steps"] - 1:
                with torch.no_grad():
                    quotient_eval = torch.linalg.qr(parameter, mode="reduced").Q
                    train_loss = float(metric_objective(
                        coordinates[train_indices], forms, target[train_indices],
                        metric[train_indices], quotient_eval,
                    ))
                    validation_loss = float(metric_objective(
                        coordinates[validation_indices], forms, target[validation_indices],
                        metric[validation_indices], quotient_eval,
                    ))
                history.append({
                    "step": step,
                    "batch_loss": float(loss.detach()),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                })
    with torch.no_grad():
        quotient = torch.linalg.qr(parameter, mode="reduced").Q.detach()
    return quotient, {"history": history, "initialization": initialization, "seed": seed}


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64) - np.mean(first)
    second = np.asarray(second, dtype=np.float64) - np.mean(second)
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def score_summary(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    target_np, prediction_np = target.cpu().numpy(), prediction.cpu().numpy()
    correlations = [correlation(target_np[:, i], prediction_np[:, i]) for i in range(target.shape[1])]
    return {
        "centered_vector_r2": centered_vector_r2(target_np, prediction_np),
        "median_axis_correlation": float(np.median(correlations)),
        "minimum_axis_correlation": float(np.min(correlations)),
    }


def summarize(records: dict[str, list[float]]) -> dict[str, float]:
    return {
        "median_exact_kl": float(np.median(records["exact_kl"])),
        "mean_exact_kl": float(np.mean(records["exact_kl"])),
        "ce_delta": float(np.mean(records["ce_delta"])),
        "top1_agreement": float(np.mean(records["top1"])),
    }


@torch.inference_mode()
def evaluate(
    model,
    spec: dict,
    config: dict,
    output_basis: torch.Tensor,
    inverse: torch.Tensor,
    candidate_basis: torch.Tensor,
    forms: torch.Tensor,
    traces: torch.Tensor,
    quotients: dict[str, torch.Tensor],
) -> dict:
    rows = corpus_rows(spec)
    positions = config["positions"]
    block = model.blocks[config["layer"]]
    dimension = model.config.n_embd
    records = {
        name: {field: [] for field in ("exact_kl", "ce_delta", "top1")}
        for name in quotients
    }
    score_target, score_predictions = [], {name: [] for name in quotients}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1)
        x = normalized[:, positions].float().flatten(0, 1)
        raw_write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x)).float()
        native_write = raw_write + block.mlp.Down_bias.float()
        native_score = raw_write @ output_basis
        z = x @ inverse
        coordinates = z @ candidate_basis
        isotropic = z.square().sum(1, keepdim=True) * traces[None]
        predictions = {
            name: continuous_prediction(coordinates, forms, quotient) + isotropic
            for name, quotient in quotients.items()
        }
        score_target.append(native_score.cpu())
        for name, prediction in predictions.items():
            score_predictions[name].append(prediction.cpu())
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
        for name, prediction in predictions.items():
            changed_write = native_write + (prediction - native_score) @ output_basis.T
            logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(pre + changed_write, (dimension,))).float() / 30
            )
            logprob = logits.log_softmax(-1)
            exact_kl = (native_probability * (native_logprob - logprob)).sum(1)
            ce = -logprob.gather(1, targets[:, None]).squeeze(1)
            records[name]["exact_kl"].extend(exact_kl.cpu().tolist())
            records[name]["ce_delta"].extend((ce - native_ce).cpu().tolist())
            records[name]["top1"].extend((logits.argmax(-1) == native_top1).cpu().tolist())
    target = torch.cat(score_target)
    return {
        "scores": {
            name: score_summary(target, torch.cat(prediction))
            for name, prediction in score_predictions.items()
        },
        "behavioral": {name: summarize(row) for name, row in records.items()},
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE_ARRAYS)
    output_basis = torch.from_numpy(source["tail_output_basis"][:, :4]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float().detach()
    left = block.mlp.Left.weight.float().detach()
    right = block.mlp.Right.weight.float().detach()

    print("discovery tensor and metric", flush=True)
    discovery_x, discovery_write = collect(model, config["layer"], config["discovery"], config)
    square, inverse, covariance = covariance_transforms(
        discovery_x, config["covariance_floor_fraction"]
    )
    cleaned_forms, traces = [], []
    for component in range(4):
        coefficients = down.T @ output_basis[:, component]
        raw = left.T @ (coefficients[:, None] * right)
        raw = 0.5 * (raw + raw.T)
        clean, trace = trace_clean(square @ raw @ square)
        cleaned_forms.append(clean)
        traces.append(trace)
    traces_tensor = torch.stack(traces)
    _, common_basis = common_input(cleaned_forms)
    candidate_basis = common_basis[:, : config["candidate_pool_rank"]]
    forms = torch.einsum(
        "di,kde,ej->kij", candidate_basis, torch.stack(cleaned_forms), candidate_basis
    ).detach()
    z = discovery_x @ inverse
    coordinates = (z @ candidate_basis).detach()
    isotropic = (z.square().sum(1, keepdim=True) * traces_tensor[None]).detach()
    target = (discovery_write @ output_basis - isotropic).detach()
    metric = collect_fisher(model, config["discovery"], config, output_basis).detach()
    row_ids = torch.arange(len(discovery_x), device="cuda") // len(config["positions"])
    even_rows = torch.nonzero((row_ids % 2) == 0, as_tuple=False).flatten()
    odd_rows = torch.nonzero((row_ids % 2) == 1, as_tuple=False).flatten()

    print("continuous fit A", flush=True)
    fit_a, info_a = optimize_subspace(
        coordinates, forms, target, metric, even_rows, odd_rows,
        config, config["seed"],
    )
    print("continuous fit B", flush=True)
    fit_b, info_b = optimize_subspace(
        coordinates, forms, target, metric, odd_rows, even_rows,
        config, config["seed"] + 1,
    )
    frobenius = torch.eye(config["candidate_pool_rank"], device="cuda")[:, : config["rank"]]
    quotients = {"continuous_a": fit_a, "continuous_b": fit_b, "frobenius": frobenius}
    discovery_metrics = {}
    for name, quotient in quotients.items():
        prediction = continuous_prediction(coordinates, forms, quotient)
        discovery_metrics[name] = {
            "even_fisher": float(fisher_quadratic(
                target[even_rows] - prediction[even_rows], metric[even_rows]
            ).mean()),
            "odd_fisher": float(fisher_quadratic(
                target[odd_rows] - prediction[odd_rows], metric[odd_rows]
            ).mean()),
            **score_summary(target + isotropic, prediction + isotropic),
        }

    results = {}
    for corpus, spec in config["evaluation"].items():
        print("external evaluation", corpus, flush=True)
        results[corpus] = evaluate(
            model, spec, config, output_basis, inverse, candidate_basis,
            forms, traces_tensor, quotients,
        )

    overlap = canonical_overlap(fit_a, fit_b)
    hypotheses = {
        "O1_continuous_fit_improves_heldout_fisher": {
            "supported": (
                discovery_metrics["continuous_a"]["odd_fisher"]
                <= 0.85 * discovery_metrics["frobenius"]["odd_fisher"]
                and discovery_metrics["continuous_b"]["even_fisher"]
                <= 0.85 * discovery_metrics["frobenius"]["even_fisher"]
            ),
        },
        "O2_continuous_fit_improves_external_mean_kl": {
            "supported": all(
                row["behavioral"]["continuous_a"]["mean_exact_kl"]
                <= 0.90 * row["behavioral"]["frobenius"]["mean_exact_kl"]
                for row in results.values()
            ),
        },
        "O3_continuous_fit_improves_external_median_kl": {
            "supported": all(
                row["behavioral"]["continuous_a"]["median_exact_kl"]
                <= 0.90 * row["behavioral"]["frobenius"]["median_exact_kl"]
                for row in results.values()
            ),
        },
        "O4_independent_fits_recover_same_subspace": {
            "supported": overlap["mean"] >= 0.80,
            "overlap": overlap,
        },
        "O5_improvement_replicates_across_fits": {
            "supported": all(
                row["behavioral"][name]["mean_exact_kl"]
                < row["behavioral"]["frobenius"]["mean_exact_kl"]
                for row in results.values() for name in ("continuous_a", "continuous_b")
            ),
        },
        "O6_continuous_rank8_reaches_behavioral_gate": {
            "supported": all(
                row["behavioral"]["continuous_a"]["median_exact_kl"] <= 0.005
                for row in results.values()
            ),
        },
    }
    np.savez_compressed(ARRAYS, **{
        "candidate_basis": candidate_basis.cpu().numpy(),
        "forms": forms.cpu().numpy(),
        "traces": traces_tensor.cpu().numpy(),
        "continuous_a": fit_a.cpu().numpy(),
        "continuous_b": fit_b.cpu().numpy(),
        "frobenius": frobenius.cpu().numpy(),
    })
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "covariance": covariance,
        "fit_a": info_a,
        "fit_b": info_b,
        "subspace_overlap": overlap,
        "discovery": discovery_metrics,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Continuous downstream-Fisher optimization of the Layer-17 tail program",
        "",
        "The tail tensor, FineWeb covariance metric, 128D tensor-derived candidate pool, and rank-eight budget are frozen. Two independent fits optimize an orthonormal 8D subspace on alternating discovery rows using pulled-back final-logit Fisher error; each is validated on the other half and on disjoint FineWeb/WikiText/TinyStories with exact finite KL.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Continuous Fisher basis](layer17_tail_continuous_fisher.png)",
        "",
        f"Independent optimized-subspace canonical cosine: mean **{overlap['mean']:.3f}**, minimum **{overlap['minimum']:.3f}**.",
        "",
        "| Corpus | Program | Score R² | Mean exact KL | Median exact KL |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        for name in ("frobenius", "continuous_a", "continuous_b"):
            lines.append(
                f"| {corpus} | {name} | {row['scores'][name]['centered_vector_r2']:.3f} | "
                f"{row['behavioral'][name]['mean_exact_kl']:.5f} | "
                f"{row['behavioral'][name]['median_exact_kl']:.5f} |"
            )
    lines += [
        "",
        "| Program | Even-row Fisher | Odd-row Fisher | Discovery score R² |",
        "|---|---:|---:|---:|",
    ]
    for name, row in discovery_metrics.items():
        lines.append(
            f"| {name} | {row['even_fisher']:.4e} | {row['odd_fisher']:.4e} | {row['centered_vector_r2']:.3f} |"
        )
    lines += [
        "",
        "A stable improvement would show that the tail is globally compact but ordered by the wrong metric. Split instability or OOD reversal would instead identify a context/distribution-dependent behavioral quotient. The optimization remains restricted to the frozen tensor-derived 128D pool and does not alter the model.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, plots = plt.subplots(1, 3, figsize=(14, 4.2))
    for label, info in (("fit A", info_a), ("fit B", info_b)):
        plots[0].plot([row["step"] for row in info["history"]], [row["validation_loss"] for row in info["history"]], marker="o", label=label)
    plots[0].set_yscale("log")
    plots[0].set_xlabel("optimization step")
    plots[0].set_ylabel("opposite-half Fisher loss")
    plots[0].set_title("Split validation")
    plots[0].legend(fontsize=8)
    x_axis = np.arange(len(results))
    width = 0.25
    for index, name in enumerate(("frobenius", "continuous_a", "continuous_b")):
        plots[1].bar(x_axis + (index - 1) * width, [results[c]["behavioral"][name]["mean_exact_kl"] for c in results], width, label=name)
        plots[2].bar(x_axis + (index - 1) * width, [results[c]["behavioral"][name]["median_exact_kl"] for c in results], width, label=name)
    for plot, title, ylabel in (
        (plots[1], "Mean behavioral error", "mean exact KL"),
        (plots[2], "Median behavioral error", "median exact KL"),
    ):
        plot.set_xticks(x_axis, list(results), rotation=15)
        plot.set_title(title)
        plot.set_ylabel(ylabel)
        plot.legend(fontsize=8)
    for plot in plots:
        plot.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
