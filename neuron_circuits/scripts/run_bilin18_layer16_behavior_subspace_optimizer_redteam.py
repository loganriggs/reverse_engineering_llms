#!/usr/bin/env python3
"""Red-team behavior-subspace optimization with deterministic full-batch gradients."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_behavior_subspace import fidelity
from run_bilin18_layer16_forward_adjoint_atlas import collect_factors, covariance_basis
from run_bilin18_layer16_upstream_tangent_channels import autograd_check, build_input_axes
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_behavior_subspace_optimizer_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_behavior_subspace_optimizer_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_BEHAVIOR_SUBSPACE_OPTIMIZER_REDTEAM.md"
BASIS_OUTPUT = TENSOR_ROOT / "bilin18_layer16_behavior_subspace_optimizer_redteam.pt"


def relative_error_objective(forward, effective, target, basis, denominator=None):
    """Aggregate relative squared error for y_s = J_s P h_s."""
    predicted = ((forward @ basis) * (effective @ basis)[:, None]).sum(-1)
    if denominator is None:
        denominator = target.double().square().sum().clamp_min(1e-30)
    return (target - predicted).double().square().sum() / denominator


def grassmann_gradient(basis, gradient):
    """Remove within-span rotations; the objective depends only on U U^T."""
    return gradient - basis @ (basis.T @ gradient)


@torch.no_grad()
def retract(basis):
    """QR retraction with a deterministic sign convention."""
    q, r = torch.linalg.qr(basis, mode="reduced")
    sign = torch.sign(torch.diag(r))
    sign[sign == 0] = 1
    return q * sign[None]


@torch.inference_mode()
def aggregate_fidelity(data, basis, chunk_size=128):
    device = basis.device
    numerator = 0.0
    denominator = 0.0
    for start in range(0, len(data["target"]), chunk_size):
        sl = slice(start, start + chunk_size)
        forward = data["forward"][sl].float().to(device)
        effective = data["effective_adjoint"][sl].float().to(device)
        target = data["target"][sl].float().to(device)
        predicted = ((forward @ basis) * (effective @ basis)[:, None]).sum(-1)
        numerator += float((target - predicted).double().square().sum())
        denominator += float(target.double().square().sum())
    return 1.0 - numerator / max(denominator, 1e-30)


def full_batch_gradient(data, basis, chunk_size):
    """Exact gradient of aggregate normalized error, accumulated by chunks."""
    device = basis.device
    parameter = basis.detach().clone().requires_grad_(True)
    denominator = sum(
        float(data["target"][start:start + chunk_size].double().square().sum())
        for start in range(0, len(data["target"]), chunk_size)
    )
    for start in range(0, len(data["target"]), chunk_size):
        sl = slice(start, start + chunk_size)
        forward = data["forward"][sl].float().to(device)
        effective = data["effective_adjoint"][sl].float().to(device)
        target = data["target"][sl].float().to(device)
        loss = relative_error_objective(
            forward, effective, target, parameter,
            torch.tensor(denominator, dtype=torch.float64, device=device),
        )
        loss.backward()
    gradient = parameter.grad.detach()
    tangent = grassmann_gradient(parameter.detach(), gradient)
    return tangent, float(gradient.norm()), float(tangent.norm())


def optimize_full_batch(train, validation, initial, learning_rate, config):
    rank = config["subspace_rank"]
    basis = initial[:, :rank].float().cuda().clone()
    initial_train = aggregate_fidelity(train, basis, config["gradient_batch_size"])
    initial_validation = aggregate_fidelity(validation, basis, config["gradient_batch_size"])
    best_validation = initial_validation
    best_step = 0
    best_basis = basis.cpu().clone()
    curve = [{
        "step": 0,
        "train_fidelity": initial_train,
        "validation_fidelity": initial_validation,
        "euclidean_gradient_norm": None,
        "grassmann_gradient_norm": None,
    }]
    for step in range(1, config["optimization_steps"] + 1):
        gradient, euclidean_norm, tangent_norm = full_batch_gradient(
            train, basis, config["gradient_batch_size"])
        basis = retract(basis - learning_rate * gradient)
        if step % config["validation_interval"] == 0 or step == config["optimization_steps"]:
            train_value = aggregate_fidelity(train, basis, config["gradient_batch_size"])
            validation_value = aggregate_fidelity(validation, basis, config["gradient_batch_size"])
            curve.append({
                "step": step,
                "train_fidelity": train_value,
                "validation_fidelity": validation_value,
                "euclidean_gradient_norm": euclidean_norm,
                "grassmann_gradient_norm": tangent_norm,
            })
            if validation_value > best_validation:
                best_validation = validation_value
                best_step = step
                best_basis = basis.cpu().clone()
        if step % 20 == 0:
            print(
                f"  lr={learning_rate:g} step={step} "
                f"train={curve[-1]['train_fidelity']:.4f} "
                f"validation={curve[-1]['validation_fidelity']:.4f}",
                flush=True,
            )
    return best_basis, {
        "initial_train_fidelity": initial_train,
        "initial_validation_fidelity": initial_validation,
        "best_validation_fidelity": best_validation,
        "best_step": best_step,
        "curve": curve,
    }


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    axes = build_input_axes(model, config)
    source = load_rows(config["subspace_train"])
    ids = torch.as_tensor(np.asarray(source[:1, :128], dtype=np.int64), device="cuda")
    upstream_error, autograd_error = autograd_check(model, ids, axes, config)

    print("collect train", flush=True)
    train, train_forward, train_contraction = collect_factors(
        model, config["subspace_train"], axes, config)
    print("collect validation", flush=True)
    validation, validation_forward, validation_contraction = collect_factors(
        model, config["subspace_validation"], axes, config)
    evaluations = {}
    maximum_forward = max(upstream_error, train_forward, validation_forward)
    maximum_contraction = max(train_contraction, validation_contraction)
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data, error, contraction = collect_factors(model, spec, axes, config)
        evaluations[corpus] = data
        maximum_forward = max(maximum_forward, error)
        maximum_contraction = max(maximum_contraction, contraction)
    del model
    torch.cuda.empty_cache()

    forward_basis, _ = covariance_basis(train["forward"].flatten(0, 1))
    runs = {}
    selected_basis = None
    selected_learning_rate = None
    selected_validation = -float("inf")
    for learning_rate in config["learning_rates"]:
        print("optimize", learning_rate, flush=True)
        basis, record = optimize_full_batch(
            train, validation, forward_basis, learning_rate, config)
        key = f"{learning_rate:g}"
        runs[key] = record
        if record["best_validation_fidelity"] > selected_validation:
            selected_validation = record["best_validation_fidelity"]
            selected_learning_rate = learning_rate
            selected_basis = basis

    rank = config["subspace_rank"]
    results = {}
    for corpus, data in evaluations.items():
        results[corpus] = {
            "sites": len(data["target"]),
            "forward_pca_fidelity": fidelity(data, forward_basis[:, :rank]),
            "optimized_fidelity": fidelity(data, selected_basis),
        }
    torch.save({
        "selected_basis": selected_basis,
        "selected_learning_rate": selected_learning_rate,
        "forward_pca_basis": forward_basis[:, :rank],
    }, BASIS_OUTPUT)

    best_record = runs[f"{selected_learning_rate:g}"]
    best_curve_row = next(
        row for row in best_record["curve"] if row["step"] == best_record["best_step"])
    exact_error = max(maximum_forward, maximum_contraction)
    hypotheses = {
        "R0_factor_collection_is_exact": exact_error <= 1e-5,
        "R1_full_batch_optimization_improves_validation": (
            selected_validation >= best_record["initial_validation_fidelity"] + .02),
        "R2_optimized_rank64_is_sufficient": all(
            row["optimized_fidelity"] >= .90 for row in results.values()),
        "R3_optimized_rank64_beats_forward_pca": all(
            row["optimized_fidelity"] >= row["forward_pca_fidelity"] + .05
            for row in results.values()),
        "R4_optimized_subspace_transfers_ood": all(
            results[name]["optimized_fidelity"] >= results["fineweb"]["optimized_fidelity"] - .10
            for name in ("wikitext", "tinystories")),
        "R5_training_improvement_is_not_pure_overfit": (
            best_curve_row["train_fidelity"] > best_record["initial_train_fidelity"]
            and selected_validation > best_record["initial_validation_fidelity"]),
        "R6_selected_basis_generalizes_to_fineweb": (
            results["fineweb"]["optimized_fidelity"] >= selected_validation - .05),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["R0_factor_collection_is_exact"].update({
        "autograd_relative_error": autograd_error,
        "maximum_forward_error": maximum_forward,
        "maximum_contraction_error": maximum_contraction,
    })
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "selected_learning_rate": selected_learning_rate,
        "selected_validation_fidelity": selected_validation,
        "optimization": runs,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Deterministic optimizer red-team for the Layer-16 behavior subspace",
        "",
        "Rank-64 projectors start from the forward-PCA span and receive exact "
        "full-training-set gradients of the `J P h` reconstruction error. The "
        "gradient is projected onto the Grassmann tangent space and each update "
        "is retracted by QR. Learning rate and checkpoint are selected only on "
        "the disjoint FineWeb validation panel.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Autograd error: `{autograd_error:.2e}`; maximum contraction error: "
        f"`{maximum_contraction:.2e}`.",
        "",
        "| Learning rate | Initial train | Best-step train | Initial validation | Best validation | Step |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for learning_rate in config["learning_rates"]:
        row = runs[f"{learning_rate:g}"]
        best_row = next(item for item in row["curve"] if item["step"] == row["best_step"])
        lines.append(
            f"| {learning_rate:g} | {row['initial_train_fidelity']:.3f} | "
            f"{best_row['train_fidelity']:.3f} | {row['initial_validation_fidelity']:.3f} | "
            f"{row['best_validation_fidelity']:.3f} | {row['best_step']} |")
    lines += [
        "",
        f"Selected learning rate: `{selected_learning_rate:g}`.",
        "",
        "| Corpus | Forward PCA | Optimized | Difference |",
        "|---|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        lines.append(
            f"| {corpus} | {row['forward_pca_fidelity']:.3f} | "
            f"{row['optimized_fidelity']:.3f} | "
            f"{row['optimized_fidelity'] - row['forward_pca_fidelity']:+.3f} |")
    lines += [
        "",
        "This red-team can distinguish stochastic-optimizer failure from a stable "
        "shared-subspace limitation. It still cannot prove global optimality: the "
        "rank-constrained projector objective is non-convex, and every tested "
        "coordinate is dense and nonsemantic.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
