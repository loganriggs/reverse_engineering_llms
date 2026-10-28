#!/usr/bin/env python3
"""Optimize a shared causal residual subspace directly in the exact J P h contraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_forward_adjoint_atlas import collect_factors, covariance_basis
from run_bilin18_layer16_upstream_tangent_channels import autograd_check, build_input_axes
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_behavior_subspace.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_behavior_subspace.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_BEHAVIOR_SUBSPACE.md"
BASIS_OUTPUT = TENSOR_ROOT / "bilin18_layer16_behavior_subspaces.pt"


@torch.inference_mode()
def prediction(data, basis):
    forward = data["forward"].float().cuda(); effective = data["effective_adjoint"].float().cuda()
    vectors = basis.float().cuda()
    return ((forward @ vectors) * (effective @ vectors)[:, None]).sum(-1)


@torch.inference_mode()
def fidelity(data, basis):
    target = data["target"].float().cuda(); predicted = prediction(data, basis)
    return 1 - float((target - predicted).double().square().sum()
                     / target.double().square().sum().clamp_min(1e-30))


def optimize_basis(train, validation, initial, config, rank):
    basis = torch.nn.Parameter(initial[:, :rank].float().cuda().clone())
    optimizer = torch.optim.Adam([basis], lr=config["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"] + rank)
    initial_validation = fidelity(validation, basis.detach())
    best_validation = initial_validation; best_step = 0; best_basis = basis.detach().cpu().clone()
    curve = [{"step": 0, "validation_fidelity": initial_validation}]
    sample_count = len(train["target"])
    for step in range(1, config["optimization_steps"] + 1):
        index = torch.randint(0, sample_count, (config["optimization_batch_size"],), generator=generator)
        forward = train["forward"][index].float().cuda()
        effective = train["effective_adjoint"][index].float().cuda()
        target = train["target"][index].float().cuda()
        predicted = ((forward @ basis) * (effective @ basis)[:, None]).sum(-1)
        loss = (target - predicted).square().sum() / target.square().sum().clamp_min(1e-30)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad(): basis.copy_(torch.linalg.qr(basis, mode="reduced").Q)
        if step % config["validation_interval"] == 0:
            value = fidelity(validation, basis.detach())
            curve.append({"step": step, "validation_fidelity": value,
                          "minibatch_fidelity": 1 - float(loss.detach())})
            if value > best_validation:
                best_validation = value; best_step = step; best_basis = basis.detach().cpu().clone()
    return best_basis, {"initial_validation_fidelity": initial_validation,
                        "best_validation_fidelity": best_validation,
                        "best_step": best_step, "curve": curve}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    axes = build_input_axes(model, config)
    source = load_rows(config["subspace_train"])
    ids = torch.as_tensor(np.asarray(source[:1, :128], dtype=np.int64), device="cuda")
    upstream_error, autograd_error = autograd_check(model, ids, axes, config)
    print("subspace train", flush=True)
    train, train_forward, train_contraction = collect_factors(model, config["subspace_train"], axes, config)
    print("subspace validation", flush=True)
    validation, validation_forward, validation_contraction = collect_factors(
        model, config["subspace_validation"], axes, config)
    evaluations = {}; maximum_forward = max(upstream_error, train_forward, validation_forward)
    maximum_contraction = max(train_contraction, validation_contraction)
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data, error, contraction = collect_factors(model, spec, axes, config)
        evaluations[corpus] = data; maximum_forward = max(maximum_forward, error)
        maximum_contraction = max(maximum_contraction, contraction)
    del model; torch.cuda.empty_cache()
    forward_basis, _ = covariance_basis(train["forward"].flatten(0, 1))
    optimized = {}; training = {}; results = {corpus: {"sites": len(data["target"]), "ranks": {}}
                                               for corpus, data in evaluations.items()}
    for rank in config["subspace_ranks"]:
        print("optimize rank", rank, flush=True)
        basis, row = optimize_basis(train, validation, forward_basis, config, rank)
        optimized[str(rank)] = basis; training[str(rank)] = row
        for corpus, data in evaluations.items():
            results[corpus]["ranks"][str(rank)] = {
                "forward_pca_fidelity": fidelity(data, forward_basis[:, :rank]),
                "optimized_fidelity": fidelity(data, basis),
            }
    torch.save(optimized, BASIS_OUTPUT)
    exact_error = max(maximum_forward, maximum_contraction)
    hypotheses = {
        "C0_factor_collection_is_exact": exact_error <= 1e-5,
        "C1_behavior_rank64_is_sufficient": all(row["ranks"]["64"]["optimized_fidelity"] >= .90 for row in results.values()),
        "C2_behavior_rank64_beats_forward_pca": all(row["ranks"]["64"]["optimized_fidelity"] >= row["ranks"]["64"]["forward_pca_fidelity"] + .05 for row in results.values()),
        "C3_behavior_rank128_is_high_fidelity": all(row["ranks"]["128"]["optimized_fidelity"] >= .95 for row in results.values()),
        "C4_behavior_subspace_transfers_ood": all(results[name]["ranks"]["64"]["optimized_fidelity"] >= results["fineweb"]["ranks"]["64"]["optimized_fidelity"] - .10 for name in ("wikitext", "tinystories")),
        "C5_behavior_optimization_improves_validation": training["64"]["best_validation_fidelity"] >= training["64"]["initial_validation_fidelity"] + .05,
        "C6_behavior_subspace_generalizes_from_validation": results["fineweb"]["ranks"]["64"]["optimized_fidelity"] >= training["64"]["best_validation_fidelity"] - .05,
        "C7_rank32_behavior_subspace_is_useful": all(row["ranks"]["32"]["optimized_fidelity"] >= .80 for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["C0_factor_collection_is_exact"].update({
        "autograd_relative_error": autograd_error,
        "maximum_forward_error": maximum_forward,
        "maximum_contraction_error": maximum_contraction})
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "training": training, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Behavior-optimized shared residual subspace for the Layer-16 loss path", "",
             "An orthonormal projector is optimized directly inside the exact `J P h` contraction, starting from the forward-PCA span. Checkpoints are selected only by disjoint FineWeb validation. This tests whether PCA's variance objective—not shared-subspace breadth—caused the earlier low fidelity.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Autograd error: `{autograd_error:.2e}`; maximum contraction error: `{maximum_contraction:.2e}`.", "",
              "| Rank | Initial validation | Best validation | Step |", "|---:|---:|---:|---:|"]
    for rank in config["subspace_ranks"]:
        row = training[str(rank)]
        lines.append(f"| {rank} | {row['initial_validation_fidelity']:.3f} | {row['best_validation_fidelity']:.3f} | {row['best_step']} |")
    lines += ["", "| Corpus | Rank | Forward PCA | Behavior optimized |", "|---|---:|---:|---:|"]
    for corpus, row in results.items():
        for rank in config["subspace_ranks"]:
            item = row["ranks"][str(rank)]
            lines.append(f"| {corpus} | {rank} | {item['forward_pca_fidelity']:.3f} | {item['optimized_fidelity']:.3f} |")
    lines += ["", "A successful optimized subspace is a shared causal interface for this loss-path family. It is not automatically semantic or native-sparse: every coordinate remains a dense residual direction, and the site-specific forward tangent and effective adjoint are still required to execute the contraction."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
