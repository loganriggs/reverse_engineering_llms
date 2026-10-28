#!/usr/bin/env python3
"""Separate model-side and target-token contributions to the Layer-16 CE adjoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_mlp_path_type_program import norm
from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_layer16_ce_aware_mode_selection import score
from run_bilin18_layer16_forward_adjoint_atlas import covariance_basis
from run_bilin18_layer16_upstream_tangent_channels import (
    autograd_check,
    build_input_axes,
    propagate_to_layer16,
)
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_ce_adjoint_decomposition.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_ce_adjoint_decomposition.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_CE_ADJOINT_DECOMPOSITION.md"
FACTOR_OUTPUT = TENSOR_ROOT / "bilin18_layer16_ce_adjoint_factors.pt"
COMPONENTS = ("ce", "prediction", "target", "margin")


@torch.inference_mode(False)
def component_gradients(model, row17, position, labels):
    """Return gradients of logsumexp, target logit, CE, and model top-2 margin."""
    block = model.blocks[17]
    base_mixed = row17["mixed"].detach().clone()
    bus = row17["bus"].detach().clone()
    delta = torch.zeros(len(labels), base_mixed.shape[-1], device="cuda", requires_grad=True)
    injection = torch.zeros_like(base_mixed)
    injection[:, position] = delta
    mixed = base_mixed + block.lambdas[0] * injection
    block.attn.rotary.seq_len_cached = None
    block.attn.rotary.cos_cached = None
    block.attn.rotary.sin_cached = None
    attention, _ = block.attn(norm(mixed), bus)
    pre = mixed + attention
    final = norm(pre + block.mlp(norm(pre)))
    raw = model.lm_head(final[:, position])
    logits = 30 * torch.tanh(raw / 30)
    top2 = logits.detach().topk(2, dim=-1).indices

    prediction_objective = torch.logsumexp(logits, dim=-1).sum()
    target_objective = logits.gather(1, labels[:, None]).sum()
    margin_objective = (
        logits.gather(1, top2[:, :1]) - logits.gather(1, top2[:, 1:2])).sum()
    prediction, prediction_output = torch.autograd.grad(
        prediction_objective, (delta, final), retain_graph=True)
    target, target_output = torch.autograd.grad(
        target_objective, (delta, final), retain_graph=True)
    margin, margin_output = torch.autograd.grad(
        margin_objective, (delta, final), retain_graph=True)
    direct_ce, = torch.autograd.grad(F.cross_entropy(logits, labels, reduction="sum"), delta)
    ce = prediction - target
    relative_error = float(
        (ce - direct_ce).double().norm() / direct_ce.double().norm().clamp_min(1e-30))
    return {
        "ce": (ce.detach(), (prediction_output - target_output)[:, position].detach()),
        "prediction": (prediction.detach(), prediction_output[:, position].detach()),
        "target": (target.detach(), target_output[:, position].detach()),
        "margin": (margin.detach(), margin_output[:, position].detach()),
    }, top2.detach(), relative_error


@torch.inference_mode(False)
def collect(model, spec, axes, config):
    source = load_rows(spec)
    stores = {
        "forward": [], "state": [], "labels": [], "top1": [], "top2": [],
        "components": {
            name: {
                "output_covector": [], "write_adjoint": [],
                "effective_adjoint": [], "target": [],
            }
            for name in COMPONENTS
        },
    }
    maximum_forward_error = 0.0
    maximum_gradient_error = 0.0
    maximum_contraction_error = 0.0
    block = model.blocks[16]
    left_weight = block.mlp.Left.weight.float()
    right_weight = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        x, cache = native_cache(model, ids, config)
        by_layer = {row["index"]: row for row in cache}
        for position in config["positions"]:
            forward, error = propagate_to_layer16(model, x, cache, position, axes, config)
            maximum_forward_error = max(maximum_forward_error, error)
            labels = torch.as_tensor(array[:, position + 1].astype(np.int64), device="cuda")
            gradients, top2, gradient_error = component_gradients(
                model, by_layer[17], position, labels)
            maximum_gradient_error = max(maximum_gradient_error, gradient_error)
            state = by_layer[16]["state"][:, position].float()
            left = state @ left_weight.T
            right = state @ right_weight.T
            dleft = torch.einsum("bnd,cd->bnc", forward.cuda(), left_weight)
            dright = torch.einsum("bnd,cd->bnc", forward.cuda(), right_weight)
            direct_write = torch.einsum(
                "bnc,dc->bnd", dleft * right[:, None] + dright * left[:, None], down)

            stores["forward"].append(forward.cpu())
            stores["state"].append(state.cpu())
            stores["labels"].append(labels.cpu())
            stores["top1"].append(top2[:, 0].cpu())
            stores["top2"].append(top2[:, 1].cpu())
            for name, (write_adjoint, output_covector) in gradients.items():
                channel_adjoint = write_adjoint @ down
                effective = (
                    (channel_adjoint * right) @ left_weight
                    + (channel_adjoint * left) @ right_weight)
                contraction = torch.einsum("bnd,bd->bn", forward.cuda(), effective)
                reference = (direct_write * write_adjoint[:, None]).sum(-1)
                relative = float(
                    (contraction - reference).double().norm()
                    / reference.double().norm().clamp_min(1e-30))
                maximum_contraction_error = max(maximum_contraction_error, relative)
                stores["components"][name]["output_covector"].append(output_covector.cpu())
                stores["components"][name]["write_adjoint"].append(write_adjoint.cpu())
                stores["components"][name]["effective_adjoint"].append(effective.cpu())
                stores["components"][name]["target"].append(reference.cpu())
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    result = {
        "forward": torch.cat(stores["forward"]),
        "state": torch.cat(stores["state"]),
        "labels": torch.cat(stores["labels"]),
        "top1": torch.cat(stores["top1"]),
        "top2": torch.cat(stores["top2"]),
        "components": {
            name: {key: torch.cat(rows) for key, rows in values.items()}
            for name, values in stores["components"].items()
        },
    }
    return result, maximum_forward_error, maximum_gradient_error, maximum_contraction_error


@torch.inference_mode()
def projected_fidelity(data, component, basis, rank):
    vectors = basis[:, :rank].float().cuda()
    effective = data["components"][component]["effective_adjoint"].float().cuda()
    prediction = torch.einsum(
        "bnd,bd->bn",
        data["forward"].float().cuda(),
        (effective @ vectors) @ vectors.T,
    ).cpu()
    return score(data["components"][component]["target"], prediction)


def component_geometry(data):
    values = {
        name: data["components"][name]["target"].double().flatten()
        for name in COMPONENTS
    }
    ce_energy = float(values["ce"].square().sum())
    prediction_energy = float(values["prediction"].square().sum())
    target_energy = float(values["target"].square().sum())
    dot = float(values["prediction"] @ values["target"])
    pred_norm = prediction_energy ** .5
    target_norm = target_energy ** .5
    return {
        "prediction_to_ce_energy": prediction_energy / max(ce_energy, 1e-30),
        "target_to_ce_energy": target_energy / max(ce_energy, 1e-30),
        "prediction_target_cosine": dot / max(pred_norm * target_norm, 1e-30),
        "triangle_cancellation_ratio": (pred_norm + target_norm) / max(ce_energy ** .5, 1e-30),
    }


def token_prototypes(discovery):
    labels = discovery["labels"].long()
    effective = discovery["components"]["target"]["effective_adjoint"].double()
    sums = {}
    counts = {}
    for label in torch.unique(labels).tolist():
        mask = labels == label
        sums[int(label)] = effective[mask].sum(0)
        counts[int(label)] = int(mask.sum())
    return sums, counts


@torch.inference_mode()
def discovery_token_audit(discovery, generator):
    sums, counts = token_prototypes(discovery)
    labels = discovery["labels"].long()
    effective = discovery["components"]["target"]["effective_adjoint"].double()
    eligible = torch.tensor([counts[int(label)] >= 2 for label in labels.tolist()])
    indices = eligible.nonzero().flatten()
    loo = torch.stack([
        (sums[int(labels[index])] - effective[index]) / (counts[int(labels[index])] - 1)
        for index in indices.tolist()
    ]).float()
    forward = discovery["forward"][indices].float()
    prediction = torch.einsum("bnd,bd->bn", forward, loo)
    target = discovery["components"]["target"]["target"][indices]
    loo_fidelity = score(target, prediction)

    normalized = effective / effective.norm(dim=1, keepdim=True).clamp_min(1e-30)
    same_cosines = []
    different_cosines = []
    permutation = torch.randperm(len(labels), generator=generator)
    for offset, index in enumerate(indices.tolist()):
        same_mean = loo[offset].double()
        same_mean = same_mean / same_mean.norm().clamp_min(1e-30)
        same_cosines.append(float(normalized[index] @ same_mean))
        chosen = None
        for candidate in permutation.tolist():
            if labels[candidate] != labels[index]:
                chosen = candidate
                break
        different_cosines.append(float(normalized[index] @ normalized[chosen]))
        permutation = torch.roll(permutation, 1)
    return {
        "sites": len(labels),
        "unique_target_tokens": len(counts),
        "repeated_token_sites": len(indices),
        "repeated_token_fraction": float(eligible.float().mean()),
        "leave_one_out_prototype_fidelity": loo_fidelity,
        "mean_same_token_cosine": float(np.mean(same_cosines)),
        "mean_matched_different_token_cosine": float(np.mean(different_cosines)),
        "same_minus_different_cosine": float(np.mean(same_cosines) - np.mean(different_cosines)),
    }


@torch.inference_mode()
def evaluate_token_prototypes(discovery, evaluation):
    sums, counts = token_prototypes(discovery)
    labels = evaluation["labels"].long()
    covered = torch.tensor([int(label) in sums for label in labels.tolist()])
    indices = covered.nonzero().flatten()
    if len(indices):
        prototypes = torch.stack([
            (sums[int(labels[index])] / counts[int(labels[index])]).float()
            for index in indices.tolist()
        ])
        prediction = torch.einsum(
            "bnd,bd->bn", evaluation["forward"][indices].float(), prototypes)
        fidelity = score(
            evaluation["components"]["target"]["target"][indices], prediction)
    else:
        fidelity = float("nan")
    return {
        "covered_sites": len(indices),
        "coverage": float(covered.float().mean()),
        "covered_target_fidelity": fidelity,
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
    source = load_rows(config["decomposition_discovery"])
    ids = torch.as_tensor(np.asarray(source[:1, :128], dtype=np.int64), device="cuda")
    upstream_error, autograd_error = autograd_check(model, ids, axes, config)

    print("collect discovery", flush=True)
    discovery, forward_error, gradient_error, contraction_error = collect(
        model, config["decomposition_discovery"], axes, config)
    evaluations = {}
    maximum_forward = max(upstream_error, forward_error)
    maximum_gradient = gradient_error
    maximum_contraction = contraction_error
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        data, error, grad_error, contraction = collect(model, spec, axes, config)
        evaluations[corpus] = data
        maximum_forward = max(maximum_forward, error)
        maximum_gradient = max(maximum_gradient, grad_error)
        maximum_contraction = max(maximum_contraction, contraction)
    layer16_weights = {
        "left": model.blocks[16].mlp.Left.weight.detach().float().cpu(),
        "right": model.blocks[16].mlp.Right.weight.detach().float().cpu(),
        "down": model.blocks[16].mlp.Down.weight.detach().float().cpu(),
    }
    del model
    torch.cuda.empty_cache()
    torch.save({
        "discovery": discovery,
        "evaluation": evaluations,
        "layer16_weights": layer16_weights,
    }, FACTOR_OUTPUT)

    forward_basis, forward_spectrum = covariance_basis(discovery["forward"].flatten(0, 1))
    bases = {"forward": forward_basis}
    spectra = {"forward": forward_spectrum, "components": {}}
    for component in COMPONENTS:
        output_basis, output_spectrum = covariance_basis(
            discovery["components"][component]["output_covector"])
        write_basis, write_spectrum = covariance_basis(
            discovery["components"][component]["write_adjoint"])
        effective_basis, effective_spectrum = covariance_basis(
            discovery["components"][component]["effective_adjoint"])
        bases[f"{component}_effective"] = effective_basis
        spectra["components"][component] = {
            "output_covector": output_spectrum,
            "write_adjoint": write_spectrum,
            "effective_adjoint": effective_spectrum,
        }

    generator = torch.Generator().manual_seed(config["seed"])
    token_audit = discovery_token_audit(discovery, generator)
    results = {}
    for corpus, data in evaluations.items():
        methods = {}
        for component in COMPONENTS:
            methods[component] = {
                "forward_basis": {
                    str(rank): projected_fidelity(data, component, forward_basis, rank)
                    for rank in config["subspace_ranks"]
                },
                "own_effective_basis": {
                    str(rank): projected_fidelity(
                        data, component, bases[f"{component}_effective"], rank)
                    for rank in config["subspace_ranks"]
                },
            }
        results[corpus] = {
            "sites": len(data["labels"]),
            "geometry": component_geometry(data),
            "methods": methods,
            "target_token_prototypes": evaluate_token_prototypes(discovery, data),
        }

    exact_error = max(maximum_forward, maximum_gradient, maximum_contraction)
    hypotheses = {
        "D0_adjoint_decomposition_is_exact": exact_error <= 1e-5,
        "D1_target_readout_dominates_ce_energy": all(
            row["geometry"]["target_to_ce_energy"]
            >= row["geometry"]["prediction_to_ce_energy"]
            for row in results.values()),
        "D2_prediction_term_has_compact_forward_interface": all(
            row["methods"]["prediction"]["forward_basis"]["64"] >= .90
            for row in results.values()),
        "D3_target_term_is_broad_at_rank64": all(
            row["methods"]["target"]["forward_basis"]["64"] < .90
            for row in results.values()),
        "D4_model_margin_has_compact_forward_interface": all(
            row["methods"]["margin"]["forward_basis"]["64"] >= .90
            for row in results.values()),
        "D5_target_adjoint_is_broader_than_prediction_adjoint": (
            spectra["components"]["target"]["effective_adjoint"]["rank90"]
            >= spectra["components"]["prediction"]["effective_adjoint"]["rank90"] + 32),
        "D6_target_identity_predicts_repeated_target_reads": (
            token_audit["leave_one_out_prototype_fidelity"] >= .50),
        "D7_target_identity_clusters_effective_adjoints": (
            token_audit["same_minus_different_cosine"] >= .20),
        "D8_ce_breadth_is_primarily_target_readout_breadth": all(
            row["methods"]["prediction"]["forward_basis"]["64"]
            >= row["methods"]["ce"]["forward_basis"]["64"] + .10
            and abs(
                row["methods"]["target"]["forward_basis"]["64"]
                - row["methods"]["ce"]["forward_basis"]["64"]
            ) <= .05
            for row in results.values()),
        "D9_target_prototypes_transfer_across_contexts": all(
            row["target_token_prototypes"]["coverage"] >= .25
            and row["target_token_prototypes"]["covered_target_fidelity"] >= .50
            for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["D0_adjoint_decomposition_is_exact"].update({
        "autograd_relative_error": autograd_error,
        "maximum_forward_error": maximum_forward,
        "maximum_gradient_decomposition_error": maximum_gradient,
        "maximum_contraction_error": maximum_contraction,
    })
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "discovery_spectra": spectra,
        "discovery_token_audit": token_audit,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Model-side versus target-token breadth in the Layer-16 CE adjoint",
        "",
        "The exact CE write adjoint is decomposed as `g_CE = g_logsumexp - "
        "g_target`. The first term is determined by the model's probability "
        "distribution; the second selects the dataset-supplied true token. A "
        "top-1-versus-runner-up margin supplies a second model-chosen endpoint. "
        "All terms are contracted through the same exact `J B(x)^T g` tensor path.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Autograd error: `{autograd_error:.2e}`; gradient decomposition error: "
        f"`{maximum_gradient:.2e}`; contraction error: `{maximum_contraction:.2e}`.",
        "",
        "| Component | Output-covector R90 | Write-adjoint R90 | Effective-adjoint R90 |",
        "|---|---:|---:|---:|",
    ]
    for component in COMPONENTS:
        row = spectra["components"][component]
        lines.append(
            f"| {component} | {row['output_covector']['rank90']} | "
            f"{row['write_adjoint']['rank90']} | "
            f"{row['effective_adjoint']['rank90']} |")
    lines += [
        "",
        "| Corpus | Component | Forward basis R32/R64/R128/R256 | Own-adjoint basis R32/R64/R128/R256 |",
        "|---|---|---:|---:|",
    ]
    for corpus, row in results.items():
        for component in COMPONENTS:
            forward_values = row["methods"][component]["forward_basis"]
            own_values = row["methods"][component]["own_effective_basis"]
            ranks = config["subspace_ranks"]
            lines.append(
                f"| {corpus} | {component} | "
                + "/".join(f"{forward_values[str(rank)]:.3f}" for rank in ranks)
                + " | "
                + "/".join(f"{own_values[str(rank)]:.3f}" for rank in ranks)
                + " |")
    lines += [
        "",
        "| Corpus | Prediction/CE energy | Target/CE energy | Pred-target cosine | Cancellation ratio | Prototype coverage/fidelity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        geometry = row["geometry"]
        prototype = row["target_token_prototypes"]
        lines.append(
            f"| {corpus} | {geometry['prediction_to_ce_energy']:.3f} | "
            f"{geometry['target_to_ce_energy']:.3f} | "
            f"{geometry['prediction_target_cosine']:.3f} | "
            f"{geometry['triangle_cancellation_ratio']:.3f} | "
            f"{prototype['coverage']:.3f}/{prototype['covered_target_fidelity']:.3f} |")
    lines += [
        "",
        "Discovery repeated-target audit: "
        f"{token_audit['repeated_token_fraction']:.3f} of sites are covered; "
        f"leave-one-out prototype fidelity is {token_audit['leave_one_out_prototype_fidelity']:.3f}; "
        f"same-minus-different effective-adjoint cosine is "
        f"{token_audit['same_minus_different_cosine']:.3f}.",
        "",
        "This experiment assigns breadth to a behavioral question, not to an "
        "abstract model once and for all. Target-logit breadth describes the "
        "family of token readouts; prediction/margin breadth more directly "
        "describes model-chosen inference interfaces.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
