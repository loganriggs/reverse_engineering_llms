#!/usr/bin/env python3
"""Audit native-channel execution inside the essential final-MLP backward path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin12_mlp_path_type_program import norm
from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_backward_mlp_native_channels.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_native_channels.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_BACKWARD_MLP_NATIVE_CHANNELS.md")
COMPONENTS = ("prediction", "target", "margin")


def fidelity(target, prediction):
    return 1.0 - float(
        (target.double() - prediction.double()).square().sum()
        / target.double().square().sum().clamp_min(1e-30))


def rms_vjp(x, adjoint):
    root = x.square().mean(-1, keepdim=True).sqrt()
    coupling = (x[:, None] * adjoint).mean(-1, keepdim=True)
    return adjoint / root[:, None] - x[:, None] * coupling / root[:, None].pow(3)


@torch.inference_mode(False)
def final_mlp_channel_adjoints(
        block, row17, position, covectors, lm_head=None, token_indices=None):
    mixed = row17["mixed"].detach().clone().requires_grad_(True)
    bus = row17["bus"].detach().clone()
    block.attn.rotary.seq_len_cached = None
    block.attn.rotary.cos_cached = None
    block.attn.rotary.sin_cached = None
    attention, _ = block.attn(norm(mixed), bus)
    pre = mixed + attention
    state = norm(pre)
    left = block.mlp.Left(state)
    right = block.mlp.Right(state)
    write = block.mlp.Down(left * right) + block.mlp.Down_bias
    residual = pre + write
    final = norm(residual)
    covectors = dict(covectors)
    if token_indices:
        if lm_head is None:
            raise ValueError("lm_head is required for token-index covectors")
        raw = lm_head(final[:, position]).float()
        for name, indices in token_indices.items():
            indices = indices.cuda()
            selected_raw = raw.gather(1, indices[:, None]).squeeze(1)
            cap_derivative = 1 - torch.tanh(selected_raw / 30).square()
            covectors[name] = cap_derivative[:, None] * lm_head.weight.float()[indices]
    results = {}
    for component, covector in covectors.items():
        full_covector = torch.zeros_like(final)
        full_covector[:, position] = covector.cuda()
        residual_adjoint, = torch.autograd.grad(
            final, residual, full_covector, retain_graph=True)
        output_adjoint = residual_adjoint[:, position]
        channel_scalar = output_adjoint @ block.mlp.Down.weight.float()
        left_position = left[:, position].float()
        right_position = right[:, position].float()
        state_adjoint = channel_scalar[:, :, None] * (
            right_position[:, :, None] * block.mlp.Left.weight.float()[None]
            + left_position[:, :, None] * block.mlp.Right.weight.float()[None])
        pre_adjoint = rms_vjp(pre[:, position].float(), state_adjoint)
        results[component] = (
            block.lambdas[0].float() * pre_adjoint).detach()
    return results


@torch.inference_mode()
def layer16_direct_write(factor_data, indices, weights):
    forward = factor_data["forward"][indices].float().cuda()
    state = factor_data["state"][indices].float().cuda()
    left_weight = weights["left"].float().cuda()
    right_weight = weights["right"].float().cuda()
    down = weights["down"].float().cuda()
    left = state @ left_weight.T
    right = state @ right_weight.T
    dleft = torch.einsum("bnd,cd->bnc", forward, left_weight)
    dright = torch.einsum("bnd,cd->bnc", forward, right_weight)
    return torch.einsum(
        "bnc,dc->bnd", dleft * right[:, None] + dright * left[:, None], down)


def factor_source(factors, spec):
    if spec["factor_dataset"] == "discovery":
        return factors["discovery"]
    return factors["evaluation"][spec["factor_dataset"]]


def discovery_scores(model, spec, factor_data, path_data, weights, config):
    source = load_rows(spec)
    channel_count = model.blocks[17].mlp.Left.weight.shape[0]
    dot = {component: torch.zeros(channel_count, dtype=torch.float64) for component in COMPONENTS}
    square = {component: torch.zeros(channel_count, dtype=torch.float64) for component in COMPONENTS}
    target_energy = {component: 0.0 for component in COMPONENTS}
    maximum_error = 0.0
    offset = spec.get("factor_start", 0)
    consumed = 0
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        _, cache = native_cache(model, ids, config)
        row17 = {row["index"]: row for row in cache}[17]
        for position in config["positions"]:
            indices = slice(offset + consumed, offset + consumed + len(ids))
            direct_write = layer16_direct_write(factor_data, indices, weights)
            covectors = {
                component: factor_data["components"][component]["output_covector"][indices]
                for component in COMPONENTS
            }
            channel_adjoints = final_mlp_channel_adjoints(
                model.blocks[17], row17, position, covectors)
            for component in COMPONENTS:
                contribution = torch.einsum(
                    "bcd,bnd->bcn", channel_adjoints[component], direct_write).cpu()
                reference_adjoint = path_data[component][indices, 2].float().cuda()
                target = torch.einsum("bd,bnd->bn", reference_adjoint, direct_write).cpu()
                reconstruction = contribution.sum(1)
                error = float(
                    (target.double() - reconstruction.double()).norm()
                    / target.double().norm().clamp_min(1e-30))
                maximum_error = max(maximum_error, error)
                dot[component] += torch.einsum(
                    "bcn,bn->c", contribution.double(), target.double())
                square[component] += contribution.double().square().sum((0, 2))
                target_energy[component] += float(target.double().square().sum())
            consumed += len(ids)
        print(f"  discovery rows {start + len(ids)}/{len(source)}", flush=True)
    score = {
        component: 2 * dot[component] - square[component]
        for component in COMPONENTS
    }
    shared_score = sum(
        score[component] / max(target_energy[component], 1e-30)
        for component in config["fit_components"])
    ordering = {
        "shared": torch.argsort(shared_score, descending=True),
        **{
            component: torch.argsort(score[component], descending=True)
            for component in COMPONENTS
        },
    }
    return ordering, maximum_error


def support_summary(contribution, target):
    norms = contribution.double().square().sum(-1).sqrt()
    norm_support = norms.sum(-1).square() / norms.square().sum(-1).clamp_min(1e-30)
    projection = torch.einsum(
        "bcn,bn->bc", contribution.double(), target.double())
    projection = projection / target.double().square().sum(-1, keepdim=True).clamp_min(1e-30)
    absolute_mass = projection.abs().sum(-1)
    projection_support = absolute_mass.square() / projection.square().sum(-1).clamp_min(1e-30)
    return norm_support, projection_support, absolute_mass


def summarize(values):
    array = np.asarray(values)
    return {
        "p10": float(np.quantile(array, .10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, .90)),
    }


def evaluate_dataset(model, spec, factor_data, path_data, weights, ordering, random_sets, config):
    source = load_rows(spec)
    budgets = config["budgets"]
    stores = {
        component: {
            "target": [],
            "shared": {budget: [] for budget in budgets},
            "specific": {budget: [] for budget in budgets},
            "random": [[] for _ in random_sets],
            "norm_support": [], "projection_support": [], "absolute_mass": [],
        } for component in COMPONENTS
    }
    maximum_error = 0.0
    offset = spec.get("factor_start", 0)
    consumed = 0
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        _, cache = native_cache(model, ids, config)
        row17 = {row["index"]: row for row in cache}[17]
        for position in config["positions"]:
            indices = slice(offset + consumed, offset + consumed + len(ids))
            direct_write = layer16_direct_write(factor_data, indices, weights)
            covectors = {
                component: factor_data["components"][component]["output_covector"][indices]
                for component in COMPONENTS
            }
            channel_adjoints = final_mlp_channel_adjoints(
                model.blocks[17], row17, position, covectors)
            for component in COMPONENTS:
                contribution = torch.einsum(
                    "bcd,bnd->bcn", channel_adjoints[component], direct_write).cpu()
                reference_adjoint = path_data[component][indices, 2].float().cuda()
                target = torch.einsum("bd,bnd->bn", reference_adjoint, direct_write).cpu()
                error = float(
                    (target.double() - contribution.sum(1).double()).norm()
                    / target.double().norm().clamp_min(1e-30))
                maximum_error = max(maximum_error, error)
                row = stores[component]
                row["target"].append(target)
                for budget in budgets:
                    row["shared"][budget].append(
                        contribution[:, ordering["shared"][:budget]].sum(1))
                    row["specific"][budget].append(
                        contribution[:, ordering[component][:budget]].sum(1))
                for random_index, selected in enumerate(random_sets):
                    row["random"][random_index].append(
                        contribution[:, selected].sum(1))
                norm_support, projection_support, absolute_mass = support_summary(
                    contribution, target)
                row["norm_support"].extend(norm_support.tolist())
                row["projection_support"].extend(projection_support.tolist())
                row["absolute_mass"].extend(absolute_mass.tolist())
            consumed += len(ids)
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)

    results = {}
    for component, row in stores.items():
        target = torch.cat(row["target"])
        shared = {
            str(budget): fidelity(target, torch.cat(row["shared"][budget]))
            for budget in budgets
        }
        specific = {
            str(budget): fidelity(target, torch.cat(row["specific"][budget]))
            for budget in budgets
        }
        random_values = [
            fidelity(target, torch.cat(predictions)) for predictions in row["random"]
        ]
        k90_shared = next((budget for budget in budgets if shared[str(budget)] >= .90), None)
        k90_specific = next((budget for budget in budgets if specific[str(budget)] >= .90), None)
        results[component] = {
            "shared_fidelity": shared,
            "specific_fidelity": specific,
            "shared_k90": k90_shared,
            "specific_k90": k90_specific,
            "random_512": {
                "median": float(np.median(random_values)),
                "p95": float(np.quantile(random_values, .95)),
                "maximum": max(random_values),
            },
            "norm_effective_support": summarize(row["norm_support"]),
            "projection_effective_support": summarize(row["projection_support"]),
            "absolute_projection_mass": summarize(row["absolute_mass"]),
        }
    return results, maximum_error


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    paths = torch.load(config["path_path"], map_location="cpu", weights_only=False)
    weights = factors["layer16_weights"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    discovery_spec = config["selection_discovery"]
    discovery_data = factor_source(factors, discovery_spec)
    print("select channels", flush=True)
    ordering, discovery_error = discovery_scores(
        model, discovery_spec, discovery_data,
        paths[discovery_spec["factor_dataset"]], weights, config)
    generator = torch.Generator().manual_seed(config["seed"])
    channel_count = model.blocks[17].mlp.Left.weight.shape[0]
    random_sets = [
        torch.randperm(channel_count, generator=generator)[:config["random_budget"]]
        for _ in range(config["random_replicates"])
    ]

    results = {}
    maximum_error = discovery_error
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = factor_source(factors, spec)
        rows, error = evaluate_dataset(
            model, spec, data, paths[spec["factor_dataset"]],
            weights, ordering, random_sets, config)
        results[corpus] = {"sites": spec["rows"] * len(config["positions"]),
                           "components": rows}
        maximum_error = max(maximum_error, error)
    del model
    torch.cuda.empty_cache()

    tested = config["fit_components"]
    hypotheses = {
        "N0_native_channels_recompose_final_mlp_path": maximum_error <= 1e-5,
        "N1_shared_512_channel_bank_is_sufficient": all(
            row["components"][component]["shared_fidelity"]["512"] >= .90
            for row in results.values() for component in tested),
        "N2_shared_2048_channel_bank_is_sufficient": all(
            row["components"][component]["shared_fidelity"]["2048"] >= .90
            for row in results.values() for component in tested),
        "N3_endpoint_specific_512_banks_are_sufficient": all(
            row["components"][component]["specific_fidelity"]["512"] >= .90
            for row in results.values() for component in tested),
        "N4_individual_paths_are_native_dense": all(
            row["components"][component]["norm_effective_support"]["median"] >= 512
            for row in results.values() for component in tested),
        "N5_native_path_uses_signed_cancellation": all(
            row["components"][component]["absolute_projection_mass"]["median"] >= 2
            for row in results.values() for component in tested),
        "N6_shared_bank_beats_random": all(
            row["components"][component]["shared_fidelity"]["512"]
            >= row["components"][component]["random_512"]["p95"] + .10
            for row in results.values() for component in tested),
        "N7_shared_native_bank_transfers_ood": all(
            results[name]["components"][component]["shared_fidelity"]["1024"]
            >= results["fineweb"]["components"][component]["shared_fidelity"]["1024"] - .10
            for name in ("wikitext", "tinystories") for component in tested),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["N0_native_channels_recompose_final_mlp_path"]["maximum_relative_error"] = maximum_error
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "shared_ordering": ordering["shared"].tolist(),
        "endpoint_orderings": {component: ordering[component].tolist() for component in COMPONENTS},
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Native channels inside the essential final-MLP backward path",
        "",
        "Each final-layer bilinear channel's exact backward contribution is "
        "contracted into the existing 32D Layer-12 behavioral tangent. Shared and "
        "endpoint-specific marginal rankings are frozen on FineWeb discovery and "
        "evaluated on disjoint FineWeb, WikiText, and TinyStories.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Maximum all-channel behavioral recomposition error: `{maximum_error:.2e}`.",
        "",
        "| Corpus | Component | Shared K32/K128/K512/K1024/K2048/K4096 | Shared K90 | Specific K90 | Random512 p95 | Median norm support | Median projection support | Median absolute projection mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    shown = (32, 128, 512, 1024, 2048, 4096)
    for corpus, row in results.items():
        for component in COMPONENTS:
            item = row["components"][component]
            values = "/".join(
                f"{item['shared_fidelity'][str(budget)]:.3f}" for budget in shown)
            lines.append(
                f"| {corpus} | {component} | {values} | "
                f"{item['shared_k90']} | {item['specific_k90']} | "
                f"{item['random_512']['p95']:.3f} | "
                f"{item['norm_effective_support']['median']:.0f} | "
                f"{item['projection_effective_support']['median']:.0f} | "
                f"{item['absolute_projection_mass']['median']:.2f} |")
    lines += [
        "",
        "Marginal rankings are stable fixed-bank tests, not globally optimal subset "
        "solutions. Broad native support would establish dense execution in the "
        "trained CP basis while leaving open a compact symmetry-aware tensor "
        "refactorization of the same quadratic form.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
