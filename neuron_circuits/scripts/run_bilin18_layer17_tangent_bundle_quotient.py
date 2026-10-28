#!/usr/bin/env python3
"""Test a role-separated tensor quotient of the final-MLP backward path.

The symmetric quotient used one state-PCA basis for both arguments of the
bilinear derivative.  This program instead treats the natural context, the
incoming tangent, and the outgoing write as distinct tensor roles.  Bases are
fit only on a discovery prefix, frozen, and evaluated on a discovery holdout,
disjoint FineWeb, WikiText, and TinyStories.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_output_conditioned_suffix_atlas import pca_basis
from run_bilin18_layer17_backward_mlp_native_channels import (
    factor_source,
    fidelity,
    layer16_direct_write,
)
from run_bilin18_layer17_backward_mlp_symmetric_tensor import collect_sites
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_tangent_bundle_quotient.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_quotient.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_TANGENT_BUNDLE_QUOTIENT.md")
TENSOR = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_tangent_bundle_quotient.pt")


def rms_jvp(x, tangent):
    root = x.square().mean(-1, keepdim=True).sqrt()
    coupling = (x[:, None] * tangent).mean(-1, keepdim=True)
    return tangent / root[:, None] - x[:, None] * coupling / root[:, None].pow(3)


@torch.inference_mode()
def mlp_jvp(block, state, tangent):
    left_weight = block.mlp.Left.weight.float()
    right_weight = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    left = state @ left_weight.T
    right = state @ right_weight.T
    dleft = torch.einsum("bnd,cd->bnc", tangent, left_weight)
    dright = torch.einsum("bnd,cd->bnc", tangent, right_weight)
    return torch.einsum(
        "bnc,dc->bnd", dleft * right[:, None] + dright * left[:, None], down)


@torch.inference_mode()
def enrich_sites(block, sites, factor_data, spec, weights, batch_size):
    count = len(sites["state"])
    offset = spec.get("factor_start", 0)
    direct_rows, tangent_rows, output_rows = [], [], []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        indices = slice(offset + start, offset + stop)
        direct = layer16_direct_write(factor_data, indices, weights)
        pre = sites["pre"][start:stop].float().cuda()
        state = sites["state"][start:stop].float().cuda()
        tangent = block.lambdas[0].float() * rms_jvp(pre, direct)
        output = mlp_jvp(block, state, tangent)
        direct_rows.append(direct.cpu())
        tangent_rows.append(tangent.cpu())
        output_rows.append(output.cpu())
    return {
        **sites,
        "direct": torch.cat(direct_rows),
        "tangent": torch.cat(tangent_rows),
        "output": torch.cat(output_rows),
    }


def slice_sites(sites, selection):
    return {
        key: ({name: value[selection] for name, value in row.items()}
              if isinstance(row, dict) else row[selection])
        for key, row in sites.items()
    }


def scaled_adjoint_basis(sites, components, rank):
    rows = []
    for component in components:
        values = sites["adjoint"][component].float()
        scale = float((len(values) / values.double().square().sum()).sqrt())
        rows.append(values * scale)
    return pca_basis(torch.cat(rows), rank)


@torch.inference_mode()
def effect_from_bases(block, sites, component, context_basis=None,
                      tangent_basis=None, output_basis=None, batch_size=4):
    left_weight = block.mlp.Left.weight.float()
    right_weight = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    context_basis = None if context_basis is None else context_basis.float().cuda()
    tangent_basis = None if tangent_basis is None else tangent_basis.float().cuda()
    output_basis = None if output_basis is None else output_basis.float().cuda()
    if output_basis is not None:
        output_channel = output_basis.T @ down
    rows = []
    for start in range(0, len(sites["state"]), batch_size):
        stop = min(start + batch_size, len(sites["state"]))
        state = sites["state"][start:stop].float().cuda()
        tangent = sites["tangent"][start:stop].float().cuda()
        adjoint = sites["adjoint"][component][start:stop].float().cuda()
        if context_basis is not None:
            state = (state @ context_basis) @ context_basis.T
        if tangent_basis is not None:
            tangent = (tangent @ tangent_basis) @ tangent_basis.T
        left = state @ left_weight.T
        right = state @ right_weight.T
        dleft = torch.einsum("bnd,cd->bnc", tangent, left_weight)
        dright = torch.einsum("bnd,cd->bnc", tangent, right_weight)
        product = dleft * right[:, None] + dright * left[:, None]
        if output_basis is None:
            channel_scalar = adjoint @ down
        else:
            channel_scalar = (adjoint @ output_basis) @ output_channel
        rows.append(torch.einsum("bnc,bc->bn", product, channel_scalar).cpu())
    return torch.cat(rows)


@torch.inference_mode()
def native_target(sites, paths, component, indices):
    native = torch.einsum(
        "bd,bnd->bn", sites["adjoint"][component].float(), sites["output"].float())
    saved_adjoint = paths[component][indices, 2].float()
    saved = torch.einsum("bd,bnd->bn", saved_adjoint, sites["direct"].float())
    error = float(
        (native.double() - saved.double()).norm()
        / saved.double().norm().clamp_min(1e-30))
    return native, error


def random_basis(dimension, rank, generator):
    return torch.linalg.qr(torch.randn(dimension, rank, generator=generator)).Q


@torch.inference_mode()
def build_role_tensor(block, context_basis, tangent_basis, output_basis):
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    context_basis = context_basis.float().cuda()
    tangent_basis = tangent_basis.float().cuda()
    output_basis = output_basis.float().cuda()
    output_channel = output_basis.T @ down
    left_tangent = left @ tangent_basis
    right_tangent = right @ tangent_basis
    left_context = left @ context_basis
    right_context = right @ context_basis
    tensor = (
        torch.einsum("oc,ci,cj->oij", output_channel, left_tangent, right_context)
        + torch.einsum("oc,ci,cj->oij", output_channel, right_tangent, left_context)
    )
    u, singular, vh = torch.linalg.svd(tensor.flatten(1), full_matrices=False)
    energy = singular.double().square()
    shares = energy / energy.sum().clamp_min(1e-30)
    rank90 = int(torch.searchsorted(
        shares.cumsum(0), torch.tensor(.90, dtype=shares.dtype, device=shares.device)
    ).item() + 1)
    return tensor.cpu(), u.cpu(), singular.cpu(), vh.cpu(), {
        "output_mode_rank90": rank90,
        "top_energy_share": float(shares[0]),
        "leading_energy_shares": shares.cpu().tolist(),
    }


@torch.inference_mode()
def tensor_effect(sites, component, context_basis, tangent_basis,
                  output_basis, tensor, mode_budget=None, batch_size=4):
    context_basis = context_basis.float().cuda()
    tangent_basis = tangent_basis.float().cuda()
    output_basis = output_basis.float().cuda()
    tensor = tensor.float().cuda()
    if mode_budget is not None:
        u, singular, vh = torch.linalg.svd(tensor.flatten(1), full_matrices=False)
        tensor = ((u[:, :mode_budget] * singular[:mode_budget]) @ vh[:mode_budget]).view_as(tensor)
    rows = []
    for start in range(0, len(sites["state"]), batch_size):
        stop = min(start + batch_size, len(sites["state"]))
        context = sites["state"][start:stop].float().cuda() @ context_basis
        tangent = sites["tangent"][start:stop].float().cuda() @ tangent_basis
        adjoint = sites["adjoint"][component][start:stop].float().cuda() @ output_basis
        rows.append(torch.einsum(
            "bo,bni,oij,bj->bn", adjoint, tangent, tensor, context).cpu())
    return torch.cat(rows)


def summary_table_row(results, corpus, condition, rank):
    row = results[corpus]["conditions"][condition][str(rank)]
    return "/".join(f"{row[name]:.3f}" for name in ("prediction", "target", "margin"))


def main():
    config = json.loads(CONFIG.read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    paths = torch.load(config["path_path"], map_location="cpu", weights_only=False)
    weights16 = factors["layer16_weights"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    block = model.blocks[17]

    discovery_spec = config["discovery"]
    discovery_factor = factor_source(factors, discovery_spec)
    discovery = collect_sites(model, discovery_spec, discovery_factor, config)
    discovery = enrich_sites(
        block, discovery, discovery_factor, discovery_spec, weights16,
        config["evaluation_batch_size"])
    fit_sites = config["fit_sites"]
    fit = slice_sites(discovery, slice(0, fit_sites))
    validation = slice_sites(discovery, slice(fit_sites, len(discovery["state"])))
    maximum_rank = max(config["ranks"])
    context_basis = pca_basis(fit["state"], maximum_rank)
    tangent_basis = pca_basis(fit["tangent"].flatten(0, 1), maximum_rank)
    output_basis = pca_basis(fit["output"].flatten(0, 1), maximum_rank)
    adjoint_basis = scaled_adjoint_basis(fit, config["components"], maximum_rank)

    generator = torch.Generator().manual_seed(config["seed"])
    dimension = fit["state"].shape[-1]
    random_bases = [
        (random_basis(dimension, config["random_rank"], generator),
         random_basis(dimension, config["random_rank"], generator),
         random_basis(dimension, config["random_rank"], generator))
        for _ in range(config["random_replicates"])
    ]

    tensor_rank = config["tensor_rank"]
    role_tensor, tensor_u, tensor_s, tensor_vh, tensor_spectrum = build_role_tensor(
        block, context_basis[:, :tensor_rank], tangent_basis[:, :tensor_rank],
        output_basis[:, :tensor_rank])
    torch.save({
        "context_basis": context_basis[:, :tensor_rank],
        "tangent_basis": tangent_basis[:, :tensor_rank],
        "output_basis": output_basis[:, :tensor_rank],
        "tensor": role_tensor,
        "tensor_u": tensor_u,
        "tensor_s": tensor_s,
        "tensor_vh": tensor_vh,
    }, TENSOR)

    datasets = {
        "fit": (fit, "discovery", slice(0, fit_sites)),
        "validation": (validation, "discovery", slice(fit_sites, len(discovery["state"]))),
    }
    for corpus, spec in config["evaluation"].items():
        print("collect", corpus, flush=True)
        factor_data = factor_source(factors, spec)
        sites = collect_sites(model, spec, factor_data, config)
        sites = enrich_sites(
            block, sites, factor_data, spec, weights16,
            config["evaluation_batch_size"])
        offset = spec.get("factor_start", 0)
        datasets[corpus] = (
            sites, spec["factor_dataset"], slice(offset, offset + len(sites["state"])))

    results = {}
    maximum_native_error = 0.0
    maximum_tensor_error = 0.0
    tensor_mode_budgets = [8, 16, 32, 64]
    for corpus, (sites, path_name, path_indices) in datasets.items():
        print("evaluate", corpus, flush=True)
        targets = {}
        for component in config["components"]:
            targets[component], error = native_target(
                sites, paths[path_name], component, path_indices)
            maximum_native_error = max(maximum_native_error, error)
        condition_rows = {
            name: {str(rank): {} for rank in config["ranks"]}
            for name in ("output_only", "context_only", "tangent_only",
                         "role_separated", "shared_state_pca", "swapped_roles")
        }
        for rank in config["ranks"]:
            cb = context_basis[:, :rank]
            tb = tangent_basis[:, :rank]
            ob = output_basis[:, :rank]
            ab = adjoint_basis[:, :rank]
            for component in config["components"]:
                target = targets[component]
                candidates = {
                    "output_only": effect_from_bases(
                        block, sites, component, output_basis=ob,
                        batch_size=config["evaluation_batch_size"]),
                    "context_only": effect_from_bases(
                        block, sites, component, context_basis=cb,
                        batch_size=config["evaluation_batch_size"]),
                    "tangent_only": effect_from_bases(
                        block, sites, component, tangent_basis=tb,
                        batch_size=config["evaluation_batch_size"]),
                    "role_separated": effect_from_bases(
                        block, sites, component, cb, tb, ob,
                        config["evaluation_batch_size"]),
                    "shared_state_pca": effect_from_bases(
                        block, sites, component, cb, cb, ab,
                        config["evaluation_batch_size"]),
                    "swapped_roles": effect_from_bases(
                        block, sites, component, tb, cb, ob,
                        config["evaluation_batch_size"]),
                }
                for name, prediction in candidates.items():
                    condition_rows[name][str(rank)][component] = fidelity(target, prediction)

        random_rows = {component: [] for component in config["components"]}
        for cb, tb, ob in random_bases:
            for component in config["components"]:
                prediction = effect_from_bases(
                    block, sites, component, cb, tb, ob,
                    config["evaluation_batch_size"])
                random_rows[component].append(fidelity(targets[component], prediction))
        random_summary = {
            component: {
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, .95)),
                "maximum": max(values),
            } for component, values in random_rows.items()
        }

        tensor_rows = {}
        for component in config["components"]:
            direct = effect_from_bases(
                block, sites, component, context_basis[:, :tensor_rank],
                tangent_basis[:, :tensor_rank], output_basis[:, :tensor_rank],
                config["evaluation_batch_size"])
            exact = tensor_effect(
                sites, component, context_basis[:, :tensor_rank],
                tangent_basis[:, :tensor_rank], output_basis[:, :tensor_rank],
                role_tensor, batch_size=config["evaluation_batch_size"])
            error = float(
                (direct.double() - exact.double()).norm()
                / direct.double().norm().clamp_min(1e-30))
            maximum_tensor_error = max(maximum_tensor_error, error)
            tensor_rows[component] = {
                "exact_error": error,
                "native_fidelity": fidelity(targets[component], exact),
                "mode_fidelity": {
                    str(budget): fidelity(
                        targets[component],
                        tensor_effect(
                            sites, component, context_basis[:, :tensor_rank],
                            tangent_basis[:, :tensor_rank], output_basis[:, :tensor_rank],
                            role_tensor, mode_budget=budget,
                            batch_size=config["evaluation_batch_size"]),
                    ) for budget in tensor_mode_budgets
                },
            }
        results[corpus] = {
            "sites": len(sites["state"]),
            "conditions": condition_rows,
            "random_rank64": random_summary,
            "tensor_rank64": tensor_rows,
        }

    eval_names = ("fineweb", "wikitext", "tinystories")
    role64 = lambda corpus, component: results[corpus]["conditions"]["role_separated"]["64"][component]
    hypotheses = {
        "T0_analytic_jvp_recomposes_native_path": maximum_native_error <= 1e-5,
        "T1_output_write_basis_is_compact": all(
            results[corpus]["conditions"]["output_only"]["64"][component] >= .95
            for corpus in eval_names for component in config["components"]),
        "T2_role_separated_rank64_recovers_scores": all(
            role64(corpus, component) >= .80 for corpus in eval_names
            for component in ("prediction", "target")),
        "T3_role_separated_rank64_recovers_margin": all(
            role64(corpus, "margin") >= .70 for corpus in eval_names),
        "T4_role_separated_rank128_is_sufficient": all(
            results[corpus]["conditions"]["role_separated"]["128"][component]
            >= (.80 if component == "margin" else .90)
            for corpus in eval_names for component in config["components"]),
        "T5_role_separation_beats_shared_pca": all(
            role64(corpus, component)
            - results[corpus]["conditions"]["shared_state_pca"]["64"][component]
            >= (.30 if component == "margin" else .20)
            for corpus in eval_names for component in config["components"]),
        "T6_role_assignment_matters": all(
            np.mean([role64(corpus, component) for component in config["components"]])
            - np.mean([
                results[corpus]["conditions"]["swapped_roles"]["64"][component]
                for component in config["components"]]) >= .10
            for corpus in eval_names),
        "T7_role_quotient_beats_random": all(
            role64(corpus, component)
            >= results[corpus]["random_rank64"][component]["p95"] + .20
            for corpus in eval_names for component in config["components"]),
        "T8_role_quotient_transfers_ood": all(
            role64(corpus, component) >= role64("fineweb", component) - .15
            for corpus in ("wikitext", "tinystories")
            for component in config["components"]),
        "T9_contracted_tensor_is_exact": maximum_tensor_error <= 1e-5,
        "T10_validation_split_is_not_fit": (
            len(fit["state"]) == config["fit_sites"]
            and len(validation["state"]) + len(fit["state"]) == len(discovery["state"])),
    }
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "maximum_native_recomposition_error": maximum_native_error,
        "maximum_tensor_contraction_error": maximum_tensor_error,
        "basis_fit_sites": len(fit["state"]),
        "basis_validation_sites": len(validation["state"]),
        "tensor_spectrum": tensor_spectrum,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Role-separated tangent-bundle quotient of the final MLP",
        "",
        "The natural state, incoming Layer-16 tangent, and outgoing final-MLP write are treated as distinct tensor roles. Uncentered PCA bases are fit on 192 FineWeb discovery sites, frozen, and tested on a 64-site discovery holdout plus disjoint FineWeb, WikiText, and TinyStories. All effects use the exact original factors and the untouched endpoint adjoints.",
        "", "## Locked hypotheses", "",
    ]
    for name, statement in config["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if hypotheses[name] else 'not supported'}.** {statement}")
    lines += [
        "", f"Maximum analytic-JVP/native-path error: `{maximum_native_error:.2e}`.",
        f"Maximum contracted-tensor/projected-CP error: `{maximum_tensor_error:.2e}`.",
        f"Rank-64 tensor output-mode energy R90: `{tensor_spectrum['output_mode_rank90']}`; top share: `{tensor_spectrum['top_energy_share']:.3f}`.",
        "", "## Quotient fidelity", "",
        "Each cell is prediction/target/margin fidelity relative to the complete native final-MLP direct path.",
        "", "| Corpus | Condition | rank 32 | rank 64 | rank 128 |", "|---|---|---:|---:|---:|",
    ]
    for corpus in results:
        for condition in ("output_only", "context_only", "tangent_only",
                          "role_separated", "shared_state_pca", "swapped_roles"):
            lines.append(
                f"| {corpus} | {condition} | {summary_table_row(results, corpus, condition, 32)} | "
                f"{summary_table_row(results, corpus, condition, 64)} | "
                f"{summary_table_row(results, corpus, condition, 128)} |")
    lines += ["", "## Random rank-64 controls", "",
              "| Corpus | Endpoint | median | p95 | maximum |", "|---|---|---:|---:|---:|"]
    for corpus, row in results.items():
        for component, values in row["random_rank64"].items():
            lines.append(
                f"| {corpus} | {component} | {values['median']:.3f} | {values['p95']:.3f} | {values['maximum']:.3f} |")
    lines += ["", "## Tensor modes", "",
              "| Corpus | Endpoint | exact rank-64 | modes 8/16/32/64 |", "|---|---|---:|---:|"]
    for corpus, row in results.items():
        for component, values in row["tensor_rank64"].items():
            modes = "/".join(f"{values['mode_fidelity'][str(k)]:.3f}" for k in tensor_mode_budgets)
            lines.append(f"| {corpus} | {component} | {values['native_fidelity']:.3f} | {modes} |")
    lines += [
        "", "The role-separated quotient is an existence and transfer test, not yet a semantic interpretation. Natural-write and tangent bases use causal path data on the discovery split. If they transfer, the next question is whether weight-only or cheaply observable proxies recover the same tensor roles.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "hypotheses": hypotheses,
        "maximum_native_error": maximum_native_error,
        "maximum_tensor_error": maximum_tensor_error,
        "role64": {
            corpus: results[corpus]["conditions"]["role_separated"]["64"]
            for corpus in results
        },
    }, indent=2))


if __name__ == "__main__":
    main()
