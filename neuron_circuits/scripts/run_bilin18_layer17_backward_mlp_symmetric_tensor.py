#!/usr/bin/env python3
"""Canonicalize the final-MLP backward path as a symmetric quadratic tensor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin12_mlp_path_type_program import norm
from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_layer16_output_conditioned_suffix_atlas import pca_basis
from run_bilin18_layer17_backward_mlp_native_channels import (
    factor_source,
    fidelity,
    layer16_direct_write,
)
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_layer17_backward_mlp_symmetric_tensor.json")
OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_symmetric_tensor.json")
REPORT = Path("artifacts/bilin18_tensor_decomposition/BILIN18_LAYER17_BACKWARD_MLP_SYMMETRIC_TENSOR.md")
TENSOR_OUTPUT = Path("artifacts/bilin18_tensor_decomposition/bilin18_layer17_backward_mlp_symmetric_tensor.pt")


def rms_vjp(x, adjoint):
    root = x.square().mean(-1, keepdim=True).sqrt()
    coupling = (x * adjoint).mean(-1, keepdim=True)
    return adjoint / root - x * coupling / root.pow(3)


@torch.inference_mode()
def collect_sites(model, spec, factor_data, config):
    source = load_rows(spec)
    block = model.blocks[17]
    stores = {
        "pre": [], "state": [],
        "adjoint": {component: [] for component in config["components"]},
    }
    offset = spec.get("factor_start", 0)
    consumed = 0
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        _, cache = native_cache(model, ids, config)
        row17 = {row["index"]: row for row in cache}[17]
        pre = row17["pre"].float()
        state = norm(pre)
        residual = pre + block.mlp(state)
        for position in config["positions"]:
            indices = slice(offset + consumed, offset + consumed + len(ids))
            stores["pre"].append(pre[:, position].cpu())
            stores["state"].append(state[:, position].cpu())
            for component in config["components"]:
                q = factor_data["components"][component]["output_covector"][indices].float().cuda()
                stores["adjoint"][component].append(
                    rms_vjp(residual[:, position], q).cpu())
            consumed += len(ids)
    return {
        "pre": torch.cat(stores["pre"]),
        "state": torch.cat(stores["state"]),
        "adjoint": {
            component: torch.cat(values)
            for component, values in stores["adjoint"].items()
        },
    }


@torch.inference_mode()
def build_tensor(block, input_basis, output_basis):
    left = block.mlp.Left.weight.float() @ input_basis.cuda()
    right = block.mlp.Right.weight.float() @ input_basis.cuda()
    output_channel = output_basis.float().cuda().T @ block.mlp.Down.weight.float()
    tensor = torch.einsum("kc,ci,cj->kij", output_channel, left, right)
    tensor = .5 * (tensor + tensor.transpose(-1, -2))
    u, singular, vh = torch.linalg.svd(tensor.flatten(1), full_matrices=False)
    energy = singular.double().square()
    shares = energy / energy.sum().clamp_min(1e-30)
    rank90 = int(torch.searchsorted(
        shares.cumsum(0), torch.tensor(.90, dtype=shares.dtype, device=shares.device)).item() + 1)
    return tensor.cpu(), u.cpu(), singular.cpu(), vh.cpu(), {
        "rank90": rank90,
        "top_energy_share": float(shares[0]),
        "leading_energy_shares": shares[:64].cpu().tolist(),
    }


@torch.inference_mode()
def execute(
        block, site_data, factor_data, path_data, weights16,
        input_basis, output_basis, tensor, tensor_svd, spec, config):
    offset = spec.get("factor_start", 0)
    count = spec["rows"] * len(config["positions"])
    indices = slice(offset, offset + count)
    direct_write = layer16_direct_write(factor_data, indices, weights16)
    input_basis = input_basis.float().cuda()
    output_basis = output_basis.float().cuda()
    pre = site_data["pre"].float().cuda()
    state = site_data["state"].float().cuda()
    state_coefficients = state @ input_basis
    projected_state = state_coefficients @ input_basis.T
    u, singular, vh = (value.float().cuda() for value in tensor_svd)
    tensor_flat = tensor.float().cuda().flatten(1)
    left_weight = block.mlp.Left.weight.float()
    right_weight = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    results = {}
    maximum_projected_error = 0.0
    for component in config["components"]:
        adjoint = site_data["adjoint"][component].float().cuda()
        adjoint_coefficients = adjoint @ output_basis
        projected_adjoint = adjoint_coefficients @ output_basis.T
        channel_scalar = projected_adjoint @ down
        left = projected_state @ left_weight.T
        right = projected_state @ right_weight.T
        direct_state_adjoint = (
            (channel_scalar * right) @ left_weight
            + (channel_scalar * left) @ right_weight)
        # The contracted quadratic tensor represents U^T Q U, so its output
        # gradient is also in span(U). Apply the identical output projection in
        # the native-CP exactness control.
        direct_state_adjoint = (
            direct_state_adjoint @ input_basis) @ input_basis.T
        direct_pre_adjoint = block.lambdas[0].float() * rms_vjp(pre, direct_state_adjoint)
        direct_effect = torch.einsum("bd,bnd->bn", direct_pre_adjoint, direct_write)

        exact_matrices = (adjoint_coefficients @ tensor_flat).view(
            len(pre), config["input_dimension"], config["input_dimension"])
        exact_gradient_coordinates = 2 * torch.einsum(
            "bij,bj->bi", exact_matrices, state_coefficients)
        exact_state_adjoint = exact_gradient_coordinates @ input_basis.T
        exact_pre_adjoint = block.lambdas[0].float() * rms_vjp(pre, exact_state_adjoint)
        exact_effect = torch.einsum("bd,bnd->bn", exact_pre_adjoint, direct_write)
        exact_error = float(
            (exact_effect - direct_effect).double().norm()
            / direct_effect.double().norm().clamp_min(1e-30))
        maximum_projected_error = max(maximum_projected_error, exact_error)

        mode_results = {}
        for budget in config["mode_budgets"]:
            mode_coefficients = (adjoint_coefficients @ u[:, :budget]) * singular[:budget]
            matrices = (mode_coefficients @ vh[:budget]).view(
                len(pre), config["input_dimension"], config["input_dimension"])
            gradient_coordinates = 2 * torch.einsum(
                "bij,bj->bi", matrices, state_coefficients)
            state_adjoint = gradient_coordinates @ input_basis.T
            pre_adjoint = block.lambdas[0].float() * rms_vjp(pre, state_adjoint)
            effect = torch.einsum("bd,bnd->bn", pre_adjoint, direct_write)
            mode_results[str(budget)] = effect.cpu()
        native_adjoint = path_data[component][indices, 2].float().cuda()
        native_effect = torch.einsum("bd,bnd->bn", native_adjoint, direct_write).cpu()
        results[component] = {
            "native": native_effect,
            "projected_direct": exact_effect.cpu(),
            "modes": mode_results,
        }
    return results, maximum_projected_error


def main(config_path=CONFIG, output_path=OUTPUT, report_path=REPORT,
         tensor_output=TENSOR_OUTPUT):
    config = json.loads(Path(config_path).read_text())
    factors = torch.load(config["factor_path"], map_location="cpu", weights_only=False)
    paths = torch.load(config["path_path"], map_location="cpu", weights_only=False)
    native_result = json.loads(Path(config["native_result_path"]).read_text())
    weights16 = factors["layer16_weights"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()

    discovery_spec = config["discovery"]
    discovery_factor = factor_source(factors, discovery_spec)
    discovery = collect_sites(model, discovery_spec, discovery_factor, config)
    input_basis = pca_basis(discovery["state"], config["input_dimension"])
    scaled_adjoints = []
    component_scales = {}
    for component in config["components"]:
        values = discovery["adjoint"][component].float()
        scale = float((len(values) / values.double().square().sum()).sqrt())
        component_scales[component] = scale
        scaled_adjoints.append(values * scale)
    output_basis = pca_basis(torch.cat(scaled_adjoints), config["output_dimension"])
    tensor, tensor_u, tensor_s, tensor_vh, spectrum = build_tensor(
        model.blocks[17], input_basis, output_basis)
    tensor_svd = (tensor_u, tensor_s, tensor_vh)

    results = {}
    maximum_projected_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        factor_data = factor_source(factors, spec)
        sites = collect_sites(model, spec, factor_data, config)
        execution, error = execute(
            model.blocks[17], sites, factor_data, paths[spec["factor_dataset"]],
            weights16, input_basis, output_basis, tensor, tensor_svd, spec, config)
        maximum_projected_error = max(maximum_projected_error, error)
        component_rows = {}
        for component, row in execution.items():
            native = row["native"]
            component_rows[component] = {
                "projected_direct_fidelity": fidelity(native, row["projected_direct"]),
                "mode_fidelity": {
                    budget: fidelity(native, prediction)
                    for budget, prediction in row["modes"].items()
                },
                "within_quotient_mode_fidelity": {
                    budget: fidelity(row["projected_direct"], prediction)
                    for budget, prediction in row["modes"].items()
                },
            }
        results[corpus] = {
            "sites": spec["rows"] * len(config["positions"]),
            "components": component_rows,
        }
    del model
    torch.cuda.empty_cache()

    if "rank64_result_path" in config:
        rank64_result = json.loads(Path(config["rank64_result_path"]).read_text())
        hypotheses = {
            "Q0_symmetric_tensor_matches_projected_native_cp": maximum_projected_error <= 1e-5,
            "Q1_rank128_quotient_improves_over_rank64": all(
                row["components"][component]["projected_direct_fidelity"]
                >= rank64_result["results"][corpus]["components"][component]["projected_direct_fidelity"] + .20
                for corpus, row in results.items() for component in config["components"]),
            "Q2_rank128_input_output_quotient_is_useful": all(
                row["components"][component]["projected_direct_fidelity"] >= .80
                for row in results.values() for component in config["components"]),
            "Q3_sixtyfour_tensor_modes_are_useful": all(
                row["components"][component]["mode_fidelity"]["64"] >= .80
                for row in results.values() for component in config["components"]),
            "Q4_sixtyfour_modes_preserve_the_quotient": all(
                row["components"][component]["within_quotient_mode_fidelity"]["64"] >= .95
                for row in results.values() for component in config["components"]),
            "Q5_symmetric_tensor_has_energy_rank64": spectrum["rank90"] <= 64,
            "Q6_rank128_tensor_transfers_ood": all(
                results[name]["components"][component]["mode_fidelity"]["64"]
                >= results["fineweb"]["components"][component]["mode_fidelity"]["64"] - .10
                for name in ("wikitext", "tinystories") for component in config["components"]),
        }
    else:
        hypotheses = {
            "Q0_symmetric_tensor_matches_projected_native_cp": maximum_projected_error <= 1e-5,
            "Q1_rank64_input_output_quotient_is_sufficient": all(
                row["components"][component]["projected_direct_fidelity"] >= .90
                for row in results.values() for component in config["components"]),
            "Q2_sixteen_tensor_modes_are_useful": all(
                row["components"][component]["mode_fidelity"]["16"] >= .80
                for row in results.values() for component in config["components"]),
            "Q3_thirtytwo_tensor_modes_are_sufficient": all(
                row["components"][component]["mode_fidelity"]["32"] >= .90
                for row in results.values() for component in config["components"]),
            "Q4_symmetric_tensor_has_energy_rank32": spectrum["rank90"] <= 32,
            "Q5_symmetric_tensor_transfers_ood": all(
                results[name]["components"][component]["mode_fidelity"]["32"]
                >= results["fineweb"]["components"][component]["mode_fidelity"]["32"] - .10
                for name in ("wikitext", "tinystories") for component in config["components"]),
            "Q6_tensor_modes_beat_native_margin_channels": all(
                results[corpus]["components"]["margin"]["mode_fidelity"]["32"]
                >= native_result["results"][corpus]["components"]["margin"]["shared_fidelity"]["512"] + .10
                for corpus in results),
        }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["Q0_symmetric_tensor_matches_projected_native_cp"]["maximum_relative_error"] = maximum_projected_error
    payload = {
        "config": config,
        "hypotheses": hypothesis_rows,
        "component_scales": component_scales,
        "tensor_spectrum": spectrum,
        "results": results,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, allow_nan=False))
    torch.save({
        "input_basis": input_basis,
        "output_basis": output_basis,
        "tensor": tensor,
        "tensor_u": tensor_u,
        "tensor_s": tensor_s,
        "tensor_vh": tensor_vh,
    }, tensor_output)

    lines = [
        "# Symmetric tensor quotient of the final-MLP backward path",
        "",
        "The native Left/Right/Down CP factors are contracted into a symmetric "
        f"output-adjoint × input × input tensor on {config['input_dimension']}D "
        "discovery-frozen input and output quotients. SVD modes are executed through native RMS normalization "
        "and the exact 32D upstream behavioral tangent.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"Projected-CP/tensor error: `{maximum_projected_error:.2e}`. Tensor energy "
        f"R90: `{spectrum['rank90']}`; top share: `{spectrum['top_energy_share']:.3f}`.",
        "",
        "| Corpus | Component | Full quotient | Native fidelity "
        + "/".join(f"K{k}" for k in config["mode_budgets"])
        + " | Within-quotient "
        + "/".join(f"K{k}" for k in config["mode_budgets"]) + " |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        for component in config["components"]:
            item = row["components"][component]
            native_values = "/".join(
                f"{item['mode_fidelity'][str(k)]:.3f}" for k in config["mode_budgets"])
            within_values = "/".join(
                f"{item['within_quotient_mode_fidelity'][str(k)]:.3f}"
                for k in config["mode_budgets"])
            lines.append(
                f"| {corpus} | {component} | {item['projected_direct_fidelity']:.3f} | "
                f"{native_values} | {within_values} |")
    lines += [
        "",
        "Full-quotient fidelity separates basis truncation from tensor-mode "
        "truncation. High within-quotient but low native fidelity means the symmetric "
        "factorization is compact only after choosing a lossy observer quotient.",
    ]
    Path(report_path).write_text("\n".join(lines) + "\n")
    print("wrote", report_path, flush=True)


if __name__ == "__main__":
    main()
