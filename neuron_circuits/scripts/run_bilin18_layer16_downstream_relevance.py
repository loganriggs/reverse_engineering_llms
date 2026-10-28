#!/usr/bin/env python3
"""Compose the Layer-16 local operator atlas with its exact output-facing map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_late_mlp_operator_atlas import derivative_targets, fidelity, project
from run_bilin18_late_mlp_tensor_canonicalization import (
    exact_reads, factor_modes, gram_modes, truncated_modes,
)
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_downstream_relevance.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_downstream_relevance.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_DOWNSTREAM_RELEVANCE.md"


@torch.inference_mode()
def collect_boundary(model, spec, config):
    stores = {"state16": [], "pre17": [], "state17": []}
    source = load_rows(spec)
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        _, cache = native_cache(model, ids, config)
        by_layer = {row["index"]: row for row in cache}
        positions = config["positions"]
        stores["state16"].append(by_layer[16]["state"][:, positions].float().flatten(0, 1).cpu())
        stores["pre17"].append(by_layer[17]["pre"][:, positions].float().flatten(0, 1).cpu())
        stores["state17"].append(by_layer[17]["state"][:, positions].float().flatten(0, 1).cpu())
    return {key: torch.cat(value) for key, value in stores.items()}


@torch.inference_mode()
def downstream_maps(pre17, output_axes, residual_scale):
    x = pre17.float().cuda(); axes = output_axes.float().cuda()
    root = x.square().mean(-1, keepdim=True).sqrt()
    coupling = x @ axes
    maps = axes[None] / root[:, :, None] - (
        x[:, :, None] * coupling[:, None, :] / (x.shape[-1] * root[:, :, None].pow(3)))
    return (residual_scale * maps).cpu()


@torch.inference_mode()
def compose(local, downstream):
    return torch.einsum("snd,sdr->snr", local.float().cuda(), downstream.float().cuda()).cpu()


@torch.inference_mode(False)
def validate_autograd(block16, state16, pre17, input_axes, output_axes, analytic, residual_scale, seed):
    count = min(8, len(state16)); generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(count, input_axes.shape[1], generator=generator).cuda()
    direction = coefficients @ input_axes.T.cuda()
    base = state16[:count].cuda().clone(); fixed_pre = pre17[:count].cuda().clone()
    native_write = (block16.mlp.Down(block16.mlp.Left(base) * block16.mlp.Right(base))
                    + block16.mlp.Down_bias)

    def function(value):
        write = block16.mlp.Down(block16.mlp.Left(value) * block16.mlp.Right(value)) + block16.mlp.Down_bias
        return F.rms_norm(fixed_pre + residual_scale * (write - native_write), (value.shape[-1],)) @ output_axes

    _, tangent = torch.func.jvp(function, (base,), (direction,))
    manual = torch.einsum("sn,snr->sr", coefficients, analytic[:count].cuda())
    return float((tangent - manual).double().norm() / tangent.double().norm().clamp_min(1e-30))


@torch.inference_mode(False)
def main():
    raw = json.loads(CONFIG.read_text()); config = raw
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    local_axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])[16]
    axis_boundary = collect_boundary(model, config["axis_discovery"], config)
    output_axes = torch.linalg.svd(axis_boundary["state17"], full_matrices=False).Vh[:config["rank"]].T.cuda()
    discovery = collect_boundary(model, config["atlas_discovery"], config)
    residual_scale = float(model.blocks[17].lambdas[0])
    discovery_local = derivative_targets(
        model.blocks[16], discovery["state16"], local_axes, config["batch_size"])
    discovery_downstream = downstream_maps(discovery["pre17"], output_axes, residual_scale)
    discovery_endpoint = compose(discovery_local.view(len(discovery_local), config["rank"], -1),
                                 discovery_downstream)
    autograd_error = validate_autograd(
        model.blocks[16], discovery["state16"], discovery["pre17"], local_axes,
        output_axes, discovery_endpoint, residual_scale, config["seed"])

    endpoint_modes, _, endpoint_uncentered = gram_modes(
        discovery_endpoint.flatten(1), max(config["endpoint_spectral_budgets"]))
    _, _, endpoint_centered = gram_modes(
        discovery_endpoint.flatten(1) - discovery_endpoint.flatten(1).mean(0),
        min(len(discovery_endpoint) - 1, 512))
    local_modes, local_singular, _ = gram_modes(discovery_local, config["operator_modes"])
    reads = exact_reads(model.blocks[16], local_axes, local_modes)
    left, inner_values, right, _ = factor_modes(
        local_modes, local_singular, config["rank"], local_axes.shape[0])
    mode_banks = {str(rank): truncated_modes(left, inner_values, right, rank)
                  for rank in config["inner_ranks"]}
    mean_local = discovery_local.view(len(discovery_local), config["rank"], -1).mean(0)

    maximum_endpoint_rank = max(config["endpoint_spectral_budgets"]); results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        boundary = collect_boundary(model, spec, config)
        local = derivative_targets(model.blocks[16], boundary["state16"], local_axes, config["batch_size"])
        local = local.view(len(local), config["rank"], -1)
        downstream = downstream_maps(boundary["pre17"], output_axes, residual_scale)
        endpoint = compose(local, downstream)
        spectral = {}
        flattened = endpoint.flatten(1)
        for budget in config["endpoint_spectral_budgets"]:
            spectral[str(budget)] = fidelity(flattened, project(flattened, endpoint_modes[:budget]))
        coefficients = boundary["state16"].float() @ reads.T
        full_local = (coefficients.cuda() @ local_modes.cuda()).view_as(local.cuda()).cpu()
        full_endpoint = compose(full_local, downstream)
        inner = {}
        for inner_rank, bank in mode_banks.items():
            predicted_local = (coefficients.cuda() @ bank.cuda()).view_as(local.cuda()).cpu()
            inner[inner_rank] = fidelity(endpoint, compose(predicted_local, downstream))
        mean_endpoint = compose(mean_local[None].expand_as(local), downstream)
        results[corpus] = {
            "sites": len(local),
            "endpoint_spectral_fidelity": spectral,
            "mean_local_operator_endpoint_fidelity": fidelity(endpoint, mean_endpoint),
            "full_rank64_local_program_endpoint_fidelity": fidelity(endpoint, full_endpoint),
            "inner_rank_local_program_endpoint_fidelity": inner,
        }

    hypotheses = {
        "D0_composed_endpoint_derivative_is_exact": autograd_error <= 1e-5,
        "D1_downstream_readout_compresses_contextual_variation": endpoint_centered["rank90"] <= 142,
        "D2_endpoint_rank16_atlas_is_sufficient": all(
            row["endpoint_spectral_fidelity"]["16"] >= .90 for row in results.values()),
        "D3_endpoint_rank64_atlas_is_high_fidelity": all(
            row["endpoint_spectral_fidelity"]["64"] >= .95 for row in results.values()),
        "D4_common_local_operator_is_behaviorally_sufficient": all(
            row["mean_local_operator_endpoint_fidelity"] >= .80 for row in results.values()),
        "D5_factorized_local_program_is_behaviorally_sufficient": all(
            row["inner_rank_local_program_endpoint_fidelity"]["8"] >= .90 for row in results.values()),
        "D6_inner_rank8_preserves_the_causal_atlas": all(
            row["inner_rank_local_program_endpoint_fidelity"]["8"]
            >= row["full_rank64_local_program_endpoint_fidelity"] - .05 for row in results.values()),
        "D7_downstream_program_transfers_ood": all(
            results[name]["inner_rank_local_program_endpoint_fidelity"]["8"]
            >= results["fineweb"]["inner_rank_local_program_endpoint_fidelity"]["8"] - .05
            for name in ("wikitext", "tinystories")),
        "D8_contextual_correction_is_behaviorally_material": all(
            row["full_rank64_local_program_endpoint_fidelity"]
            >= row["mean_local_operator_endpoint_fidelity"] + .05 for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["D0_composed_endpoint_derivative_is_exact"]["relative_error"] = autograd_error
    payload = {"config": raw, "hypotheses": hypothesis_rows,
               "endpoint_discovery_spectrum": {"uncentered": endpoint_uncentered,
                                                "centered": endpoint_centered},
               "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Downstream relevance of the Layer-16 common operator and contextual correction", "",
             "The exact Layer-16 local MLP derivative is composed site-by-site with the Layer-17 residual scale, RMS-normalization Jacobian, and a frozen natural rank-32 output chart. Attention is fixed at its native response, matching the established MLP path boundary. This distinguishes locally large variation from variation that reaches the output-facing state.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Independent autograd relative error: `{autograd_error:.2e}`.",
              f"Discovery endpoint spectrum: uncentered R90 `{endpoint_uncentered['rank90']}`, centered R90 `{endpoint_centered['rank90']}`, centered top share `{endpoint_centered['top_energy_share']:.3f}`.", "",
              "| Corpus | Endpoint spectral R8 / R16 / R64 | Mean local | Full local64 | Local64 inner R1 / R2 / R4 / R8 |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        spectral = row["endpoint_spectral_fidelity"]
        inner = row["inner_rank_local_program_endpoint_fidelity"]
        lines.append(f"| {corpus} | {spectral['8']:.3f} / {spectral['16']:.3f} / {spectral['64']:.3f} | {row['mean_local_operator_endpoint_fidelity']:.3f} | {row['full_rank64_local_program_endpoint_fidelity']:.3f} | {inner['1']:.3f} / {inner['2']:.3f} / {inner['4']:.3f} / {inner['8']:.3f} |")
    lines += ["", "The endpoint spectral atlas is a hindsight coordinate bound. The local programs are executable: their coefficients come from exact weight-derived state reads, and the site-specific downstream RMS map is computed from the native state. Neither result yet assigns natural-language semantics to the reads."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
