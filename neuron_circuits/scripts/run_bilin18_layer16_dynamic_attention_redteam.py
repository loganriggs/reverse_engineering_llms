#!/usr/bin/env python3
"""Red-team Layer-16 causal compression with dynamic Layer-17 two-QK attention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_mlp_path_type_program import norm
from run_bilin18_attention_layer_path_program import (
    native_cache, sequence_rms_jvp, two_qk_attention_jvp,
)
from run_bilin18_layer16_downstream_relevance import collect_boundary, compose, downstream_maps
from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_late_mlp_operator_atlas import derivative_targets, fidelity, project
from run_bilin18_late_mlp_tensor_canonicalization import (
    exact_reads, factor_modes, gram_modes, truncated_modes,
)
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_dynamic_attention_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_dynamic_attention_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_DYNAMIC_ATTENTION_REDTEAM.md"


@torch.inference_mode()
def local_programs(block, states, axes, modes, reads, mode_banks, mean_local, batch_size):
    exact = derivative_targets(block, states, axes, batch_size).view(len(states), axes.shape[1], -1)
    coefficients = states.float() @ reads.T
    rows = {"exact": exact, "mean": mean_local[None].expand_as(exact),
            "full64": (coefficients.cuda() @ modes.cuda()).view_as(exact.cuda()).cpu()}
    for name, bank in mode_banks.items():
        rows[f"inner{name}"] = (coefficients.cuda() @ bank.cuda()).view_as(exact.cuda()).cpu()
    return rows


@torch.inference_mode()
def propagate_dynamic(block17, row17, position, programs, output_axes, chunk):
    names = list(programs); batch, input_rank, dimension = programs[names[0]].shape
    stacked = torch.stack([programs[name] for name in names], 1).cuda()
    flat = stacked.flatten(1, 2); outputs = []; maximum_forward_error = 0.0
    residual_scale = block17.lambdas[0].float()
    for start in range(0, flat.shape[1], chunk):
        local = flat[:, start:start + chunk]
        tangent = torch.zeros(batch, local.shape[1], row17["mixed"].shape[1], dimension, device="cuda")
        tangent[:, :, position] = residual_scale * local
        attention_input_tangent = sequence_rms_jvp(row17["mixed"].float(), tangent)
        attention_tangent, error = two_qk_attention_jvp(
            block17.attn, row17["attention_input"].float(), row17["bus"], attention_input_tangent)
        maximum_forward_error = max(maximum_forward_error, error)
        state_tangent = sequence_rms_jvp(row17["pre"].float(), tangent + attention_tangent)
        outputs.append(state_tangent[:, :, position] @ output_axes)
    values = torch.cat(outputs, 1).view(batch, len(names), input_rank, -1).cpu()
    return {name: values[:, index] for index, name in enumerate(names)}, maximum_forward_error


@torch.inference_mode()
def collect_dynamic(model, spec, config, local_axes, output_axes, modes, reads,
                    mode_banks, mean_local, include_programs):
    source = load_rows(spec); stores = {}; frozen_rows = []; maximum_forward_error = 0.0
    for start in range(0, len(source), config["batch_size"]):
        ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
        _, cache = native_cache(model, ids, config); by_layer = {row["index"]: row for row in cache}
        state16 = by_layer[16]["state"][:, config["positions"]].float()
        flat_states = state16.flatten(0, 1).cpu()
        programs = local_programs(
            model.blocks[16], flat_states, local_axes, modes, reads, mode_banks,
            mean_local, config["batch_size"])
        shaped = {name: value.view(len(ids), len(config["positions"]), config["rank"], -1)
                  for name, value in programs.items()}
        for offset, position in enumerate(config["positions"]):
            selected = ({name: value[:, offset] for name, value in shaped.items()}
                        if include_programs else {"exact": shaped["exact"][:, offset]})
            dynamic, error = propagate_dynamic(
                model.blocks[17], by_layer[17], position, selected, output_axes,
                config["tangent_chunk"])
            maximum_forward_error = max(maximum_forward_error, error)
            for name, value in dynamic.items(): stores.setdefault(name, []).append(value)
            if include_programs:
                pre = by_layer[17]["pre"][:, position].float().cpu()
                downstream = downstream_maps(pre, output_axes, float(model.blocks[17].lambdas[0].detach()))
                frozen_rows.append(compose(shaped["exact"][:, offset], downstream))
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    result = {name: torch.cat(value) for name, value in stores.items()}
    if include_programs: result["frozen_exact"] = torch.cat(frozen_rows)
    return result, maximum_forward_error


@torch.inference_mode(False)
def autograd_check(model, ids, config, local_axes, output_axes):
    _, cache = native_cache(model, ids, config); by_layer = {row["index"]: row for row in cache}
    position = config["positions"][0]; block16 = model.blocks[16]; block17 = model.blocks[17]
    state16 = by_layer[16]["state"][:, position].detach().clone().cuda()
    exact = derivative_targets(block16, state16.cpu(), local_axes, config["batch_size"]).view(len(ids), config["rank"], -1)
    analytic, _ = propagate_dynamic(
        block17, by_layer[17], position, {"exact": exact}, output_axes, config["tangent_chunk"])
    direction = local_axes[:, 0].cuda()[None].expand(len(ids), -1).clone()
    base_mixed = by_layer[17]["mixed"].detach().clone()
    bus = by_layer[17]["bus"].detach().clone()
    native_write = block16.mlp.Down(block16.mlp.Left(state16) * block16.mlp.Right(state16)) + block16.mlp.Down_bias
    block17.attn.rotary.seq_len_cached = None
    block17.attn.rotary.cos_cached = None
    block17.attn.rotary.sin_cached = None

    def function(value):
        write = block16.mlp.Down(block16.mlp.Left(value) * block16.mlp.Right(value)) + block16.mlp.Down_bias
        mixed = base_mixed.clone()
        mixed[:, position] = mixed[:, position] + block17.lambdas[0] * (write - native_write)
        attention, _ = block17.attn(norm(mixed), bus)
        return norm(mixed + attention)[:, position] @ output_axes

    _, expected = torch.func.jvp(function, (state16,), (direction,))
    manual = analytic["exact"][:, 0].cuda()
    return float((manual - expected).double().norm() / expected.double().norm().clamp_min(1e-30))


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    local_axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])[16]
    axis_boundary = collect_boundary(model, config["axis_discovery"], config)
    output_axes = torch.linalg.svd(axis_boundary["state17"], full_matrices=False).Vh[:config["rank"]].T.cuda()
    local_states = collect_states(model, config["local_atlas_discovery"], config)[16]
    local_target = derivative_targets(model.blocks[16], local_states, local_axes, config["batch_size"])
    modes, singular, _ = gram_modes(local_target, config["operator_modes"])
    reads = exact_reads(model.blocks[16], local_axes, modes)
    left, inner_values, right, _ = factor_modes(modes, singular, config["rank"], local_axes.shape[0])
    mode_banks = {str(rank): truncated_modes(left, inner_values, right, rank)
                  for rank in config["inner_ranks"]}
    mean_local = local_target.view(len(local_target), config["rank"], -1).mean(0)
    check_source = load_rows(config["dynamic_discovery"])
    check_ids = torch.as_tensor(np.asarray(check_source[:1, :128], dtype=np.int64), device="cuda")
    autograd_error = autograd_check(model, check_ids, config, local_axes, output_axes)
    discovery, maximum_forward_error = collect_dynamic(
        model, config["dynamic_discovery"], config, local_axes, output_axes,
        modes, reads, mode_banks, mean_local, False)
    endpoint_modes, _, endpoint_spectrum = gram_modes(
        discovery["exact"].flatten(1), max(config["endpoint_spectral_budgets"]))
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        values, error = collect_dynamic(
            model, spec, config, local_axes, output_axes, modes, reads,
            mode_banks, mean_local, True)
        maximum_forward_error = max(maximum_forward_error, error); target = values["exact"]
        spectral = {str(rank): fidelity(target.flatten(1), project(target.flatten(1), endpoint_modes[:rank]))
                    for rank in config["endpoint_spectral_budgets"]}
        results[corpus] = {
            "sites": len(target),
            "dynamic_vs_frozen_fidelity": fidelity(target, values["frozen_exact"]),
            "endpoint_spectral_fidelity": spectral,
            "mean_local_operator_fidelity": fidelity(target, values["mean"]),
            "full_rank64_local_program_fidelity": fidelity(target, values["full64"]),
            "inner_rank4_local_program_fidelity": fidelity(target, values["inner4"]),
            "inner_rank8_local_program_fidelity": fidelity(target, values["inner8"]),
        }
    exact_error = max(autograd_error, maximum_forward_error)
    hypotheses = {
        "E0_dynamic_composition_is_exact": exact_error <= 1e-5,
        "E1_dynamic_attention_is_a_small_boundary_change": all(row["dynamic_vs_frozen_fidelity"] >= .95 for row in results.values()),
        "E2_dynamic_endpoint_rank16_atlas_is_sufficient": all(row["endpoint_spectral_fidelity"]["16"] >= .90 for row in results.values()),
        "E3_common_local_operator_survives_dynamic_routing": all(row["mean_local_operator_fidelity"] >= .80 for row in results.values()),
        "E4_factorized_local_program_survives_dynamic_routing": all(row["inner_rank8_local_program_fidelity"] >= .90 for row in results.values()),
        "E5_inner_rank8_preserves_the_dynamic_program": all(row["inner_rank8_local_program_fidelity"] >= row["full_rank64_local_program_fidelity"] - .05 for row in results.values()),
        "E6_dynamic_program_transfers_ood": all(results[name]["inner_rank8_local_program_fidelity"] >= results["fineweb"]["inner_rank8_local_program_fidelity"] - .05 for name in ("wikitext", "tinystories")),
        "E7_contextual_correction_remains_material": all(row["full_rank64_local_program_fidelity"] >= row["mean_local_operator_fidelity"] + .05 for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["E0_dynamic_composition_is_exact"].update({
        "autograd_relative_error": autograd_error,
        "maximum_native_forward_relative_error": maximum_forward_error})
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "dynamic_discovery_spectrum": endpoint_spectrum, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Dynamic-attention red team of the Layer-16 compact causal program", "",
             "Layer-16 local derivative programs are propagated through the exact full-sequence Layer-17 two-QK attention JVP, residual route, RMS normalization, and the frozen natural output chart. The panel is deliberately smaller than the frozen-attention run because every site carries many full-sequence tangent worlds.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Autograd error: `{autograd_error:.2e}`; maximum native-forward error: `{maximum_forward_error:.2e}`.",
              f"Dynamic discovery endpoint R90: `{endpoint_spectrum['rank90']}`; top energy share: `{endpoint_spectrum['top_energy_share']:.3f}`.", "",
              "| Corpus | Dynamic vs frozen | Endpoint R8 / R16 / R64 | Mean | Full64 | Inner4 / Inner8 |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        spectral = row["endpoint_spectral_fidelity"]
        lines.append(f"| {corpus} | {row['dynamic_vs_frozen_fidelity']:.3f} | {spectral['8']:.3f} / {spectral['16']:.3f} / {spectral['64']:.3f} | {row['mean_local_operator_fidelity']:.3f} | {row['full_rank64_local_program_fidelity']:.3f} | {row['inner_rank4_local_program_fidelity']:.3f} / {row['inner_rank8_local_program_fidelity']:.3f} |")
    lines += ["", "The direct endpoint spectrum is an oracle coordinate bound. The mean and factorized local programs are state-executable and use the exact dynamic routing derivative. This remains a rank-32 natural output quotient rather than a full-vocabulary logit or cross-entropy result."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
