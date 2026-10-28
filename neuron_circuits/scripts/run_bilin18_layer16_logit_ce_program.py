#!/usr/bin/env python3
"""Score the compact Layer-16 program in logits, Fisher geometry, and CE."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_mlp_path_type_program import norm
from run_bilin18_attention_layer_path_program import (
    mlp_post_jvp, native_cache, sequence_rms_jvp, two_qk_attention_jvp,
)
from run_bilin18_layer16_downstream_relevance import collect_boundary
from run_bilin18_layer16_dynamic_attention_redteam import local_programs
from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_late_mlp_operator_atlas import derivative_targets
from run_bilin18_late_mlp_tensor_canonicalization import (
    exact_reads, factor_modes, gram_modes, truncated_modes,
)
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_logit_ce_program.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_logit_ce_program.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_LOGIT_CE_PROGRAM.md"
PROGRAMS = ("mean", "full64", "inner4", "inner8")
METRICS = ("logit", "fisher", "cross_entropy", "margin")


@torch.inference_mode()
def base_output(model, row17, position):
    block = model.blocks[17]
    final_pre = row17["pre"].float() + block.mlp(row17["state"].float())
    final_state = norm(final_pre[:, position])
    raw_logits = model.lm_head(final_state)
    logits = 30 * torch.tanh(raw_logits / 30)
    return final_pre, raw_logits, logits


@torch.inference_mode()
def propagate_logits(model, row17, position, local, chunk):
    block = model.blocks[17]; batch, input_rank, dimension = local.shape
    final_pre, raw_logits, logits = base_output(model, row17, position)
    outputs = []; maximum_forward_error = 0.0
    for start in range(0, input_rank, chunk):
        selected = local[:, start:start + chunk].cuda()
        tangent = torch.zeros(batch, selected.shape[1], row17["mixed"].shape[1], dimension, device="cuda")
        tangent[:, :, position] = block.lambdas[0].float() * selected
        normalized = sequence_rms_jvp(row17["mixed"].float(), tangent)
        attention, error = two_qk_attention_jvp(
            block.attn, row17["attention_input"].float(), row17["bus"], normalized)
        maximum_forward_error = max(maximum_forward_error, error)
        pre_tangent = tangent + attention
        final_pre_tangent = mlp_post_jvp(
            block, row17["pre"].float(), row17["state"].float(), pre_tangent)
        final_state_tangent = sequence_rms_jvp(
            final_pre[:, position], final_pre_tangent[:, :, position])
        raw_tangent = F.linear(final_state_tangent, model.lm_head.weight.float())
        cap_derivative = 1 - torch.tanh(raw_logits[:, None] / 30).square()
        outputs.append(raw_tangent * cap_derivative)
    return torch.cat(outputs, 1), logits, maximum_forward_error


def empty_accumulator():
    return {program: {metric: {"error": 0.0, "target": 0.0} for metric in METRICS}
            for program in PROGRAMS}


@torch.inference_mode()
def update_metrics(accumulator, target_tangent, predicted, logits, labels):
    probability = logits.softmax(-1)
    target_mean = (probability[:, None] * target_tangent).sum(-1, keepdim=True)
    target_centered = target_tangent - target_mean
    target_ce = target_mean.squeeze(-1) - target_tangent.gather(
        -1, labels[:, None, None].expand(-1, target_tangent.shape[1], 1)).squeeze(-1)
    competitor_logits = logits.clone(); competitor_logits.scatter_(1, labels[:, None], float("-inf"))
    competitor = competitor_logits.argmax(-1)
    target_margin = (target_tangent.gather(-1, labels[:, None, None].expand(-1, target_tangent.shape[1], 1))
                     - target_tangent.gather(-1, competitor[:, None, None].expand(-1, target_tangent.shape[1], 1))).squeeze(-1)
    for name, value in predicted.items():
        error = value - target_tangent
        prediction_mean = (probability[:, None] * value).sum(-1, keepdim=True)
        error_centered = error - (probability[:, None] * error).sum(-1, keepdim=True)
        prediction_ce = prediction_mean.squeeze(-1) - value.gather(
            -1, labels[:, None, None].expand(-1, value.shape[1], 1)).squeeze(-1)
        prediction_margin = (value.gather(-1, labels[:, None, None].expand(-1, value.shape[1], 1))
                             - value.gather(-1, competitor[:, None, None].expand(-1, value.shape[1], 1))).squeeze(-1)
        rows = accumulator[name]
        rows["logit"]["error"] += float(error.double().square().sum())
        rows["logit"]["target"] += float(target_tangent.double().square().sum())
        rows["fisher"]["error"] += float((probability[:, None] * error_centered.square()).double().sum())
        rows["fisher"]["target"] += float((probability[:, None] * target_centered.square()).double().sum())
        rows["cross_entropy"]["error"] += float((prediction_ce - target_ce).double().square().sum())
        rows["cross_entropy"]["target"] += float(target_ce.double().square().sum())
        rows["margin"]["error"] += float((prediction_margin - target_margin).double().square().sum())
        rows["margin"]["target"] += float(target_margin.double().square().sum())


def finish(accumulator):
    return {program: {metric: 1 - values["error"] / max(values["target"], 1e-30)
                      for metric, values in rows.items()}
            for program, rows in accumulator.items()}


@torch.inference_mode()
def collect(model, spec, config, local_axes, modes, reads, mode_banks, mean_local):
    source = load_rows(spec); accumulator = empty_accumulator(); maximum_forward_error = 0.0
    ce_sum = 0.0; token_count = 0
    for start in range(0, len(source), config["batch_size"]):
        batch_array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(batch_array[:, :128].astype(np.int64), device="cuda")
        _, cache = native_cache(model, ids, config); by_layer = {row["index"]: row for row in cache}
        states = by_layer[16]["state"][:, config["positions"]].float(); flat_states = states.flatten(0, 1).cpu()
        programs = local_programs(
            model.blocks[16], flat_states, local_axes, modes, reads, mode_banks,
            mean_local, config["batch_size"])
        shaped = {name: value.view(len(ids), len(config["positions"]), config["rank"], -1)
                  for name, value in programs.items()}
        for offset, position in enumerate(config["positions"]):
            target_tangent, logits, error = propagate_logits(
                model, by_layer[17], position, shaped["exact"][:, offset], config["tangent_chunk"])
            maximum_forward_error = max(maximum_forward_error, error)
            predictions = {}
            for name in PROGRAMS:
                predictions[name], _, error = propagate_logits(
                    model, by_layer[17], position, shaped[name][:, offset], config["tangent_chunk"])
                maximum_forward_error = max(maximum_forward_error, error)
            labels = torch.as_tensor(batch_array[:, position + 1].astype(np.int64), device="cuda")
            update_metrics(accumulator, target_tangent, predictions, logits, labels)
            ce_sum += float(F.cross_entropy(logits, labels, reduction="sum")); token_count += len(labels)
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    return {"sites": token_count, "native_cross_entropy": ce_sum / token_count,
            "fidelity": finish(accumulator)}, maximum_forward_error


@torch.inference_mode(False)
def autograd_check(model, ids, config, local_axes):
    _, cache = native_cache(model, ids, config); by_layer = {row["index"]: row for row in cache}
    position = config["positions"][0]; block16 = model.blocks[16]; block17 = model.blocks[17]
    state16 = by_layer[16]["state"][:, position].detach().clone().cuda()
    exact = derivative_targets(block16, state16.cpu(), local_axes, config["batch_size"]).view(len(ids), config["rank"], -1)
    analytic, _, _ = propagate_logits(model, by_layer[17], position, exact, config["tangent_chunk"])
    direction = local_axes[:, 0].cuda()[None].expand(len(ids), -1).clone()
    base_mixed = by_layer[17]["mixed"].detach().clone(); bus = by_layer[17]["bus"].detach().clone()
    native_write = block16.mlp(state16)
    block17.attn.rotary.seq_len_cached = None; block17.attn.rotary.cos_cached = None; block17.attn.rotary.sin_cached = None

    def function(value):
        mixed = base_mixed.clone()
        mixed[:, position] = mixed[:, position] + block17.lambdas[0] * (block16.mlp(value) - native_write)
        attention, _ = block17.attn(norm(mixed), bus)
        pre = mixed + attention
        final = norm(pre + block17.mlp(norm(pre)))
        raw = model.lm_head(final[:, position])
        return 30 * torch.tanh(raw / 30)

    _, expected = torch.func.jvp(function, (state16,), (direction,))
    manual = analytic[:, 0]
    return float((manual - expected).double().norm() / expected.double().norm().clamp_min(1e-30))


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    local_axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])[16]
    local_states = collect_states(model, config["local_atlas_discovery"], config)[16]
    local_target = derivative_targets(model.blocks[16], local_states, local_axes, config["batch_size"])
    modes, singular, _ = gram_modes(local_target, config["operator_modes"])
    reads = exact_reads(model.blocks[16], local_axes, modes)
    left, inner_values, right, _ = factor_modes(modes, singular, config["rank"], local_axes.shape[0])
    mode_banks = {str(rank): truncated_modes(left, inner_values, right, rank)
                  for rank in config["inner_ranks"]}
    mean_local = local_target.view(len(local_target), config["rank"], -1).mean(0)
    check_source = load_rows(config["evaluation"]["fineweb"])
    check_ids = torch.as_tensor(np.asarray(check_source[:1, :128], dtype=np.int64), device="cuda")
    autograd_error = autograd_check(model, check_ids, config, local_axes)
    results = {}; maximum_forward_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        results[corpus], error = collect(
            model, spec, config, local_axes, modes, reads, mode_banks, mean_local)
        maximum_forward_error = max(maximum_forward_error, error)
    hypotheses = {
        "L0_full_suffix_logit_derivative_is_exact": max(autograd_error, maximum_forward_error) <= 1e-5,
        "L1_inner8_preserves_full_logit_tangents": all(row["fidelity"]["inner8"]["logit"] >= .95 for row in results.values()),
        "L2_inner4_preserves_full_logit_tangents": all(row["fidelity"]["inner4"]["logit"] >= .90 for row in results.values()),
        "L3_inner8_preserves_fisher_geometry": all(row["fidelity"]["inner8"]["fisher"] >= .95 for row in results.values()),
        "L4_inner8_preserves_cross_entropy_derivatives": all(row["fidelity"]["inner8"]["cross_entropy"] >= .90 for row in results.values()),
        "L5_common_operator_is_loss_sufficient": all(row["fidelity"]["mean"]["cross_entropy"] >= .70 for row in results.values()),
        "L6_contextual_program_materially_improves_loss": all(row["fidelity"]["full64"]["cross_entropy"] >= row["fidelity"]["mean"]["cross_entropy"] + .10 for row in results.values()),
        "L7_loss_program_transfers_ood": all(results[name]["fidelity"]["inner8"]["cross_entropy"] >= results["fineweb"]["fidelity"]["inner8"]["cross_entropy"] - .05 for name in ("wikitext", "tinystories")),
        "L8_inner8_nearly_matches_full64_on_loss": all(row["fidelity"]["inner8"]["cross_entropy"] >= row["fidelity"]["full64"]["cross_entropy"] - .05 for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["L0_full_suffix_logit_derivative_is_exact"].update({
        "autograd_relative_error": autograd_error,
        "maximum_native_forward_relative_error": maximum_forward_error})
    payload = {"config": config, "hypotheses": hypothesis_rows, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Layer-16 compact program at the final logits and loss", "",
             "The exact and approximate Layer-16 local derivative programs pass through dynamic Layer-17 attention, the final bilinear MLP, final RMS normalization, unembedding, and tanh logit cap. `Fisher` is probability-weighted centered-logit squared fidelity, the local quadratic geometry of predictive KL. `CE` is fidelity of the true-next-token loss directional derivatives over the 32 local input axes.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Autograd error: `{autograd_error:.2e}`; maximum native-forward error: `{maximum_forward_error:.2e}`.", "",
              "| Corpus | Native CE | Program | Logit | Fisher | CE derivative | Margin |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for program in PROGRAMS:
            values = row["fidelity"][program]
            lines.append(f"| {corpus} | {row['native_cross_entropy']:.3f} | {program} | {values['logit']:.3f} | {values['fisher']:.3f} | {values['cross_entropy']:.3f} | {values['margin']:.3f} |")
    lines += ["", "These are first-order causal comparisons around native text states. CE and margin can be noisier than full-logit fidelity because each collapses a 50,304-dimensional tangent to a single scalar per direction; aggregate denominators and OOD replication are therefore reported rather than per-example ratios."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
