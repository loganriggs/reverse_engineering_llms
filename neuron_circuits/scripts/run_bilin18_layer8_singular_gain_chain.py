#!/usr/bin/env python3
"""Locate amplification along the singular ordered attention path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer8_direct_tensor_jacobian import (
    bilinear_mlp_jvp,
    head_rms_norm_jvp,
    output_fisher_response,
    rms_norm_jvp,
    rotate_heads,
    squared_bilinear_attention_jvp,
)
from run_bilin18_layer8_singular_source_trace import factor_residual
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_singular_gain_chain.json")
OUTPUT = TENSOR_ROOT / "layer8_singular_gain_chain.json"
REPORT = TENSOR_ROOT / "LAYER8_SINGULAR_GAIN_CHAIN.md"


@torch.inference_mode()
def to_attention_input(model, site, position, selected, target):
    dimension = site["x8"].shape[-1]
    x = site["x8"]; bus = site["value_bus"]
    tangent = site["tangents"].float()
    for layer, block in enumerate(model.blocks[9:target], start=9):
        mixed = block.lambdas[0] * x + block.lambdas[1] * site["x0"]
        attention_input = F.rms_norm(mixed, (dimension,))
        attention, bus = block.attn(attention_input, bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        x = pre + write
        mixed_tangent = block.lambdas[0].float() * tangent
        if layer in selected:
            attention_input_tangent = rms_norm_jvp(mixed[0, position].float(), mixed_tangent)
            attention_tangent = squared_bilinear_attention_jvp(
                block.attn, attention_input[0].float(), site["value_bus"],
                attention_input_tangent, position,
            )
        else:
            attention_tangent = torch.zeros_like(mixed_tangent)
        pre_tangent = mixed_tangent + attention_tangent
        normalized_tangent = rms_norm_jvp(pre[0, position].float(), pre_tangent)
        tangent = pre_tangent + bilinear_mlp_jvp(
            block, normalized[0, position].float(), normalized_tangent,
        )
    block = model.blocks[target]
    mixed = block.lambdas[0] * x + block.lambdas[1] * site["x0"]
    attention_input = F.rms_norm(mixed, (dimension,))
    attention, bus_after = block.attn(attention_input, bus)
    pre = mixed + attention
    normalized = F.rms_norm(pre, (dimension,))
    write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    layer_output = pre + write
    input_tangent = rms_norm_jvp(
        mixed[0, position].float(), block.lambdas[0].float() * tangent,
    )
    return {"block": block, "attention_input": attention_input, "bus_before": bus,
            "bus_after": bus_after, "pre": pre, "normalized": normalized,
            "layer_output": layer_output, "input_tangent": input_tangent}


@torch.inference_mode()
def suffix_from_path(model, target, context, x0, position, tangent):
    dimension = context["layer_output"].shape[-1]
    x = context["layer_output"]; bus = context["bus_after"]
    direct = tangent
    for block in model.blocks[target + 1:]:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_input = F.rms_norm(mixed, (dimension,))
        attention, bus = block.attn(attention_input, bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        x = pre + write
        mixed_tangent = block.lambdas[0].float() * direct
        normalized_tangent = rms_norm_jvp(pre[0, position].float(), mixed_tangent)
        direct = mixed_tangent + bilinear_mlp_jvp(
            block, normalized[0, position].float(), normalized_tangent,
        )
    final = x[0, position].float(); normalized = F.rms_norm(final, (dimension,))
    final_tangent = rms_norm_jvp(final, direct)
    raw = model.lm_head(normalized).float(); logits = 30 * torch.tanh(raw / 30)
    derivative = 1 - torch.tanh(raw / 30).square()
    logit_tangent = F.linear(final_tangent, model.lm_head.weight.float()) * derivative[None]
    return logits, output_fisher_response(logits, logit_tangent)


@torch.inference_mode()
def audit(model, site, config, position):
    i, j, target = config["triple"]
    contexts = {mask: to_attention_input(model, site, position, set(mask), target)
                for mask in ((), (i,), (j,), (i, j))}
    pair_input = (contexts[(i, j)]["input_tangent"] - contexts[(i,)]["input_tangent"]
                  - contexts[(j,)]["input_tangent"] + contexts[()]["input_tangent"])
    context = contexts[()]; block = context["block"]
    head = config["head"]
    q2_raw = F.linear(context["attention_input"][0], block.attn.c_q2.weight.float()).view(
        -1, block.attn.n_head, block.attn.head_dim,
    )
    dq2_raw = F.linear(pair_input, block.attn.c_q2.weight.float()).view(
        -1, block.attn.n_head, block.attn.head_dim,
    )
    cosine, sine = block.attn.rotary(q2_raw[None])
    dq2 = rotate_heads(
        head_rms_norm_jvp(q2_raw[position].float(), dq2_raw),
        cosine[0, position, 0].float()[None, None],
        sine[0, position, 0].float()[None, None],
    )[:, head]
    attention_path = squared_bilinear_attention_jvp(
        block.attn, context["attention_input"][0].float(), context["bus_before"],
        pair_input, position, heads=(head,), factors=(config["factor"],),
    )
    normalized_path = rms_norm_jvp(context["pre"][0, position].float(), attention_path)
    post_mlp_path = attention_path + bilinear_mlp_jvp(
        block, context["normalized"][0, position].float(), normalized_path,
    )
    logits, final_response = suffix_from_path(
        model, target, context, site["x0"], position, post_mlp_path,
    )
    subtraction, _ = factor_residual(
        model, site, tuple(config["triple"]), position, head, config["factor"], None,
    )
    input_norm = float(pair_input.norm()); q2_norm = float(dq2.norm())
    attention_norm = float(attention_path.norm()); post_norm = float(post_mlp_path.norm())
    final_norm = float(final_response.norm())
    return {
        "pair_input_norm": input_norm,
        "q2_raw_rms": float(q2_raw[position, head].square().mean().sqrt()),
        "q2_derivative_norm": q2_norm,
        "q2_gain": q2_norm / max(input_norm, 1e-30),
        "attention_path_norm": attention_norm,
        "routing_gain": attention_norm / max(q2_norm, 1e-30),
        "post_mlp_path_norm": post_norm,
        "mlp_gain": post_norm / max(attention_norm, 1e-30),
        "final_response_norm": final_norm,
        "downstream_gain": final_norm / max(post_norm, 1e-30),
        "direct_to_mobius_error": float(
            (final_response - subtraction).norm() / subtraction.norm().clamp_min(1e-30)
        ),
    }


def ratio(failure, control, key):
    return failure[key] / max(control[key], 1e-30)


def main():
    config = json.loads(CONFIG.read_text())
    source_config_spec = json.loads(Path(config["parent_config"]).read_text())
    factor_config = json.loads(Path(source_config_spec["parent_config"]).read_text())
    head_config = json.loads(Path(factor_config["parent_config"]).read_text())
    stability_config = json.loads(Path(head_config["parent_config"]).read_text())
    mobius_config = json.loads(Path(stability_config["parent_config"]).read_text())
    synergy_config = json.loads(Path(mobius_config["parent_config"]).read_text())
    singleton_config = json.loads(Path(synergy_config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"], "sites_per_query": 2,
                     "budget": parent["budget"]}
    trace = json.loads((TENSOR_ROOT / "layer8_singular_source_trace.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    corpus_config = parent["evaluation"][source_config_spec["corpus"]]
    rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                         parent["queries"], 2)
    bank = {row_index: raw for query, row_index, raw in rows if query == source_config_spec["query"]}
    failure_site = prepare_site(model, bank[source_config_spec["row"]], source_config_spec["query"],
                                source_config, tensor_basis, directions, vectors)
    control_site = prepare_site(model, bank[trace["control_row"]], source_config_spec["query"],
                                source_config, tensor_basis, directions, vectors)
    failure = audit(model, failure_site, source_config_spec, source_config["causal_position"])
    control = audit(model, control_site, source_config_spec, source_config["causal_position"])
    ratios = {key: ratio(failure, control, key) for key in
              ("pair_input_norm", "q2_gain", "routing_gain", "mlp_gain", "downstream_gain",
               "final_response_norm", "q2_raw_rms")}
    hypotheses = {
        "G0_upstream_pair_tangent_amplifies": {"supported": bool(ratios["pair_input_norm"] >= 5)},
        "G1_qk2_local_gain_amplifies": {"supported": bool(ratios["q2_gain"] >= 5)},
        "G2_value_bus_routing_gain_amplifies": {"supported": bool(ratios["routing_gain"] >= 5)},
        "G3_downstream_gain_amplifies": {"supported": bool(ratios["downstream_gain"] >= 5)},
        "G4_direct_ordered_path_matches_mobius": {"supported": bool(
            max(failure["direct_to_mobius_error"], control["direct_to_mobius_error"]) <= 1e-3)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "ratios": ratios, "failure": failure, "control": control}, indent=2))
    lines = ["# Gain chain of the singular ordered QK/value-bus path", "",
             "The `[11,13,14]` head-6 QK2/value-bus path is propagated directly, avoiding Mobius subtraction at the final measurement. Stage gains compare the singular WikiText context with its matched query-1 control.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Stage | Failure | Control | Failure/control |",
              "|---|---:|---:|---:|",
              f"| Pair tangent arriving L14 | {failure['pair_input_norm']:.3e} | {control['pair_input_norm']:.3e} | {ratios['pair_input_norm']:.2f}x |",
              f"| QK2 derivative gain | {failure['q2_gain']:.3e} | {control['q2_gain']:.3e} | {ratios['q2_gain']:.2f}x |",
              f"| Value-bus routing gain | {failure['routing_gain']:.3e} | {control['routing_gain']:.3e} | {ratios['routing_gain']:.2f}x |",
              f"| Layer-14 MLP gain | {failure['mlp_gain']:.3e} | {control['mlp_gain']:.3e} | {ratios['mlp_gain']:.2f}x |",
              f"| Downstream suffix gain | {failure['downstream_gain']:.3e} | {control['downstream_gain']:.3e} | {ratios['downstream_gain']:.2f}x |",
              f"| Final response norm | {failure['final_response_norm']:.3e} | {control['final_response_norm']:.3e} | {ratios['final_response_norm']:.2f}x |",
              "", f"Direct-to-Mobius errors: failure `{failure['direct_to_mobius_error']:.3e}`, control `{control['direct_to_mobius_error']:.3e}`. Native head-6 QK2 raw RMS: failure `{failure['q2_raw_rms']:.3e}`, control `{control['q2_raw_rms']:.3e}`.", "",
              "A large upstream ratio means the singularity is created before the final head. A large local-gain ratio identifies QK normalization or value-bus readout. A large downstream ratio means a modest route is amplified after layer 14."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
