#!/usr/bin/env python3
"""Propagate Layer-8 modes through analytic residual/bilinear suffix tangents."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_downstream_fisher_oracle import mode_write_tangents
from run_bilin18_layer8_frozen_fisher_quotient import finite_routes
from run_bilin18_layer8_joint_fisher_program import (
    greedy_normalized_joint,
    normalized_joint_objective,
    weighted_responses,
)
from run_bilin18_layer8_router_causal_validation import (
    prefix_to_layer,
    select_sites,
    suffix_from_layer,
)
from run_bilin18_layer8_stratified_fisher_oracle import stratified_fisher_sketch
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_direct_tensor_jacobian.json")
OUTPUT = TENSOR_ROOT / "layer8_direct_tensor_jacobian.json"
REPORT = TENSOR_ROOT / "LAYER8_DIRECT_TENSOR_JACOBIAN.md"
FIGURE = TENSOR_ROOT / "layer8_direct_tensor_jacobian.png"


def rms_norm_jvp(x, tangents):
    eps = torch.finfo(x.dtype).eps
    mean_square = x.square().mean() + eps
    root = mean_square.sqrt()
    coupling = (tangents * x[None]).mean(-1, keepdim=True)
    return tangents / root - x[None] * coupling / mean_square.pow(1.5)


def bilinear_mlp_jvp(block, normalized, tangents):
    left_x = block.mlp.Left(normalized).float()
    right_x = block.mlp.Right(normalized).float()
    left_tangent = F.linear(tangents, block.mlp.Left.weight.float())
    right_tangent = F.linear(tangents, block.mlp.Right.weight.float())
    product_tangent = left_tangent * right_x[None] + right_tangent * left_x[None]
    return F.linear(product_tangent, block.mlp.Down.weight.float())


def head_rms_norm_jvp(x, tangents):
    eps = torch.finfo(x.dtype).eps
    mean_square = x.square().mean(-1, keepdim=True) + eps
    root = mean_square.sqrt()
    coupling = (tangents * x[None]).mean(-1, keepdim=True)
    return tangents / root[None] - x[None] * coupling / mean_square.pow(1.5)[None]


def rotate_heads(values, cos, sin):
    half = values.shape[-1] // 2
    first, second = values[..., :half], values[..., half:]
    return torch.cat((first * cos + second * sin, first * (-sin) + second * cos), dim=-1)


def squared_bilinear_attention_jvp(attention, normalized, value_bus, tangents, position,
                                   heads=None, factors=None, source_positions=None):
    """Exact final-position JVP for the checkpoint's squared bilinear attention."""
    length, dimension = normalized.shape
    head_count, head_dim = attention.n_head, attention.head_dim
    q_raw = F.linear(normalized, attention.c_q.weight.float()).view(length, head_count, head_dim)
    k_raw = F.linear(normalized, attention.c_k.weight.float()).view(length, head_count, head_dim)
    q2_raw = F.linear(normalized, attention.c_q2.weight.float()).view(length, head_count, head_dim)
    k2_raw = F.linear(normalized, attention.c_k2.weight.float()).view(length, head_count, head_dim)
    v_current = F.linear(normalized, attention.c_v.weight.float()).view(length, head_count, head_dim)
    current_value = (1 - attention.lamb.float()) * v_current
    bus_value = attention.lamb.float() * value_bus[0].float()
    value = current_value + bus_value

    cosine, sine = attention.rotary(q_raw[None])
    cosine = cosine[0, :, 0].float()
    sine = sine[0, :, 0].float()
    q = rotate_heads(F.rms_norm(q_raw, (head_dim,)), cosine[:, None], sine[:, None])
    k = rotate_heads(F.rms_norm(k_raw, (head_dim,)), cosine[:, None], sine[:, None])
    q2 = rotate_heads(F.rms_norm(q2_raw, (head_dim,)), cosine[:, None], sine[:, None])
    k2 = rotate_heads(F.rms_norm(k2_raw, (head_dim,)), cosine[:, None], sine[:, None])

    dq_raw = F.linear(tangents, attention.c_q.weight.float()).view(-1, head_count, head_dim)
    dk_raw = F.linear(tangents, attention.c_k.weight.float()).view(-1, head_count, head_dim)
    dq2_raw = F.linear(tangents, attention.c_q2.weight.float()).view(-1, head_count, head_dim)
    dk2_raw = F.linear(tangents, attention.c_k2.weight.float()).view(-1, head_count, head_dim)
    dv = (1 - attention.lamb.float()) * F.linear(
        tangents, attention.c_v.weight.float(),
    ).view(-1, head_count, head_dim)
    dq = rotate_heads(
        head_rms_norm_jvp(q_raw[position].float(), dq_raw),
        cosine[position][None, None], sine[position][None, None],
    )
    dk = rotate_heads(
        head_rms_norm_jvp(k_raw[position].float(), dk_raw),
        cosine[position][None, None], sine[position][None, None],
    )
    dq2 = rotate_heads(
        head_rms_norm_jvp(q2_raw[position].float(), dq2_raw),
        cosine[position][None, None], sine[position][None, None],
    )
    dk2 = rotate_heads(
        head_rms_norm_jvp(k2_raw[position].float(), dk2_raw),
        cosine[position][None, None], sine[position][None, None],
    )

    visible = slice(0, position + 1)
    score = torch.einsum("hd,thd->ht", q[position], k[visible])
    score2 = torch.einsum("hd,thd->ht", q2[position], k2[visible])
    dscore = torch.einsum("nhd,thd->nht", dq, k[visible])
    dscore2 = torch.einsum("nhd,thd->nht", dq2, k2[visible])
    dscore[:, :, position] += torch.einsum("hd,nhd->nh", q[position], dk)
    dscore2[:, :, position] += torch.einsum("hd,nhd->nh", q2[position], dk2)
    pattern = (score / head_dim) * (score2 / head_dim)
    dpattern_qk1 = (dscore * score2[None]) / (head_dim ** 2)
    dpattern_qk2 = (score[None] * dscore2) / (head_dim ** 2)
    if source_positions is not None:
        source_keep = torch.zeros(position + 1, device=dscore.device, dtype=dscore.dtype)
        source_keep[torch.as_tensor(
            list(source_positions), device=dscore.device, dtype=torch.long,
        )] = 1
        dpattern_qk1 = dpattern_qk1 * source_keep[None, None, :]
        dpattern_qk2 = dpattern_qk2 * source_keep[None, None, :]
    components = {
        "qk1_current_route": torch.einsum(
            "nht,thd->nhd", dpattern_qk1, current_value[visible],
        ),
        "qk1_bus_route": torch.einsum(
            "nht,thd->nhd", dpattern_qk1, bus_value[visible],
        ),
        "qk2_current_route": torch.einsum(
            "nht,thd->nhd", dpattern_qk2, current_value[visible],
        ),
        "qk2_bus_route": torch.einsum(
            "nht,thd->nhd", dpattern_qk2, bus_value[visible],
        ),
        "current_value": pattern[:, position][None, :, None] * dv,
    }
    selected_factors = components if factors is None else {
        name: components[name] for name in factors
    }
    dz = sum(selected_factors.values(), torch.zeros_like(dv))
    if heads is not None:
        keep = torch.zeros(attention.n_head, device=dz.device, dtype=dz.dtype)
        keep[torch.as_tensor(list(heads), device=dz.device, dtype=torch.long)] = 1
        dz = dz * keep[None, :, None]
    return F.linear(dz.flatten(1), attention.c_proj.weight.float())


def output_fisher_response(logits, logit_tangents):
    probability = F.softmax(logits.float(), -1)
    centered = logit_tangents - (logit_tangents * probability[None]).sum(-1, keepdim=True)
    return probability.sqrt()[:, None] * centered.T


@torch.no_grad()
def analytic_suffix(model, x8, x0, value_bus, position, initial_tangents,
                    include_attention=False, attention_layers=None,
                    attention_heads=None, attention_factors=None,
                    attention_sources=None):
    """Propagate Layer-8 tangents through the suffix.

    ``attention_layers`` optionally restricts attention JVPs to an explicit set
    of absolute block indices.  The native forward pass is unchanged.  Keeping
    ``include_attention`` as the master switch preserves the original direct
    and full-tensor endpoints exactly.
    """
    if attention_layers is not None:
        attention_layers = frozenset(int(layer) for layer in attention_layers)
    attention_heads = {} if attention_heads is None else attention_heads
    attention_factors = {} if attention_factors is None else attention_factors
    attention_sources = {} if attention_sources is None else attention_sources
    dimension = x8.shape[-1]
    x = x8
    direct = initial_tangents.float()
    residual = initial_tangents.float()
    bus = value_bus
    for layer, block in enumerate(model.blocks[9:], start=9):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_input = F.rms_norm(mixed, (dimension,))
        attention, bus = block.attn(attention_input, bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        write = block.mlp.Down(product) + block.mlp.Down_bias
        x = pre + write

        direct_mixed = block.lambdas[0].float() * direct
        if include_attention and (attention_layers is None or layer in attention_layers):
            attention_input_tangent = rms_norm_jvp(
                mixed[0, position].float(), direct_mixed,
            )
            attention_tangent = squared_bilinear_attention_jvp(
                block.attn, attention_input[0].float(), value_bus,
                attention_input_tangent, position,
                heads=attention_heads.get(layer),
                factors=attention_factors.get(layer),
                source_positions=attention_sources.get(layer),
            )
        else:
            attention_tangent = torch.zeros_like(direct_mixed)
        direct_pre = direct_mixed + attention_tangent
        direct_normalized = rms_norm_jvp(pre[0, position].float(), direct_pre)
        direct = direct_mixed + bilinear_mlp_jvp(
            block, normalized[0, position].float(), direct_normalized,
        ) + attention_tangent
        residual = block.lambdas[0].float() * residual

    final_state = x[0, position].float()
    final_normalized = F.rms_norm(final_state, (dimension,))
    direct_final = rms_norm_jvp(final_state, direct)
    residual_final = rms_norm_jvp(final_state, residual)
    raw = model.lm_head(final_normalized).float()
    logits = 30 * torch.tanh(raw / 30)
    bounded_derivative = 1 - torch.tanh(raw / 30).square()
    direct_logits = F.linear(direct_final, model.lm_head.weight.float()) * bounded_derivative[None]
    residual_logits = F.linear(residual_final, model.lm_head.weight.float()) * bounded_derivative[None]
    return logits, direct_logits, residual_logits


def selected_tangent(logit_tangents, amplitudes, indices, target_norm):
    raw_input_norm = amplitudes[indices].square().sum().sqrt().clamp_min(1e-30)
    scale = target_norm / raw_input_norm
    return scale * logit_tangents[indices].sum(0)


def summarize(records, routes):
    result = {}
    for route in routes:
        keys = [
            "native_kl", "fidelity_to_oracle", "cosine_to_oracle",
            "heldout_objective_efficiency", "oracle_support_recall",
        ]
        if route in ("direct_tensor", "residual_only"):
            keys += ["tangent_to_finite_fidelity", "tangent_to_finite_cosine"]
        result[route] = {
            key: float(np.median([record["routes"][route][key] for record in records]))
            for key in keys
        }
    return result


def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    global_arrays = np.load(config["global_arrays"])
    global_singular = torch.from_numpy(global_arrays["singular"]).cuda()
    global_vh = torch.from_numpy(global_arrays["vh"]).cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    tensor_config = {**source_config, "positions": source_config["discovery_positions"]}
    with torch.no_grad():
        discovery_x, discovery_write = collect(
            model, 8, source_config["tensor_discovery"], tensor_config,
        )
        _, _, tensor_vh = torch.linalg.svd(discovery_x, full_matrices=False)
        tensor_basis = tensor_vh.T[:, :source_config["dimension"]]
        centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
        _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
        directions = write_vh.T[:, :source_config["queries"]]
        block8 = model.blocks[8]
        left = block8.mlp.Left.weight.float() @ tensor_basis
        right = block8.mlp.Right.weight.float() @ tensor_basis
        coefficients = block8.mlp.Down.weight.float().T @ directions
        matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
        matrices = 0.5 * (matrices + matrices.transpose(1, 2))
        values, vectors = ordered_eigensystem(matrices)

    results = {}
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        scan_config = {**source_config, "positions": [source_config["causal_position"]]}
        states, _ = collect(model, 8, corpus_config, scan_config)
        selected_sites = select_sites(
            states, tensor_basis, values, vectors, source_config["sites_per_query"],
        )
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        records = []
        gradient_cache = {}
        for query, sites in enumerate(selected_sites):
            mode_directions = tensor_basis @ vectors[query]
            for site_number, row_index in enumerate(sites):
                ids = torch.as_tensor(
                    np.asarray(rows[row_index, :source_config["causal_position"] + 1], dtype=np.int64),
                    device="cuda",
                )[None]
                with torch.no_grad():
                    x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
                    native_write = block8.mlp.Down(
                        block8.mlp.Left(normalized) * block8.mlp.Right(normalized)
                    ) + block8.mlp.Down_bias
                    x8 = pre + native_write
                    x = normalized[0, source_config["causal_position"]].float()
                    mode_tangents, amplitudes = mode_write_tangents(block8, x, mode_directions)
                    analytic_logits, direct_logit_tangents, residual_logit_tangents = analytic_suffix(
                        model, x8, x0, value_bus, source_config["causal_position"], mode_tangents,
                    )
                # prefix_to_layer and analytic_suffix use inference mode. Clone
                # into ordinary tensors before building the assessment-only
                # autograd graph for the full suffix oracle.
                with torch.inference_mode(False):
                    assessment_x8 = x8.detach().clone()
                    assessment_x0 = x0.detach().clone()
                    assessment_bus = value_bus.detach().clone()
                if row_index not in gradient_cache:
                    seed = config["seed"] + corpus_index * 100000 + int(row_index)
                    gradient_cache[row_index] = stratified_fisher_sketch(
                        model, assessment_x8, assessment_x0, assessment_bus,
                        source_config["causal_position"], source_config["exact_top_tokens"],
                        source_config["tail_samples_per_split"], seed,
                    )[:4]
                first_g, first_w, second_g, second_w = gradient_cache[row_index]
                oracle_response = weighted_responses(first_g, first_w, mode_tangents)
                heldout_response = weighted_responses(second_g, second_w, mode_tangents)
                direct_response = output_fisher_response(analytic_logits, direct_logit_tangents)
                residual_response = output_fisher_response(analytic_logits, residual_logit_tangents)
                global_response = global_singular[:, None] * (global_vh @ mode_tangents.T)
                local_contribution = (values[query] * (x @ mode_directions).square()).abs()
                route_indices = {
                    "oracle": greedy_normalized_joint(
                        oracle_response, amplitudes, source_config["budget"],
                    ),
                    "direct_tensor": greedy_normalized_joint(
                        direct_response, amplitudes, source_config["budget"],
                    ),
                    "residual_only": greedy_normalized_joint(
                        residual_response, amplitudes, source_config["budget"],
                    ),
                    "global_full": greedy_normalized_joint(
                        global_response, amplitudes, source_config["budget"],
                    ),
                    "local_query": local_contribution.topk(source_config["budget"]).indices,
                }
                with torch.no_grad():
                    native_single = suffix_from_layer(
                        model, 8, x0, value_bus, x8,
                        source_config["causal_position"],
                    )[0]
                    native_logits, route_rows = finite_routes(
                        model, pre, normalized, x0, value_bus,
                        source_config["causal_position"], mode_directions, route_indices,
                        directions[:, query], source_config["edit_norm_fraction"],
                    )
                recomposition_error = float((native_single - analytic_logits).abs().max())
                if recomposition_error > 1e-5:
                    raise AssertionError(f"analytic native logit mismatch {recomposition_error:.3e}")
                batch_baseline_error = float((native_logits - native_single).abs().max())
                if batch_baseline_error > 1e-3:
                    raise AssertionError(f"batched native logit mismatch {batch_baseline_error:.3e}")
                oracle_effect = route_rows["oracle"]["effect"]
                oracle_objective = normalized_joint_objective(
                    heldout_response, amplitudes, route_indices["oracle"],
                )
                oracle_set = set(route_indices["oracle"].cpu().tolist())
                for name in config["routes"]:
                    fidelity, cosine, _ = fisher_fidelity(
                        native_logits, oracle_effect, route_rows[name]["effect"],
                    )
                    selected = route_indices[name]
                    route_rows[name].update({
                        "fidelity_to_oracle": fidelity,
                        "cosine_to_oracle": cosine,
                        "heldout_objective_efficiency": normalized_joint_objective(
                            heldout_response, amplitudes, selected,
                        ) / max(oracle_objective, 1e-30),
                        "oracle_support_recall": len(set(selected.cpu().tolist()) & oracle_set)
                        / source_config["budget"],
                    })
                    if name in ("direct_tensor", "residual_only"):
                        tangents = direct_logit_tangents if name == "direct_tensor" else residual_logit_tangents
                        predicted = selected_tangent(
                            tangents, amplitudes, selected,
                            source_config["edit_norm_fraction"] * x.norm(),
                        )
                        tangent_fidelity, tangent_cosine, _ = fisher_fidelity(
                            native_logits, route_rows[name]["effect"], predicted,
                        )
                        route_rows[name].update({
                            "tangent_to_finite_fidelity": tangent_fidelity,
                            "tangent_to_finite_cosine": tangent_cosine,
                        })
                    route_rows[name].pop("effect")
                records.append({
                    "query": query,
                    "row": int(row_index),
                    "analytic_recomposition_error": recomposition_error,
                    "batch_baseline_error": batch_baseline_error,
                    "routes": route_rows,
                })
            print(corpus, "query", query, "complete", flush=True)
        results[corpus] = {"summary": summarize(records, config["routes"]), "records": records}

    def route(corpus, name):
        return results[corpus]["summary"][name]

    hypotheses = {
        "T1_direct_tensor_recovers_oracle_direction": {
            "supported": all(route(c, "direct_tensor")["fidelity_to_oracle"] >= 0.75 for c in results),
        },
        "T2_direct_tensor_recovers_oracle_magnitude": {
            "supported": all(
                route(c, "direct_tensor")["native_kl"] >= 0.75 * route(c, "oracle")["native_kl"]
                for c in results
            ),
        },
        "T3_direct_tensor_recovers_oracle_objective": {
            "supported": all(route(c, "direct_tensor")["heldout_objective_efficiency"] >= 0.75 for c in results),
        },
        "T4_direct_tangent_predicts_finite_edit": {
            "supported": all(route(c, "direct_tensor")["tangent_to_finite_fidelity"] >= 0.85 for c in results),
        },
        "T5_bilinear_jacobians_add_value": {
            "supported": all(
                route(c, "direct_tensor")["fidelity_to_oracle"]
                - route(c, "residual_only")["fidelity_to_oracle"] >= 0.20
                for c in results
            ),
        },
        "T6_direct_tensor_beats_global_metric": {
            "supported": all(
                route(c, "direct_tensor")["fidelity_to_oracle"]
                - route(c, "global_full")["fidelity_to_oracle"] >= 0.20
                for c in results
            ),
        },
        "T7_direct_tensor_ood_stability": {
            "supported": abs(
                route("fineweb", "direct_tensor")["fidelity_to_oracle"]
                - route("wikitext", "direct_tensor")["fidelity_to_oracle"]
            ) <= 0.15,
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Analytic direct tensor Jacobian through Layers 9–17",
        "",
        "All 512 Layer-8 mode write tangents are propagated through the exact learned residual coefficients, RMSNorm Jacobians, and bilinear MLP Jacobians of Layers 9–17. Native attention writes are recomputed, but their derivatives are clamped to zero. The final RMSNorm, unembedding, bounded-logit derivative, and full-vocabulary softmax Fisher metric are exact. `Residual only` omits every downstream MLP Jacobian. No corpus bank, learned router, or suffix backpropagation is used for these two routes.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Direct tensor Jacobian](layer8_direct_tensor_jacobian.png)",
        "",
        "| Corpus | Route | Exact KL | KL/oracle | Oracle fidelity/cosine | Objective efficiency | Support recall | Tangent→finite fidelity/cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        oracle = result["summary"]["oracle"]
        for name in config["routes"]:
            row = result["summary"][name]
            tangent = (
                f"{row['tangent_to_finite_fidelity']:.3f}/{row['tangent_to_finite_cosine']:.3f}"
                if name in ("direct_tensor", "residual_only") else "—"
            )
            lines.append(
                f"| {corpus} | {name} | {row['native_kl']:.3e} | "
                f"{row['native_kl'] / max(oracle['native_kl'], 1e-30):.3f} | "
                f"{row['fidelity_to_oracle']:.3f}/{row['cosine_to_oracle']:.3f} | "
                f"{row['heldout_objective_efficiency']:.3f} | {row['oracle_support_recall']:.3f} | {tangent} |"
            )
    lines += [
        "",
        "Success would identify a weight-derived, state-specific causal metric in the tensor/residual path. Failure would show that the small attention-only finite effect can still be essential for choosing the right program, or that tangent selection does not survive the finite joint edit.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    names = config["routes"][1:]
    x = np.arange(len(names))
    width = 0.36
    for corpus_index, (corpus, result) in enumerate(results.items()):
        offset = (corpus_index - 0.5) * width
        axes[0].bar(x + offset, [result["summary"][n]["fidelity_to_oracle"] for n in names], width, label=corpus)
        axes[1].bar(x + offset, [result["summary"][n]["heldout_objective_efficiency"] for n in names], width, label=corpus)
    axes[0].set_title("Effect recovery")
    axes[0].set_ylabel("median Fisher fidelity")
    axes[1].set_title("Objective recovery")
    axes[1].set_ylabel("median efficiency")
    for axis in axes:
        axis.set_xticks(x, names, rotation=20)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
