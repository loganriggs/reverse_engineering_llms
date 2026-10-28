#!/usr/bin/env python3
"""Test reusable native-head sparsity inside the four-layer Q+V role program."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_attention_layer_path_program import rotate
from run_bilin18_destination_path_predictor import (
    analytic_candidate_paths,
    fidelity,
    median_angle_partner,
    propagate_candidate_tangent,
    response,
)
from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import build_function_bank
from run_bilin18_layer1_output_aware_coalitions import clear_rotary
from run_bilin18_layer1_suffix_degree_pilot import mlp_vjp
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_transfer_state_dynamic_read_selector import select_sites
from run_bilin18_transfer_state_fisher_chart import score_adjoint_set
from run_bilin18_transfer_state_invocation_chart import invocation_project, local_residual_basis
from run_bilin18_transfer_state_signed_role_address import backward_role_program
from run_bilin18_variable_function_atlas import spearman


CONFIG = Path("configs/bilin18_transfer_state_role_head_sparsity.json")


def invocation_labels(layers: list[int], heads: int) -> list[tuple[int, int]]:
    return [(layer, head) for layer in layers for head in range(heads)]


def masked_attention_role_vjp(block, mixed, bus, adjoints, roles, selected_heads):
    """VJP through selected Q/K/value roles and native heads."""
    clear_rotary(type("Wrapper", (), {"blocks": [block]})(), 0)
    attention = block.attn
    with torch.enable_grad():
        variable = mixed.detach().clone().requires_grad_(True)
        state = F.rms_norm(variable, (variable.shape[-1],))
        batch, length, dimension = state.shape
        heads, head_dim = attention.n_head, attention.head_dim

        def projected(module):
            raw = F.linear(state, module.weight.float()).view(batch, length, heads, head_dim)
            cosine, sine = attention.rotary(raw)
            return rotate(F.rms_norm(raw, (head_dim,)), cosine.float(), sine.float())

        q1, k1 = projected(attention.c_q), projected(attention.c_k)
        q2, k2 = projected(attention.c_q2), projected(attention.c_k2)
        current_value = F.linear(state, attention.c_v.weight.float()).view(batch, length, heads, head_dim)
        value = ((1 - attention.lamb.float()) * current_value
                 + attention.lamb.float() * bus.detach().view_as(current_value).float())
        causal = torch.tril(torch.ones(length, length, device=state.device, dtype=torch.bool))
        mask = torch.zeros(heads, device=state.device)
        mask[list(selected_heads)] = 1

        def score(query, key):
            return torch.einsum("bqhd,bkhd->bhqk", query, key) / head_dim

        def write(score1, score2, values):
            pattern = (score1 * score2).masked_fill(~causal, 0)
            attended = torch.einsum("bhqk,bkhd->bhqd", pattern, values)
            attended = attended * mask[None, :, None, None]
            flat = attended.transpose(1, 2).contiguous().view(batch, length, dimension)
            return F.linear(flat, attention.c_proj.weight.float())

        score1, score2 = score(q1, k1), score(q2, k2)
        outputs = {
            "q1": write(score(q1, k1.detach()), score2.detach(), value.detach()),
            "k1": write(score(q1.detach(), k1), score2.detach(), value.detach()),
            "q2": write(score1.detach(), score(q2, k2.detach()), value.detach()),
            "k2": write(score1.detach(), score(q2.detach(), k2), value.detach()),
            "value": write(score1.detach(), score2.detach(), value),
        }
        selected = sum((outputs[role] for role in roles), torch.zeros_like(outputs["q1"]))
        gradient, = torch.autograd.grad(selected, variable, adjoints, is_grads_batched=True)
    return gradient.detach().float()


def backward_head_program(model, trace, final_adjoint, boundary, roles, head_map):
    current = final_adjoint.float()
    for layer in range(len(model.blocks) - 1, boundary, -1):
        block = model.blocks[layer]
        updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current)
        selected_heads = head_map.get(layer, set())
        if selected_heads:
            previous = trace[layer - 1]["output"].float()
            x0 = trace[layer]["x0"].float()
            mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
            bus = trace[layer - 1]["value_bus"]
            updated = updated + masked_attention_role_vjp(
                block, mixed, bus, updated, roles, selected_heads,
            )
        current = block.lambdas[0].float() * updated
    return current


def head_map_from_labels(labels: list[tuple[int, int]]) -> dict[int, set[int]]:
    result = {}
    for layer, head in labels:
        result.setdefault(layer, set()).add(head)
    return result


def weighted_relative_error(approximate, target, weights):
    shape = (len(weights),) + (1,) * (target.ndim - 1)
    weight = weights.float().reshape(shape)
    numerator = ((approximate - target).square() * weight).sum()
    denominator = (target.square() * weight).sum().clamp_min(1e-30)
    return float(torch.sqrt(numerator / denominator))


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    localization_config = json.loads(Path(config["source_config"]).read_text())
    role_config = json.loads(Path(localization_config["source_config"]).read_text())
    attention_config = json.loads(Path(role_config["source_config"]).read_text())
    selector_config = json.loads(Path(attention_config["source_config"]).read_text())
    invocation_config = json.loads(Path(selector_config["source_config"]).read_text())
    read_config = json.loads(Path(invocation_config["source_config"]).read_text())
    read_result = json.loads(Path(invocation_config["source_result"]).read_text())
    fisher_config = json.loads(Path(read_config["source_config"]).read_text())
    dynamic_config = json.loads(Path(fisher_config["source_config"]).read_text())
    state_config = json.loads(Path(dynamic_config["source_config"]).read_text())
    base = json.loads(Path(state_config["source_config"]).read_text())
    boundary = config["boundary"]
    queries = base["positions"]
    count = role_config["site_count"]
    residual_rank = read_config["coordinate_budget"] // count
    output = TENSOR_ROOT / f"{config['artifact_stem']}.json"
    report = TENSOR_ROOT / f"{config['artifact_stem'].upper()}.md"

    corpus = np.load(base["corpus"], mmap_mode="r")
    indices = np.asarray(read_result["evaluation_indices"], dtype=np.int64)
    panel = np.asarray(corpus[indices, :base["sequence_length"]], dtype=np.int64)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    labels = invocation_labels(config["role_layers"], model.config.n_head)
    full_map = head_map_from_labels(labels)
    roles = set(config["roles"])
    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    bank_labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in read_config["targets"]:
        target = bank_labels.index(f"L{layer}H{head}")
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((bank_labels[target], target, partner, scale, controls))

    context_data = []
    maximum_recomposition_error = 0.0
    maximum_local_error = 0.0
    for row_index, row in enumerate(panel):
        print("head-sparsity discovery", row_index + 1, "/", len(panel), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        (first, weights), _, _ = score_adjoint_set(
            model, trace[-1]["output"], queries,
            config["exact_top_tokens"], config["tail_samples_per_split"],
            config["seed"] + row_index,
        )
        ordinary, local_error = backward_role_program(
            model, trace, first, [boundary],
            {"mlp": True, "roles": roles, "layers": set(config["role_layers"])},
            validate=True,
        )
        target = ordinary[boundary]
        full = backward_head_program(model, trace, first, boundary, roles, full_map)
        maximum_recomposition_error = max(
            maximum_recomposition_error,
            float((full - target).norm() / target.norm().clamp_min(1e-30)),
        )
        maximum_local_error = max(maximum_local_error, local_error)
        importance = []
        for index, invocation in enumerate(labels):
            kept = [label for label in labels if label != invocation]
            omitted = backward_head_program(
                model, trace, first, boundary, roles, head_map_from_labels(kept),
            )
            importance.append(weighted_relative_error(omitted, full, weights))
            print("  omit", index + 1, "/", len(labels), invocation, flush=True)
        context_data.append({
            "ids": ids,
            "trace": trace,
            "first": first,
            "weights": weights,
            "full": full,
            "importance": np.asarray(importance),
        })

    split = config["discovery_rows"]
    discovery_importance = np.mean([row["importance"] for row in context_data[:split]], axis=0)
    evaluation_importance = np.mean([row["importance"] for row in context_data[split:]], axis=0)
    ordering = np.argsort(discovery_importance)[::-1]
    rng = np.random.default_rng(config["seed"] + 100)
    random_order = rng.permutation(len(labels))
    selected = {
        f"top{rank}": [labels[index] for index in ordering[:rank]] for rank in config["ranks"]
    }
    selected.update({
        f"random{rank}": [labels[index] for index in random_order[:rank]] for rank in config["ranks"]
    })
    methods = list(selected)
    accumulators = {
        label: {method: {"target2": 0., "prediction2": 0., "dot": 0.}
                for method in methods} for label, *_ in target_specs
    }
    diagnostics = []
    positions_tensor = torch.as_tensor(queries, device="cuda")
    maximum_forward_error = 0.0
    for evaluation_index, data in enumerate(context_data[split:], start=split):
        print("head-sparsity evaluation", evaluation_index + 1, "/", len(context_data), flush=True)
        charts = {}
        context = {"row": evaluation_index, "programs": {}}
        for method, invocation_set in selected.items():
            program = backward_head_program(
                model, data["trace"], data["first"], boundary, roles,
                head_map_from_labels(invocation_set),
            )
            sites = select_sites(program, data["weights"], count)
            charts[method] = {
                position: local_residual_basis(program, data["weights"], position, residual_rank)
                for position in sites
            }
            context["programs"][method] = {
                "weighted_relative_error_to_all": weighted_relative_error(
                    program, data["full"], data["weights"],
                ),
                "sites": sites,
            }
        diagnostics.append(context)
        ids = data["ids"]
        for label, target, partner, scale, controls in target_specs:
            print(" evaluate", label, flush=True)
            with torch.no_grad():
                analytic = analytic_candidate_paths(
                    model, ids, bank[target], bank[partner], scale, controls, bank,
                    positions_tensor, capture_layers=[boundary], return_cache=True,
                )
                _, full_response, _ = response(model, analytic["final_state"], analytic["full"])
            target2 = float(full_response.square().sum())
            tangent = analytic["captured_tangents"][boundary]
            for method in methods:
                projected = invocation_project(tangent, charts[method])
                with torch.no_grad():
                    propagated, error = propagate_candidate_tangent(
                        model, projected, analytic["later_cache"], boundary,
                    )
                    maximum_forward_error = max(maximum_forward_error, error)
                    _, predicted, _ = response(
                        model, analytic["final_state"], propagated[:, :, positions_tensor],
                    )
                acc = accumulators[label][method]
                acc["target2"] += target2
                acc["prediction2"] += float(predicted.square().sum())
                acc["dot"] += float((predicted * full_response).sum())

    results = {
        label: {method: {"response_fidelity": fidelity(**acc)}
                for method, acc in methods_row.items()}
        for label, methods_row in accumulators.items()
    }
    top8_layers = len({layer for layer, _ in selected["top8"]})
    h1 = spearman(discovery_importance, evaluation_importance) >= .50
    h2 = all(results[label]["top8"]["response_fidelity"] >= results[label]["top36"]["response_fidelity"] - .10 for label, *_ in target_specs)
    h3 = all(results[label]["top8"]["response_fidelity"] >= results[label]["random8"]["response_fidelity"] + .10 for label, *_ in target_specs)
    h4 = all(results[label]["top16"]["response_fidelity"] >= results[label]["top36"]["response_fidelity"] - .05 for label, *_ in target_specs)
    h5 = top8_layers >= 3
    hypotheses = {
        "H0_all_head_program_recomposes_near4_qv": {"supported": maximum_recomposition_error <= 1e-5 and maximum_local_error <= 1e-5,
                                                      "maximum_recomposition_error": maximum_recomposition_error,
                                                      "maximum_local_error": maximum_local_error},
        "H1_head_importance_is_context_stable": {"supported": h1, "spearman": spearman(discovery_importance, evaluation_importance)},
        "H2_top8_heads_are_behaviorally_sufficient": {"supported": h2},
        "H3_top8_heads_beat_random": {"supported": h3},
        "H4_top16_heads_are_behaviorally_sufficient": {"supported": h4},
        "H5_selected_heads_span_multiple_layers": {"supported": h5, "layers": top8_layers},
        "H6_native_head_sparsity_is_supported": {"supported": h1 and h2 and h3 and h5},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5,
                                             "maximum_relative_error": maximum_forward_error},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    importance_rows = [{
        "layer": layer,
        "head": head,
        "discovery_importance": float(discovery_importance[index]),
        "evaluation_importance": float(evaluation_importance[index]),
        "discovery_rank": int(np.where(ordering == index)[0][0]) + 1,
    } for index, (layer, head) in enumerate(labels)]
    output.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "selected": {name: [[layer, head] for layer, head in rows] for name, rows in selected.items()},
        "importance": importance_rows,
        "diagnostics": diagnostics,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Native-head sparsity inside the four-layer Q+V program", "",
        "The 36 Layer-7--10 head invocations in the supported query+value recurrence are ranked only by leave-one-head-out weighted suffix-adjoint error on two discovery contexts. Frozen top-k and random-k sets are evaluated on two disjoint contexts. Each program must generate both its read sites and local rank-16 payload bases before held-out candidate-operator interventions.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Selected top eight", "", ", ".join(f"L{layer}H{head}" for layer, head in selected["top8"]), "", "## Held-out causal fidelity", "", "| Target | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for label, *_ in target_specs:
        lines.append("| " + label + " | " + " | ".join(f"{results[label][method]['response_fidelity']:.3f}" for method in methods) + " |")
    lines += ["", "## Head importance", "", "| Rank | Invocation | Discovery LOO error | Evaluation LOO error |", "|---:|---|---:|---:|"]
    for row in sorted(importance_rows, key=lambda item: item["discovery_rank"]):
        lines.append(f"| {row['discovery_rank']} | L{row['layer']}H{row['head']} | {row['discovery_importance']:.5f} | {row['evaluation_importance']:.5f} |")
    lines += [
        "", "## Guardrails", "",
        "Leave-one-out error is conditional on the other 35 invocations and is not additive importance. A sparse top-k win supports a reusable native execution subset for this Q+V program; failure can mean context-dependent routing or distributed redundancy rather than absence of head-level structure. Only two contexts are used per split, so ranking stability and matched random controls are required before interpreting individual heads.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "selected_top8": selected["top8"], "results": results}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
