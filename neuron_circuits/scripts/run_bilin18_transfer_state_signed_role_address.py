#!/usr/bin/env python3
"""Decompose dynamic read addresses into signed Q/K/value/MLP suffix programs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_destination_path_predictor import (
    analytic_candidate_paths,
    fidelity,
    median_angle_partner,
    propagate_candidate_tangent,
    response,
)
from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import build_function_bank
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, mlp_vjp
from run_bilin18_layer1_suffix_qk_role_pilot import ROLES, attention_role_vjps
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_transfer_state_dynamic_read_selector import position_demand, select_sites
from run_bilin18_transfer_state_fisher_chart import backward_boundaries, score_adjoint_set
from run_bilin18_transfer_state_invocation_chart import invocation_project, local_residual_basis
from run_bilin18_variable_function_atlas import spearman


CONFIG = Path("configs/bilin18_transfer_state_signed_role_address.json")


def role_specs() -> dict[str, dict]:
    return {
        "carry": {"mlp": False, "roles": set()},
        "mlp": {"mlp": True, "roles": set()},
        "queries": {"mlp": True, "roles": {"q1", "q2"}},
        "keys": {"mlp": True, "roles": {"k1", "k2"}},
        "value": {"mlp": True, "roles": {"value"}},
        "query_key": {"mlp": True, "roles": {"q1", "q2", "k1", "k2"}},
        "query_value": {"mlp": True, "roles": {"q1", "q2", "value"}},
        "key_value": {"mlp": True, "roles": {"k1", "k2", "value"}},
        "attention_no_mlp": {"mlp": False, "roles": set(ROLES)},
        "full": {"mlp": True, "roles": set(ROLES)},
    }


def backward_role_program(model, trace, final_adjoint, boundaries, spec, validate=False):
    current = final_adjoint.float()
    captured = {}
    maximum_local_error = 0.0
    final_layer = len(model.blocks) - 1
    if final_layer in boundaries:
        captured[final_layer] = current
    for layer in range(final_layer, min(boundaries), -1):
        block = model.blocks[layer]
        updated = current
        if spec["mlp"]:
            updated = updated + mlp_vjp(block, trace[layer]["pre"].float(), current)
        role_enabled = spec["roles"] and (
            "layers" not in spec or layer in spec["layers"]
        )
        if role_enabled:
            previous = trace[layer - 1]["output"].float()
            x0 = trace[layer]["x0"].float()
            mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
            bus = trace[layer - 1]["value_bus"]
            pieces = attention_role_vjps(block, mixed, bus, updated)
            if validate:
                native = attention_vjp(block, mixed, bus, updated)
                recomposed = sum(pieces.values())
                maximum_local_error = max(
                    maximum_local_error,
                    float((recomposed - native).norm() / native.norm().clamp_min(1e-30)),
                )
            routed = sum(
                (pieces[role] for role in spec["roles"]),
                torch.zeros_like(updated),
            )
            updated = updated + routed
        current = block.lambdas[0].float() * updated
        source = layer - 1
        if source in boundaries:
            captured[source] = current
    return captured, maximum_local_error


def demand_similarity(approximate, exact, weights, count):
    approximate_demand = position_demand(approximate, weights).cpu().numpy()
    exact_demand = position_demand(exact, weights).cpu().numpy()
    approximate_sites = np.argsort(approximate_demand)[-count:][::-1].tolist()
    exact_sites = np.argsort(exact_demand)[-count:][::-1].tolist()
    a, b = set(approximate_sites), set(exact_sites)
    cosine = float(
        approximate_demand @ exact_demand
        / max(np.linalg.norm(approximate_demand) * np.linalg.norm(exact_demand), 1e-30)
    )
    return {
        "cosine": cosine,
        "spearman": spearman(approximate_demand, exact_demand),
        "top_jaccard": len(a & b) / len(a | b),
        "sites": approximate_sites,
    }


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    attention_config = json.loads(Path(config["source_config"]).read_text())
    selector_config = json.loads(Path(attention_config["source_config"]).read_text())
    invocation_config = json.loads(Path(selector_config["source_config"]).read_text())
    read_config = json.loads(Path(invocation_config["source_config"]).read_text())
    read_result = json.loads(Path(invocation_config["source_result"]).read_text())
    fisher_config = json.loads(Path(read_config["source_config"]).read_text())
    dynamic_config = json.loads(Path(fisher_config["source_config"]).read_text())
    state_config = json.loads(Path(dynamic_config["source_config"]).read_text())
    base = json.loads(Path(state_config["source_config"]).read_text())
    boundaries = state_config["boundaries"]
    queries = base["positions"]
    count = config["site_count"]
    rank = read_config["coordinate_budget"] // count
    specs = role_specs()
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
    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in read_config["targets"]:
        target = labels.index(f"L{layer}H{head}")
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((labels[target], target, partner, scale, controls))

    causal_methods = []
    for name in specs:
        causal_methods.append(f"address_{name}")
        if name != "full":
            causal_methods.append(f"self_{name}")
    accumulators = {
        label: {boundary: {method: {"target2": 0., "prediction2": 0., "dot": 0.}
                           for method in causal_methods} for boundary in boundaries}
        for label, *_ in target_specs
    }
    diagnostics = []
    positions_tensor = torch.as_tensor(queries, device="cuda")
    maximum_local_error = 0.0
    maximum_full_error = 0.0
    maximum_forward_error = 0.0

    for row_index, row in enumerate(panel):
        print("signed-role context", row_index + 1, "/", len(panel), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        (first, weights), _, _ = score_adjoint_set(
            model, trace[-1]["output"], queries,
            config["exact_top_tokens"], config["tail_samples_per_split"],
            config["seed"] + row_index,
        )
        exact_by = backward_boundaries(model, trace, first, boundaries)
        programs = {}
        for name, spec in specs.items():
            print(" role", name, flush=True)
            programs[name], error = backward_role_program(
                model, trace, first, boundaries, spec, validate=name == "full",
            )
            maximum_local_error = max(maximum_local_error, error)
        charts = {}
        context = {"row": row_index, "boundaries": {}}
        for boundary in boundaries:
            full = programs["full"][boundary]
            exact = exact_by[boundary]
            maximum_full_error = max(
                maximum_full_error,
                float((full - exact).norm() / exact.norm().clamp_min(1e-30)),
            )
            full_sites = select_sites(full, weights, count)
            charts[boundary] = {}
            context["boundaries"][str(boundary)] = {"programs": {}}
            for name in specs:
                program = programs[name][boundary]
                metrics = demand_similarity(program, full, weights, count)
                sites = metrics.pop("sites")
                address_bases = {
                    position: local_residual_basis(full, weights, position, rank)
                    for position in sites
                }
                self_bases = {
                    position: local_residual_basis(program, weights, position, rank)
                    for position in sites
                }
                charts[boundary][f"address_{name}"] = address_bases
                if name != "full":
                    charts[boundary][f"self_{name}"] = self_bases
                context["boundaries"][str(boundary)]["programs"][name] = {
                    **metrics,
                    "sites": sites,
                }
            context["boundaries"][str(boundary)]["exact_sites"] = full_sites
        diagnostics.append(context)

        for label, target, partner, scale, controls in target_specs:
            print(" evaluate", label, flush=True)
            with torch.no_grad():
                analytic = analytic_candidate_paths(
                    model, ids, bank[target], bank[partner], scale, controls, bank,
                    positions_tensor, capture_layers=boundaries, return_cache=True,
                )
                _, full_response, _ = response(model, analytic["final_state"], analytic["full"])
            target2 = float(full_response.square().sum())
            for boundary in boundaries:
                tangent = analytic["captured_tangents"][boundary]
                for method in causal_methods:
                    projected = invocation_project(tangent, charts[boundary][method])
                    with torch.no_grad():
                        propagated, error = propagate_candidate_tangent(
                            model, projected, analytic["later_cache"], boundary,
                        )
                        maximum_forward_error = max(maximum_forward_error, error)
                        _, predicted, _ = response(
                            model, analytic["final_state"], propagated[:, :, positions_tensor],
                        )
                    acc = accumulators[label][boundary][method]
                    acc["target2"] += target2
                    acc["prediction2"] += float(predicted.square().sum())
                    acc["dot"] += float((predicted * full_response).sum())

    results = {
        label: {str(boundary): {
            method: {"response_fidelity": fidelity(**acc)}
            for method, acc in accumulators[label][boundary].items()
        } for boundary in boundaries} for label, *_ in target_specs
    }
    rows = [(label, boundary, results[label][str(boundary)])
            for label, *_ in target_specs for boundary in boundaries]
    singles = ["queries", "keys", "value"]
    pairs = ["query_key", "query_value", "key_value"]
    proper = [name for name in specs if name != "full"]
    median_jaccard = {
        name: {boundary: float(np.median([
            item["boundaries"][str(boundary)]["programs"][name]["top_jaccard"]
            for item in diagnostics
        ])) for boundary in boundaries} for name in specs
    }
    hypotheses = {
        "C0_full_role_program_recomposes_exact_suffix": {"supported": maximum_local_error <= 1e-5 and maximum_full_error <= 1e-5,
                                                           "maximum_local_error": maximum_local_error,
                                                           "maximum_full_error": maximum_full_error},
        "C1_mlp_only_does_not_recover_early_addresses": {"supported": median_jaccard["mlp"][6] < .50},
        "C2_transport_coalitions_approach_exact_address_effects": {"supported": sum(
            max(row[f"address_{name}"]["response_fidelity"] for name in pairs)
            >= row["address_full"]["response_fidelity"] - .10
            for _, _, row in rows
        ) >= len(rows) / 2},
        "C3_no_single_attention_role_is_early_sufficient": {"supported": not any(
            all(
                row[f"address_{name}"]["response_fidelity"]
                >= row["address_full"]["response_fidelity"] - .10
                for _, boundary, row in rows if boundary == 6
            ) for name in singles
        )},
        "C4_self_contained_subset_program_is_useful": {"supported": sum(
            max(row[f"self_{name}"]["response_fidelity"] for name in proper)
            >= row["address_full"]["response_fidelity"] - .10
            for _, _, row in rows
        ) >= len(rows) / 2},
        "C5_late_addressing_simplifies_to_mlp_carry": {"supported": all(
            row["address_mlp"]["response_fidelity"]
            >= row["address_full"]["response_fidelity"] - .10
            for _, boundary, row in rows if boundary == 17
        )},
        "C6_role_coalitions_have_nonadditive_value": {"supported": all(
            max(row[f"address_{name}"]["response_fidelity"] for name in pairs)
            >= max(row[f"address_{name}"]["response_fidelity"] for name in singles) + .05
            for _, boundary, row in rows if boundary == 6
        )},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5,
                                             "maximum_relative_error": maximum_forward_error},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    output.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "median_jaccard": median_jaccard,
        "diagnostics": diagnostics,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Signed suffix-role address programs", "",
        "The exact probability-weighted score adjoints are propagated backward with selected local Jacobian terms enabled: residual carry, bilinear-MLP VJP, Q1/Q2, K1/K2, and current-value/OV. Each truncated recurrence prospectively ranks eight read sites. `address` tests its selected sites with exact local Fisher payload bases; `self` also requires the truncated program to generate its own local payload bases. Every causal chart has 8 sites × rank 16 = 128 runtime coordinates and sees no candidate operator tangents during construction.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Address recovery", "", "| Program | " + " | ".join(f"b{boundary}" for boundary in boundaries) + " |", "|---|" + "---:|" * len(boundaries)]
    for name in specs:
        lines.append("| " + name + " | " + " | ".join(f"{median_jaccard[name][boundary]:.3f}" for boundary in boundaries) + " |")
    lines += ["", "## Causal address fidelity", "", "| Target | Boundary | Exact | MLP | Queries | Keys | Value | Q+K | Q+V | K+V | Attn no MLP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, boundary, row in rows:
        f = lambda name: row[f"address_{name}"]["response_fidelity"]
        lines.append(
            f"| {label} | {boundary} | {f('full'):.3f} | {f('mlp'):.3f} | {f('queries'):.3f} | "
            f"{f('keys'):.3f} | {f('value'):.3f} | {f('query_key'):.3f} | {f('query_value'):.3f} | "
            f"{f('key_value'):.3f} | {f('attention_no_mlp'):.3f} |"
        )
    lines += ["", "## Self-contained fidelity", "", "| Target | Boundary | MLP | Queries | Keys | Value | Q+K | Q+V | K+V | Attn no MLP |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, boundary, row in rows:
        f = lambda name: row[f"self_{name}"]["response_fidelity"]
        lines.append(
            f"| {label} | {boundary} | {f('mlp'):.3f} | {f('queries'):.3f} | {f('keys'):.3f} | "
            f"{f('value'):.3f} | {f('query_key'):.3f} | {f('query_value'):.3f} | "
            f"{f('key_value'):.3f} | {f('attention_no_mlp'):.3f} |"
        )
    lines += [
        "", "## Guardrails", "",
        "These are ablated recurrent programs, not an additive attribution: removing one role changes every earlier adjoint on which later role terms operate. Pairwise gains therefore establish compositional necessity, not a Shapley percentage. `Address` supplies exact payload geometry after testing only the router; `self` is the stricter surrogate. Score-adjoint sampling is smaller than the high-sample chart, so the exact full-program recomposition is the implementation gate and conclusions concern role families rather than literal axes.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "median_jaccard": median_jaccard}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
