#!/usr/bin/env python3
"""Localize signed address/payload role invocations across suffix depth."""

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
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_transfer_state_dynamic_read_selector import select_sites
from run_bilin18_transfer_state_fisher_chart import backward_boundaries, score_adjoint_set
from run_bilin18_transfer_state_invocation_chart import invocation_project, local_residual_basis
from run_bilin18_transfer_state_signed_role_address import backward_role_program, demand_similarity


CONFIG = Path("configs/bilin18_transfer_state_role_layer_localization.json")


def localization_specs(boundary: int, final_layer: int, widths: list[int]) -> dict[str, dict]:
    suffix = list(range(boundary + 1, final_layer + 1))
    result = {
        "full": {"mlp": True, "roles": {"q1", "q2", "k1", "k2", "value"}, "layers": set(suffix)},
        "kv_all": {"mlp": True, "roles": {"k1", "k2", "value"}, "layers": set(suffix)},
        "qv_all": {"mlp": True, "roles": {"q1", "q2", "value"}, "layers": set(suffix)},
    }
    for width in widths:
        result[f"kv_near{width}"] = {"mlp": True, "roles": {"k1", "k2", "value"}, "layers": set(suffix[:width])}
        result[f"kv_far{width}"] = {"mlp": True, "roles": {"k1", "k2", "value"}, "layers": set(suffix[-width:])}
        result[f"qv_near{width}"] = {"mlp": True, "roles": {"q1", "q2", "value"}, "layers": set(suffix[:width])}
        result[f"qv_far{width}"] = {"mlp": True, "roles": {"q1", "q2", "value"}, "layers": set(suffix[-width:])}
    return result


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    role_config = json.loads(Path(config["source_config"]).read_text())
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
    rank = read_config["coordinate_budget"] // count
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
    specs = localization_specs(boundary, len(model.blocks) - 1, config["widths"])
    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in read_config["targets"]:
        target = labels.index(f"L{layer}H{head}")
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((labels[target], target, partner, scale, controls))

    methods = []
    for name in specs:
        methods.extend((f"address_{name}", f"self_{name}"))
    accumulators = {
        label: {method: {"target2": 0., "prediction2": 0., "dot": 0.}
                for method in methods} for label, *_ in target_specs
    }
    diagnostics = []
    positions_tensor = torch.as_tensor(queries, device="cuda")
    maximum_full_error = 0.0
    maximum_local_error = 0.0
    maximum_forward_error = 0.0

    for row_index, row in enumerate(panel):
        print("role-localization context", row_index + 1, "/", len(panel), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        (first, weights), _, _ = score_adjoint_set(
            model, trace[-1]["output"], queries,
            config["exact_top_tokens"], config["tail_samples_per_split"],
            config["seed"] + row_index,
        )
        exact = backward_boundaries(model, trace, first, [boundary])[boundary]
        programs = {}
        for name, spec in specs.items():
            print(" local role", name, flush=True)
            captured, error = backward_role_program(
                model, trace, first, [boundary], spec, validate=name == "full",
            )
            programs[name] = captured[boundary]
            maximum_local_error = max(maximum_local_error, error)
        maximum_full_error = max(
            maximum_full_error,
            float((programs["full"] - exact).norm() / exact.norm().clamp_min(1e-30)),
        )
        charts = {}
        context = {"row": row_index, "programs": {}}
        for name, program in programs.items():
            metrics = demand_similarity(program, programs["full"], weights, count)
            sites = metrics.pop("sites")
            charts[f"address_{name}"] = {
                position: local_residual_basis(programs["full"], weights, position, rank)
                for position in sites
            }
            charts[f"self_{name}"] = {
                position: local_residual_basis(program, weights, position, rank)
                for position in sites
            }
            context["programs"][name] = {**metrics, "sites": sites}
        diagnostics.append(context)

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
    median_jaccard = {
        name: float(np.median([item["programs"][name]["top_jaccard"] for item in diagnostics]))
        for name in specs
    }
    near8_contexts = [item["programs"]["kv_near8"]["top_jaccard"] for item in diagnostics]
    hypotheses = {
        "L0_full_recomposition": {"supported": maximum_full_error <= 1e-5 and maximum_local_error <= 1e-5,
                                   "maximum_full_error": maximum_full_error,
                                   "maximum_local_error": maximum_local_error},
        "L1_near4_key_value_recovers_addresses": {"supported": median_jaccard["kv_near4"] >= .75},
        "L2_near8_key_value_is_address_sufficient": {"supported": all(value >= .90 for value in near8_contexts)},
        "L3_near_roles_beat_far_roles": {"supported": median_jaccard["kv_near8"] >= median_jaccard["kv_far8"] + .10},
        "L4_near8_query_value_is_self_contained": {"supported": all(
            results[label]["self_qv_near8"]["response_fidelity"]
            >= results[label]["self_qv_all"]["response_fidelity"] - .10
            for label, *_ in target_specs
        )},
        "L5_near4_query_value_is_not_uniformly_sufficient": {"supported": any(
            results[label]["self_qv_near4"]["response_fidelity"]
            < results[label]["self_qv_all"]["response_fidelity"] - .10
            for label, *_ in target_specs
        )},
        "L6_role_invocations_are_depth_local": {"supported": all(value >= .90 for value in near8_contexts) and all(
            results[label]["self_qv_near8"]["response_fidelity"]
            >= results[label]["self_qv_all"]["response_fidelity"] - .10
            for label, *_ in target_specs
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
        "# Layer localization of signed role programs", "",
        "At Boundary 6, key+value address programs and query+value payload programs are restricted to the nearest or farthest 2, 4, or 8 downstream attention layers. MLP VJPs and residual carry remain enabled throughout. Address programs receive exact local payload bases; self-contained programs generate both site rankings and local bases. Every causal chart retains the same 128 runtime coordinates.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Address recovery", "", "| Program | Median top-8 Jaccard |", "|---|---:|"]
    for name in specs:
        lines.append(f"| {name} | {median_jaccard[name]:.3f} |")
    lines += ["", "## Causal fidelity", "", "| Target | Exact address | K+V all address | Q+V all self | Q+V near2 | near4 | near8 | far2 | far4 | far8 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, *_ in target_specs:
        row = results[label]
        f = lambda name: row[name]["response_fidelity"]
        lines.append(
            f"| {label} | {f('address_full'):.3f} | {f('address_kv_all'):.3f} | {f('self_qv_all'):.3f} | "
            f"{f('self_qv_near2'):.3f} | {f('self_qv_near4'):.3f} | {f('self_qv_near8'):.3f} | "
            f"{f('self_qv_far2'):.3f} | {f('self_qv_far4'):.3f} | {f('self_qv_far8'):.3f} |"
        )
    lines += [
        "", "## Guardrails", "",
        "Near and far programs are component-removal recurrences, not additive layer attribution. MLP and residual paths remain active everywhere, so a near-eight success means eight attention-role invocations suffice under that interface; it does not mean later layers are behaviorally unused. The test is localized to Boundary 6, where the previous read-program gap was largest.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "median_jaccard": median_jaccard, "results": results}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
