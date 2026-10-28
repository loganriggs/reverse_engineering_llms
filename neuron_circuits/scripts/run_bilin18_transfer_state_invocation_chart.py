#!/usr/bin/env python3
"""Allow readable position calls to bind invocation-specific residual transforms."""

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
from run_bilin18_transfer_state_fisher_chart import backward_boundaries, score_adjoint_set
from run_bilin18_transfer_state_read_function_audit import orthonormalize
from run_bilin18_transfer_state_vocabulary import canonical_stability


CONFIG = Path("configs/bilin18_transfer_state_invocation_chart.json")


def local_residual_basis(adjoint: torch.Tensor, weights: torch.Tensor,
                         position: int, rank: int) -> torch.Tensor:
    matrix = adjoint[:, 0, position].float() * weights.float().sqrt()[:, None]
    gram = matrix @ matrix.T
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    keep = values > values[0].clamp_min(1e-30) * 1e-8
    values = values[keep]
    vectors = vectors[:, keep]
    basis = matrix.T @ vectors / values.sqrt()[None]
    return basis[:, :rank].float()


def invocation_project(tangent: torch.Tensor, bases: dict[int, torch.Tensor]) -> torch.Tensor:
    result = torch.zeros_like(tangent)
    for position, basis in bases.items():
        local = tangent[:, :, position]
        result[:, :, position] = (local @ basis) @ basis.T
    return result


def sites_for_offsets(queries: list[int], offsets: range | list[int]) -> list[int]:
    return [query - offset for offset in offsets for query in queries if query - offset >= 0]


def chart_specs(queries: list[int], budget: int) -> dict[str, tuple[list[int], int]]:
    specs = {
        "current_specific32": sites_for_offsets(queries, [0]),
        "current_previous_specific16": sites_for_offsets(queries, [0, 1]),
        "recent4_specific8": sites_for_offsets(queries, range(4)),
        "recent8_specific4": sites_for_offsets(queries, range(8)),
    }
    return {name: (sites, budget // len(sites)) for name, sites in specs.items()}


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    read_config = json.loads(Path(config["source_config"]).read_text())
    read_result = json.loads(Path(config["source_result"]).read_text())
    fisher_config = json.loads(Path(read_config["source_config"]).read_text())
    dynamic_config = json.loads(Path(fisher_config["source_config"]).read_text())
    state_config = json.loads(Path(dynamic_config["source_config"]).read_text())
    base = json.loads(Path(state_config["source_config"]).read_text())
    boundaries = state_config["boundaries"]
    queries = base["positions"]
    length = base["sequence_length"]
    budget = read_config["coordinate_budget"]
    specs = chart_specs(queries, budget)
    output = TENSOR_ROOT / f"{config['artifact_stem']}.json"
    report = TENSOR_ROOT / f"{config['artifact_stem'].upper()}.md"

    corpus = np.load(base["corpus"], mmap_mode="r")
    indices = np.asarray(read_result["evaluation_indices"], dtype=np.int64)
    panel = np.asarray(corpus[indices, :length], dtype=np.int64)
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

    methods = [*specs, "current_random32"]
    accumulators = {
        label: {boundary: {method: {"target2": 0., "prediction2": 0., "dot": 0.}
                           for method in methods} for boundary in boundaries}
        for label, *_ in target_specs
    }
    diagnostics = []
    positions_tensor = torch.as_tensor(queries, device="cuda")
    maximum_forward_error = 0.0
    generator = torch.Generator(device="cuda").manual_seed(read_config["evaluation_random_seed"] + 2)
    random_current = {
        position: orthonormalize(torch.randn(model.config.n_embd, 32, device="cuda", generator=generator))
        for position in queries
    }

    for row_index, row in enumerate(panel):
        print("invocation-chart context", row_index + 1, "/", len(panel), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        (first, first_weights), (second, second_weights), _ = score_adjoint_set(
            model, trace[-1]["output"], queries,
            fisher_config["exact_top_tokens"], fisher_config["tail_samples_per_split"],
            fisher_config["seed"] + 2000 + row_index,
        )
        first_by = backward_boundaries(model, trace, first, boundaries)
        second_by = backward_boundaries(model, trace, second, boundaries)
        charts = {}
        context_diagnostics = {"row": row_index, "boundaries": {}}
        for boundary in boundaries:
            charts[boundary] = {}
            for name, (sites, rank) in specs.items():
                charts[boundary][name] = {
                    position: local_residual_basis(first_by[boundary], first_weights, position, rank)
                    for position in sites
                }
            current_overlaps = []
            for position in queries:
                first_basis = charts[boundary]["current_specific32"][position]
                second_basis = local_residual_basis(second_by[boundary], second_weights, position, 32)
                available = min(first_basis.shape[1], second_basis.shape[1], 32)
                overlap, _ = canonical_stability(first_basis[:, :available], second_basis[:, :available])
                current_overlaps.append(overlap)
            context_diagnostics["boundaries"][str(boundary)] = {
                "mean_current_overlap32": float(np.mean(current_overlaps)),
                "per_query_current_overlap32": current_overlaps,
            }
        diagnostics.append(context_diagnostics)

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
                projections = {
                    name: invocation_project(tangent, charts[boundary][name]) for name in specs
                }
                projections["current_random32"] = invocation_project(tangent, random_current)
                for method, projected in projections.items():
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
    rows = []
    for label, *_ in target_specs:
        for boundary in boundaries:
            specific = results[label][str(boundary)]
            source = read_result["results"][label][str(boundary)]
            rows.append((label, boundary, specific, source))
    invocation_names = list(specs)
    best_specific = [max(row[name]["response_fidelity"] for name in invocation_names)
                     for _, _, row, _ in rows]
    stable = [item["boundaries"][str(boundary)]["mean_current_overlap32"]
              for item in diagnostics for boundary in boundaries]
    hypotheses = {
        "I1_invocation_specific_current_beats_shared": {"supported": all(
            row["current_specific32"]["response_fidelity"]
            >= source["current_sites4"]["response_fidelity"] + .10
            for _, _, row, source in rows
        )},
        "I2_invocation_specific_reads_approach_joint": {"supported": sum(
            best_value >= source["joint128"]["response_fidelity"] - .10
            for best_value, (_, _, _, source) in zip(best_specific, rows)
        ) >= len(rows) / 2},
        "I3_invocation_specific_current_beats_random": {"supported": all(
            row["current_specific32"]["response_fidelity"]
            >= row["current_random32"]["response_fidelity"] + .15
            for _, _, row, _ in rows
        )},
        "I4_current_invocation_subspaces_are_stable": {"supported": all(value >= .70 for value in stable)},
        "I5_joint_retains_unexplained_structure": {"supported": any(
            source["joint128"]["response_fidelity"] >= best_value + .10
            for best_value, (_, _, _, source) in zip(best_specific, rows)
        )},
        "I6_noncurrent_reads_are_necessary_somewhere": {"supported": any(
            max(row[name]["response_fidelity"] for name in invocation_names if name != "current_specific32")
            >= row["current_specific32"]["response_fidelity"] + .05
            for _, _, row, _ in rows
        )},
        "attention_jvp_forward_invariant": {
            "supported": maximum_forward_error <= 1e-5,
            "maximum_relative_error": maximum_forward_error,
        },
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    output.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "diagnostics": diagnostics,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Invocation-specific read-function charts", "",
        "This follow-up keeps the same four locked unseen contexts, three target heads, and 128-coordinate budget. The previous readable charts reused one suffix-Fisher residual basis across calls. Here each declared read invocation receives its own candidate-independent residual basis: 4 current sites × rank 32, 8 current/previous sites × rank 16, 16 recent sites × rank 8, or 32 recent sites × rank 4. This tests whether the unexplained joint-field advantage is merely call-specific variable/transform binding.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Causal results", "", "| Target | Boundary | Joint | Shared current | Specific current | Specific current+previous | Specific recent4 | Specific recent8 | Random current |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, boundary, row, source in rows:
        f = lambda name: row[name]["response_fidelity"]
        lines.append(
            f"| {label} | {boundary} | {source['joint128']['response_fidelity']:.3f} | "
            f"{source['current_sites4']['response_fidelity']:.3f} | {f('current_specific32'):.3f} | "
            f"{f('current_previous_specific16'):.3f} | {f('recent4_specific8'):.3f} | "
            f"{f('recent8_specific4'):.3f} | {f('current_random32'):.3f} |"
        )
    lines += ["", "## Stability", "", "| Boundary | Median current-site rank-32 split overlap |", "|---:|---:|"]
    for boundary in boundaries:
        values = [item["boundaries"][str(boundary)]["mean_current_overlap32"] for item in diagnostics]
        lines.append(f"| {boundary} | {np.median(values):.3f} |")
    lines += [
        "", "## Guardrails", "",
        "Invocation-specific bases have the same number of runtime coefficients as the shared and joint charts, but higher declaration cost than a reused residual transform. They remain output-metric and context conditioned. A gain supports typed call-specific binding, not human-readable residual variables. A remaining joint advantage can arise from nonseparable combinations spread over nonlocal positions, not necessarily a new semantic concept.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
