#!/usr/bin/env python3
"""Test readable positional functions inside dynamic suffix-Fisher charts."""

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
from run_bilin18_transfer_state_fisher_chart import (
    adjoint_covariance,
    backward_boundaries,
    score_adjoint_set,
)
from run_bilin18_transfer_state_joint_fisher_chart import joint_eigensystem, joint_project
from run_bilin18_transfer_state_vocabulary import canonical_stability, eigensystem


CONFIG = Path("configs/bilin18_transfer_state_read_function_audit.json")


def orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(matrix.float(), mode="reduced").Q


def site_basis(length: int, queries: list[int], offsets: range | list[int], device) -> torch.Tensor:
    columns = []
    for offset in offsets:
        for query in queries:
            position = query - offset
            if position >= 0:
                column = torch.zeros(length, device=device)
                column[position] = 1
                columns.append(column)
    return orthonormalize(torch.stack(columns, 1))


def block_basis(length: int, queries: list[int], device) -> torch.Tensor:
    columns = []
    start = 0
    for query in queries:
        column = torch.zeros(length, device=device)
        column[start:query + 1] = 1
        columns.append(column)
        start = query + 1
    return orthonormalize(torch.stack(columns, 1))


def prefix_basis(length: int, queries: list[int], device) -> torch.Tensor:
    columns = []
    for query in queries:
        column = torch.zeros(length, device=device)
        column[:query + 1] = 1
        columns.append(column)
    return orthonormalize(torch.stack(columns, 1))


def position_covariance(adjoint: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    scaled = adjoint.double() * weights.sqrt()[:, None, None, None]
    return torch.einsum("sbld,sbmd->lm", scaled, scaled)


def factorized_project(tangent: torch.Tensor, positions: torch.Tensor,
                       residuals: torch.Tensor) -> torch.Tensor:
    coefficients = torch.einsum("lp,bnld,dr->bnpr", positions, tangent, residuals)
    return torch.einsum("lp,bnpr,dr->bnld", positions, coefficients, residuals)


def subspace_coverage(target: torch.Tensor, container: torch.Tensor) -> float:
    target = target.float()
    container = container.float()
    return float((container.T @ target).square().sum() / target.shape[1])


def make_read_bases(length: int, queries: list[int], device) -> dict[str, torch.Tensor]:
    return {
        "current_sites4": site_basis(length, queries, [0], device),
        "current_previous8": site_basis(length, queries, [0, 1], device),
        "recent4x4": site_basis(length, queries, range(4), device),
        "recent8x4": site_basis(length, queries, range(8), device),
        "segment4": block_basis(length, queries, device),
        "causal_prefix4": prefix_basis(length, queries, device),
    }


def method_residual_rank(position_rank: int, budget: int) -> int:
    if budget % position_rank:
        raise ValueError("coordinate budget must divide by position rank")
    return budget // position_rank


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    fisher_config = json.loads(Path(config["source_config"]).read_text())
    dynamic_config = json.loads(Path(fisher_config["source_config"]).read_text())
    state_config = json.loads(Path(dynamic_config["source_config"]).read_text())
    base = json.loads(Path(state_config["source_config"]).read_text())
    boundaries = state_config["boundaries"]
    queries = base["positions"]
    length = base["sequence_length"]
    budget = config["coordinate_budget"]
    output = TENSOR_ROOT / f"{config['artifact_stem']}.json"
    report = TENSOR_ROOT / f"{config['artifact_stem'].upper()}.md"

    corpus = np.load(base["corpus"], mmap_mode="r")
    rng = np.random.default_rng(config["evaluation_random_seed"])
    indices = np.sort(rng.choice(
        np.arange(config["evaluation_index_low"], config["evaluation_index_high"]),
        config["evaluation_rows"], replace=False,
    ))
    panel = np.asarray(corpus[indices, :length], dtype=np.int64)

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in config["targets"]:
        target = labels.index(f"L{layer}H{head}")
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((labels[target], target, partner, scale, controls))

    read_bases = make_read_bases(length, queries, "cuda")
    method_position_ranks = {name: basis.shape[1] for name, basis in read_bases.items()}
    method_position_ranks.update({"fisher_position8": 8, "random_position8": 8})
    methods = ["joint128", "collapsed1", *method_position_ranks]
    accumulators = {
        label: {boundary: {method: {"target2": 0., "prediction2": 0., "dot": 0.}
                           for method in methods} for boundary in boundaries}
        for label, *_ in target_specs
    }
    diagnostics = []
    positions_tensor = torch.as_tensor(queries, device="cuda")
    maximum_forward_error = 0.0

    generator = torch.Generator(device="cuda").manual_seed(config["evaluation_random_seed"] + 1)
    random_position8 = orthonormalize(torch.randn(length, 8, device="cuda", generator=generator))

    for row_index, row in enumerate(panel):
        print("read-function context", row_index + 1, "/", len(panel), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        (first, first_weights), (second, second_weights), _ = score_adjoint_set(
            model, trace[-1]["output"], queries,
            fisher_config["exact_top_tokens"], fisher_config["tail_samples_per_split"],
            fisher_config["seed"] + 1000 + row_index,
        )
        first_by = backward_boundaries(model, trace, first, boundaries)
        second_by = backward_boundaries(model, trace, second, boundaries)
        charts = {}
        context_diagnostics = {"row": row_index, "boundaries": {}}
        for boundary in boundaries:
            residual_values, residual_vectors = eigensystem(adjoint_covariance(
                first_by[boundary], first_weights,
            ))
            joint_values, joint_axes = joint_eigensystem(first_by[boundary], first_weights)
            position_values, position_vectors = eigensystem(position_covariance(
                first_by[boundary], first_weights,
            ))
            _, second_position_vectors = eigensystem(position_covariance(
                second_by[boundary], second_weights,
            ))
            overlap8, _ = canonical_stability(position_vectors[:, :8], second_position_vectors[:, :8])
            readable_coverage = {
                name: subspace_coverage(position_vectors[:, :8], basis)
                for name, basis in read_bases.items()
            }
            context_diagnostics["boundaries"][str(boundary)] = {
                "position_overlap8": overlap8,
                "leading8_readable_coverage": readable_coverage,
                "position_rank90": int(torch.searchsorted(
                    position_values.cumsum(0) / position_values.sum().clamp_min(1e-30),
                    torch.tensor(.9, device=position_values.device, dtype=position_values.dtype),
                ).item() + 1),
                "top_mode_positions": torch.topk(position_vectors[:, 0].abs(), 8).indices.cpu().tolist(),
            }
            charts[boundary] = {
                "residual": residual_vectors[:, :budget].float(),
                "joint": joint_axes[:, :budget],
                "fisher_position8": position_vectors[:, :8].float(),
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
                residual = charts[boundary]["residual"]
                projections = {
                    "joint128": joint_project(tangent, charts[boundary]["joint"]),
                    "collapsed1": factorized_project(
                        tangent, torch.eye(length, device="cuda"), residual[:, :1],
                    ),
                }
                for name, position_basis in read_bases.items():
                    residual_rank = method_residual_rank(position_basis.shape[1], budget)
                    projections[name] = factorized_project(
                        tangent, position_basis, residual[:, :residual_rank],
                    )
                projections["fisher_position8"] = factorized_project(
                    tangent, charts[boundary]["fisher_position8"], residual[:, :16],
                )
                projections["random_position8"] = factorized_project(
                    tangent, random_position8, residual[:, :16],
                )
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

    results = {}
    for label, *_ in target_specs:
        results[label] = {}
        for boundary in boundaries:
            results[label][str(boundary)] = {
                method: {"response_fidelity": fidelity(**acc)}
                for method, acc in accumulators[label][boundary].items()
            }
    rows = [results[label][str(boundary)] for label, *_ in target_specs for boundary in boundaries]
    readable = list(read_bases)
    best_readable = [max(row[name]["response_fidelity"] for name in readable) for row in rows]
    position_overlaps = [
        item["boundaries"][str(boundary)]["position_overlap8"]
        for item in diagnostics for boundary in boundaries
    ]
    hypotheses = {
        "R1_known_reads_are_early_useful": {"supported": all(
            max(results[label]["6"][name]["response_fidelity"] for name in readable) >= .50
            for label, *_ in target_specs
        )},
        "R2_known_reads_beat_random": {"supported": all(
            best_value >= row["random_position8"]["response_fidelity"] + .10
            for best_value, row in zip(best_readable, rows)
        )},
        "R3_learned_positions_beat_random": {"supported": all(
            row["fisher_position8"]["response_fidelity"]
            >= row["random_position8"]["response_fidelity"] + .15 for row in rows
        )},
        "R4_position_chart_is_sampling_stable": {"supported": all(value >= .70 for value in position_overlaps)},
        "R5_unrestricted_joint_has_unexplained_structure": {"supported": any(
            row["joint128"]["response_fidelity"] >= best_value + .10
            for best_value, row in zip(best_readable, rows)
        )},
        "R6_some_known_read_approaches_joint": {"supported": sum(
            best_value >= row["joint128"]["response_fidelity"] - .10
            for best_value, row in zip(best_readable, rows)
        ) >= len(rows) / 2},
        "attention_jvp_forward_invariant": {
            "supported": maximum_forward_error <= 1e-5,
            "maximum_relative_error": maximum_forward_error,
        },
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])

    output.write_text(json.dumps({
        "config": config,
        "evaluation_indices": indices.tolist(),
        "hypotheses": hypotheses,
        "diagnostics": diagnostics,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Read-function audit of dynamic sequence fields", "",
        "Four previously unused FineWeb rows (indices 19000--19999) and three new destination heads were fixed before execution. Every factorized chart uses exactly 128 runtime coefficients: p positional functions times 128/p suffix-Fisher residual directions. The unrestricted joint rank-128 chart and collapsed residual rank-1 chart are matched 128-coordinate references. Charts use native state, weights, and the output Fisher metric but never the candidate replacement tangents.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Causal results", "", "| Target | Boundary | Joint | Collapsed | Current | Current+previous | Recent4 | Recent8 | Segment | Prefix | Fisher-pos | Random-pos |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, *_ in target_specs:
        for boundary in boundaries:
            row = results[label][str(boundary)]
            f = lambda name: row[name]["response_fidelity"]
            lines.append(
                f"| {label} | {boundary} | {f('joint128'):.3f} | {f('collapsed1'):.3f} | "
                f"{f('current_sites4'):.3f} | {f('current_previous8'):.3f} | {f('recent4x4'):.3f} | "
                f"{f('recent8x4'):.3f} | {f('segment4'):.3f} | {f('causal_prefix4'):.3f} | "
                f"{f('fisher_position8'):.3f} | {f('random_position8'):.3f} |"
            )
    lines += ["", "## Positional generator diagnostics", "", "| Boundary | Median split overlap (rank 8) | Median readable coverage of Fisher position span | Median position rank90 |", "|---:|---:|---:|---:|"]
    for boundary in boundaries:
        cells = [item["boundaries"][str(boundary)] for item in diagnostics]
        coverage = [max(cell["leading8_readable_coverage"].values()) for cell in cells]
        lines.append(
            f"| {boundary} | {np.median([cell['position_overlap8'] for cell in cells]):.3f} | "
            f"{np.median(coverage):.3f} | {np.median([cell['position_rank90'] for cell in cells]):.0f} |"
        )
    lines += [
        "", "## Guardrails", "",
        "A readable position span paired with opaque Fisher residual directions tests only the read-function part of the ontology; it does not make the transported variable itself human-readable. The Fisher-position chart is a candidate-independent dynamic oracle, not a semantic label. The random control shares rank and residual directions but not causal positional structure. All methods are matched in runtime coefficients, not literal basis-storage cost. Negative results can mean nonseparable position/residual coupling rather than absence of reusable functions.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
