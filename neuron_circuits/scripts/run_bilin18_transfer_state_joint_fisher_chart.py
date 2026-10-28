#!/usr/bin/env python3
"""Prospective joint sequence-residual suffix-Fisher charts on new targets."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_destination_path_predictor import (
    analytic_candidate_paths,
    cosine,
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
from run_bilin18_transfer_state_vocabulary import canonical_stability, eigensystem, project, rank_summary


CONFIG = Path("configs/bilin18_transfer_state_joint_fisher_chart.json")
OUTPUT = TENSOR_ROOT / "bilin18_transfer_state_joint_fisher_chart.json"
REPORT = TENSOR_ROOT / "BILIN18_TRANSFER_STATE_JOINT_FISHER_CHART.md"
STATIC_RESULT = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.json"


def joint_eigensystem(adjoint, weights):
    """Right singular field axes from a probability-weighted adjoint sketch."""
    weighted = adjoint.float() * weights.float().sqrt()[:, None, None, None]
    matrix = weighted.flatten(1)
    gram = matrix @ matrix.T
    values, vectors = torch.linalg.eigh(gram)
    order = torch.argsort(values, descending=True)
    values = values[order].clamp_min(0)
    vectors = vectors[:, order]
    keep = values > values[0].clamp_min(1e-30) * 1e-8
    values = values[keep]
    vectors = vectors[:, keep]
    axes = matrix.T @ vectors / values.sqrt()[None]
    return values.double(), axes.float()


def joint_project(tangent, axes):
    shape = tangent.shape
    flat = tangent.flatten(2)
    return ((flat @ axes) @ axes.T).reshape(shape)


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    artifact_stem = config.get("artifact_stem", "bilin18_transfer_state_joint_fisher_chart")
    output_path = TENSOR_ROOT / f"{artifact_stem}.json"
    report_path = TENSOR_ROOT / f"{artifact_stem.upper()}.md"
    fisher_config = json.loads(Path(config["source_config"]).read_text())
    dynamic_config = json.loads(Path(fisher_config["source_config"]).read_text())
    state_config = json.loads(Path(dynamic_config["source_config"]).read_text())
    base = json.loads(Path(state_config["source_config"]).read_text())
    static_result = json.loads(STATIC_RESULT.read_text())
    boundaries = state_config["boundaries"]
    positions = base["positions"]
    corpus = np.load(base["corpus"], mmap_mode="r")
    indices = np.asarray(static_result["evaluation_indices"], dtype=np.int64)
    raw = np.asarray(corpus[indices, :base["sequence_length"]], dtype=np.int64)
    heldout = raw[len(raw) // 2:][:state_config["causal_rows"]]

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in config["targets"]:
        label = f"L{layer}H{head}"; target = labels.index(label)
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((label, target, partner, scale, controls))
    positions_tensor = torch.as_tensor(positions, device="cuda")
    methods = ("collapsed", "joint")
    accumulators = {
        label: {
            boundary: {
                method: {
                    rank: {"target2": 0.0, "prediction2": 0.0, "dot": 0.0}
                    for rank in config["ranks"]
                }
                for method in methods
            }
            for boundary in boundaries
        }
        for label, *_ in target_specs
    }
    chart_rows = []
    maximum_forward_error = 0.0
    for row_index, row in enumerate(heldout):
        print("joint Fisher chart context", row_index + 1, "/", len(heldout), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        (first, first_weights), (second, second_weights), metadata = score_adjoint_set(
            model, trace[-1]["output"], positions,
            fisher_config["exact_top_tokens"], fisher_config["tail_samples_per_split"],
            fisher_config["seed"] + row_index,
        )
        first_by = backward_boundaries(model, trace, first, boundaries)
        second_by = backward_boundaries(model, trace, second, boundaries)
        charts = {}
        context_summary = {"row": row_index, "boundaries": {}, "top_mass": [item["top_probability_mass"] for item in metadata]}
        for boundary in boundaries:
            residual_values, residual_vectors = eigensystem(adjoint_covariance(
                first_by[boundary], first_weights,
            ))
            second_residual_values, second_residual_vectors = eigensystem(adjoint_covariance(
                second_by[boundary], second_weights,
            ))
            joint_values, joint_axes = joint_eigensystem(first_by[boundary], first_weights)
            second_joint_values, second_joint_axes = joint_eigensystem(second_by[boundary], second_weights)
            residual_overlap, _ = canonical_stability(residual_vectors[:, :64], second_residual_vectors[:, :64])
            joint_overlap, _ = canonical_stability(joint_axes[:, :64], second_joint_axes[:, :64])
            charts[boundary] = {
                "collapsed": residual_vectors[:, :max(config["ranks"])].float(),
                "joint": joint_axes[:, :max(config["ranks"])].float(),
            }
            context_summary["boundaries"][str(boundary)] = {
                "collapsed_overlap64": residual_overlap,
                "joint_overlap64": joint_overlap,
                "collapsed_rank": rank_summary(residual_values),
                "joint_rank": rank_summary(joint_values),
                "second_collapsed_rank": rank_summary(second_residual_values),
                "second_joint_rank": rank_summary(second_joint_values),
            }
        chart_rows.append(context_summary)

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
                    for method in methods:
                        full_basis = charts[boundary][method]
                        for rank in config["ranks"]:
                            available = min(rank, full_basis.shape[1])
                            if method == "collapsed":
                                projected = project(tangent, full_basis[:, :available])
                            else:
                                projected = joint_project(tangent, full_basis[:, :available])
                            propagated, error = propagate_candidate_tangent(
                                model, projected, analytic["later_cache"], boundary,
                            )
                            maximum_forward_error = max(maximum_forward_error, error)
                            _, predicted, _ = response(
                                model, analytic["final_state"], propagated[:, :, positions_tensor],
                            )
                            accumulator = accumulators[label][boundary][method][rank]
                            accumulator["target2"] += target2
                            accumulator["prediction2"] += float(predicted.square().sum())
                            accumulator["dot"] += float((predicted * full_response).sum())

    results = {}
    for label, *_ in target_specs:
        results[label] = {}
        for boundary in boundaries:
            row = {"methods": {}}
            for method in methods:
                row["methods"][method] = {}
                for rank in config["ranks"]:
                    item = accumulators[label][boundary][method][rank]
                    row["methods"][method][str(rank)] = {
                        "response_fidelity": fidelity(item["target2"], item["prediction2"], item["dot"]),
                        "response_cosine": cosine(item["target2"], item["prediction2"], item["dot"]),
                    }
            row["joint_overlap64_median"] = float(np.median([
                item["boundaries"][str(boundary)]["joint_overlap64"] for item in chart_rows
            ]))
            row["collapsed_overlap64_median"] = float(np.median([
                item["boundaries"][str(boundary)]["collapsed_overlap64"] for item in chart_rows
            ]))
            row["joint_rank90_median"] = float(np.median([
                item["boundaries"][str(boundary)]["joint_rank"]["rank90"] for item in chart_rows
            ]))
            results[label][str(boundary)] = row

    flat = [results[label][str(boundary)] for label, *_ in target_specs for boundary in boundaries]
    j1 = all(row["methods"]["joint"]["64"]["response_fidelity"] >= row["methods"]["collapsed"]["64"]["response_fidelity"] + .10 for row in flat)
    j2 = all(row["methods"]["joint"]["64"]["response_fidelity"] >= .85 for row in flat)
    j3 = all(row["methods"]["joint"]["128"]["response_fidelity"] >= .95 for row in flat)
    j4 = all(row["joint_overlap64_median"] >= .70 for row in flat)
    per_target = {
        label: all(
            results[label][str(boundary)]["methods"]["joint"]["64"]["response_fidelity"]
            >= results[label][str(boundary)]["methods"]["collapsed"]["64"]["response_fidelity"] + .10
            and results[label][str(boundary)]["methods"]["joint"]["64"]["response_fidelity"] >= .85
            for boundary in boundaries
        )
        for label, *_ in target_specs
    }
    hypotheses = {
        "J1_joint_chart_beats_position_collapsed_chart": {"supported": j1},
        "J2_joint_rank64_is_causally_useful": {"supported": j2},
        "J3_joint_rank128_is_high_fidelity": {"supported": j3},
        "J4_joint_chart_is_sampling_stable": {"supported": j4},
        "J5_cross_position_structure_generalizes_to_new_targets": {"supported": all(per_target.values()), "per_target": per_target},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5, "maximum_relative_error": maximum_forward_error},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    output_path.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "chart_rows": chart_rows,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Joint sequence-residual suffix-Fisher charts", "",
        "Three new target heads were fixed before this experiment. The prior chart collapses each suffix score gradient over positions into a residual-axis covariance, allowing separate coefficients at every position. The joint chart instead diagonalizes complete sequence×residual gradient fields, preserving structured cross-position relationships. Both charts are generated from the same native context and Fisher sketch and see no target replacement tangents.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Prospective new-target results", "", "| Target | Boundary | Joint overlap / rank90 | Collapsed r64/r128 | Joint r64/r128 |", "|---|---:|---:|---:|---:|"]
    for label, *_ in target_specs:
        for boundary in boundaries:
            row = results[label][str(boundary)]
            collapsed, joint = row["methods"]["collapsed"], row["methods"]["joint"]
            lines.append(
                f"| {label} | {boundary} | {row['joint_overlap64_median']:.3f} / {row['joint_rank90_median']:.0f} | "
                f"{collapsed['64']['response_fidelity']:.3f} / {collapsed['128']['response_fidelity']:.3f} | "
                f"{joint['64']['response_fidelity']:.3f} / {joint['128']['response_fidelity']:.3f} |"
            )
    lines += [
        "", "## Guardrails", "",
        "The joint chart has rank coordinates over the complete sequence field, while the collapsed residual chart has rank residual directions with independent coefficients at every token. They therefore have different runtime degrees of freedom despite sharing the same nominal rank. A joint-chart win supports structured routing but is not a parameter-count-matched compression result. Tail-sketch overlap and exact suffix fidelity are reported separately; unstable literal axes should not be assigned individual semantic names.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(report_path, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
