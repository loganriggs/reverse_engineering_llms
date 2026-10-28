#!/usr/bin/env python3
"""Permutation red team for native-head subset specificity."""

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
from run_bilin18_transfer_state_fisher_chart import score_adjoint_set
from run_bilin18_transfer_state_invocation_chart import invocation_project, local_residual_basis
from run_bilin18_transfer_state_role_head_sparsity import (
    backward_head_program,
    head_map_from_labels,
    invocation_labels,
    weighted_relative_error,
)


CONFIG = Path("configs/bilin18_transfer_state_role_head_random_redteam.json")


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    head_config = json.loads(Path(config["source_config"]).read_text())
    head_result = json.loads(Path(config["source_result"]).read_text())
    localization_config = json.loads(Path(head_config["source_config"]).read_text())
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
    boundary = head_config["boundary"]
    queries = base["positions"]
    count = role_config["site_count"]
    residual_rank = read_config["coordinate_budget"] // count
    output = TENSOR_ROOT / f"{config['artifact_stem']}.json"
    report = TENSOR_ROOT / f"{config['artifact_stem'].upper()}.md"

    corpus = np.load(base["corpus"], mmap_mode="r")
    indices = np.asarray(read_result["evaluation_indices"], dtype=np.int64)
    split = head_config["discovery_rows"]
    panel = np.asarray(corpus[indices[split:], :base["sequence_length"]], dtype=np.int64)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    labels = invocation_labels(head_config["role_layers"], model.config.n_head)
    roles = set(head_config["roles"])
    full_map = head_map_from_labels(labels)
    rng = np.random.default_rng(config["seed"])
    selections = {}
    for rank in config["ranks"]:
        selections[f"top{rank}"] = [tuple(value) for value in head_result["selected"][f"top{rank}"]]
        for draw in range(config["random_draws"]):
            choice = rng.choice(len(labels), rank, replace=False)
            selections[f"random{rank}_{draw}"] = [labels[index] for index in choice]

    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    bank_labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in read_config["targets"]:
        target = bank_labels.index(f"L{layer}H{head}")
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((bank_labels[target], target, partner, scale, controls))
    accumulators = {
        label: {method: {"target2": 0., "prediction2": 0., "dot": 0.}
                for method in selections} for label, *_ in target_specs
    }
    errors = {method: [] for method in selections}
    positions_tensor = torch.as_tensor(queries, device="cuda")
    maximum_forward_error = 0.0
    for local_index, row in enumerate(panel):
        original_index = split + local_index
        print("head-random context", local_index + 1, "/", len(panel), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        (first, weights), _, _ = score_adjoint_set(
            model, trace[-1]["output"], queries,
            head_config["exact_top_tokens"], head_config["tail_samples_per_split"],
            head_config["seed"] + original_index,
        )
        full = backward_head_program(model, trace, first, boundary, roles, full_map)
        charts = {}
        for method, invocation_set in selections.items():
            program = backward_head_program(
                model, trace, first, boundary, roles, head_map_from_labels(invocation_set),
            )
            errors[method].append(weighted_relative_error(program, full, weights))
            sites = select_sites(program, weights, count)
            charts[method] = {
                position: local_residual_basis(program, weights, position, residual_rank)
                for position in sites
            }
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
            for method in selections:
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
                for method, acc in methods.items()}
        for label, methods in accumulators.items()
    }
    summary = {}
    for rank in config["ranks"]:
        random_methods = [f"random{rank}_{draw}" for draw in range(config["random_draws"])]
        top_method = f"top{rank}"
        top_fidelity = float(np.mean([
            results[label][top_method]["response_fidelity"] for label, *_ in target_specs
        ]))
        random_fidelity = np.asarray([
            np.mean([results[label][method]["response_fidelity"] for label, *_ in target_specs])
            for method in random_methods
        ])
        top_error = float(np.mean(errors[top_method]))
        random_error = np.asarray([np.mean(errors[method]) for method in random_methods])
        summary[str(rank)] = {
            "top_mean_fidelity": top_fidelity,
            "random_mean_fidelity": random_fidelity.tolist(),
            "random_fidelity_quantiles": np.quantile(random_fidelity, [.05, .5, .95]).tolist(),
            "top_mean_adjoint_error": top_error,
            "random_mean_adjoint_error": random_error.tolist(),
            "random_error_quantiles": np.quantile(random_error, [.05, .5, .95]).tolist(),
            "fidelity_random_percentile": float(np.mean(random_fidelity <= top_fidelity)),
            "error_random_percentile": float(np.mean(random_error <= top_error)),
        }
    p1 = summary["8"]["top_mean_fidelity"] > summary["8"]["random_fidelity_quantiles"][2]
    p2 = summary["16"]["top_mean_fidelity"] > summary["16"]["random_fidelity_quantiles"][2]
    p3 = summary["8"]["top_mean_adjoint_error"] < summary["8"]["random_error_quantiles"][0]
    p4 = summary["16"]["top_mean_adjoint_error"] < summary["16"]["random_error_quantiles"][0]
    hypotheses = {
        "P1_top8_beats_random_distribution": {"supported": p1},
        "P2_top16_beats_random_distribution": {"supported": p2},
        "P3_top8_has_lower_adjoint_error": {"supported": p3},
        "P4_top16_has_lower_adjoint_error": {"supported": p4},
        "P5_specific_native_head_sparsity_survives_redteam": {"supported": p1 and p3},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5,
                                             "maximum_relative_error": maximum_forward_error},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    output.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summary": summary,
        "selections": {name: [[layer, head] for layer, head in rows] for name, rows in selections.items()},
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Random-subset red team of native-head sparsity", "",
        "The frozen leave-one-out top-8 and top-16 Q+V invocation sets are compared with sixteen independently sampled native-head subsets at each rank. Every subset is evaluated on the same two held-out contexts and must prospectively generate its own sites and local payload bases before candidate-operator causal testing.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Permutation summary", "", "| Rank | Top fidelity | Random fidelity q05/median/q95 | Fidelity percentile | Top adjoint error | Random error q05/median/q95 | Error percentile |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for rank in config["ranks"]:
        row = summary[str(rank)]
        fq = row["random_fidelity_quantiles"]; eq = row["random_error_quantiles"]
        lines.append(
            f"| {rank} | {row['top_mean_fidelity']:.3f} | {fq[0]:.3f}/{fq[1]:.3f}/{fq[2]:.3f} | "
            f"{row['fidelity_random_percentile']:.3f} | {row['top_mean_adjoint_error']:.3f} | "
            f"{eq[0]:.3f}/{eq[1]:.3f}/{eq[2]:.3f} | {row['error_random_percentile']:.3f} |"
        )
    lines += [
        "", "## Guardrails", "",
        "Sixteen permutations give only a coarse empirical tail probability (resolution 1/16). The ranking is conditional leave-one-out importance and can miss cooperative subsets. Failure means the selected native identities are not distinguished from random subsets under these held-out behavioral and adjoint metrics; it does not mean the four-layer Q+V role program itself is unstructured.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
