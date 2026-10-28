#!/usr/bin/env python3
"""Select context-specific read sites from candidate-independent Fisher demand."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
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
from run_bilin18_transfer_state_invocation_chart import invocation_project, local_residual_basis


CONFIG = Path("configs/bilin18_transfer_state_dynamic_read_selector.json")


def position_demand(adjoint: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (adjoint.double().square() * weights[:, None, None, None]).sum((0, 1, 3))


def select_sites(adjoint: torch.Tensor, weights: torch.Tensor, count: int) -> list[int]:
    return torch.topk(position_demand(adjoint, weights), count).indices.cpu().tolist()


def lag_to_next_query(position: int, queries: list[int]) -> int:
    return min(query - position for query in queries if query >= position)


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    invocation_config = json.loads(Path(config["source_config"]).read_text())
    invocation_result = json.loads(Path(config["source_result"]).read_text())
    read_config = json.loads(Path(invocation_config["source_config"]).read_text())
    read_result = json.loads(Path(invocation_config["source_result"]).read_text())
    fisher_config = json.loads(Path(read_config["source_config"]).read_text())
    dynamic_config = json.loads(Path(fisher_config["source_config"]).read_text())
    state_config = json.loads(Path(dynamic_config["source_config"]).read_text())
    base = json.loads(Path(state_config["source_config"]).read_text())
    boundaries = state_config["boundaries"]
    queries = base["positions"]
    length = base["sequence_length"]
    budget = read_config["coordinate_budget"]
    counts = config["site_counts"]
    output = TENSOR_ROOT / f"{config['artifact_stem']}.json"
    report = TENSOR_ROOT / f"{config['artifact_stem'].upper()}.md"

    corpus = np.load(base["corpus"], mmap_mode="r")
    indices = np.asarray(read_result["evaluation_indices"], dtype=np.int64)
    panel = np.asarray(corpus[indices, :length], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
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

    methods = [f"dynamic_top{count}" for count in counts]
    accumulators = {
        label: {boundary: {method: {"target2": 0., "prediction2": 0., "dot": 0.}
                           for method in methods} for boundary in boundaries}
        for label, *_ in target_specs
    }
    diagnostics = []
    positions_tensor = torch.as_tensor(queries, device="cuda")
    maximum_forward_error = 0.0

    for row_index, row in enumerate(panel):
        print("dynamic-read context", row_index + 1, "/", len(panel), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        (first, first_weights), (second, second_weights), _ = score_adjoint_set(
            model, trace[-1]["output"], queries,
            fisher_config["exact_top_tokens"], fisher_config["tail_samples_per_split"],
            fisher_config["seed"] + 3000 + row_index,
        )
        first_by = backward_boundaries(model, trace, first, boundaries)
        second_by = backward_boundaries(model, trace, second, boundaries)
        charts = {}
        context = {"row": row_index, "boundaries": {}}
        for boundary in boundaries:
            charts[boundary] = {}
            first_top8 = set(select_sites(first_by[boundary], first_weights, 8))
            second_top8 = set(select_sites(second_by[boundary], second_weights, 8))
            selected = {}
            for count in counts:
                sites = select_sites(first_by[boundary], first_weights, count)
                rank = budget // count
                selected[str(count)] = sites
                charts[boundary][f"dynamic_top{count}"] = {
                    position: local_residual_basis(first_by[boundary], first_weights, position, rank)
                    for position in sites
                }
            context["boundaries"][str(boundary)] = {
                "top8_jaccard": len(first_top8 & second_top8) / len(first_top8 | second_top8),
                "selected_sites": selected,
                "top8_examples": [{
                    "position": position,
                    "lag_to_next_query": lag_to_next_query(position, queries),
                    "token": tokenizer.decode([int(row[position])]),
                } for position in selected["8"]],
            }
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
                for method in methods:
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
    fixed_names = {
        4: "current_specific32",
        8: "current_previous_specific16",
        16: "recent4_specific8",
        32: "recent8_specific4",
    }
    rows = []
    for label, *_ in target_specs:
        for boundary in boundaries:
            dynamic = results[label][str(boundary)]
            fixed = invocation_result["results"][label][str(boundary)]
            source = read_result["results"][label][str(boundary)]
            rows.append((label, boundary, dynamic, fixed, source))
    best_dynamic = [max(row[name]["response_fidelity"] for name in methods)
                    for _, _, row, _, _ in rows]
    nonlocal_flags = []
    for item in diagnostics:
        for boundary in boundaries:
            nonlocal_flags.extend(
                example["lag_to_next_query"] > 7
                for example in item["boundaries"][str(boundary)]["top8_examples"]
            )
    jaccards = {
        boundary: [item["boundaries"][str(boundary)]["top8_jaccard"] for item in diagnostics]
        for boundary in boundaries
    }
    hypotheses = {
        "S1_dynamic_sites_beat_fixed_reads": {"supported": all(
            best_value >= max(
                fixed[fixed_names[count]]["response_fidelity"] for count in counts
            ) + .05 for best_value, (_, _, _, fixed, _) in zip(best_dynamic, rows)
        )},
        "S2_dynamic_sites_approach_joint": {"supported": sum(
            best_value >= source["joint128"]["response_fidelity"] - .10
            for best_value, (_, _, _, _, source) in zip(best_dynamic, rows)
        ) >= len(rows) / 2},
        "S3_dynamic_reads_are_nonlocal": {"supported": np.mean(nonlocal_flags) >= .25,
                                            "fraction": float(np.mean(nonlocal_flags))},
        "S4_selected_sites_are_sampling_stable": {"supported": all(
            np.median(values) >= .50 for values in jaccards.values()
        )},
        "S5_joint_still_has_nonseparable_advantage": {"supported": any(
            source["joint128"]["response_fidelity"] >= best_value + .10
            for best_value, (_, _, _, _, source) in zip(best_dynamic, rows)
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
    }, indent=2, ensure_ascii=False, allow_nan=False))

    lines = [
        "# Dynamic read-site selector", "",
        "This follow-up asks whether the early joint-field gap comes from context-specific addressing. At each boundary, candidate-independent suffix-Fisher demand selects 4, 8, 16, or 32 token sites. Each selected invocation receives a local residual basis so every chart still uses exactly 128 runtime coordinates. The candidate replacement tangents are unseen during site and basis construction.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Causal results", "", "| Target | Boundary | Joint | Fixed best | Dynamic top4 | top8 | top16 | top32 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, boundary, dynamic, fixed, source in rows:
        fixed_best = max(fixed[fixed_names[count]]["response_fidelity"] for count in counts)
        lines.append(
            f"| {label} | {boundary} | {source['joint128']['response_fidelity']:.3f} | {fixed_best:.3f} | "
            + " | ".join(f"{dynamic[f'dynamic_top{count}']['response_fidelity']:.3f}" for count in counts)
            + " |"
        )
    lines += ["", "## Selector diagnostics", "", "| Boundary | Median top-8 split Jaccard |", "|---:|---:|"]
    for boundary in boundaries:
        lines.append(f"| {boundary} | {np.median(jaccards[boundary]):.3f} |")
    lines += [
        "", f"Across top-8 selections, {np.mean(nonlocal_flags):.1%} of sites are more than seven tokens before the next scored position.",
        "", "## Concrete selected reads", "",
    ]
    for item in diagnostics:
        lines.append(f"- Context row {item['row']}:")
        for boundary in boundaries:
            examples = item["boundaries"][str(boundary)]["top8_examples"]
            rendered = ", ".join(
                f"p{example['position']} lag{example['lag_to_next_query']} `{example['token'].replace('`', '')}`"
                for example in examples
            )
            lines.append(f"  - boundary {boundary}: {rendered}")
    lines += [
        "", "## Guardrails", "",
        "Fisher demand is an output-sensitive dynamic address generator, not yet a human-readable function. Selected token examples are descriptive and were not used to choose the locked thresholds. A win would localize the missing object to context-dependent addressing; it would not explain why those tokens were selected. A loss would support genuinely joint position-by-residual structure or a selector that cannot be reduced to hard sites.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
