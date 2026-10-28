#!/usr/bin/env python3
"""Predict dynamic Fisher read sites from native QK attention routing."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

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
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_transfer_state_dynamic_read_selector import select_sites
from run_bilin18_transfer_state_fisher_chart import backward_boundaries, score_adjoint_set
from run_bilin18_transfer_state_invocation_chart import invocation_project, local_residual_basis


CONFIG = Path("configs/bilin18_transfer_state_attention_address.json")


@torch.inference_mode()
def normalized_attention_patterns(model, trace) -> list[torch.Tensor]:
    patterns = []
    x0 = trace[0]["x0"].float()
    previous = x0
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
        normalized = F.rms_norm(mixed, (model.config.n_embd,))
        pattern = attention_details(
            block.attn, normalized, None, model.config, model.tt,
        )["pattern"].float().abs().mean(1)[0]
        pattern = pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-12)
        patterns.append(pattern)
        previous = trace[layer]["output"].float()
    return patterns


def attention_address_scores(patterns: list[torch.Tensor], boundary: int,
                             queries: list[int]) -> dict[str, torch.Tensor]:
    length = patterns[0].shape[0]
    direct = torch.zeros(length, device=patterns[0].device)
    for pattern in patterns[boundary + 1:]:
        direct += pattern[queries].sum(0)
    rollout = torch.zeros(length, device=patterns[0].device)
    rollout[queries] = 1
    for pattern in reversed(patterns[boundary + 1:]):
        rollout = rollout + pattern.T @ rollout
        rollout = rollout / rollout.sum().clamp_min(1e-12)
    return {"direct": direct, "rollout": rollout}


def top_jaccard(first: list[int], second: list[int]) -> float:
    a, b = set(first), set(second)
    return len(a & b) / len(a | b)


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    selector_config = json.loads(Path(config["source_config"]).read_text())
    selector_result = json.loads(Path(config["dynamic_result"]).read_text())
    invocation_config = json.loads(Path(selector_config["source_config"]).read_text())
    invocation_result = json.loads(Path(selector_config["source_result"]).read_text())
    read_config = json.loads(Path(invocation_config["source_config"]).read_text())
    read_result = json.loads(Path(invocation_config["source_result"]).read_text())
    fisher_config = json.loads(Path(read_config["source_config"]).read_text())
    dynamic_config = json.loads(Path(fisher_config["source_config"]).read_text())
    state_config = json.loads(Path(dynamic_config["source_config"]).read_text())
    base = json.loads(Path(state_config["source_config"]).read_text())
    boundaries = state_config["boundaries"]
    queries = base["positions"]
    length = base["sequence_length"]
    count = config["site_count"]
    rank = read_config["coordinate_budget"] // count
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

    methods = ["direct", "rollout"]
    accumulators = {
        label: {boundary: {method: {"target2": 0., "prediction2": 0., "dot": 0.}
                           for method in methods} for boundary in boundaries}
        for label, *_ in target_specs
    }
    diagnostics = []
    positions_tensor = torch.as_tensor(queries, device="cuda")
    maximum_forward_error = 0.0

    for row_index, row in enumerate(panel):
        print("attention-address context", row_index + 1, "/", len(panel), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
            patterns = normalized_attention_patterns(model, trace)
        (first, first_weights), _, _ = score_adjoint_set(
            model, trace[-1]["output"], queries,
            fisher_config["exact_top_tokens"], fisher_config["tail_samples_per_split"],
            fisher_config["seed"] + 4000 + row_index,
        )
        first_by = backward_boundaries(model, trace, first, boundaries)
        charts = {}
        context = {"row": row_index, "boundaries": {}}
        for boundary in boundaries:
            fisher_sites = select_sites(first_by[boundary], first_weights, count)
            address_scores = attention_address_scores(patterns, boundary, queries)
            selected = {
                method: torch.topk(scores, count).indices.cpu().tolist()
                for method, scores in address_scores.items()
            }
            charts[boundary] = {
                method: {
                    position: local_residual_basis(first_by[boundary], first_weights, position, rank)
                    for position in sites
                } for method, sites in selected.items()
            }
            context["boundaries"][str(boundary)] = {
                "fisher_sites": fisher_sites,
                "direct_sites": selected["direct"],
                "rollout_sites": selected["rollout"],
                "direct_jaccard": top_jaccard(fisher_sites, selected["direct"]),
                "rollout_jaccard": top_jaccard(fisher_sites, selected["rollout"]),
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
    rows = []
    for label, *_ in target_specs:
        for boundary in boundaries:
            rows.append((
                label, boundary, results[label][str(boundary)],
                selector_result["results"][label][str(boundary)]["dynamic_top8"]["response_fidelity"],
                invocation_result["results"][label][str(boundary)]["current_previous_specific16"]["response_fidelity"],
                read_result["results"][label][str(boundary)]["joint128"]["response_fidelity"],
            ))
    median_jaccard = {
        method: {boundary: float(np.median([
            item["boundaries"][str(boundary)][f"{method}_jaccard"] for item in diagnostics
        ])) for boundary in boundaries} for method in methods
    }
    mean_fidelity = {
        method: float(np.mean([row[method]["response_fidelity"] for _, _, row, *_ in rows]))
        for method in methods
    }
    hypotheses = {
        "A1_attention_rollout_recovers_fisher_sites": {"supported": all(
            median_jaccard["rollout"][boundary] >= .50 for boundary in boundaries
        )},
        "A2_attention_addresses_preserve_causal_effect": {"supported": sum(
            row["rollout"]["response_fidelity"] >= fisher_top8 - .10
            for _, _, row, fisher_top8, _, _ in rows
        ) >= len(rows) / 2},
        "A3_attention_addresses_improve_fixed_reads_early": {"supported": all(
            row["rollout"]["response_fidelity"] >= fixed + .05
            for _, boundary, row, _, fixed, _ in rows if boundary == 6
        )},
        "A4_rollout_beats_direct_attention": {"supported": (
            np.median(list(median_jaccard["rollout"].values())) > np.median(list(median_jaccard["direct"].values()))
            and mean_fidelity["rollout"] > mean_fidelity["direct"]
        )},
        "A5_attention_addresses_do_not_fully_explain_joint": {"supported": any(
            joint >= row["rollout"]["response_fidelity"] + .10
            for _, _, row, _, _, joint in rows
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
        "median_jaccard": median_jaccard,
        "mean_fidelity": mean_fidelity,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Attention-derived dynamic read addresses", "",
        "Native bilinear QK patterns are converted to nonnegative row-normalized routing weights. `direct` sums query-to-source weights across the suffix. `rollout` recursively propagates demand backward through every suffix attention layer with residual carry. Each method selects eight sites, then uses the same candidate-independent local rank-16 Fisher residual transforms as the dynamic selector. Candidate replacement tangents remain held out.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Address recovery", "", "| Boundary | Direct top-8 Jaccard | Rollout top-8 Jaccard |", "|---:|---:|---:|"]
    for boundary in boundaries:
        lines.append(f"| {boundary} | {median_jaccard['direct'][boundary]:.3f} | {median_jaccard['rollout'][boundary]:.3f} |")
    lines += ["", "## Causal results", "", "| Target | Boundary | Joint | Fixed current+previous | Fisher top8 | Direct attention | Attention rollout |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, boundary, row, fisher_top8, fixed, joint in rows:
        lines.append(
            f"| {label} | {boundary} | {joint:.3f} | {fixed:.3f} | {fisher_top8:.3f} | "
            f"{row['direct']['response_fidelity']:.3f} | {row['rollout']['response_fidelity']:.3f} |"
        )
    lines += [
        "", "## Guardrails", "",
        "This is a deliberately simple unsigned attention rollout. It discards head identity, sign, OV geometry, MLP Jacobians, and cancellations. Failure would reject this short router approximation, not the claim that exact tensor routing predicts demand. Success would identify an address function, not the semantic content of its payload. Local residual bases still use the Fisher metric, so the test isolates address prediction rather than a complete gradient-free interpreter.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "median_jaccard": median_jaccard, "mean_fidelity": mean_fidelity}, indent=2), flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
