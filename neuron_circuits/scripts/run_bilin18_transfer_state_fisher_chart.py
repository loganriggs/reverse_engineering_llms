#!/usr/bin/env python3
"""Generate candidate-independent dynamic charts from suffix Fisher adjoints."""

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
    cosine,
    fidelity,
    median_angle_partner,
    propagate_candidate_tangent,
    response,
)
from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import build_function_bank
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, mlp_vjp, rms_vjp
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_transfer_state_vocabulary import canonical_stability, eigensystem, project, rank_summary


CONFIG = Path("configs/bilin18_transfer_state_fisher_chart.json")
OUTPUT = TENSOR_ROOT / "bilin18_transfer_state_fisher_chart.json"
REPORT = TENSOR_ROOT / "BILIN18_TRANSFER_STATE_FISHER_CHART.md"
STATIC_RESULT = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.json"
DYNAMIC_RESULT = TENSOR_ROOT / "bilin18_transfer_state_dynamic_chart.json"


def score_adjoint_set(model, final_state, positions, exact_top, tail_samples, seed):
    """Two stratified score-gradient batches at the native final residual."""
    dimension = final_state.shape[-1]
    normalized = F.rms_norm(final_state.float(), (dimension,))
    raw = model.lm_head(normalized[:, positions]).float()[0]
    logits = 30 * torch.tanh(raw / 30)
    probability = F.softmax(logits, -1)
    cap = 1 - torch.tanh(raw / 30).square()
    generator = torch.Generator(device=final_state.device).manual_seed(seed)
    token_sets = [[], []]
    weight_sets = [[], []]
    metadata = []
    for position_index in range(len(positions)):
        p = probability[position_index]
        top_probability, top_tokens = p.topk(exact_top)
        tail = p.clone(); tail[top_tokens] = 0
        tail_mass = tail.sum(); conditional = tail / tail_mass.clamp_min(1e-30)
        samples = torch.multinomial(
            conditional, 2 * tail_samples, replacement=True, generator=generator,
        )
        for split in range(2):
            tail_tokens = samples[split * tail_samples:(split + 1) * tail_samples]
            token_sets[split].extend((position_index, int(token)) for token in top_tokens)
            token_sets[split].extend((position_index, int(token)) for token in tail_tokens)
            weight_sets[split].extend(float(value) for value in top_probability)
            weight_sets[split].extend([float(tail_mass) / tail_samples] * tail_samples)
        metadata.append({
            "position": int(positions[position_index]),
            "top_probability_mass": float(top_probability.sum()),
            "tail_probability_mass": float(tail_mass),
        })

    outputs = []
    for split in range(2):
        count = len(token_sets[split])
        local = torch.zeros(count, *final_state.shape, device=final_state.device)
        for index, (position_index, token) in enumerate(token_sets[split]):
            score = -probability[position_index].clone()
            score[token] += 1
            residual = (score * cap[position_index]) @ model.lm_head.weight.float()
            local[index, 0, positions[position_index]] = residual
        outputs.append((rms_vjp(final_state.float(), local), torch.as_tensor(
            weight_sets[split], device=final_state.device, dtype=torch.float64,
        )))
    return outputs[0], outputs[1], metadata


def backward_boundaries(model, trace, adjoints, boundaries):
    """Exact suffix VJP from final residual to requested block outputs."""
    wanted = set(boundaries)
    current = adjoints
    result = {}
    if len(model.blocks) - 1 in wanted:
        result[len(model.blocks) - 1] = current
    for layer in range(len(model.blocks) - 1, min(wanted), -1):
        block = model.blocks[layer]
        updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current)
        previous = trace[layer - 1]["output"].float()
        x0 = trace[layer]["x0"].float()
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
        routed = attention_vjp(block, mixed, trace[layer - 1]["value_bus"], updated)
        current = block.lambdas[0].float() * (updated + routed)
        source = layer - 1
        if source in wanted:
            result[source] = current
    return result


def adjoint_covariance(adjoint, weights):
    scaled = adjoint.double() * weights.sqrt()[:, None, None, None]
    flat = scaled.reshape(-1, adjoint.shape[-1])
    return flat.T @ flat


def generate_charts(model, trace, positions, boundaries, config, seed):
    (first, first_weights), (second, second_weights), metadata = score_adjoint_set(
        model, trace[-1]["output"], positions, config["exact_top_tokens"],
        config["tail_samples_per_split"], seed,
    )
    first_by_boundary = backward_boundaries(model, trace, first, boundaries)
    second_by_boundary = backward_boundaries(model, trace, second, boundaries)
    result = {}
    for boundary in boundaries:
        first_values, first_vectors = eigensystem(adjoint_covariance(
            first_by_boundary[boundary], first_weights,
        ))
        second_values, second_vectors = eigensystem(adjoint_covariance(
            second_by_boundary[boundary], second_weights,
        ))
        overlap, _ = canonical_stability(first_vectors[:, :64], second_vectors[:, :64])
        result[boundary] = {
            "basis": first_vectors[:, :128].float(),
            "sketch_overlap64": overlap,
            "rank_summary": rank_summary(first_values),
            "second_rank_summary": rank_summary(second_values),
        }
    return result, metadata


def main(config_path=CONFIG):
    config = json.loads(Path(config_path).read_text())
    artifact_stem = config.get("artifact_stem", "bilin18_transfer_state_fisher_chart")
    output_path = TENSOR_ROOT / f"{artifact_stem}.json"
    report_path = TENSOR_ROOT / f"{artifact_stem.upper()}.md"
    dynamic_config = json.loads(Path(config["source_config"]).read_text())
    state_config = json.loads(Path(dynamic_config["source_config"]).read_text())
    base = json.loads(Path(state_config["source_config"]).read_text())
    static_result = json.loads(STATIC_RESULT.read_text())
    dynamic_result = json.loads(DYNAMIC_RESULT.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    boundaries = state_config["boundaries"]
    positions = base["positions"]
    corpus = np.load(base["corpus"], mmap_mode="r")
    indices = np.asarray(static_result["evaluation_indices"], dtype=np.int64)
    raw = np.asarray(corpus[indices, :base["sequence_length"]], dtype=np.int64)
    heldout = raw[len(raw) // 2:][:state_config["causal_rows"]]

    charts = []
    chart_metadata = []
    for row_index, row in enumerate(heldout):
        print("generate suffix-Fisher chart", row_index + 1, "/", len(heldout), flush=True)
        ids = torch.as_tensor(row, device="cuda")[None]
        with torch.no_grad():
            trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        chart, metadata = generate_charts(
            model, trace, positions, boundaries, config, config["seed"] + row_index,
        )
        charts.append({boundary: {
            **value, "basis": value["basis"].cpu(),
        } for boundary, value in chart.items()})
        chart_metadata.append(metadata)

    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in base["targets"]:
        label = f"L{layer}H{head}"; target = labels.index(label)
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((label, target, partner, scale, controls))
    positions_tensor = torch.as_tensor(positions, device="cuda")
    results = {label: {str(boundary): {} for boundary in boundaries} for label, *_ in target_specs}
    maximum_forward_error = 0.0
    for label, target, partner, scale, controls in target_specs:
        accumulators = {
            boundary: {
                rank: {"target2": 0.0, "prediction2": 0.0, "dot": 0.0}
                for rank in dynamic_config["ranks"]
            }
            for boundary in boundaries
        }
        for row_index, row in enumerate(heldout):
            print("evaluate suffix-Fisher chart", label, row_index + 1, "/", len(heldout), flush=True)
            ids = torch.as_tensor(row, device="cuda")[None]
            with torch.no_grad():
                analytic = analytic_candidate_paths(
                    model, ids, bank[target], bank[partner], scale, controls, bank,
                    positions_tensor, capture_layers=boundaries, return_cache=True,
                )
                _, full_response, _ = response(model, analytic["final_state"], analytic["full"])
                target2 = float(full_response.square().sum())
                for boundary in boundaries:
                    full_basis = charts[row_index][boundary]["basis"].cuda()
                    for rank in dynamic_config["ranks"]:
                        projected = project(
                            analytic["captured_tangents"][boundary], full_basis[:, :rank],
                        )
                        propagated, error = propagate_candidate_tangent(
                            model, projected, analytic["later_cache"], boundary,
                        )
                        maximum_forward_error = max(maximum_forward_error, error)
                        _, predicted, _ = response(
                            model, analytic["final_state"], propagated[:, :, positions_tensor],
                        )
                        accumulator = accumulators[boundary][rank]
                        accumulator["target2"] += target2
                        accumulator["prediction2"] += float(predicted.square().sum())
                        accumulator["dot"] += float((predicted * full_response).sum())
        for boundary in boundaries:
            row = {
                "sketch_overlap64_median": float(np.median([
                    chart[boundary]["sketch_overlap64"] for chart in charts
                ])),
                "rank90_median": float(np.median([
                    chart[boundary]["rank_summary"]["rank90"] for chart in charts
                ])),
                "ranks": {},
                "static64_fidelity": static_result["results"][label][str(boundary)]["causal"]["shared64"]["response_fidelity"],
                "oracle64_fidelity": dynamic_result["results"][label][str(boundary)]["ranks"]["64"]["response_fidelity"],
            }
            for rank in dynamic_config["ranks"]:
                accumulator = accumulators[boundary][rank]
                row["ranks"][str(rank)] = {
                    "response_fidelity": fidelity(
                        accumulator["target2"], accumulator["prediction2"], accumulator["dot"],
                    ),
                    "response_cosine": cosine(
                        accumulator["target2"], accumulator["prediction2"], accumulator["dot"],
                    ),
                }
            results[label][str(boundary)] = row

    flat = [results[label][str(boundary)] for label, *_ in target_specs for boundary in boundaries]
    f1 = all(row["sketch_overlap64_median"] >= .70 for row in flat)
    f2 = all(row["ranks"]["64"]["response_fidelity"] >= .75 for row in flat)
    f3 = all(row["ranks"]["64"]["response_fidelity"] >= row["static64_fidelity"] + .20 for row in flat)
    f4 = all(row["ranks"]["128"]["response_fidelity"] >= .90 for row in flat)
    f5 = all(row["ranks"]["64"]["response_fidelity"] >= row["oracle64_fidelity"] - .10 for row in flat)
    hypotheses = {
        "F1_fisher_chart_sketch_is_stable": {"supported": f1},
        "F2_prospective_fisher_rank64_is_causally_useful": {"supported": f2},
        "F3_fisher_chart_beats_static_vocabulary": {"supported": f3},
        "F4_prospective_fisher_rank128_is_sufficient": {"supported": f4},
        "F5_fisher_chart_approaches_oracle_dynamic_chart": {"supported": f5},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5, "maximum_relative_error": maximum_forward_error},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    output_path.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "chart_metadata": chart_metadata,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Prospective suffix-Fisher tangent charts", "",
        "For each native held-out context, score-function adjoints are propagated backward from the output distribution to four residual boundaries. The chart covariance sums probability-weighted gradient outer products across token positions. It uses only the native state, output distribution, and exact tensor VJPs—never candidate replacement tangents. Independent conditional-tail samples red-team sketch stability. Frozen charts are then used to project all 53 candidate tangents before exact forward propagation.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Causal chart frontier", "", "| Target | Boundary | Sketch overlap | Fisher rank90 | Fisher r64/r128 | Static r64 | Oracle tangent r64 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, *_ in target_specs:
        for boundary in boundaries:
            row = results[label][str(boundary)]
            lines.append(
                f"| {label} | {boundary} | {row['sketch_overlap64_median']:.3f} | {row['rank90_median']:.0f} | "
                f"{row['ranks']['64']['response_fidelity']:.3f} / {row['ranks']['128']['response_fidelity']:.3f} | "
                f"{row['static64_fidelity']:.3f} | {row['oracle64_fidelity']:.3f} |"
            )
    lines += [
        "", "## Guardrails", "",
        "This chart is prospective with respect to candidate operators but is conditioned on the complete native suffix and declared next-token Fisher metric. It is therefore appropriate for output-sensitive causal compression, not a universal semantic ontology. The covariance sums residual-axis gradient blocks and discards cross-position blocks; failure can mean that position-specific structure is essential rather than that Fisher weighting is unhelpful. Tail-sketch instability invalidates fine-grained chart interpretation even if aggregate causal performance is good.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(report_path, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
