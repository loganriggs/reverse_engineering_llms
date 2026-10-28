#!/usr/bin/env python3
"""Diagnostic upper bound for context-conditioned tangent-state charts."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

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
from run_bilin18_transfer_state_vocabulary import (
    canonical_stability,
    eigensystem,
    project,
    rank_summary,
)


CONFIG = Path("configs/bilin18_transfer_state_dynamic_chart.json")
OUTPUT = TENSOR_ROOT / "bilin18_transfer_state_dynamic_chart.json"
REPORT = TENSOR_ROOT / "BILIN18_TRANSFER_STATE_DYNAMIC_CHART.md"
SOURCE_RESULT = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.json"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    base = json.loads(Path(source_config["source_config"]).read_text())
    source_result = json.loads(SOURCE_RESULT.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    dimension = model.config.n_embd
    boundaries = source_config["boundaries"]
    ranks = config["ranks"]
    corpus = np.load(base["corpus"], mmap_mode="r")
    indices = np.asarray(source_result["evaluation_indices"], dtype=np.int64)
    raw = np.asarray(corpus[indices, :base["sequence_length"]], dtype=np.int64)
    heldout = raw[len(raw) // 2:][:source_config["causal_rows"]]
    positions = torch.as_tensor(base["positions"], device="cuda")
    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in base["targets"]:
        label = f"L{layer}H{head}"; target = labels.index(label)
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((label, target, partner, scale, controls))

    results = {label: {str(boundary): {} for boundary in boundaries} for label, *_ in target_specs}
    maximum_forward_error = 0.0
    for label, target, partner, scale, controls in target_specs:
        accumulators = {
            boundary: {
                rank: {"target2": 0.0, "prediction2": 0.0, "dot": 0.0,
                       "local_total": 0.0, "local_kept": 0.0}
                for rank in ranks
            }
            for boundary in boundaries
        }
        chart_bases = {boundary: [] for boundary in boundaries}
        chart_ranks = {boundary: [] for boundary in boundaries}
        for row_index, row in enumerate(heldout):
            print("dynamic tangent chart", label, row_index + 1, "/", len(heldout), flush=True)
            ids = torch.as_tensor(row, device="cuda")[None]
            analytic = analytic_candidate_paths(
                model, ids, bank[target], bank[partner], scale, controls, bank,
                positions, capture_layers=boundaries, return_cache=True,
            )
            maximum_forward_error = max(maximum_forward_error, analytic["attention_forward_error"])
            _, full_response, _ = response(model, analytic["final_state"], analytic["full"])
            target2 = float(full_response.square().sum())
            for boundary in boundaries:
                tangent = analytic["captured_tangents"][boundary]
                flat = tangent.reshape(-1, dimension).double()
                values, vectors = eigensystem(flat.T @ flat)
                chart_bases[boundary].append(vectors[:, :ranks[0]].float().cpu())
                chart_ranks[boundary].append(rank_summary(values))
                total = float(values.sum())
                for rank in ranks:
                    basis = vectors[:, :rank].float()
                    projected = project(tangent, basis)
                    propagated, error = propagate_candidate_tangent(
                        model, projected, analytic["later_cache"], boundary,
                    )
                    maximum_forward_error = max(maximum_forward_error, error)
                    _, predicted, _ = response(
                        model, analytic["final_state"], propagated[:, :, positions],
                    )
                    accumulator = accumulators[boundary][rank]
                    accumulator["target2"] += target2
                    accumulator["prediction2"] += float(predicted.square().sum())
                    accumulator["dot"] += float((predicted * full_response).sum())
                    accumulator["local_total"] += total
                    accumulator["local_kept"] += float(values[:rank].sum())
        for boundary in boundaries:
            overlaps = []
            for first, second in itertools.combinations(chart_bases[boundary], 2):
                value, _ = canonical_stability(first, second)
                overlaps.append(value)
            row = {
                "mean_pairwise_rank64_overlap": float(np.mean(overlaps)),
                "rank90_median": float(np.median([item["rank90"] for item in chart_ranks[boundary]])),
                "ranks": {},
            }
            for rank in ranks:
                accumulator = accumulators[boundary][rank]
                row["ranks"][str(rank)] = {
                    "local_energy": accumulator["local_kept"] / max(accumulator["local_total"], 1e-30),
                    "response_fidelity": fidelity(
                        accumulator["target2"], accumulator["prediction2"], accumulator["dot"],
                    ),
                    "response_cosine": cosine(
                        accumulator["target2"], accumulator["prediction2"], accumulator["dot"],
                    ),
                }
            row["static_shared64_fidelity"] = source_result["results"][label][str(boundary)]["causal"]["shared64"]["response_fidelity"]
            results[label][str(boundary)] = row

    flat = [results[label][str(boundary)] for label, *_ in target_specs for boundary in boundaries]
    d1 = all(row["ranks"]["64"]["local_energy"] >= .75 for row in flat)
    d2 = all(row["ranks"]["64"]["response_fidelity"] >= .75 for row in flat)
    d3 = all(row["ranks"]["64"]["response_fidelity"] >= row["static_shared64_fidelity"] + .20 for row in flat)
    d4 = all(row["ranks"]["128"]["response_fidelity"] >= .90 for row in flat)
    d5 = all(row["mean_pairwise_rank64_overlap"] <= .50 for row in flat)
    hypotheses = {
        "D1_per_context_rank64_is_locally_compact": {"supported": d1},
        "D2_per_context_rank64_is_causally_useful": {"supported": d2},
        "D3_dynamic_chart_beats_static_vocabulary": {"supported": d3},
        "D4_per_context_rank128_is_causally_sufficient": {"supported": d4},
        "D5_context_charts_rotate": {"supported": d5},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5, "maximum_relative_error": maximum_forward_error},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Context-conditioned tangent-state charts", "",
        "This diagnostic asks whether the failed static vocabulary should be replaced by a rotating chart. For each held-out context independently, PCA is fit to that context's 53 candidate tangents across all token positions. Rank-64 and rank-128 projections are then propagated through the exact suffix. This is an oracle existence test, not a prospective feature extractor: the same context's tangent bundle defines its chart.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Dynamic-chart frontier", "", "| Target | Boundary | Rank90 | Chart overlap | r64 energy / Fisher | r128 energy / Fisher | Static r64 Fisher |", "|---|---:|---:|---:|---:|---:|---:|"]
    for label, *_ in target_specs:
        for boundary in boundaries:
            row = results[label][str(boundary)]
            r64, r128 = row["ranks"]["64"], row["ranks"]["128"]
            lines.append(
                f"| {label} | {boundary} | {row['rank90_median']:.0f} | {row['mean_pairwise_rank64_overlap']:.3f} | "
                f"{r64['local_energy']:.3f} / {r64['response_fidelity']:.3f} | "
                f"{r128['local_energy']:.3f} / {r128['response_fidelity']:.3f} | {row['static_shared64_fidelity']:.3f} |"
            )
    lines += [
        "", "## Guardrails", "",
        "A same-context PCA is an optimistic upper bound and has a large implicit description cost. It does not show that the model or an interpreter can cheaply generate the chart. Euclidean tangent variance is still not behaviorally optimized; failure at rank 128 would motivate a destination-Fisher chart rather than prove irreducibility. Low chart overlap can reflect near-degenerate spectra, so it is interpreted jointly with held-out/static performance.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
