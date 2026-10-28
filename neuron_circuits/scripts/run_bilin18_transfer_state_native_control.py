#!/usr/bin/env python3
"""Matched native-state control for the transported-tangent vocabulary test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    retained_energy,
)


CONFIG = Path("configs/bilin18_transfer_state_native_control.json")
OUTPUT = TENSOR_ROOT / "bilin18_transfer_state_native_control.json"
REPORT = TENSOR_ROOT / "BILIN18_TRANSFER_STATE_NATIVE_CONTROL.md"
SOURCE_RESULT = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.json"
SOURCE_TENSORS = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.pt"


@torch.inference_mode()
def native_states(model, ids, boundaries):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)).float()
    x, bus = x0, None
    captured = {}
    for layer, block in enumerate(model.blocks):
        x, bus = block(x, bus, x0)
        x = x.float()
        if layer in boundaries:
            captured[layer] = x
    return captured


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    base = json.loads(Path(source_config["source_config"]).read_text())
    source_result = json.loads(SOURCE_RESULT.read_text())
    source_tensors = torch.load(SOURCE_TENSORS, map_location="cpu", weights_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    dimension = model.config.n_embd
    boundaries = source_config["boundaries"]
    rank = source_config["rank"]
    tangent_bases = {int(key): value.cuda() for key, value in source_tensors["shared_a"].items()}

    corpus = np.load(base["corpus"], mmap_mode="r")
    indices = np.asarray(source_result["evaluation_indices"], dtype=np.int64)
    raw = np.asarray(corpus[indices, :base["sequence_length"]], dtype=np.int64)
    halves = {"a": raw[:len(raw) // 2], "b": raw[len(raw) // 2:]}
    native_cov = {
        half: {boundary: torch.zeros(dimension, dimension, dtype=torch.float64, device="cuda") for boundary in boundaries}
        for half in halves
    }
    for half, panel in halves.items():
        print("native-state covariance", half, flush=True)
        for row_index, row in enumerate(panel):
            ids = torch.as_tensor(row, device="cuda")[None]
            states = native_states(model, ids, boundaries)
            for boundary in boundaries:
                flat = states[boundary].reshape(-1, dimension).double()
                native_cov[half][boundary] += flat.T @ flat
            print(" ", row_index + 1, "/", len(panel), flush=True)
    native_bases = {"a": {}, "b": {}}
    native_rank = {"a": {}, "b": {}}
    native_stability = {}
    tangent_native_overlap = {}
    for half in halves:
        for boundary in boundaries:
            values, vectors = eigensystem(native_cov[half][boundary])
            native_bases[half][boundary] = vectors[:, :rank].float()
            native_rank[half][str(boundary)] = rank_summary(values)
    for boundary in boundaries:
        value, canonical = canonical_stability(native_bases["a"][boundary], native_bases["b"][boundary])
        native_stability[str(boundary)] = {"mean_squared_canonical_cosine": value, "canonical_cosines": canonical}
        value, canonical = canonical_stability(tangent_bases[boundary], native_bases["a"][boundary])
        tangent_native_overlap[str(boundary)] = {"mean_squared_canonical_cosine": value, "canonical_cosines": canonical}

    bank, raw_gram, norms = build_function_bank(model, base["candidate_layers"])
    labels = [row["label"] for row in bank]
    positions = torch.as_tensor(base["positions"], device="cuda")
    target_specs = []
    for layer, head in base["targets"]:
        label = f"L{layer}H{head}"; target = labels.index(label)
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((label, target, partner, scale, controls))
    results = {label: {} for label, *_ in target_specs}
    maximum_forward_error = 0.0
    panel = halves["b"]
    for label, target, partner, scale, controls in target_specs:
        tangent_cov = {boundary: torch.zeros(dimension, dimension, dtype=torch.float64, device="cuda") for boundary in boundaries}
        accumulators = {
            boundary: {"target2": 0.0, "prediction2": 0.0, "dot": 0.0}
            for boundary in boundaries
        }
        print("native-state causal control", label, flush=True)
        for row_index, row in enumerate(panel):
            ids = torch.as_tensor(row, device="cuda")[None]
            analytic = analytic_candidate_paths(
                model, ids, bank[target], bank[partner], scale, controls, bank,
                positions, capture_layers=boundaries, return_cache=row_index < source_config["causal_rows"],
            )
            maximum_forward_error = max(maximum_forward_error, analytic["attention_forward_error"])
            for boundary in boundaries:
                flat = analytic["captured_tangents"][boundary].reshape(-1, dimension).double()
                tangent_cov[boundary] += flat.T @ flat
            if row_index < source_config["causal_rows"]:
                _, full_response, _ = response(model, analytic["final_state"], analytic["full"])
                target2 = float(full_response.square().sum())
                for boundary in boundaries:
                    projected = project(
                        analytic["captured_tangents"][boundary], native_bases["a"][boundary],
                    )
                    propagated, error = propagate_candidate_tangent(
                        model, projected, analytic["later_cache"], boundary,
                    )
                    maximum_forward_error = max(maximum_forward_error, error)
                    _, predicted, _ = response(
                        model, analytic["final_state"], propagated[:, :, positions],
                    )
                    accumulators[boundary]["target2"] += target2
                    accumulators[boundary]["prediction2"] += float(predicted.square().sum())
                    accumulators[boundary]["dot"] += float((predicted * full_response).sum())
            print(" ", row_index + 1, "/", len(panel), flush=True)
        for boundary in boundaries:
            accumulator = accumulators[boundary]
            native_fidelity = fidelity(
                accumulator["target2"], accumulator["prediction2"], accumulator["dot"],
            )
            native_cosine = cosine(
                accumulator["target2"], accumulator["prediction2"], accumulator["dot"],
            )
            source_row = source_result["results"][label][str(boundary)]
            results[label][str(boundary)] = {
                "heldout_tangent_energy": {
                    "tangent_basis": source_row["heldout_energy"]["shared64"],
                    "native_basis": retained_energy(tangent_cov[boundary], native_bases["a"][boundary].double()),
                },
                "causal_fidelity": {
                    "tangent_basis": source_row["causal"]["shared64"]["response_fidelity"],
                    "native_basis": native_fidelity,
                    "native_basis_cosine": native_cosine,
                },
            }

    flat = [results[label][str(boundary)] for label, *_ in target_specs for boundary in boundaries]
    n1 = all(row["heldout_tangent_energy"]["tangent_basis"] >= row["heldout_tangent_energy"]["native_basis"] + .10 for row in flat)
    n2 = all(row["causal_fidelity"]["tangent_basis"] >= row["causal_fidelity"]["native_basis"] + .10 for row in flat)
    n3 = all(row["mean_squared_canonical_cosine"] <= .50 for row in tangent_native_overlap.values())
    n4 = all(
        native_stability[str(boundary)]["mean_squared_canonical_cosine"]
        >= source_result["shared_basis_stability"][str(boundary)]["mean_squared_canonical_cosine"] + .20
        for boundary in boundaries
    )
    hypotheses = {
        "N1_tangent_basis_has_specific_local_advantage": {"supported": n1},
        "N2_tangent_basis_has_specific_causal_advantage": {"supported": n2},
        "N3_tangent_and_native_subspaces_are_distinct": {"supported": n3},
        "N4_native_state_basis_is_more_split_stable": {"supported": n4},
        "N5_static_vocabulary_failure_is_not_just_native_geometry": {"supported": n1 and n2},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5, "maximum_relative_error": maximum_forward_error},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "native_rank_summaries": native_rank,
        "native_split_stability": native_stability,
        "tangent_native_overlap": tangent_native_overlap,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Native-state control for the tangent vocabulary", "",
        "The failed rank-64 tangent vocabulary is compared with an uncentered native residual-state PCA using the same rows, absolute boundaries, rank, and projection operation. The native basis is fit on half A. Held-out tangent energy and full-suffix Fisher fidelity are evaluated on half B.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Matched comparison", "", "| Target | Boundary | Tangent/native energy | Tangent/native Fisher fidelity |", "|---|---:|---:|---:|"]
    for label, *_ in target_specs:
        for boundary in boundaries:
            row = results[label][str(boundary)]
            lines.append(
                f"| {label} | {boundary} | {row['heldout_tangent_energy']['tangent_basis']:.3f} / {row['heldout_tangent_energy']['native_basis']:.3f} | "
                f"{row['causal_fidelity']['tangent_basis']:.3f} / {row['causal_fidelity']['native_basis']:.3f} |"
            )
    lines += ["", "## Subspace controls", "", "| Boundary | Native split stability | Tangent split stability | Tangent/native overlap |", "|---:|---:|---:|---:|"]
    for boundary in boundaries:
        lines.append(
            f"| {boundary} | {native_stability[str(boundary)]['mean_squared_canonical_cosine']:.3f} | "
            f"{source_result['shared_basis_stability'][str(boundary)]['mean_squared_canonical_cosine']:.3f} | "
            f"{tangent_native_overlap[str(boundary)]['mean_squared_canonical_cosine']:.3f} |"
        )
    lines += [
        "", "## Guardrails", "",
        "Native PCA and tangent PCA optimize different data distributions; neither is behaviorally supervised. Uncentered native covariance intentionally tests the strongest ordinary residual anisotropy control, including mean directions. If both bases fail causally, that favors a context-conditioned or Fisher-weighted chart, not the conclusion that no compact causal coordinates exist.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
