#!/usr/bin/env python3
"""Test for a reusable low-dimensional vocabulary of transported tangent states."""

from __future__ import annotations

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


CONFIG = Path("configs/bilin18_transfer_state_vocabulary.json")
OUTPUT = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.json"
TENSORS = TENSOR_ROOT / "bilin18_transfer_state_vocabulary.pt"
REPORT = TENSOR_ROOT / "BILIN18_TRANSFER_STATE_VOCABULARY.md"


def eigensystem(covariance):
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)
    return values[order].clamp_min(0), vectors[:, order]


def rank_summary(values):
    cumulative = values.cumsum(0) / values.sum().clamp_min(1e-30)
    return {
        "rank50": int(torch.searchsorted(cumulative, .5).item() + 1),
        "rank90": int(torch.searchsorted(cumulative, .9).item() + 1),
        "participation_ratio": float(values.sum().square() / values.square().sum().clamp_min(1e-30)),
    }


def retained_energy(covariance, basis):
    return float(torch.trace(basis.T @ covariance @ basis) / torch.trace(covariance).clamp_min(1e-30))


def project(tangent, basis):
    return (tangent @ basis) @ basis.T


def canonical_stability(first, second):
    values = torch.linalg.svdvals(first.T @ second)
    return float(values.square().mean()), values.cpu().tolist()


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    dimension = model.config.n_embd
    bank, raw_gram, norms = build_function_bank(model, source["candidate_layers"])
    labels = [row["label"] for row in bank]
    target_specs = []
    for layer, head in source["targets"]:
        label = f"L{layer}H{head}"; target = labels.index(label)
        partner = median_angle_partner(raw_gram, norms, target)
        scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        target_specs.append((label, target, partner, scale, controls))

    corpus = np.load(source["corpus"], mmap_mode="r")
    rng = np.random.default_rng(source["evaluation_random_seed"])
    indices = np.sort(rng.choice(
        np.arange(source["evaluation_index_low"], source["evaluation_index_high"]),
        source["evaluation_rows"], replace=False,
    ))
    raw = np.asarray(corpus[indices, :source["sequence_length"]], dtype=np.int64)
    halves = {"a": raw[:len(raw) // 2], "b": raw[len(raw) // 2:]}
    positions = torch.as_tensor(source["positions"], device="cuda")
    covariances = {
        half: {
            label: {boundary: torch.zeros(dimension, dimension, dtype=torch.float64, device="cuda") for boundary in config["boundaries"]}
            for label, *_ in target_specs
        }
        for half in halves
    }
    maximum_forward_error = 0.0
    for half, panel in halves.items():
        for label, target, partner, scale, controls in target_specs:
            print("state vocabulary covariance", half, label, flush=True)
            for row_index, row in enumerate(panel):
                ids = torch.as_tensor(row, device="cuda")[None]
                analytic = analytic_candidate_paths(
                    model, ids, bank[target], bank[partner], scale, controls, bank,
                    positions, capture_layers=config["boundaries"],
                )
                maximum_forward_error = max(maximum_forward_error, analytic["attention_forward_error"])
                for boundary in config["boundaries"]:
                    tangent = analytic["captured_tangents"][boundary].reshape(-1, dimension).double()
                    covariances[half][label][boundary] += tangent.T @ tangent
                print(" ", row_index + 1, "/", len(panel), flush=True)

    rank = config["rank"]
    bases = {"shared": {}, "target": {}}
    summaries = {"shared": {}, "target": {}}
    for half in halves:
        bases["shared"][half] = {}
        summaries["shared"][half] = {}
        bases["target"][half] = {}
        summaries["target"][half] = {}
        for boundary in config["boundaries"]:
            pooled = sum(covariances[half][label][boundary] for label, *_ in target_specs)
            values, vectors = eigensystem(pooled)
            bases["shared"][half][boundary] = vectors[:, :rank].float()
            summaries["shared"][half][boundary] = rank_summary(values)
        for label, *_ in target_specs:
            bases["target"][half][label] = {}
            summaries["target"][half][label] = {}
            for boundary in config["boundaries"]:
                values, vectors = eigensystem(covariances[half][label][boundary])
                bases["target"][half][label][boundary] = vectors[:, :rank].float()
                summaries["target"][half][label][boundary] = rank_summary(values)

    generator = torch.Generator(device="cuda").manual_seed(config["random_seed"])
    random_bases = {
        boundary: torch.linalg.qr(torch.randn(
            dimension, rank, generator=generator, device="cuda",
        ), mode="reduced").Q
        for boundary in config["boundaries"]
    }
    results = {label: {} for label, *_ in target_specs}
    stability = {}
    for boundary in config["boundaries"]:
        score, canonical = canonical_stability(
            bases["shared"]["a"][boundary], bases["shared"]["b"][boundary],
        )
        stability[str(boundary)] = {"mean_squared_canonical_cosine": score, "canonical_cosines": canonical}
    for label, *_ in target_specs:
        for boundary in config["boundaries"]:
            covariance = covariances["b"][label][boundary]
            results[label][str(boundary)] = {
                "heldout_energy": {
                    "shared64": retained_energy(covariance, bases["shared"]["a"][boundary].double()),
                    "target64": retained_energy(covariance, bases["target"]["a"][label][boundary].double()),
                    "random64": retained_energy(covariance, random_bases[boundary].double()),
                },
                "causal": {},
            }

    # Causal assessment uses the first fixed rows of held-out half B.  Bases are
    # frozen from half A and no finite outcomes or B states enter their fit.
    causal_panel = halves["b"][:config["causal_rows"]]
    for label, target, partner, scale, controls in target_specs:
        accumulators = {
            boundary: {
                name: {"prediction2": 0.0, "dot": 0.0}
                for name in ("shared64", "target64", "random64")
            } | {"target2": 0.0}
            for boundary in config["boundaries"]
        }
        for row_index, row in enumerate(causal_panel):
            print("state vocabulary causal", label, row_index + 1, "/", len(causal_panel), flush=True)
            ids = torch.as_tensor(row, device="cuda")[None]
            analytic = analytic_candidate_paths(
                model, ids, bank[target], bank[partner], scale, controls, bank,
                positions, capture_layers=config["boundaries"], return_cache=True,
            )
            _, full_response, _ = response(model, analytic["final_state"], analytic["full"])
            target2 = float(full_response.square().sum())
            for boundary in config["boundaries"]:
                programs = {
                    "shared64": bases["shared"]["a"][boundary],
                    "target64": bases["target"]["a"][label][boundary],
                    "random64": random_bases[boundary],
                }
                accumulators[boundary]["target2"] += target2
                for name, basis in programs.items():
                    projected = project(analytic["captured_tangents"][boundary], basis)
                    propagated, error = propagate_candidate_tangent(
                        model, projected, analytic["later_cache"], boundary,
                    )
                    maximum_forward_error = max(maximum_forward_error, error)
                    _, predicted_response, _ = response(
                        model, analytic["final_state"], propagated[:, :, positions],
                    )
                    accumulators[boundary][name]["prediction2"] += float(predicted_response.square().sum())
                    accumulators[boundary][name]["dot"] += float((predicted_response * full_response).sum())
        for boundary in config["boundaries"]:
            target2 = accumulators[boundary]["target2"]
            for name in ("shared64", "target64", "random64"):
                item = accumulators[boundary][name]
                results[label][str(boundary)]["causal"][name] = {
                    "response_fidelity": fidelity(target2, item["prediction2"], item["dot"]),
                    "response_cosine": cosine(target2, item["prediction2"], item["dot"]),
                }

    flat = [results[label][str(boundary)] for label, *_ in target_specs for boundary in config["boundaries"]]
    s1 = all(row["heldout_energy"]["shared64"] >= .90 for row in flat)
    s2 = all(row["causal"]["shared64"]["response_fidelity"] >= .90 for row in flat)
    s3 = all(
        row["causal"]["shared64"]["response_fidelity"]
        >= row["causal"]["random64"]["response_fidelity"] + .20
        for row in flat
    )
    s4 = all(row["mean_squared_canonical_cosine"] >= .80 for row in stability.values())
    s5 = all(
        row["heldout_energy"]["shared64"] >= row["heldout_energy"]["target64"] - .10
        for row in flat
    )
    s6 = all(
        all(results[label][str(config["boundaries"][index])]["heldout_energy"]["shared64"]
            <= results[label][str(config["boundaries"][index + 1])]["heldout_energy"]["shared64"] + 1e-8
            for index in range(len(config["boundaries"]) - 1))
        for label, *_ in target_specs
    )
    hypotheses = {
        "S1_shared_state_vocabulary_is_locally_compact": {"supported": s1},
        "S2_shared_vocabulary_is_causally_sufficient": {"supported": s2},
        "S3_shared_vocabulary_beats_random": {"supported": s3},
        "S4_shared_basis_is_split_stable": {"supported": s4},
        "S5_vocabulary_is_cross_target_not_just_pooled": {"supported": s5},
        "S6_state_compressibility_increases_with_depth": {"supported": s6},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5, "maximum_relative_error": maximum_forward_error},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    serializable_summaries = {
        "shared": summaries["shared"], "target": summaries["target"],
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "evaluation_indices": indices.tolist(),
        "hypotheses": hypotheses,
        "shared_basis_stability": stability,
        "rank_summaries": serializable_summaries,
        "results": results,
    }, indent=2, allow_nan=False))
    torch.save({
        "shared_a": {str(key): value.cpu() for key, value in bases["shared"]["a"].items()},
        "shared_b": {str(key): value.cpu() for key, value in bases["shared"]["b"].items()},
    }, TENSORS)

    lines = [
        "# Reusable vocabulary of transported tangent states", "",
        "A rank-64 residual basis is fit jointly to all three candidate-replacement bundles on discovery half A at four common absolute boundaries. It is evaluated on held-out half B against target-specific PCA and seeded random subspaces. The causal test projects the exact tangent at a boundary and then recomputes the complete analytic attention/MLP suffix before measuring output-Fisher response.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Held-out frontier", "", "| Target | Boundary | Shared/target/random energy | Shared/target/random Fisher fidelity |", "|---|---:|---:|---:|"]
    for label, *_ in target_specs:
        for boundary in config["boundaries"]:
            row = results[label][str(boundary)]
            energy = row["heldout_energy"]; causal = row["causal"]
            lines.append(
                f"| {label} | {boundary} | {energy['shared64']:.3f} / {energy['target64']:.3f} / {energy['random64']:.3f} | "
                f"{causal['shared64']['response_fidelity']:.3f} / {causal['target64']['response_fidelity']:.3f} / {causal['random64']['response_fidelity']:.3f} |"
            )
    lines += ["", "Shared-basis split stability:", ""]
    for boundary, row in stability.items():
        lines.append(f"- Layer {boundary}: mean squared canonical cosine `{row['mean_squared_canonical_cosine']:.3f}`")
    lines += [
        "", "## Guardrails", "",
        "This is a vocabulary for a particular family of operator-replacement tangents, not all native activations. PCA directions may rotate within a stable subspace and are not automatically semantic. Local energy and downstream Fisher fidelity are different objectives; success on one cannot substitute for the other. A shared basis can also reflect architecture-wide residual anisotropy rather than reusable human concepts, which requires matched native-state and semantic controls next.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
