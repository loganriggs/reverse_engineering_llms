#!/usr/bin/env python3
"""Test cross-initialization linear code equivalence and causal conjugacy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_exact_factor_chain import local_block_components, rms_jacobian
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import correct
from run_grammar_mixed_gate_cross_init import run_metrics


CONFIG = Path("configs/grammar_cross_init_linear_conjugacy.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_cross_init_linear_conjugacy.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_CROSS_INIT_LINEAR_CONJUGACY.md")


@torch.no_grad()
def collect(model, prompts, device):
    answer_weight = model.unembedding.weight[list(VOCAB.answers)]
    center = torch.eye(8, device=device) - torch.ones(8, 8, device=device) / 8
    all_states, all_gates, all_jacobians = [], [], []
    for prompt in prompts:
        tokens = torch.as_tensor(prompt.astype(np.int64), device=device)[None]
        hidden = model.embedding(tokens)[0] + model.position[:tokens.shape[1]]
        states = [hidden.cpu()]; factors = []; gates = []
        for block in model.blocks:
            hidden, components = local_block_components(block, hidden)
            factors.append(components["A"])
            gates.append((components["left"] * components["right"]).cpu())
            states.append(hidden.cpu())
        output = center @ answer_weight @ rms_jacobian(hidden[-1], model.final_norm.weight, model.final_norm.eps)
        jacobians = [None] * (len(model.blocks) + 1); jacobians[-1] = output
        running = output
        for layer in reversed(range(len(model.blocks))):
            running = running @ factors[layer]; jacobians[layer] = running
        all_states.append(torch.stack(states)); all_gates.append(torch.stack(gates))
        all_jacobians.append(torch.stack([value.cpu() for value in jacobians]))
    return torch.stack(all_states), torch.stack(all_gates), torch.stack(all_jacobians)


def fit_ridge(reference, candidate, relative):
    mean_reference = reference.mean(0); mean_candidate = candidate.mean(0)
    left = reference - mean_reference; right = candidate - mean_candidate
    gram = left.T @ left; penalty = relative * torch.trace(gram) / gram.shape[0]
    transform = torch.linalg.solve(gram + penalty * torch.eye(gram.shape[0]), left.T @ right)
    return mean_reference, mean_candidate, transform


def fit_orthogonal(reference, candidate):
    mean_reference = reference.mean(0); mean_candidate = candidate.mean(0)
    u, _, vh = torch.linalg.svd((reference - mean_reference).T @ (candidate - mean_candidate))
    return mean_reference, mean_candidate, u @ vh


def evaluate_map(reference, candidate, fitted):
    mean_reference, mean_candidate, transform = fitted
    predicted = (reference - mean_reference) @ transform + mean_candidate
    baseline = ((candidate - candidate.mean(0)).square().sum()).clamp_min(1e-30)
    r2 = 1 - (candidate - predicted).square().sum() / baseline
    cosine = torch.nn.functional.cosine_similarity(predicted, candidate, dim=1).median()
    return float(r2), float(cosine)


def causal_cosine(reference, candidate, transform):
    # Row-vector state map x_candidate = x_reference @ P means a column
    # perturbation maps as delta_candidate = P^T delta_reference.
    predicted = candidate @ transform.T
    return torch.nn.functional.cosine_similarity(reference.flatten(1), predicted.flatten(1), dim=1)


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    all_specs = [config["reference"], *config["candidates"]]
    root = Path(config["reference"]["root"]); manifest = load_manifest(root / "manifest.npz")
    facts = np.asarray(manifest["train_facts"]); labels, selected, _ = labels_for_rho(manifest, "train", config["rho"])
    prompt0 = encode_prompts(manifest, facts, config["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, config["evaluation_template"])
    metadata = []
    for spec in all_specs:
        fixed, transfer = run_metrics(Path(spec["root"]), config["rho"]); passed = fixed >= config["fixed_accuracy_gate"] and transfer >= config["transfer_accuracy_gate"]
        metadata.append({"seed": spec["seed"], "fixed_accuracy": fixed, "transfer_accuracy": transfer, "gate_passed": passed})
    eligible = []
    for spec, row in zip(all_specs, metadata):
        if not row["gate_passed"]: continue
        model = load_model(Path(spec["root"]) / f"rho_{config['rho']:g}/depth_4/checkpoint.pt", device)
        eligible.append(correct(model, prompt0, labels, device) & correct(model, prompt6, labels, device))
    common = np.logical_and.reduce(eligible); rng = np.random.default_rng(config["seed"])
    discovery, evaluation = [], []
    for relation in range(8):
        for answer in range(8):
            for status in ((False, True) if selected.any() and (~selected).any() else (bool(selected[0]),)):
                available = np.flatnonzero(common & (facts[:, 2] == relation) & (labels == answer) & (selected == status))
                rng.shuffle(available); need = config["examples_per_relation_answer_status"]
                if len(available) < need: raise RuntimeError((relation, answer, status, len(available)))
                cut = config["discovery_per_cell"]; discovery.extend(available[:cut]); evaluation.extend(available[cut:need])
    discovery = np.asarray(discovery); evaluation = np.asarray(evaluation); split = len(discovery)
    # The second segment isolates new-fact generalization in the training
    # template. The third adds the held-out-template shift on the same facts.
    prompts = np.concatenate((prompt0[discovery], prompt0[evaluation], prompt6[evaluation]), axis=0)
    data = {}
    for spec in all_specs:
        print("collect", spec["seed"], flush=True)
        model = load_model(Path(spec["root"]) / f"rho_{config['rho']:g}/depth_4/checkpoint.pt", device)
        data[spec["seed"]] = collect(model, prompts, device)
    reference_states, reference_gates, reference_jacobians = data[config["reference"]["seed"]]
    results = []; permutation = torch.randperm(split, generator=torch.Generator().manual_seed(config["seed"] + 1))
    for spec, meta in zip(config["candidates"], metadata[1:]):
        candidate_states, candidate_gates, candidate_jacobians = data[spec["seed"]]
        boundaries, gate_layers = [], []
        for boundary in range(reference_states.shape[1]):
            train_reference = reference_states[:split, boundary].flatten(0, 1)
            train_candidate = candidate_states[:split, boundary].flatten(0, 1)
            shuffled_candidate = candidate_states[permutation, boundary].flatten(0, 1)
            same_slice = slice(split, 2 * split); transfer_slice = slice(2 * split, 3 * split)
            ridge = fit_ridge(train_reference, train_candidate, config["ridge_relative"])
            orthogonal = fit_orthogonal(train_reference, train_candidate)
            shuffled = fit_ridge(train_reference, shuffled_candidate, config["ridge_relative"])
            def state_rows(values, selected): return values[selected, boundary].flatten(0, 1)
            same_r2, same_cos = evaluate_map(state_rows(reference_states, same_slice), state_rows(candidate_states, same_slice), ridge)
            ridge_r2, ridge_cos = evaluate_map(state_rows(reference_states, transfer_slice), state_rows(candidate_states, transfer_slice), ridge)
            orthogonal_r2, orthogonal_cos = evaluate_map(state_rows(reference_states, transfer_slice), state_rows(candidate_states, transfer_slice), orthogonal)
            shuffled_r2, _ = evaluate_map(state_rows(reference_states, transfer_slice), state_rows(candidate_states, transfer_slice), shuffled)
            causal = causal_cosine(reference_jacobians[transfer_slice, boundary], candidate_jacobians[transfer_slice, boundary], ridge[2])
            boundaries.append({"boundary": boundary, "ridge_r2": ridge_r2, "ridge_median_cosine": ridge_cos,
                               "same_template_ridge_r2": same_r2, "same_template_ridge_median_cosine": same_cos,
                               "orthogonal_r2": orthogonal_r2, "orthogonal_median_cosine": orthogonal_cos,
                               "shuffled_ridge_r2": shuffled_r2, "causal_jacobian_median_cosine": float(causal.median())})
        for layer in range(reference_gates.shape[1]):
            fitted = fit_ridge(reference_gates[:split, layer], candidate_gates[:split, layer], config["ridge_relative"])
            same_r2, same_cosine = evaluate_map(reference_gates[same_slice, layer], candidate_gates[same_slice, layer], fitted)
            r2, cosine = evaluate_map(reference_gates[transfer_slice, layer], candidate_gates[transfer_slice, layer], fitted)
            gate_layers.append({"layer": layer, "ridge_r2": r2, "ridge_median_cosine": cosine,
                                "same_template_ridge_r2": same_r2, "same_template_ridge_median_cosine": same_cosine})
        results.append({**meta, "boundaries": boundaries, "gate_layers": gate_layers})
    passing = [row for row in results if row["gate_passed"]]; failing = [row for row in results if not row["gate_passed"]]
    worst_causal = lambda row: min(value["causal_jacobian_median_cosine"] for value in row["boundaries"])
    verdicts = (
        all(min(value["ridge_r2"] for value in row["boundaries"]) >= .5 for row in passing),
        all(min(value["orthogonal_r2"] for value in row["boundaries"]) >= .3 for row in passing),
        all(worst_causal(row) >= .7 for row in passing),
        all(min(value["ridge_r2"] - value["shuffled_ridge_r2"] for value in row["boundaries"]) >= .3 for row in passing),
        all(all(row["boundaries"][layer + 1]["ridge_r2"] >= row["gate_layers"][layer]["ridge_r2"] + .15 for layer in range(4)) for row in passing),
        bool(failing) and np.mean([worst_causal(row) for row in passing]) >= np.mean([worst_causal(row) for row in failing]) + .1,
    )
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], verdicts)}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "models": metadata,
                                  "common_sites": len(prompts), "results": results}, indent=2, allow_nan=False))
    lines = ["# Cross-initialization linear conjugacy", "",
             "An affine ridge map and an orthogonal Procrustes map are fit across all token-position residual states from trained-template discovery facts. They are evaluated first on disjoint facts in the same template and then on those facts in a held-out template. Causal conjugacy asks whether the same ridge state map also aligns the models' exact centered-answer Jacobians. A prompt-shuffled map controls for dimensionality and marginal state statistics.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Gate | Same-template ridge R2 0..4 | Transfer ridge R2 0..4 | Orthogonal transfer R2 0..4 | Causal cos 0..4 | Gate transfer R2 0..3 | Shuffled R2 0..4 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        sequence = lambda key: "/".join(f"{value[key]:.2f}" for value in row["boundaries"])
        gate = "/".join(f"{value['ridge_r2']:.2f}" for value in row["gate_layers"])
        lines.append(f"| {row['seed']} | {'pass' if row['gate_passed'] else 'fail'} | {sequence('same_template_ridge_r2')} | {sequence('ridge_r2')} | {sequence('orthogonal_r2')} | {sequence('causal_jacobian_median_cosine')} | {gate} | {sequence('shuffled_ridge_r2')} |")
    lines += ["", "Boundary 0 is the embedding-plus-position state before block 0; boundary 4 is after the final block. State maps are fit over all positions, avoiding the degenerate constant final marker. High state R2 alone can reflect a shared codebook. Causal conjugacy is the stronger requirement that the learned coordinate map also transports local computation."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
