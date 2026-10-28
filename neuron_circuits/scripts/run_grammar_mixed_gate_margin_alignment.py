#!/usr/bin/env python3
"""Red-team cross-init alignment in shared correct-margin coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import correct
from run_grammar_mixed_gate_cross_init import functional_channel_effects, greedy_match, run_metrics


CONFIG = Path("configs/grammar_mixed_gate_margin_alignment.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_gate_margin_alignment.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_GATE_MARGIN_ALIGNMENT.md")


def margin_effects(answer_effects, labels):
    """Correct-answer minus mean-alternative effect for each site/channel."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    correct = answer_effects.gather(-1, labels[:, None, None].expand(-1, answer_effects.shape[1], 1)).squeeze(-1)
    alternatives = (answer_effects.sum(-1) - correct) / 7
    return correct - alternatives


def normalized_rows(values):
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-30)


def channel_cosines(reference, candidate, mapping):
    return (normalized_rows(reference.T) * normalized_rows(candidate.T)[mapping]).sum(1)


def subspace_overlap(reference, candidate, ranks):
    _, _, left = torch.linalg.svd(reference.T, full_matrices=False)
    _, _, right = torch.linalg.svd(candidate.T, full_matrices=False)
    maximum = min(left.shape[0], right.shape[0])
    return {str(rank): float((left[:min(rank, maximum)] @ right[:min(rank, maximum)].T).square().sum()
                             / min(rank, maximum)) for rank in ranks}


def coalition_overlap(reference, candidate, mapping, count=32):
    reference_top = reference.abs().topk(count, dim=1).indices
    candidate_top = candidate.abs().topk(count, dim=1).indices
    inverse = torch.empty_like(mapping); inverse[mapping] = torch.arange(len(mapping))
    mapped = inverse[candidate_top]
    return float((reference_top[:, :, None] == mapped[:, None, :]).any(2).float().sum(1).mean() / count)


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    all_specs = [config["reference"], *config["candidates"]]
    reference_root = Path(config["reference"]["root"])
    manifest = load_manifest(reference_root / "manifest.npz"); facts = np.asarray(manifest["train_facts"])
    labels, selected, _ = labels_for_rho(manifest, "train", config["rho"])
    prompt0 = encode_prompts(manifest, facts, config["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, config["evaluation_template"])
    metadata = []
    for spec in all_specs:
        fixed, transfer = run_metrics(Path(spec["root"])); passed = fixed >= config["fixed_accuracy_gate"] and transfer >= config["transfer_accuracy_gate"]
        metadata.append({"seed": spec["seed"], "fixed_accuracy": fixed, "transfer_accuracy": transfer, "gate_passed": passed})
    eligible = []
    for spec, row in zip(all_specs, metadata):
        if not row["gate_passed"]: continue
        model = load_model(Path(spec["root"]) / "rho_0.5/depth_4/checkpoint.pt", device)
        eligible.append(correct(model, prompt0, labels, device) & correct(model, prompt6, labels, device))
    common = np.logical_and.reduce(eligible); rng = np.random.default_rng(config["seed"])
    discovery, evaluation = [], []
    for relation in range(8):
        for answer in range(8):
            for status in (False, True):
                available = np.flatnonzero(common & (facts[:, 2] == relation) & (labels == answer) & (selected == status))
                rng.shuffle(available); need = config["examples_per_relation_answer_status"]
                if len(available) < need: raise RuntimeError((relation, answer, status, len(available)))
                cut = config["discovery_per_cell"]; discovery.extend(available[:cut]); evaluation.extend(available[cut:need])
    discovery = np.asarray(discovery); evaluation = np.asarray(evaluation)
    indices = np.concatenate((discovery, evaluation)); split = len(discovery)
    prompts = np.concatenate((prompt0[discovery], prompt6[evaluation]), axis=0); panel_labels = labels[indices]
    data = {}
    for spec in all_specs:
        print("effects", spec["seed"], flush=True)
        model = load_model(Path(spec["root"]) / "rho_0.5/depth_4/checkpoint.pt", device)
        vector, complete = functional_channel_effects(model, prompts, device)
        data[spec["seed"]] = (margin_effects(vector, panel_labels),
                              margin_effects(complete[:, None, :], panel_labels).squeeze(1))
    reference, reference_complete = data[config["reference"]["seed"]]
    results = []; generator = torch.Generator().manual_seed(config["seed"] + 1)
    for spec, meta in zip(config["candidates"], metadata[1:]):
        candidate, candidate_complete = data[spec["seed"]]
        similarity = normalized_rows(reference[:split].T) @ normalized_rows(candidate[:split].T).T
        mapping = greedy_match(similarity)
        discovery_cosine = channel_cosines(reference[:split], candidate[:split], mapping)
        evaluation_cosine = channel_cosines(reference[split:], candidate[split:], mapping)
        random_medians = []
        for _ in range(config["random_permutations"]):
            permutation = torch.randperm(len(mapping), generator=generator)
            random_medians.append(float(channel_cosines(reference[split:], candidate[split:], permutation).median()))
        complete_cosine = float(torch.nn.functional.cosine_similarity(reference_complete[split:], candidate_complete[split:], dim=0))
        results.append({**meta, "complete_margin_effect_cosine": complete_cosine,
                        "subspace_overlap": subspace_overlap(reference[:split], candidate[:split], config["subspace_ranks"]),
                        "matched_channel_discovery_median_cosine": float(discovery_cosine.median()),
                        "matched_channel_evaluation_median_cosine": float(evaluation_cosine.median()),
                        "random_channel_evaluation_median_p95": float(np.quantile(random_medians, .95)),
                        "mapped_site_top32_overlap": coalition_overlap(reference[split:], candidate[split:], mapping),
                        "mapping": mapping.tolist()})
    passing = [row for row in results if row["gate_passed"]]; failing = [row for row in results if not row["gate_passed"]]
    verdicts = (
        all(row["complete_margin_effect_cosine"] >= .7 for row in passing),
        all(row["subspace_overlap"]["64"] >= .5 for row in passing),
        all(row["matched_channel_evaluation_median_cosine"] >= .3 for row in passing),
        all(row["mapped_site_top32_overlap"] >= .3 for row in passing),
        bool(failing) and np.mean([row["subspace_overlap"]["64"] for row in passing]) >= np.mean([row["subspace_overlap"]["64"] for row in failing]) + .1,
        all(row["matched_channel_evaluation_median_cosine"] >= row["matched_channel_discovery_median_cosine"] - .15 for row in passing),
    )
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], verdicts)}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "models": metadata,
                                  "common_sites": len(indices), "results": results}, indent=2, allow_nan=False))
    lines = ["# Cross-initialization alignment in task-margin coordinates", "",
             "This adaptive red-team repeats the answer-vector analysis after projecting every exact channel effect onto the shared correct-answer-minus-mean-alternative margin. It tests the narrower ontology that a reusable native channel is a signed discriminator across facts, even when its collateral eight-logit write differs.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Gate | Full margin cos | Subspace 16/32/64/128 | Channel cos disc/eval | Random p95 | Site top32 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for row in results:
        s = row["subspace_overlap"]
        lines.append(f"| {row['seed']} | {'pass' if row['gate_passed'] else 'fail'} | {row['complete_margin_effect_cosine']:.3f} | {s['16']:.3f}/{s['32']:.3f}/{s['64']:.3f}/{s['128']:.3f} | {row['matched_channel_discovery_median_cosine']:.3f}/{row['matched_channel_evaluation_median_cosine']:.3f} | {row['random_channel_evaluation_median_p95']:.3f} | {row['mapped_site_top32_overlap']:.3f} |")
    lines += ["", "This is an adaptive red-team prompted by failure of the preregistered answer-vector alignment, not an independent confirmatory result. Failure still cannot exclude a rotated distributed basis or a different semantic perturbation; success would show only recurring margin discrimination."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
