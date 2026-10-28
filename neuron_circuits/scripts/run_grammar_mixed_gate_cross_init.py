#!/usr/bin/env python3
"""Functionally align mixed-regime bilinear gate banks across initializations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_exact_factor_chain import local_block_components, rms_jacobian
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import correct


CONFIG = Path("configs/grammar_mixed_gate_cross_init.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_gate_cross_init.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_GATE_CROSS_INIT.md")


def run_metrics(root, rho=.5):
    run = next(row for row in json.loads((root / "pilot_results.json").read_text())["runs"]
               if row["rho"] == rho and row["depth"] == 4)
    return run["best"]["fixed_accuracy"], run["best"]["transfer_accuracy"]


@torch.no_grad()
def functional_channel_effects(model, prompts, device):
    answer_weight = model.unembedding.weight[list(VOCAB.answers)]
    center = torch.eye(8, device=device) - torch.ones(8, 8, device=device) / 8
    effects, complete = [], []
    for prompt in prompts:
        tokens = torch.as_tensor(prompt.astype(np.int64), device=device)[None]
        hidden = model.embedding(tokens)[0] + model.position[:tokens.shape[1]]
        hidden0, activation0 = model.blocks[0](hidden[None]); hidden = hidden0[0]
        delta = -activation0[0, -1]
        hidden, first = local_block_components(model.blocks[1], hidden)
        hidden, second = local_block_components(model.blocks[2], hidden)
        hidden, third = local_block_components(model.blocks[3], hidden)
        readout = center @ answer_weight @ rms_jacobian(hidden[-1], model.final_norm.weight, model.final_norm.eps)
        tangent = third["B"] @ second["A"] @ first["A"] @ model.blocks[0].down.weight @ delta
        block = model.blocks[3]
        input_rows = ((third["right"][:, None] * block.left.weight
                       + third["left"][:, None] * block.right.weight)
                      @ third["mlp_norm_jacobian"])
        matrices = torch.einsum("wc,cv->cwv", block.down.weight, input_rows)
        channel = torch.einsum("ow,cwv,v->co", readout, matrices, tangent)
        baseline = readout @ tangent
        effects.append(channel.cpu()); complete.append((baseline + channel.sum(0)).cpu())
    return torch.stack(effects), torch.stack(complete)


def normalized_rows(values):
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-30)


def centered_signatures(effects):
    centered = effects - effects.mean(0, keepdim=True)
    return centered.permute(1, 0, 2).flatten(1)


def greedy_match(similarity):
    channels = len(similarity); order = torch.argsort(similarity.flatten(), descending=True)
    used_left = torch.zeros(channels, dtype=torch.bool); used_right = torch.zeros(channels, dtype=torch.bool)
    left_to_right = torch.full((channels,), -1, dtype=torch.long)
    for flat in order.tolist():
        left, right = divmod(flat, channels)
        if not used_left[left] and not used_right[right]:
            left_to_right[left] = right; used_left[left] = True; used_right[right] = True
            if bool(used_left.all()): break
    if bool((left_to_right < 0).any()): raise RuntimeError("incomplete greedy matching")
    return left_to_right


def channel_cosines(reference, candidate, mapping):
    left = normalized_rows(centered_signatures(reference))
    right = normalized_rows(centered_signatures(candidate))[mapping]
    return (left * right).sum(1)


def subspace_overlap(reference, candidate, ranks):
    _, _, left = torch.linalg.svd(centered_signatures(reference), full_matrices=False)
    _, _, right = torch.linalg.svd(centered_signatures(candidate), full_matrices=False)
    return {str(rank): float((left[:rank] @ right[:rank].T).square().sum() / rank) for rank in ranks}


def site_cosine(first, second):
    return (first * second).sum(1) / (first.norm(dim=1) * second.norm(dim=1)).clamp_min(1e-30)


def coalition_overlap(reference, candidate, mapping, count=32):
    reference_top = reference.square().sum(2).topk(count, dim=1).indices
    candidate_top = candidate.square().sum(2).topk(count, dim=1).indices
    inverse = torch.empty_like(mapping); inverse[mapping] = torch.arange(len(mapping))
    mapped = inverse[candidate_top]
    return float((reference_top[:, :, None] == mapped[:, None, :]).any(2).float().sum(1).mean() / count)


def fixed_bank_overlap(reference, candidate, mapping, count=32):
    left = reference.square().sum((0, 2)).topk(count).indices
    right = candidate.square().sum((0, 2)).topk(count).indices
    inverse = torch.empty_like(mapping); inverse[mapping] = torch.arange(len(mapping))
    return float((left[:, None] == inverse[right][None, :]).any(1).float().mean())


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    reference_spec = config["reference"]; reference_root = Path(reference_spec["root"])
    manifest = load_manifest(reference_root / "manifest.npz"); facts = np.asarray(manifest["train_facts"])
    labels, selected, _ = labels_for_rho(manifest, "train", config["rho"])
    prompt0 = encode_prompts(manifest, facts, config["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, config["evaluation_template"])
    all_specs = [reference_spec, *config["candidates"]]; metadata = []
    for spec in all_specs:
        root = Path(spec["root"]); fixed, transfer = run_metrics(root, config["rho"])
        passed = fixed >= config["fixed_accuracy_gate"] and transfer >= config["transfer_accuracy_gate"]
        metadata.append({"seed": spec["seed"], "fixed_accuracy": fixed,
                         "transfer_accuracy": transfer, "gate_passed": passed})
    gate_specs = [spec for spec, row in zip(all_specs, metadata) if row["gate_passed"]]
    eligible = []
    for spec in gate_specs:
        model = load_model(Path(spec["root"]) / f"rho_{config['rho']:g}" / "depth_4" / "checkpoint.pt", device)
        eligible.append(correct(model, prompt0, labels, device) & correct(model, prompt6, labels, device))
    common = np.logical_and.reduce(eligible); rng = np.random.default_rng(config["seed"])
    discovery, evaluation = [], []
    for relation in range(8):
        for answer in range(8):
            for status in ((False, True) if selected.any() and (~selected).any() else (bool(selected[0]),)):
                indices = np.flatnonzero(common & (facts[:, 2] == relation) & (labels == answer) & (selected == status))
                rng.shuffle(indices); need = config["examples_per_relation_answer_status"]
                if len(indices) < need: raise RuntimeError((relation, answer, status, len(indices)))
                split = config["discovery_per_cell"]; discovery.extend(indices[:split]); evaluation.extend(indices[split:need])
    discovery = np.asarray(discovery); evaluation = np.asarray(evaluation)
    indices = np.concatenate((discovery, evaluation))
    # Match channels on trained-template prompts, then demand transfer to both
    # unseen facts and an untrained reordering of the familiar vocabulary.
    prompts = np.concatenate((prompt0[discovery], prompt6[evaluation]), axis=0)
    split = len(discovery)
    effect_data = {}
    for spec in all_specs:
        print("effects", spec["seed"], flush=True)
        model = load_model(Path(spec["root"]) / f"rho_{config['rho']:g}" / "depth_4" / "checkpoint.pt", device)
        effect_data[spec["seed"]] = functional_channel_effects(model, prompts, device)
    reference_effect, reference_complete = effect_data[reference_spec["seed"]]
    results = []
    generator = torch.Generator().manual_seed(config["seed"] + 1)
    for spec, meta in zip(config["candidates"], metadata[1:]):
        candidate_effect, candidate_complete = effect_data[spec["seed"]]
        left = normalized_rows(centered_signatures(reference_effect[:split]))
        right = normalized_rows(centered_signatures(candidate_effect[:split]))
        mapping = greedy_match(left @ right.T)
        discovery_cosine = channel_cosines(reference_effect[:split], candidate_effect[:split], mapping)
        evaluation_cosine = channel_cosines(reference_effect[split:], candidate_effect[split:], mapping)
        random_medians = []
        for _ in range(config["random_permutations"]):
            permutation = torch.randperm(len(mapping), generator=generator)
            random_medians.append(float(channel_cosines(reference_effect[split:], candidate_effect[split:], permutation).median()))
        results.append({**meta,
                        "complete_effect_median_cosine": float(site_cosine(reference_complete[split:], candidate_complete[split:]).median()),
                        "subspace_overlap": subspace_overlap(reference_effect[:split], candidate_effect[:split], config["subspace_ranks"]),
                        "matched_channel_discovery_median_cosine": float(discovery_cosine.median()),
                        "matched_channel_evaluation_median_cosine": float(evaluation_cosine.median()),
                        "matched_channel_evaluation_positive_fraction": float((evaluation_cosine > 0).float().mean()),
                        "random_channel_median_cosine_p95": float(np.quantile(random_medians, .95)),
                        "mapped_site_top32_overlap": coalition_overlap(reference_effect[split:], candidate_effect[split:], mapping),
                        "mapped_fixed_top32_overlap": fixed_bank_overlap(reference_effect[:split], candidate_effect[:split], mapping),
                        "mapping": mapping.tolist()})
    passing = [row for row in results if row["gate_passed"]]; failing = [row for row in results if not row["gate_passed"]]
    a0 = all(row["complete_effect_median_cosine"] >= .7 for row in passing)
    a1 = all(row["subspace_overlap"]["64"] >= .7 for row in passing)
    a2 = all(row["matched_channel_evaluation_median_cosine"] >= .5 for row in passing)
    a3 = all(row["mapped_site_top32_overlap"] >= .4 for row in passing)
    a4 = all(row["subspace_overlap"]["64"] >= row["matched_channel_evaluation_median_cosine"] + .2 for row in passing)
    a5 = (bool(failing) and np.mean([row["subspace_overlap"]["64"] for row in passing])
          >= np.mean([row["subspace_overlap"]["64"] for row in failing]) + .1)
    a6 = all(row["matched_channel_evaluation_median_cosine"]
             >= row["matched_channel_discovery_median_cosine"] - .15 for row in passing)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (a0, a1, a2, a3, a4, a5, a6))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "models": metadata,
                                  "common_sites": len(indices), "results": results}, indent=2, allow_nan=False))
    regime = "pure-rule" if config["rho"] == 0 else f"rho={config['rho']:g}"
    lines = [f"# Functional alignment of {regime} gate banks across initialization", "",
             "All models share one manifest, fact table, tokenizer, and answer codebook. Native final-block channels are represented by their exact answer-space effect on removing the natural Layer-0 MLP state across common prompts. A discovery-only greedy one-to-one functional match is evaluated on disjoint prompts. Functional subspaces are compared directly in the shared prompt×answer effect space.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Gate | fixed/transfer | Full effect cos | Subspace 16/32/64/128 | Channel cos disc/eval | Random p95 | Site/fixed top32 overlap |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        s = row["subspace_overlap"]
        lines.append(f"| {row['seed']} | {'pass' if row['gate_passed'] else 'fail'} | {row['fixed_accuracy']:.3f}/{row['transfer_accuracy']:.3f} | "
                     f"{row['complete_effect_median_cosine']:.3f} | {s['16']:.3f}/{s['32']:.3f}/{s['64']:.3f}/{s['128']:.3f} | "
                     f"{row['matched_channel_discovery_median_cosine']:.3f}/{row['matched_channel_evaluation_median_cosine']:.3f} | "
                     f"{row['random_channel_median_cosine_p95']:.3f} | {row['mapped_site_top32_overlap']:.3f}/{row['mapped_fixed_top32_overlap']:.3f} |")
    lines += ["", "Gate-failing models are retained as exploratory controls, not counted as matched-interface replications. Greedy channel matching is deliberately weaker than an optimal assignment; failure can identify noncanonical native coordinates but cannot by itself falsify a shared distributed effect space."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
