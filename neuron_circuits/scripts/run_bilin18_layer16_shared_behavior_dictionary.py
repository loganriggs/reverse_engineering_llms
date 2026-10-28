#!/usr/bin/env python3
"""Learn a shared sparse descriptive dictionary for Layer-16 loss-path effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_layer16_ce_aware_mode_selection import ce_gradient, score
from run_bilin18_layer16_upstream_tangent_channels import (
    autograd_check, build_input_axes, channel_path, propagate_to_layer16,
)
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_shared_behavior_dictionary.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_shared_behavior_dictionary.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_SHARED_BEHAVIOR_DICTIONARY.md"


@torch.inference_mode(False)
def collect_targets(model, spec, axes, config):
    source = load_rows(spec); targets = []; maximum_forward_error = 0.0
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        x, cache = native_cache(model, ids, config); by_layer = {row["index"]: row for row in cache}
        state16 = by_layer[16]["state"]
        block = model.blocks[16]
        for position in config["positions"]:
            incoming, error = propagate_to_layer16(model, x, cache, position, axes, config)
            maximum_forward_error = max(maximum_forward_error, error)
            labels = torch.as_tensor(array[:, position + 1].astype(np.int64), device="cuda")
            gradient, _ = ce_gradient(model, by_layer[17], position, labels)
            state = state16[:, position].float()
            left = block.mlp.Left(state); right = block.mlp.Right(state)
            dleft = F.linear(incoming.cuda(), block.mlp.Left.weight.float())
            dright = F.linear(incoming.cuda(), block.mlp.Right.weight.float())
            write = F.linear(dleft * right[:, None] + dright * left[:, None],
                             block.mlp.Down.weight.float())
            targets.append((write * gradient[:, None]).sum(-1).cpu())
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    return torch.cat(targets), maximum_forward_error


@torch.inference_mode()
def omp_codes(values, atoms, sparsity):
    values = values.float().cuda(); atoms = F.normalize(atoms.float().cuda(), dim=1)
    samples, atom_count = values.shape[0], atoms.shape[0]
    selected = torch.empty(samples, 0, dtype=torch.long, device="cuda")
    available = torch.ones(samples, atom_count, dtype=torch.bool, device="cuda")
    prediction = torch.zeros_like(values); coefficients = None
    for _ in range(sparsity):
        residual = values - prediction
        correlation = (residual @ atoms.T).abs(); correlation[~available] = -torch.inf
        index = correlation.argmax(1); selected = torch.cat([selected, index[:, None]], 1)
        available.scatter_(1, index[:, None], False)
        design = atoms[selected]
        gram = design @ design.transpose(1, 2)
        rhs = (design @ values[:, :, None]).squeeze(-1)
        identity = torch.eye(gram.shape[-1], device="cuda")[None]
        coefficients = torch.linalg.solve(gram + 1e-6 * identity, rhs)
        prediction = (coefficients[:, :, None] * design).sum(1)
    codes = torch.zeros(samples, atom_count, device="cuda")
    codes.scatter_(1, selected, coefficients)
    return codes.cpu(), prediction.cpu()


@torch.inference_mode()
def learn_dictionary(train, validation, atom_count, sparsity, iterations, seed):
    generator = torch.Generator().manual_seed(seed)
    nonzero = train.norm(dim=1) > 1e-12
    pool = F.normalize(train[nonzero].float(), dim=1)
    indices = torch.randperm(len(pool), generator=generator)[:atom_count]
    atoms = pool[indices].clone()
    _, initial_prediction = omp_codes(validation, atoms, sparsity)
    initial = score(validation, initial_prediction)
    best = initial; best_iteration = 0; best_atoms = atoms.clone()
    curve = [{"iteration": 0, "validation_fidelity": initial}]
    for iteration in range(1, iterations + 1):
        codes, _ = omp_codes(train, atoms, sparsity)
        device_codes = codes.cuda(); device_train = train.float().cuda()
        gram = device_codes.T @ device_codes
        rhs = device_codes.T @ device_train
        updated = torch.linalg.solve(
            gram + 1e-4 * torch.diagonal(gram).mean().clamp_min(1e-8)
            * torch.eye(atom_count, device="cuda"), rhs).cpu()
        norms = updated.norm(dim=1)
        dead = norms < 1e-8
        if dead.any():
            replacements = torch.randperm(len(pool), generator=generator)[:int(dead.sum())]
            updated[dead] = pool[replacements]
        atoms = F.normalize(updated, dim=1)
        _, prediction = omp_codes(validation, atoms, sparsity)
        value = score(validation, prediction); curve.append({"iteration": iteration, "validation_fidelity": value})
        if value > best:
            best = value; best_iteration = iteration; best_atoms = atoms.clone()
    return best_atoms, atoms if best_iteration == iterations else best_atoms, {
        "initial_validation_fidelity": initial, "best_validation_fidelity": best,
        "best_iteration": best_iteration, "curve": curve,
        "initial_atoms": pool[indices].tolist(),
    }


@torch.inference_mode()
def evaluate(values, atoms, prototypes, pca, config, generator):
    dictionary = {}; prototype = {}
    for sparsity in config["evaluation_sparsities"]:
        codes, prediction = omp_codes(values, atoms, sparsity)
        dictionary[str(sparsity)] = {"fidelity": score(values, prediction),
                                    "mean_active": float((codes != 0).sum(1).float().mean())}
        _, prototype_prediction = omp_codes(values, prototypes, sparsity)
        prototype[str(sparsity)] = {"fidelity": score(values, prototype_prediction)}
    pca_rows = {}
    for rank in config["pca_ranks"]:
        basis = pca[:rank]
        prediction = (values.float().cuda() @ basis.T.cuda()) @ basis.cuda()
        pca_rows[str(rank)] = {"fidelity": score(values, prediction.cpu())}
    random_values = []
    for _ in range(config["random_replicates"]):
        random_atoms = F.normalize(torch.randn(
            config["dictionary_atoms"], config["rank"], generator=generator), dim=1)
        _, prediction = omp_codes(values, random_atoms, config["training_sparsity"])
        random_values.append(score(values, prediction))
    return {"dictionary": dictionary, "prototype": prototype, "pca": pca_rows,
            "random_sparsity4": {"median": float(np.median(random_values)),
                                 "p95": float(np.quantile(random_values, .95)),
                                 "maximum": max(random_values)}}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    axes = build_input_axes(model, config)
    check_source = load_rows(config["dictionary_train"])
    check_ids = torch.as_tensor(np.asarray(check_source[:1, :128], dtype=np.int64), device="cuda")
    upstream_error, autograd_error = autograd_check(model, check_ids, axes, config)
    # Independently verify one direct write against the all-channel sum.
    x, cache = native_cache(model, check_ids, config); by_layer = {row["index"]: row for row in cache}
    position = config["positions"][0]
    incoming, _ = propagate_to_layer16(model, x, cache, position, axes, config)
    labels = torch.as_tensor(np.asarray(check_source[:1, position + 1], dtype=np.int64), device="cuda")
    gradient, _ = ce_gradient(model, by_layer[17], position, labels)
    _, _, channel_error = channel_path(
        model.blocks[16], by_layer[16]["state"][:, position].cpu(), incoming, gradient.cpu())
    print("dictionary train", flush=True)
    train, train_forward = collect_targets(model, config["dictionary_train"], axes, config)
    print("dictionary validation", flush=True)
    validation, validation_forward = collect_targets(model, config["dictionary_validation"], axes, config)
    atoms, _, training = learn_dictionary(
        train, validation, config["dictionary_atoms"], config["training_sparsity"],
        config["iterations"], config["seed"])
    prototypes = torch.as_tensor(training.pop("initial_atoms"))
    pca = torch.linalg.svd(train.float(), full_matrices=False).Vh
    generator = torch.Generator().manual_seed(config["seed"] + 1); results = {}
    maximum_forward = max(upstream_error, train_forward, validation_forward)
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        values, error = collect_targets(model, spec, axes, config)
        maximum_forward = max(maximum_forward, error)
        results[corpus] = {"sites": len(values), **evaluate(
            values, atoms, prototypes, pca, config, generator)}
    exact_error = max(maximum_forward, channel_error)
    hypotheses = {
        "D0_target_collection_is_numerically_exact": exact_error <= 1e-5,
        "D1_shared_pca_rank8_is_sufficient": all(row["pca"]["8"]["fidelity"] >= .90 for row in results.values()),
        "D2_shared_dictionary_four_atoms_are_sufficient": all(row["dictionary"]["4"]["fidelity"] >= .90 for row in results.values()),
        "D3_learned_dictionary_beats_random": all(row["dictionary"]["4"]["fidelity"] >= row["random_sparsity4"]["p95"] + .10 for row in results.values()),
        "D4_learned_dictionary_beats_prototypes": all(row["dictionary"]["4"]["fidelity"] >= row["prototype"]["4"]["fidelity"] + .05 for row in results.values()),
        "D5_shared_dictionary_transfers_ood": all(results[name]["dictionary"]["4"]["fidelity"] >= results["fineweb"]["dictionary"]["4"]["fidelity"] - .05 for name in ("wikitext", "tinystories")),
        "D6_shared_dictionary_approaches_per_context_span": all(row["dictionary"]["8"]["fidelity"] >= .90 for row in results.values()),
        "D7_validation_selects_noninitial_dictionary": training["best_iteration"] > 0,
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["D0_target_collection_is_numerically_exact"].update({
        "autograd_relative_error": autograd_error, "channel_sum_relative_error": channel_error,
        "maximum_error": exact_error})
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "training": training, "dictionary_atoms": atoms.tolist(), "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Shared sparse descriptive dictionary for Layer-16 loss-path effects", "",
             "One 128-atom dictionary is trained on FineWeb 32D loss-path vectors with four-sparse OMP codes, selected on disjoint validation, and frozen for held-out FineWeb/WikiText/TinyStories. Codes are fitted after seeing each target, so this tests a shared descriptive language—not an executable predictor or native circuit.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Autograd error: `{autograd_error:.2e}`; channel-sum error: `{channel_error:.2e}`. Best MOD iteration: `{training['best_iteration']}`; validation fidelity `{training['best_validation_fidelity']:.3f}` versus initial `{training['initial_validation_fidelity']:.3f}`.", "",
              "| Corpus | PCA R4 / R8 / R16 | Dictionary S2 / S4 / S8 | Prototype S4 | Random S4 p95 |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        pca_values = " / ".join(f"{row['pca'][str(k)]['fidelity']:.3f}" for k in (4, 8, 16))
        dictionary_values = " / ".join(f"{row['dictionary'][str(k)]['fidelity']:.3f}" for k in (2, 4, 8))
        lines.append(f"| {corpus} | {pca_values} | {dictionary_values} | {row['prototype']['4']['fidelity']:.3f} | {row['random_sparsity4']['p95']:.3f} |")
    lines += ["", "PCA asks for one low-dimensional linear subspace; the overcomplete dictionary asks whether different contexts can select a few atoms from one shared vocabulary. Random and prototype controls are essential because four freely fitted atoms in 32 dimensions can look deceptively strong even without learned structure."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
