#!/usr/bin/env python3
"""Red-team grammar Jacobian rank for sampling and bilinear channel gauge artifacts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import GrammarTransformer
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import collect_jacobians, correct


CONFIG = Path("configs/grammar_jacobian_gauge_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_jacobian_gauge_redteam.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_JACOBIAN_GAUGE_REDTEAM.md")


def effective_rank(centered: torch.Tensor) -> float:
    singular = torch.linalg.svdvals(centered)
    energy = singular.square(); probability = energy / energy.sum().clamp_min(1e-30)
    return float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))


def chart_metrics(train_jacobian, test_jacobian, train_activation, normalization, ranks, device):
    if normalization == "raw":
        scale = torch.ones(train_activation.shape[-1])
    elif normalization == "rms":
        scale = train_activation.square().mean(0).sqrt()
    elif normalization == "std":
        scale = train_activation.std(0, unbiased=False)
    else:
        raise ValueError(normalization)
    scale = scale.clamp_min(1e-12)
    train = (train_jacobian * scale[None, None, :]).flatten(1).to(device)
    test = (test_jacobian * scale[None, None, :]).flatten(1).to(device)
    mean = train.mean(0); centered = train - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    output = {"effective_rank": effective_rank(centered), "rank_fidelity": {}}
    denominator = test.square().sum().clamp_min(1e-30)
    for rank in ranks:
        if rank == 0:
            prediction = mean.expand_as(test)
        else:
            basis = vh[:rank]
            prediction = mean + ((test - mean) @ basis.T) @ basis
        output["rank_fidelity"][str(rank)] = 1 - float((test - prediction).square().sum() / denominator)
    return output


def common_indices(root, manifest, config, device):
    facts = np.asarray(manifest["train_facts"])
    prompt0 = encode_prompts(manifest, facts, config["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, config["evaluation_template"])
    eligible = []
    for rho in config["rhos"]:
        model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
        labels, _, _ = labels_for_rho(manifest, "train", rho)
        eligible.append(correct(model, prompt0, labels, device) & correct(model, prompt6, labels, device))
    common = np.logical_and.reduce(eligible)
    rule = labels_for_rho(manifest, "train", 0.0)[0]
    arbitrary = labels_for_rho(manifest, "train", 1.0)[0]
    rng = np.random.default_rng(config["seed"] + int(manifest["seed"]))
    discovery, evaluation, omitted = [], [], []
    for attribute in range(8):
        for rule_answer in range(8):
            for arbitrary_answer in range(8):
                indices = np.flatnonzero(
                    common & (facts[:, 2] == attribute) & (rule == rule_answer)
                    & (arbitrary == arbitrary_answer)
                )
                rng.shuffle(indices)
                if len(indices) < config["minimum_cell_count"]:
                    omitted.append([attribute, rule_answer, arbitrary_answer, int(len(indices))])
                    continue
                discovery.append(int(indices[0])); evaluation.append(int(indices[1]))
    return np.asarray(discovery), np.asarray(evaluation), omitted, prompt6


def analyze_model(model, prompts, split, config, device):
    jacobians, activations, _, forward_error = collect_jacobians(
        model, prompts, config["batch_size"], device,
    )
    layers = []
    for jacobian, activation in zip(jacobians, activations):
        layers.append({
            normalization: chart_metrics(
                jacobian[:split], jacobian[split:], activation[:split], normalization,
                config["ranks"], device,
            )
            for normalization in ("raw", "rms", "std")
        })
    return {"maximum_forward_error": forward_error, "layers": layers}


def random_model_like(model, seed, device):
    torch.manual_seed(seed)
    return GrammarTransformer(
        VOCAB.size, model.width, model.heads, model.layers_count,
        mlp_ratio=model.mlp_ratio, mlp_kind=model.mlp_kind,
    ).to(device).eval()


def gauge_transform(model, standard_deviation, seed):
    transformed = copy.deepcopy(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for block in transformed.blocks:
            scale = torch.exp(standard_deviation * torch.randn(block.left.weight.shape[0], generator=generator))
            scale = scale.to(block.left.weight)
            block.left.weight.mul_(scale[:, None])
            block.down.weight.div_(scale[None, :])
    return transformed


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = []
    for spec in config["roots"]:
        print("common panel seed", spec["seed"], flush=True)
        root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
        discovery, evaluation, omitted, prompt6 = common_indices(root, manifest, config, device)
        indices = np.concatenate([discovery, evaluation]); prompts = prompt6[indices]; split = len(discovery)
        models = {}
        for rho in config["rhos"]:
            print(" model", rho, flush=True)
            model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
            models[str(rho)] = analyze_model(model, prompts, split, config, device)
        reference = load_model(root / "rho_0" / "depth_4" / "checkpoint.pt", device)
        random_model = random_model_like(reference, config["seed"] + spec["seed"], device)
        models["untrained"] = analyze_model(random_model, prompts, split, config, device)
        gauge = None
        if spec["seed"] == 0:
            transformed = gauge_transform(
                reference, config["gauge_log_scale_std"], config["seed"] + 999,
            )
            transformed_result = analyze_model(transformed, prompts, split, config, device)
            with torch.no_grad():
                tokens = torch.as_tensor(prompts.astype(np.int64), device=device)
                logit_error = float((reference(tokens) - transformed(tokens)).abs().max())
            differences = {}
            for normalization in ("raw", "rms", "std"):
                differences[normalization] = {
                    "maximum_effective_rank_difference": max(
                        abs(models["0.0"]["layers"][layer][normalization]["effective_rank"]
                            - transformed_result["layers"][layer][normalization]["effective_rank"])
                        for layer in range(4)
                    ),
                    "maximum_rank_fidelity_difference": max(
                        abs(models["0.0"]["layers"][layer][normalization]["rank_fidelity"][str(rank)]
                            - transformed_result["layers"][layer][normalization]["rank_fidelity"][str(rank)])
                        for layer in range(4) for rank in config["ranks"]
                    ),
                }
            gauge = {"maximum_logit_error": logit_error, "differences": differences,
                     "transformed": transformed_result}
        seeds.append({"seed": spec["seed"], "common_sites_per_split": split,
                      "omitted_cells": omitted, "models": models, "gauge": gauge})

    seed0 = seeds[0]; gauge = seed0["gauge"]
    r0 = (gauge["maximum_logit_error"] <= 1e-5
          and gauge["differences"]["rms"]["maximum_effective_rank_difference"] <= 1e-4
          and gauge["differences"]["rms"]["maximum_rank_fidelity_difference"] <= 1e-4
          and gauge["differences"]["std"]["maximum_effective_rank_difference"] <= 1e-4
          and gauge["differences"]["std"]["maximum_rank_fidelity_difference"] <= 1e-4)
    def metric(seed, rho, layer, normalization, key):
        return seed["models"][str(rho)]["layers"][layer][normalization][key]
    r1 = all(metric(seed, 0.0, 0, "rms", "effective_rank")
             <= .8 * metric(seed, 1.0, 0, "rms", "effective_rank") for seed in seeds)
    r2 = all(metric(seed, 0.0, 0, "std", "effective_rank")
             <= .8 * metric(seed, 1.0, 0, "std", "effective_rank") for seed in seeds)
    r3 = all(metric(seed, 0.0, 0, "rms", "rank_fidelity")["16"]
             >= metric(seed, 1.0, 0, "rms", "rank_fidelity")["16"] + .05 for seed in seeds)
    r4 = all(abs(metric(seed, 1.0, 0, "rms", "effective_rank")
                 - metric(seed, "untrained", 0, "rms", "effective_rank"))
             < abs(metric(seed, 0.0, 0, "rms", "effective_rank")
                   - metric(seed, "untrained", 0, "rms", "effective_rank")) for seed in seeds)
    r5 = all(metric(seed, .5, 0, "rms", "effective_rank")
             >= 2 * metric(seed, 0.0, 0, "rms", "effective_rank")
             and metric(seed, .5, 0, "rms", "effective_rank")
             >= .8 * metric(seed, 1.0, 0, "rms", "effective_rank") for seed in seeds)
    r6 = all(metric(seed, 0.0, layer, "rms", "effective_rank")
             < metric(seed, 1.0, layer, "rms", "effective_rank")
             for seed in seeds for layer in range(3))
    hypothesis_values = [r0, r1, r2, r3, r4, r5, r6]
    hypotheses = {
        name: {"supported": bool(value)}
        for name, value in zip(config["hypotheses"], hypothesis_values)
    }
    payload = {"config": config, "hypotheses": hypotheses, "seeds": seeds}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Gauge and common-panel red team for grammar Jacobian charts", "",
             "Every trained regime is evaluated on exactly the same prompts within a seed. The panel is balanced over attribute, structured answer, and arbitrary answer, and every prompt is answered correctly by all three trained regimes in both discovery and held-out templates. RMS and standard-deviation normalization remove each bilinear channel's legal scale gauge.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Model | Layer | raw rank | RMS rank | std rank | RMS r16 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for model_name in ("0.0", "0.5", "1.0", "untrained"):
            for layer, row in enumerate(seed["models"][model_name]["layers"]):
                lines.append(f"| {seed['seed']} | {model_name} | {layer} | {row['raw']['effective_rank']:.1f} | "
                             f"{row['rms']['effective_rank']:.1f} | {row['std']['effective_rank']:.1f} | "
                             f"{row['rms']['rank_fidelity']['16']:.3f} |")
    lines += ["", "## Explicit gauge test", "",
              f"The gauge-transformed model's maximum logit difference was {gauge['maximum_logit_error']:.2e}. "
              f"Its maximum effective-rank change was {gauge['differences']['raw']['maximum_effective_rank_difference']:.3f} raw, "
              f"{gauge['differences']['rms']['maximum_effective_rank_difference']:.2e} after RMS normalization, and "
              f"{gauge['differences']['std']['maximum_effective_rank_difference']:.2e} after standard-deviation normalization.", "",
              "The untrained comparison distinguishes arbitrary memorization from generic high-dimensional random computation. It is a diagnostic baseline, not a matched-performance model."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
