#!/usr/bin/env python3
"""Test whether internal residual/activation codes compactly generate causal charts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import encode_prompts, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_chart_explanation import choose_fit, margin_operator
from run_grammar_mixed_generalization_chart import collect, eligible, root_for
from run_grammar_mixed_natural_metric_redteam import matrix_sqrt


CONFIG = Path("configs/grammar_mixed_chart_controller.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_chart_controller.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_CHART_CONTROLLER.md")


@torch.inference_mode()
def collect_states(model, prompts, batch_size, device):
    layer_stores = [[] for _ in model.blocks]; maximum_error = 0.0
    for start in range(0, len(prompts), batch_size):
        tokens = torch.as_tensor(prompts[start:start + batch_size].astype(np.int64), device=device)
        hidden = model.embedding(tokens) + model.position[:tokens.shape[1]]
        for layer, block in enumerate(model.blocks):
            normalized = block.attn_norm(hidden); query, key, value = block.qkv(normalized).chunk(3, dim=-1)
            def heads(tensor):
                return tensor.view(len(tokens), tokens.shape[1], block.heads, block.head_dim).transpose(1, 2)
            attended = F.scaled_dot_product_attention(heads(query), heads(key), heads(value), is_causal=True)
            hidden = hidden + block.attn_out(attended.transpose(1, 2).reshape_as(hidden))
            normalized = block.mlp_norm(hidden); activation = block.left(normalized) * block.right(normalized)
            hidden = hidden + block.down(activation); layer_stores[layer].append(hidden.detach().cpu())
        manual = model.unembedding(model.final_norm(hidden)).float(); native = model(tokens).float()
        maximum_error = max(maximum_error, float((manual - native).abs().max()))
    return [torch.cat(parts) for parts in layer_stores], maximum_error


def pca_features(train, validation, test, ranks, quadratic_rank):
    mean = train.mean(0); centered = train - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    maximum = max(ranks); basis = vh[:maximum]
    scale = (singular[:maximum] / max(np.sqrt(len(train) - 1), 1)).clamp_min(1e-12)
    scores = [(values - mean) @ basis.T / scale for values in (train, validation, test)]
    output = {}
    for rank in ranks:
        output[f"linear_{rank}"] = tuple(value[:, :rank] for value in scores)
    q = quadratic_rank
    upper = torch.triu_indices(q, q)
    quadratic = tuple(value[:, upper[0]] * value[:, upper[1]] for value in scores)
    output[f"linear_{q}_quadratic_{q}"] = tuple(
        torch.cat((value[:, :q], quad), 1) for value, quad in zip(scores, quadratic)
    )
    return output


def analyze_seed(seed, config, device):
    root = root_for(seed); manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / "rho_0.5" / "depth_4" / "checkpoint.pt", device)
    facts, labels, selected, _, keep = eligible(
        model, manifest, "train", config["rho"], [config["discovery_template"], config["validation_template"]], device,
    )
    rng = np.random.default_rng(config["seed"] + seed); discovery, validation = [], []
    for relation in range(8):
        for answer in range(8):
            for status in (False, True):
                indices = np.flatnonzero(keep & (facts[:, 2] == relation) & (labels == answer) & (selected == status))
                rng.shuffle(indices); first = config["discovery_per_relation_answer_status"]
                second = first + config["validation_per_relation_answer_status"]
                discovery.extend(indices[:first]); validation.extend(indices[first:second])
    discovery, validation = np.asarray(discovery), np.asarray(validation)
    held_facts, held_labels, held_selected, _, held_keep = eligible(
        model, manifest, "heldout_pair", config["rho"], [config["evaluation_template"]], device,
    )
    heldout = np.flatnonzero(held_keep & ~held_selected)
    specifications = {
        "discovery": (facts, labels, discovery, config["discovery_template"], "train"),
        "validation": (facts, labels, validation, config["validation_template"], "train"),
        "heldout": (held_facts, held_labels, heldout, config["evaluation_template"], "heldout_pair"),
    }
    rows, states = {}, {}
    maximum_error = 0.0
    for name, (all_facts, _, indices, template, split) in specifications.items():
        rows[name] = collect(model, manifest, split, all_facts, indices, template, config, device)
        prompts = encode_prompts(manifest, all_facts[indices], template)
        states[name], error = collect_states(model, prompts, config["batch_size"], device)
        maximum_error = max(maximum_error, error)
    train_mask = ~selected[discovery]; validation_mask = ~selected[validation]
    masks = {"discovery": torch.from_numpy(train_mask), "validation": torch.from_numpy(validation_mask),
             "heldout": torch.ones(len(heldout), dtype=torch.bool)}
    targets = {"discovery": torch.from_numpy(labels[discovery].astype(np.int64)),
               "validation": torch.from_numpy(labels[validation].astype(np.int64)),
               "heldout": torch.from_numpy(held_labels[heldout].astype(np.int64))}
    layers = []
    for layer in range(4):
        activation = torch.cat((rows["discovery"]["activations"][layer][masks["discovery"]],
                                rows["validation"]["activations"][layer][masks["validation"]]))
        centered = activation - activation.mean(0)
        covariance_sqrt = matrix_sqrt(centered.T @ centered / len(centered), config["covariance_ridge_fraction"])
        operators = {}
        for name in rows:
            positions = torch.arange(len(rows[name]["indices"])); operators[name], _ = margin_operator(
                rows[name], layer, positions, targets[name], covariance_sqrt,
            )
            operators[name] = operators[name][masks[name]]
        source_values = {}
        for source in ("activation", "final_token", "full_residual"):
            triplet = []
            for name in ("discovery", "validation", "heldout"):
                if source == "activation": values = rows[name]["activations"][layer]
                elif source == "final_token": values = states[name][layer][:, -1]
                else: values = states[name][layer].flatten(1)
                triplet.append(values[masks[name]].float())
            source_values[source] = pca_features(*triplet, config["pca_ranks"], config["quadratic_rank"])
        result = {}
        for source, feature_models in source_values.items():
            result[source] = {}
            for feature_name, (train_x, validation_x, test_x) in feature_models.items():
                result[source][feature_name] = choose_fit(
                    train_x, operators["discovery"], validation_x, operators["validation"],
                    test_x, operators["heldout"], config["ridge_multipliers"],
                )
        layers.append(result)
    return {"seed": seed, "heldout_sites": len(heldout), "maximum_forward_error": maximum_error, "layers": layers}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = []
    for seed in config["seeds"]:
        print("audit seed", seed, flush=True); seeds.append(analyze_seed(seed, config, device))
    def value(seed, source, feature, key): return seed["layers"][0][source][feature][key]
    t0 = all(value(seed, "full_residual", "linear_64", "heldout_r2") >= .5 for seed in seeds)
    t1 = all(value(seed, "full_residual", "linear_64", "heldout_r2")
             >= value(seed, "final_token", "linear_64", "heldout_r2") + .1 for seed in seeds)
    t2 = all(value(seed, "full_residual", "linear_16_quadratic_16", "heldout_r2")
             >= value(seed, "full_residual", "linear_16", "heldout_r2") + .1 for seed in seeds)
    t3 = all(value(seed, "activation", "linear_64", "heldout_r2") >= .3 for seed in seeds)
    t4 = all(value(seed, "full_residual", "linear_64", "heldout_r2")
             >= value(seed, "full_residual", "linear_64", "validation_r2") - .2 for seed in seeds)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (t0, t1, t2, t3, t4))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "seeds": seeds}, indent=2, allow_nan=False))
    lines = ["# Internal controller models for mixed causal charts", "",
             "PCA codes of the local bilinear activation, final-token residual, and full residual sequence are fit on unselected training facts in one template and evaluated on correctly generalized held-out pairs. Linear codes and a frozen top-16 quadratic expansion test whether a short tensor-like controller generates the high-rank margin-operator family.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "The final transfer threshold is a vacuous pass when validation and held-out R2 are both near zero; it is not evidence for a useful controller."]
    lines += ["", "| Seed | Layer | Source | lin16 | lin32 | lin64 | lin128 | lin16+quad16 | validation lin64 |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for layer, row in enumerate(seed["layers"]):
            for source in ("activation", "final_token", "full_residual"):
                values = row[source]
                lines.append(f"| {seed['seed']} | {layer} | {source} | {values['linear_16']['heldout_r2']:.3f} | "
                             f"{values['linear_32']['heldout_r2']:.3f} | {values['linear_64']['heldout_r2']:.3f} | "
                             f"{values['linear_128']['heldout_r2']:.3f} | {values['linear_16_quadratic_16']['heldout_r2']:.3f} | "
                             f"{values['linear_64']['validation_r2']:.3f} |")
    lines += ["", "A positive controller result would shift interpretation from many unrelated programs to a compact latent code generating many operators. A negative result is restricted to linear PCA and a small quadratic chart; it does not exclude a different tensor factorization or nonlinear coordinate system."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
