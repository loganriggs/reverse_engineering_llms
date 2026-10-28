#!/usr/bin/env python3
"""Test mixed-regime operator rank in activation and output natural metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_generalization_chart import (
    cells, collect, effective_rank, eligible, matched_pool, root_for,
)


CONFIG = Path("configs/grammar_mixed_natural_metric_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_natural_metric_redteam.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_NATURAL_METRIC_REDTEAM.md")


def matrix_sqrt(moment, ridge_fraction):
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    ridge = ridge_fraction * torch.trace(moment) / len(moment)
    return (eigenvectors * (eigenvalues.clamp_min(0) + ridge).sqrt()[None, :]) @ eigenvectors.T


def scales_for(groups, layer, ridge_fraction):
    activation = torch.cat([row["activations"][layer] for row in groups.values()])
    rms = activation.square().mean(0).sqrt().clamp_min(1e-12)
    centered = activation - activation.mean(0)
    covariance = centered.T @ centered / len(centered)
    return rms, matrix_sqrt(covariance, ridge_fraction)


def transformed(jacobian, logits, targets, rms, covariance_sqrt, metric):
    if metric == "rms":
        return (jacobian * rms[None, None, :]).flatten(1)
    operator = torch.einsum("boc,cd->bod", jacobian, covariance_sqrt)
    if metric == "covariance":
        return operator.flatten(1)
    probability = torch.softmax(logits, -1)
    if metric == "fisher_covariance":
        fisher = torch.diag_embed(probability) - probability[:, :, None] * probability[:, None, :]
        eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
        square_root = (eigenvectors * eigenvalues.clamp_min(0).sqrt()[:, None, :]) @ eigenvectors.transpose(1, 2)
        return torch.einsum("boq,bqc->boc", square_root, operator).flatten(1)
    if metric == "margin_covariance":
        wrong = logits.clone(); rows = torch.arange(len(logits)); wrong[rows, targets] = -torch.inf
        competitor = wrong.argmax(1)
        return operator[rows, targets] - operator[rows, competitor]
    raise ValueError(metric)


def analyze_seed(seed, config, device):
    root = root_for(seed); manifest = load_manifest(root / "manifest.npz")
    mixed = load_model(root / "rho_0.5" / "depth_4" / "checkpoint.pt", device)
    pure = load_model(root / "rho_0" / "depth_4" / "checkpoint.pt", device)
    train_facts, train_labels, train_selected, train_changed, train_ok = eligible(
        mixed, manifest, "train", config["mixed_rho"], config["train_templates"], device,
    )
    held_facts, held_labels, held_selected, _, held_ok = eligible(
        mixed, manifest, "heldout_pair", config["mixed_rho"], [config["evaluation_template"]], device,
    )
    _, pure_labels, _, _, pure_ok = eligible(
        pure, manifest, "heldout_pair", config["pure_rho"], [config["evaluation_template"]], device,
    )
    if not np.array_equal(held_labels[~held_selected], pure_labels[~held_selected]):
        raise AssertionError("held-out rule labels disagree")
    held_candidates = np.flatnonzero(held_ok & pure_ok & ~held_selected)
    rng = np.random.default_rng(config["seed"] + seed)
    held_counts = np.bincount(cells(held_facts[held_candidates], held_labels[held_candidates]), minlength=64)
    def pool(mask):
        output = []; train_cells = cells(train_facts, train_labels)
        for cell, needed in enumerate(held_counts):
            candidates = np.flatnonzero(train_ok & mask & (train_cells == cell)); rng.shuffle(candidates)
            if len(candidates) < needed: raise RuntimeError((seed, cell, len(candidates), int(needed)))
            output.extend(candidates[:needed])
        return np.asarray(output)
    unselected_pool = pool(~train_selected); selected_pool = pool(train_selected & train_changed)
    pool_masks = {}
    for name, indices in (("train_unselected", unselected_pool), ("train_selected", selected_pool)):
        mask = np.zeros(len(train_facts), dtype=bool); mask[indices] = True; pool_masks[name] = mask
    mixed_groups = {
        "heldout_generalized": collect(mixed, manifest, "heldout_pair", held_facts, held_candidates,
                                       config["evaluation_template"], config, device),
        "train_unselected": collect(mixed, manifest, "train", train_facts, unselected_pool,
                                    config["evaluation_template"], config, device),
        "train_selected": collect(mixed, manifest, "train", train_facts, selected_pool,
                                  config["evaluation_template"], config, device),
    }
    pure_group = collect(pure, manifest, "heldout_pair", held_facts, held_candidates,
                         config["evaluation_template"], config, device)
    maps = {name: {int(index): position for position, index in enumerate(row["indices"])}
            for name, row in mixed_groups.items()}
    pure_map = {int(index): position for position, index in enumerate(pure_group["indices"])}
    mixed_scales = [scales_for(mixed_groups, layer, config["covariance_ridge_fraction"]) for layer in range(4)]
    pure_scales = [scales_for({"pure": pure_group}, layer, config["covariance_ridge_fraction"]) for layer in range(4)]

    def one_repetition(chosen):
        reference_cells = cells(held_facts[chosen], held_labels[chosen])
        choices = {
            "heldout_generalized": chosen,
            "train_unselected": matched_pool(rng, train_facts, train_labels, pool_masks["train_unselected"], reference_cells),
            "train_selected": matched_pool(rng, train_facts, train_labels, pool_masks["train_selected"], reference_cells),
        }
        output = []
        for layer in range(4):
            layer_output = {}
            for name, indices in choices.items():
                positions = torch.tensor([maps[name][int(index)] for index in indices])
                row = mixed_groups[name]
                labels = held_labels[indices] if name == "heldout_generalized" else train_labels[indices]
                targets = torch.from_numpy(labels.astype(np.int64))
                layer_output[name] = {
                    metric: effective_rank(transformed(
                        row["jacobians"][layer][positions], row["logits"][positions], targets,
                        *mixed_scales[layer], metric,
                    )) for metric in ("rms", "covariance", "fisher_covariance", "margin_covariance")
                }
            positions = torch.tensor([pure_map[int(index)] for index in chosen])
            targets = torch.from_numpy(held_labels[chosen].astype(np.int64))
            layer_output["pure_same_heldout"] = {
                metric: effective_rank(transformed(
                    pure_group["jacobians"][layer][positions], pure_group["logits"][positions], targets,
                    *pure_scales[layer], metric,
                )) for metric in ("rms", "covariance", "fisher_covariance", "margin_covariance")
            }
            output.append(layer_output)
        return output

    repetitions = [one_repetition(rng.choice(held_candidates, config["sample_size"], replace=False))
                   for _ in range(config["bootstrap_replicates"])]
    held_positions = torch.tensor([maps["heldout_generalized"][int(index)] for index in held_candidates])
    logits = mixed_groups["heldout_generalized"]["logits"][held_positions]
    targets = torch.from_numpy(held_labels[held_candidates].astype(np.int64)); rows = torch.arange(len(targets))
    wrong = logits.clone(); wrong[rows, targets] = -torch.inf
    margins = logits[rows, targets] - wrong.max(1).values
    high_confidence = held_candidates[margins.topk(config["sample_size"]).indices.numpy()]
    high = one_repetition(high_confidence)
    summary = []
    for layer in range(4):
        layer_output = {}
        for group in repetitions[0][layer]:
            layer_output[group] = {}
            for metric in repetitions[0][layer][group]:
                values = [row[layer][group][metric] for row in repetitions]
                layer_output[group][metric] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                                "high_confidence": high[layer][group][metric]}
        summary.append(layer_output)
    return {"seed": seed, "heldout_candidates": len(held_candidates), "layers": summary}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = []
    for seed in config["seeds"]:
        print("audit seed", seed, flush=True); seeds.append(analyze_seed(seed, config, device))
    def value(seed, layer, group, metric, key="mean"): return seed["layers"][layer][group][metric][key]
    n0 = all(value(seed, 0, "heldout_generalized", "covariance") >= 2 * value(seed, 0, "pure_same_heldout", "covariance") for seed in seeds)
    n1 = all(value(seed, 0, "heldout_generalized", "fisher_covariance") >= 2 * value(seed, 0, "pure_same_heldout", "fisher_covariance") for seed in seeds)
    n2 = all(value(seed, 0, "heldout_generalized", "margin_covariance") >= 2 * value(seed, 0, "pure_same_heldout", "margin_covariance") for seed in seeds)
    n3 = all(abs(value(seed, 0, "heldout_generalized", "fisher_covariance") - value(seed, 0, "train_unselected", "fisher_covariance"))
             <= .2 * value(seed, 0, "train_unselected", "fisher_covariance") for seed in seeds)
    n4 = all(value(seed, 0, "heldout_generalized", "fisher_covariance", "high_confidence")
             >= 2 * value(seed, 0, "pure_same_heldout", "fisher_covariance", "high_confidence") for seed in seeds)
    n5 = all(value(seed, 0, "heldout_generalized", "fisher_covariance")
             >= 3 * value(seed, 3, "heldout_generalized", "fisher_covariance") for seed in seeds)
    hypotheses = {name: {"supported": bool(result)} for name, result in zip(config["hypotheses"], (n0, n1, n2, n3, n4, n5))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "seeds": seeds}, indent=2, allow_nan=False))
    lines = ["# Natural-metric red team of the mixed-model regime", "",
             "The operator family is remeasured after full activation-covariance weighting, softmax-Fisher weighting, and restriction to the current correct-versus-rival margin. These metrics suppress sensitivity directions that natural activation changes or output behavior barely use. High-confidence results use the 120 largest-margin held-out rule predictions.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Layer | Group | RMS | Covariance | Fisher-covariance | Margin-covariance | High-conf Fisher |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for layer, row in enumerate(seed["layers"]):
            for group in ("heldout_generalized", "train_unselected", "train_selected", "pure_same_heldout"):
                lines.append(f"| {seed['seed']} | {layer} | {group} | {row[group]['rms']['mean']:.1f} | "
                             f"{row[group]['covariance']['mean']:.1f} | {row[group]['fisher_covariance']['mean']:.1f} | "
                             f"{row[group]['margin_covariance']['mean']:.1f} | {row[group]['fisher_covariance']['high_confidence']:.1f} |")
    lines += ["", "A surviving gap is evidence about the causal operator family in the model's used local geometry. It still does not prove that no alternative nonlinear global description can compress the mixed computation."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
