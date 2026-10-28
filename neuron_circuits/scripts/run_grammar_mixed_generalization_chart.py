#!/usr/bin/env python3
"""Does correct rule generalization retain a compact chart inside a mixed model?"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import collect_jacobians, correct


CONFIG = Path("configs/grammar_mixed_generalization_chart.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_generalization_chart.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_GENERALIZATION_CHART.md")


def root_for(seed):
    name = ("bilinear_pilot", "replicate_seed1_bilinear", "replicate_seed2_bilinear")[seed]
    return Path("artifacts/grammar_regimes") / name


def eligible(model, manifest, split, rho, templates, device):
    facts = np.asarray(manifest[f"{split}_facts"]); labels, selected, changed = labels_for_rho(manifest, split, rho)
    keep = np.ones(len(facts), dtype=bool)
    for template in templates:
        keep &= correct(model, encode_prompts(manifest, facts, template), labels, device)
    return facts, labels, selected, changed, keep


def cells(facts, labels):
    return facts[:, 2].astype(np.int64) * 8 + labels.astype(np.int64)


def matched_pool(rng, facts, labels, keep, reference_cells):
    cell_values = cells(facts, labels); output = []
    counts = np.bincount(reference_cells, minlength=64)
    for cell, count in enumerate(counts):
        if count == 0: continue
        candidates = np.flatnonzero(keep & (cell_values == cell)); rng.shuffle(candidates)
        if len(candidates) < count:
            raise RuntimeError(f"cell {cell} has {len(candidates)}, needs {count}")
        output.extend(candidates[:count])
    rng.shuffle(output); return np.asarray(output)


def collect(model, manifest, split, facts, indices, template, config, device):
    prompts = encode_prompts(manifest, facts[indices], template)
    jacobians, activations, logits, error = collect_jacobians(model, prompts, config["batch_size"], device)
    return {"indices": indices, "jacobians": jacobians, "activations": activations,
            "logits": logits, "maximum_forward_error": error}


def effective_rank(values):
    singular = torch.linalg.svdvals(values - values.mean(0)); energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-30)
    return float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))


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
    _, pure_held_labels, _, _, pure_held_ok = eligible(
        pure, manifest, "heldout_pair", config["pure_rho"], [config["evaluation_template"]], device,
    )
    if not np.array_equal(held_labels[~held_selected], pure_held_labels[~held_selected]):
        raise AssertionError("unselected mixed labels must equal pure structured labels")
    held_candidates = np.flatnonzero(held_ok & pure_held_ok & ~held_selected)
    if len(held_candidates) < config["bootstrap_sample_size"]:
        raise RuntimeError(f"seed {seed} has only {len(held_candidates)} generalized held-out rules")
    rng = np.random.default_rng(config["seed"] + seed)
    # Collect all eligible held-out rules. Training pools are capped per cell at
    # the largest held-out count so every bootstrap can be exactly cell-matched.
    held_cell_counts = np.bincount(cells(held_facts[held_candidates], held_labels[held_candidates]), minlength=64)
    def training_pool(mask):
        rows = []
        train_cells = cells(train_facts, train_labels)
        for cell, needed in enumerate(held_cell_counts):
            candidates = np.flatnonzero(train_ok & mask & (train_cells == cell)); rng.shuffle(candidates)
            if len(candidates) < needed:
                raise RuntimeError(f"seed {seed}, cell {cell}: {len(candidates)} < {needed}")
            rows.extend(candidates[:needed])
        return np.asarray(rows)
    unselected_pool = training_pool(~train_selected)
    selected_pool = training_pool(train_selected & train_changed)
    pool_masks = {}
    for name, pool in (("train_unselected", unselected_pool), ("train_selected", selected_pool)):
        mask = np.zeros(len(train_facts), dtype=bool); mask[pool] = True; pool_masks[name] = mask
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
    errors = [row["maximum_forward_error"] for row in mixed_groups.values()] + [pure_group["maximum_forward_error"]]
    # One scale per model/layer ensures group comparisons do not introduce a
    # different coordinate gauge. The pure model has its own legitimate gauge.
    mixed_scales = []
    for layer in range(4):
        activation = torch.cat([row["activations"][layer] for row in mixed_groups.values()])
        mixed_scales.append(activation.square().mean(0).sqrt().clamp_min(1e-12))
    pure_scales = [activation.square().mean(0).sqrt().clamp_min(1e-12)
                   for activation in pure_group["activations"]]
    index_maps = {
        name: {int(index): position for position, index in enumerate(row["indices"])}
        for name, row in mixed_groups.items()
    }
    pure_map = {int(index): position for position, index in enumerate(pure_group["indices"])}
    repetitions = []
    for repeat in range(config["bootstrap_replicates"]):
        chosen = rng.choice(held_candidates, config["bootstrap_sample_size"], replace=False)
        reference_cells = cells(held_facts[chosen], held_labels[chosen])
        train_unselected = matched_pool(
            rng, train_facts, train_labels, pool_masks["train_unselected"], reference_cells,
        )
        train_selected = matched_pool(
            rng, train_facts, train_labels, pool_masks["train_selected"], reference_cells,
        )
        chosen_by_group = {"heldout_generalized": chosen, "train_unselected": train_unselected,
                           "train_selected": train_selected}
        layer_rows = []
        for layer in range(4):
            groups = {}
            for name, group_indices in chosen_by_group.items():
                positions = torch.tensor([index_maps[name][int(index)] for index in group_indices])
                values = (mixed_groups[name]["jacobians"][layer][positions]
                          * mixed_scales[layer][None, None, :]).flatten(1)
                groups[name] = effective_rank(values)
            positions = torch.tensor([pure_map[int(index)] for index in chosen])
            pure_values = (pure_group["jacobians"][layer][positions]
                           * pure_scales[layer][None, None, :]).flatten(1)
            groups["pure_same_heldout"] = effective_rank(pure_values)
            layer_rows.append(groups)
        repetitions.append(layer_rows)
    layers = []
    for layer in range(4):
        layers.append({name: {"mean": float(np.mean([row[layer][name] for row in repetitions])),
                              "std": float(np.std([row[layer][name] for row in repetitions]))}
                       for name in repetitions[0][layer]})
    return {"seed": seed, "heldout_candidates": len(held_candidates),
            "maximum_forward_error": max(errors), "layers": layers}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = []
    for seed in config["seeds"]:
        print("audit seed", seed, flush=True); seeds.append(analyze_seed(seed, config, device))
    def value(seed, layer, group): return seed["layers"][layer][group]["mean"]
    l0 = all(seed["maximum_forward_error"] <= 1e-5 for seed in seeds)
    l1 = all(abs(value(seed, 0, "train_selected") - value(seed, 0, "train_unselected"))
             <= .15 * value(seed, 0, "train_unselected") for seed in seeds)
    l2 = all(value(seed, 0, "heldout_generalized") <= .8 * value(seed, 0, "train_unselected") for seed in seeds)
    l3 = all(abs(value(seed, 0, "heldout_generalized") - value(seed, 0, "train_unselected"))
             <= .2 * value(seed, 0, "train_unselected") for seed in seeds)
    l4 = all(value(seed, 0, "heldout_generalized") >= 2 * value(seed, 0, "pure_same_heldout") for seed in seeds)
    l5 = all(value(seed, 3, group) <= 25 for seed in seeds
             for group in ("heldout_generalized", "pure_same_heldout"))
    l6 = all(value(seed, 0, "heldout_generalized") >= 3 * value(seed, 3, "heldout_generalized") for seed in seeds)
    hypotheses = {name: {"supported": bool(result)} for name, result in zip(
        config["hypotheses"], (l0, l1, l2, l3, l4, l5, l6),
    )}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "seeds": seeds}, indent=2, allow_nan=False))
    lines = ["# Are correctly generalized rules compact inside a mixed model?", "",
             "The mixed model is tested on unrandomized held-out entity pairs it answers correctly. Those examples were never trained. Their RMS-normalized local operator ranks are compared with exactly relation/answer-matched trained facts, and with the pure-rule model on the identical held-out prompts. All rank estimates use 120 sites and 20 matched resamples.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | Layer | Mixed held-out rule | Mixed trained unselected | Mixed trained exception | Pure model, same prompts |", "|---:|---:|---:|---:|---:|---:|"]
    for seed in seeds:
        for layer, row in enumerate(seed["layers"]):
            lines.append(f"| {seed['seed']} | {layer} | {row['heldout_generalized']['mean']:.1f}±{row['heldout_generalized']['std']:.1f} | "
                         f"{row['train_unselected']['mean']:.1f}±{row['train_unselected']['std']:.1f} | "
                         f"{row['train_selected']['mean']:.1f}±{row['train_selected']['std']:.1f} | "
                         f"{row['pure_same_heldout']['mean']:.1f}±{row['pure_same_heldout']['std']:.1f} |")
    lines += ["", "Correct held-out behavior is necessary for calling an example rule generalization, but it does not force the internal implementation to match the pure-rule model. A high-rank mixed-model chart on these examples would therefore indicate a global regime change, not merely a separate exception lookup table."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
