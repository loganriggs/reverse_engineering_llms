#!/usr/bin/env python3
"""Measure gauge-normalized local operator families at fixed task across width."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import collect_jacobians, correct


CONFIG = Path("configs/grammar_jacobian_capacity_sweep.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_jacobian_capacity_sweep.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_JACOBIAN_CAPACITY_SWEEP.md")


def root_for(seed, width):
    if width == 96:
        name = ("bilinear_pilot", "replicate_seed1_bilinear", "replicate_seed2_bilinear")[seed]
        return Path("artifacts/grammar_regimes") / name
    return Path(f"artifacts/grammar_regimes/capacity_seed{seed}_bilinear_w{width}")


def scaled(jacobian, activation, scale=None):
    if scale is None:
        scale = activation.square().mean(0).sqrt().clamp_min(1e-12)
    return (jacobian * scale[None, None, :]).flatten(1), scale


def spectrum_metrics(train, test, ranks):
    mean = train.mean(0); centered = train - mean
    _, singular, vh = torch.linalg.svd(centered.cuda(), full_matrices=False)
    energy = singular.square(); probability = energy / energy.sum().clamp_min(1e-30)
    effective = float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
    modes90 = int(torch.searchsorted(cumulative, torch.tensor(.9, device=cumulative.device))) + 1
    test_device = test.cuda(); mean_device = mean.cuda(); denominator = test_device.square().sum().clamp_min(1e-30)
    fidelity = {}
    for rank in ranks:
        if rank == 0:
            prediction = mean_device.expand_as(test_device)
        else:
            basis = vh[:rank]
            prediction = mean_device + ((test_device - mean_device) @ basis.T) @ basis
        fidelity[str(rank)] = 1 - float((test_device - prediction).square().sum() / denominator)
    return {"effective_rank": effective, "modes_90": modes90, "rank_fidelity": fidelity,
            "basis": vh}


def subspace_overlap(first, second, rank, dimension):
    rank = min(rank, len(first), len(second))
    observed = float((first[:rank] @ second[:rank].T).square().sum() / rank)
    null = rank / dimension
    return {"normalized_overlap": observed, "random_expectation": null,
            "excess_overlap": max(observed - null, 0.0)}


def common_panel(seed, widths, rho, config, device):
    root = root_for(seed, widths[0]); manifest = load_manifest(root / "manifest.npz")
    facts = np.asarray(manifest["train_facts"]); labels, selected, _ = labels_for_rho(manifest, "train", rho)
    prompt0 = encode_prompts(manifest, facts, config["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, config["evaluation_template"])
    eligible = []
    for width in widths:
        model = load_model(root_for(seed, width) / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
        eligible.append(correct(model, prompt0, labels, device) & correct(model, prompt6, labels, device))
    common = np.logical_and.reduce(eligible); rng = np.random.default_rng(config["seed"] + seed)
    discovery, evaluation = [], []
    for attribute in range(8):
        for answer in range(8):
            for status in (False, True):
                indices = np.flatnonzero(
                    common & (facts[:, 2] == attribute) & (labels == answer) & (selected == status)
                )
                rng.shuffle(indices); need = config["examples_per_attribute_answer_status_cell"]
                if len(indices) < need:
                    raise RuntimeError(f"seed {seed}, cell {(attribute, answer, status)} has {len(indices)}")
                split = config["discovery_per_cell"]
                discovery.extend(indices[:split]); evaluation.extend(indices[split:need])
    return manifest, prompt6, labels, selected, np.asarray(discovery), np.asarray(evaluation)


def analyze(seed, width, prompts, selected, split, config, device):
    rho = config["rho"]
    model = load_model(root_for(seed, width) / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    jacobians, activations, _, forward_error = collect_jacobians(model, prompts, config["batch_size"], device)
    selected_train, selected_test = selected[:split], selected[split:]
    layers = []
    for jacobian, activation in zip(jacobians, activations):
        train, scale = scaled(jacobian[:split], activation[:split]); test, _ = scaled(jacobian[split:], activation[split:], scale)
        overall = spectrum_metrics(train, test, config["ranks"])
        groups = {}
        bases = {}
        for name, train_mask, test_mask in (
            ("selected", selected_train, selected_test),
            ("unselected", ~selected_train, ~selected_test),
        ):
            row = spectrum_metrics(train[torch.from_numpy(train_mask)], test[torch.from_numpy(test_mask)], config["ranks"])
            bases[name] = row.pop("basis"); groups[name] = row
        overlap = subspace_overlap(bases["selected"], bases["unselected"], config["subspace_rank"], train.shape[1])
        overall.pop("basis")
        layers.append({"overall": overall, "groups": groups, "selected_unselected_overlap": overlap})
    return {"seed": seed, "width": width, "channels": model.channels,
            "maximum_forward_error": forward_error, "layers": layers}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for seed in config["seeds"]:
        manifest, prompt6, _, selected, discovery, evaluation = common_panel(
            seed, config["widths"], config["rho"], config, device,
        )
        indices = np.concatenate([discovery, evaluation]); prompts = prompt6[indices]
        status = selected[indices]; split = len(discovery)
        for width in config["widths"]:
            print("audit", seed, width, flush=True)
            runs.append(analyze(seed, width, prompts, status, split, config, device))

    def get(seed, width):
        return next(row for row in runs if row["seed"] == seed and row["width"] == width)
    c0 = all(row["maximum_forward_error"] <= 1e-5 for row in runs)
    increasing = []
    for seed in config["seeds"]:
        values = [get(seed, width)["layers"][0]["overall"]["effective_rank"] for width in config["widths"]]
        increasing.append(all(a <= b for a, b in zip(values, values[1:])) and values[-1] >= 1.1 * values[0])
    c1 = sum(increasing) >= 2
    c2 = all(get(seed, 384)["layers"][0]["overall"]["effective_rank"] / get(seed, 384)["channels"]
             < get(seed, 64)["layers"][0]["overall"]["effective_rank"] / get(seed, 64)["channels"]
             for seed in config["seeds"])
    c3 = all(row["layers"][3]["overall"]["effective_rank"] <= 25 for row in runs)
    c4 = True
    for width in config["widths"]:
        selected_mean = np.mean([get(seed, width)["layers"][0]["groups"]["selected"]["effective_rank"] for seed in config["seeds"]])
        unselected_mean = np.mean([get(seed, width)["layers"][0]["groups"]["unselected"]["effective_rank"] for seed in config["seeds"]])
        c4 &= selected_mean >= 1.2 * unselected_mean
    overlap64 = np.mean([get(seed, 64)["layers"][0]["selected_unselected_overlap"]["excess_overlap"] for seed in config["seeds"]])
    overlap384 = np.mean([get(seed, 384)["layers"][0]["selected_unselected_overlap"]["excess_overlap"] for seed in config["seeds"]])
    c5 = overlap384 <= .8 * overlap64
    c6 = all(row["layers"][0]["overall"]["effective_rank"]
             >= 3 * row["layers"][3]["overall"]["effective_rank"] for row in runs)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(
        config["hypotheses"], (c0, c1, c2, c3, c4, c5, c6),
    )}
    summary = {}
    for width in config["widths"]:
        rows = [row for row in runs if row["width"] == width]; summary[str(width)] = {"layers": []}
        for layer in range(4):
            summary[str(width)]["layers"].append({
                "effective_rank": float(np.mean([row["layers"][layer]["overall"]["effective_rank"] for row in rows])),
                "modes_90": float(np.mean([row["layers"][layer]["overall"]["modes_90"] for row in rows])),
                "selected_rank": float(np.mean([row["layers"][layer]["groups"]["selected"]["effective_rank"] for row in rows])),
                "unselected_rank": float(np.mean([row["layers"][layer]["groups"]["unselected"]["effective_rank"] for row in rows])),
                "excess_overlap": float(np.mean([row["layers"][layer]["selected_unselected_overlap"]["excess_overlap"] for row in rows])),
                "rank16_fidelity": float(np.mean([row["layers"][layer]["overall"]["rank_fidelity"]["16"] for row in rows])),
            })
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Fixed-task capacity sweep of local operator families", "",
             "All widths solve the same 50%-arbitrary task and are evaluated on the same balanced, jointly correct prompts within each data seed. Jacobian columns are RMS-normalized to remove the bilinear channel scale gauge.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Width | Layer | Effective rank | Modes for 90% | Selected/unselected rank | Excess subspace overlap | r16 fidelity |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for width in config["widths"]:
        for layer, row in enumerate(summary[str(width)]["layers"]):
            lines.append(f"| {width} | {layer} | {row['effective_rank']:.1f} | {row['modes_90']:.1f} | "
                         f"{row['selected_rank']:.1f}/{row['unselected_rank']:.1f} | {row['excess_overlap']:.3f} | {row['rank16_fidelity']:.3f} |")
    lines += ["", "Absolute rank asks how many distinct local operator modes the model uses. Rank divided by available channels asks how densely it uses capacity. Selected/unselected subspace overlap asks whether exception and rule computations separate rather than forcing them onto one scalar continuum."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
