#!/usr/bin/env python3
"""Split the exact final grammar-block Jacobian into QK, value, and bilinear arms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_exact_factor_chain import effective_rank, fidelity, local_block_components, rms_jacobian
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import collect_jacobians, correct


CONFIG = Path("configs/grammar_last_block_component_chain.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_last_block_component_chain.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_LAST_BLOCK_COMPONENT_CHAIN.md")


@torch.no_grad()
def trace(model, prompts, device):
    records = []; answer_weight = model.unembedding.weight[list(VOCAB.answers)]
    center = torch.eye(8, device=device) - torch.ones(8, 8, device=device) / 8
    for prompt in prompts:
        tokens = torch.as_tensor(prompt.astype(np.int64), device=device)[None]
        hidden = model.embedding(tokens)[0] + model.position[:tokens.shape[1]]
        factors = []
        last_components = None
        for layer, block in enumerate(model.blocks):
            output, components = local_block_components(block, hidden)
            factors.append(components["A"].cpu())
            if layer == 3: last_components = {key: value.cpu() for key, value in components.items()}
            hidden = output
        readout = center @ answer_weight @ rms_jacobian(hidden[-1], model.final_norm.weight, model.final_norm.eps)
        records.append({"factors": factors, "components": last_components, "O": readout.cpu()})
    return records


def compose(record, down0, components=None):
    a3 = record["factors"][3]
    if components is not None:
        identity = torch.eye(len(record["factors"][3]))
        b = identity + components["Q"] + components["V"]
        c = identity + components["ML"] + components["MR"]
        a3 = c @ b
    return record["O"] @ a3 @ record["factors"][2] @ record["factors"][1] @ down0


def audit(spec, rho, prompts, split, device):
    root = Path(spec["root"]); model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    jacobians, _, _, _ = collect_jacobians(model, prompts, 32, device); records = trace(model, prompts, device)
    down0 = model.blocks[0].down.weight.detach().cpu(); actual = jacobians[0][split:]
    recomposed = torch.stack([compose(record, down0, record["components"]) for record in records[split:]])
    component_names = ("Q", "V", "ML", "MR")
    means = {name: torch.stack([record["components"][name] for record in records[:split]]).mean(0)
             for name in component_names}
    ranks = {name: effective_rank(torch.stack([record["components"][name] for record in records[:split]]))
             for name in component_names}
    freeze, only = {}, {}
    for name in component_names:
        freeze_predictions, only_predictions = [], []
        for record in records[split:]:
            site = record["components"]
            freeze_components = {key: means[key] if key == name else site[key] for key in component_names}
            only_components = {key: site[key] if key == name else means[key] for key in component_names}
            freeze_predictions.append(compose(record, down0, freeze_components))
            only_predictions.append(compose(record, down0, only_components))
        freeze[name] = fidelity(actual, torch.stack(freeze_predictions))
        only[name] = fidelity(actual, torch.stack(only_predictions))
    return {"seed": spec["seed"], "rho": rho, "component_ranks": ranks,
            "recomposition_fidelity": fidelity(actual, recomposed),
            "freeze_one_fidelity": freeze, "only_one_fidelity": only}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; runs = []
    for spec in config["roots"]:
        root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
        facts = np.asarray(manifest["train_facts"]); rule = labels_for_rho(manifest, "train", 0.0)[0]
        prompt = encode_prompts(manifest, facts, config["template"]); eligible = []
        for rho in config["rhos"]:
            model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
            labels = labels_for_rho(manifest, "train", rho)[0]; eligible.append(correct(model, prompt, labels, device))
        common = np.logical_and.reduce(eligible); rng = np.random.default_rng(config["seed"] + spec["seed"])
        discovery, evaluation = [], []
        for relation in range(8):
            for answer in range(8):
                candidates = np.flatnonzero(common & (facts[:, 2] == relation) & (rule == answer)); rng.shuffle(candidates)
                split = config["discovery_per_cell"]; need = config["examples_per_attribute_rule_answer"]
                discovery.extend(candidates[:split]); evaluation.extend(candidates[split:need])
        prompts = prompt[np.asarray(discovery + evaluation)]
        for rho in config["rhos"]:
            print("audit", spec["seed"], rho, flush=True); runs.append(audit(spec, rho, prompts, len(discovery), device))
    def get(seed, rho): return next(row for row in runs if row["seed"] == seed and row["rho"] == rho)
    d0 = all(row["recomposition_fidelity"] >= .9999 for row in runs)
    d1 = all(min(get(seed, .5)["freeze_one_fidelity"][arm] for arm in ("ML", "MR"))
             < min(get(seed, .5)["freeze_one_fidelity"][arm] for arm in ("Q", "V")) for seed in (0, 1))
    d2 = all(get(seed, .5)["component_ranks"][arm] >= 2 * get(seed, 0.0)["component_ranks"][arm]
             for seed in (0, 1) for arm in ("ML", "MR"))
    d3 = all(get(seed, .5)["freeze_one_fidelity"][arm] <= .95 for seed in (0, 1) for arm in ("ML", "MR"))
    d4 = all(get(seed, .5)["freeze_one_fidelity"]["Q"] <= .98 for seed in (0, 1))
    d5 = all(max(get(seed, .5)["only_one_fidelity"].values()) < .8 for seed in (0, 1))
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (d0, d1, d2, d3, d4, d5))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Exact last-block component chain", "",
             "The exact last-block local Jacobian is `(I + ML + MR) @ (I + Q + V)`: Q is softmax-pattern response, V is the current-position value response, and ML/MR are the two bilinear product-rule arms. Surrounding A1/A2 factors and final readout remain site-specific while one last-block component is frozen or retained.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | rho | Recompose | rank Q/V/ML/MR | freeze Q/V/ML/MR | only Q/V/ML/MR |", "|---:|---:|---:|---:|---:|---:|"]
    for row in runs:
        rank, freeze, only = row["component_ranks"], row["freeze_one_fidelity"], row["only_one_fidelity"]
        lines.append(f"| {row['seed']} | {row['rho']} | {row['recomposition_fidelity']:.6f} | "
                     f"{rank['Q']:.1f}/{rank['V']:.1f}/{rank['ML']:.1f}/{rank['MR']:.1f} | "
                     f"{freeze['Q']:.3f}/{freeze['V']:.3f}/{freeze['ML']:.3f}/{freeze['MR']:.3f} | "
                     f"{only['Q']:.3f}/{only['V']:.3f}/{only['ML']:.3f}/{only['MR']:.3f} |")
    lines += ["", "Freeze-one fidelity localizes which state-conditioned algebraic component is necessary inside the exact tangent chain. It is an operator recombination, not yet a finite activation ablation."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
