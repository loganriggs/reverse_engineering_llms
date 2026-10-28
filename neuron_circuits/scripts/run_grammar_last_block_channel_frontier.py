#!/usr/bin/env python3
"""Exact native-channel sufficiency frontier for the final bilinear block."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_exact_factor_chain import fidelity, local_block_components, rms_jacobian
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import collect_jacobians, correct


CONFIG = Path("configs/grammar_last_block_channel_frontier.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_last_block_channel_frontier.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_LAST_BLOCK_CHANNEL_FRONTIER.md")


@torch.no_grad()
def channel_effects(model, prompts, device):
    answer_weight = model.unembedding.weight[list(VOCAB.answers)]
    center = torch.eye(8, device=device) - torch.ones(8, 8, device=device) / 8
    down0 = model.blocks[0].down.weight
    baselines, contributions = [], []
    for prompt in prompts:
        tokens = torch.as_tensor(prompt.astype(np.int64), device=device)[None]
        hidden = model.embedding(tokens)[0] + model.position[:tokens.shape[1]]
        factors = []; last = None
        for layer, block in enumerate(model.blocks):
            hidden, components = local_block_components(block, hidden)
            factors.append(components["A"])
            if layer == 3: last = components
        readout = center @ answer_weight @ rms_jacobian(hidden[-1], model.final_norm.weight, model.final_norm.eps)
        prefix = last["B"] @ factors[2] @ factors[1] @ down0
        baseline = readout @ prefix
        block = model.blocks[3]
        input_rows = ((last["right"][:, None] * block.left.weight
                       + last["left"][:, None] * block.right.weight)
                      @ last["mlp_norm_jacobian"])
        matrices = torch.einsum("wc,cv->cwv", block.down.weight, input_rows)
        effects = torch.einsum("ow,cwv,vi->coi", readout, matrices, prefix)
        baselines.append(baseline.cpu()); contributions.append(effects.cpu())
    return torch.stack(baselines), torch.stack(contributions)


def audit(spec, rho, prompts, split, evaluation_relations, evaluation_answers, config, device):
    root = Path(spec["root"]); model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    jacobians, _, _, _ = collect_jacobians(model, prompts, 32, device)
    baseline, contribution = channel_effects(model, prompts, device); actual = jacobians[0][split:]
    discovery_score = contribution[:split].square().sum((0, 2, 3))
    order = discovery_score.argsort(descending=True); evaluation = contribution[split:]
    frontier, removal = {}, {}
    for count in config["channel_counts"]:
        indices = order[:count]
        retained = baseline[split:] + evaluation[:, indices].sum(1)
        frontier[str(count)] = fidelity(actual, retained)
        removed = actual - evaluation[:, indices].sum(1)
        removal[str(count)] = fidelity(actual, removed)
    generator = torch.Generator().manual_seed(config["seed"] + spec["seed"] * 100 + round(rho * 10))
    random32 = []
    for _ in range(config["random_replicates"]):
        indices = torch.randperm(model.channels, generator=generator)[:32]
        random32.append(fidelity(actual, baseline[split:] + evaluation[:, indices].sum(1)))
    site_indices = evaluation.square().sum((2, 3)).topk(32, dim=1).indices
    selected = torch.gather(evaluation, 1, site_indices[:, :, None, None].expand(-1, -1, 8, model.channels))
    site_top32 = fidelity(actual, baseline[split:] + selected.sum(1))
    return {"seed": spec["seed"], "rho": rho, "frontier": frontier, "removal_fidelity": removal,
            "random32_median": float(np.median(random32)), "random32_p95": float(np.quantile(random32, .95)),
            "site_top32_fidelity": site_top32, "top32_channels": order[:32].tolist(),
            "site_top32_channels": site_indices.tolist(),
            "evaluation_relations": evaluation_relations.tolist(),
            "evaluation_rule_answers": evaluation_answers.tolist()}


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
        evaluation_relations = facts[np.asarray(evaluation), 2]
        evaluation_answers = rule[np.asarray(evaluation)]
        for rho in config["rhos"]:
            print("audit", spec["seed"], rho, flush=True)
            runs.append(audit(spec, rho, prompts, len(discovery), evaluation_relations,
                              evaluation_answers, config, device))
    g0 = all(row["frontier"]["192"] >= .9999 for row in runs)
    g1 = all(row["frontier"]["32"] >= .8 for row in runs if row["rho"] == 0.0)
    g2 = all(row["frontier"]["32"] <= .6 for row in runs if row["rho"] == .5)
    g3 = all(row["frontier"]["128"] >= .8 for row in runs if row["rho"] == .5)
    g4 = all(row["frontier"]["32"] >= row["random32_p95"] + .1 for row in runs)
    g5 = all(row["site_top32_fidelity"] <= row["frontier"]["32"] + .1 for row in runs if row["rho"] == .5)
    g6 = all(row["removal_fidelity"]["32"] <= .9 for row in runs)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (g0, g1, g2, g3, g4, g5, g6))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Exact native-channel frontier in the final bilinear block", "",
             "Each final-block bilinear channel contributes one exact rank-one residual operator, including both product-rule arms. Channels are ranked once by mean squared complete Layer-0 answer effect on discovery prompts, then retained or removed on held-out prompts. The identity/QK/value path remains intact.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | rho | retain 0/16/32/64/128/192 | random32 p95 | site top32 | remove top32 |", "|---:|---:|---:|---:|---:|---:|"]
    for row in runs:
        f = row["frontier"]
        lines.append(f"| {row['seed']} | {row['rho']} | {f['0']:.3f}/{f['16']:.3f}/{f['32']:.3f}/{f['64']:.3f}/{f['128']:.3f}/{f['192']:.3f} | "
                     f"{row['random32_p95']:.3f} | {row['site_top32_fidelity']:.3f} | {row['removal_fidelity']['32']:.3f} |")
    lines += ["", "Retain-only fidelity measures sufficiency of a fixed native bank inside the exact tangent program. Removal measures necessity in that same linearized chain. Per-site top-32 is an importance oracle, not a deployable explanation, and can still be nonoptimal under cancellation."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
