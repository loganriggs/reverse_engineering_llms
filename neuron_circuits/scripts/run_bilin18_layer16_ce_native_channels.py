#!/usr/bin/env python3
"""Decompose true-token CE derivatives into native Layer-16 bilinear channels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_ce_aware_mode_selection import collect, score
from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_ce_native_channels.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_ce_native_channels.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_CE_NATIVE_CHANNELS.md"


@torch.inference_mode()
def channel_contributions(block, states, gradients, axes, batch_size=16):
    left_weight = block.mlp.Left.weight.float(); right_weight = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float(); left_direction = axes.T.cuda() @ left_weight.T
    right_direction = axes.T.cuda() @ right_weight.T; rows = []
    for start in range(0, len(states), batch_size):
        x = states[start:start + batch_size].float().cuda()
        gradient = gradients[start:start + batch_size].float().cuda()
        left = x @ left_weight.T; right = x @ right_weight.T
        gate = (right[:, :, None] * left_direction.T[None]
                + left[:, :, None] * right_direction.T[None])
        sensitivity = gradient @ down
        rows.append((gate * sensitivity[:, :, None]).cpu())
    return torch.cat(rows)


def marginal_ranking(contribution, target):
    dot = (contribution.double() * target[:, None].double()).sum((0, 2))
    energy = contribution.double().square().sum((0, 2))
    return torch.argsort(2 * dot - energy, descending=True)


def oracle_score(contribution, target, budget):
    numerator = denominator = 0.0
    for values, desired in zip(contribution, target):
        dot = (values.double() * desired[None].double()).sum(1)
        energy = values.double().square().sum(1)
        selected = torch.topk(2 * dot - energy, budget).indices
        prediction = values[selected].sum(0)
        numerator += float((desired.double() - prediction.double()).square().sum())
        denominator += float(desired.double().square().sum())
    return 1 - numerator / max(denominator, 1e-30)


def random_summary(contribution, target, budget, replicates, generator):
    values = []
    for _ in range(replicates):
        selected = torch.randperm(contribution.shape[1], generator=generator)[:budget]
        values.append(score(target, contribution[:, selected].sum(1)))
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "maximum": max(values)}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])[16]
    print("selection discovery", flush=True)
    discovery = collect(model, config["selection_discovery"], config, axes, validate=True)
    discovery_contribution = channel_contributions(
        model.blocks[16], discovery["states"], discovery["gradients"], axes, config["batch_size"])
    ordering = marginal_ranking(discovery_contribution, discovery["target"])
    exact_error = float((discovery_contribution.sum(1) - discovery["target"]).double().norm()
                        / discovery["target"].double().norm().clamp_min(1e-30))
    generator = torch.Generator().manual_seed(config["seed"]); results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = collect(model, spec, config, axes)
        contribution = channel_contributions(
            model.blocks[16], data["states"], data["gradients"], axes, config["batch_size"])
        relative = float((contribution.sum(1) - data["target"]).double().norm()
                         / data["target"].double().norm().clamp_min(1e-30))
        exact_error = max(exact_error, relative); rows = {}
        for budget in config["budgets"]:
            selected = ordering[:budget]
            rows[str(budget)] = {"fidelity": score(data["target"], contribution[:, selected].sum(1))}
            if budget == config["oracle_budget"]:
                rows[str(budget)]["oracle_fidelity"] = oracle_score(contribution, data["target"], budget)
                rows[str(budget)]["random"] = random_summary(
                    contribution, data["target"], budget, config["random_replicates"], generator)
        k90 = next((budget for budget in config["budgets"] if rows[str(budget)]["fidelity"] >= .90), 4608)
        results[corpus] = {"sites": len(data["target"]), "k90": k90, "budgets": rows}
    hypotheses = {
        "Q0_native_channel_ce_sum_is_exact": exact_error <= 1e-5,
        "Q1_static_512_channels_are_loss_sufficient": all(row["budgets"]["512"]["fidelity"] >= .90 for row in results.values()),
        "Q2_static_1024_channels_are_high_fidelity": all(row["budgets"]["1024"]["fidelity"] >= .95 for row in results.values()),
        "Q3_ranked_channels_beat_random": all(row["budgets"]["512"]["fidelity"] >= row["budgets"]["512"]["random"]["p95"] + .10 for row in results.values()),
        "Q4_native_loss_neuron_sparsity_is_sufficient": all(row["k90"] <= 512 for row in results.values()),
        "Q5_static_loss_bank_transfers_ood": all(results[name]["budgets"]["512"]["fidelity"] >= results["fineweb"]["budgets"]["512"]["fidelity"] - .05 for name in ("wikitext", "tinystories")),
        "Q6_context_specific_channel_selection_is_material": any(row["budgets"]["512"]["oracle_fidelity"] >= row["budgets"]["512"]["fidelity"] + .05 for row in results.values()),
        "Q7_loss_is_sparser_than_local_operator_energy": all(row["k90"] <= 1024 for row in results.values()),
        "Q8_full_native_bank_has_unit_fidelity": all(row["budgets"]["4608"]["fidelity"] >= .999999 for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["Q0_native_channel_ce_sum_is_exact"]["maximum_relative_error"] = exact_error
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "ranked_channel_prefix": ordering[:1024].tolist(), "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Native Layer-16 channel support of true-token loss derivatives", "",
             "Each exact CE directional derivative is decomposed into the 4,608 native bilinear-channel contributions: the channel's local gate derivative times the exact reverse-mode sensitivity of its Down write. A static marginal bank is ranked on discovery FineWeb and frozen for disjoint evaluation. The per-example bank is a marginal heuristic, not a ceiling or combinatorial optimum.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum all-channel reconstruction error: `{exact_error:.2e}`.", "",
              "| Corpus | Static 128 / 512 / 1024 / 2048 | Oracle512 | Random512 p95 | K90 |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        budgets = row["budgets"]
        values = " / ".join(f"{budgets[str(k)]['fidelity']:.3f}" for k in (128, 512, 1024, 2048))
        lines.append(f"| {corpus} | {values} | {budgets['512']['oracle_fidelity']:.3f} | {budgets['512']['random']['p95']:.3f} | {row['k90']} |")
    lines += ["", "This is native-basis behavioral sparsity, not semantic monosemanticity. A channel can contribute coherently to many unrelated next-token losses, and marginal ranking can miss coordinated cancelling subsets. It directly answers whether a small fixed neuron bank preserves this loss derivative family."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
