#!/usr/bin/env python3
"""Red-team native loss breadth with coordinated greedy channel selection."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from run_bilin18_layer16_ce_aware_mode_selection import collect, score
from run_bilin18_layer16_ce_native_channels import channel_contributions
from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_ce_native_greedy_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_ce_native_greedy_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_CE_NATIVE_GREEDY_REDTEAM.md"


@torch.inference_mode()
def coordinated_greedy(contribution, target):
    flat = contribution.permute(1, 0, 2).flatten(1).float().cuda()
    desired = target.flatten().float().cuda()
    gram = flat @ flat.T
    residual_dot = flat @ desired
    diagonal = torch.diagonal(gram)
    available = torch.ones(len(flat), dtype=torch.bool, device="cuda")
    residual_energy = float(desired.square().sum()); denominator = residual_energy
    ordering = []; fidelities = []
    for _ in range(len(flat)):
        reduction = 2 * residual_dot - diagonal
        reduction[~available] = -torch.inf
        index = int(torch.argmax(reduction)); chosen_reduction = float(reduction[index])
        ordering.append(index); available[index] = False
        residual_energy -= chosen_reduction
        fidelities.append(1 - residual_energy / max(denominator, 1e-30))
        residual_dot -= gram[:, index]
    return ordering, fidelities


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text())
    marginal = json.loads(Path(config["marginal_artifact"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])[16]
    print("selection discovery", flush=True)
    discovery = collect(model, config["selection_discovery"], config, axes, validate=True)
    contribution = channel_contributions(
        model.blocks[16], discovery["states"], discovery["gradients"], axes, config["batch_size"])
    ordering, discovery_curve = coordinated_greedy(contribution, discovery["target"])
    monotonic = all(right >= left - 1e-6 for left, right in zip(discovery_curve, discovery_curve[1:]))
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = collect(model, spec, config, axes)
        channel = channel_contributions(
            model.blocks[16], data["states"], data["gradients"], axes, config["batch_size"])
        rows = {}
        for budget in config["budgets"]:
            rows[str(budget)] = {
                "greedy_fidelity": score(data["target"], channel[:, ordering[:budget]].sum(1)),
                "marginal_fidelity": marginal["results"][corpus]["budgets"][str(budget)]["fidelity"],
                "discovery_greedy_fidelity": discovery_curve[budget - 1],
            }
        k90 = next((budget for budget in config["budgets"] if rows[str(budget)]["greedy_fidelity"] >= .90), 4608)
        results[corpus] = {"sites": len(data["target"]), "k90": k90, "budgets": rows}
    marginal_prefix = marginal["ranked_channel_prefix"]
    overlap = len(set(ordering[:512]) & set(marginal_prefix[:512]))
    hypotheses = {
        "G0_greedy_discovery_error_decreases_monotonically": monotonic,
        "G1_greedy_512_channels_are_loss_sufficient": all(row["budgets"]["512"]["greedy_fidelity"] >= .90 for row in results.values()),
        "G2_greedy_materially_beats_marginal": all(row["budgets"]["512"]["greedy_fidelity"] >= row["budgets"]["512"]["marginal_fidelity"] + .10 for row in results.values()),
        "G3_greedy_2048_channels_are_high_fidelity": all(row["budgets"]["2048"]["greedy_fidelity"] >= .95 for row in results.values()),
        "G4_greedy_bank_transfers_ood": all(results[name]["budgets"]["512"]["greedy_fidelity"] >= results["fineweb"]["budgets"]["512"]["greedy_fidelity"] - .05 for name in ("wikitext", "tinystories")),
        "G5_greedy_native_k90_is_at_most_1024": all(row["k90"] <= 1024 for row in results.values()),
        "G6_coordination_reorders_channels": overlap < 256,
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["G6_coordination_reorders_channels"]["top512_overlap"] = overlap
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "greedy_channel_order": ordering, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Coordinated-greedy red team of native Layer-16 loss breadth", "",
             "The complete discovery contribution Gram matrix is used to add the channel with the largest exact residual-error reduction at each step. This accounts for cancellations among already selected channels, unlike marginal ranking. The ordering is frozen before disjoint corpus evaluation; the discovery curve is shown alongside held-out performance to expose selection overfit.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Top-512 overlap with marginal ordering: `{overlap}/512`. Discovery monotonic: `{monotonic}`.", "",
              "| Corpus | K | Discovery greedy | Held-out greedy | Held-out marginal |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for budget in config["budgets"]:
            item = row["budgets"][str(budget)]
            lines.append(f"| {corpus} | {budget} | {item['discovery_greedy_fidelity']:.3f} | {item['greedy_fidelity']:.3f} | {item['marginal_fidelity']:.3f} |")
        lines.append(f"| {corpus} | K90 | — | {row['k90']} | {marginal['results'][corpus]['k90']} |")
    lines += ["", "Coordinated greedy is exact for its one-step criterion but is not a global subset optimizer. A large discovery/held-out gap would weaken any sparsity conclusion from this selector; similar held-out breadth under both rankings would strengthen the conclusion that no small stable native bank captures this loss family."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
