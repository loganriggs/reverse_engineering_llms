#!/usr/bin/env python3
"""Select frozen local operator modes by their true-token CE relevance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_mlp_path_type_program import norm
from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_layer16_logit_ce_program import propagate_logits
from run_bilin18_late_mlp_native_channel_program import build_axes, collect_states
from run_bilin18_late_mlp_operator_atlas import derivative_targets
from run_bilin18_late_mlp_tensor_canonicalization import exact_reads, gram_modes
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_ce_aware_mode_selection.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_ce_aware_mode_selection.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_CE_AWARE_MODE_SELECTION.md"


def score(target, prediction):
    return 1 - float((target.double() - prediction.double()).square().sum()
                     / target.double().square().sum().clamp_min(1e-30))


@torch.inference_mode(False)
def ce_gradient(model, row17, position, labels):
    block = model.blocks[17]
    base_mixed = row17["mixed"].detach().clone()
    bus = row17["bus"].detach().clone()
    delta = torch.zeros(len(labels), base_mixed.shape[-1], device="cuda", requires_grad=True)
    injection = torch.zeros_like(base_mixed)
    injection[:, position] = delta
    mixed = base_mixed + block.lambdas[0] * injection
    block.attn.rotary.seq_len_cached = None
    block.attn.rotary.cos_cached = None
    block.attn.rotary.sin_cached = None
    attention, _ = block.attn(norm(mixed), bus)
    pre = mixed + attention
    final = norm(pre + block.mlp(norm(pre)))
    raw = model.lm_head(final[:, position])
    logits = 30 * torch.tanh(raw / 30)
    loss = F.cross_entropy(logits, labels, reduction="sum")
    gradient, = torch.autograd.grad(loss, delta)
    return gradient.detach(), float(loss.detach())


@torch.inference_mode(False)
def collect(model, spec, config, local_axes, validate=False):
    source = load_rows(spec); states = []; gradients = []; targets = []
    ce_sum = 0.0; count = 0; validation_error = 0.0
    for start in range(0, len(source), config["batch_size"]):
        batch_array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(batch_array[:, :128].astype(np.int64), device="cuda")
        _, cache = native_cache(model, ids, config); by_layer = {row["index"]: row for row in cache}
        state = by_layer[16]["state"][:, config["positions"]].float()
        flat = state.flatten(0, 1).cpu()
        local = derivative_targets(model.blocks[16], flat, local_axes, config["batch_size"])
        local = local.view(len(ids), len(config["positions"]), config["rank"], -1)
        for offset, position in enumerate(config["positions"]):
            labels = torch.as_tensor(batch_array[:, position + 1].astype(np.int64), device="cuda")
            gradient, loss = ce_gradient(model, by_layer[17], position, labels)
            directional = torch.einsum("bnd,bd->bn", local[:, offset].cuda(), gradient).cpu()
            if validate and start == 0 and offset == 0:
                tangent, logits, _ = propagate_logits(
                    model, by_layer[17], position, local[:, offset], config.get("tangent_chunk", 8))
                probability = logits.softmax(-1)
                forward = ((probability[:, None] * tangent).sum(-1)
                           - tangent.gather(-1, labels[:, None, None].expand(-1, config["rank"], 1)).squeeze(-1)).cpu()
                validation_error = float((directional - forward).double().norm()
                                         / forward.double().norm().clamp_min(1e-30))
            states.append(state[:, offset].cpu()); gradients.append(gradient.cpu()); targets.append(directional)
            ce_sum += loss; count += len(labels)
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    return {"states": torch.cat(states), "gradients": torch.cat(gradients),
            "target": torch.cat(targets), "native_cross_entropy": ce_sum / count,
            "forward_reverse_relative_error": validation_error}


@torch.inference_mode()
def contributions(data, modes, reads, input_rank):
    coefficients = data["states"].float().cuda() @ reads.T.cuda()
    matrices = modes.view(len(modes), input_rank, -1).cuda()
    projected = torch.einsum("knd,sd->skn", matrices, data["gradients"].float().cuda())
    return (coefficients[:, :, None] * projected).cpu()


def greedy_order(contribution, target):
    flat = contribution.permute(1, 0, 2).flatten(1).double()
    target_flat = target.flatten().double()
    gram = flat @ flat.T; target_dot = flat @ target_flat
    selected = []; residual_dot = target_dot.clone()
    available = torch.ones(len(flat), dtype=torch.bool)
    for _ in range(len(flat)):
        reduction = 2 * residual_dot - torch.diagonal(gram)
        reduction[~available] = -torch.inf
        index = int(torch.argmax(reduction)); selected.append(index); available[index] = False
        residual_dot -= gram[:, index]
    return selected


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
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    local_axes = build_axes(collect_states(model, config["axis_discovery"], config), config["rank"])[16]
    local_states = collect_states(model, config["local_atlas_discovery"], config)[16]
    local_target = derivative_targets(model.blocks[16], local_states, local_axes, config["batch_size"])
    modes, _, _ = gram_modes(local_target, config["operator_modes"])
    reads = exact_reads(model.blocks[16], local_axes, modes)
    print("selection discovery", flush=True)
    discovery = collect(model, config["selection_discovery"], config, local_axes, validate=True)
    discovery_contribution = contributions(discovery, modes, reads, config["rank"])
    ordering = greedy_order(discovery_contribution, discovery["target"])
    discovery_budgets = {
        str(budget): {
            "ce_aware_fidelity": score(discovery["target"], discovery_contribution[:, ordering[:budget]].sum(1)),
            "energy_prefix_fidelity": score(discovery["target"], discovery_contribution[:, :budget].sum(1)),
        }
        for budget in config["budgets"]
    }
    generator = torch.Generator().manual_seed(config["seed"]); results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = collect(model, spec, config, local_axes)
        contribution = contributions(data, modes, reads, config["rank"])
        budgets = {}
        for budget in config["budgets"]:
            selected = ordering[:budget]
            budgets[str(budget)] = {
                "ce_aware_fidelity": score(data["target"], contribution[:, selected].sum(1)),
                "energy_prefix_fidelity": score(data["target"], contribution[:, :budget].sum(1)),
            }
            if budget == 64:
                budgets[str(budget)]["random"] = random_summary(
                    contribution, data["target"], budget, config["random_replicates"], generator)
        results[corpus] = {"sites": len(data["target"]),
                           "native_cross_entropy": data["native_cross_entropy"],
                           "budgets": budgets}
    overlap = len(set(ordering[:64]) & set(range(64)))
    validation_error = discovery["forward_reverse_relative_error"]
    hypotheses = {
        "B0_reverse_ce_derivative_matches_forward_jvp": validation_error <= 1e-5,
        "B1_ce_aware_64_mode_program_is_sufficient": all(row["budgets"]["64"]["ce_aware_fidelity"] >= .90 for row in results.values()),
        "B2_ce_aware_selection_beats_energy_prefix": all(row["budgets"]["64"]["ce_aware_fidelity"] >= row["budgets"]["64"]["energy_prefix_fidelity"] + .10 for row in results.values()),
        "B3_ce_aware_128_mode_program_is_high_fidelity": all(row["budgets"]["128"]["ce_aware_fidelity"] >= .95 for row in results.values()),
        "B4_ce_aware_selection_transfers_ood": all(results[name]["budgets"]["64"]["ce_aware_fidelity"] >= results["fineweb"]["budgets"]["64"]["ce_aware_fidelity"] - .10 for name in ("wikitext", "tinystories")),
        "B5_ce_aware_selection_beats_random": all(row["budgets"]["64"]["ce_aware_fidelity"] >= row["budgets"]["64"]["random"]["p95"] + .10 for row in results.values()),
        "B6_top256_local_modes_cover_loss": all(row["budgets"]["256"]["ce_aware_fidelity"] >= .98 for row in results.values()),
        "B7_loss_relevance_reorders_local_energy_modes": overlap < 32,
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["B0_reverse_ce_derivative_matches_forward_jvp"]["relative_error"] = validation_error
    hypothesis_rows["B7_loss_relevance_reorders_local_energy_modes"]["top64_overlap"] = overlap
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "ce_aware_mode_order": ordering,
               "selection_discovery_budgets": discovery_budgets, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# CE-aware selection from the frozen Layer-16 operator atlas", "",
             "The local operator modes and exact state read vectors are unchanged. A fixed bank is selected only by its contribution to true-next-token CE derivatives on discovery FineWeb, using exact reverse-mode gradients from the Layer-16 write through the remaining model. This is metric-aware mode selection, not per-example fitting.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Forward/reverse CE derivative relative error: `{validation_error:.2e}`. Top-64 overlap with local-energy prefix: `{overlap}/64`.",
              f"On selection discovery, K64 CE-aware/energy-prefix fidelity is `{discovery_budgets['64']['ce_aware_fidelity']:.3f}` / `{discovery_budgets['64']['energy_prefix_fidelity']:.3f}`.", "",
              "| Corpus | K | CE-aware | Energy prefix | Random p95 (K64) |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for budget in config["budgets"]:
            item = row["budgets"][str(budget)]
            random = f"{item['random']['p95']:.3f}" if "random" in item else "—"
            lines.append(f"| {corpus} | {budget} | {item['ce_aware_fidelity']:.3f} | {item['energy_prefix_fidelity']:.3f} | {random} |")
    lines += ["", "Selection can improve one quotient while harming another. These banks are justified for local true-token loss derivatives only; they are not automatically the best state, full-logit, Fisher, or semantic basis. The 256-mode ceiling also tests whether the discarded local spectral tail carries disproportionate loss relevance."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
