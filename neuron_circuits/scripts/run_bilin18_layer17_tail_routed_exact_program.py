#!/usr/bin/env python3
"""Test whether Layer-17 tail complexity is conditionally low-rank."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_bottleneck_tail_ontology import corpus_rows
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_continuous_fisher import continuous_prediction, score_summary, summarize
from run_bilin18_layer17_tail_fisher_basis import collect_fisher, fisher_quadratic
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_routed_exact_program.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
FRONTIER_ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_rank_frontier.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_routed_exact_program.json"
CHECKPOINT = TENSOR_ROOT / "layer17_tail_routed_exact_program.pt"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_ROUTED_EXACT_PROGRAM.md"
FIGURE = TENSOR_ROOT / "layer17_tail_routed_exact_program.png"


class RoutedTensorProgram(torch.nn.Module):
    def __init__(self, pool_rank, expert_rank, experts, router_features, base, seed):
        super().__init__()
        generator = torch.Generator(device=base.device).manual_seed(seed)
        initial = base[None].repeat(experts, 1, 1)
        initial = initial + 0.08 * torch.randn(initial.shape, generator=generator, device=base.device)
        self.expert_parameters = torch.nn.Parameter(initial)
        self.router_weight = torch.nn.Parameter(
            0.01 * torch.randn(router_features, experts, generator=generator, device=base.device)
        )
        self.router_bias = torch.nn.Parameter(torch.zeros(experts, device=base.device))
        self.router_features = router_features

    def quotients(self):
        return torch.linalg.qr(self.expert_parameters, mode="reduced").Q

    def forward(self, coordinates, forms, hard=False):
        quotients = self.quotients()
        reduced_coordinates = torch.einsum("bp,epr->ber", coordinates, quotients)
        reduced_forms = torch.einsum("epa,kpq,eqb->ekab", quotients, forms, quotients)
        expert_scores = torch.einsum(
            "bei,ekij,bej->bek", reduced_coordinates, reduced_forms, reduced_coordinates
        )
        logits = coordinates[:, :self.router_features] @ self.router_weight + self.router_bias
        soft_gates = logits.softmax(-1)
        if hard:
            gates = F.one_hot(soft_gates.argmax(-1), soft_gates.shape[1]).to(soft_gates.dtype)
        else:
            gates = soft_gates
        return torch.einsum("be,bek->bk", gates, expert_scores), gates


def route_summary(gates):
    mean_gate = gates.mean(0)
    entropy = -(mean_gate * mean_gate.clamp_min(1e-12).log()).sum()
    site_entropy = -(gates * gates.clamp_min(1e-12).log()).sum(1).mean()
    return {
        "mean_gate": mean_gate.detach().cpu().tolist(),
        "effective_experts": float(entropy.exp()),
        "mean_site_entropy": float(site_entropy),
        "hard_shares": torch.bincount(gates.argmax(1), minlength=gates.shape[1]).float().div(len(gates)).cpu().tolist(),
    }


def fit_fisher(program, coordinates, forms, target, metric, train, validation, config):
    optimizer = torch.optim.Adam(program.parameters(), lr=config["fisher_learning_rate"])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 10)
    history = []
    for step in range(config["fisher_steps"]):
        sampled = train[torch.randint(len(train), (config["fisher_batch_size"],), generator=generator, device="cuda")]
        prediction, gates = program(coordinates[sampled], forms)
        causal = fisher_quadratic(target[sampled] - prediction, metric[sampled]).mean()
        mean_gate = gates.mean(0)
        balance = gates.shape[1] * (mean_gate - 1 / gates.shape[1]).square().sum()
        entropy = -(gates * gates.clamp_min(1e-12).log()).sum(1).mean()
        loss = causal + config["balance_weight"] * balance + config["entropy_weight"] * entropy
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 50 == 0 or step == config["fisher_steps"] - 1:
            with torch.no_grad():
                val_prediction, val_gates = program(coordinates[validation], forms)
                val = fisher_quadratic(target[validation] - val_prediction, metric[validation]).mean()
            history.append({"step": step, "train_fisher": float(causal.detach()), "validation_fisher": float(val), **route_summary(val_gates)})
    return history


@torch.inference_mode()
def collect_exact_context(model, spec, config):
    rows = corpus_rows(spec); positions = config["positions"]
    pre_rows, write_rows, target_rows = [], [], []
    block = model.blocks[config["layer"]]
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1)
        x = normalized[:, positions].float().flatten(0, 1)
        write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x)).float() + block.mlp.Down_bias.float()
        targets = torch.as_tensor(rows[start:start + len(raw), np.asarray(positions) + 1].reshape(-1), device="cuda", dtype=torch.long)
        pre_rows.append(pre); write_rows.append(write); target_rows.append(targets)
    return torch.cat(pre_rows), torch.cat(write_rows), torch.cat(target_rows)


def exact_batch_loss(program, indices, coordinates, forms, isotropic, pre, native_write,
                     native_score, targets, output_basis, model, ce_weight):
    dimension = model.config.n_embd
    prediction, _ = program(coordinates[indices], forms)
    prediction = prediction + isotropic[indices]
    changed_write = native_write[indices] + (prediction - native_score[indices]) @ output_basis.T
    changed_logits = 30 * torch.tanh(
        model.lm_head(F.rms_norm(pre[indices] + changed_write, (dimension,))).float() / 30
    )
    changed_logprob = changed_logits.log_softmax(-1)
    with torch.no_grad():
        native_logits = 30 * torch.tanh(
            model.lm_head(F.rms_norm(pre[indices] + native_write[indices], (dimension,))).float() / 30
        )
        native_logprob = native_logits.log_softmax(-1)
        native_probability = native_logprob.exp()
    kl = (native_probability * (native_logprob - changed_logprob)).sum(1).mean()
    ce = -changed_logprob.gather(1, targets[indices, None]).mean()
    return kl + ce_weight * ce, kl, ce


def fit_exact(program, coordinates, forms, isotropic, pre, native_write, native_score,
              targets, output_basis, model, train, validation, config, ce_weight, seed):
    optimizer = torch.optim.Adam(program.parameters(), lr=config["exact_learning_rate"])
    generator = torch.Generator(device="cuda").manual_seed(seed)
    history = []
    for step in range(config["exact_steps"]):
        sampled = train[torch.randint(len(train), (config["exact_batch_size"],), generator=generator, device="cuda")]
        loss, kl, ce = exact_batch_loss(
            program, sampled, coordinates, forms, isotropic, pre, native_write,
            native_score, targets, output_basis, model, ce_weight,
        )
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 20 == 0 or step == config["exact_steps"] - 1:
            sampled_val = validation[torch.randint(len(validation), (128,), generator=generator, device="cuda")]
            with torch.no_grad():
                val_loss, val_kl, val_ce = exact_batch_loss(
                    program, sampled_val, coordinates, forms, isotropic, pre, native_write,
                    native_score, targets, output_basis, model, ce_weight,
                )
            history.append({"step": step, "train_loss": float(loss.detach()), "train_kl": float(kl.detach()), "train_ce": float(ce.detach()), "validation_loss": float(val_loss), "validation_kl": float(val_kl), "validation_ce": float(val_ce)})
    return history


@torch.inference_mode()
def evaluate(model, spec, config, output_basis, inverse, candidate_basis, forms, traces, fixed, routed):
    rows = corpus_rows(spec); positions = config["positions"]
    block = model.blocks[config["layer"]]; dimension = model.config.n_embd
    names = list(fixed) + [f"{name}_{mode}" for name in routed for mode in ("soft", "hard")]
    records = {name: {field: [] for field in ("exact_kl", "ce_delta", "top1")} for name in names}
    scores = {name: [] for name in names}; target_scores = []; gate_rows = {name: [] for name in routed}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda"); pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1); x = normalized[:, positions].float().flatten(0, 1)
        raw_write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x)).float()
        native_write = raw_write + block.mlp.Down_bias.float(); native_score = raw_write @ output_basis
        z = x @ inverse; coordinates = z @ candidate_basis
        isotropic = z.square().sum(1, keepdim=True) * traces[None]
        predictions = {name: continuous_prediction(coordinates, forms, quotient) + isotropic for name, quotient in fixed.items()}
        for name, program in routed.items():
            soft, gates = program(coordinates, forms, hard=False); hard, _ = program(coordinates, forms, hard=True)
            predictions[f"{name}_soft"] = soft + isotropic; predictions[f"{name}_hard"] = hard + isotropic
            gate_rows[name].append(gates.cpu())
        target_scores.append(native_score.cpu())
        native_logits = 30 * torch.tanh(model.lm_head(F.rms_norm(pre + native_write, (dimension,))).float() / 30)
        native_logprob = native_logits.log_softmax(-1); native_probability = native_logprob.exp(); native_top1 = native_logits.argmax(-1)
        targets = torch.as_tensor(rows[start:start + len(raw), np.asarray(positions) + 1].reshape(-1), device="cuda", dtype=torch.long)
        native_ce = -native_logprob.gather(1, targets[:, None]).squeeze(1)
        for name, prediction in predictions.items():
            scores[name].append(prediction.cpu())
            changed_write = native_write + (prediction - native_score) @ output_basis.T
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(pre + changed_write, (dimension,))).float() / 30)
            logprob = logits.log_softmax(-1)
            exact_kl = (native_probability * (native_logprob - logprob)).sum(1)
            ce = -logprob.gather(1, targets[:, None]).squeeze(1)
            records[name]["exact_kl"].extend(exact_kl.cpu().tolist()); records[name]["ce_delta"].extend((ce-native_ce).cpu().tolist()); records[name]["top1"].extend((logits.argmax(-1)==native_top1).cpu().tolist())
    target = torch.cat(target_scores)
    return {
        "scores": {name: score_summary(target, torch.cat(value)) for name, value in scores.items()},
        "behavioral": {name: summarize(value) for name, value in records.items()},
        "routing": {name: route_summary(torch.cat(value)) for name, value in gate_rows.items()},
    }


def main():
    config = json.loads(CONFIG.read_text()); source = np.load(SOURCE_ARRAYS); frontier = np.load(FRONTIER_ARRAYS)
    output_basis = torch.from_numpy(source["tail_output_basis"][:, :4]).cuda().float()
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    block = model.blocks[config["layer"]]; down = block.mlp.Down.weight.float().detach(); left = block.mlp.Left.weight.float().detach(); right = block.mlp.Right.weight.float().detach()
    print("constructing tensor and discovery objective", flush=True)
    discovery_x, discovery_write = collect(model, config["layer"], config["discovery"], config)
    square, inverse, covariance = covariance_transforms(discovery_x, config["covariance_floor_fraction"])
    cleaned_forms, traces = [], []
    for component in range(4):
        coefficients = down.T @ output_basis[:, component]; raw = left.T @ (coefficients[:, None] * right); raw = .5 * (raw + raw.T)
        clean, trace = trace_clean(square @ raw @ square); cleaned_forms.append(clean); traces.append(trace)
    traces_tensor = torch.stack(traces); _, common_basis = common_input(cleaned_forms); candidate_basis = common_basis[:, :config["candidate_pool_rank"]]
    forms = torch.einsum("di,kde,ej->kij", candidate_basis, torch.stack(cleaned_forms), candidate_basis).detach()
    z = discovery_x @ inverse; coordinates = (z @ candidate_basis).detach(); isotropic = (z.square().sum(1, keepdim=True) * traces_tensor[None]).detach()
    target = (discovery_write @ output_basis - isotropic).detach(); metric = collect_fisher(model, config["discovery"], config, output_basis).detach()
    row_ids = torch.arange(len(discovery_x), device="cuda") // len(config["positions"]); even = torch.nonzero(row_ids % 2 == 0, as_tuple=False).flatten(); odd = torch.nonzero(row_ids % 2 == 1, as_tuple=False).flatten()
    global8 = torch.from_numpy(frontier["rank_8"]).cuda().float(); global32 = torch.from_numpy(frontier["rank_32"]).cuda().float()
    fisher_program = RoutedTensorProgram(config["candidate_pool_rank"], config["expert_rank"], config["experts"], config["router_features"], global8, config["seed"]).cuda()
    print("training Fisher router", flush=True); fisher_history = fit_fisher(fisher_program, coordinates, forms, target, metric, even, odd, config)
    with torch.no_grad():
        fisher_prediction, _ = fisher_program(coordinates[odd], forms)
        global32_prediction = continuous_prediction(coordinates[odd], forms, global32)
        opposite_half_fisher = {
            "routed": float(fisher_quadratic(target[odd]-fisher_prediction, metric[odd]).mean()),
            "global32": float(fisher_quadratic(target[odd]-global32_prediction, metric[odd]).mean()),
        }
    print("collecting exact discovery context", flush=True); pre, native_write, targets = collect_exact_context(model, config["discovery"], config)
    native_score = (discovery_write @ output_basis).detach()
    kl_program = copy.deepcopy(fisher_program); ce_program = copy.deepcopy(fisher_program)
    print("fine-tuning exact KL", flush=True); kl_history = fit_exact(kl_program, coordinates, forms, isotropic, pre, native_write, native_score, targets, output_basis, model, even, odd, config, 0.0, config["seed"]+20)
    print("fine-tuning KL plus CE", flush=True); ce_history = fit_exact(ce_program, coordinates, forms, isotropic, pre, native_write, native_score, targets, output_basis, model, even, odd, config, config["ce_weight"], config["seed"]+30)
    fixed = {"global8": global8, "global32": global32}; routed = {"fisher": fisher_program, "kl": kl_program, "ce": ce_program}
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("external evaluation", corpus, flush=True); results[corpus] = evaluate(model, spec, config, output_basis, inverse, candidate_basis, forms, traces_tensor, fixed, routed)
    c1 = opposite_half_fisher["routed"] <= .8 * opposite_half_fisher["global32"]
    c2 = all(r["behavioral"]["kl_soft"]["mean_exact_kl"] <= .8*r["behavioral"]["global32"]["mean_exact_kl"] for r in results.values())
    c3 = all(r["behavioral"]["ce_soft"]["ce_delta"] <= 0 for r in results.values())
    c4 = all(r["behavioral"]["kl_hard"]["mean_exact_kl"] <= r["behavioral"]["global32"]["mean_exact_kl"] - .8*(r["behavioral"]["global32"]["mean_exact_kl"]-r["behavioral"]["kl_soft"]["mean_exact_kl"]) for r in results.values())
    c5 = all(r["routing"]["kl"]["effective_experts"] >= 3 for r in results.values())
    c6 = all(r["behavioral"]["kl_soft"]["median_exact_kl"] <= .005 for r in results.values())
    c7 = all(r["behavioral"]["kl_soft"]["mean_exact_kl"] < r["behavioral"]["fisher_soft"]["mean_exact_kl"] for r in results.values())
    hypotheses = {
        "C1_routing_beats_global32_local_fisher": {"supported": c1}, "C2_exact_kl_router_beats_global32_ood": {"supported": c2},
        "C3_task_aware_router_improves_ce": {"supported": c3}, "C4_hard_routing_retains_gain": {"supported": c4},
        "C5_router_uses_multiple_experts": {"supported": c5}, "C6_routed_rank8_reaches_behavioral_gate": {"supported": c6},
        "C7_exact_objective_improves_fisher_router": {"supported": c7},
    }
    torch.save({"fisher": fisher_program.state_dict(), "kl": kl_program.state_dict(), "ce": ce_program.state_dict(), "config": config}, CHECKPOINT)
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "covariance": covariance, "opposite_half_fisher": opposite_half_fisher, "fisher_history": fisher_history, "kl_history": kl_history, "ce_history": ce_history, "results": results}, indent=2, allow_nan=False))
    lines = ["# Context-routed exact Layer-17 tail program", "", "Four rank-eight tensor experts are routed from 32 frozen input coordinates. The program is trained first with pulled-back Fisher error, then cloned for exact native-distribution KL and KL-plus-observed-token-CE fine-tuning. Fixed global rank 8/32 and hard-routing controls use the same tensor pool.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Opposite-half Fisher: routed **{opposite_half_fisher['routed']:.4e}**, global32 **{opposite_half_fisher['global32']:.4e}**.", "", "![Routed program](layer17_tail_routed_exact_program.png)", "", "| Corpus | Program | Mean KL | Median KL | CE Δ | Top1 |", "|---|---|---:|---:|---:|---:|"]
    display = ["global8", "global32", "fisher_soft", "fisher_hard", "kl_soft", "kl_hard", "ce_soft", "ce_hard"]
    for corpus, row in results.items():
        for name in display:
            b = row["behavioral"][name]; lines.append(f"| {corpus} | {name} | {b['mean_exact_kl']:.5f} | {b['median_exact_kl']:.5f} | {b['ce_delta']:.5f} | {b['top1_agreement']:.3f} |")
        route = row["routing"]["kl"]; lines.append(f"| {corpus} | KL router support | — | — | — | effective experts {route['effective_experts']:.2f} |")
    lines += ["", "A soft-only gain indicates an ensemble code, not sparse conditional execution. A hard-route gain with multiple balanced experts supports conditional compactness. Failure against global rank 32 supports a genuinely broad simultaneous residual or an inadequate linear router."]
    REPORT.write_text("\n".join(lines)+"\n")
    figure, plots = plt.subplots(1,3,figsize=(15,4.5)); x=np.arange(len(results)); width=.12
    for index,name in enumerate(display):
        offset=(index-(len(display)-1)/2)*width
        plots[0].bar(x+offset,[results[c]["behavioral"][name]["mean_exact_kl"] for c in results],width,label=name)
        plots[1].bar(x+offset,[results[c]["behavioral"][name]["ce_delta"] for c in results],width,label=name)
    for corpus,row in results.items(): plots[2].plot(range(config["experts"]),row["routing"]["kl"]["mean_gate"],marker="o",label=corpus)
    plots[0].set_title("Native-distribution fidelity"); plots[0].set_ylabel("mean exact KL"); plots[1].set_title("Observed-token utility"); plots[1].set_ylabel("mean CE delta"); plots[2].set_title("KL-router utilization"); plots[2].set_ylabel("mean gate")
    for plot in plots[:2]: plot.set_xticks(x,list(results)); plot.tick_params(axis="x",rotation=15); plot.grid(alpha=.2)
    plots[0].legend(fontsize=6,ncol=2); plots[2].set_xlabel("expert"); plots[2].legend(fontsize=8); plots[2].grid(alpha=.2)
    figure.tight_layout(); figure.savefig(FIGURE,dpi=180); plt.close(figure); print("wrote",OUTPUT,CHECKPOINT,REPORT,FIGURE,flush=True)


if __name__ == "__main__": main()
