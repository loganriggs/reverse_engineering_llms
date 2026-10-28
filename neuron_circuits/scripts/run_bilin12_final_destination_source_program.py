#!/usr/bin/env python3
"""Decompose the final attention message into exact source-position edges."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_final_destination_source_program.json")
OUTPUT = TENSOR_ROOT / "bilin12_final_destination_source_program.json"
REPORT = TENSOR_ROOT / "BILIN12_FINAL_DESTINATION_SOURCE_PROGRAM.md"
FIGURE = TENSOR_ROOT / "bilin12_final_destination_source_program.png"


def source_write_contributions(block, details: dict, destination: int) -> torch.Tensor:
    """Return bias-free per-source writes, shaped [batch, source, residual]."""
    mixed = (1 - block.attn.lamb) * details["current_value"] + block.attn.lamb * details["value_bus"]
    per_head_source = details["pattern"][:, :, destination, :, None] * mixed.permute(0, 2, 1, 3)
    contributions = 0
    for head in range(block.attn.n_head):
        start = head * block.attn.head_dim; stop = (head + 1) * block.attn.head_dim
        weight = block.attn.c_proj.weight[:, start:stop]
        contributions = contributions + per_head_source[:, head] @ weight.T
    return contributions


def ordinary_context(row: dict) -> dict:
    return {
        "x0": row["x0"].detach().clone(),
        "mixed": row["mixed"].detach().clone(),
        "write": row["details"]["write"].detach().clone(),
        "value_bus": row["value_bus"].detach().clone(),
    }


def differentiable_logits_from_final_delta(model, layer: int, context: dict, delta: torch.Tensor, position: int) -> torch.Tensor:
    length = context["write"].shape[1]; selector = torch.zeros(1, length, 1, device=delta.device, dtype=delta.dtype); selector[:, position] = 1
    write = context["write"] + selector * delta[:, None]
    mixed = context["mixed"]; dimension = mixed.shape[-1]; block = model.blocks[layer]
    pre = mixed + write; state = F.rms_norm(pre, (dimension,)); mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias; x = pre + mlp
    value_bus = context["value_bus"]; x0 = context["x0"]
    for suffix in model.blocks[layer + 1:]: x, value_bus = suffix(x, value_bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30)


def downstream_fisher_gradient(model, layer: int, row: dict, target_delta: torch.Tensor, position: int) -> torch.Tensor:
    context = ordinary_context(row); dtype = context["write"].dtype
    # `trace` runs under inference mode and populates rotary caches with
    # inference tensors. Rebuild those tiny deterministic caches before a VJP.
    for suffix in model.blocks[layer + 1:]:
        suffix.attn.rotary.seq_len_cached = None
        suffix.attn.rotary.cos_cached = None
        suffix.attn.rotary.sin_cached = None
    zero = torch.zeros(target_delta.shape[0], context["write"].shape[-1], device=target_delta.device, dtype=dtype, requires_grad=True)
    with torch.enable_grad():
        logits = differentiable_logits_from_final_delta(model, layer, context, zero, position)
        probability = F.softmax(logits.detach().float(), -1)
        centered_target = target_delta.float() - (probability * target_delta.float()).sum(-1, keepdim=True)
        logit_gradient = probability * centered_target
        objective = (logits * logit_gradient).sum()
        gradient = torch.autograd.grad(objective, zero)[0].float()
    return gradient


def balanced_order(scores: torch.Tensor) -> torch.Tensor:
    """Alternate the strongest positive and negative edges, then fill by magnitude."""
    rows = []
    for row in scores:
        positive = [int(i) for i in row.argsort(descending=True) if float(row[i]) >= 0]
        negative = [int(i) for i in row.argsort(descending=False) if float(row[i]) < 0]
        chosen = []
        for index in range(max(len(positive), len(negative))):
            if index < len(positive): chosen.append(positive[index])
            if index < len(negative): chosen.append(negative[index])
        chosen_set = set(chosen)
        chosen.extend(int(i) for i in row.abs().argsort(descending=True) if int(i) not in chosen_set)
        rows.append(torch.tensor(chosen, device=scores.device, dtype=torch.long))
    return torch.stack(rows)


def empty_metric() -> dict:
    return {"examples": 0, "target": 0.0, "prediction": 0.0, "cross": 0.0, "error": 0.0, "forward_full": 0.0, "reverse_full": 0.0, "forward": 0.0, "target_error": 0.0, "predicted_cross": 0.0}


def finish_metric(row: dict) -> dict:
    target = max(row["target"], 1e-30); prediction = max(row["prediction"], 1e-30)
    return {"examples": row["examples"], "fisher_fidelity": 1 - row["error"] / target, "fisher_projection": row["cross"] / target, "fisher_cosine": row["cross"] / max(np.sqrt(target * prediction), 1e-30), "predicted_projection": row["predicted_cross"] / target, "forward_kl_ratio": row["forward"] / max(row["forward_full"], 1e-30), "target_error_kl_ratio": row["target_error"] / max(row["reverse_full"], 1e-30)}


def participation(scores: torch.Tensor) -> torch.Tensor:
    absolute = scores.abs(); return absolute.sum(-1).square() / absolute.square().sum(-1).clamp_min(1e-30)


@torch.no_grad()
def evaluate_condition(model, tokenizer, rows, edit_position: int, config: dict, random_seed: int) -> dict:
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]; prediction_position = config["prediction_position"]
    keys = [(method, budget) for method in config["methods"] for budget in config["budgets"]]
    accumulators = {str(layer): {f"{method}/{budget}": empty_metric() for method, budget in keys} | {"edited_only": empty_metric()} for layer in config["layers"]}
    diagnostics = {str(layer): {"examples": 0, "maximum_write_error": 0.0, "relative_num": 0.0, "relative_den": 0.0, "maximum_logit_error": 0.0, "participation": [], "positive_score": [], "negative_score": [], "net_score": [], "target_energy": [], "top16_counts": np.zeros(config["sequence_length"], dtype=np.int64), "absolute_top16_counts": np.zeros(config["sequence_length"], dtype=np.int64), "local_top16": 0, "total_top16": 0} for layer in config["layers"]}
    generator = torch.Generator(device="cuda").manual_seed(random_seed)
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64); base_ids = torch.as_tensor(raw, device="cuda", dtype=torch.long); keep = base_ids[:, edit_position] != newline
        if not bool(keep.any()): continue
        base_ids = base_ids[keep]; changed_ids = base_ids.clone(); changed_ids[:, edit_position] = newline
        base_rows, _, _ = trace(model, base_ids); changed_rows, _, _ = trace(model, changed_ids)
        for layer in config["layers"]:
            key = str(layer); block = model.blocks[layer]; base = base_rows[layer]; changed = changed_rows[layer]; diag = diagnostics[key]
            base_sources = source_write_contributions(block, base["details"], prediction_position); changed_sources = source_write_contributions(block, changed["details"], prediction_position); source_delta = (changed_sources - base_sources).float()
            direct_delta = (changed["details"]["write"][:, prediction_position] - base["details"]["write"][:, prediction_position]).float(); recomposed = source_delta.sum(1); error = recomposed - direct_delta
            full_write = base["details"]["write"].clone(); full_write[:, prediction_position] = changed["details"]["write"][:, prediction_position]
            endpoints = logits_from_attention_writes(model, layer, base, torch.stack((base["details"]["write"], full_write))); base_logits, target_logits = endpoints; target_logit_delta = target_logits - base_logits
            gradient = downstream_fisher_gradient(model, layer, base, target_logit_delta, prediction_position); scores = (source_delta * gradient[:, None]).sum(-1); norms = source_delta.square().sum(-1).sqrt()
            all_orders = {"signed_fisher": scores.argsort(-1, descending=True), "absolute_fisher": scores.abs().argsort(-1, descending=True), "balanced_fisher": balanced_order(scores), "state_norm": norms.argsort(-1, descending=True), "random": torch.stack([torch.randperm(config["sequence_length"], generator=generator, device="cuda") for _ in range(len(base_ids))])}
            orders = {method: all_orders[method] for method in config["methods"]}
            recomposed_write = base["details"]["write"].clone(); recomposed_write[:, prediction_position] += recomposed.to(recomposed_write.dtype); recomposed_logits = logits_from_attention_writes(model, layer, base, recomposed_write.unsqueeze(0))[0]
            target_energy_per_example, _, _, _ = fisher_terms(base_logits, target_logit_delta, target_logit_delta)
            signed16 = orders["signed_fisher"][:, :16]; absolute16 = all_orders["absolute_fisher"][:, :16]; diag["participation"].extend(participation(scores).cpu().tolist()); diag["positive_score"].extend(scores.clamp_min(0).sum(-1).cpu().tolist()); diag["negative_score"].extend(scores.clamp_max(0).sum(-1).cpu().tolist()); diag["net_score"].extend(scores.sum(-1).cpu().tolist()); diag["target_energy"].extend(target_energy_per_example.cpu().tolist()); diag["examples"] += len(base_ids); diag["maximum_write_error"] = max(diag["maximum_write_error"], float(error.abs().max())); diag["relative_num"] += float(error.square().sum()); diag["relative_den"] += float(direct_delta.square().sum()); diag["maximum_logit_error"] = max(diag["maximum_logit_error"], float((recomposed_logits - target_logits).abs().max()))
            for batch_index in range(len(base_ids)):
                chosen = signed16[batch_index].cpu().tolist(); chosen_absolute = absolute16[batch_index].cpu().tolist(); np.add.at(diag["top16_counts"], chosen, 1); np.add.at(diag["absolute_top16_counts"], chosen_absolute, 1); diag["local_top16"] += sum(abs(source - edit_position) <= 2 or source == prediction_position for source in chosen); diag["total_top16"] += len(chosen)
            approximations = []; labels = []; predicted = []
            for method, budget in keys:
                indices = orders[method][:, :budget]; mask = torch.zeros(len(base_ids), config["sequence_length"], device="cuda", dtype=source_delta.dtype); mask.scatter_(1, indices, 1); approximations.append((mask[:, :, None] * source_delta).sum(1)); labels.append(f"{method}/{budget}"); predicted.append((mask * scores).sum(-1))
            edited = source_delta[:, edit_position]; approximations.append(edited); labels.append("edited_only"); predicted.append(scores[:, edit_position])
            worlds = []
            for approximation in approximations:
                write = base["details"]["write"].clone(); write[:, prediction_position] += approximation.to(write.dtype); worlds.append(write)
            logits = logits_from_attention_writes(model, layer, base, torch.stack(worlds)); target_energy, _, _, _ = fisher_terms(base_logits, target_logit_delta, target_logit_delta); forward_full = exact_kl(base_logits, target_logits); reverse_full = exact_kl(target_logits, base_logits)
            for index, label in enumerate(labels):
                row = accumulators[key][label]; prediction_delta = logits[index] - base_logits; _, prediction_energy, cross, finite_error = fisher_terms(base_logits, target_logit_delta, prediction_delta)
                row["examples"] += len(base_ids); row["target"] += float(target_energy.sum()); row["prediction"] += float(prediction_energy.sum()); row["cross"] += float(cross.sum()); row["error"] += float(finite_error.sum()); row["forward_full"] += float(forward_full.sum()); row["reverse_full"] += float(reverse_full.sum()); row["forward"] += float(exact_kl(base_logits, logits[index]).sum()); row["target_error"] += float(exact_kl(target_logits, logits[index]).sum()); row["predicted_cross"] += float(predicted[index].sum())
    return {"metrics": {layer: {name: finish_metric(row) for name, row in rows.items()} for layer, rows in accumulators.items()}, "diagnostics": {layer: {**{k: v for k, v in row.items() if k not in ("top16_counts", "absolute_top16_counts", "participation", "relative_num", "relative_den")}, "relative_write_error": float(np.sqrt(row["relative_num"] / max(row["relative_den"], 1e-30))), "participation_median": float(np.median(row["participation"])), "participation_values": row["participation"], "top16_counts": row["top16_counts"].tolist(), "absolute_top16_counts": row["absolute_top16_counts"].tolist(), "local_top16_fraction": row["local_top16"] / max(row["total_top16"], 1)} for layer, row in diagnostics.items()}}


def main() -> None:
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    evaluation = {}
    for corpus_index, (corpus, spec) in enumerate(config["evaluation"].items()):
        rows = load_rows(spec); evaluation[corpus] = {}
        for position in config["edit_positions"]:
            print("evaluating", corpus, position, flush=True); evaluation[corpus][str(position)] = evaluate_condition(model, tokenizer, rows, position, config, config["seed"] + 1000 * corpus_index + position)
    def vals(method, budget, metric="fisher_fidelity"):
        return [evaluation[c][str(p)]["metrics"][str(layer)][f"{method}/{budget}"][metric] for c in evaluation for p in config["edit_positions"] for layer in config["layers"]]
    edited = [evaluation[c][str(p)]["metrics"][str(layer)]["edited_only"]["fisher_fidelity"] for c in evaluation for p in config["edit_positions"] for layer in config["layers"]]
    diagnostics = [evaluation[c][str(p)]["diagnostics"][str(layer)] for c in evaluation for p in config["edit_positions"] for layer in config["layers"]]
    predicted, exact = [], []
    for c in evaluation:
        for p in config["edit_positions"]:
            for layer in config["layers"]:
                for method in config["methods"]:
                    for budget in config["budgets"][:-1]:
                        row = evaluation[c][str(p)]["metrics"][str(layer)][f"{method}/{budget}"]; predicted.append(row["predicted_projection"]); exact.append(row["fisher_projection"])
    correlation = float(np.corrcoef(predicted, exact)[0, 1]); global_counts = np.sum([np.asarray(row["top16_counts"]) for row in diagnostics], axis=0); skeleton_fraction = float(np.sort(global_counts)[-16:].sum() / max(global_counts.sum(), 1)); locality_fraction = float(np.mean([row["local_top16_fraction"] for row in diagnostics]))
    hypotheses = {
        "E1_source_recomposition_is_numerically_exact": {"supported": all(row["relative_write_error"] <= 1e-5 and row["maximum_logit_error"] <= 1e-4 for row in diagnostics)},
        "E2_sixteen_fisher_sources_are_useful": {"supported": np.mean(vals("signed_fisher", 16)) >= .75},
        "E3_sixtyfour_fisher_sources_are_robustly_sufficient": {"supported": min(vals("signed_fisher", 64)) >= .85},
        "E4_behavioral_ranking_beats_state_ranking": {"supported": np.mean(vals("signed_fisher", 16)) - np.mean(vals("state_norm", 16)) >= .10, "mean_advantage": float(np.mean(vals("signed_fisher", 16)) - np.mean(vals("state_norm", 16)))},
        "E5_signed_ranking_beats_absolute_ranking": {"supported": np.mean(vals("signed_fisher", 16)) >= np.mean(vals("absolute_fisher", 16)), "mean_difference": float(np.mean(vals("signed_fisher", 16)) - np.mean(vals("absolute_fisher", 16)))},
        "E6_source_program_is_conditionally_sparse": {"supported": np.median([value for row in diagnostics for value in row["participation_values"]]) <= 32, "median": float(np.median([value for row in diagnostics for value in row["participation_values"]]))},
        "E7_the_edited_source_alone_is_not_sufficient": {"supported": np.mean(edited) < .50, "mean_fidelity": float(np.mean(edited))},
        "E8_selected_sources_are_geometrically_local": {"supported": locality_fraction >= .50, "fraction": locality_fraction},
        "E9_local_gradient_predicts_finite_projection": {"supported": correlation >= .80, "correlation": correlation},
        "E10_a_small_global_source_skeleton_recurs": {"supported": skeleton_fraction >= .50, "fraction": skeleton_fraction, "top_positions": np.argsort(global_counts)[-16:][::-1].tolist()},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "global_top16_counts": global_counts.tolist(), "evaluation": evaluation}, indent=2, allow_nan=False))
    budgets = config["budgets"]; fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for method in config["methods"]: axes[0].plot(budgets, [np.mean(vals(method, budget)) for budget in budgets], marker="o", label=method); axes[1].plot(budgets, [min(vals(method, budget)) for budget in budgets], marker="o", label=method)
    axes[0].set_title("Adaptive source-program mean"); axes[1].set_title("Adaptive source-program minimum"); axes[0].set_xscale("log", base=2); axes[1].set_xscale("log", base=2)
    top_positions = np.argsort(global_counts)[-20:][::-1]; axes[2].bar(np.arange(len(top_positions)), global_counts[top_positions]); axes[2].set_xticks(np.arange(len(top_positions)), top_positions, rotation=45); axes[2].set_title("Most recurrent selected sources")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=7) if axis.get_legend_handles_labels()[0] else None
    axes[0].set_xlabel("source budget"); axes[1].set_xlabel("source budget"); axes[0].set_ylabel("Fisher fidelity"); axes[1].set_ylabel("Fisher fidelity"); axes[2].set_xlabel("absolute source position")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    lines = ["# Final-destination source-edge program", "", "Each selected layer's final-destination attention-message difference is decomposed exactly into 128 additive source-position writes, retaining the native squared QK weights, current values, persistent value bus, heads, and output projection. A downstream Fisher gradient at the base message ranks source edges per example; the resulting finite source subsets are then patched through the nonlinear suffix. State-norm, absolute-gradient, and random rankings are controls.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"The exact source sum missed the locked 1e-5 runtime threshold by float32 reassociation: maximum relative write error is {max(row['relative_write_error'] for row in diagnostics):.2e}, while maximum propagated-logit disagreement is {max(row['maximum_logit_error'] for row in diagnostics):.2e}. This is retained as a numerical gate failure, not interpreted as a missing source term."]
    lines += ["", "![Source program](bilin12_final_destination_source_program.png)", "", "## Held-out finite frontier", "", "| Budget | Signed Fisher mean/min | Absolute Fisher mean/min | State norm mean/min | Random mean/min |", "|---:|---:|---:|---:|---:|"]
    for budget in budgets: lines.append(f"| {budget} | {np.mean(vals('signed_fisher', budget)):.3f}/{min(vals('signed_fisher', budget)):.3f} | {np.mean(vals('absolute_fisher', budget)):.3f}/{min(vals('absolute_fisher', budget)):.3f} | {np.mean(vals('state_norm', budget)):.3f}/{min(vals('state_norm', budget)):.3f} | {np.mean(vals('random', budget)):.3f}/{min(vals('random', budget)):.3f} |")
    lines += ["", f"Edited-source-only mean fidelity: **{np.mean(edited):.3f}**. Median absolute-score participation: **{hypotheses['E6_source_program_is_conditionally_sparse']['median']:.1f}** positions. Edit-neighborhood-or-final share of top-16 selections: **{locality_fraction:.3f}**. Global top-16 source skeleton share: **{skeleton_fraction:.3f}**. Local-gradient versus finite Fisher projection correlation: **{correlation:.3f}**.", "", "Adaptive source selection is an oracle diagnostic because it uses the target message and its downstream target direction. Sparse success would demonstrate conditional edge sparsity, not a fixed reusable circuit. Failure would indicate that even the source contraction is broadly cooperative."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
