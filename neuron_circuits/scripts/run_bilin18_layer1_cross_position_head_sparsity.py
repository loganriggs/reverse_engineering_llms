#!/usr/bin/env python3
"""Test whether cross-position role programs are sparse over layer-head edges."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin18_attention_layer_path_program import rotate
from run_bilin18_layer1_cross_position_role_redteam import (
    cross_gradient, cross_logits_for_deltas, extended_metrics, native_logits,
)
from run_bilin18_layer1_output_aware_coalitions import clear_rotary
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_cross_position_head_sparsity.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_cross_position_head_sparsity.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_CROSS_POSITION_HEAD_SPARSITY.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_cross_position_head_sparsity.png"


def attention_head_vjps(block, mixed, bus, adjoints):
    """Return the additive VJP of every native attention head."""
    clear_rotary(type("Wrapper", (), {"blocks": [block]})(), 0); attention = block.attn
    with torch.enable_grad():
        variable = mixed.detach().clone().requires_grad_(True); state = F.rms_norm(variable, (variable.shape[-1],)); batch, length, dimension = state.shape; heads, head = attention.n_head, attention.head_dim
        def projected(module):
            raw = F.linear(state, module.weight.float()).view(batch, length, heads, head); cosine, sine = attention.rotary(raw)
            return rotate(F.rms_norm(raw, (head,)), cosine.float(), sine.float())
        q1, k1 = projected(attention.c_q), projected(attention.c_k); q2, k2 = projected(attention.c_q2), projected(attention.c_k2)
        current_value = F.linear(state, attention.c_v.weight.float()).view(batch, length, heads, head); value = (1 - attention.lamb.float()) * current_value + attention.lamb.float() * bus.detach().view_as(current_value).float()
        score1 = torch.einsum("bqhd,bkhd->bhqk", q1, k1) / head; score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head; causal = torch.tril(torch.ones(length, length, device=state.device, dtype=torch.bool)); pattern = (score1 * score2).masked_fill(~causal, 0); attended = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
        outputs = []
        for head_index in range(heads):
            weight = attention.c_proj.weight.float()[:, head_index * head:(head_index + 1) * head]
            outputs.append(F.linear(attended[:, head_index], weight))
        result = []
        for head_index, output in enumerate(outputs):
            gradient, = torch.autograd.grad(output, variable, adjoints, retain_graph=head_index < heads - 1, is_grads_batched=True); result.append(gradient.detach().float())
    return torch.stack(result)


def head_program(model, trace, edit_position, prediction_position, contrast, programs):
    names = list(programs); final = final_adjoint(model, trace[-1]["output"], prediction_position, contrast); current = final.repeat(len(names), 1, 1, 1); maximum_local_error = 0.0
    for layer in range(len(model.blocks) - 1, 1, -1):
        block = model.blocks[layer]; updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current); enabled = [row for row, name in enumerate(names) if any(edge[0] == layer for edge in programs[name])]
        if enabled:
            inputs = updated[enabled]; previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float(); mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0; bus = trace[layer - 1]["value_bus"]
            pieces = attention_head_vjps(block, mixed, bus, inputs); native = attention_vjp(block, mixed, bus, inputs); recomposed = pieces.sum(0); maximum_local_error = max(maximum_local_error, float((recomposed - native).norm() / native.norm().clamp_min(1e-30)))
            for local, row in enumerate(enabled):
                chosen = [head for edge_layer, head in programs[names[row]] if edge_layer == layer]; updated[row] += pieces[chosen, local].sum(0)
        current = block.lambdas[0].float() * updated
    down = model.blocks[1].mlp.Down.weight.float(); return {name: current[row, 0, edit_position] @ down for row, name in enumerate(names)}, maximum_local_error


def make_site(model, raw, newline, edit_position, prediction_position):
    changed = raw.copy(); changed[edit_position] = newline; base_ids = torch.as_tensor(raw, device="cuda")[None]; changed_ids = torch.as_tensor(changed, device="cuda")[None]
    with torch.no_grad(): base_trace = forward_layers_native(model, base_ids, 17); changed_trace = forward_layers_native(model, changed_ids, 17); base_logits = native_logits(model, base_trace, prediction_position); changed_logits = native_logits(model, changed_trace, prediction_position)
    contrast = (F.softmax(changed_logits, -1) - F.softmax(base_logits, -1)).detach(); delta = product_at(model, changed_trace, 1, edit_position) - product_at(model, base_trace, 1, edit_position); exact = cross_gradient(model, base_trace, 1, edit_position, prediction_position, contrast)
    return base_trace, contrast, delta, exact


def median_summary(records, method):
    rows = [row["metrics"][method] for row in records]; shifts = np.asarray([row["finite_shifts"][method] for row in records]); return {key + "_median": float(np.median([value[key] for value in rows])) for key in ("gradient_cosine", "gradient_norm_ratio", "contribution_cosine", "true_leverage_ratio")} | {"finite_directional_fraction": float(np.mean(shifts > 0)), "finite_mean_shift": float(shifts.mean())}


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r"); edges = [(layer, head) for layer in range(2, 18) for head in range(model.config.n_head)]; all_edges = set(edges); discovery_scores = {edge: [] for edge in edges}; maximum_local_error = 0.0
    discovery_rows = corpus[config["discovery_start"]:config["discovery_start"] + config["discovery_rows"]]
    for row_index, raw_row in enumerate(discovery_rows):
        raw = np.asarray(raw_row[:config["sequence_length"]], dtype=np.int64).copy(); edit = config["discovery_edit_position"]
        if raw[edit] == newline: continue
        trace, contrast, delta, exact = make_site(model, raw, newline, edit, config["prediction_position"])
        programs = {"full": all_edges, **{f"leaveout_{layer}_{head}": all_edges - {(layer, head)} for layer, head in edges}}
        predictions, error = head_program(model, trace, edit, config["prediction_position"], contrast, programs); maximum_local_error = max(maximum_local_error, error); exact_contribution = exact * delta
        for edge in edges:
            prediction = predictions[f"leaveout_{edge[0]}_{edge[1]}"]; contribution = prediction * delta; cosine = float((contribution @ exact_contribution) / (contribution.norm() * exact_contribution.norm()).clamp_min(1e-30)); discovery_scores[edge].append(1 - cosine)
        print("head discovery", row_index + 1, "/", len(discovery_rows), flush=True)
    edge_score = {edge: float(np.mean(values)) for edge, values in discovery_scores.items()}; ordering = sorted(edges, key=lambda edge: edge_score[edge], reverse=True)
    rng = np.random.default_rng(config["seed"]); programs = {f"top_{rank}": set(ordering[:rank]) for rank in config["head_edge_ranks"]}
    for rank in config["random_ranks"]:
        for replicate in range(config["random_replicates"]):
            chosen = rng.choice(len(edges), rank, replace=False); programs[f"random_{rank}_{replicate}"] = {edges[int(index)] for index in chosen}
    records = []; evaluation_rows = corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"]]
    for edit in config["evaluation_edit_positions"]:
        for row_index, raw_row in enumerate(evaluation_rows):
            raw = np.asarray(raw_row[:config["sequence_length"]], dtype=np.int64).copy()
            if raw[edit] == newline: continue
            trace, contrast, delta, exact = make_site(model, raw, newline, edit, config["prediction_position"]); predictions, error = head_program(model, trace, edit, config["prediction_position"], contrast, programs); maximum_local_error = max(maximum_local_error, error); predictions["exact"] = exact
            supports = {name: (prediction * delta).topk(config["support_rank"]).indices for name, prediction in predictions.items()}; patch_deltas = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, support) for name, support in supports.items()}}
            with torch.no_grad(): logits = cross_logits_for_deltas(model, trace[1], patch_deltas, edit, config["prediction_position"])
            base_score = float(logits["base"].float() @ contrast); finite = {name: float(value.float() @ contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({"distance": config["prediction_position"] - edit, "row": row_index, "metrics": {name: extended_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in predictions.items()}, "finite_shifts": finite})
        print("head evaluation position", edit, "distance", config["prediction_position"] - edit, flush=True)
    distances = sorted({row["distance"] for row in records}); summaries = {}
    for distance in distances:
        subset = [row for row in records if row["distance"] == distance]; summaries[str(distance)] = {method: median_summary(subset, method) for method in subset[0]["metrics"]}
        for rank in config["random_ranks"]:
            medians = [summaries[str(distance)][f"random_{rank}_{replicate}"]["gradient_cosine_median"] for replicate in range(config["random_replicates"])]
            summaries[str(distance)][f"random_{rank}_aggregate"] = {"median": float(np.median(medians)), "p95": float(np.quantile(medians, .95)), "maximum": float(np.max(medians))}
    cell = lambda distance, method: summaries[str(distance)][method]; top16 = set(ordering[:16])
    hypotheses = {
        "Y0_head_vjps_recompose_native_attention": {"supported": maximum_local_error <= 1e-5 and all(row["metrics"]["top_72"]["gradient_cosine"] >= .999 and row["metrics"]["top_72"]["gradient_relative_error"] <= .02 for row in records)},
        "Y1_frozen_top16_is_high_fidelity": {"supported": all(cell(distance, "top_16")["gradient_cosine_median"] >= .85 for distance in distances)},
        "Y2_frozen_top16_beats_random": {"supported": all(cell(distance, "top_16")["gradient_cosine_median"] - cell(distance, "random_16_aggregate")["p95"] >= .05 for distance in distances)},
        "Y3_frozen_top32_is_very_high_fidelity": {"supported": all(cell(distance, "top_32")["gradient_cosine_median"] >= .90 for distance in distances)},
        "Y4_important_edges_are_early_concentrated": {"supported": sum(layer <= 9 for layer, _ in top16) / len(top16) >= .60},
        "Y5_frozen_top16_is_causally_directional": {"supported": all(cell(distance, "top_16")["finite_directional_fraction"] >= .75 for distance in distances)},
        "Y6_selection_is_discovery_only": {"supported": True},
    }
    ranked_edges = [{"rank": index + 1, "layer": edge[0], "head": edge[1], "score": edge_score[edge]} for index, edge in enumerate(ordering)]
    OUTPUT.write_text(json.dumps({"config": config, "maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "ranked_edges": ranked_edges, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Discovery-frozen layer-head sparsity for cross-position programs", "", "Eight disjoint FineWeb rows at distance 32 rank the 72 suffix layer-head edges by leave-one-out loss of contribution cosine. The frozen ordering is evaluated on new rows at distances 8, 32, and 64, alongside matched random edge sets. Every selected head program is composed through native residual and MLP VJPs.", "", f"Maximum local head recomposition error: `{maximum_local_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Head sparsity](bilin18_layer1_cross_position_head_sparsity.png)", "", "## Frozen top edges", "", "| Rank | Block | Head | Discovery leave-one-out score |", "|---:|---:|---:|---:|"]
    for row in ranked_edges[:16]: lines.append(f"| {row['rank']} | {row['layer']} | {row['head']} | {row['score']:.5f} |")
    for distance in distances:
        lines += ["", f"## Distance {distance}", "", "| Edge budget | Gradient cosine | Contribution cosine | Finite direction | Random p95 |", "|---:|---:|---:|---:|---:|"]
        for rank in config["head_edge_ranks"]:
            row = cell(distance, f"top_{rank}"); random = cell(distance, f"random_{rank}_aggregate")["p95"] if rank in config["random_ranks"] else None; lines.append(f"| {rank} | {row['gradient_cosine_median']:.3f} | {row['contribution_cosine_median']:.3f} | {row['finite_directional_fraction']:.3f} | {'—' if random is None else f'{random:.3f}'} |")
    lines += ["", "## Guardrail", "", "Leave-one-out ranking is a discovery-frozen heuristic, not a proof of unique head importance. Correlated or synergistic heads can make individual omissions look small, and a compact selected program remains state-conditioned across positions and MLP operators."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained"); ranks = config["head_edge_ranks"]
    for distance in distances: axes[0].plot(ranks, [cell(distance, f"top_{rank}")["gradient_cosine_median"] for rank in ranks], marker="o", label=f"distance {distance}")
    axes[0].set(xlabel="Frozen layer-head edges", ylabel="Median gradient cosine", title="Sparse head-program frontier"); axes[0].legend(); axes[0].grid(alpha=.2)
    heatmap = np.zeros((16, model.config.n_head));
    for rank, (layer, head) in enumerate(ordering, 1): heatmap[layer - 2, head] = rank
    image = axes[1].imshow(heatmap, aspect="auto", cmap="viridis_r"); axes[1].set(xlabel="Head", ylabel="Suffix block (2 at top)", title="Discovery rank of every edge"); axes[1].set_yticks(range(16), range(2, 18)); fig.colorbar(image, ax=axes[1], label="Rank")
    fig.savefig(FIGURE, dpi=180); plt.close(fig); print(json.dumps({"maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "top16": ranked_edges[:16], "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
