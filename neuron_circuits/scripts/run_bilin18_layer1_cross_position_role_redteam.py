#!/usr/bin/env python3
"""Red-team same-position query dominance with genuinely cross-position effects."""

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
from run_bilin12_layer01_frozen_taylor_behavior import differentiable_suffix
from run_bilin18_layer1_output_aware_coalitions import clear_rotary
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_suffix_qk_role_pilot import ROLES, attention_role_vjps
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_cross_position_role_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_cross_position_role_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_CROSS_POSITION_ROLE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_cross_position_role_redteam.png"


def native_logits(model, trace, position):
    state = F.rms_norm(trace[-1]["output"], (model.config.n_embd,))[:, position]
    return (30 * torch.tanh(model.lm_head(state).float() / 30))[0]


def cross_gradient(model, trace, layer, edit_position, prediction_position, contrast):
    decoder = model.blocks[layer].mlp.Down.weight.float(); x0 = trace[layer]["x0"].detach().clone(); bus = trace[layer]["value_bus"].detach().clone(); output = trace[layer]["output"].detach().clone(); clear_rotary(model, layer + 1)
    with torch.enable_grad():
        variable = torch.zeros(decoder.shape[1], device="cuda", requires_grad=True); mask = torch.zeros_like(output); mask[:, edit_position] = F.linear(variable, decoder).to(output.dtype)
        logits = differentiable_suffix(
            model, layer, x0, bus, output + mask, prediction_position,
        )[0]
        score = logits.float() @ contrast; gradient, = torch.autograd.grad(score, variable)
    return gradient.detach().float()


def cross_role_program(model, trace, edit_position, prediction_position, contrast, specifications):
    names = list(specifications); final = final_adjoint(model, trace[-1]["output"], prediction_position, contrast); current = final.repeat(len(names), 1, 1, 1); maximum_local_error = 0.0
    for layer in range(len(model.blocks) - 1, 1, -1):
        block = model.blocks[layer]; updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current)
        enabled = [row for row, name in enumerate(names) if layer in specifications[name]["layers"] and specifications[name]["roles"]]
        if enabled:
            inputs = updated[enabled]; previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float(); mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0; bus = trace[layer - 1]["value_bus"]
            pieces = attention_role_vjps(block, mixed, bus, inputs); native = attention_vjp(block, mixed, bus, inputs); recomposed = sum(pieces.values())
            maximum_local_error = max(maximum_local_error, float((recomposed - native).norm() / native.norm().clamp_min(1e-30)))
            for local, row in enumerate(enabled):
                updated[row] += sum((pieces[role][local] for role in specifications[names[row]]["roles"]), torch.zeros_like(updated[row]))
        current = block.lambdas[0].float() * updated
    down = model.blocks[1].mlp.Down.weight.float()
    return {name: current[row, 0, edit_position] @ down for row, name in enumerate(names)}, maximum_local_error


def specifications():
    all_layers = set(range(2, 18)); early = set(range(2, 10)); late = set(range(10, 18))
    return {
        "mlp_only": {"layers": set(), "roles": set()},
        "queries": {"layers": all_layers, "roles": {"q1", "q2"}},
        "keys": {"layers": all_layers, "roles": {"k1", "k2"}},
        "value": {"layers": all_layers, "roles": {"value"}},
        "query_key": {"layers": all_layers, "roles": {"q1", "q2", "k1", "k2"}},
        "query_value": {"layers": all_layers, "roles": {"q1", "q2", "value"}},
        "key_value": {"layers": all_layers, "roles": {"k1", "k2", "value"}},
        "early_full": {"layers": early, "roles": set(ROLES)},
        "late_full": {"layers": late, "roles": set(ROLES)},
        "all_full": {"layers": all_layers, "roles": set(ROLES)},
    }


def cross_logits_for_deltas(model, base, deltas, edit_position, prediction_position):
    names = list(deltas); decoder = model.blocks[1].mlp.Down.weight.float(); outputs = []; x0s = []; buses = []
    for delta in deltas.values():
        output = base["output"].clone(); output[:, edit_position] += F.linear(delta, decoder).to(output.dtype); outputs.append(output); x0s.append(base["x0"]); buses.append(base["value_bus"])
    logits = suffix_from_layer(model, 1, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), prediction_position)
    return {name: logits[index] for index, name in enumerate(names)}


def extended_metrics(prediction, exact, delta, rank):
    result = metrics(prediction, exact, delta, rank); result["gradient_norm_ratio"] = float(prediction.norm() / exact.norm().clamp_min(1e-30)); return result


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r"); rows = corpus[config["start_row"]:config["start_row"] + config["rows"]]; specs = specifications(); k = config["support_rank"]; records = []; maximum_local_error = 0.0
    for edit_position in config["edit_positions"]:
        used = 0
        for row_index, raw_row in enumerate(rows):
            raw = np.asarray(raw_row[:config["sequence_length"]], dtype=np.int64).copy()
            if raw[edit_position] == newline: continue
            changed = raw.copy(); changed[edit_position] = newline
            base_ids = torch.as_tensor(raw, device="cuda")[None]; changed_ids = torch.as_tensor(changed, device="cuda")[None]
            with torch.no_grad(): base_trace = forward_layers_native(model, base_ids, 17); changed_trace = forward_layers_native(model, changed_ids, 17); base_logits = native_logits(model, base_trace, config["prediction_position"]); changed_logits = native_logits(model, changed_trace, config["prediction_position"])
            contrast = (F.softmax(changed_logits, -1) - F.softmax(base_logits, -1)).detach()
            delta = product_at(model, changed_trace, 1, edit_position) - product_at(model, base_trace, 1, edit_position)
            exact = cross_gradient(model, base_trace, 1, edit_position, config["prediction_position"], contrast)
            predictions, error = cross_role_program(model, base_trace, edit_position, config["prediction_position"], contrast, specs); maximum_local_error = max(maximum_local_error, error); predictions["exact"] = exact
            supports = {name: (prediction * delta).topk(k).indices for name, prediction in predictions.items()}; patch_deltas = {"base": torch.zeros_like(delta), "full_product": delta, **{name: selected_delta(delta, support) for name, support in supports.items()}}
            with torch.no_grad(): logits = cross_logits_for_deltas(model, base_trace[1], patch_deltas, edit_position, config["prediction_position"])
            base_score = float(logits["base"].float() @ contrast); finite = {name: float(value.float() @ contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({"distance": config["prediction_position"] - edit_position, "edit_position": edit_position, "row": row_index, "source_token": int(raw[edit_position]), "source_text": tokenizer.decode([int(raw[edit_position])]).replace("\n", "\\n"), "native_behavior_score": float((changed_logits - base_logits) @ contrast), "metrics": {name: extended_metrics(prediction, exact, delta, k) for name, prediction in predictions.items()}, "finite_shifts": finite})
            used += 1
        print("cross-position", edit_position, "distance", config["prediction_position"] - edit_position, "sites", used, flush=True)
    methods = list(records[0]["metrics"]); distances = sorted({row["distance"] for row in records}); summaries = {}
    for distance in distances:
        subset = [row for row in records if row["distance"] == distance]; summaries[str(distance)] = {}
        for method in methods:
            values = [row["metrics"][method] for row in subset]; shifts = np.asarray([row["finite_shifts"][method] for row in subset]); summaries[str(distance)][method] = {key + "_median": float(np.median([value[key] for value in values])) for key in ("gradient_cosine", "gradient_relative_error", "gradient_norm_ratio", "contribution_cosine", "true_leverage_ratio")} | {"finite_directional_fraction": float(np.mean(shifts > 0)), "finite_mean_shift": float(shifts.mean())}
        full_shifts = np.asarray([row["finite_shifts"]["full_product"] for row in subset]); summaries[str(distance)]["full_product"] = {"finite_directional_fraction": float(np.mean(full_shifts > 0)), "finite_mean_shift": float(full_shifts.mean())}
    cell = lambda distance, method: summaries[str(distance)][method]
    hypotheses = {
        "X0_role_program_is_exact": {"supported": maximum_local_error <= 1e-5 and all(row["metrics"]["all_full"]["gradient_cosine"] >= .999 and row["metrics"]["all_full"]["gradient_relative_error"] <= .02 for row in records)},
        "X1_nontransport_programs_are_structurally_zero": {"supported": all(max(row["metrics"][method]["gradient_norm_ratio"] for row in records) <= 1e-5 for method in ("mlp_only", "queries"))},
        "X2_key_or_value_transport_is_material": {"supported": all(max(cell(distance, "keys")["gradient_cosine_median"], cell(distance, "value")["gradient_cosine_median"]) >= .50 for distance in distances)},
        "X3_queries_refine_transported_signals": {"supported": all(max(cell(distance, "query_key")["gradient_cosine_median"], cell(distance, "query_value")["gradient_cosine_median"]) - max(cell(distance, "keys")["gradient_cosine_median"], cell(distance, "value")["gradient_cosine_median"]) >= .05 for distance in distances)},
        "X4_no_single_transport_family_is_universally_sufficient": {"supported": not (all(cell(distance, "keys")["gradient_cosine_median"] >= .90 for distance in distances) or all(cell(distance, "value")["gradient_cosine_median"] >= .90 for distance in distances))},
        "X5_early_eight_blocks_are_not_always_sufficient": {"supported": any(cell(distance, "early_full")["gradient_cosine_median"] < .90 for distance in distances)},
        "X6_exact_coalitions_have_finite_direction": {"supported": all(cell(distance, "exact")["finite_directional_fraction"] >= .75 for distance in distances)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    display = ["mlp_only", "queries", "keys", "value", "query_key", "query_value", "key_value", "early_full", "late_full", "all_full"]
    lines = ["# Cross-position red team of suffix role programs", "", "A natural FineWeb token at an earlier position is replaced by newline, while behavior is scored at Position 127 along the full model's own probability-change direction. The exact Layer-1 product gradient at the edit position is decomposed through query, key, and value VJPs. This changes the causal geometry: some key or value derivative must transport sensitivity backward to the edited position.", "", f"Maximum local role recomposition error: `{maximum_local_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Cross-position roles](bilin18_layer1_cross_position_role_redteam.png)", ""]
    for distance in distances:
        lines += [f"### Distance {distance}", "", "| Program | Gradient cosine | Norm / exact | Contribution cosine | Finite direction |", "|---|---:|---:|---:|---:|"]
        for method in display:
            value = cell(distance, method); lines.append(f"| {method} | {value['gradient_cosine_median']:.3f} | {value['gradient_norm_ratio_median']:.3e} | {value['contribution_cosine_median']:.3f} | {value['finite_directional_fraction']:.3f} |")
        lines.append("")
    lines += ["## Guardrail", "", "The output direction is defined from each example's full token-substitution effect, so this assay explains a known behavioral change rather than prospectively predicting which change will occur. Its purpose is to distinguish structural transport roles, not to claim a reusable semantic feature."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained"); plot_methods = ["queries", "keys", "value", "query_key", "query_value", "early_full", "late_full"]
    width = .8 / len(distances); x = np.arange(len(plot_methods))
    for index, distance in enumerate(distances): axes[0].bar(x + (index - 1) * width, [cell(distance, method)["gradient_cosine_median"] for method in plot_methods], width=width, label=f"distance {distance}")
    axes[0].set_xticks(x, plot_methods, rotation=28, ha="right"); axes[0].set(ylabel="Median gradient cosine", title="Cross-position role sufficiency"); axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    for method in ("queries", "keys", "value", "query_key", "query_value"):
        axes[1].plot(distances, [cell(distance, method)["gradient_cosine_median"] for distance in distances], marker="o", label=method)
    axes[1].set(xlabel="Source-to-output distance", ylabel="Median gradient cosine", title="Transport changes the role hierarchy"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig); print(json.dumps({"maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
