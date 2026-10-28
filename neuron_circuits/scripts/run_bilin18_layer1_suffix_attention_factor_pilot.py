#!/usr/bin/env python3
"""Split early suffix attention sensitivity into QK1, QK2, and value VJPs."""

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
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_attention_layer_path_program import rotate
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import clear_rotary, semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_suffix_layer_localization_pilot import summarize
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_attention_factor_pilot.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_attention_factor_pilot.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_ATTENTION_FACTOR_PILOT.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_attention_factor_pilot.png"
FACTORS = ("qk1", "qk2", "value")


def attention_factor_vjps(block, mixed, bus, adjoints):
    """Exact additive local VJPs through QK1, QK2, and current-value branches."""
    clear_rotary(type("Wrapper", (), {"blocks": [block]})(), 0)
    attention = block.attn
    with torch.enable_grad():
        variable = mixed.detach().clone().requires_grad_(True)
        state = F.rms_norm(variable, (variable.shape[-1],))
        batch, length, dimension = state.shape
        heads, head = attention.n_head, attention.head_dim

        def projected(module):
            raw = F.linear(state, module.weight.float()).view(batch, length, heads, head)
            cosine, sine = attention.rotary(raw)
            return rotate(F.rms_norm(raw, (head,)), cosine.float(), sine.float())

        q1, k1 = projected(attention.c_q), projected(attention.c_k)
        q2, k2 = projected(attention.c_q2), projected(attention.c_k2)
        current_value = F.linear(state, attention.c_v.weight.float()).view(batch, length, heads, head)
        value = ((1 - attention.lamb.float()) * current_value
                 + attention.lamb.float() * bus.detach().view_as(current_value).float())
        score1 = torch.einsum("bqhd,bkhd->bhqk", q1, k1) / head
        score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head
        causal = torch.tril(torch.ones(length, length, device=state.device, dtype=torch.bool))

        def write(first, second, values):
            pattern = (first * second).masked_fill(~causal, 0)
            attended = torch.einsum("bhqk,bkhd->bhqd", pattern, values)
            return F.linear(
                attended.transpose(1, 2).contiguous().view(batch, length, dimension),
                attention.c_proj.weight.float(),
            )

        outputs = {
            "qk1": write(score1, score2.detach(), value.detach()),
            "qk2": write(score1.detach(), score2, value.detach()),
            "value": write(score1.detach(), score2.detach(), value),
        }
        result = {}
        for index, name in enumerate(FACTORS):
            gradient, = torch.autograd.grad(
                outputs[name], variable, adjoints,
                retain_graph=index < len(FACTORS) - 1, is_grads_batched=True,
            )
            result[name] = gradient.detach().float()
    return result


def factor_program(model, trace, position, contrast, specifications):
    names = list(specifications)
    final = final_adjoint(model, trace[-1]["output"], position, contrast)
    current = final.repeat(len(names), 1, 1, 1)
    maximum_local_error = 0.0
    for layer in range(len(model.blocks) - 1, 1, -1):
        block = model.blocks[layer]
        mlp_rows = mlp_vjp(block, trace[layer]["pre"].float(), current)
        updated = current + mlp_rows
        enabled = [row for row, name in enumerate(names) if layer in specifications[name]["layers"] and specifications[name]["factors"]]
        if enabled:
            inputs = updated[enabled]
            previous = trace[layer - 1]["output"].float(); x0 = trace[layer]["x0"].float()
            mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
            bus = trace[layer - 1]["value_bus"]
            pieces = attention_factor_vjps(block, mixed, bus, inputs)
            native = attention_vjp(block, mixed, bus, inputs)
            recomposed = sum(pieces.values())
            maximum_local_error = max(
                maximum_local_error,
                float((recomposed - native).norm() / native.norm().clamp_min(1e-30)),
            )
            for local, row in enumerate(enabled):
                factors = specifications[names[row]]["factors"]
                updated[row] += sum((pieces[factor][local] for factor in factors), torch.zeros_like(updated[row]))
        current = block.lambdas[0].float() * updated
    down = model.blocks[1].mlp.Down.weight.float()
    return {
        name: current[row, 0, position] @ down for row, name in enumerate(names)
    }, maximum_local_error


def make_specifications(early_layers):
    early = set(early_layers); all_layers = set(range(2, 18))
    return {
        "mlp_only": {"layers": set(), "factors": set()},
        "early_qk1": {"layers": early, "factors": {"qk1"}},
        "early_qk2": {"layers": early, "factors": {"qk2"}},
        "early_qk": {"layers": early, "factors": {"qk1", "qk2"}},
        "early_value": {"layers": early, "factors": {"value"}},
        "early_qk1_value": {"layers": early, "factors": {"qk1", "value"}},
        "early_qk2_value": {"layers": early, "factors": {"qk2", "value"}},
        "early_full": {"layers": early, "factors": set(FACTORS)},
        "all_full": {"layers": all_layers, "factors": set(FACTORS)},
    }


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrast = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]
    selected = []
    for feature in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature]
        selected.extend((next(row for row in rows if row["selection"] == "extreme"), next(row for row in rows if row["selection"] == "uniform")))
    specifications = make_specifications(config["attention_layers"]); k = config["support_rank"]; records = []; maximum_local_error = 0.0
    for site, row in enumerate(selected):
        feature = FEATURES.index(row["feature"]); vector = contrast[:, feature]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64)
        recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, 1); recipient_trace = forward_layers_native(model, recipient_ids, 17)
        delta = product_at(model, donor_trace, 1, row["donor_position"]) - product_at(model, recipient_trace, 1, row["recipient_position"])
        exact = semantic_gradient(model, recipient_trace, 1, row["recipient_position"], vector)
        predictions, local_error = factor_program(model, recipient_trace, row["recipient_position"], vector, specifications)
        maximum_local_error = max(maximum_local_error, local_error); predictions["exact"] = exact
        supports = {name: (prediction * delta).topk(k).indices for name, prediction in predictions.items()}
        deltas = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, support) for name, support in supports.items()}}
        with torch.no_grad(): logits = logits_for_deltas(model, 1, recipient_trace[1], deltas, row["recipient_position"])
        base = class_score(logits["base"], vector); finite = {name: class_score(value, vector) - base for name, value in logits.items() if name != "base"}
        records.append({"feature": row["feature"], "selection": row["selection"], "metrics": {name: metrics(prediction, exact, delta, k) for name, prediction in predictions.items()}, "finite_shifts": finite})
        print("attention-factor site", site + 1, "/", len(selected), row["feature"], row["selection"], flush=True)
    methods = list(records[0]["metrics"]); summaries = {method: summarize(records, method) for method in methods}
    qk = summaries["early_qk"]; value = summaries["early_value"]; full = summaries["early_full"]
    hypotheses = {
        "M0_factor_vjps_recompose_native_attention": {"supported": maximum_local_error <= 1e-5 and all(row["metrics"]["all_full"]["gradient_cosine"] >= .999 and row["metrics"]["all_full"]["gradient_relative_error"] <= .02 for row in records)},
        "M1_full_early_factor_program_replicates": {"supported": full["gradient_cosine_median"] >= .90},
        "M2_qk_routing_alone_is_sufficient": {"supported": qk["gradient_cosine_median"] >= .90},
        "M3_value_transport_alone_is_sufficient": {"supported": value["gradient_cosine_median"] >= .90},
        "M4_routing_value_complementarity_is_material": {"supported": full["gradient_cosine_median"] - max(qk["gradient_cosine_median"], value["gradient_cosine_median"]) >= .05},
        "M5_two_qk_factors_are_complementary": {"supported": qk["gradient_cosine_median"] - max(summaries["early_qk1"]["gradient_cosine_median"], summaries["early_qk2"]["gradient_cosine_median"]) >= .05},
        "M6_one_qk_factor_plus_value_is_sufficient": {"supported": max(summaries["early_qk1_value"]["gradient_cosine_median"], summaries["early_qk2_value"]["gradient_cosine_median"]) >= .90},
    }
    OUTPUT.write_text(json.dumps({"config": config, "maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    display = ["mlp_only", "early_qk1", "early_qk2", "early_qk", "early_value", "early_qk1_value", "early_qk2_value", "early_full", "all_full"]
    exact_shift = summaries["exact"]["finite_mean_shift"]
    lines = ["# Early suffix attention-factor pilot", "", "For attention responses in Blocks 2–9, the exact local VJP is split into first-QK, second-QK, and current-value/OV derivatives. Each program composes its selected derivative families through the native MLP and residual VJPs, then selects and finitely patches its own top-64 Layer-1 channels.", "", f"Maximum local QK1+QK2+value recomposition error: `{maximum_local_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Attention factor pilot](bilin18_layer1_suffix_attention_factor_pilot.png)", "", "| Enabled early attention derivative | Gradient cosine / Q10 | Contribution cosine | Finite direction | Shift / exact |", "|---|---:|---:|---:|---:|"]
    for method in display:
        value = summaries[method]; lines.append(f"| {method} | {value['gradient_cosine_median']:.3f}/{value['gradient_cosine_q10']:.3f} | {value['contribution_cosine_median']:.3f} | {value['finite_directional_fraction']:.3f} | {value['finite_mean_shift'] / exact_shift:.3f} |")
    lines += ["", "## Guardrail", "", "These are derivatives at the native forward state. QK is routing sensitivity and value is transported-content sensitivity; neither is a claim that semantic information is stored exclusively in those weights."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained"); names = display[:-1]
    axes[0].bar(range(len(names)), [summaries[name]["gradient_cosine_median"] for name in names]); axes[0].set_xticks(range(len(names)), names, rotation=30, ha="right"); axes[0].set(ylabel="Median gradient cosine", title="QK and OV explanatory sufficiency"); axes[0].grid(axis="y", alpha=.2)
    axes[1].bar(range(len(names)), [summaries[name]["finite_mean_shift"] / exact_shift for name in names]); axes[1].set_xticks(range(len(names)), names, rotation=30, ha="right"); axes[1].axhline(1, color="gray", linestyle=":"); axes[1].set(ylabel="Mean finite shift / exact", title="Causal coalition strength"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"maximum_local_recomposition_error": maximum_local_error, "hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
