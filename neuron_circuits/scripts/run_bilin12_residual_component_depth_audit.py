#!/usr/bin/env python3
"""Open the residual carrier into exact per-block additive components."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_carrier_depth_audit import exact_kl, fisher_terms
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_residual_component_depth_audit.json")
OUTPUT = TENSOR_ROOT / "bilin12_residual_component_depth_audit.json"
REPORT = TENSOR_ROOT / "BILIN12_RESIDUAL_COMPONENT_DEPTH_AUDIT.md"
FIGURE = TENSOR_ROOT / "bilin12_residual_component_depth_audit.png"
COMPONENTS = ("carried_state", "current_embedding_injection", "attention_write", "mlp_write")


def vector_shapley(worlds):
    result = {}
    count = len(COMPONENTS)
    for index, component in enumerate(COMPONENTS):
        bit = 1 << index
        effect = torch.zeros_like(worlds[0])
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            effect += weight * (worlds[mask | bit] - worlds[mask])
        result[component] = effect
    return result


@torch.inference_mode()
def trace_components(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    rows = []
    for block in model.blocks:
        previous = x
        bus_before = value_bus
        carried = block.lambdas[0] * previous
        injected = block.lambdas[1] * x0
        mixed = carried + injected
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        state = F.rms_norm(pre, (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + mlp
        rows.append({
            "previous": previous,
            "x0": x0,
            "bus_before": bus_before,
            "value_bus": value_bus,
            "carried_state": carried,
            "current_embedding_injection": injected,
            "attention_write": attention,
            "mlp_write": mlp,
            "output": x,
        })
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return rows, 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def component_worlds(model, layer, base, changed):
    batch = base["output"].shape[0]
    deltas = {component: changed[component] - base[component] for component in COMPONENTS}
    states = []
    for mask in range(16):
        value = base["output"]
        for index, component in enumerate(COMPONENTS):
            if mask & (1 << index):
                value = value + deltas[component]
        states.append(value)
    x = torch.cat(states)
    x0 = base["x0"].repeat(16, 1, 1)
    value_bus = base["value_bus"].repeat(16, 1, 1, 1)
    for block in model.blocks[layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    dimension = model.config.n_embd
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    worlds = (30 * torch.tanh(raw / 30)).reshape(16, batch, -1)
    independent_x = changed["output"]
    independent_bus = base["value_bus"]
    for block in model.blocks[layer + 1:]:
        independent_x, independent_bus = block(independent_x, independent_bus, base["x0"])
    independent_raw = model.lm_head(F.rms_norm(independent_x, (dimension,))[:, -1]).float()
    independent = 30 * torch.tanh(independent_raw / 30)
    return worlds, deltas, independent


def empty_accumulator():
    return {
        "examples": 0,
        "residual_energy": 0.0,
        "full_energy": 0.0,
        "shapley_residual_cross": {component: 0.0 for component in COMPONENTS},
        "shapley_full_cross": {component: 0.0 for component in COMPONENTS},
        "state_denominator": 0.0,
        "state_projection": {component: 0.0 for component in COMPONENTS},
        "state_error": 0.0,
        "logit_error": 0.0,
        "residual_kl": 0.0,
    }


def update(acc, worlds, deltas, independent_residual, native_base, native_changed, base_output, changed_output):
    base, residual_changed = worlds[0], worlds[15]
    residual_target = residual_changed - base
    full_target = native_changed - native_base
    residual_energy, _, _, _ = fisher_terms(base, residual_target, residual_target)
    full_energy, _, _, _ = fisher_terms(base, full_target, full_target)
    acc["examples"] += len(base)
    acc["residual_energy"] += float(residual_energy.sum())
    acc["full_energy"] += float(full_energy.sum())
    effects = vector_shapley(worlds)
    for component, effect in effects.items():
        _, _, residual_cross, _ = fisher_terms(base, residual_target, effect)
        _, _, full_cross, _ = fisher_terms(base, full_target, effect)
        acc["shapley_residual_cross"][component] += float(residual_cross.sum())
        acc["shapley_full_cross"][component] += float(full_cross.sum())
    output_delta = changed_output - base_output
    acc["state_denominator"] += float(output_delta.square().sum())
    for component, delta in deltas.items():
        acc["state_projection"][component] += float((delta * output_delta).sum())
    acc["state_error"] = max(
        acc["state_error"],
        float((sum(deltas.values()) - output_delta).abs().max()),
    )
    acc["logit_error"] = max(acc["logit_error"], float((residual_changed - independent_residual).abs().max()))
    acc["residual_kl"] += float(exact_kl(base, residual_changed).sum())


def finalize(acc):
    residual_energy = max(acc["residual_energy"], 1e-30)
    full_energy = max(acc["full_energy"], 1e-30)
    state_denominator = max(acc["state_denominator"], 1e-30)
    within = {component: acc["shapley_residual_cross"][component] / residual_energy for component in COMPONENTS}
    full = {component: acc["shapley_full_cross"][component] / full_energy for component in COMPONENTS}
    state = {component: acc["state_projection"][component] / state_denominator for component in COMPONENTS}
    return {
        "examples": acc["examples"],
        "within_residual_shapley": within,
        "full_target_projection": full,
        "state_projection": state,
        "within_residual_sum": sum(within.values()),
        "full_projection_sum": sum(full.values()),
        "state_projection_sum": sum(state.values()),
        "maximum_state_recomposition_error": acc["state_error"],
        "maximum_logit_recomposition_error": acc["logit_error"],
        "mean_residual_route_kl": acc["residual_kl"] / max(acc["examples"], 1),
    }


@torch.inference_mode()
def evaluate(model, tokenizer, spec, config):
    rows = load_rows(spec)
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    accumulators = [empty_accumulator() for _ in model.blocks]
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        base_ids = torch.as_tensor(raw, device="cuda")
        keep = base_ids[:, config["position"]] != newline
        if not bool(keep.any()): continue
        base_ids = base_ids[keep]
        changed_ids = base_ids.clone(); changed_ids[:, config["position"]] = newline
        base_rows, native_base = trace_components(model, base_ids)
        changed_rows, native_changed = trace_components(model, changed_ids)
        for layer in range(len(model.blocks)):
            worlds, deltas, independent = component_worlds(model, layer, base_rows[layer], changed_rows[layer])
            update(
                accumulators[layer], worlds, deltas, independent, native_base, native_changed,
                base_rows[layer]["output"], changed_rows[layer]["output"],
            )
    return {str(layer): finalize(acc) for layer, acc in enumerate(accumulators)}


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluating", corpus, flush=True)
        results[corpus] = evaluate(model, tokenizer, spec, config)
    rows = [results[c][str(layer)] for c in results for layer in range(12)]
    largest_noncarried = {
        corpus: [
            max(COMPONENTS[1:], key=lambda name: abs(results[corpus][str(layer)]["within_residual_shapley"][name]))
            for layer in range(12)
        ] for corpus in results
    }
    hypotheses = {
        "D1_exact_component_recomposition": {"supported": all(
            max(row["maximum_state_recomposition_error"], row["maximum_logit_recomposition_error"]) <= 1e-4
            for row in rows
        )},
        "D2_some_attention_write_is_behaviorally_material": {"supported": all(
            max(abs(results[c][str(layer)]["within_residual_shapley"]["attention_write"]) for layer in range(12)) >= .10
            for c in results
        )},
        "D3_some_mlp_write_is_behaviorally_material": {"supported": all(
            max(abs(results[c][str(layer)]["within_residual_shapley"]["mlp_write"]) for layer in range(12)) >= .10
            for c in results
        )},
        "D4_current_embedding_injection_is_locally_material": {"supported": all(
            max(abs(results[c][str(layer)]["within_residual_shapley"]["current_embedding_injection"]) for layer in range(11)) >= .10
            for c in results
        )},
        "D5_late_residual_is_not_just_carried_state": {"supported": all(
            abs(results[c]["10"]["within_residual_shapley"]["attention_write"])
            + abs(results[c]["10"]["within_residual_shapley"]["mlp_write"]) >= .10
            for c in results
        )},
        "D6_component_depth_order_transfers_ood": {"supported": all(
            largest_noncarried["fineweb"] == largest_noncarried[corpus] for corpus in ("wikitext", "tinystories")
        ), "largest_noncarried": largest_noncarried},
        "D7_state_energy_is_not_behavioral_credit": {"supported": any(
            abs(row["state_projection"][component] - row["within_residual_shapley"][component]) >= .15
            for row in rows for component in COMPONENTS
        )},
        "D8_attention_is_not_the_dominant_noncarried_source": {"supported": all(
            sum(
                results[c][str(layer)]["within_residual_shapley"]["mlp_write"]
                > results[c][str(layer)]["within_residual_shapley"]["attention_write"]
                for layer in range(12)
            ) >= 8 for c in results
        )},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))

    layers = np.arange(12); colors = dict(zip(COMPONENTS, ("#4c78a8", "#f58518", "#54a24b", "#e45756")))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    for component in COMPONENTS:
        values = np.asarray([[results[c][str(layer)]["within_residual_shapley"][component] for layer in layers] for c in results])
        axes[0].plot(layers, values.mean(0), marker="o", color=colors[component], label=component)
        axes[0].fill_between(layers, values.min(0), values.max(0), color=colors[component], alpha=.12)
        states = np.asarray([[results[c][str(layer)]["state_projection"][component] for layer in layers] for c in results])
        axes[1].plot(layers, states.mean(0), marker="o", color=colors[component], label=component)
        full = np.asarray([[results[c][str(layer)]["full_target_projection"][component] for layer in layers] for c in results])
        axes[2].plot(layers, full.mean(0), marker="o", color=colors[component], label=component)
    axes[0].set_title("Behavior within residual route"); axes[1].set_title("Euclidean state projection"); axes[2].set_title("Projection onto full token effect")
    for axis in axes:
        axis.axhline(0, color="black", lw=.8); axis.set_xlabel("layer"); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = [
        "# Residual-carrier component audit across depth",
        "",
        "For a final-token-to-newline edit, each block output change is exactly split into four additive vectors: carried prior residual, the block's fresh original-embedding injection, attention write, and bilinear-MLP write. All 16 combinations are propagated through the suffix with the base embedding stream and base value bus. Vector Shapley therefore allocates the changed-residual-only behavioral effect, while a separate projection shows alignment with the complete token-edit effect.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Residual components](bilin12_residual_component_depth_audit.png)",
        "",
        "| Corpus | Layer | Carried / inject / attention / MLP behavioral Shapley | State projection | Full-target sum |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus in results:
        for layer in range(12):
            row = results[corpus][str(layer)]
            behavioral = " / ".join(f"{row['within_residual_shapley'][name]:.3f}" for name in COMPONENTS)
            state = " / ".join(f"{row['state_projection'][name]:.3f}" for name in COMPONENTS)
            lines.append(f"| {corpus} | {layer} | {behavioral} | {state} | {row['full_projection_sum']:.3f} |")
    lines += [
        "",
        "These are two distinct ontologies. Euclidean state projection asks which additive vector resembles the block's residual change. Behavioral Shapley asks which vector matters after the nonlinear suffix under a particular hybrid game. A component can be large in state space and behaviorally suppressed, or small and downstream-amplified.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
