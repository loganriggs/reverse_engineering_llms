#!/usr/bin/env python3
"""Factor the construction of transferable semantic variables at the destination."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_semantic_destination_construction.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_destination_construction.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_DESTINATION_CONSTRUCTION.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_destination_construction.png"
FACTORS = ("carried", "attention", "mlp")
WORLDS = tuple(itertools.product((0, 1), repeat=len(FACTORS)))


def shapley(scores):
    count = len(FACTORS)
    result = {}
    for factor_index, name in enumerate(FACTORS):
        value = torch.zeros_like(next(iter(scores.values())))
        others = [index for index in range(count) if index != factor_index]
        for size in range(count):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset in itertools.combinations(others, size):
                absent = [0] * count
                for index in subset:
                    absent[index] = 1
                present = absent.copy(); present[factor_index] = 1
                value += weight * (scores[tuple(present)] - scores[tuple(absent)])
        result[name] = value
    return result


@torch.inference_mode()
def trace_model(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    layers = []
    for block in model.blocks:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        x = pre + mlp
        layers.append({"mixed": mixed, "attention": attention, "mlp": mlp, "output": x, "bus": value_bus})
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, 127]).float() / 30)
    return x0, layers, logits


@torch.inference_mode()
def suffix(model, layer, state, x0, value_bus):
    dimension = model.config.n_embd
    x = state
    for block in model.blocks[layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    return 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, 127]).float() / 30)


def projection(target, value):
    target, value = target.double(), value.double()
    energy = target.square().sum().clamp_min(1e-30)
    return {
        "mean": float(value.mean()),
        "projection_share": float((value * target).sum() / energy),
        "energy_fidelity": float(1 - (target - value).square().sum() / energy),
        "direction_agreement": float((value * target > 0).double().mean()),
        "pearson": float(np.corrcoef(target.numpy(), value.numpy())[0, 1]) if float(value.std()) > 0 else 0.0,
    }


@torch.inference_mode()
def collect_cell(model, raw, records, contrast, layers, batch_size):
    targets = {layer: [] for layer in layers}
    endpoints = {layer: [] for layer in layers}
    shapley_values = {layer: {name: [] for name in FACTORS} for layer in layers}
    singleton_values = {layer: {name: [] for name in FACTORS} for layer in layers}
    max_state_error = max_shapley_error = max_base_error = 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]; size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        changed = original.copy()
        for index, row in enumerate(batch):
            changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda")
        x0, traces, native_logits = trace_model(model, ids)
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        native_target = direction * ((native_logits[size:] @ contrast) - (native_logits[:size] @ contrast))
        for layer in layers:
            trace = traces[layer]
            base_state, changed_state = trace["output"][:size], trace["output"][size:]
            components = {
                "carried": trace["mixed"][size:, 127] - trace["mixed"][:size, 127],
                "attention": trace["attention"][size:, 127] - trace["attention"][:size, 127],
                "mlp": trace["mlp"][size:, 127] - trace["mlp"][:size, 127],
            }
            reconstructed = base_state[:, 127] + sum(components.values())
            relative_state_error = (reconstructed - changed_state[:, 127]).norm() / (changed_state[:, 127] - base_state[:, 127]).norm().clamp_min(1e-30)
            max_state_error = max(max_state_error, float(relative_state_error))
            states = []
            for world in WORLDS:
                state = base_state.clone()
                state[:, 127] = base_state[:, 127] + sum(world[index] * components[name] for index, name in enumerate(FACTORS))
                states.append(state)
            world_logits = suffix(
                model,
                layer,
                torch.cat(states),
                torch.cat([x0[:size]] * len(states)),
                torch.cat([trace["bus"][:size]] * len(states)),
            ).reshape(len(WORLDS), size, -1)
            max_base_error = max(max_base_error, float((world_logits[0] - native_logits[:size]).abs().max()))
            scores = {world: world_logits[index] @ contrast for index, world in enumerate(WORLDS)}
            base_score = scores[(0, 0, 0)]
            endpoint = direction * (scores[(1, 1, 1)] - base_score)
            allocated = shapley(scores)
            max_shapley_error = max(max_shapley_error, float((sum(allocated.values()) - (scores[(1, 1, 1)] - base_score)).norm() / (scores[(1, 1, 1)] - base_score).norm().clamp_min(1e-30)))
            targets[layer].append(native_target.cpu()); endpoints[layer].append(endpoint.cpu())
            for factor_index, name in enumerate(FACTORS):
                shapley_values[layer][name].append((direction * allocated[name]).cpu())
                singleton = [0, 0, 0]; singleton[factor_index] = 1
                singleton_values[layer][name].append((direction * (scores[tuple(singleton)] - base_score)).cpu())
        print(f"construction batch {min(start + size, len(records))}/{len(records)}", flush=True)
    result = {}
    for layer in layers:
        target = torch.cat(targets[layer]); endpoint = torch.cat(endpoints[layer])
        row = {"examples": len(target), "destination": projection(target, endpoint), "shapley": {}, "singletons": {}}
        for name in FACTORS:
            row["shapley"][name] = projection(target, torch.cat(shapley_values[layer][name]))
            row["singletons"][name] = projection(target, torch.cat(singleton_values[layer][name]))
        result[str(layer)] = row
    return result, {"state": max_state_error, "shapley": max_shapley_error, "base": max_base_error}


def summarize(results, families, layers):
    summary = {}
    for family in families:
        rows = [row for row in results if row["family"] == family]
        summary[family] = {}
        for layer in layers:
            key = str(layer)
            summary[family][key] = {
                "destination": {metric: float(np.mean([row["layers"][key]["destination"][metric] for row in rows])) for metric in ("projection_share", "energy_fidelity", "direction_agreement", "pearson")},
                "shapley": {}, "singletons": {},
            }
            for name in FACTORS:
                summary[family][key]["shapley"][name] = {metric: float(np.mean([row["layers"][key]["shapley"][name][metric] for row in rows])) for metric in ("projection_share", "energy_fidelity", "direction_agreement", "pearson")}
                summary[family][key]["singletons"][name] = {metric: float(np.mean([row["layers"][key]["singletons"][name][metric] for row in rows])) for metric in ("projection_share", "energy_fidelity", "direction_agreement", "pearson")}
    return summary


def main():
    config = json.loads(CONFIG.read_text())
    carrier_config = json.loads(Path(config["source_config"]).read_text())
    base = json.loads(Path(carrier_config["source_config"]).read_text())
    family_source = json.loads(Path(carrier_config["family_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    family_pairs = {"spelling": token_pairs(tokenizer, base["spelling_pairs"])}
    family_pairs.update({name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()})
    family_pairs = {name: family_pairs[name] for name in config["families"]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results, errors = [], {"state": 0.0, "shapley": 0.0, "base": 0.0}
    for family_index, (family, pairs) in enumerate(family_pairs.items()):
        midpoint = len(pairs) // 2
        folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0, config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
                records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + 1000 * family_index + 10 * fold + split)
                print(f"construction cell {family} fold={fold} split={split} n={len(records)}", flush=True)
                layers, cell_errors = collect_cell(model, raw, records, contrast, config["layers"], config["batch_size"])
                for name in errors: errors[name] = max(errors[name], cell_errors[name])
                results.append({"family": family, "fold": fold, "split": split, "layers": layers})
                torch.cuda.empty_cache()
    summary = summarize(results, config["families"], config["layers"])
    crossings = {}
    for family in config["families"]:
        layer = next((layer for layer in config["layers"] if summary[family][str(layer)]["destination"]["projection_share"] >= .5), None)
        crossings[family] = None if layer is None else {
            "layer": layer,
            "destination": summary[family][str(layer)]["destination"]["projection_share"],
            "shapley": {name: summary[family][str(layer)]["shapley"][name]["projection_share"] for name in FACTORS},
        }
    robust_cells = []
    for family in config["families"]:
        for layer in config["layers"]:
            row = summary[family][str(layer)]
            if row["destination"]["projection_share"] >= .25:
                for name in FACTORS:
                    robust_cells.append(np.sign(row["shapley"][name]["projection_share"]) == np.sign(row["singletons"][name]["projection_share"]))
    sign_agreement = float(np.mean(robust_cells))
    attention_crossings = [row["shapley"]["attention"] for row in crossings.values() if row is not None]
    mlp_crossings = [abs(row["shapley"]["mlp"]) for row in crossings.values() if row is not None]
    terminal_carried = {family: summary[family]["17"]["shapley"]["carried"]["projection_share"] for family in config["families"]}
    hypotheses = {
        "C0_exact_finite_factorial": {"supported": errors["state"] <= 1e-5 and errors["shapley"] <= 1e-5 and errors["base"] <= 1e-4, "observed": errors},
        "C1_fresh_attention_constructs_some_crossing_variables": {"supported": sum(value >= .20 for value in attention_crossings) >= 2, "observed": {family: None if row is None else row["shapley"]["attention"] for family, row in crossings.items()}},
        "C2_mlp_is_usually_a_consolidator_not_primary_source": {"supported": sum(value < .20 for value in mlp_crossings) >= 3, "observed": {family: None if row is None else row["shapley"]["mlp"] for family, row in crossings.items()}},
        "C3_terminal_interface_is_carried": {"supported": all(value >= .60 for value in terminal_carried.values()), "observed": terminal_carried},
        "C4_branch_credit_is_interaction_robust": {"supported": sign_agreement >= .80, "observed": sign_agreement, "cells": len(robust_cells)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "errors": errors, "crossings": crossings, "summary": summary, "results": results}, indent=2, allow_nan=False))
    lines = ["# Construction of destination-aligned semantic variables", "", "At each audited block, the changed destination state is decomposed exactly into carried/reinjected state, fresh attention write, and realized bilinear-MLP write. All eight finite coalitions pass through the untouched nonlinear suffix. Shapley values allocate interactions and are measured against the complete native semantic counterfactual.", "", "## Locked decisions", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name, row in hypotheses.items())
    lines += ["", "![Destination construction](bilin18_semantic_destination_construction.png)", "", "## First audited destination crossing", "", "| Family | Layer | Destination | Carried | Attention | MLP |", "|---|---:|---:|---:|---:|---:|"]
    for family, row in crossings.items():
        if row is None:
            lines.append(f"| {family} | — | — | — | — | — |")
        else:
            lines.append(f"| {family} | {row['layer']} | {row['destination']:.3f} | {row['shapley']['carried']:.3f} | {row['shapley']['attention']:.3f} | {row['shapley']['mlp']:.3f} |")
    lines += ["", "## Layerwise destination projection and branch allocation", ""]
    for family in config["families"]:
        lines += [f"### {family}", "", "| Layer | Destination | Carried | Attention | MLP |", "|---:|---:|---:|---:|---:|"]
        for layer in config["layers"]:
            row = summary[family][str(layer)]
            lines.append(f"| {layer} | {row['destination']['projection_share']:.3f} | {row['shapley']['carried']['projection_share']:.3f} | {row['shapley']['attention']['projection_share']:.3f} | {row['shapley']['mlp']['projection_share']:.3f} |")
        lines.append("")
    lines += ["## Interpretation boundary", "", "The components are additive changes in the realized destination state, not independent mechanisms: the changed MLP write was itself computed from changed carried and attention inputs. Finite Shapley values give an exact symmetric allocation for the declared output contrast, while singleton comparisons expose interaction sensitivity. A large carried share means the current block preserves an already available destination variable; it does not identify the earlier layer or path that created it."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    colors = {"carried": "#4c78a8", "attention": "#f58518", "mlp": "#54a24b"}
    for axis, family in zip(axes.flat, config["families"]):
        axis.plot(config["layers"], [summary[family][str(layer)]["destination"]["projection_share"] for layer in config["layers"]], color="black", marker="o", label="destination total")
        for name in FACTORS:
            axis.plot(config["layers"], [summary[family][str(layer)]["shapley"][name]["projection_share"] for layer in config["layers"]], marker=".", color=colors[name], label=name)
        axis.axhline(.5, color="gray", linestyle="--", linewidth=.8); axis.axhline(0, color="black", linewidth=.6)
        axis.set(title=family, xlabel="Layer", ylabel="Projection onto native effect"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "crossings": crossings}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
