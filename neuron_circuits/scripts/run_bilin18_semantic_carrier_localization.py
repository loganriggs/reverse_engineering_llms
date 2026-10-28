#!/usr/bin/env python3
"""Localize transferable variables across residual, embedding, and value-bus carriers."""

from __future__ import annotations

import json
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


CONFIG = Path("configs/bilin18_semantic_carrier_localization.json")
OUTPUT = TENSOR_ROOT / "bilin18_semantic_carrier_localization.json"
REPORT = TENSOR_ROOT / "BILIN18_SEMANTIC_CARRIER_LOCALIZATION.md"
FIGURE = TENSOR_ROOT / "bilin18_semantic_carrier_localization.png"

PROGRAMS = (
    "base", "native", "residual_full", "residual_cue", "residual_boundary",
    "residual_destination", "residual_post_boundary", "embedding", "value_bus",
    "embedding_value_bus", "residual_embedding", "residual_value_bus",
)


@torch.inference_mode()
def trace_model(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    outputs = []
    for block in model.blocks:
        x, value_bus = block(x, value_bus, x0)
        outputs.append(x)
    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, 127]).float() / 30)
    return x0, value_bus, outputs, logits


@torch.inference_mode()
def suffix(model, layer, states, x0, value_bus):
    dimension = model.config.n_embd
    x = states
    for block in model.blocks[layer + 1:]:
        x, value_bus = block(x, value_bus, x0)
    return 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[:, 127]).float() / 30)


def hybrid_states(base, changed, cue, boundary, destination=127):
    size = len(base)
    result = {}
    result["base"] = base
    result["native"] = changed
    result["residual_full"] = changed
    for name in ("residual_cue", "residual_boundary", "residual_destination", "residual_post_boundary"):
        result[name] = base.clone()
    rows = torch.arange(size, device=base.device)
    result["residual_cue"][rows, cue] = changed[rows, cue]
    result["residual_boundary"][rows, boundary] = changed[rows, boundary]
    result["residual_destination"][:, destination] = changed[:, destination]
    positions = torch.arange(base.shape[1], device=base.device)[None]
    mask = positions >= boundary[:, None]
    result["residual_post_boundary"][mask] = changed[mask]
    for name in ("embedding", "value_bus", "embedding_value_bus"):
        result[name] = base
    result["residual_embedding"] = changed
    result["residual_value_bus"] = changed
    return result


@torch.inference_mode()
def collect_cell(model, raw, records, contrast, layers, batch_size):
    storage = {layer: {name: [] for name in PROGRAMS if name != "base"} for layer in layers}
    targets = {layer: [] for layer in layers}
    maximum_error = 0.0
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        changed = original.copy()
        for index, row in enumerate(batch):
            changed[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate((original, changed)), device="cuda")
        x0, bus, outputs, native_logits = trace_model(model, ids)
        size = len(batch)
        x0_base, x0_changed = x0[:size], x0[size:]
        bus_base, bus_changed = bus[:size], bus[size:]
        cue = torch.as_tensor([row[3] for row in batch], device="cuda")
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        native_base_score = native_logits[:size] @ contrast
        native_changed_score = native_logits[size:] @ contrast
        native_target = direction * (native_changed_score - native_base_score)
        for layer in layers:
            base_state, changed_state = outputs[layer][:size], outputs[layer][size:]
            states = hybrid_states(base_state, changed_state, cue, boundary)
            state_batches, x0_batches, bus_batches = [], [], []
            for name in PROGRAMS:
                state_batches.append(states[name])
                x0_batches.append(x0_changed if name in ("native", "embedding", "embedding_value_bus", "residual_embedding") else x0_base)
                bus_batches.append(bus_changed if name in ("native", "value_bus", "embedding_value_bus", "residual_value_bus") else bus_base)
            logits = suffix(model, layer, torch.cat(state_batches), torch.cat(x0_batches), torch.cat(bus_batches))
            logits = logits.reshape(len(PROGRAMS), size, -1)
            scores = torch.einsum("pbv,v->pb", logits, contrast)
            base_score = scores[0]
            target = direction * (scores[1] - base_score)
            targets[layer].append(target.cpu())
            maximum_error = max(maximum_error, float((logits[0] - native_logits[:size]).abs().max()),
                                float((logits[1] - native_logits[size:]).abs().max()),
                                float((target - native_target).abs().max()))
            for program_index, name in enumerate(PROGRAMS[1:], 1):
                storage[layer][name].append((direction * (scores[program_index] - base_score)).cpu())
        print(f"carrier batch {min(start + size, len(records))}/{len(records)}", flush=True)
    result = {}
    for layer in layers:
        target = torch.cat(targets[layer]).double()
        energy = target.square().sum().clamp_min(1e-30)
        result[str(layer)] = {"examples": len(target), "full_mean": float(target.mean()), "full_win": float((target > 0).double().mean()), "programs": {}}
        for name, pieces in storage[layer].items():
            value = torch.cat(pieces).double()
            result[str(layer)]["programs"][name] = {
                "mean": float(value.mean()),
                "expected_direction_fraction": float((value > 0).double().mean()),
                "full_direction_agreement": float((value * target > 0).double().mean()),
                "projection_share": float((value * target).sum() / energy),
                "energy_fidelity": float(1 - (target - value).square().sum() / energy),
                "pearson": float(np.corrcoef(target.numpy(), value.numpy())[0, 1]) if float(value.std()) > 0 else 0.0,
            }
    return result, maximum_error


def summarize_cells(results, families, layers):
    summary = {}
    for family in families:
        summary[family] = {}
        rows = [row for row in results if row["family"] == family]
        for layer in layers:
            key = str(layer)
            summary[family][key] = {
                "examples": int(sum(row["layers"][key]["examples"] for row in rows)),
                "full_win": float(np.mean([row["layers"][key]["full_win"] for row in rows])),
                "programs": {},
            }
            for program in PROGRAMS[1:]:
                summary[family][key]["programs"][program] = {
                    metric: float(np.mean([row["layers"][key]["programs"][program][metric] for row in rows]))
                    for metric in ("projection_share", "energy_fidelity", "expected_direction_fraction", "full_direction_agreement", "pearson")
                }
    return summary


def earliest(summary, family, layers, program, threshold):
    return next((layer for layer in layers if summary[family][str(layer)]["programs"][program]["projection_share"] >= threshold), None)


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    family_source = json.loads(Path(config["family_config"]).read_text())
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
    results = []; maximum_error = 0.0
    for family_index, (family, pairs) in enumerate(family_pairs.items()):
        midpoint = len(pairs) // 2
        folds = ((pairs[:midpoint], pairs[midpoint:]), (pairs[midpoint:], pairs[:midpoint]))
        for fold, (targets, cues) in enumerate(folds):
            contrast = make_contrasts(model.lm_head.weight.shape[0], targets, 0,
                                      config["seed"] + 100 * family_index + fold, "cuda")[:, 0]
            for split, (begin, end) in enumerate(base["splits"]):
                raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
                records = select_records(raw, boundary_flags, cues, counts, 127,
                                         config["examples_per_label_per_split"],
                                         config["seed"] + 1000 * family_index + 10 * fold + split)
                print(f"carrier cell {family} fold={fold} split={split} n={len(records)}", flush=True)
                layers, error = collect_cell(model, raw, records, contrast, config["layers"], config["batch_size"])
                maximum_error = max(maximum_error, error)
                results.append({"family": family, "fold": fold, "split": split, "layers": layers})
    summary = summarize_cells(results, config["families"], config["layers"])
    layer8 = "8"; layer12 = "12"
    other_destination = [summary[family][layer12]["programs"]["residual_destination"]["projection_share"] for family in ("sentiment", "gender", "tense")]
    hypotheses = {
        "L0_hybrid_recomposition_is_exact": {"supported": maximum_error <= 1e-4, "maximum_absolute_error": maximum_error},
        "L1_spelling_reaches_boundary_state": {"supported": summary["spelling"][layer8]["programs"]["residual_boundary"]["projection_share"] >= .50, "observed": summary["spelling"][layer8]["programs"]["residual_boundary"]["projection_share"]},
        "L2_other_variables_reach_destination_state": {"supported": sum(value >= .50 for value in other_destination) >= 2, "observed": dict(zip(("sentiment", "gender", "tense"), other_destination))},
        "L3_full_residual_internalizes_transfer": {"supported": all(summary[family][layer12]["programs"]["residual_full"]["projection_share"] >= .80 for family in config["families"]), "observed": {family: summary[family][layer12]["programs"]["residual_full"]["projection_share"] for family in config["families"]}},
        "L4_external_carriers_decline": {"supported": all(max(summary[family][layer12]["programs"]["embedding"]["projection_share"], summary[family][layer12]["programs"]["value_bus"]["projection_share"]) <= .20 for family in config["families"]), "observed": {family: {name: summary[family][layer12]["programs"][name]["projection_share"] for name in ("embedding", "value_bus")} for family in config["families"]}},
        "L5_spelling_boundary_path_is_exceptional": {"supported": summary["spelling"][layer8]["programs"]["residual_boundary"]["projection_share"] >= max(summary[family][layer8]["programs"]["residual_boundary"]["projection_share"] for family in ("sentiment", "gender", "tense")) + .20, "observed": {family: summary[family][layer8]["programs"]["residual_boundary"]["projection_share"] for family in config["families"]}},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    earliest_layers = {family: {program: earliest(summary, family, config["layers"], program, .50) for program in ("residual_cue", "residual_boundary", "residual_destination", "residual_post_boundary", "residual_full")} for family in config["families"]}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "maximum_absolute_error": maximum_error, "earliest_projection_half": earliest_layers, "summary": summary, "results": results}, indent=2, allow_nan=False))
    lines = ["# Depth and spatial localization of transferable semantic carriers", "",
             "Natural cue swaps are replayed from ten layer boundaries. At each boundary, the counterfactual residual is transplanted either at the cue, latest boundary, destination, post-boundary field, or full sequence. Repeated embedding and persistent Layer-0 value-bus changes are independently retained or removed. The output contrast uses disjoint lexical pairs from the same family.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Carrier localization](bilin18_semantic_carrier_localization.png)", "", "## Earliest layer reaching 0.50 projection", "", "| Family | Cue | Boundary | Destination | Post-boundary | Full residual |", "|---|---:|---:|---:|---:|---:|"]
    for family in config["families"]:
        row = earliest_layers[family]
        lines.append("| " + family + " | " + " | ".join("—" if row[name] is None else str(row[name]) for name in ("residual_cue", "residual_boundary", "residual_destination", "residual_post_boundary", "residual_full")) + " |")
    lines += ["", "## Projection-share curves", ""]
    for family in config["families"]:
        lines += [f"### {family}", "", "| Layer | Cue | Boundary | Destination | Post-boundary | Full residual | Embedding | Value bus |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for layer in config["layers"]:
            p = summary[family][str(layer)]["programs"]
            lines.append(f"| {layer} | {p['residual_cue']['projection_share']:.3f} | {p['residual_boundary']['projection_share']:.3f} | {p['residual_destination']['projection_share']:.3f} | {p['residual_post_boundary']['projection_share']:.3f} | {p['residual_full']['projection_share']:.3f} | {p['embedding']['projection_share']:.3f} | {p['value_bus']['projection_share']:.3f} |")
        lines.append("")
    lines += ["## Guardrails", "", "A residual transplant is a sufficiency test under the original suffix, not a declaration that the patched positions exclusively store the variable. Projection shares are not additive because the spatial programs overlap and interact nonlinearly with embedding reinjection and the value bus. A cue-only success can be literal lexical transport; boundary/destination success is stronger evidence that the edit has been contextualized, but semantic naming still rests on disjoint-pair behavioral transfer."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for axis, family in zip(axes.flat, config["families"]):
        for program in ("residual_cue", "residual_boundary", "residual_destination", "residual_post_boundary", "residual_full"):
            axis.plot(config["layers"], [summary[family][str(layer)]["programs"][program]["projection_share"] for layer in config["layers"]], marker="o", label=program.replace("residual_", ""))
        axis.axhline(.5, color="black", linestyle="--", linewidth=.8); axis.set(title=family, xlabel="Layer boundary", ylabel="Projection onto full semantic effect"); axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "earliest_projection_half": earliest_layers}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
