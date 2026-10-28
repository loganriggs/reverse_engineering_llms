#!/usr/bin/env python3
"""Factor counterfactual L9H8 boundary transport into QK and value sources."""

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
from run_bilin18_boundary_dialect_counterfactual import (
    make_contrasts,
    select_records,
)
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import leading_mode


CONFIG = Path("configs/bilin18_boundary_style_factorial.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_factorial.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_FACTORIAL.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_factorial.png"
FACTORS = ("qk1", "qk2", "current_value", "persistent_bus")
WORLDS = tuple(itertools.product((0, 1), repeat=len(FACTORS)))


def shapley(outputs: dict[tuple[int, ...], torch.Tensor]) -> dict[str, torch.Tensor]:
    """Exact vector-valued Shapley decomposition between 0000 and 1111."""
    count = len(FACTORS)
    result = {}
    for factor_index, factor in enumerate(FACTORS):
        value = torch.zeros_like(next(iter(outputs.values())))
        others = [index for index in range(count) if index != factor_index]
        for size in range(count):
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            for subset in itertools.combinations(others, size):
                absent = [0] * count
                for index in subset:
                    absent[index] = 1
                present = absent.copy(); present[factor_index] = 1
                value = value + weight * (outputs[tuple(present)] - outputs[tuple(absent)])
        result[factor] = value
    return result


def summarize_vector(target: torch.Tensor, pieces: dict[str, torch.Tensor]) -> dict:
    target = target.double(); energy = float(target.square().sum().clamp_min(1e-30))
    rows = {}
    for name, value in pieces.items():
        value = value.double(); inner = float((value * target).sum()); own = float(value.square().sum())
        rows[name] = {
            "projection_share": inner / energy,
            "energy_ratio": own / energy,
            "cosine": inner / max(math.sqrt(own * energy), 1e-30),
        }
    return rows


def summarize_scalar(target: torch.Tensor, pieces: dict[str, torch.Tensor]) -> dict:
    target = target.double(); energy = float(target.square().sum().clamp_min(1e-30))
    rows = {}
    for name, value in pieces.items():
        value = value.double(); inner = float((value * target).sum()); own = float(value.square().sum())
        rows[name] = {
            "win": float((value > 0).double().mean()),
            "mean": float(value.mean()),
            "projection_share": inner / energy,
            "energy_ratio": own / energy,
            "cosine": inner / max(math.sqrt(own * energy), 1e-30),
        }
    return rows


@torch.inference_mode()
def collect_factorial(model, raw, records, layer, head, source_kind, output_mode,
                      contrasts, config):
    dimension = model.config.n_embd; heads = model.config.n_head
    head_dim = dimension // heads; output_mode = output_mode.to("cuda")
    all_local_target, all_local_parts = [], {name: [] for name in FACTORS}
    all_output_target, all_output_parts = [], {name: [] for name in FACTORS}
    all_leading_target, all_leading_parts = [], {name: [] for name in FACTORS}
    random_total = []
    direct = {name: {"declared": [], "random": []} for name in ("current_value_only", "qk_only")}
    local_error = output_error = 0.0
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]
        original = np.stack([raw[row[0], :128].copy() for row in batch]); swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate([original, swapped]), device="cuda")
        selected_source = 1 if source_kind == "boundary" else 2
        source_one = torch.as_tensor([row[selected_source] for row in batch], device="cuda")
        source = torch.cat([source_one, source_one]); destination = torch.full_like(source, 127)
        rows = torch.arange(len(ids), device="cuda"); batch_size = len(batch)
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks):
            if block_index < layer:
                x, bus = block(x, bus, x0); continue
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus,
                                        model.config, model.tt)
            projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
            factors = {
                "qk1": details["score1"][rows, head, destination, source].float().reshape(2, batch_size),
                "qk2": details["score2"][rows, head, destination, source].float().reshape(2, batch_size),
                "current_value": details["current_value"][rows, source, head].float().reshape(2, batch_size, head_dim),
                "persistent_bus": details["value_bus"][rows, source, head].float().reshape(2, batch_size, head_dim),
            }
            writes = {}
            for world in WORLDS:
                qk1 = factors["qk1"][world[0]]; qk2 = factors["qk2"][world[1]]
                current = factors["current_value"][world[2]]; first = factors["persistent_bus"][world[3]]
                value = (1 - block.attn.lamb.float()) * current + block.attn.lamb.float() * first
                writes[world] = (qk1 * qk2)[:, None] * value @ projection

            local_target = writes[(1, 1, 1, 1)] - writes[(0, 0, 0, 0)]
            local_parts = shapley(writes)
            local_error = max(local_error, float((sum(local_parts.values()) - local_target).norm()
                                                  / local_target.norm().clamp_min(1e-30)))
            leading_worlds = {world: write @ output_mode for world, write in writes.items()}
            leading_parts = shapley(leading_worlds)

            # All worlds share the original global context. Only this one native edge
            # receives counterfactual factors, making the endpoint contrast a causal
            # mediation assay rather than another whole-prefix association test.
            original_attention = details["write"][:batch_size].float()
            original_edge = writes[(0, 0, 0, 0)]
            attention_variants = []
            for world in WORLDS:
                modified = original_attention.clone()
                modified[torch.arange(batch_size, device="cuda"), destination[:batch_size]] += writes[world] - original_edge
                attention_variants.append(modified)
            attention = torch.cat(attention_variants)
            mixed0 = mixed[:batch_size].float().repeat(len(WORLDS), 1, 1)
            x00 = x0[:batch_size].float().repeat(len(WORLDS), 1, 1)
            bus0 = details["value_bus"][:batch_size].float().repeat(len(WORLDS), 1, 1, 1)
            state = F.rms_norm(mixed0 + attention, (dimension,))
            suffix_x = mixed0 + attention + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            suffix_bus = bus0
            for suffix in model.blocks[block_index + 1:]:
                suffix_x, suffix_bus = suffix(suffix_x, suffix_bus, x00)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(suffix_x, (dimension,))[:, 127]).float() / 30)
            scores = (logits @ contrasts).reshape(len(WORLDS), batch_size, -1)
            score_worlds = {world: scores[index] for index, world in enumerate(WORLDS)}
            output_parts = shapley(score_worlds)
            output_target = score_worlds[(1, 1, 1, 1)] - score_worlds[(0, 0, 0, 0)]
            output_error = max(output_error, float((sum(output_parts.values()) - output_target).norm()
                                                    / output_target.norm().clamp_min(1e-30)))

            direction = torch.as_tensor([1 if not row[4] else -1 for row in batch],
                                        device="cuda", dtype=torch.float32)
            all_local_target.append(local_target.cpu())
            all_leading_target.append((direction * (leading_worlds[(1,1,1,1)] - leading_worlds[(0,0,0,0)])).cpu())
            all_output_target.append((direction * output_target[:, 0]).cpu())
            random_total.append((direction[:, None] * output_target[:, 1:]).cpu())
            for direct_name, world in (("current_value_only", (0, 0, 1, 0)),
                                       ("qk_only", (1, 1, 0, 0))):
                change = score_worlds[world] - score_worlds[(0, 0, 0, 0)]
                direct[direct_name]["declared"].append((direction * change[:, 0]).cpu())
                direct[direct_name]["random"].append((direction[:, None] * change[:, 1:]).cpu())
            for name in FACTORS:
                all_local_parts[name].append(local_parts[name].cpu())
                all_leading_parts[name].append((direction * leading_parts[name]).cpu())
                all_output_parts[name].append((direction * output_parts[name][:, 0]).cpu())
            break

    local_target = torch.cat(all_local_target); output_target = torch.cat(all_output_target)
    leading_target = torch.cat(all_leading_target); random_total = torch.cat(random_total)
    local_parts = {name: torch.cat(values) for name, values in all_local_parts.items()}
    output_parts = {name: torch.cat(values) for name, values in all_output_parts.items()}
    leading_parts = {name: torch.cat(values) for name, values in all_leading_parts.items()}
    for pieces in (local_parts, output_parts, leading_parts):
        pieces["qk_combined"] = pieces["qk1"] + pieces["qk2"]
        pieces["value_combined"] = pieces["current_value"] + pieces["persistent_bus"]
    random_means = random_total.mean(0); total_mean = float(output_target.mean())
    direct_summary = {}
    for name, values in direct.items():
        declared = torch.cat(values["declared"]); random = torch.cat(values["random"]).mean(0)
        direct_summary[name] = {
            "win": float((declared > 0).float().mean()),
            "mean": float(declared.mean()),
            "declared_random_percentile": float((random < declared.mean()).float().mean()),
        }
    return {
        "examples": len(output_target),
        "local_recomposition_error": local_error,
        "output_recomposition_error": output_error,
        "total": {
            "win": float((output_target > 0).float().mean()),
            "mean": total_mean,
            "leading_win": float((leading_target > 0).float().mean()),
            "leading_mean": float(leading_target.mean()),
            "declared_random_percentile": float((random_means < total_mean).float().mean()),
            "random_p95_mean": float(torch.quantile(random_means, .95)),
        },
        "local": summarize_vector(local_target, local_parts),
        "leading": summarize_scalar(leading_target, leading_parts),
        "output": summarize_scalar(output_target, output_parts),
        "direct": direct_summary,
    }


def main():
    config = json.loads(CONFIG.read_text()); base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    pairs = [(tokenizer.encode(us, add_special_tokens=False)[0], tokenizer.encode(uk, add_special_tokens=False)[0])
             for us, uk in base["spelling_pairs"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    programs = [("L9H8_boundary", 9, 8, "boundary"),
                ("L9H8_nonboundary", 9, 8, "control"),
                ("L9H7_boundary", 9, 7, "boundary")]
    modes = {}
    for _, layer, head, _ in programs:
        key = (layer, head)
        if key not in modes:
            block = model.blocks[layer].attn; head_dim = model.config.n_embd // model.config.n_head
            sl = slice(head * head_dim, (head + 1) * head_dim)
            modes[key] = leading_mode(block.c_proj.weight[:, sl].detach().float().cpu(),
                                      block.c_v.weight[sl].detach().float().cpu())[0]
    results = []
    for fold in range(config["folds"]):
        target_indices = set(range(fold * config["target_pairs_per_fold"], (fold + 1) * config["target_pairs_per_fold"]))
        target_pairs = [pair for index, pair in enumerate(pairs) if index in target_indices]
        cues = [pair for index, pair in enumerate(pairs) if index not in target_indices]
        contrasts = make_contrasts(model.lm_head.weight.shape[0], target_pairs,
                                   config["random_contrasts"], config["seed"] + fold, "cuda")
        for split, (corpus_start, corpus_end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[corpus_start:corpus_end, :128], dtype=np.int64)
            records = select_records(raw, boundary, cues, counts, 127,
                                     config["examples_per_label_per_split"], config["seed"] + fold * 10 + split)
            for name, layer, head, source_kind in programs:
                print("factorial", fold, split, name, len(records), flush=True)
                result = collect_factorial(model, raw, records, layer, head, source_kind,
                                           modes[(layer, head)], contrasts, config)
                results.append({"fold": fold, "split": split, "program": name, **result})

    selected = [row for row in results if row["program"] == "L9H8_boundary"]
    controls = [row for row in results if row["program"] == "L9H8_nonboundary"]
    heads = [row for row in results if row["program"] == "L9H7_boundary"]
    lookup = lambda rows, fold, split: next(row for row in rows if row["fold"] == fold and row["split"] == split)
    f0 = max(max(row["local_recomposition_error"], row["output_recomposition_error"]) for row in results) <= 1e-5
    f1 = all(row["total"]["win"] >= .60 for row in selected)
    f2_cells = sum(row["output"]["current_value"]["mean"] > row["output"]["qk_combined"]["mean"] for row in selected)
    f2 = f2_cells >= 5 and all(row["local"]["current_value"]["projection_share"] > 0 for row in selected)
    f3 = all(abs(row[metric]["persistent_bus"][field]) <= 1e-6
             for row in selected for metric, field in (("local", "energy_ratio"), ("output", "mean")))
    ratios = [row["total"]["mean"] / max(abs(lookup(controls, row["fold"], row["split"])["total"]["mean"]), 1e-30)
              for row in selected]
    f4 = float(np.median(ratios)) >= 1.5
    f5_cells = sum(row["total"]["win"] >= lookup(heads, row["fold"], row["split"])["total"]["win"] + .05
                   for row in selected); f5 = f5_cells >= 5
    f6_cells = sum(row["total"]["declared_random_percentile"] >= .95 for row in selected); f6 = f6_cells >= 5
    hypotheses = {
        "F0_factorial_recomposes": {"supported": f0},
        "F1_boundary_edge_hybrid_transfers_style": {"supported": f1},
        "F2_contextual_value_dominates_routing": {"supported": f2, "cells": f2_cells},
        "F3_persistent_bus_is_negative_control": {"supported": f3},
        "F4_factorized_transport_is_boundary_specific": {"supported": f4, "median_ratio": float(np.median(ratios))},
        "F5_factorized_transport_is_head_specific": {"supported": f5, "cells": f5_cells},
        "F6_declared_contrast_beats_random": {"supported": f6, "cells": f6_cells},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# QK/value factorial of boundary style transport", "",
             "A US/UK cue is swapped, but every finite endpoint world keeps the original global context. Only one Layer-9 attention edge receives original or counterfactual QK1, QK2, current-value, and persistent-value-bus factors. Exact vector-valued Shapley decomposition therefore asks which native tensor factors make that edge transport the disjoint spelling contrast.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Fold/split | Program | N | total win / mean | current-V mean/share | QK mean/share | direct V/QK win | bus mean | random percentile |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['fold']}/{row['split']} | {row['program']} | {row['examples']} | {row['total']['win']:.3f} / {row['total']['mean']:+.4g} | {row['output']['current_value']['mean']:+.4g} / {row['output']['current_value']['projection_share']:.3f} | {row['output']['qk_combined']['mean']:+.4g} / {row['output']['qk_combined']['projection_share']:.3f} | {row['direct']['current_value_only']['win']:.3f}/{row['direct']['qk_only']['win']:.3f} | {row['output']['persistent_bus']['mean']:+.2g} | {row['total']['declared_random_percentile']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "This factors one invocation of one native edge. Current value means the contextual Layer-9 value at the source boundary; the persistent bus is the Layer-0 value at that same unchanged boundary token. QK factors describe routing, not payload. A dominant current-value term supports an accumulated contextual payload, but does not by itself identify which earlier layers created it. Finite Shapley values are exact for the declared output contrast, not universal percentages of head function."]
    REPORT.write_text("\n".join(lines) + "\n")
    labels = [f"f{row['fold']}s{row['split']}" for row in selected]; x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].bar(x - .2, [row["output"]["current_value"]["mean"] for row in selected], .4, label="current value")
    axes[0].bar(x + .2, [row["output"]["qk_combined"]["mean"] for row in selected], .4, label="QK combined")
    axes[0].axhline(0, color="black", linewidth=.8); axes[0].set(xticks=x, xticklabels=labels, ylabel="Directed spelling-contrast shift", title="Which edge factor carries style?"); axes[0].legend()
    for index, rows in enumerate((selected, controls, heads)):
        axes[1].bar(x + (index - 1) * .25, [row["total"]["win"] for row in rows], .24, label=rows[0]["program"])
    axes[1].axhline(.5, color="black", linewidth=.8); axes[1].set(xticks=x, xticklabels=labels, ylabel="Correct shift fraction", title="Specificity controls"); axes[1].legend(fontsize=8)
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "selected": selected}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
