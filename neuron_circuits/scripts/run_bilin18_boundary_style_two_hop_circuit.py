#!/usr/bin/env python3
"""Finite native intervention on the proposed L8H2-cue -> L9H8-boundary route."""

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
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model, reference_logits
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_two_hop_circuit.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_two_hop_circuit.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_TWO_HOP_CIRCUIT.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_two_hop_circuit.png"
UPSTREAM = (("L8H2_cue", 2, "cue"), ("L8H1_cue", 1, "cue"), ("L8H2_noncue", 2, "control"))
DOWNSTREAM = (("L9H8_boundary", 8, "boundary"), ("L9H7_boundary", 7, "boundary"), ("L9H8_nonboundary", 8, "control"))


def matched_cue_controls(tokens, cues, boundaries, counts, boundary_flags):
    controls = []
    positions = np.arange(tokens.shape[1])
    for row, cue, boundary in zip(tokens, cues, boundaries):
        distance = max(int(boundary - cue), 1); low = 2 ** int(np.floor(np.log2(distance))); high = 2 * low
        candidate = positions[(positions < boundary) & (positions != cue)
                              & ((boundary - positions) >= low) & ((boundary - positions) < high)
                              & ~boundary_flags[row]]
        if not len(candidate): candidate = positions[(positions < boundary) & (positions != cue) & ~boundary_flags[row]]
        target = np.log1p(counts[int(row[cue])]); controls.append(int(candidate[np.argmin(np.abs(np.log1p(counts[row[candidate]]) - target))]))
    return controls


def edge_write(details, block, head, destination, source, rows, dimension, heads):
    head_dim = dimension // heads
    pattern = (details["score1"][rows, head, destination, source]
               * details["score2"][rows, head, destination, source]).float()
    value = details["mixed_value"][rows, source, head].float()
    projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
    return pattern[:, None] * value @ projection


def empty_stats(): return {"native": [], "frozen": [], "mediated": [], "random_native": [], "random_mediated": []}


@torch.inference_mode()
def collect(model, raw, records, contrasts, counts, boundary_flags, config):
    dimension = model.config.n_embd; heads = model.config.n_head; destination = 127
    values = {(up[0], down[0]): empty_stats() for up in UPSTREAM for down in DOWNSTREAM}
    native_error = 0.0
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]; batch_size = len(batch); paired = torch.arange(batch_size, device="cuda")
        original = np.stack([raw[row[0], :128].copy() for row in batch]); swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        boundaries_np = np.asarray([row[1] for row in batch]); cues_np = np.asarray([row[3] for row in batch])
        controls_np = np.asarray(matched_cue_controls(original, cues_np, boundaries_np, counts, boundary_flags))
        boundary = torch.as_tensor(boundaries_np, device="cuda"); cue = torch.as_tensor(cues_np, device="cuda"); control = torch.as_tensor(controls_np, device="cuda")
        nonboundary9 = torch.as_tensor([row[2] for row in batch], device="cuda")
        ids = torch.as_tensor(np.concatenate([original, swapped]), device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks[:9]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            pre = mixed + details["write"]; state = F.rms_norm(pre, (dimension,))
            mlp_write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            if layer < 8:
                x = pre + mlp_write; bus = details["value_bus"]; continue

            original_attention = details["write"][:batch_size].float(); original_x0 = x0[:batch_size].float(); original_bus = details["value_bus"][:batch_size].float()
            direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
            for upstream_name, upstream_head, source_kind in UPSTREAM:
                source = cue if source_kind == "cue" else control
                edge0 = edge_write(details, block, upstream_head, boundary, source, paired, dimension, heads)
                edge1 = edge_write(details, block, upstream_head, boundary, source, paired + batch_size, dimension, heads)
                patched_attention = original_attention.clone(); patched_attention[paired, boundary] += edge1 - edge0
                attention8 = torch.cat([original_attention, patched_attention])
                mixed8 = mixed[:batch_size].float().repeat(2, 1, 1); x08 = original_x0.repeat(2, 1, 1); bus8 = original_bus.repeat(2, 1, 1, 1)
                state8 = F.rms_norm(mixed8 + attention8, (dimension,))
                x8 = mixed8 + attention8 + block.mlp.Down(block.mlp.Left(state8) * block.mlp.Right(state8)) + block.mlp.Down_bias

                block9 = model.blocks[9]; mixed9 = block9.lambdas[0] * x8 + block9.lambdas[1] * x08
                details9 = attention_details(block9.attn, F.rms_norm(mixed9, (dimension,)), bus8, model.config, model.tt)
                for downstream_name, downstream_head, downstream_source_kind in DOWNSTREAM:
                    source9 = torch.cat([boundary, boundary]) if downstream_source_kind == "boundary" else torch.cat([nonboundary9, nonboundary9])
                    rows9 = torch.arange(2 * batch_size, device="cuda"); destination9 = torch.full_like(source9, destination)
                    downstream_edge = edge_write(details9, block9, downstream_head, destination9, source9, rows9, dimension, heads).reshape(2, batch_size, dimension)
                    native_attention9 = details9["write"].float()
                    frozen_attention9 = native_attention9[batch_size:].clone()
                    frozen_attention9[paired, destination] += downstream_edge[0] - downstream_edge[1]
                    attention9 = torch.cat([native_attention9[:batch_size], native_attention9[batch_size:], frozen_attention9])
                    mixed9v = torch.cat([mixed9[:batch_size], mixed9[batch_size:], mixed9[batch_size:]])
                    x09v = torch.cat([x08[:batch_size], x08[batch_size:], x08[batch_size:]])
                    bus9v = torch.cat([details9["value_bus"][:batch_size], details9["value_bus"][batch_size:], details9["value_bus"][batch_size:]]).float()
                    state9 = F.rms_norm(mixed9v + attention9, (dimension,))
                    suffix_x = mixed9v + attention9 + block9.mlp.Down(block9.mlp.Left(state9) * block9.mlp.Right(state9)) + block9.mlp.Down_bias
                    suffix_bus = bus9v
                    for suffix in model.blocks[10:]: suffix_x, suffix_bus = suffix(suffix_x, suffix_bus, x09v)
                    logits = 30 * torch.tanh(model.lm_head(F.rms_norm(suffix_x, (dimension,))[:, destination]).float() / 30)
                    logits = logits.reshape(3, batch_size, -1)
                    if upstream_name == "L8H2_cue" and downstream_name == "L9H8_boundary":
                        reference = torch.stack([reference_logits(model, torch.as_tensor(row[None], device="cuda")) for row in original])
                        native_error = max(native_error, float((logits[0] - reference).abs().max()))
                    scores = logits @ contrasts; native = direction[:, None] * (scores[1] - scores[0]); frozen = direction[:, None] * (scores[2] - scores[0]); mediated = native - frozen
                    row = values[(upstream_name, downstream_name)]
                    row["native"].append(native[:, 0].cpu()); row["frozen"].append(frozen[:, 0].cpu()); row["mediated"].append(mediated[:, 0].cpu())
                    row["random_native"].append(native[:, 1:].cpu()); row["random_mediated"].append(mediated[:, 1:].cpu())
            break
    result = {}
    for key, row in values.items():
        native = torch.cat(row["native"]); frozen = torch.cat(row["frozen"]); mediated = torch.cat(row["mediated"])
        random_native = torch.cat(row["random_native"]).mean(0); random_mediated = torch.cat(row["random_mediated"]).mean(0)
        native_mean = float(native.mean()); mediated_mean = float(mediated.mean())
        result["__".join(key)] = {
            "examples": len(native), "native_win": float((native > 0).float().mean()), "native_mean": native_mean,
            "frozen_win": float((frozen > 0).float().mean()), "frozen_mean": float(frozen.mean()),
            "mediated_win": float((mediated > 0).float().mean()), "mediated_mean": mediated_mean,
            "mediation_fraction": mediated_mean / max(abs(native_mean), 1e-30),
            "native_random_percentile": float((random_native < native_mean).float().mean()),
            "mediated_random_percentile": float((random_mediated < mediated_mean).float().mean()),
        }
    return result, native_error


def main():
    config = json.loads(CONFIG.read_text()); base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    pairs = [(tokenizer.encode(us, add_special_tokens=False)[0], tokenizer.encode(uk, add_special_tokens=False)[0]) for us, uk in base["spelling_pairs"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = []; max_native_error = 0.0
    for fold in range(config["folds"]):
        target_indices = set(range(fold * config["target_pairs_per_fold"], (fold + 1) * config["target_pairs_per_fold"]))
        target_pairs = [pair for index, pair in enumerate(pairs) if index in target_indices]; cues = [pair for index, pair in enumerate(pairs) if index not in target_indices]
        contrasts = make_contrasts(model.lm_head.weight.shape[0], target_pairs, config["random_contrasts"], config["seed"] + fold, "cuda")
        for split, (corpus_start, corpus_end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[corpus_start:corpus_end, :128], dtype=np.int64)
            records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + fold * 10 + split)
            print("two-hop", fold, split, len(records), flush=True)
            rows, error = collect(model, raw, records, contrasts, counts, boundary_flags, config)
            max_native_error = max(max_native_error, error); results.append({"fold": fold, "split": split, "programs": rows})
    predicted = "L8H2_cue__L9H8_boundary"
    c0 = max_native_error <= 1e-4
    c1 = all(row["programs"][predicted]["native_win"] >= .75 for row in results)
    c2 = all(row["programs"][predicted]["native_mean"] >= 5 * max(abs(row["programs"][f"{control}__L9H8_boundary"]["native_mean"]), 1e-30)
             for row in results for control in ("L8H1_cue", "L8H2_noncue"))
    c3 = all(row["programs"][predicted]["mediation_fraction"] >= .50 for row in results)
    specificity_cells = 0
    for row in results:
        target = row["programs"][predicted]["mediation_fraction"]
        controls = [row["programs"][f"L8H2_cue__{name}"]["mediation_fraction"] for name in ("L9H7_boundary", "L9H8_nonboundary")]
        specificity_cells += int(all(target >= control + .20 for control in controls))
    c4 = specificity_cells >= 5
    contrast_cells = sum(row["programs"][predicted]["native_random_percentile"] >= .95 and row["programs"][predicted]["mediated_random_percentile"] >= .95 for row in results); c5 = contrast_cells >= 5
    hypotheses = {
        "C0_native_branch_recomposes": {"supported": c0, "max_logit_error": max_native_error},
        "C1_native_L8H2_cue_edge_transfers_style": {"supported": c1},
        "C2_upstream_effect_is_specific": {"supported": c2},
        "C3_L9H8_boundary_change_mediates": {"supported": c3},
        "C4_downstream_mediation_is_specific": {"supported": c4, "cells": specificity_cells},
        "C5_declared_contrast_beats_random": {"supported": c5, "cells": contrast_cells},
        "C6_native_two_hop_circuit_is_supported": {"supported": all((c0,c1,c2,c3,c4,c5))},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite native test of the L8H2-cue to L9H8-boundary circuit", "",
             "The actual L8H2 cue-edge write is replaced by its counterfactual write inside the original prefix, after which the Layer-8 MLP and every later block are recomputed. At Layer 9, the proposed downstream edge is either left native or frozen to its value before the upstream edit. Matched upstream head/source and downstream head/source controls receive the same treatment.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Fold/split | native win/mean | frozen mean | mediated fraction | random percentiles native/mediated | control mediation H7/nonboundary |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        target = row["programs"][predicted]; h7 = row["programs"]["L8H2_cue__L9H7_boundary"]; non = row["programs"]["L8H2_cue__L9H8_nonboundary"]
        lines.append(f"| {row['fold']}/{row['split']} | {target['native_win']:.3f}/{target['native_mean']:+.4g} | {target['frozen_mean']:+.4g} | {target['mediation_fraction']:.3f} | {target['native_random_percentile']:.3f}/{target['mediated_random_percentile']:.3f} | {h7['mediation_fraction']:.3f}/{non['mediation_fraction']:.3f} |")
    lines += ["", "## Interpretation boundary", "",
              "Replacing one upstream edge and freezing one downstream edge establishes a finite path-specific dependency. It does not imply that either complete head is monosemantic, nor that this route explains every contextual consequence of the spelling cue. The counterfactual upstream write comes from a full swapped-prefix execution, so the edge packages both changed routing and changed value; a subsequent factorization is needed to distinguish lexical payload from route selection."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(results)); labels = [f"f{r['fold']}s{r['split']}" for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].bar(x-.2, [r["programs"][predicted]["native_mean"] for r in results], .4, label="native downstream")
    axes[0].bar(x+.2, [r["programs"][predicted]["frozen_mean"] for r in results], .4, label="L9H8 frozen"); axes[0].set(xticks=x, xticklabels=labels, ylabel="Directed spelling shift", title="Does the upstream edge act through L9H8?"); axes[0].legend()
    for name in ("L9H8_boundary", "L9H7_boundary", "L9H8_nonboundary"):
        axes[1].plot(x, [r["programs"][f"L8H2_cue__{name}"]["mediation_fraction"] for r in results], marker="o", label=name)
    axes[1].axhline(.5, color="black", linestyle="--", linewidth=.8); axes[1].set(xticks=x, xticklabels=labels, ylabel="Mediated fraction", title="Downstream edge specificity"); axes[1].legend(fontsize=8)
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
