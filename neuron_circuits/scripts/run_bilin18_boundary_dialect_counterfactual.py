#!/usr/bin/env python3
"""Red-team boundary dialect transport with single-token counterfactual swaps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import leading_mode


CONFIG = Path("configs/bilin18_boundary_dialect_counterfactual.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_dialect_counterfactual.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_DIALECT_COUNTERFACTUAL.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_dialect_counterfactual.png"


def select_records(raw, boundary, cue_pairs, counts, destination, maximum, seed):
    cue_to_pair = {token: (us, uk) for us, uk in cue_pairs for token in (us, uk)}
    records = []
    for row_index, tokens in enumerate(raw[:, :destination + 1]):
        boundaries = np.flatnonzero(boundary[tokens])
        if not len(boundaries): continue
        source = int(boundaries[-1]); hits = [(position, int(token)) for position, token in enumerate(tokens[:source]) if int(token) in cue_to_pair]
        if len(hits) != 1: continue
        cue_position, token = hits[0]; us, uk = cue_to_pair[token]; original_us = int(token == us); replacement = uk if original_us else us
        distance = destination - source; low = 0 if distance == 0 else 2 ** int(np.floor(np.log2(distance))); high = 1 if distance == 0 else min(2 * low, destination + 1)
        candidate = np.arange(max(0, destination - high + 1), destination - low + 1)
        candidate = candidate[(candidate <= destination) & ~boundary[tokens[candidate]]]
        if not len(candidate): continue
        target_frequency = counts[int(tokens[source])]
        control = int(candidate[np.argmin(np.abs(np.log1p(counts[tokens[candidate]]) - np.log1p(target_frequency)))])
        records.append((row_index, source, control, cue_position, original_us, replacement))
    rng = np.random.default_rng(seed); by_label = {label: [row for row in records if row[4] == label] for label in (0, 1)}
    count = min(maximum, len(by_label[0]), len(by_label[1])); selected = []
    for label in (0, 1): selected.extend([by_label[label][index] for index in rng.choice(len(by_label[label]), count, replace=False)])
    rng.shuffle(selected); return selected


def make_contrasts(vocab_size, target_pairs, random_count, seed, device):
    tokens = [token for pair in target_pairs for token in pair]
    matrix = torch.zeros(vocab_size, random_count + 1, device=device)
    for us, uk in target_pairs: matrix[us, 0] = 1 / len(target_pairs); matrix[uk, 0] = -1 / len(target_pairs)
    rng = np.random.default_rng(seed)
    correct_positive = {us for us, _ in target_pairs}
    for column in range(1, random_count + 1):
        while True:
            positive = set(rng.choice(tokens, len(target_pairs), replace=False).tolist())
            if positive != correct_positive: break
        for token in tokens: matrix[token, column] = (1 if token in positive else -1) / len(target_pairs)
    return matrix


@torch.inference_mode()
def collect_swaps(model, raw, records, layer, head, source_kind, output_mode, contrasts, config):
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    output_mode = output_mode.to("cuda")
    leading_changes, native_changes, delta_changes = [], [], []
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]
        original = np.stack([raw[row[0], :128].copy() for row in batch]); swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        ids = torch.as_tensor(np.concatenate([original, swapped]), device="cuda")
        selected_source = 1 if source_kind == "boundary" else 2
        source = torch.as_tensor([row[selected_source] for row in batch] * 2, device="cuda")
        destination = torch.full_like(source, 127); rows = torch.arange(len(ids), device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks):
            if block_index < layer:
                x, bus = block(x, bus, x0); continue
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
            selected_value = details["pattern"][rows, head, destination, source, None].float() * details["mixed_value"][rows, source, head].float()
            write = selected_value @ projection
            modified = details["write"].float().clone(); modified[rows, destination] -= write
            attention = torch.cat([details["write"].float(), modified], 0)
            x0 = torch.cat([x0.float(), x0.float()], 0); mixed = torch.cat([mixed.float(), mixed.float()], 0); bus = torch.cat([details["value_bus"].float(), details["value_bus"].float()], 0)
            state = F.rms_norm(mixed + attention, (dimension,)); x = mixed + attention + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
            for suffix in model.blocks[block_index + 1:]: x, bus = suffix(x, bus, x0)
            destination2 = torch.cat([destination, destination]); rows2 = torch.arange(2 * len(ids), device="cuda")
            logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,))[rows2, destination2]).float() / 30)).reshape(2, len(ids), -1)
            native = logits[0] @ contrasts; delta = (logits[0] - logits[1]) @ contrasts
            batch_size = len(batch); direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)[:, None]
            leading = (write @ output_mode).reshape(2, batch_size)
            leading_changes.extend((direction[:, 0] * (leading[1] - leading[0])).cpu().tolist())
            native_changes.append((direction * (native[batch_size:] - native[:batch_size])).cpu())
            delta_changes.append((direction * (delta[batch_size:] - delta[:batch_size])).cpu())
            break
    return {"leading": np.asarray(leading_changes), "native": torch.cat(native_changes).numpy(), "delta": torch.cat(delta_changes).numpy()}


def summarize(values):
    correct = values["delta"][:, 0]; random = values["delta"][:, 1:]
    random_means = random.mean(0); correct_mean = float(correct.mean())
    return {"examples": len(correct), "leading_win": float(np.mean(values["leading"] > 0)), "leading_mean": float(values["leading"].mean()), "native_win": float(np.mean(values["native"][:, 0] > 0)), "native_mean": float(values["native"][:, 0].mean()), "delta_win": float(np.mean(correct > 0)), "delta_mean": correct_mean, "random_p95_mean": float(np.quantile(random_means, .95)), "declared_random_percentile": float(np.mean(random_means < correct_mean))}


def main():
    config = json.loads(CONFIG.read_text()); base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r"); counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    pairs = [(tokenizer.encode(us, add_special_tokens=False)[0], tokenizer.encode(uk, add_special_tokens=False)[0]) for us, uk in base["spelling_pairs"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    programs = [("L9H8_boundary", 9, 8, "boundary"), ("L9H8_nonboundary", 9, 8, "control"), ("L9H7_boundary", 9, 7, "boundary")]
    modes = {}
    for _, layer, head, _ in programs:
        key = (layer, head)
        if key in modes: continue
        block = model.blocks[layer].attn; head_dim = model.config.n_embd // model.config.n_head; sl = slice(head * head_dim, (head + 1) * head_dim)
        modes[key] = leading_mode(block.c_proj.weight[:, sl].detach().float().cpu(), block.c_v.weight[sl].detach().float().cpu())[0]
    results = []
    for fold in range(config["folds"]):
        target_indices = set(range(fold * config["target_pairs_per_fold"], (fold + 1) * config["target_pairs_per_fold"])); target_pairs = [pair for index, pair in enumerate(pairs) if index in target_indices]; cues = [pair for index, pair in enumerate(pairs) if index not in target_indices]
        contrasts = make_contrasts(model.lm_head.weight.shape[0], target_pairs, config["random_contrasts"], config["seed"] + fold, "cuda")
        for split, (start, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[start:end, :128], dtype=np.int64)
            records = select_records(raw, boundary, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + fold * 10 + split)
            for name, layer, head, source_kind in programs:
                print("swap", fold, split, name, len(records), flush=True)
                values = collect_swaps(model, raw, records, layer, head, source_kind, modes[(layer, head)], contrasts, config)
                results.append({"fold": fold, "split": split, "program": name, **summarize(values)})
    boundary_rows = [row for row in results if row["program"] == "L9H8_boundary"]
    nonboundary_rows = [row for row in results if row["program"] == "L9H8_nonboundary"]
    control_rows = [row for row in results if row["program"] == "L9H7_boundary"]
    lookup = lambda rows, fold, split: next(row for row in rows if row["fold"] == fold and row["split"] == split)
    s0 = all(row["native_win"] >= .60 for row in boundary_rows)
    s1 = all(row["leading_win"] >= .60 for row in boundary_rows)
    s2 = all(row["delta_win"] >= .60 for row in boundary_rows)
    ratios = [row["delta_mean"] / max(abs(lookup(nonboundary_rows, row["fold"], row["split"])["delta_mean"]), 1e-30) for row in boundary_rows]
    s3 = float(np.median(ratios)) >= 1.5
    head_wins = sum(row["delta_win"] >= lookup(control_rows, row["fold"], row["split"])["delta_win"] + .05 for row in boundary_rows)
    s4 = head_wins >= 5
    random_wins = sum(row["declared_random_percentile"] >= .95 for row in boundary_rows); s5 = random_wins >= 5
    hypotheses = {"S0_native_model_responds_to_spelling_swap": {"supported": s0}, "S1_boundary_leading_mode_responds": {"supported": s1}, "S2_boundary_edge_transmits_disjoint_style": {"supported": s2}, "S3_transport_is_boundary_specific": {"supported": s3, "median_ratio": float(np.median(ratios))}, "S4_transport_is_head_specific": {"supported": s4, "cells": head_wins}, "S5_declared_contrast_beats_random": {"supported": s5, "cells": random_wins}, "S6_counterfactual_dialect_transport_is_supported": {"supported": s0 and s2 and s4 and s5}}
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Counterfactual red team of boundary dialect transport", "", "One US/UK spelling cue is swapped in an otherwise identical prefix. The tested output contrast uses four disjoint spelling pairs, rotated over three folds. L9H8's latest-boundary edge is compared with a frequency-near non-boundary edge from the same distance bin and with L9H7 at the same boundary. Thirty-one balanced random signings of the target tokens test whether the declared contrast was privileged.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Fold/split | Program | N | leading win | native disjoint win | edge-delta win / mean | contrast percentile |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in results: lines.append(f"| {row['fold']}/{row['split']} | {row['program']} | {row['examples']} | {row['leading_win']:.3f} | {row['native_win']:.3f} | {row['delta_win']:.3f} / {row['delta_mean']:+.4g} | {row['declared_random_percentile']:.3f} |")
    lines += ["", "## Guardrails", "", "A one-token swap is a much stronger control for topic than observational labels, but it can be locally unnatural and still perturb generic model state. Disjoint cue and target pairs rule out copying the same spelling token; they do not eliminate a broader lexical-style association learned from correlated domains. The non-boundary control is matched by log-distance and approximate token frequency, not by source residual content. This supports a transportable spelling/style variable, not the claim that L9H8 represents an abstract sociolinguistic concept in every context."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained"); labels = [f"f{row['fold']}s{row['split']}" for row in boundary_rows]; x = np.arange(len(labels))
    for index, rows in enumerate((boundary_rows, nonboundary_rows, control_rows)): axes[0].bar(x + (index - 1) * .25, [row["delta_win"] for row in rows], width=.24, label=rows[0]["program"])
    axes[0].axhline(.5, color="black", linewidth=.8); axes[0].set(xticks=x, xticklabels=labels, ylabel="Correct counterfactual shift rate", title="Disjoint spelling-logit transport"); axes[0].legend(fontsize=8)
    axes[1].bar(x, [row["declared_random_percentile"] for row in boundary_rows]); axes[1].axhline(.95, color="black", linewidth=.8, linestyle="--"); axes[1].set(xticks=x, xticklabels=labels, ylabel="Percentile among random contrasts", title="Declared contrast specificity")
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "boundary": boundary_rows}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
