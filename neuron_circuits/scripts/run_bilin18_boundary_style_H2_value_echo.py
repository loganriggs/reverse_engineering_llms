#!/usr/bin/env python3
"""Test whether L17H2 post-boundary values are literal low-rank state echoes."""

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
from run_bilin18_boundary_style_downstream_screen import upstream_states
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_H2_value_echo.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_H2_value_echo.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_H2_VALUE_ECHO.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_H2_value_echo.png"


def project(values, basis):
    return (values @ basis.T) @ basis


@torch.inference_mode()
def layer17_details(model, original, swapped, cue, boundary):
    dimension = model.config.n_embd
    x, x0, bus = upstream_states(model, original, swapped, cue, boundary)
    for layer in range(9, 18):
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(
            block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt
        )
        pre = mixed + details["write"]
        normalized = F.rms_norm(pre, (dimension,))
        mlp = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        if layer < 17:
            x = pre + mlp
            bus = details["value_bus"]
            continue
        return block, mixed, details
    raise AssertionError("Layer 17 was not reached")


@torch.inference_mode()
def collect_value_deltas(model, raw, records, config):
    chunks = []
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start : start + config["batch_size"]]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch):
            swapped[index, row[3]] = row[5]
        cue = torch.as_tensor([row[3] for row in batch], device="cuda")
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        _, _, details = layer17_details(model, original, swapped, cue, boundary)
        values = details["mixed_value"][:, :, config["head"]]
        delta = values[size:] - values[:size]
        position = torch.arange(original.shape[1], device="cuda")[None]
        mask = (position > boundary[:, None]) & (position < 127)
        chunks.append(delta[mask].float().cpu())
    return torch.cat(chunks)


def fit_bases(values, config):
    covariance = values.double().T @ values.double()
    _, vectors = torch.linalg.eigh(covariance)
    pca = vectors.flip(1).T.float()
    generator = torch.Generator().manual_seed(config["seed"])
    random = []
    for _ in range(config["random_subspaces"]):
        matrix = torch.randn(values.shape[1], max(config["ranks"]), generator=generator)
        q, _ = torch.linalg.qr(matrix)
        random.append(q.T.contiguous())
    return pca, random


def fidelity(target, prediction):
    denominator = target.double().square().sum().clamp_min(1e-30)
    return float(1 - (target.double() - prediction.double()).square().sum() / denominator)


@torch.inference_mode()
def evaluate(model, raw, records, contrasts, pca_basis, random_bases, config):
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    head = config["head"]
    destination = 127
    ranks = config["ranks"]
    value_targets = []
    value_predictions = {rank: [] for rank in ranks}
    random_value_predictions = {(draw, rank): [] for draw in range(len(random_bases)) for rank in ranks}
    output = {"full": [], "routing": []}
    output.update({f"pca_{rank}": [] for rank in ranks})
    output.update({f"random_{draw}_{rank}": [] for draw in range(len(random_bases)) for rank in ranks})

    projection = model.blocks[17].attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
    pca_gpu = pca_basis.to("cuda")
    random_gpu = [basis.to("cuda") for basis in random_bases]

    for start in range(0, len(records), config["batch_size"]):
        batch = records[start : start + config["batch_size"]]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch):
            swapped[index, row[3]] = row[5]
        cue = torch.as_tensor([row[3] for row in batch], device="cuda")
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        direction = torch.as_tensor(
            [1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32
        )
        block, mixed, details = layer17_details(model, original, swapped, cue, boundary)
        pattern = (details["score1"] * details["score2"])[:, head, destination].float()
        values = details["mixed_value"][:, :, head].float()
        p0, p1 = pattern[:size], pattern[size:]
        v0, v1 = values[:size], values[size:]
        delta = v1 - v0
        position = torch.arange(original.shape[1], device="cuda")[None]
        mask = (position > boundary[:, None]) & (position < destination)
        value_targets.append(delta[mask].cpu())

        reconstructed = {}
        for rank in ranks:
            predicted = project(delta, pca_gpu[:rank])
            value_predictions[rank].append(predicted[mask].cpu())
            reconstructed[f"pca_{rank}"] = predicted
            for draw, basis in enumerate(random_gpu):
                random_prediction = project(delta, basis[:rank])
                random_value_predictions[(draw, rank)].append(random_prediction[mask].cpu())
                reconstructed[f"random_{draw}_{rank}"] = random_prediction

        mask_float = mask[:, :, None].float()
        original_edge = p0[:, :, None] * v0
        routing_edge = p1[:, :, None] * v0
        full_edge = p1[:, :, None] * v1
        edge_changes = {
            "routing": ((routing_edge - original_edge) * mask_float).sum(1) @ projection,
            "full": ((full_edge - original_edge) * mask_float).sum(1) @ projection,
        }
        for name, prediction in reconstructed.items():
            changed = p1[:, :, None] * (v0 + prediction)
            edge_changes[name] = ((changed - original_edge) * mask_float).sum(1) @ projection

        variants = [details["write"][:size].float()]
        for name in output:
            modified = details["write"][:size].float().clone()
            modified[:, destination] += edge_changes[name]
            variants.append(modified)
        repeated = len(variants)
        attention = torch.cat(variants)
        mixed_original = mixed[:size].float().repeat(repeated, 1, 1)
        normalized = F.rms_norm(mixed_original + attention, (dimension,))
        final = mixed_original + attention + block.mlp.Down(
            block.mlp.Left(normalized) * block.mlp.Right(normalized)
        ) + block.mlp.Down_bias
        logits = 30 * torch.tanh(
            model.lm_head(F.rms_norm(final, (dimension,))[:, destination]).float() / 30
        )
        scores = (logits @ contrasts[:, :1]).reshape(repeated, size)
        for index, name in enumerate(output, start=1):
            output[name].append((direction * (scores[index] - scores[0])).cpu())

    target_values = torch.cat(value_targets)
    value_fidelity = {
        str(rank): fidelity(target_values, torch.cat(value_predictions[rank])) for rank in ranks
    }
    random_value_fidelity = {
        str(rank): [
            fidelity(target_values, torch.cat(random_value_predictions[(draw, rank)]))
            for draw in range(len(random_bases))
        ]
        for rank in ranks
    }
    effects = {name: torch.cat(chunks) for name, chunks in output.items()}
    full = effects["full"]
    routing = effects["routing"]
    value_target = full - routing
    causal_fidelity = {
        str(rank): fidelity(value_target, effects[f"pca_{rank}"] - routing) for rank in ranks
    }
    random_causal_fidelity = {
        str(rank): [
            fidelity(value_target, effects[f"random_{draw}_{rank}"] - routing)
            for draw in range(len(random_bases))
        ]
        for rank in ranks
    }
    full_mean = float(full.mean())
    return {
        "examples": len(full),
        "value_positions": len(target_values),
        "value_fidelity": value_fidelity,
        "random_value_fidelity": random_value_fidelity,
        "causal_value_fidelity": causal_fidelity,
        "random_causal_value_fidelity": random_causal_fidelity,
        "full_effect_mean": full_mean,
        "routing_effect_mean": float(routing.mean()),
        "routing_fraction": float(routing.mean()) / max(abs(full_mean), 1e-30),
        "pca_effect_means": {str(rank): float(effects[f"pca_{rank}"].mean()) for rank in ranks},
    }


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack(
        [vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]
    )
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[
        :, FEATURES.index("period")
    ].astype(bool)
    pairs = [
        (
            tokenizer.encode(first, add_special_tokens=False)[0],
            tokenizer.encode(second, add_special_tokens=False)[0],
        )
        for first, second in base["spelling_pairs"]
    ]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    cells = []
    for fold in range(config["folds"]):
        target_indices = set(
            range(
                fold * config["target_pairs_per_fold"],
                (fold + 1) * config["target_pairs_per_fold"],
            )
        )
        target_pairs = [pair for index, pair in enumerate(pairs) if index in target_indices]
        cues = [pair for index, pair in enumerate(pairs) if index not in target_indices]
        contrasts = make_contrasts(
            model.lm_head.weight.shape[0], target_pairs, 0, config["seed"] + fold, "cuda"
        )
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(
                raw,
                boundary_flags,
                cues,
                counts,
                127,
                config["examples_per_label_per_split"],
                config["seed"] + fold * 10 + split,
            )
            cells.append((fold, split, raw, records, contrasts))

    discovery = next(cell for cell in cells if cell[0] == 0 and cell[1] == 0)
    print("fit value echo", len(discovery[3]), flush=True)
    discovery_values = collect_value_deltas(model, discovery[2], discovery[3], config)
    pca_basis, random_bases = fit_bases(discovery_values, config)

    results = []
    for fold, split, raw, records, contrasts in cells:
        print("evaluate value echo", fold, split, len(records), flush=True)
        results.append(
            {
                "fold": fold,
                "split": split,
                **evaluate(model, raw, records, contrasts, pca_basis, random_bases, config),
            }
        )
    heldout = [row for row in results if not (row["fold"] == 0 and row["split"] == 0)]
    v0 = True
    v1 = all(row["value_fidelity"]["4"] >= 0.50 for row in heldout)
    v2 = all(row["causal_value_fidelity"]["4"] >= 0.70 for row in heldout)
    v3 = all(
        row["causal_value_fidelity"]["4"]
        >= max(row["random_causal_value_fidelity"]["4"]) + 0.20
        for row in heldout
    )
    v4_cells = sum(
        row["causal_value_fidelity"]["1"]
        >= 0.8 * max(row["causal_value_fidelity"]["16"], 1e-30)
        for row in heldout
    )
    v4 = v4_cells >= 4
    v5 = all(row["routing_fraction"] < 0.50 for row in heldout)
    hypotheses = {
        "V0_discovery_basis_is_frozen": {"supported": v0},
        "V1_rank4_is_a_literal_value_echo": {"supported": v1},
        "V2_rank4_is_causally_sufficient": {"supported": v2},
        "V3_rank4_beats_random_subspaces": {"supported": v3},
        "V4_rank1_is_nearly_sufficient": {"supported": v4, "cells": v4_cells},
        "V5_routing_alone_is_incomplete": {"supported": v5},
        "V6_literal_low_rank_echo_ontology": {"supported": all((v0, v1, v2, v3, v4, v5))},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(
        json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2)
    )

    lines = [
        "# Does L17H2 read a literal low-rank spelling echo?",
        "",
        "An uncentered value-delta basis is fit only on fold 0/split 0. On all cells, native H2 post-boundary value changes are projected into ranks 1, 4, and 16 while native routing is retained, then the final MLP and output are recomputed. Eight frozen random subspaces per rank are controls.",
        "",
        "## Locked results",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += [
        "",
        "| Fold/split | raw value R1/R4/R16 | causal value R1/R4/R16 | routing/full | best random R4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        raw_values = "/".join(f"{row['value_fidelity'][str(rank)]:.3f}" for rank in config["ranks"])
        causal_values = "/".join(f"{row['causal_value_fidelity'][str(rank)]:.3f}" for rank in config["ranks"])
        lines.append(
            f"| {row['fold']}/{row['split']} | {raw_values} | {causal_values} | "
            f"{row['routing_fraction']:.3f} | {max(row['random_causal_value_fidelity']['4']):.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "High raw and causal fidelity would support a literal repeated value direction. Low raw but high causal fidelity would instead identify a compact output-relevant quotient. Low fidelity on both would mean that the sparse H2 read function binds genuinely different positional value transforms. The test concerns intervention-induced differences, not the full semantic content of source states.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = [f"f{row['fold']}s{row['split']}" for row in results]
    x = np.arange(len(results))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for rank in config["ranks"]:
        axes[0].plot(x, [row["value_fidelity"][str(rank)] for row in results], marker="o", label=f"rank {rank}")
        axes[1].plot(x, [row["causal_value_fidelity"][str(rank)] for row in results], marker="o", label=f"rank {rank}")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Explained energy", title="Raw H2 value delta")
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Finite causal fidelity", title="Value-dependent H2 effect")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
