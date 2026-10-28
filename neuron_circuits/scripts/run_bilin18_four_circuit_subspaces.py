#!/usr/bin/env python3
"""Collect ordinary-site propagation data for four predeclared Bilin18 circuits.

The source neurons and semantic labels are frozen from the existing layer-1
atlas before this script evaluates causal behavior.  Sites exclude activation
tails: each source activation is 0.5--1.5 positional standard deviations from
its calibration mean, balanced across signs and disjoint document panels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, compare_logits, load_full_model


CIRCUITS = {
    "boundary_reset": [1825, 291, 2590, 4574],
    "capital_word_start": [1690, 2976, 175, 210],
    "number_format": [1535, 3448, 4562, 2812],
    "determiner_syntax": [4361, 1775, 3691, 3996],
}
UNITS = [unit for values in CIRCUITS.values() for unit in values]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--calibration-start", type=int, default=2000)
    parser.add_argument("--calibration-rows", type=int, default=1000)
    parser.add_argument("--discovery-start", type=int, default=4000)
    parser.add_argument("--transfer-start", type=int, default=8000)
    parser.add_argument("--candidate-rows", type=int, default=3000)
    parser.add_argument("--sites-per-sign", type=int, default=32)
    parser.add_argument("--min-position", type=int, default=8)
    parser.add_argument("--z-min", type=float, default=.5)
    parser.add_argument("--z-max", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ood-dir", type=Path, default=Path("artifacts/bilin18_ood_corpora"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/bilin18_four_circuits"))
    return parser.parse_args()


def freeze(model):
    for module in [model.embedding, model.blocks, model.lm_head]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def token_flags(tokenizer):
    flags = np.zeros((tokenizer.vocab_size, len(CIRCUITS)), dtype=np.uint8)
    determiners = {
        "the", "a", "an", "this", "that", "these", "those", "my", "your",
        "his", "her", "its", "our", "their", "some", "any", "each", "every",
        "either", "neither",
    }
    eot = tokenizer.eos_token_id
    for token in range(tokenizer.vocab_size):
        text = tokenizer.decode([token])
        stripped = text.strip()
        flags[token, 0] = token == eot
        flags[token, 1] = bool(stripped and stripped[0].isalpha() and stripped[0].isupper())
        flags[token, 2] = any(char.isdigit() for char in stripped)
        flags[token, 3] = stripped.lower() in determiners
    return flags


@torch.inference_mode()
def layer1_selected(model, ids, unit_index):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0, value_bus = x, None
    for layer in range(2):
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1:
            return product[..., unit_index].float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raise AssertionError("unreachable")


@torch.inference_mode()
def calibration(model, corpus, args, unit_index):
    sums = torch.zeros(args.sequence_length, len(UNITS), device=args.device)
    squares = torch.zeros_like(sums)
    rows = corpus[args.calibration_start:args.calibration_start + args.calibration_rows]
    count = 0
    for start in range(0, len(rows), args.batch_size):
        ids = torch.as_tensor(np.asarray(rows[start:start + args.batch_size,
                                             :args.sequence_length], dtype=np.int64),
                              device=args.device)
        values = layer1_selected(model, ids, unit_index)
        sums += values.sum(0)
        squares += values.square().sum(0)
        count += len(ids)
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(1e-5).sqrt()
    return mean, std


def merge_reservoir(saved, candidates, keep):
    saved.extend(candidates)
    saved.sort(key=lambda row: row[0], reverse=True)
    del saved[keep:]


@torch.inference_mode()
def select_sites(model, corpus, split, start_row, mean, std, flags, args, unit_index):
    rng = np.random.default_rng(args.seed + start_row)
    reservoirs = {(column, sign): [] for column in range(len(UNITS)) for sign in (-1, 1)}
    rows = corpus[start_row:start_row + args.candidate_rows]
    for start in range(0, len(rows), args.batch_size):
        raw = np.asarray(rows[start:start + args.batch_size, :args.sequence_length], dtype=np.int64)
        ids = torch.as_tensor(raw, device=args.device)
        values = layer1_selected(model, ids, unit_index)
        z = ((values - mean[None]) / std[None]).cpu().numpy()
        for column in range(len(UNITS)):
            for sign in (-1, 1):
                mask = ((z[..., column] * sign >= args.z_min) &
                        (z[..., column] * sign <= args.z_max))
                mask[:, :args.min_position] = False
                batch_rows, positions = np.where(mask)
                if not len(batch_rows):
                    continue
                priorities = rng.random(len(batch_rows))
                take = np.argpartition(priorities, -min(args.sites_per_sign, len(priorities)))[
                    -min(args.sites_per_sign, len(priorities)):]
                candidates = []
                for selected in take:
                    batch_row, position = int(batch_rows[selected]), int(positions[selected])
                    token = int(raw[batch_row, position])
                    candidates.append((float(priorities[selected]), {
                        "split": split,
                        "source_unit": UNITS[column],
                        "source_column": column,
                        "sign": sign,
                        "row": start_row + start + batch_row,
                        "position": position,
                        "token_id": token,
                        "z": float(z[batch_row, position, column]),
                        "activation": float(values[batch_row, position, column]),
                        "semantic_flags": flags[token].tolist(),
                    }))
                merge_reservoir(reservoirs[(column, sign)], candidates,
                                args.sites_per_sign)
    sites = []
    for column in range(len(UNITS)):
        for sign in (-1, 1):
            chosen = reservoirs[(column, sign)]
            if len(chosen) < args.sites_per_sign:
                raise RuntimeError(f"only {len(chosen)} sites for unit {UNITS[column]} sign {sign}")
            sites.extend(row for _, row in chosen)
    return sites


@torch.inference_mode()
def run_site(model, ids, position, source_unit, replacement=None):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0, value_bus = x, None
    captured = {}
    source_native = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1:
            source_native = float(product[0, position, source_unit])
            if replacement is not None:
                product = product.clone()
                product[0, position, source_unit] = replacement
        if layer in (2, 3):
            captured[layer] = product[0, position].float().cpu()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return (30 * torch.tanh(raw / 30))[0].cpu(), captured, source_native


@torch.inference_mode()
def semantic_panel(model, rows, flags, unit_index, args):
    features, labels = [], []
    for start in range(0, len(rows), args.batch_size):
        raw = np.asarray(rows[start:start + args.batch_size, :args.sequence_length], dtype=np.int64)
        ids = torch.as_tensor(raw, device=args.device)
        features.append(layer1_selected(model, ids, unit_index).cpu().to(torch.float16).numpy())
        labels.append(flags[raw])
    return np.concatenate(features).reshape(-1, len(UNITS)), np.concatenate(labels).reshape(-1, len(CIRCUITS))


def main():
    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = np.load(args.corpus, mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = token_flags(tokenizer)
    model = load_full_model(args.device)
    freeze(model)
    unit_index = torch.tensor(UNITS, device=args.device)
    mean, std = calibration(model, corpus, args, unit_index)
    sites = []
    for split, start in (("discovery", args.discovery_start),
                         ("transfer", args.transfer_start)):
        selected = select_sites(model, corpus, split, start, mean, std, flags,
                                args, unit_index)
        sites.extend(selected)
        print(f"selected {len(selected)} {split} ordinary sites", flush=True)

    deltas = {2: [], 3: []}
    records = []
    for index, site in enumerate(sites):
        row = np.asarray(corpus[site["row"]], dtype=np.int64)
        position = site["position"]
        ids = torch.as_tensor(row[:position + 1], device=args.device)[None]
        target = int(row[position + 1])
        native_logits, native_products, native_source = run_site(
            model, ids, position, site["source_unit"])
        replacement = mean[position, site["source_column"]]
        changed_logits, changed_products, _ = run_site(
            model, ids, position, site["source_unit"], replacement)
        for layer in (2, 3):
            deltas[layer].append((changed_products[layer] - native_products[layer]).to(torch.float16).numpy())
        records.append({
            **site,
            "target_id": target,
            "source_native_recomputed": native_source,
            "source_replacement": float(replacement),
            "source_delta": float(replacement) - native_source,
            "output_effect": compare_logits(native_logits, changed_logits, target),
        })
        if (index + 1) % 128 == 0:
            print(f"propagation {index + 1}/{len(sites)}", flush=True)

    np.savez_compressed(
        args.output / "ordinary_propagation.npz",
        layer2=np.asarray(deltas[2], dtype=np.float16),
        layer3=np.asarray(deltas[3], dtype=np.float16),
        positional_mean=mean.cpu().numpy().astype(np.float32),
        positional_std=std.cpu().numpy().astype(np.float32),
    )

    # Frozen FineWeb training/held-out and two OOD semantic panels.
    semantic_arrays = {}
    panels = {
        "fineweb_train": corpus[12000:12800],
        "fineweb_test": corpus[12800:13600],
        "wikitext103": np.load(args.ood_dir / "wikitext103_1200x129.npy", mmap_mode="r")[:800],
        "tinystories": np.load(args.ood_dir / "tinystories_1200x129.npy", mmap_mode="r")[:800],
    }
    for name, rows in panels.items():
        features, labels = semantic_panel(model, rows, flags, unit_index, args)
        semantic_arrays[f"{name}_features"] = features
        semantic_arrays[f"{name}_labels"] = labels
        print(f"semantic panel {name}: {len(features)} positions", flush=True)
    np.savez_compressed(args.output / "semantic_panels.npz", **semantic_arrays)

    manifest = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "circuits": CIRCUITS,
        "unit_order": UNITS,
        "semantic_order": list(CIRCUITS),
        "ordinary_site_definition": {
            "z_min": args.z_min, "z_max": args.z_max,
            "sites_per_sign_per_source_per_split": args.sites_per_sign,
            "min_position": args.min_position,
        },
        "sites": records,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "Circuit names are hypotheses frozen from the earlier coarse atlas, not established meanings.",
            "Boundary/reset uses the literal GPT-2 end-of-text token as its semantic proxy.",
            "Ordinary sites are moderate deviations for each source neuron, not an unbiased sample of every token.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "output": str(args.output), "sites": len(records),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
