#!/usr/bin/env python3
"""Freeze novel layer-1 circuit hypotheses without looking at natural text.

Every GPT-2 vocabulary item is run as an isolated one-token sequence. Candidate
semantics come only from the tokenizer string; neuron selection, decoder fitting,
and downstream layer-2 path ranking use model activations and weights. Token IDs
of opposite parity form discovery and held-out splits. Natural corpora are not
opened by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import load_full_model


OLD_UNITS = {1825, 291, 2590, 4574, 1690, 2976, 175, 210,
             1535, 3448, 4562, 2812, 4361, 1775, 3691, 3996}
CATEGORIES = (
    "newline", "opening_delimiter", "closing_delimiter", "quote",
    "code_operator", "url_web", "non_ascii", "word_continuation",
    "suffix_ing", "suffix_ed", "suffix_ly", "all_caps",
    "comma_period", "colon_semicolon",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--units-per-circuit", type=int, default=4)
    parser.add_argument("--circuits", type=int, default=6)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--min-positive", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/bilin18_weight_only_circuits"))
    return parser.parse_args()


def token_flags(text: str):
    stripped = text.strip()
    lower = stripped.lower()
    quote_chars = set("'\"`\u2018\u2019\u201c\u201d")
    operators = {"=", "==", "+", "-", "*", "/", "%", "=>", "!=",
                 "<=", ">=", "&&", "||", "+=", "-=", "::", "->", "&", "|"}
    return np.asarray([
        "\n" in text,
        bool(stripped) and set(stripped) <= set("([{<"),
        bool(stripped) and set(stripped) <= set(")]}>") ,
        bool(stripped) and set(stripped) <= quote_chars,
        stripped in operators,
        any(fragment in lower for fragment in ("http", "www", ".com", ".org", "://")),
        any(ord(char) > 127 for char in text),
        bool(stripped) and not text[:1].isspace() and stripped.isalpha() and len(stripped) >= 2,
        stripped.isalpha() and len(stripped) > 4 and lower.endswith("ing"),
        stripped.isalpha() and len(stripped) > 3 and lower.endswith("ed"),
        stripped.isalpha() and len(stripped) > 3 and lower.endswith("ly"),
        stripped.isalpha() and len(stripped) >= 2 and stripped.isupper(),
        stripped in {",", ".", "...", ",."},
        stripped in {":", ";", ":;", "::"},
    ], dtype=np.uint8)


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2
        start = stop
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.inference_mode()
def layer1_products(model, ids):
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
            return product[:, 0].float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raise AssertionError("unreachable")


def standardized_gaps(sum_all, square_all, count_all, sum_pos, count_pos):
    count_pos = count_pos.to(dtype=sum_all.dtype)
    mean = sum_all / count_all
    variance = square_all / count_all - mean.square()
    std = variance.clamp_min(1e-8).sqrt()
    count_neg = count_all - count_pos[:, None]
    mean_pos = sum_pos / count_pos[:, None].clamp_min(1)
    mean_neg = (sum_all[None] - sum_pos) / count_neg.clamp_min(1)
    return (mean_pos - mean_neg) / std[None], mean, std


def effective_support(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.sum() ** 2 / max(np.square(values).sum(), 1e-30))


def main():
    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    flags = np.stack([token_flags(text) for text in texts])
    splits = (np.arange(tokenizer.vocab_size) & 1).astype(np.int64)
    model = load_full_model(args.device)
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    width = model.blocks[1].mlp.Left.weight.shape[0]
    sums = [torch.zeros(width, device=args.device) for _ in range(2)]
    squares = [torch.zeros(width, device=args.device) for _ in range(2)]
    positive_sums = [torch.zeros(len(CATEGORIES), width, device=args.device) for _ in range(2)]
    counts = np.bincount(splits, minlength=2)
    positive_counts = np.stack([flags[splits == split].sum(0) for split in range(2)])
    all_values = np.empty((tokenizer.vocab_size, width), dtype=np.float16)
    for start in range(0, tokenizer.vocab_size, args.batch_size):
        stop = min(start + args.batch_size, tokenizer.vocab_size)
        ids = torch.arange(start, stop, device=args.device)[:, None]
        values = layer1_products(model, ids)
        all_values[start:stop] = values.cpu().to(torch.float16).numpy()
        batch_flags = torch.as_tensor(flags[start:stop], device=args.device, dtype=torch.float32)
        batch_splits = splits[start:stop]
        for split in range(2):
            mask = torch.as_tensor(batch_splits == split, device=args.device)
            selected = values[mask]
            sums[split] += selected.sum(0)
            squares[split] += selected.square().sum(0)
            positive_sums[split] += batch_flags[mask].T @ selected
        if start % (args.batch_size * 20) == 0:
            print(f"isolated tokens {stop}/{tokenizer.vocab_size}", flush=True)

    gaps, means, stds = [], [], []
    for split in range(2):
        gap, mean, std = standardized_gaps(
            sums[split], squares[split], counts[split], positive_sums[split],
            torch.as_tensor(positive_counts[split], device=args.device))
        gaps.append(gap.cpu().numpy())
        means.append(mean.cpu().numpy())
        stds.append(std.cpu().numpy())

    discovery = splits == 0
    heldout = ~discovery
    values32 = all_values.astype(np.float32)
    category_rows = []
    for ci, name in enumerate(CATEGORIES):
        valid = bool(np.all(positive_counts[:, ci] >= args.min_positive))
        order = np.argsort(np.abs(gaps[0][ci]))[::-1]
        units = [int(unit) for unit in order if unit not in OLD_UNITS][:args.units_per_circuit]
        x_train = values32[discovery][:, units]
        y_train = flags[discovery, ci].astype(bool)
        mean = x_train.mean(0)
        std = x_train.std(0).clip(1e-5)
        z_train = (x_train - mean) / std
        weight = z_train[y_train].mean(0) - z_train[~y_train].mean(0)
        score = ((values32[:, units] - mean) / std) @ weight
        heldout_auc = auc(flags[heldout, ci], score[heldout])
        ranked = np.argsort(score[heldout])
        held_ids = np.flatnonzero(heldout)
        examples = {
            "highest": [{"id": int(i), "text": texts[i], "score": float(score[i])}
                        for i in held_ids[ranked[-8:][::-1]]],
            "lowest": [{"id": int(i), "text": texts[i], "score": float(score[i])}
                       for i in held_ids[ranked[:8]]],
        }
        category_rows.append({
            "name": name, "valid": valid, "positive_counts": positive_counts[:, ci].tolist(),
            "units": units, "unit_discovery_gaps": [float(gaps[0][ci, unit]) for unit in units],
            "decoder_mean": mean.tolist(), "decoder_std": std.tolist(),
            "decoder_weight": weight.tolist(), "heldout_auc": heldout_auc,
            "examples": examples,
        })

    # Freeze the best diverse hypotheses before any natural-data validation.
    frozen = []
    for row in sorted(category_rows, key=lambda item: item["heldout_auc"], reverse=True):
        units = set(row["units"])
        if not row["valid"] or not np.isfinite(row["heldout_auc"]):
            continue
        if any(len(units & set(other["units"])) / len(units | set(other["units"])) > .5
               for other in frozen):
            continue
        frozen.append(row)
        if len(frozen) == args.circuits:
            break

    down = model.blocks[1].mlp.Down.weight.detach().float()
    left2 = model.blocks[2].mlp.Left.weight.detach().float()
    right2 = model.blocks[2].mlp.Right.weight.detach().float()
    paths = {}
    for row in frozen:
        source = torch.as_tensor(row["units"], device=args.device)
        writes = down[:, source]
        left = left2 @ writes
        right = right2 @ writes
        per_source = (left.square() + right.square()).sqrt()
        score = per_source.square().mean(1).sqrt()
        order = torch.argsort(score, descending=True)
        top = order[:args.targets].cpu().tolist()
        scores = score.cpu().numpy()
        paths[row["name"]] = {
            "targets": top,
            "target_scores": [float(scores[index]) for index in top],
            "effective_support": effective_support(scores),
            "top32_energy_fraction": float(np.square(scores[top]).sum() / np.square(scores).sum()),
            "per_source_top_scores": per_source[top].cpu().tolist(),
        }

    output = {
        "method": {
            "source": "isolated vocabulary items only; no natural corpus",
            "split": "even token IDs discovery, odd token IDs held out",
            "selection": "top absolute standardized mean-gap channels; linear four-channel decoder",
            "path_score": "RMS layer1 Down -> layer2 Left/Right weight product",
        },
        "model": {"vocab_size": tokenizer.vocab_size, "mlp_width": width},
        "categories": category_rows,
        "frozen_names": [row["name"] for row in frozen],
        "paths": paths,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "discovery.json").write_text(json.dumps(output, indent=2))
    print("frozen", [(row["name"], round(row["heldout_auc"], 4), row["units"])
                     for row in frozen], flush=True)
    print(f"wrote {args.output / 'discovery.json'} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
