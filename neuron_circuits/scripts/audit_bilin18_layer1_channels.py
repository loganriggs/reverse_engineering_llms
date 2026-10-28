#!/usr/bin/env python3
"""Causally rank layer-1 Bilin18 channels and measure sparse sufficiency.

The breadth atlas showed that replacing the whole layer-1 MLP by its positional
mean is unusually damaging.  This script asks whether that effect is carried by
a small stable channel set.  It ranks channels on a discovery panel with
|gradient * (activation - positional mean)|, freezes the ranking, and then runs
finite necessity and sufficiency interventions on disjoint FineWeb, WikiText,
and TinyStories panels.  Every intervention recomputes layers 2--17.
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

from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--mean-start", type=int, default=12000)
    parser.add_argument("--mean-rows", type=int, default=400)
    parser.add_argument("--rank-start", type=int, default=13200)
    parser.add_argument("--rank-rows", type=int, default=256)
    parser.add_argument("--eval-start", type=int, default=13600)
    parser.add_argument("--eval-rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rank-batch-size", type=int, default=1)
    parser.add_argument("--ks", default="0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,4608")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ood-dir", type=Path, default=Path("artifacts/bilin18_ood_corpora"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/bilin18_layer1_deep/iteration1_channels.json"))
    return parser.parse_args()


def freeze(model):
    for module in [model.embedding, model.blocks, model.lm_head]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def rows129(array, start, count, length):
    raw = np.asarray(array[start:start + count, :length + 1], dtype=np.int64)
    if len(raw) != count:
        raise ValueError(f"requested {count} rows at {start}, found {len(raw)}")
    return raw


def forward(model, ids, layer, positional_mean=None, replace_mask=None,
            capture_leaf=False, capture_product=False, replacement_scale=None,
            product_override=None):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0, value_bus = x, None
    leaf = None
    for index, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if index == layer:
            if capture_leaf:
                product = product.detach().requires_grad_(True)
                leaf = product
            elif capture_product:
                leaf = product.detach()
            if product_override is not None:
                product = product_override.to(product.dtype)
            if replacement_scale is not None:
                mean = positional_mean[:product.shape[1]].to(product.dtype)[None]
                product = mean + replacement_scale * (product - mean)
            if replace_mask is not None:
                mean = positional_mean[:product.shape[1]].to(product.dtype)[None]
                mask = replace_mask.to(product.device)[None, None]
                product = torch.where(mask, mean, product)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30), leaf


@torch.inference_mode()
def positional_product_mean(model, rows, args):
    units = model.config.expansion_factor * model.config.n_embd
    sums = torch.zeros(args.sequence_length, units, device=args.device)
    count = 0
    block = model.blocks[args.layer]
    dimension = model.config.n_embd
    for start in range(0, len(rows), args.batch_size):
        ids = torch.as_tensor(rows[start:start + args.batch_size, :-1], device=args.device)
        x = F.rms_norm(model.embedding(ids), (dimension,))
        x0, value_bus = x, None
        for index in range(args.layer + 1):
            current = model.blocks[index]
            mixed = current.lambdas[0] * x + current.lambdas[1] * x0
            attention_write, value_bus = current.attn(F.rms_norm(mixed, (dimension,)), value_bus)
            pre = mixed + attention_write
            normalized = F.rms_norm(pre, (dimension,))
            product = current.mlp.Left(normalized) * current.mlp.Right(normalized)
            if index == args.layer:
                sums += product.float().sum(0)
            x = pre + current.mlp.Down(product) + current.mlp.Down_bias
        count += len(ids)
    return sums / count


def gradient_rank(model, rows, mean, args, label):
    units = mean.shape[-1]
    absolute = torch.zeros(units, device=args.device)
    signed = torch.zeros(units, device=args.device)
    grad_norm = torch.zeros(units, device=args.device)
    count = 0
    for start in range(0, len(rows), args.rank_batch_size):
        raw = rows[start:start + args.rank_batch_size]
        ids = torch.as_tensor(raw[:, :-1], device=args.device)
        labels = torch.as_tensor(raw[:, -1], device=args.device)
        with torch.enable_grad():
            logits, leaf = forward(model, ids, args.layer, capture_leaf=True)
            loss = F.cross_entropy(logits, labels)
            gradient, = torch.autograd.grad(loss, leaf)
        delta = leaf.detach().float() - mean[:leaf.shape[1]][None]
        contribution = gradient.float() * delta
        absolute += contribution.abs().sum((0, 1))
        # First-order NLL effect of replacing the native activation by its mean.
        signed += (-contribution).sum((0, 1))
        grad_norm += gradient.float().abs().sum((0, 1))
        count += leaf.shape[0] * leaf.shape[1]
        if (start // args.rank_batch_size + 1) % 32 == 0:
            print(f"{label} gradient rank: {start + len(raw)}/{len(rows)}", flush=True)
    return {
        "absolute": absolute / count,
        "signed": signed / count,
        "gradient_abs": grad_norm / count,
    }


@torch.inference_mode()
def evaluate_curve(model, rows, mean, order, args, panel):
    units = len(order)
    ks = sorted(set(min(int(value), units) for value in args.ks.split(",")))
    accumulators = {
        mode: {k: {"nll": 0.0, "kl": 0.0, "same": 0, "n": 0} for k in ks}
        for mode in ("necessity", "sufficiency")
    }
    base_nll = 0.0
    base_n = 0
    base_masks = {}
    for mode in accumulators:
        for k in ks:
            mask = torch.zeros(units, dtype=torch.bool, device=args.device)
            if mode == "necessity":
                mask[order[:k]] = True
            else:
                mask[:] = True
                mask[order[:k]] = False
            base_masks[(mode, k)] = mask
    for start in range(0, len(rows), args.batch_size):
        raw = rows[start:start + args.batch_size]
        ids = torch.as_tensor(raw[:, :-1], device=args.device)
        labels = torch.as_tensor(raw[:, -1], device=args.device)
        native, _ = forward(model, ids, args.layer)
        native_logp = F.log_softmax(native, -1)
        native_p = native_logp.exp()
        base_nll += float(F.nll_loss(native_logp, labels, reduction="sum"))
        base_n += len(ids)
        for mode in accumulators:
            for k in ks:
                changed, _ = forward(model, ids, args.layer, mean, base_masks[(mode, k)])
                logp = F.log_softmax(changed, -1)
                item = accumulators[mode][k]
                item["nll"] += float(F.nll_loss(logp, labels, reduction="sum"))
                item["kl"] += float((native_p * (native_logp - logp)).sum())
                item["same"] += int((native.argmax(-1) == changed.argmax(-1)).sum())
                item["n"] += len(ids)
        if (start // args.batch_size + 1) % 16 == 0:
            print(f"{panel} finite curves: {start + len(raw)}/{len(rows)}", flush=True)
    base_ce = base_nll / base_n
    result = {"examples": base_n, "baseline_ce": base_ce}
    for mode, values in accumulators.items():
        result[mode] = []
        for k, item in values.items():
            ce = item["nll"] / item["n"]
            result[mode].append({
                "k": k,
                "ce_delta": ce - base_ce,
                "perplexity_ratio": math.exp(min(ce - base_ce, 20)),
                "kl_from_native": item["kl"] / item["n"],
                "top1_agreement": item["same"] / item["n"],
            })
    return result


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def compare_ranks(reference, other):
    first, second = rankdata(reference), rankdata(other)
    pearson = float(np.corrcoef(first, second)[0, 1])
    descending_ref = np.argsort(reference)[::-1]
    descending_other = np.argsort(other)[::-1]
    overlaps = {}
    for k in (16, 32, 64, 128, 256):
        overlaps[str(k)] = len(set(descending_ref[:k]) & set(descending_other[:k])) / k
    return {"spearman": pearson, "top_k_overlap": overlaps}


def main():
    args = parse_args()
    started = time.time()
    fineweb = np.load(args.corpus, mmap_mode="r")
    mean_rows = rows129(fineweb, args.mean_start, args.mean_rows, args.sequence_length)
    rank_rows = rows129(fineweb, args.rank_start, args.rank_rows, args.sequence_length)
    panels = {
        "fineweb": rows129(fineweb, args.eval_start, args.eval_rows, args.sequence_length),
        "wikitext103": rows129(np.load(args.ood_dir / "wikitext103_1200x129.npy", mmap_mode="r"),
                                0, args.eval_rows, args.sequence_length),
        "tinystories": rows129(np.load(args.ood_dir / "tinystories_1200x129.npy", mmap_mode="r"),
                                0, args.eval_rows, args.sequence_length),
    }
    model = load_full_model(args.device)
    freeze(model)
    mean = positional_product_mean(model, mean_rows, args)
    ranking = gradient_rank(model, rank_rows, mean, args, "fineweb")
    order = ranking["absolute"].argsort(descending=True)
    # A smaller independent OOD gradient sample tests whether the priority list
    # is peculiar to FineWeb without using OOD ranks to choose interventions.
    ood_ranks = {}
    for panel in ("wikitext103", "tinystories"):
        sample = panels[panel][:min(96, len(panels[panel]))]
        scores = gradient_rank(model, sample, mean, args, panel)
        ood_ranks[panel] = {
            "comparison": compare_ranks(ranking["absolute"].cpu().numpy(),
                                        scores["absolute"].cpu().numpy()),
            "absolute_scores": scores["absolute"].cpu().tolist(),
        }
    curves = {
        panel: evaluate_curve(model, rows, mean, order, args, panel)
        for panel, rows in panels.items()
    }
    down_norm = model.blocks[args.layer].mlp.Down.weight.float().norm(dim=0)
    top = []
    for rank, unit in enumerate(order[:256].tolist(), 1):
        top.append({
            "rank": rank,
            "unit": unit,
            "mean_abs_gradient_delta": float(ranking["absolute"][unit]),
            "mean_signed_predicted_ablation_nll": float(ranking["signed"][unit]),
            "mean_abs_gradient": float(ranking["gradient_abs"][unit]),
            "down_norm": float(down_norm[unit]),
        })
    result = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "layer": args.layer,
        "intervention": "replace selected native product channels by a position-specific FineWeb mean and recompute all downstream layers",
        "ranking": "mean absolute gradient * (native product - positional mean) on a frozen FineWeb discovery panel",
        "mean_rows": args.mean_rows,
        "rank_rows": args.rank_rows,
        "eval_rows_per_panel": args.eval_rows,
        "top_channels": top,
        "fineweb_absolute_scores": ranking["absolute"].cpu().tolist(),
        "ood_rank_stability": ood_ranks,
        "curves": curves,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "elapsed_seconds": result["elapsed_seconds"],
        "top_units": [row["unit"] for row in top[:16]],
        "ood_rank_stability": {k: v["comparison"] for k, v in ood_ranks.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
