#!/usr/bin/env python3
"""Test whether Bilin18's final 4,608-channel MLP can be replaced by a subset.

Channels are ranked once on a discovery split by activation variance times the
norm of their direct Down→unembedding direction. Evaluation uses disjoint
FineWeb rows. Omitted channels are either zeroed or replaced by their discovery
mean, the latter corresponding to folding their constant contribution into a
new Down bias. Random fixed subsets and per-token oracle subsets are controls.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--discovery-start", type=int, default=2000)
    parser.add_argument("--discovery-rows", type=int, default=400)
    parser.add_argument("--eval-start", type=int, default=2600)
    parser.add_argument("--eval-rows", type=int, default=400)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-chunk", type=int, default=512)
    parser.add_argument("--random-replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/bilin18_final_mlp_compression"),
    )
    return parser.parse_args()


@torch.inference_mode()
def final_inputs(model, ids):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0 = x
    value_bus = None
    for block in model.blocks[:-1]:
        x, value_bus = block(x, value_bus, x0)
    block = model.blocks[-1]
    mixed = block.lambdas[0] * x + block.lambdas[1] * x0
    attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    pre_mlp = mixed + attention_write
    normalized = F.rms_norm(pre_mlp, (dimension,))
    left = block.mlp.Left(normalized)
    right = block.mlp.Right(normalized)
    return pre_mlp, left * right


@torch.inference_mode()
def logits_from_products(model, pre_mlp, products):
    block = model.blocks[-1]
    mlp_write = block.mlp.Down(products) + block.mlp.Down_bias
    post = pre_mlp + mlp_write
    normalized = F.rms_norm(post, (model.config.n_embd,))
    raw = model.lm_head(normalized).float()
    return 30 * torch.tanh(raw / 30), mlp_write, post


@torch.inference_mode()
def activation_moments(model, corpus, args):
    total = 0
    sums = torch.zeros(4608, device=args.device)
    squares = torch.zeros_like(sums)
    stop = args.discovery_start + args.discovery_rows
    rows = corpus[args.discovery_start:stop]
    for start in range(0, len(rows), args.batch_size):
        ids = torch.as_tensor(
            np.asarray(rows[start:start + args.batch_size, :args.sequence_length], dtype=np.int64),
            device=args.device,
        )
        _, products = final_inputs(model, ids)
        flat = products.float().reshape(-1, products.shape[-1])
        sums += flat.sum(0)
        squares += flat.square().sum(0)
        total += flat.shape[0]
        if (start // args.batch_size + 1) % 50 == 0:
            print(f"discovery {start + len(ids)}/{len(rows)} rows", flush=True)
    mean = sums / total
    variance = (squares / total - mean.square()).clamp_min(0)
    return mean, variance, total


@torch.inference_mode()
def direct_logit_norms(model, args):
    down = model.blocks[-1].mlp.Down.weight.float()
    unembedding = model.lm_head.weight.float()
    square = torch.zeros(down.shape[1], device=args.device)
    max_abs = torch.zeros_like(square)
    for start in range(0, unembedding.shape[0], args.vocab_chunk):
        response = unembedding[start:start + args.vocab_chunk] @ down
        square += response.square().sum(0)
        max_abs = torch.maximum(max_abs, response.abs().max(0).values)
    norm = square.sqrt()
    return norm, max_abs / norm.clamp_min(1e-8)


def make_fixed_conditions(order, units, args):
    ks = [0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, units]
    conditions = []
    for k in ks:
        chosen = order[:k]
        conditions.append((f"ranked_mean_k{k}", chosen, "mean", k, None))
        conditions.append((f"ranked_zero_k{k}", chosen, "zero", k, None))
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for k in (32, 128, 512, 2048):
        for replicate in range(args.random_replicates):
            chosen = torch.randperm(units, generator=generator)[:k].to(order.device)
            conditions.append((f"random_mean_k{k}_r{replicate}", chosen, "mean", k, replicate))
    return conditions


def accumulator():
    return {
        "tokens": 0,
        "nll_sum": 0.0,
        "kl_sum": 0.0,
        "top1_same": 0,
        "write_error_sq": 0.0,
        "write_base_sq": 0.0,
        "post_error_sq": 0.0,
        "post_base_sq": 0.0,
    }


@torch.inference_mode()
def evaluate(model, corpus, mean, order, conditions, args):
    stats = {name: accumulator() for name, *_ in conditions}
    baseline = accumulator()
    rows = corpus[args.eval_start:args.eval_start + args.eval_rows]
    units = mean.numel()
    for start in range(0, len(rows), args.batch_size):
        raw = np.asarray(rows[start:start + args.batch_size, :args.sequence_length + 1], dtype=np.int64)
        ids = torch.as_tensor(
            raw[:, :args.sequence_length],
            device=args.device,
        )
        pre, products = final_inputs(model, ids)
        pre = pre[:, -1:]
        products = products[:, -1:]
        base_logits, base_write, base_post = logits_from_products(model, pre, products)
        target = torch.as_tensor(raw[:, args.sequence_length], device=args.device)
        base_eval = base_logits[:, 0]
        base_logp = F.log_softmax(base_eval, -1)
        base_p = base_logp.exp()
        n = target.numel()
        baseline["tokens"] += n
        baseline["nll_sum"] += float(F.nll_loss(base_logp, target, reduction="sum"))
        baseline["top1_same"] += n

        for name, chosen, replacement, k, replicate in conditions:
            if k == units:
                changed_products = products
            else:
                fill = mean.to(products.dtype) if replacement == "mean" else torch.zeros_like(mean, dtype=products.dtype)
                changed_products = fill.view(1, 1, -1).expand_as(products).clone()
                if k:
                    changed_products[..., chosen] = products[..., chosen]
            logits, write, post = logits_from_products(model, pre, changed_products)
            changed_eval = logits[:, 0]
            changed_logp = F.log_softmax(changed_eval, -1)
            item = stats[name]
            item["tokens"] += n
            item["nll_sum"] += float(F.nll_loss(changed_logp, target, reduction="sum"))
            item["kl_sum"] += float((base_p * (base_logp - changed_logp)).sum())
            item["top1_same"] += int((base_eval.argmax(-1) == changed_eval.argmax(-1)).sum())
            dw = (write - base_write).float()
            dp = (post - base_post).float()
            item["write_error_sq"] += float(dw.square().sum())
            item["write_base_sq"] += float(base_write.float().square().sum())
            item["post_error_sq"] += float(dp.square().sum())
            item["post_base_sq"] += float(base_post.float().square().sum())
        if (start // args.batch_size + 1) % 25 == 0:
            print(f"evaluation {start + len(ids)}/{len(rows)} rows", flush=True)
    base_ce = baseline["nll_sum"] / baseline["tokens"]
    output = {"baseline": {"tokens": baseline["tokens"], "cross_entropy": base_ce}, "conditions": []}
    for name, chosen, replacement, k, replicate in conditions:
        item = stats[name]
        output["conditions"].append({
            "name": name,
            "selection": "ranked" if replicate is None else "random",
            "replacement": replacement,
            "k": k,
            "replicate": replicate,
            "cross_entropy": item["nll_sum"] / item["tokens"],
            "cross_entropy_delta": item["nll_sum"] / item["tokens"] - base_ce,
            "mean_kl_from_full": item["kl_sum"] / item["tokens"],
            "top1_agreement": item["top1_same"] / item["tokens"],
            "relative_mlp_write_l2": math.sqrt(item["write_error_sq"] / max(item["write_base_sq"], 1e-12)),
            "relative_post_residual_l2": math.sqrt(item["post_error_sq"] / max(item["post_base_sq"], 1e-12)),
        })
    return output


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = np.load(args.corpus, mmap_mode="r")
    model = load_full_model(args.device)
    mean, variance, discovery_tokens = activation_moments(model, corpus, args)
    logit_norm, concentration = direct_logit_norms(model, args)
    score = variance * logit_norm.square()
    order = score.argsort(descending=True)
    conditions = make_fixed_conditions(order, mean.numel(), args)
    evaluated = evaluate(model, corpus, mean, order, conditions, args)
    result = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "layer": 17,
        "config": {**vars(args), "corpus": str(args.corpus), "output": str(args.output)},
        "discovery_tokens": discovery_tokens,
        "ranking": "activation variance × squared direct Down→unembedding norm",
        "ranking_order": [int(unit) for unit in order.tolist()],
        "channel_scores": [float(value) for value in score.tolist()],
        "score_effective_support": float(score.sum().square() / score.square().sum().clamp_min(1e-12)),
        "score_fraction_top_k": {
            str(k): float(score[order[:k]].sum() / score.sum().clamp_min(1e-12))
            for k in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4608)
        },
        "top_units": [
            {
                "rank": rank + 1,
                "unit": int(unit),
                "activation_mean": float(mean[unit]),
                "activation_variance": float(variance[unit]),
                "direct_logit_norm": float(logit_norm[unit]),
                "max_logit_fraction": float(concentration[unit]),
                "score": float(score[unit]),
            }
            for rank, unit in enumerate(order[:256].tolist())
        ],
        **evaluated,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(evaluated, indent=2))
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
