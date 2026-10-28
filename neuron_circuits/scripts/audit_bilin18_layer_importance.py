#!/usr/bin/env python3
"""Rank Bilin18 MLP layers by finite mean-replacement effects."""

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
    parser.add_argument("--mean-start", type=int, default=12000)
    parser.add_argument("--mean-rows", type=int, default=400)
    parser.add_argument("--eval-start", type=int, default=12600)
    parser.add_argument("--eval-rows", type=int, default=600)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/bilin18_full_mlp_atlas/layer_importance.json"))
    return parser.parse_args()


@torch.inference_mode()
def forward(model, ids, replacement_layer=None, replacement=None):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0, value_bus = x, None
    writes = []
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        write = block.mlp.Down(product) + block.mlp.Down_bias
        writes.append(write)
        if layer == replacement_layer:
            replacement = replacement.to(write.dtype)
            if replacement.ndim == 1:
                write = replacement.view(1, 1, -1).expand_as(write)
            else:
                write = replacement[None, :write.shape[1]].expand_as(write)
        x = pre + write
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30), writes


@torch.inference_mode()
def collect_means(model, corpus, args):
    sums = torch.zeros(model.config.n_layer, args.sequence_length,
                       model.config.n_embd, device=args.device)
    count = 0
    rows = corpus[args.mean_start:args.mean_start + args.mean_rows]
    for start in range(0, len(rows), args.batch_size):
        ids = torch.as_tensor(np.asarray(rows[start:start + args.batch_size,
                                             :args.sequence_length], dtype=np.int64),
                              device=args.device)
        _, writes = forward(model, ids)
        for layer, write in enumerate(writes):
            sums[layer] += write.float().sum(0)
        count += len(ids)
    return sums / count


@torch.inference_mode()
def main():
    args = parse_args()
    corpus = np.load(args.corpus, mmap_mode="r")
    model = load_full_model(args.device)
    means = collect_means(model, corpus, args)
    layers = model.config.n_layer
    stats = [{"n": 0, "nll": 0., "kl": 0., "same": 0} for _ in range(layers)]
    baseline_nll, baseline_n = 0., 0
    rows = corpus[args.eval_start:args.eval_start + args.eval_rows]
    for start in range(0, len(rows), args.batch_size):
        raw = np.asarray(rows[start:start + args.batch_size,
                              :args.sequence_length + 1], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :args.sequence_length], device=args.device)
        labels = torch.as_tensor(raw[:, args.sequence_length], device=args.device)
        native, _ = forward(model, ids)
        native_logp = F.log_softmax(native, -1)
        native_p = native_logp.exp()
        baseline_nll += float(F.nll_loss(native_logp, labels, reduction="sum"))
        baseline_n += len(ids)
        for layer in range(layers):
            changed, _ = forward(model, ids, layer, means[layer])
            changed_logp = F.log_softmax(changed, -1)
            item = stats[layer]
            item["n"] += len(ids)
            item["nll"] += float(F.nll_loss(changed_logp, labels, reduction="sum"))
            item["kl"] += float((native_p * (native_logp - changed_logp)).sum())
            item["same"] += int((native.argmax(-1) == changed.argmax(-1)).sum())
        if (start // args.batch_size + 1) % 50 == 0:
            print(f"evaluation: {start + len(ids)}/{len(rows)} rows", flush=True)
    base_ce = baseline_nll / baseline_n
    results = []
    for layer, item in enumerate(stats):
        ce = item["nll"] / item["n"]
        results.append({
            "layer": layer, "cross_entropy_delta": ce - base_ce,
            "perplexity_ratio": math.exp(ce - base_ce),
            "kl_from_full": item["kl"] / item["n"],
            "top1_agreement": item["same"] / item["n"],
            "mean_write_norm": float(means[layer].norm()),
        })
    result = {
        "intervention": "replace each position's MLP write in one layer by its position-specific FineWeb discovery mean",
        "baseline_cross_entropy": base_ce,
        "mean_tokens": args.mean_rows * args.sequence_length,
        "evaluation_examples": baseline_n,
        "layers": results,
        "rank_by_ce_delta": [item["layer"] for item in sorted(results, key=lambda x: x["cross_entropy_delta"], reverse=True)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
