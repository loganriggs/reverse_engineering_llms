#!/usr/bin/env python3
"""Compare the best square replacement's neurons with native layer 17."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import CLASSES, TOKENIZER, classify_token, context_for
from run_bilin18_final_mlp_compression import final_inputs
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from train_bilin18_final_mlp_architectures import NarrowSquare


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "artifacts/bilin18_final_mlp_architectures/square_k128.pt"))
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--discovery-rows", type=int, default=16000)
    parser.add_argument("--heldout-rows", type=int, default=4000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tail-size", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/bilin18_final_mlp_architectures/square_k128_audit.json"))
    return parser.parse_args()


@torch.inference_mode()
def collect(model, student, rows, class_lookup, args, keep_tails):
    units = student.reader.shape[0]
    count = 0
    sums = torch.zeros(units, device=args.device)
    squares = torch.zeros_like(sums)
    class_abs = torch.zeros(len(CLASSES), units, device=args.device)
    class_signed = torch.zeros_like(class_abs)
    class_counts = torch.zeros(len(CLASSES), device=args.device)
    top_v = torch.full((args.tail_size, units), -torch.inf, device=args.device) if keep_tails else None
    top_i = torch.full((args.tail_size, units), -1, dtype=torch.long, device=args.device) if keep_tails else None
    reader = F.normalize(student.reader, dim=1)
    for start in range(0, len(rows), args.batch_size):
        ids = torch.as_tensor(np.asarray(rows[start:start + args.batch_size,
                                             :args.sequence_length], dtype=np.int64),
                              device=args.device)
        pre, _ = final_inputs(model, ids)
        x = F.rms_norm(pre, (model.config.n_embd,)).float()
        activation = F.linear(x, reader).square().reshape(-1, units)
        labels = class_lookup[ids].reshape(-1)
        one_hot = F.one_hot(labels, len(CLASSES)).float()
        sums += activation.sum(0)
        squares += activation.square().sum(0)
        class_abs += one_hot.T @ activation.abs()
        class_signed += one_hot.T @ activation
        class_counts += one_hot.sum(0)
        if keep_tails:
            indices = torch.arange(count, count + len(activation), device=args.device)
            cv = torch.cat((top_v, activation), 0)
            ci = torch.cat((top_i, indices[:, None].expand_as(activation)), 0)
            top_v, selected = torch.topk(cv, args.tail_size, dim=0)
            top_i = torch.gather(ci, 0, selected)
        count += len(activation)
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt()
    class_mean = class_abs / class_counts[:, None].clamp_min(1)
    global_mean = class_abs.sum(0) / class_counts.sum()
    lift = class_mean / global_mean.clamp_min(1e-8)
    signed_mean = class_signed / class_counts[:, None].clamp_min(1)
    return {"count": count, "mean": mean, "std": std, "lift": lift,
            "signed_mean": signed_mean, "top_v": top_v, "top_i": top_i}


@torch.inference_mode()
def main():
    args = parse_args()
    corpus = np.load(args.corpus, mmap_mode="r")
    discovery_rows = corpus[:args.discovery_rows]
    heldout_rows = corpus[args.discovery_rows:args.discovery_rows + args.heldout_rows]
    contexts = torch.from_numpy(np.asarray(discovery_rows).copy())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    class_lookup = torch.tensor([classify_token(text) for text in texts], device=args.device)
    model = load_full_model(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    units = saved["state_dict"]["reader"].shape[0]
    student = NarrowSquare(model.config.n_embd, units).to(args.device)
    student.load_state_dict(saved["state_dict"])
    student.eval()
    discovery = collect(model, student, discovery_rows, class_lookup, args, True)
    heldout = collect(model, student, heldout_rows, class_lookup, args, False)
    winner = discovery["lift"].argmax(0)
    heldout_lift = torch.gather(heldout["lift"], 0, winner[None]).squeeze(0)
    positive = discovery["signed_mean"].argmax(0)
    negative = discovery["signed_mean"].argmin(0)
    positive_mean = torch.gather(heldout["signed_mean"], 0, positive[None]).squeeze(0)
    negative_mean = torch.gather(heldout["signed_mean"], 0, negative[None]).squeeze(0)
    gap = (positive_mean - negative_mean) / heldout["std"].clamp_min(1e-8)
    response = model.lm_head.weight.float() @ student.down
    logit_norm = response.norm(dim=0)
    score = discovery["std"] * logit_norm
    order = score.argsort(descending=True)
    channels = []
    for unit in range(units):
        top_tokens = response[:, unit].topk(8).indices.tolist()
        bottom_tokens = response[:, unit].topk(8, largest=False).indices.tolist()
        indices = discovery["top_i"][:, unit].cpu().tolist()
        channels.append({
            "unit": unit, "importance_rank": int((order == unit).nonzero()[0]) + 1,
            "preferred_class": CLASSES[int(winner[unit])],
            "positive_class": CLASSES[int(positive[unit])],
            "negative_class": CLASSES[int(negative[unit])],
            "heldout_lift": round(float(heldout_lift[unit]), 5),
            "heldout_gap_sd": round(float(gap[unit]), 5),
            "activation_std": round(float(discovery["std"][unit]), 5),
            "direct_logit_norm": round(float(logit_norm[unit]), 5),
            "importance_score": round(float(score[unit]), 5),
            "top_contexts": [context_for(i, contexts, tokenizer, args.sequence_length) for i in indices],
            "top_output_tokens": [texts[i] for i in top_tokens],
            "bottom_output_tokens": [texts[i] for i in bottom_tokens],
        })
    active = score > score.median()
    result = {
        "architecture": "128 symmetric-square channels",
        "discovery_tokens": discovery["count"], "heldout_tokens": heldout["count"],
        "active_half_lift_median": float(heldout_lift[active].median()),
        "active_half_fraction_lift_ge_2": float((heldout_lift[active] >= 2).float().mean()),
        "active_half_gap_median": float(gap[active].median()),
        "active_half_fraction_gap_ge_1": float((gap[active] >= 1).float().mean()),
        "importance_effective_support": float(score.sum().square() / score.square().sum()),
        "importance_order": order.tolist(), "channels": channels,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("importance_order", "channels")}, indent=2))
    for unit in order[:12]:
        print(json.dumps(channels[int(unit)], indent=2), flush=True)


if __name__ == "__main__":
    main()
