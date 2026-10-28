#!/usr/bin/env python3
"""Output-first weight census for Bilin18 MLP layers 15, 16, and 17."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from audit_bilin18_mlp_discriminators import (
    CLASSES,
    MODEL_REPO,
    SNAPSHOT,
    TOKENIZER,
    classify_token,
)
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="17,16,15")
    parser.add_argument("--vocab-chunk", type=int, default=512)
    parser.add_argument("--tail-size", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/bilin18_layers_15_17/output_geometry"),
    )
    return parser.parse_args()


def update(values, ids, saved_values, saved_ids, largest):
    expanded = ids[:, None].expand_as(values)
    cv = torch.cat((saved_values, values), 0)
    ci = torch.cat((saved_ids, expanded), 0)
    best, rows = torch.topk(cv, saved_values.shape[0], dim=0, largest=largest)
    return best, torch.gather(ci, 0, rows)


def quantiles(values):
    return {
        "median": round(float(values.median()), 6),
        "q10": round(float(values.quantile(.1)), 6),
        "q90": round(float(values.quantile(.9)), 6),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    token_class = torch.tensor([classify_token(x) for x in texts], device=args.device)
    state = torch.load(
        SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True,
    )
    unembedding = state["lm_head.weight"].to(args.device).float()[:tokenizer.vocab_size]
    result = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "layers": layers,
        "classes": list(CLASSES),
        "layer_results": [],
    }
    for layer in layers:
        prefix = f"transformer.h.{layer}.mlp."
        down = state[prefix + "Down.weight"].to(args.device).float()
        units = down.shape[1]
        norm_sq = torch.zeros(units, device=args.device)
        fourth = torch.zeros_like(norm_sq)
        max_abs = torch.zeros_like(norm_sq)
        top_v = torch.full((args.tail_size, units), -torch.inf, device=args.device)
        low_v = torch.full_like(top_v, torch.inf)
        top_i = torch.full((args.tail_size, units), -1, dtype=torch.long, device=args.device)
        low_i = torch.full_like(top_i, -1)
        class_abs = torch.zeros(len(CLASSES), units, device=args.device)
        class_signed = torch.zeros_like(class_abs)
        class_counts = torch.zeros(len(CLASSES), device=args.device)
        for start in range(0, tokenizer.vocab_size, args.vocab_chunk):
            stop = min(start + args.vocab_chunk, tokenizer.vocab_size)
            ids = torch.arange(start, stop, device=args.device)
            response = unembedding[start:stop] @ down
            norm_sq += response.square().sum(0)
            fourth += response.pow(4).sum(0)
            max_abs = torch.maximum(max_abs, response.abs().max(0).values)
            one_hot = torch.nn.functional.one_hot(token_class[ids], len(CLASSES)).float()
            class_abs += one_hot.T @ response.abs()
            class_signed += one_hot.T @ response
            class_counts += one_hot.sum(0)
            top_v, top_i = update(response, ids, top_v, top_i, True)
            low_v, low_i = update(response, ids, low_v, low_i, False)
        norm = norm_sq.sqrt()
        class_mean = class_abs / class_counts[:, None].clamp_min(1)
        global_mean = class_abs.sum(0) / class_counts.sum()
        lift = class_mean / global_mean.clamp_min(1e-8)
        winner = lift.argmax(0)
        winner_lift = torch.gather(lift, 0, winner[None]).squeeze(0)
        signed_mean = class_signed / class_counts[:, None].clamp_min(1)
        signed_sd = (signed_mean - signed_mean.mean(0)).std(0).clamp_min(1e-8)
        signed_gap = (signed_mean.max(0).values - signed_mean.min(0).values) / signed_sd
        positive = signed_mean.argmax(0)
        negative = signed_mean.argmin(0)
        energy = norm_sq
        energy_order = energy.argsort(descending=True)
        cumulative = energy[energy_order].cumsum(0) / energy.sum().clamp_min(1e-12)
        channels = []
        for unit in range(units):
            channels.append({
                "unit": unit,
                "down_norm": round(float(down[:, unit].norm()), 6),
                "direct_logit_norm": round(float(norm[unit]), 6),
                "max_logit_fraction": round(float(max_abs[unit] / norm[unit].clamp_min(1e-8)), 6),
                "vocab_effective_support": round(float(norm_sq[unit].square() / fourth[unit].clamp_min(1e-12)), 3),
                "output_class": CLASSES[int(winner[unit])],
                "output_class_lift": round(float(winner_lift[unit]), 5),
                "output_positive_class": CLASSES[int(positive[unit])],
                "output_negative_class": CLASSES[int(negative[unit])],
                "output_signed_gap": round(float(signed_gap[unit]), 5),
                "top_output_tokens": [texts[int(i)] for i in top_i[:, unit]],
                "top_output_values": [round(float(v), 5) for v in top_v[:, unit]],
                "bottom_output_tokens": [texts[int(i)] for i in low_i[:, unit]],
                "bottom_output_values": [round(float(v), 5) for v in low_v[:, unit]],
            })
        aggregate = {
            "direct_logit_norm": quantiles(norm),
            "max_logit_fraction": quantiles(max_abs / norm.clamp_min(1e-8)),
            "vocab_effective_support": quantiles(norm_sq.square() / fourth.clamp_min(1e-12)),
            "output_class_lift": quantiles(winner_lift),
            "output_signed_gap": quantiles(signed_gap),
            "channel_energy_effective_support": round(float(energy.sum().square() / energy.square().sum()), 3),
            "energy_fraction_top_k": {
                str(k): round(float(cumulative[min(k, units) - 1]), 6)
                for k in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, units)
            },
        }
        result["layer_results"].append({"layer": layer, "aggregate": aggregate, "channels": channels})
        print(json.dumps({"layer": layer, **aggregate}, indent=2), flush=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
