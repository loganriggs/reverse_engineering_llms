#!/usr/bin/env python3
"""One-pass breadth-first activation atlas for all 18 Bilin18 MLPs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import CLASSES, TOKENIZER, classify_token, context_for
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--discovery-rows", type=int, default=16000)
    parser.add_argument("--heldout-rows", type=int, default=4000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tail-size", type=int, default=4)
    parser.add_argument("--selected-per-layer", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bilin18_full_mlp_atlas"))
    return parser.parse_args()


class Stats:
    def __init__(self, layers, units, classes, tail, device, keep_tails):
        shape = (layers, units)
        self.count = 0
        self.sums = torch.zeros(shape, device=device)
        self.squares = torch.zeros(shape, device=device)
        self.class_abs = torch.zeros(layers, classes, units, device=device)
        self.class_signed = torch.zeros_like(self.class_abs)
        self.class_counts = torch.zeros(classes, device=device)
        self.top_values = torch.full((layers, tail, units), -torch.inf, device=device) if keep_tails else None
        self.bottom_values = torch.full((layers, tail, units), torch.inf, device=device) if keep_tails else None
        self.top_indices = torch.full((layers, tail, units), -1, dtype=torch.long, device=device) if keep_tails else None
        self.bottom_indices = torch.full((layers, tail, units), -1, dtype=torch.long, device=device) if keep_tails else None


def update_tail(values, indices, saved_values, saved_indices, largest):
    candidate_values = torch.cat((saved_values, values), 0)
    candidate_indices = torch.cat((saved_indices, indices[:, None].expand_as(values)), 0)
    chosen_values, chosen_rows = torch.topk(candidate_values, len(saved_values), dim=0, largest=largest)
    return chosen_values, torch.gather(candidate_indices, 0, chosen_rows)


@torch.inference_mode()
def collect(model, rows, class_lookup, stats, args, label):
    dimension = model.config.n_embd
    for start in range(0, len(rows), args.batch_size):
        ids = torch.as_tensor(np.asarray(rows[start:start + args.batch_size,
                                             :args.sequence_length], dtype=np.int64),
                              device=args.device)
        labels = class_lookup[ids].reshape(-1)
        one_hot = F.one_hot(labels, len(CLASSES)).float()
        stats.class_counts += one_hot.sum(0)
        global_indices = torch.arange(stats.count, stats.count + ids.numel(), device=args.device)
        x = F.rms_norm(model.embedding(ids), (dimension,))
        x0, value_bus = x, None
        for layer, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
            pre = mixed + attention_write
            normalized = F.rms_norm(pre, (dimension,))
            product = (block.mlp.Left(normalized) * block.mlp.Right(normalized)).float()
            flat = product.reshape(-1, product.shape[-1])
            stats.sums[layer] += flat.sum(0)
            stats.squares[layer] += flat.square().sum(0)
            stats.class_abs[layer] += one_hot.T @ flat.abs()
            stats.class_signed[layer] += one_hot.T @ flat
            if stats.top_values is not None:
                stats.top_values[layer], stats.top_indices[layer] = update_tail(
                    flat, global_indices, stats.top_values[layer], stats.top_indices[layer], True)
                stats.bottom_values[layer], stats.bottom_indices[layer] = update_tail(
                    flat, global_indices, stats.bottom_values[layer], stats.bottom_indices[layer], False)
            x = pre + block.mlp.Down(product.to(pre.dtype)) + block.mlp.Down_bias
        stats.count += ids.numel()
        batch_index = start // args.batch_size + 1
        if batch_index % 250 == 0 or start + len(ids) == len(rows):
            print(f"{label}: {start + len(ids)}/{len(rows)} rows", flush=True)


def metric_tensors(stats):
    mean = stats.sums / stats.count
    std = (stats.squares / stats.count - mean.square()).clamp_min(0).sqrt()
    class_abs_mean = stats.class_abs / stats.class_counts[None, :, None].clamp_min(1)
    global_abs = stats.class_abs.sum(1) / stats.class_counts.sum()
    lift = class_abs_mean / global_abs[:, None, :].clamp_min(1e-8)
    winner = lift.argmax(1)
    class_signed_mean = stats.class_signed / stats.class_counts[None, :, None].clamp_min(1)
    standardized = (class_signed_mean - mean[:, None, :]) / std[:, None, :].clamp_min(1e-8)
    positive = standardized.argmax(1)
    negative = standardized.argmin(1)
    return mean, std, lift, winner, class_signed_mean, positive, negative


def gather_class(values, indices):
    return torch.gather(values, 1, indices[:, None, :]).squeeze(1)


def choose_units(use_score, lift, gap, count):
    candidates = []
    each = max(count // 3, 1)
    for score in (use_score, lift * use_score.sqrt(), gap.abs() * use_score.sqrt()):
        for unit in torch.topk(score, min(each * 2, len(score))).indices.tolist():
            if unit not in candidates:
                candidates.append(unit)
            if len(candidates) >= count:
                return candidates
    return candidates


@torch.inference_mode()
def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = np.load(args.corpus, mmap_mode="r")
    required = args.discovery_rows + args.heldout_rows
    if required > len(corpus):
        raise ValueError(f"need {required} rows; corpus has {len(corpus)}")
    discovery_rows = corpus[:args.discovery_rows]
    heldout_rows = corpus[args.discovery_rows:required]
    discovery_context_rows = torch.from_numpy(np.asarray(discovery_rows).copy())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    model = load_full_model(args.device)
    layers, units = model.config.n_layer, model.config.expansion_factor * model.config.n_embd
    class_lookup = torch.tensor([classify_token(text) for text in texts], device=args.device)
    discovery = Stats(layers, units, len(CLASSES), args.tail_size, args.device, True)
    heldout = Stats(layers, units, len(CLASSES), args.tail_size, args.device, False)
    collect(model, discovery_rows, class_lookup, discovery, args, "discovery")
    collect(model, heldout_rows, class_lookup, heldout, args, "heldout")

    dmean, dstd, dlifts, dwinner, dsigned, dpositive, dnegative = metric_tensors(discovery)
    _, hstd, hlifts, _, hsigned, _, _ = metric_tensors(heldout)
    frozen_lift = gather_class(hlifts, dwinner)
    positive_mean = gather_class(hsigned, dpositive)
    negative_mean = gather_class(hsigned, dnegative)
    signed_gap = (positive_mean - negative_mean) / hstd.clamp_min(1e-8)
    down_norm = torch.stack([block.mlp.Down.weight.float().norm(dim=0) for block in model.blocks])
    use_score = dstd * down_norm

    layer_results = []
    for layer in range(layers):
        active = use_score[layer] > use_score[layer].median()
        selected = choose_units(use_score[layer], frozen_lift[layer], signed_gap[layer],
                                args.selected_per_layer)
        selected_rows = []
        for unit in selected:
            pos_ids = discovery.top_indices[layer, :, unit].cpu().tolist()
            neg_ids = discovery.bottom_indices[layer, :, unit].cpu().tolist()
            selected_rows.append({
                "unit": unit,
                "preferred_class": CLASSES[int(dwinner[layer, unit])],
                "positive_class": CLASSES[int(dpositive[layer, unit])],
                "negative_class": CLASSES[int(dnegative[layer, unit])],
                "heldout_lift": round(float(frozen_lift[layer, unit]), 5),
                "heldout_signed_gap_sd": round(float(signed_gap[layer, unit]), 5),
                "product_std": round(float(dstd[layer, unit]), 5),
                "down_norm": round(float(down_norm[layer, unit]), 5),
                "use_score": round(float(use_score[layer, unit]), 5),
                "positive_values": [round(float(v), 5) for v in discovery.top_values[layer, :, unit]],
                "positive_contexts": [context_for(i, discovery_context_rows,
                                                  tokenizer, args.sequence_length) for i in pos_ids],
                "negative_values": [round(float(v), 5) for v in discovery.bottom_values[layer, :, unit]],
                "negative_contexts": [context_for(i, discovery_context_rows,
                                                  tokenizer, args.sequence_length) for i in neg_ids],
            })
        channels = [{
            "unit": unit,
            "preferred_class": CLASSES[int(dwinner[layer, unit])],
            "positive_class": CLASSES[int(dpositive[layer, unit])],
            "negative_class": CLASSES[int(dnegative[layer, unit])],
            "heldout_lift": round(float(frozen_lift[layer, unit]), 5),
            "heldout_signed_gap_sd": round(float(signed_gap[layer, unit]), 5),
            "product_std": round(float(dstd[layer, unit]), 5),
            "down_norm": round(float(down_norm[layer, unit]), 5),
            "use_score": round(float(use_score[layer, unit]), 5),
        } for unit in range(units)]
        result = {
            "layer": layer,
            "active_half_lift_median": round(float(frozen_lift[layer, active].median()), 5),
            "active_half_fraction_lift_ge_2": round(float((frozen_lift[layer, active] >= 2).float().mean()), 5),
            "active_half_signed_gap_median": round(float(signed_gap[layer, active].median()), 5),
            "active_half_fraction_gap_ge_1": round(float((signed_gap[layer, active] >= 1).float().mean()), 5),
            "use_score_effective_support": round(float(
                use_score[layer].sum().square() / use_score[layer].square().sum().clamp_min(1e-12)), 3),
            "selected": selected_rows,
            "channels": channels,
        }
        layer_results.append(result)
        (args.output / f"layer_{layer}.json").write_text(json.dumps(result, indent=2))
        print({key: value for key, value in result.items() if key not in ("selected", "channels")}, flush=True)

    summary = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "parameters": 545_902_902,
        "corpus": str(args.corpus),
        "discovery_tokens_per_layer": discovery.count,
        "heldout_tokens_per_layer": heldout.count,
        "classes": list(CLASSES),
        "layers": [{key: value for key, value in result.items()
                    if key not in ("selected", "channels")} for result in layer_results],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
