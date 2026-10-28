#!/usr/bin/env python3
"""Operational monosemanticity probe for the final MLP on natural text."""

from __future__ import annotations

import argparse
from collections import Counter
import heapq
import json
import math
from pathlib import Path
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_corpus_mean import window_batches
from neuron_circuits.hf_adapter import GatedMLPMaskAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--screen-tokens", type=int, default=1_000_000)
    parser.add_argument("--example-tokens", type=int, default=2_000_000)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--neurons", type=int, default=12)
    parser.add_argument("--events-per-split", type=int, default=128)
    parser.add_argument("--context", type=int, default=18)
    parser.add_argument("--output", type=Path, default=Path("artifacts/last_mlp_monosemanticity/results.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def stream(args):
    return load_dataset(args.dataset, args.dataset_config, split="train", streaming=True)


def moments_screen(args, model, projection, tokenizer):
    width = projection.in_features
    sums = torch.zeros(width, dtype=torch.float64)
    squares = torch.zeros(width, dtype=torch.float64)
    fourths = torch.zeros(width, dtype=torch.float64)
    maxima = torch.full((width,), -torch.inf)
    minima = torch.full((width,), torch.inf)
    count = 0
    captured = []
    handle = projection.register_forward_pre_hook(lambda _m, x: captured.append(x[0].detach()))
    windows = math.ceil(args.screen_tokens / args.positions)
    try:
        with torch.inference_mode():
            for batch_no, tokens in enumerate(window_batches(stream(args), tokenizer, args.positions, args.batch_size, windows), 1):
                captured.clear()
                model.model(input_ids=tokens.to(args.device), use_cache=False)
                values = captured[0].float().reshape(-1, width)
                # Reduce on the GPU; copying every token × neuron activation to CPU
                # would dominate both memory traffic and runtime.
                sums += values.sum(0).double().cpu()
                squares += values.square().sum(0).double().cpu()
                fourths += values.pow(4).sum(0).double().cpu()
                maxima = torch.maximum(maxima, values.max(0).values.cpu())
                minima = torch.minimum(minima, values.min(0).values.cpu())
                count += values.shape[0]
                if batch_no % 50 == 0 or count >= args.screen_tokens:
                    print(f"screened {count:,}/{windows * args.positions:,} activations per neuron", flush=True)
    finally:
        handle.remove()
    mean = sums / count
    variance = (squares / count - mean.square()).clamp_min(1e-12)
    # Raw fourth moment is sufficient for ranking sparse, heavy-tailed activations.
    kurtosis_proxy = fourths / count / variance.square()
    max_z = (maxima - mean.float()) / variance.sqrt().float()
    return {
        "count": count,
        "mean": mean.float(),
        "std": variance.sqrt().float(),
        "max": maxima,
        "min": minima,
        "kurtosis_proxy": kurtosis_proxy.float(),
        "max_z": max_z,
    }


def choose_neurons(stats, number):
    valid = (stats["std"] > 0.02) & torch.isfinite(stats["max_z"]) & (stats["max"] > 0)
    score = stats["max_z"].clone()
    score[~valid] = -torch.inf
    # Half extremely selective and half drawn across the active distribution gives a spectrum.
    high_count = max(1, number // 2)
    high = torch.topk(score, high_count).indices.tolist()
    active = torch.where(valid)[0]
    ordered = active[torch.argsort(score[active])]
    quantile_count = number - high_count
    quantiles = []
    if quantile_count:
        for index in range(quantile_count):
            offset = round((index + 1) * (len(ordered) - 1) / (quantile_count + 1))
            quantiles.append(int(ordered[offset]))
    return list(dict.fromkeys(high + quantiles))[:number]


def heap_add(heap, value, serial, item, limit, positive=True):
    key = value if positive else -value
    entry = (key, serial, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, entry)


def collect_events(args, model, projection, tokenizer, channels):
    activation_box, residual_box = [], []
    act_handle = projection.register_forward_pre_hook(lambda _m, x: activation_box.append(x[0].detach()))
    norm_handle = model.model.norm.register_forward_pre_hook(lambda _m, x: residual_box.append(x[0].detach()))
    heaps = {c: {"a_pos": [], "b_pos": [], "a_neg": [], "b_neg": []} for c in channels}
    next_token_counts = Counter()
    next_token_total = 0
    windows = math.ceil(args.example_tokens / args.positions)
    seen = 0
    try:
        with torch.inference_mode():
            for batch_no, tokens in enumerate(window_batches(stream(args), tokenizer, args.positions, args.batch_size, windows), 1):
                activation_box.clear(); residual_box.clear()
                model.model(input_ids=tokens.to(args.device), use_cache=False)
                acts = activation_box[0][..., channels].float().cpu()
                residuals = residual_box[0].float().cpu()
                batch, positions, _ = acts.shape
                valid = torch.ones((batch, positions), dtype=torch.bool)
                valid[:, :args.context] = False
                valid[:, -args.context - 1:] = False
                if tokenizer.eos_token_id is not None:
                    valid &= tokens != tokenizer.eos_token_id
                    valid &= torch.roll(tokens != tokenizer.eos_token_id, shifts=-1, dims=1)
                ordinary_next = tokens[:, 1:][valid[:, :-1]].tolist()
                next_token_counts.update(ordinary_next)
                next_token_total += len(ordinary_next)
                for local, channel in enumerate(channels):
                    values = acts[..., local]
                    for positive in (True, False):
                        ranked = values.masked_fill(~valid, -torch.inf if positive else torch.inf)
                        top = torch.topk(ranked.flatten(), k=min(12, int(valid.sum())), largest=positive)
                        for value, flat in zip(top.values.tolist(), top.indices.tolist()):
                            row, pos = divmod(flat, positions)
                            split = "a" if (seen + row) % 2 == 0 else "b"
                            item = {
                                "activation": float(value),
                                "token_ids": tokens[row].tolist(),
                                "position": pos,
                                "residual": residuals[row, pos].tolist(),
                                "next_token_id": int(tokens[row, pos + 1]),
                                "window": seen + row,
                            }
                            heap_add(
                                heaps[channel][f"{split}_{'pos' if positive else 'neg'}"],
                                value, (seen + row) * positions + pos, item,
                                args.events_per_split, positive,
                            )
                seen += batch
                if batch_no % 50 == 0 or seen == windows:
                    print(f"searched {seen * args.positions:,}/{windows * args.positions:,} example tokens", flush=True)
    finally:
        act_handle.remove(); norm_handle.remove()
    return heaps, next_token_counts, next_token_total


def distribution(events, field):
    counts = Counter(item[field] for item in events)
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()}


def js_similarity(left, right):
    keys = set(left) | set(right)
    midpoint = {k: (left.get(k, 0) + right.get(k, 0)) / 2 for k in keys}
    def kl(p):
        return sum(value * math.log2(value / midpoint[key]) for key, value in p.items() if value)
    return 1.0 - (kl(left) + kl(right)) / 2


def simpson_concentration(probabilities):
    return sum(value * value for value in probabilities.values())


def token_text(tokenizer, token_id):
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def readable_event(tokenizer, item, radius):
    ids, pos = item["token_ids"], item["position"]
    start, stop = pos - radius, pos + radius + 1
    text = re.sub(r"\s+", " ", tokenizer.decode(ids[start:stop], clean_up_tokenization_spaces=False)).strip()
    return {
        "activation": item["activation"],
        "token": token_text(tokenizer, ids[pos]),
        "next_token": token_text(tokenizer, item["next_token_id"]),
        "text": text,
        "window": item["window"],
    }


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype) * weight


def analyze_neuron(args, model, tokenizer, channel, stats, heap_group, unembedding, direct, corpus_counts, corpus_total):
    projection = model.model.layers[-1].mlp.down_proj
    direction = projection.weight[:, channel].detach().float().cpu()
    promoted = torch.topk(direct, 30)
    suppressed = torch.topk(direct, 30, largest=False)
    direct_sorted = torch.sort(direct).values
    promoted_100 = set(torch.topk(direct, 100).indices.tolist())
    promoted_1000 = set(torch.topk(direct, 1000).indices.tolist())
    baseline_100 = sum(corpus_counts[token] for token in promoted_100) / corpus_total
    baseline_1000 = sum(corpus_counts[token] for token in promoted_1000) / corpus_total
    norm_weight = model.model.norm.weight.detach().float().cpu()
    eps = model.model.norm.variance_epsilon

    def exact_metrics(events):
        percentiles, deltas, ranks = [], [], []
        for item in events:
            h = item["activation"]
            residual = torch.tensor(item["residual"], dtype=torch.float32)
            ablated = residual - h * direction
            full_hidden = rms_norm(residual, norm_weight, eps)
            ablated_hidden = rms_norm(ablated, norm_weight, eps)
            target_weight = unembedding[item["next_token_id"]]
            delta = float(target_weight @ (full_hidden - ablated_hidden))
            deltas.append(delta)
            effect = float(direct[item["next_token_id"]])
            ascending_rank = int(torch.searchsorted(direct_sorted, torch.tensor(effect)))
            percentile = ascending_rank / len(direct_sorted)
            ranks.append(len(direct_sorted) - ascending_rank)
            percentiles.append(percentile)
        hit_100 = sum(rank <= 100 for rank in ranks) / len(ranks)
        hit_1000 = sum(rank <= 1000 for rank in ranks) / len(ranks)
        return {
            "mean_decoder_percentile_of_actual_next_token": sum(percentiles) / len(percentiles),
            "median_decoder_rank_of_actual_next_token": float(torch.tensor(ranks).median()),
            "decoder_hit_rate_at_100": hit_100,
            "decoder_hit_rate_at_1000": hit_1000,
            "corpus_baseline_rate_at_100": baseline_100,
            "corpus_baseline_rate_at_1000": baseline_1000,
            "enrichment_at_100": hit_100 / baseline_100 if baseline_100 else None,
            "enrichment_at_1000": hit_1000 / baseline_1000 if baseline_1000 else None,
            "causal_help_rate": sum(delta > 0 for delta in deltas) / len(deltas),
            "mean_exact_next_token_logit_effect": sum(deltas) / len(deltas),
            "worst_exact_next_token_logit_effect": min(deltas),
        }

    splits = {}
    for split in ("a", "b"):
        positive = sorted((x[2] for x in heap_group[f"{split}_pos"]), key=lambda x: x["activation"], reverse=True)
        negative = sorted((x[2] for x in heap_group[f"{split}_neg"]), key=lambda x: x["activation"])
        # token_ids[position] is deliberately computed here to keep stored events compact.
        current = Counter(x["token_ids"][x["position"]] for x in positive)
        current = {key: value / len(positive) for key, value in current.items()}
        following = Counter(x["next_token_id"] for x in positive)
        following = {key: value / len(positive) for key, value in following.items()}
        splits[split] = {
            "events": positive,
            "current_distribution": current,
            "following_distribution": following,
            "lexical_concentration": simpson_concentration(current),
            "outcome_concentration": simpson_concentration(following),
            "exact": exact_metrics(positive),
            "top_positive_examples": [readable_event(tokenizer, x, args.context) for x in positive[:8]],
            "top_negative_examples": [readable_event(tokenizer, x, args.context) for x in negative[:5]],
        }
    lexical_stability = js_similarity(splits["a"]["current_distribution"], splits["b"]["current_distribution"])
    outcome_stability = js_similarity(splits["a"]["following_distribution"], splits["b"]["following_distribution"])
    lexical_concentration = math.sqrt(splits["a"]["lexical_concentration"] * splits["b"]["lexical_concentration"])
    outcome_concentration = math.sqrt(splits["a"]["outcome_concentration"] * splits["b"]["outcome_concentration"])
    output_agreement = max(0.0, 2 * (splits["b"]["exact"]["mean_decoder_percentile_of_actual_next_token"] - 0.5))
    causal = splits["b"]["exact"]["causal_help_rate"]
    composite = (max(outcome_concentration, 1e-9) * max(outcome_stability, 1e-9) * max(output_agreement, 1e-9) * max(causal, 1e-9)) ** 0.25
    return {
        "neuron": f"L{len(model.model.layers)-1}/N{channel}",
        "channel": channel,
        "screen": {key: float(stats[key][channel]) for key in ("mean", "std", "max", "min", "max_z", "kurtosis_proxy")},
        "decoder": {
            "promoted": [{"token": token_text(tokenizer, int(i)), "effect": float(v)} for v, i in zip(promoted.values, promoted.indices)],
            "suppressed": [{"token": token_text(tokenizer, int(i)), "effect": float(v)} for v, i in zip(suppressed.values, suppressed.indices)],
        },
        "activation_lexical_concentration": lexical_concentration,
        "split_half_lexical_stability": lexical_stability,
        "activation_outcome_concentration": outcome_concentration,
        "split_half_outcome_stability": outcome_stability,
        "heldout_output_agreement": output_agreement,
        "heldout_causal_help_rate": causal,
        "operational_monosemanticity_score": composite,
        "split_a": {k: v for k, v in splits["a"].items() if k not in ("events", "current_distribution", "following_distribution")},
        "split_b": {k: v for k, v in splits["b"].items() if k not in ("events", "current_distribution", "following_distribution")},
    }


def main():
    args = parse_args()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    projection = model.model.layers[-1].mlp.down_proj
    stats = moments_screen(args, model, projection, tokenizer)
    channels = choose_neurons(stats, args.neurons)
    print(f"selected final-layer channels: {channels}", flush=True)
    heaps, corpus_counts, corpus_total = collect_events(args, model, projection, tokenizer, channels)
    with torch.inference_mode():
        directions = projection.weight[:, channels].float()
        direct_effects = (model.lm_head.weight.float() @ directions).cpu()
    unembedding = model.lm_head.weight.detach().float().cpu()
    results = [
        analyze_neuron(
            args, model, tokenizer, c, stats, heaps[c], unembedding,
            direct_effects[:, index], corpus_counts, corpus_total,
        )
        for index, c in enumerate(channels)
    ]
    results.sort(key=lambda row: row["operational_monosemanticity_score"], reverse=True)
    output = {
        "model": args.model,
        "layer": len(model.model.layers) - 1,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "screen_tokens": stats["count"],
        "example_tokens": math.ceil(args.example_tokens / args.positions) * args.positions,
        "selection": "half highest positive max-z among active neurons; half activation-selectivity quantiles",
        "metric": {
            "name": "operational lexical-predictive monosemanticity score",
            "formula": "geometric mean(activation-outcome concentration, split-half outcome stability, held-out decoder agreement, held-out exact causal help rate)",
            "warning": "A predictive functional proxy, not a proof of monosemanticity; it rewards one stable output role and separately reports lexical recurrence.",
            "primary_cross_prediction": "held-out enrichment@100 = P(next token is in decoder top 100 | top activation) / ordinary-corpus baseline",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "neurons": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output} in {output['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
