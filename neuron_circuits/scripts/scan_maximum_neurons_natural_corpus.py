#!/usr/bin/env python3
"""Scan a large natural-text corpus for extrema of maximum-circuit neurons."""

from __future__ import annotations

import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compute_corpus_mean import window_batches
from neuron_circuits.hf_adapter import GatedMLPMaskAdapter


DEFAULT_UNITS = ((0, 2905), (0, 2206), (21, 2550), (22, 30))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokens", type=int, default=10_000_000)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context", type=int, default=24)
    parser.add_argument("--heap-size", type=int, default=1000)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/maximum_natural_corpus/examples.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add_to_heap(heap, key: float, serial: int, item: dict, limit: int) -> None:
    entry = (key, serial, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, entry)


def readable_context(tokenizer, item: dict, radius: int) -> dict | None:
    ids = item["token_ids"]
    pos = item["position"]
    start, stop = pos - radius, pos + radius + 1
    context_ids = ids[start:stop]
    text = tokenizer.decode(context_ids, clean_up_tokenization_spaces=False)
    text = re.sub(r"\s+", " ", text).strip()
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 16 or "<|" in text or len(text) < 100:
        return None
    return {
        "activation": item["activation"],
        "target_token": tokenizer.decode([ids[pos]], clean_up_tokenization_spaces=False),
        "target_index": radius,
        "text": text,
        "tokens": [
            tokenizer.decode([token], clean_up_tokenization_spaces=False)
            for token in context_ids
        ],
        "activations": item["activations"][start:stop],
        "source_window": item["source_window"],
        "source_position": pos,
    }


def finalize(tokenizer, heap, radius: int, count: int, positive: bool) -> list[dict]:
    ordered = sorted((row[2] for row in heap), key=lambda x: x["activation"], reverse=positive)
    output, seen = [], set()
    for item in ordered:
        context = readable_context(tokenizer, item, radius)
        if context is None:
            continue
        fingerprint = context["text"].lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        output.append(context)
        if len(output) == count:
            break
    return output


def raw_extreme(tokenizer, heap, radius: int, positive: bool) -> dict:
    """Return the literal corpus extreme, even when its context fails readability filters."""
    item = max((row[2] for row in heap), key=lambda x: x["activation"]) if positive else min(
        (row[2] for row in heap), key=lambda x: x["activation"]
    )
    ids, pos = item["token_ids"], item["position"]
    start, stop = pos - radius, pos + radius + 1
    return {
        "activation": item["activation"],
        "target_token": tokenizer.decode([ids[pos]], clean_up_tokenization_spaces=False),
        "text": re.sub(
            r"\s+",
            " ",
            tokenizer.decode(ids[start:stop], clean_up_tokenization_spaces=False),
        ).strip(),
        "source_window": item["source_window"],
        "source_position": pos,
    }


def main() -> None:
    args = parse_args()
    if args.positions <= 2 * args.context:
        raise ValueError("positions must exceed twice the context radius")
    windows = (args.tokens + args.positions - 1) // args.positions
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    adapter = GatedMLPMaskAdapter(model)
    units_by_layer = defaultdict(list)
    for layer, channel in DEFAULT_UNITS:
        units_by_layer[layer].append(channel)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer, channels in units_by_layer.items():
        def hook(_module, hook_args, *, layer=layer, channels=tuple(channels)):
            captured[layer] = hook_args[0][..., list(channels)].detach().float().cpu()
        handles.append(adapter.output_projections[layer].register_forward_pre_hook(hook))

    stats = {
        f"L{layer}/N{channel}": {
            "layer": layer,
            "channel": channel,
            "count": 0,
            "sum": 0.0,
            "sum_squares": 0.0,
            "positive": [],
            "negative": [],
        }
        for layer, channel in DEFAULT_UNITS
    }
    stream = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    seen_windows = 0
    try:
        with torch.inference_mode():
            for batch_index, tokens in enumerate(
                window_batches(stream, tokenizer, args.positions, args.batch_size, windows), 1
            ):
                captured.clear()
                model(input_ids=tokens.to(args.device), use_cache=False)
                batch_count = tokens.shape[0]
                usable = torch.ones_like(tokens, dtype=torch.bool)
                usable[:, : args.context] = False
                usable[:, -args.context :] = False
                if tokenizer.eos_token_id is not None:
                    usable &= tokens != tokenizer.eos_token_id
                for layer, channels in units_by_layer.items():
                    layer_values = captured[layer]
                    for local_index, channel in enumerate(channels):
                        unit = stats[f"L{layer}/N{channel}"]
                        values = layer_values[..., local_index]
                        selected = values[usable]
                        unit["count"] += selected.numel()
                        unit["sum"] += selected.double().sum().item()
                        unit["sum_squares"] += selected.double().square().sum().item()
                        for positive in (True, False):
                            scored = values.masked_fill(~usable, -torch.inf if positive else torch.inf)
                            k = min(16, int(usable.sum()))
                            top = torch.topk(scored.flatten(), k=k, largest=positive)
                            for value, flat_index in zip(top.values.tolist(), top.indices.tolist()):
                                row, pos = divmod(flat_index, args.positions)
                                item = {
                                    "activation": float(value),
                                    "token_ids": tokens[row].tolist(),
                                    "activations": values[row].tolist(),
                                    "position": pos,
                                    "source_window": seen_windows + row,
                                }
                                serial = (seen_windows + row) * args.positions + pos
                                heap = unit["positive" if positive else "negative"]
                                add_to_heap(heap, value if positive else -value, serial, item, args.heap_size)
                seen_windows += batch_count
                if batch_index % 100 == 0 or seen_windows == windows:
                    print(
                        f"scanned {seen_windows * args.positions:,}/{windows * args.positions:,} tokens",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    output_units = []
    for name, unit in stats.items():
        count = unit["count"]
        mean = unit["sum"] / count
        variance = max(0.0, unit["sum_squares"] / count - mean * mean)
        output_units.append({
            "unit": name,
            "layer": unit["layer"],
            "channel": unit["channel"],
            "activation_count": count,
            "mean": mean,
            "standard_deviation": variance ** 0.5,
            "raw_maximum": raw_extreme(tokenizer, unit["positive"], args.context, True),
            "raw_minimum": raw_extreme(tokenizer, unit["negative"], args.context, False),
            "top_positive": finalize(tokenizer, unit["positive"], args.context, args.examples, True),
            "top_negative": finalize(tokenizer, unit["negative"], args.context, args.examples, False),
        })
    output = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "streaming": True,
        "windows": seen_windows,
        "window_tokens": args.positions,
        "tokens_forwarded": seen_windows * args.positions,
        "tokens_considered_for_extrema": sum(x["activation_count"] for x in output_units) // len(output_units),
        "context_tokens_each_side": args.context,
        "elapsed_seconds": time.perf_counter() - started,
        "units": output_units,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output} in {output['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
