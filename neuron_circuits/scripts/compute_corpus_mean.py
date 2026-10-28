#!/usr/bin/env python3
"""Estimate a position-specific MLP mean from a large streaming text corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.hf_adapter import GatedMLPMaskAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--windows", type=int, default=50_000)
    parser.add_argument("--positions", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def window_batches(stream, tokenizer, positions: int, batch_size: int, limit: int):
    """Yield deterministic non-overlapping token windows from streamed records."""
    buffer: list[int] = []
    batch: list[list[int]] = []
    produced = 0
    eos = tokenizer.eos_token_id
    for record in stream:
        text = record.get("text", "")
        if not text.strip():
            continue
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        if eos is not None:
            buffer.append(eos)
        while len(buffer) >= positions and produced < limit:
            batch.append(buffer[:positions])
            del buffer[:positions]
            produced += 1
            if len(batch) == batch_size:
                yield torch.tensor(batch, dtype=torch.long)
                batch = []
        if produced >= limit:
            break
    if batch:
        yield torch.tensor(batch, dtype=torch.long)
    if produced < limit:
        raise RuntimeError(f"stream ended after {produced:,} windows; requested {limit:,}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    adapter = GatedMLPMaskAdapter(model)
    stream = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
        streaming=True,
    )

    accumulator = torch.zeros(
        adapter.mask_shape(args.positions), dtype=torch.float32, device=args.device
    )
    seen = 0
    with torch.no_grad():
        for batch_index, tokens in enumerate(
            window_batches(stream, tokenizer, args.positions, args.batch_size, args.windows), 1
        ):
            tokens = tokens.to(args.device)
            traced = adapter.forward(tokens)
            for layer, activations in enumerate(traced.activations):
                accumulator[layer] += activations.float().sum(dim=0)
            seen += tokens.shape[0]
            if batch_index % 100 == 0 or seen == args.windows:
                print(f"mean baseline: {seen:,}/{args.windows:,} windows", flush=True)

    mean = (accumulator / seen).cpu()
    metadata = {
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "streaming": True,
        "seed": args.seed,
        "windows": seen,
        "positions": args.positions,
        "tokens": seen * args.positions,
        "batch_size": args.batch_size,
        "mask_shape": list(mean.shape),
        "dtype": "float32",
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "metadata": metadata}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"saved {seen:,} windows / {metadata['tokens']:,} tokens to {args.output} "
        f"in {metadata['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
