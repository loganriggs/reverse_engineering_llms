#!/usr/bin/env python3
"""Find ordinary WikiText contexts that maximally activate sparse-circuit neurons."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.hf_adapter import GatedMLPMaskAdapter
from compute_corpus_mean import window_batches


SOURCE = Path("artifacts/natural_neuron_examples/examples.json")
OUTPUT = Path("artifacts/corpus_neuron_examples/examples.json")
WINDOWS = 5_000
POSITIONS = 64
BATCH_SIZE = 64
PER_BATCH = 12
CONTEXT = 14


def selected_units(source):
    output = []
    for family in ("brackets", "arithmetic"):
        for neuron in source["families"][family]["neurons"]:
            output.append({
                "family": family,
                "unit": neuron["unit"],
                "layer": neuron["layer"],
                "channel": neuron["channel"],
                "counterfactual_position": neuron["position"],
                "recurrence_pairs": neuron["pairs"],
            })
    return output


def push_candidates(heap, values, tokens, batch_offset, positive, eos_id):
    usable = values.clone()
    usable[:, :CONTEXT] = -torch.inf if positive else torch.inf
    usable[:, -CONTEXT:] = -torch.inf if positive else torch.inf
    if eos_id is not None:
        usable[tokens == eos_id] = -torch.inf if positive else torch.inf
    flat = usable.flatten()
    chosen = torch.topk(flat, k=PER_BATCH, largest=positive)
    for value, raw_index in zip(chosen.values.tolist(), chosen.indices.tolist()):
        row, position = divmod(raw_index, POSITIONS)
        item = {
            "activation": float(value),
            "window_index": batch_offset + row,
            "position": position,
            "token_ids": tokens[row].tolist(),
            "window_activations": values[row].tolist(),
        }
        key = value if positive else -value
        if len(heap) < 200:
            heapq.heappush(heap, (key, batch_offset * POSITIONS + raw_index, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, batch_offset * POSITIONS + raw_index, item))


def natural_context(tokenizer, item):
    ids = item["token_ids"]
    position = item["position"]
    start, stop = position - CONTEXT, position + CONTEXT + 1
    context_ids = ids[start:stop]
    text = tokenizer.decode(context_ids, clean_up_tokenization_spaces=False).replace("\n", " ").strip()
    target = tokenizer.decode([ids[position]], clean_up_tokenization_spaces=False)
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 10 or "<|" in text or text.count("=") > 4:
        return None
    return {
        "activation": item["activation"],
        "target_token": target,
        "target_offset": CONTEXT,
        "text": text,
        "tokens": [tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in context_ids],
        "activations": item["window_activations"][start:stop],
        "token_ids": context_ids,
        "source_window": item["window_index"],
    }


def finalize(tokenizer, heap, positive):
    ordered = sorted((entry[2] for entry in heap), key=lambda row: row["activation"], reverse=positive)
    result = []
    seen = set()
    for item in ordered:
        context = natural_context(tokenizer, item)
        if context is None:
            continue
        fingerprint = re.sub(r"\s+", " ", context["text"].lower())
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.append(context)
        if len(result) == 5:
            break
    return result


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    units = selected_units(source)
    model_name = source["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    adapter = GatedMLPMaskAdapter(model)
    stream = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    heaps = {unit["unit"]: {"positive": [], "negative": []} for unit in units}
    seen = 0
    with torch.no_grad():
        for batch_number, tokens in enumerate(window_batches(stream, tokenizer, POSITIONS, BATCH_SIZE, WINDOWS), 1):
            count = tokens.shape[0]
            device_tokens = tokens.to(device)
            trace = adapter.forward(device_tokens)
            for unit in units:
                values = trace.activations[unit["layer"]][:, :, unit["channel"]].detach().float().cpu()
                push_candidates(heaps[unit["unit"]]["positive"], values, tokens, seen, True, tokenizer.eos_token_id)
                push_candidates(heaps[unit["unit"]]["negative"], values, tokens, seen, False, tokenizer.eos_token_id)
            seen += count
            if batch_number % 20 == 0 or seen == WINDOWS:
                print(f"scanned {seen:,}/{WINDOWS:,} WikiText windows", flush=True)

    output = {
        "model": model_name,
        "dataset": "Salesforce/wikitext wikitext-103-raw-v1 train",
        "windows": seen,
        "tokens_scanned": seen * POSITIONS,
        "context_tokens_each_side": CONTEXT,
        "units": [],
    }
    for unit in units:
        output["units"].append({
            **unit,
            "top_positive": finalize(tokenizer, heaps[unit["unit"]]["positive"], True),
            "top_negative": finalize(tokenizer, heaps[unit["unit"]]["negative"], False),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
