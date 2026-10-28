#!/usr/bin/env python3
"""Measure whether the model reliably performs candidate list algorithms."""

from __future__ import annotations

import json
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.algorithmic_tasks import algorithm_examples


OUTPUT = Path("artifacts/hierarchical_circuits/capability_screen.json")
MODEL = "Qwen/Qwen2.5-0.5B"


SECOND_VALUE_SHOTS = (
    ((1, 3, 2, 0), 2), ((8, 4, 6, 2), 6), ((0, 9, 5, 7), 7),
    ((6, 1, 3, 8), 6), ((4, 2, 9, 5), 5), ((7, 0, 1, 6), 6),
)
SECOND_INDEX_SHOTS = (
    ((1, 3, 2, 0), 2), ((8, 4, 6, 2), 2), ((0, 9, 5, 7), 3),
    ((6, 1, 3, 8), 0), ((4, 2, 9, 5), 3), ((7, 0, 1, 6), 3),
    ((2, 8, 7, 4), 2), ((9, 5, 6, 3), 2),
)


def few_shot_prompt(example, algorithm):
    if algorithm == "second_largest_few_shot":
        header, shots = "Second largest:\n", SECOND_VALUE_SHOTS
    else:
        header, shots = "0-based index of second largest:\n", SECOND_INDEX_SHOTS
    demonstrations = "".join(f"[{','.join(map(str, values))}] -> {answer}\n" for values, answer in shots)
    return header + demonstrations + f"[{','.join(map(str, example.values))}] -> "


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    rng = random.Random(1)
    results = {}
    for condition in (
        "maximum", "minimum", "second_largest", "argmax", "second_argmax",
        "second_largest_few_shot", "second_argmax_few_shot",
    ):
        task = condition.removesuffix("_few_shot")
        examples = algorithm_examples(task)
        rng.shuffle(examples)
        examples = examples[:100]
        correct = 0
        for example in examples:
            prompt = few_shot_prompt(example, condition) if condition.endswith("_few_shot") else example.prompt
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                prediction = int(model(ids, use_cache=False).logits[0, -1].argmax())
            answer = tokenizer.encode(str(example.answer), add_special_tokens=False)
            if len(answer) != 1:
                raise ValueError(f"answer is not one token: {example.answer}")
            correct += prediction == answer[0]
        results[condition] = {"correct": correct, "total": len(examples), "accuracy": correct / len(examples)}
        print(f"{condition}: {correct}/{len(examples)}", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"model": MODEL, "seed": 1, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
