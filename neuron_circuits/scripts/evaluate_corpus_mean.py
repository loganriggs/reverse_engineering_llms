#!/usr/bin/env python3
"""Evaluate retained-neuron masks using a large-corpus mean replacement."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.arithmetic import ArithmeticExample
from neuron_circuits.extraction import (
    evaluate_mask,
    importance_scores,
    mask_from_ranking,
    retained_counts,
    stable_descending_ranking,
)
from neuron_circuits.hf_adapter import GatedMLPMaskAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/arithmetic_pilot"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(metadata["model"])
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(metadata["model"], dtype=dtype).to(args.device).eval()
    adapter = GatedMLPMaskAdapter(model)
    payload = torch.load(args.baseline, map_location="cpu", weights_only=True)
    baseline = payload["mean"].to(args.device)
    baseline_metadata = payload["metadata"]
    if tuple(baseline.shape) != tuple(metadata["mask_shape"]):
        raise ValueError(
            f"baseline shape {tuple(baseline.shape)} does not match {tuple(metadata['mask_shape'])}"
        )

    number_ids = {}
    for number in range(10):
        encoded = tokenizer.encode(str(number), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{number} is not a single token")
        number_ids[number] = encoded[0]

    total = metadata["total_masked_activations"]
    counts = retained_counts(total, metadata["fractions"])
    examples = [ArithmeticExample(**item) for item in metadata["selected_examples"][: args.examples]]
    new_rows = []
    for index, example in enumerate(examples, 1):
        ids = tokenizer(example.prompt, return_tensors="pt").input_ids.to(args.device)
        answer_id = number_ids[example.answer]
        full_logits, scores, _ = importance_scores(
            adapter, ids, answer_id, difference_baseline=baseline
        )
        rankings = {
            "gradient_activation": stable_descending_ranking(scores["gradient_activation"]),
            "gradient_difference": stable_descending_ranking(scores["gradient_difference"]),
        }
        for method, ranking in rankings.items():
            phase = (
                "large_corpus_baseline"
                if method == "gradient_activation"
                else "large_corpus_baseline_adjusted"
            )
            for retained in counts:
                mask = mask_from_ranking(ranking, retained, tuple(metadata["mask_shape"]))
                result = evaluate_mask(
                    adapter,
                    ids,
                    mask,
                    full_logits,
                    example_id=example.example_id,
                    method=method,
                    answer=example.answer,
                    answer_token_id=answer_id,
                    number_token_ids=number_ids,
                    baseline=baseline,
                    baseline_name="corpus_mean",
                )
                new_rows.append(result.as_dict() | {"phase": phase})
        print(f"corpus-mean evaluation {index}/{len(examples)}", flush=True)

    evaluations_path = args.root / "evaluations.jsonl"
    existing = [
        json.loads(line)
        for line in evaluations_path.read_text().splitlines()
        if json.loads(line).get("phase")
        not in {"large_corpus_baseline", "large_corpus_baseline_adjusted"}
    ]
    temporary = evaluations_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in [*existing, *new_rows]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(evaluations_path)
    metadata["large_corpus_mean"] = baseline_metadata
    metadata["large_corpus_mean_evaluation_examples"] = len(examples)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"appended {len(new_rows)} corpus-mean evaluations", flush=True)


if __name__ == "__main__":
    main()
