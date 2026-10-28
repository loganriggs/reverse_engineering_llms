#!/usr/bin/env python3
"""Test whether sparse MLP counterfactual control survives location exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.hf_adapter import GatedMLPMaskAdapter
from neuron_circuits.language_tasks import language_counterfactual_pairs
from neuron_circuits.restrictions import (
    CONDITIONS,
    eligible_instance_mask,
    restricted_ranking,
)
try:
    from scripts.run_counterfactual_patching import (
        counts,
        evaluate,
        one_token,
        patch_mask,
        refine_first_crossing,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    from run_counterfactual_patching import (
        counts,
        evaluate,
        one_token,
        patch_mask,
        refine_first_crossing,
    )


SOURCE_ROOT = Path("artifacts/counterfactual_patching")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--pair-metadata", type=Path, default=SOURCE_ROOT / "metadata.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts/restricted_counterfactuals"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=53)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()

    source_metadata = json.loads(args.pair_metadata.read_text(encoding="utf-8"))
    pair_lookup = {pair.pair_id: pair for pair in language_counterfactual_pairs()}
    selected_ids = [item["pair_id"] for item in source_metadata["selected_pairs"]]
    missing = [pair_id for pair_id in selected_ids if pair_id not in pair_lookup]
    if missing:
        raise RuntimeError(f"selected pairs no longer exist: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    adapter = GatedMLPMaskAdapter(model)

    rows = []
    saved = {}
    condition_metadata = {}
    for pair_index, pair_id in enumerate(selected_ids, 1):
        pair = pair_lookup[pair_id]
        base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(args.device)
        donor_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(args.device)
        base_answer_id = one_token(tokenizer, pair.base.answer_text)
        donor_answer_id = one_token(tokenizer, pair.source.answer_text)
        if base_answer_id is None or donor_answer_id is None:
            raise RuntimeError(f"answers are not single tokens for {pair_id}")

        with torch.no_grad():
            donor_trace = adapter.forward(donor_ids)
        donor_activations = torch.stack([item[0].float() for item in donor_trace.activations])
        donor_logits = donor_trace.logits[0].float()
        base_trace = adapter.forward(base_ids, retain_activation_grads=True)
        base_logits = base_trace.logits[0].float()
        objective = base_logits[donor_answer_id] - base_logits[base_answer_id]
        gradients = torch.autograd.grad(objective, base_trace.activations)
        base_activations = torch.stack([item[0].detach().float() for item in base_trace.activations])
        gradient = torch.stack([item[0].detach().float() for item in gradients])
        scores = gradient * (donor_activations - base_activations)
        shape = adapter.mask_shape(base_ids.shape[1])

        for condition in CONDITIONS:
            allowed = eligible_instance_mask(shape, condition, args.device)
            ranking = restricted_ranking(scores, allowed)
            eligible_total = int(ranking.numel())
            cache = {}

            def evaluate_at(patched: int):
                row = evaluate(
                    adapter,
                    tokenizer,
                    pair,
                    base_ids,
                    patch_mask(ranking, patched, shape, "instance"),
                    donor_activations,
                    base_logits,
                    donor_logits,
                    base_answer_id,
                    donor_answer_id,
                    "counterfactual_gradient",
                    "instance",
                    patched,
                )
                row.update({
                    "condition": condition,
                    "eligible_total": eligible_total,
                    "eligible_patched_fraction": patched / eligible_total,
                })
                return row

            for patched in counts(eligible_total):
                cache[patched] = evaluate_at(patched)
            refine_first_crossing(cache, "pairwise_flip", evaluate_at, refinements=12)
            refine_first_crossing(cache, "target_top1", evaluate_at, refinements=12)
            rows.extend(cache[count] for count in sorted(cache))
            successes = [count for count, row in cache.items() if row["target_top1"]]
            if successes:
                crossover = min(successes)
                saved[f"{pair_id}__{condition}__top1"] = (
                    ranking[:crossover].cpu().numpy().astype(np.int32)
                )
            condition_metadata.setdefault(condition, {
                "excluded_layers": [],
                "excludes_final_position": False,
            })
        print(f"restricted {pair_index}/{len(selected_ids)}: {pair_id}", flush=True)

    condition_metadata["exclude_final_layer"] = {
        "excluded_layers": [23], "excludes_final_position": False,
    }
    condition_metadata["exclude_final_four_layers"] = {
        "excluded_layers": [20, 21, 22, 23], "excludes_final_position": False,
    }
    condition_metadata["exclude_final_position"] = {
        "excluded_layers": [], "excludes_final_position": True,
    }
    condition_metadata["exclude_final_four_layers_and_position"] = {
        "excluded_layers": [20, 21, 22, 23], "excludes_final_position": True,
    }
    with (args.output / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(args.output / "crossover_indices.npz", **saved)
    metadata = {
        "model": args.model,
        "seed": args.seed,
        "device": args.device,
        "dtype": str(dtype),
        "source_pair_metadata": str(args.pair_metadata),
        "selected_pair_ids": selected_ids,
        "conditions": condition_metadata,
        "ranking": "original signed gradient × donor-minus-base activation, filtered to eligible units",
        "criterion": "donor answer is full-vocabulary top-1",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(rows):,} evaluations in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
