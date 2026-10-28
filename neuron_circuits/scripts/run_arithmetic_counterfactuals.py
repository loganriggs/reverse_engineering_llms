#!/usr/bin/env python3
"""Run strict location-restricted MLP counterfactual patching on arithmetic."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.arithmetic_counterfactuals import arithmetic_counterfactual_pairs
from neuron_circuits.hf_adapter import GatedMLPMaskAdapter
from neuron_circuits.restrictions import eligible_instance_mask, restricted_ranking
from run_counterfactual_patching import counts, patch_mask, refine_first_crossing


OUTPUT = Path("artifacts/arithmetic_counterfactuals")
OPERATIONS = ("add", "subtract", "multiply")
CONDITION = "exclude_final_four_layers_and_position"
PAIRS_PER_OPERATION = 12


def answer_id(tokenizer, answer: int) -> int:
    ids = tokenizer.encode(str(answer), add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"answer {answer} is not a single token")
    return ids[0]


def evaluate(
    adapter, tokenizer, pair, base_ids, mask, donor_activations,
    base_logits, donor_logits, base_answer_id, donor_answer_id, patched,
):
    with torch.no_grad():
        logits = adapter.forward(base_ids, mask, baseline=donor_activations).logits[0].float()
    log_prob = F.log_softmax(logits, dim=-1)
    base_log = F.log_softmax(base_logits, dim=-1)
    donor_log = F.log_softmax(donor_logits, dim=-1)
    prediction = int(logits.argmax())
    return {
        "pair_id": pair.pair_id,
        "operation": pair.family,
        "condition": CONDITION,
        "method": "counterfactual_gradient",
        "resolution": "instance",
        "patched": patched,
        "pairwise_margin": float(logits[donor_answer_id] - logits[base_answer_id]),
        "pairwise_flip": bool(logits[donor_answer_id] >= logits[base_answer_id]),
        "target_top1": prediction == donor_answer_id,
        "predicted_token_id": prediction,
        "predicted_text": tokenizer.decode([prediction]),
        "target_probability": float(log_prob[donor_answer_id].exp()),
        "kl_from_base": float(torch.sum(base_log.exp() * (base_log - log_prob)).detach()),
        "kl_from_donor": float(torch.sum(donor_log.exp() * (donor_log - log_prob)).detach()),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    seed = 67
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    adapter = GatedMLPMaskAdapter(model)
    started = time.perf_counter()

    candidates = arithmetic_counterfactual_pairs()
    groups = {(operation, direction): [] for operation in OPERATIONS for direction in ("down", "up")}
    for pair in candidates:
        direction = "up" if pair.source.answer > pair.base.answer else "down"
        groups[pair.family, direction].append(pair)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    for operation in OPERATIONS:
        used_unordered = set()
        for direction in ("up", "down"):
            accepted = 0
            for pair in groups[operation, direction]:
                unordered = tuple(sorted((pair.base.example_id, pair.source.example_id)))
                if unordered in used_unordered:
                    continue
                base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(device)
                donor_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(device)
                if base_ids.shape != donor_ids.shape:
                    continue
                base_id = answer_id(tokenizer, pair.base.answer)
                donor_id = answer_id(tokenizer, pair.source.answer)
                with torch.no_grad():
                    intact = model(torch.cat((base_ids, donor_ids)), use_cache=False).logits[:, -1].argmax(dim=-1)
                if intact.tolist() != [base_id, donor_id]:
                    continue
                with torch.no_grad():
                    donor_trace = adapter.forward(donor_ids)
                    donor_activations = torch.stack([item[0].float() for item in donor_trace.activations])
                    full_mask = torch.zeros(adapter.mask_shape(base_ids.shape[1]), device=device)
                    patched_prediction = int(adapter.forward(base_ids, full_mask, baseline=donor_activations).logits[0].argmax())
                if patched_prediction != donor_id:
                    continue
                selected.append(pair)
                used_unordered.add(unordered)
                accepted += 1
                if accepted == PAIRS_PER_OPERATION // 2:
                    break
            if accepted != PAIRS_PER_OPERATION // 2:
                raise RuntimeError(f"not enough patchable {operation} {direction} pairs")

    rows = []
    saved = {}
    pair_metadata = []
    for index, pair in enumerate(selected, 1):
        base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(device)
        donor_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(device)
        base_id = answer_id(tokenizer, pair.base.answer)
        donor_id = answer_id(tokenizer, pair.source.answer)
        with torch.no_grad():
            donor_trace = adapter.forward(donor_ids)
        donor_activations = torch.stack([item[0].float() for item in donor_trace.activations])
        donor_logits = donor_trace.logits[0].float()
        base_trace = adapter.forward(base_ids, retain_activation_grads=True)
        base_logits = base_trace.logits[0].float()
        objective = base_logits[donor_id] - base_logits[base_id]
        gradients = torch.autograd.grad(objective, base_trace.activations)
        base_activations = torch.stack([item[0].detach().float() for item in base_trace.activations])
        gradient = torch.stack([item[0].detach().float() for item in gradients])
        scores = gradient * (donor_activations - base_activations)
        shape = adapter.mask_shape(base_ids.shape[1])
        allowed = eligible_instance_mask(shape, CONDITION, device)
        ranking = restricted_ranking(scores, allowed)
        eligible_total = int(ranking.numel())
        cache = {}

        def evaluate_at(patched):
            row = evaluate(
                adapter, tokenizer, pair, base_ids,
                patch_mask(ranking, patched, shape, "instance"), donor_activations,
                base_logits, donor_logits, base_id, donor_id, patched,
            )
            row["eligible_total"] = eligible_total
            return row

        for patched in counts(eligible_total):
            cache[patched] = evaluate_at(patched)
        refine_first_crossing(cache, "pairwise_flip", evaluate_at, refinements=12)
        refine_first_crossing(cache, "target_top1", evaluate_at, refinements=12)
        rows.extend(cache[count] for count in sorted(cache))
        successes = [count for count, row in cache.items() if row["target_top1"]]
        if successes:
            crossover = min(successes)
            saved[f"{pair.pair_id}__{CONDITION}__top1"] = ranking[:crossover].cpu().numpy().astype(np.int32)
        pair_metadata.append({
            **asdict(pair),
            "sequence_length": base_ids.shape[1],
            "base_pairwise_margin": float(base_logits[donor_id] - base_logits[base_id]),
            "donor_pairwise_margin": float(donor_logits[donor_id] - donor_logits[base_id]),
        })
        print(f"arithmetic {index}/{len(selected)}: {pair.pair_id}", flush=True)

    with (OUTPUT / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(OUTPUT / "crossover_indices.npz", **saved)
    metadata = {
        "model": model_name, "seed": seed, "device": device, "dtype": str(dtype),
        "condition": CONDITION, "pairs_per_operation": PAIRS_PER_OPERATION,
        "selected_pairs": pair_metadata,
        "ranking": "signed gradient × donor-minus-base activation, filtered to layers 0–19 and positions before the final token",
        "criterion": "donor answer is full-vocabulary top-1",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(rows):,} rows in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
