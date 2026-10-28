#!/usr/bin/env python3
"""Estimate how many MLP activations must change to cause counterfactual answers."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.hf_adapter import GatedMLPMaskAdapter
from neuron_circuits.language_tasks import CounterfactualPair, language_counterfactual_pairs


FAMILIES = ("ioi", "agreement", "brackets")
METHODS = ("counterfactual_gradient", "delta_magnitude", "random")
FRACTIONS = [
    0, 0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005,
    0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
    0.05, 0.1, 0.2, 0.5, 1.0,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output", type=Path, default=Path("artifacts/counterfactual_patching"))
    parser.add_argument("--pairs-per-family", type=int, default=12)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def one_token(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def screen_pairs(model, tokenizer, pairs, device):
    valid = []
    for pair in pairs:
        base_ids = tokenizer.encode(pair.base.prompt, add_special_tokens=False)
        source_ids = tokenizer.encode(pair.source.prompt, add_special_tokens=False)
        base_answer = one_token(tokenizer, pair.base.answer_text)
        source_answer = one_token(tokenizer, pair.source.answer_text)
        if len(base_ids) != len(source_ids) or base_answer is None or source_answer is None:
            continue
        ids = torch.tensor([base_ids, source_ids], device=device)
        with torch.no_grad():
            predictions = model(ids, use_cache=False).logits[:, -1].argmax(dim=-1).tolist()
        if predictions == [base_answer, source_answer]:
            valid.append(pair)
    return valid


def screen_patchable_pairs(adapter, tokenizer, pairs, device):
    """Keep pairs where replacing all MLP activations recovers the source answer."""
    patchable = []
    for pair in pairs:
        base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(device)
        source_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(device)
        source_answer = one_token(tokenizer, pair.source.answer_text)
        with torch.no_grad():
            source_trace = adapter.forward(source_ids)
            source_activations = torch.stack([item[0].float() for item in source_trace.activations])
            mask = torch.zeros(adapter.mask_shape(base_ids.shape[1]), device=device)
            prediction = int(
                adapter.forward(base_ids, mask, baseline=source_activations).logits[0].argmax()
            )
        if prediction == source_answer:
            patchable.append(pair)
    return patchable


def balanced_take(pairs: list[CounterfactualPair], count: int, rng: random.Random):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair.source.answer_label].append(pair)
    for group in groups.values():
        rng.shuffle(group)
    output = []
    while len(output) < count:
        progressed = False
        for label in sorted(groups):
            if groups[label] and len(output) < count:
                output.append(groups[label].pop())
                progressed = True
        if not progressed:
            break
    return output


def counts(total: int) -> list[int]:
    return sorted({min(total, max(0, round(total * fraction))) for fraction in FRACTIONS})


def patch_mask(ranking: torch.Tensor, patched: int, shape, resolution: str):
    mask = torch.ones(shape, device=ranking.device)
    if not patched:
        return mask
    chosen = ranking[:patched]
    if resolution == "instance":
        mask.reshape(-1)[chosen] = 0
    else:
        width = shape[2]
        layers = torch.div(chosen, width, rounding_mode="floor")
        channels = chosen % width
        mask[layers, :, channels] = 0
    return mask


def evaluate(
    adapter, tokenizer, pair, base_ids, mask, source_activations,
    base_logits, source_logits, base_answer_id, source_answer_id,
    method, resolution, patched,
):
    with torch.no_grad():
        logits = adapter.forward(base_ids, mask, baseline=source_activations).logits[0].float()
    log_prob = F.log_softmax(logits, dim=-1)
    base_log = F.log_softmax(base_logits, dim=-1)
    source_log = F.log_softmax(source_logits, dim=-1)
    prediction = int(logits.argmax())
    if prediction == source_answer_id:
        kind = "counterfactual_answer"
    elif prediction == base_answer_id:
        kind = "original_answer"
    else:
        kind = "other"
    total = mask.numel() if resolution == "instance" else mask.shape[0] * mask.shape[2]
    return {
        "pair_id": pair.pair_id,
        "family": pair.family,
        "method": method,
        "resolution": resolution,
        "patched": patched,
        "total": total,
        "patched_fraction": patched / total,
        "induced_activation_instances": int((mask == 0).sum()),
        "base_answer": pair.base.answer_text,
        "source_answer": pair.source.answer_text,
        "pairwise_margin": float(logits[source_answer_id] - logits[base_answer_id]),
        "pairwise_flip": bool(logits[source_answer_id] >= logits[base_answer_id]),
        "target_top1": prediction == source_answer_id,
        "predicted_token_id": prediction,
        "predicted_text": tokenizer.decode([prediction]),
        "outcome": kind,
        "target_probability": float(log_prob[source_answer_id].exp()),
        "kl_from_base": float(torch.sum(base_log.exp() * (base_log - log_prob)).detach()),
        "kl_from_source": float(torch.sum(source_log.exp() * (source_log - log_prob)).detach()),
    }


def refine_first_crossing(
    cache, criterion, evaluate_at, refinements=9,
):
    ordered = sorted(cache)
    successes = [count for count in ordered if cache[count][criterion]]
    if not successes:
        return
    high = min(successes)
    failures = [count for count in ordered if count < high and not cache[count][criterion]]
    if not failures:
        return
    low = max(failures)
    for _ in range(refinements):
        middle = (low + high) // 2
        if middle in cache or middle in (low, high):
            break
        cache[middle] = evaluate_at(middle)
        if cache[middle][criterion]:
            high = middle
        else:
            low = middle


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    adapter = GatedMLPMaskAdapter(model)

    valid = screen_pairs(model, tokenizer, language_counterfactual_pairs(), args.device)
    patchable = screen_patchable_pairs(adapter, tokenizer, valid, args.device)
    selected = {}
    for family in FAMILIES:
        family_pairs = [pair for pair in patchable if pair.family == family]
        by_length = defaultdict(list)
        for pair in family_pairs:
            length = len(tokenizer.encode(pair.base.prompt, add_special_tokens=False))
            by_length[length].append(pair)
        length, pool = max(by_length.items(), key=lambda item: len(item[1]))
        selected[family] = balanced_take(pool, args.pairs_per_family, rng)
        if len(selected[family]) != args.pairs_per_family:
            raise RuntimeError(f"not enough valid {family} counterfactual pairs")
        print(f"{family}: {len(pool)} valid aligned pairs at length {length}", flush=True)

    rows = []
    saved = {}
    pair_metadata = []
    for family in FAMILIES:
        for pair_index, pair in enumerate(selected[family], 1):
            base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(args.device)
            source_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(args.device)
            base_answer_id = one_token(tokenizer, pair.base.answer_text)
            source_answer_id = one_token(tokenizer, pair.source.answer_text)
            with torch.no_grad():
                source_trace = adapter.forward(source_ids)
            source_activations = torch.stack([item[0].float() for item in source_trace.activations])
            source_logits = source_trace.logits[0].float()
            base_trace = adapter.forward(base_ids, retain_activation_grads=True)
            base_logits = base_trace.logits[0].float()
            objective = base_logits[source_answer_id] - base_logits[base_answer_id]
            gradients = torch.autograd.grad(objective, base_trace.activations)
            base_activations = torch.stack([item[0].detach().float() for item in base_trace.activations])
            gradient = torch.stack([item[0].detach().float() for item in gradients])
            delta = source_activations - base_activations
            signed = gradient * delta
            shape = adapter.mask_shape(base_ids.shape[1])
            full_mask = torch.zeros(shape, device=args.device)
            full_patch = evaluate(
                adapter, tokenizer, pair, base_ids, full_mask, source_activations,
                base_logits, source_logits, base_answer_id, source_answer_id,
                "all", "instance", int(np.prod(shape)),
            )
            if not full_patch["target_top1"]:
                raise RuntimeError(f"full patch failed for {pair.pair_id}")

            pair_metadata.append({
                **asdict(pair),
                "sequence_length": base_ids.shape[1],
                "base_pairwise_margin": float(
                    (base_logits[source_answer_id] - base_logits[base_answer_id]).detach()
                ),
                "source_pairwise_margin": float(
                    (source_logits[source_answer_id] - source_logits[base_answer_id]).detach()
                ),
            })
            for resolution in ("instance", "neuron"):
                if resolution == "instance":
                    causal_scores = signed.reshape(-1)
                    delta_scores = delta.abs().reshape(-1)
                else:
                    causal_scores = signed.sum(dim=1).reshape(-1)
                    delta_scores = delta.square().sum(dim=1).sqrt().reshape(-1)
                total = causal_scores.numel()
                rankings = {
                    "counterfactual_gradient": torch.argsort(causal_scores, descending=True, stable=True),
                    "delta_magnitude": torch.argsort(delta_scores, descending=True, stable=True),
                    "random": torch.randperm(total, generator=torch.Generator(device="cpu").manual_seed(
                        args.seed * 10000 + pair_index + 100 * FAMILIES.index(family) + (1 if resolution == "neuron" else 0)
                    )).to(args.device),
                }
                for method in METHODS:
                    ranking = rankings[method]
                    cache = {}

                    def evaluate_at(patched):
                        return evaluate(
                            adapter, tokenizer, pair, base_ids,
                            patch_mask(ranking, patched, shape, resolution), source_activations,
                            base_logits, source_logits, base_answer_id, source_answer_id,
                            method, resolution, patched,
                        )

                    for patched in counts(total):
                        cache[patched] = evaluate_at(patched)
                    if method != "random":
                        refine_first_crossing(cache, "pairwise_flip", evaluate_at)
                        refine_first_crossing(cache, "target_top1", evaluate_at)
                    rows.extend(cache[count] for count in sorted(cache))
                    if method == "counterfactual_gradient":
                        successes = [count for count, row in cache.items() if row["target_top1"]]
                        if successes:
                            crossover = min(successes)
                            saved[f"{pair.pair_id}__{resolution}__top1"] = ranking[:crossover].cpu().numpy().astype(np.int32)
            print(f"patched {family} {pair_index}/{len(selected[family])}: {pair.pair_id}", flush=True)

    with (args.output / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(args.output / "crossover_indices.npz", **saved)
    metadata = {
        "model": args.model, "seed": args.seed, "device": args.device, "dtype": str(dtype),
        "pairs_per_family": args.pairs_per_family,
        "valid_pair_counts": {family: sum(pair.family == family for pair in valid) for family in FAMILIES},
        "patchable_pair_counts": {family: sum(pair.family == family for pair in patchable) for family in FAMILIES},
        "selected_pairs": pair_metadata,
        "methods": METHODS, "resolutions": ["instance", "neuron"], "fractions": FRACTIONS,
        "ranking": "descending gradient_at_base * (counterfactual_activation - base_activation)",
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(rows):,} evaluations in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
