#!/usr/bin/env python3
"""Run counterfactual Q/K/V/pattern/head-output patching on language pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.attention_patching import (
    AttentionPatch,
    QwenAttentionPatchAdapter,
    counterfactual_scores,
    mask_from_unit_ranking,
)
from neuron_circuits.language_tasks import language_counterfactual_pairs


COMPONENTS = ("query", "key", "value", "pattern", "message", "head_output")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--pair-metadata", type=Path, default=Path("artifacts/counterfactual_patching/metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/attention_counterfactuals"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"expected one token for {text!r}, got {ids}")
    return ids[0]


def patch_counts(total):
    fixed = {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512}
    fixed.update({round(total * fraction) for fraction in (0.1, 0.25, 0.5, 1.0)})
    return sorted(max(0, min(total, value)) for value in fixed)


def evaluate(
    adapter, tokenizer, pair, base_ids, patch, base_logits, donor_logits,
    original_id, target_id, component, method, patched, total, extra=None,
):
    with torch.no_grad():
        logits = adapter.forward(base_ids, patch).logits[0].float()
    log_prob = F.log_softmax(logits, dim=-1)
    base_log = F.log_softmax(base_logits, dim=-1)
    donor_log = F.log_softmax(donor_logits, dim=-1)
    prediction = int(logits.argmax())
    row = {
        "pair_id": pair.pair_id, "family": pair.family,
        "component": component, "method": method,
        "patched": patched, "total": total, "patched_fraction": patched / total,
        "pairwise_margin": float(logits[target_id] - logits[original_id]),
        "pairwise_flip": bool(logits[target_id] >= logits[original_id]),
        "target_top1": prediction == target_id,
        "predicted_token_id": prediction, "predicted_text": tokenizer.decode([prediction]),
        "target_probability": float(log_prob[target_id].exp()),
        "kl_from_base": float(torch.sum(base_log.exp() * (base_log - log_prob))),
        "kl_from_donor": float(torch.sum(donor_log.exp() * (donor_log - log_prob))),
    }
    if extra:
        row.update(extra)
    return row


def refine(cache, criterion, evaluate_at, rounds=10):
    successes = [count for count, row in cache.items() if row[criterion]]
    if not successes:
        return
    high = min(successes)
    failures = [count for count, row in cache.items() if count < high and not row[criterion]]
    if not failures:
        return
    low = max(failures)
    for _ in range(rounds):
        middle = (low + high) // 2
        if middle in cache or middle in (low, high):
            break
        cache[middle] = evaluate_at(middle)
        if cache[middle][criterion]:
            high = middle
        else:
            low = middle


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager"
    ).to(args.device).eval()
    adapter = QwenAttentionPatchAdapter(model)

    prior = json.loads(args.pair_metadata.read_text())
    pair_lookup = {pair.pair_id: pair for pair in language_counterfactual_pairs()}
    selected_ids = [item["pair_id"] for item in prior["selected_pairs"]]
    selected = [pair_lookup[pair_id] for pair_id in selected_ids]
    rows = []
    saved = {}

    for pair_index, pair in enumerate(selected, 1):
        base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(args.device)
        donor_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(args.device)
        original_id = token_id(tokenizer, pair.base.answer_text)
        target_id = token_id(tokenizer, pair.source.answer_text)
        with torch.no_grad():
            donor = adapter.forward(donor_ids)
        base = adapter.forward(base_ids, retain_grads=True)
        if int(base.logits[0].argmax()) != original_id or int(donor.logits[0].argmax()) != target_id:
            raise RuntimeError(f"intact answer mismatch for {pair.pair_id}")
        scores = counterfactual_scores(base, donor, target_id, original_id)
        base_logits = base.logits[0].detach().float()
        donor_logits = donor.logits[0].detach().float()
        rankings = {}

        for component_index, component in enumerate(COMPONENTS):
            shape = adapter.unit_shape(component, base_ids.shape[1])
            total = int(np.prod(shape))
            targeted = torch.argsort(scores[component].reshape(-1), descending=True, stable=True)
            generator = torch.Generator(device="cpu").manual_seed(
                args.seed * 10000 + pair_index * 100 + component_index
            )
            rankings[component] = targeted
            component_donor = tuple(item.detach() for item in donor.component(component))
            for method, ranking in (
                ("counterfactual_gradient", targeted),
                ("random", torch.randperm(total, generator=generator).to(args.device)),
            ):
                cache = {}

                def evaluate_at(count):
                    patch = AttentionPatch(
                        component,
                        mask_from_unit_ranking(ranking, count, shape),
                        component_donor,
                    )
                    return evaluate(
                        adapter, tokenizer, pair, base_ids, patch, base_logits, donor_logits,
                        original_id, target_id, component, method, count, total,
                    )

                for count in patch_counts(total):
                    cache[count] = evaluate_at(count)
                if method == "counterfactual_gradient":
                    refine(cache, "pairwise_flip", evaluate_at)
                    refine(cache, "target_top1", evaluate_at)
                    successes = [count for count, row in cache.items() if row["target_top1"]]
                    if successes:
                        crossing = min(successes)
                        saved[f"{pair.pair_id}__{component}__top1"] = targeted[:crossing].cpu().numpy().astype(np.int32)
                rows.extend(cache[count] for count in sorted(cache))

        # Joint Q+K patching diagnoses routing interactions that isolated patches miss.
        query_shape = adapter.unit_shape("query", base_ids.shape[1])
        key_shape = adapter.unit_shape("key", base_ids.shape[1])
        query_total, key_total = int(np.prod(query_shape)), int(np.prod(key_shape))
        query_donor = tuple(item.detach() for item in donor.query)
        key_donor = tuple(item.detach() for item in donor.key)
        joint_counts = (0, 1, 2, 4, 8, 16, 32, 64)
        joint_rows = []
        for query_count in joint_counts:
            for key_count in joint_counts:
                if query_count == key_count == 0:
                    continue
                patches = (
                    AttentionPatch(
                        "query",
                        mask_from_unit_ranking(rankings["query"], min(query_count, query_total), query_shape),
                        query_donor,
                    ),
                    AttentionPatch(
                        "key",
                        mask_from_unit_ranking(rankings["key"], min(key_count, key_total), key_shape),
                        key_donor,
                    ),
                )
                row = evaluate(
                    adapter, tokenizer, pair, base_ids, patches, base_logits, donor_logits,
                    original_id, target_id, "query_key_joint", "counterfactual_gradient",
                    query_count + key_count, query_total + key_total,
                    {"query_patched": query_count, "key_patched": key_count},
                )
                joint_rows.append(row)
        rows.extend(joint_rows)
        joint_successes = [row for row in joint_rows if row["target_top1"]]
        if joint_successes:
            best = min(joint_successes, key=lambda row: (row["patched"], row["query_patched"]))
            saved[f"{pair.pair_id}__qk_query__top1"] = rankings["query"][: best["query_patched"]].cpu().numpy().astype(np.int32)
            saved[f"{pair.pair_id}__qk_key__top1"] = rankings["key"][: best["key_patched"]].cpu().numpy().astype(np.int32)
        print(f"attention {pair_index}/{len(selected)} {pair.family}: {pair.pair_id}", flush=True)

    with (args.output / "evaluations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(args.output / "crossover_indices.npz", **saved)
    metadata = {
        "model": args.model, "seed": args.seed, "device": args.device, "dtype": str(dtype),
        "pair_metadata": str(args.pair_metadata), "selected_pair_ids": selected_ids,
        "pair_count": len(selected), "components": COMPONENTS,
        "num_layers": adapter.num_layers, "num_query_heads": adapter.num_query_heads,
        "num_key_value_heads": adapter.num_kv_heads, "head_dim": adapter.head_dim,
        "grouped_query_factor": adapter.num_query_heads // adapter.num_kv_heads,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(rows):,} evaluations in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
