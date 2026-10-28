#!/usr/bin/env python3
"""Run fixed-length neuron-ablation pilots on three language tasks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.extraction import (
    importance_scores,
    mask_from_ranking,
    random_ranking,
    retained_counts,
    stable_descending_ranking,
)
from neuron_circuits.hf_adapter import (
    GatedMLPMaskAdapter,
    validate_all_one_mask,
    validate_downstream_recomputation,
)
from neuron_circuits.language_metrics import LanguageMaskEvaluation, evaluate_language_mask
from neuron_circuits.language_tasks import LanguageExample, all_language_examples


FAMILIES = ("ioi", "agreement", "brackets")
FRACTIONS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
DELETION_FRACTIONS = [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output", type=Path, default=Path("artifacts/language_pilot"))
    parser.add_argument("--mean-dir", type=Path, default=Path("artifacts/language_pilot"))
    parser.add_argument("--examples-per-family", type=int, default=9)
    parser.add_argument("--family-eval", type=int, default=9)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def token_id(tokenizer, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"expected one token for {text!r}, got {encoded}")
    return encoded[0]


def screen_by_length(model, tokenizer, examples, device) -> dict[tuple[str, int], list[LanguageExample]]:
    grouped: dict[tuple[str, int], list[LanguageExample]] = defaultdict(list)
    for example in examples:
        length = len(tokenizer.encode(example.prompt, add_special_tokens=False))
        grouped[(example.family, length)].append(example)
    correct: dict[tuple[str, int], list[LanguageExample]] = defaultdict(list)
    for key, group in grouped.items():
        for start in range(0, len(group), 64):
            batch = group[start : start + 64]
            ids = tokenizer([item.prompt for item in batch], return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                predictions = model(ids, use_cache=False).logits[:, -1].argmax(dim=-1).tolist()
            for item, prediction in zip(batch, predictions, strict=True):
                if prediction == token_id(tokenizer, item.answer_text):
                    correct[key].append(item)
    return correct


def choose_length(correct: dict[tuple[str, int], list[LanguageExample]], family: str) -> int:
    groups = [(length, rows) for (name, length), rows in correct.items() if name == family]
    if not groups:
        raise RuntimeError(f"no intact-correct examples for {family}")
    # Largest aligned pool; the bracket set deliberately makes length 18 dominant.
    return max(groups, key=lambda item: len(item[1]))[0]


def balanced_take(
    examples: list[LanguageExample], count: int, rng: random.Random
) -> list[LanguageExample]:
    groups: dict[str, list[LanguageExample]] = defaultdict(list)
    for example in examples:
        groups[example.answer_label].append(example)
    for group in groups.values():
        rng.shuffle(group)
    labels = sorted(groups)
    selected: list[LanguageExample] = []
    while len(selected) < count:
        progressed = False
        for label in labels:
            if groups[label] and len(selected) < count:
                selected.append(groups[label].pop())
                progressed = True
        if not progressed:
            break
    return selected


def smallest_feasible(rows: list[LanguageMaskEvaluation], alpha: float = 0.9):
    feasible = [row for row in rows if row.masked_margin >= alpha * row.intact_margin]
    return min(feasible, key=lambda row: row.retained) if feasible else None


def refine_transition(
    adapter, tokenizer, example, ids, ranking, shape, existing, full_logits,
    method, baseline, baseline_name, refinements: int = 5,
) -> list[LanguageMaskEvaluation]:
    bad = [row.retained for row in existing if row.masked_margin < 0.9 * row.intact_margin]
    good = [row.retained for row in existing if row.masked_margin >= 0.9 * row.intact_margin]
    if not bad or not good:
        return []
    high = min(good)
    lower = [value for value in bad if value < high]
    if not lower:
        return []
    low = max(lower)
    seen = {row.retained for row in existing}
    output = []
    for _ in range(refinements):
        middle = (low + high) // 2
        if middle in seen or middle in (low, high):
            break
        row = evaluate_language_mask(
            adapter, tokenizer, example, ids, mask_from_ranking(ranking, middle, shape),
            full_logits, method=method, baseline=baseline, baseline_name=baseline_name,
        )
        output.append(row)
        seen.add(middle)
        if row.masked_margin >= 0.9 * row.intact_margin:
            high = middle
        else:
            low = middle
    return output


def load_means(directory: Path, lengths: set[int], device: str) -> dict[int, torch.Tensor]:
    result = {}
    for length in lengths:
        path = directory / f"wikitext103_mean_pos{length}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mean = payload["mean"].to(device)
        if mean.shape[1] != length:
            raise ValueError(f"{path} has incompatible shape {tuple(mean.shape)}")
        result[length] = mean
    return result


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    adapter = GatedMLPMaskAdapter(model)

    candidates = all_language_examples()
    correct = screen_by_length(model, tokenizer, candidates, args.device)
    lengths = {family: choose_length(correct, family) for family in FAMILIES}
    selected: dict[str, list[LanguageExample]] = {}
    held_out: dict[str, list[LanguageExample]] = {}
    for family in FAMILIES:
        pool = list(correct[(family, lengths[family])])
        chosen = balanced_take(pool, args.examples_per_family, rng)
        chosen_ids = {row.example_id for row in chosen}
        remaining = [row for row in pool if row.example_id not in chosen_ids]
        held = balanced_take(remaining, args.family_eval, rng)
        if len(chosen) < args.examples_per_family or len(held) < args.family_eval:
            raise RuntimeError(f"not enough aligned, intact-correct {family} examples")
        selected[family], held_out[family] = chosen, held

    means = load_means(args.mean_dir, set(lengths.values()), args.device)
    probe = tokenizer(selected["agreement"][0].prompt, return_tensors="pt").input_ids.to(args.device)
    all_one_error = validate_all_one_mask(adapter, probe)
    recompute_probe, downstream_change = validate_downstream_recomputation(adapter, probe)

    rows: list[dict[str, object]] = []
    extraction_costs: list[dict[str, object]] = []
    rankings_by_example: dict[str, dict[str, torch.Tensor]] = {}
    full_by_example: dict[str, torch.Tensor] = {}
    family_scores: dict[tuple[str, str], torch.Tensor] = {}
    saved_masks: dict[str, np.ndarray] = {}

    for family in FAMILIES:
        length = lengths[family]
        shape = adapter.mask_shape(length)
        total = int(np.prod(shape))
        counts = retained_counts(total, FRACTIONS)
        family_scores[(family, "zero")] = torch.zeros(shape, device=args.device)
        family_scores[(family, "corpus_mean")] = torch.zeros(shape, device=args.device)
        for index, example in enumerate(selected[family], 1):
            ids = tokenizer(example.prompt, return_tensors="pt").input_ids.to(args.device)
            answer_id = token_id(tokenizer, example.answer_text)
            full_logits, scores, seconds = importance_scores(
                adapter, ids, answer_id, difference_baseline=means[length]
            )
            full_by_example[example.example_id] = full_logits.cpu()
            zero_score = scores["gradient_activation"]
            mean_score = scores["gradient_difference"]
            family_scores[(family, "zero")] += zero_score / zero_score.sum().clamp_min(1e-12)
            family_scores[(family, "corpus_mean")] += mean_score / mean_score.sum().clamp_min(1e-12)
            rankings = {
                "random": random_ranking(total, args.seed * 1000 + index + 100 * FAMILIES.index(family), torch.device(args.device)),
                "gradient_activation": stable_descending_ranking(zero_score),
                "gradient_difference": stable_descending_ranking(mean_score),
            }
            rankings_by_example[example.example_id] = {
                name: ranking.cpu().to(torch.int32) for name, ranking in rankings.items()
            }
            extraction_costs.append({"example_id": example.example_id, "seconds": seconds, "backward_evaluations": 1})

            settings = (
                ("random", rankings["random"], None, "zero"),
                ("gradient_activation", rankings["gradient_activation"], None, "zero"),
                ("random", rankings["random"], means[length], "corpus_mean"),
                ("gradient_difference", rankings["gradient_difference"], means[length], "corpus_mean"),
            )
            for method, ranking, baseline, baseline_name in settings:
                evaluations = [
                    evaluate_language_mask(
                        adapter, tokenizer, example, ids,
                        mask_from_ranking(ranking, retained, shape), full_logits,
                        method=method, baseline=baseline, baseline_name=baseline_name,
                    )
                    for retained in counts
                ]
                evaluations += refine_transition(
                    adapter, tokenizer, example, ids, ranking, shape, evaluations,
                    full_logits, method, baseline, baseline_name,
                )
                rows.extend(row.as_dict() | {"phase": "per_example"} for row in evaluations)
                feasible = smallest_feasible(evaluations)
                if feasible is not None and method != "random":
                    key = f"{example.example_id}__{baseline_name}__{method}__a0_9"
                    saved_masks[key] = ranking[: feasible.retained].cpu().numpy().astype(np.int32)
            print(f"per-example {family} {index}/{len(selected[family])} {example.example_id}", flush=True)

    # Frozen masks answer whether a circuit survives a changed input of the same task.
    for family in FAMILIES:
        length = lengths[family]
        shape = adapter.mask_shape(length)
        for source, target in zip(selected[family], held_out[family], strict=True):
            target_ids = tokenizer(target.prompt, return_tensors="pt").input_ids.to(args.device)
            with torch.no_grad():
                target_full = adapter.forward(target_ids).logits[0].float()
            source_rows = [
                LanguageMaskEvaluation(**{key: value for key, value in row.items() if key != "phase"})
                for row in rows if row["phase"] == "per_example" and row["example_id"] == source.example_id
            ]
            for method, baseline_name, baseline in (
                ("gradient_activation", "zero", None),
                ("gradient_difference", "corpus_mean", means[length]),
            ):
                feasible = smallest_feasible([
                    row for row in source_rows if row.method == method and row.baseline == baseline_name
                ])
                if feasible is None:
                    continue
                ranking = rankings_by_example[source.example_id][method].to(args.device, dtype=torch.long)
                result = evaluate_language_mask(
                    adapter, tokenizer, target, target_ids,
                    mask_from_ranking(ranking, feasible.retained, shape), target_full,
                    method=method, baseline=baseline, baseline_name=baseline_name,
                )
                rows.append(result.as_dict() | {
                    "phase": "frozen_transfer", "source_example_id": source.example_id,
                    "source_answer_label": source.answer_label,
                })

    # Shared task circuits aggregate normalized attribution over all extraction examples.
    for family in FAMILIES:
        length = lengths[family]
        shape = adapter.mask_shape(length)
        total = int(np.prod(shape))
        for baseline_name, method, baseline in (
            ("zero", "family_gradient_activation", None),
            ("corpus_mean", "family_gradient_difference", means[length]),
        ):
            ranking = stable_descending_ranking(family_scores[(family, baseline_name)])
            for retained in retained_counts(total, FRACTIONS):
                mask = mask_from_ranking(ranking, retained, shape)
                for example in held_out[family]:
                    ids = tokenizer(example.prompt, return_tensors="pt").input_ids.to(args.device)
                    with torch.no_grad():
                        full_logits = adapter.forward(ids).logits[0].float()
                    result = evaluate_language_mask(
                        adapter, tokenizer, example, ids, mask, full_logits,
                        method=method, baseline=baseline, baseline_name=baseline_name,
                    )
                    rows.append(result.as_dict() | {"phase": "task_family"})

    # Necessity: delete the highest zero-baseline attribution units.
    for family in FAMILIES:
        length = lengths[family]
        shape = adapter.mask_shape(length)
        total = int(np.prod(shape))
        for example in selected[family][:6]:
            ids = tokenizer(example.prompt, return_tensors="pt").input_ids.to(args.device)
            ranking = rankings_by_example[example.example_id]["gradient_activation"].to(args.device, dtype=torch.long)
            full_logits = full_by_example[example.example_id].to(args.device)
            for removed in retained_counts(total, DELETION_FRACTIONS):
                if removed == total:
                    continue
                result = evaluate_language_mask(
                    adapter, tokenizer, example, ids,
                    mask_from_ranking(ranking, removed, shape, invert=True), full_logits,
                    method="gradient_activation_deletion", baseline_name="zero",
                )
                rows.append(result.as_dict() | {
                    "phase": "necessity", "removed": removed, "removed_fraction": removed / total,
                })

    np.savez_compressed(args.output / "successful_masks.npz", **saved_masks)
    write_jsonl(args.output / "evaluations.jsonl", rows)
    write_jsonl(args.output / "extraction_costs.jsonl", extraction_costs)
    candidate_counts = Counter(example.family for example in candidates)
    correct_counts = {
        family: len(correct[(family, lengths[family])]) for family in FAMILIES
    }
    metadata = {
        "model": args.model, "seed": args.seed, "device": args.device, "dtype": str(dtype),
        "families": list(FAMILIES), "sequence_lengths": lengths,
        "candidate_counts": dict(candidate_counts), "aligned_intact_correct_counts": correct_counts,
        "selected_examples": {family: [asdict(row) for row in selected[family]] for family in FAMILIES},
        "held_out_examples": {family: [asdict(row) for row in held_out[family]] for family in FAMILIES},
        "mask_shapes": {family: adapter.mask_shape(lengths[family]) for family in FAMILIES},
        "fractions": FRACTIONS, "mean_windows": 50_000,
        "all_one_max_logit_error": all_one_error,
        "downstream_recompute_probe": recompute_probe,
        "downstream_recompute_max_activation_change": downstream_change,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {len(rows):,} evaluations to {args.output} in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
