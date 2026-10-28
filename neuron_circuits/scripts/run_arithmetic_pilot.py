#!/usr/bin/env python3
"""Run the Stage A neuron-sufficiency pilot on single-token arithmetic."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.arithmetic import ArithmeticExample, arithmetic_examples, operation_counts
from neuron_circuits.extraction import (
    MaskEvaluation,
    evaluate_mask,
    importance_scores,
    mask_from_ranking,
    random_ranking,
    retained_counts,
    smallest_feasible,
    stable_descending_ranking,
)
from neuron_circuits.hf_adapter import (
    GatedMLPMaskAdapter,
    validate_all_one_mask,
    validate_downstream_recomputation,
)


METHODS = ("random", "activation", "write_magnitude", "gradient_activation")
FRACTIONS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
ALPHAS = (0.5, 0.9, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output", type=Path, default=Path("artifacts/arithmetic_pilot"))
    parser.add_argument("--examples", type=int, default=24)
    parser.add_argument("--family-eval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def select_balanced(
    examples: list[ArithmeticExample], count: int, rng: random.Random
) -> list[ArithmeticExample]:
    groups: dict[str, list[ArithmeticExample]] = defaultdict(list)
    for example in examples:
        groups[example.operation].append(example)
    for group in groups.values():
        rng.shuffle(group)
    selected: list[ArithmeticExample] = []
    while len(selected) < count:
        progressed = False
        for operation in ("add", "subtract", "multiply"):
            if groups[operation] and len(selected) < count:
                selected.append(groups[operation].pop())
                progressed = True
        if not progressed:
            break
    return selected


def encode_prompt(tokenizer, example: ArithmeticExample, device: str) -> torch.Tensor:
    return tokenizer(example.prompt, return_tensors="pt").input_ids.to(device)


def number_tokens(tokenizer) -> dict[int, int]:
    result = {}
    for number in range(10):
        encoded = tokenizer.encode(str(number), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"{number} does not map to one token: {encoded}")
        result[number] = encoded[0]
    return result


def screen_examples(model, tokenizer, examples, tokens, device) -> list[ArithmeticExample]:
    correct = []
    for start in range(0, len(examples), 64):
        batch = examples[start : start + 64]
        encoded = tokenizer([item.prompt for item in batch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            predictions = model(**encoded, use_cache=False).logits[:, -1].argmax(dim=-1).cpu().tolist()
        correct.extend(
            item for item, prediction in zip(batch, predictions, strict=True)
            if prediction == tokens[item.answer]
        )
    return correct


def mean_baseline(adapter, tokenizer, examples, device) -> torch.Tensor:
    encoded = tokenizer([item.prompt for item in examples], return_tensors="pt", padding=True).input_ids.to(device)
    with torch.no_grad():
        traced = adapter.forward(encoded)
    return torch.stack([activation.float().mean(dim=0) for activation in traced.activations])


def refine_transition(
    adapter,
    ids,
    ranking,
    shape,
    existing: list[MaskEvaluation],
    full_logits,
    example,
    answer_id,
    number_ids,
    method,
    refinements: int = 4,
) -> list[MaskEvaluation]:
    ordered = sorted(existing, key=lambda item: item.retained)
    infeasible = [item.retained for item in ordered if item.masked_margin < 0.9 * item.intact_margin]
    feasible = [item.retained for item in ordered if item.masked_margin >= 0.9 * item.intact_margin]
    if not infeasible or not feasible:
        return []
    high = min(feasible)
    lower_candidates = [value for value in infeasible if value < high]
    if not lower_candidates:
        return []
    low = max(lower_candidates)
    results = []
    seen = {item.retained for item in ordered}
    for _ in range(refinements):
        middle = (low + high) // 2
        if middle in seen or middle == low or middle == high:
            break
        mask = mask_from_ranking(ranking, middle, shape)
        result = evaluate_mask(
            adapter, ids, mask, full_logits,
            example_id=example.example_id, method=method, answer=example.answer,
            answer_token_id=answer_id, number_token_ids=number_ids,
        )
        results.append(result)
        seen.add(middle)
        if result.masked_margin >= 0.9 * result.intact_margin:
            high = middle
        else:
            low = middle
    return results


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
    number_ids = number_tokens(tokenizer)

    candidates = arithmetic_examples()
    lengths = {len(tokenizer.encode(example.prompt)) for example in candidates}
    if len(lengths) != 1:
        raise ValueError(f"prompts are not position aligned: {sorted(lengths)}")
    correct = screen_examples(model, tokenizer, candidates, number_ids, args.device)
    selected = select_balanced(correct, args.examples, rng)
    selected_ids = {item.example_id for item in selected}
    held_out = [item for item in correct if item.example_id not in selected_ids]
    if len(selected) != args.examples:
        raise RuntimeError("not enough correctly answered prompts")

    probe_ids = encode_prompt(tokenizer, selected[0], args.device)
    all_one_error = validate_all_one_mask(adapter, probe_ids)
    recompute_probe, downstream_change = validate_downstream_recomputation(adapter, probe_ids)
    shape = adapter.mask_shape(probe_ids.shape[1])
    total = int(np.prod(shape))
    counts = retained_counts(total, FRACTIONS)
    print(
        f"model={args.model} correct={len(correct)}/{len(candidates)} "
        f"mask_shape={shape} total={total:,} all_one_error={all_one_error:.3g}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    extraction_costs: list[dict[str, object]] = []
    score_accumulator = torch.zeros(shape, dtype=torch.float32, device=args.device)
    saved_masks: dict[str, np.ndarray] = {}
    structure: dict[str, dict[str, object]] = {}
    rankings_by_example: dict[str, dict[str, torch.Tensor]] = {}
    full_by_example: dict[str, torch.Tensor] = {}

    for index, example in enumerate(selected, 1):
        ids = encode_prompt(tokenizer, example, args.device)
        answer_id = number_ids[example.answer]
        full_logits, scores, extraction_seconds = importance_scores(adapter, ids, answer_id)
        full_by_example[example.example_id] = full_logits.cpu()
        normalized = scores["gradient_activation"] / scores["gradient_activation"].sum().clamp_min(1e-12)
        score_accumulator += normalized
        rankings = {
            "random": random_ranking(total, args.seed * 1000 + index, torch.device(args.device)),
            **{name: stable_descending_ranking(value) for name, value in scores.items()},
        }
        rankings_by_example[example.example_id] = {
            key: value.cpu().to(torch.int32) for key, value in rankings.items()
        }
        extraction_costs.append({
            "example_id": example.example_id,
            "backward_evaluations": 1,
            "seconds": extraction_seconds,
        })

        for method in METHODS:
            method_results = []
            for retained in counts:
                mask = mask_from_ranking(rankings[method], retained, shape)
                result = evaluate_mask(
                    adapter, ids, mask, full_logits,
                    example_id=example.example_id, method=method, answer=example.answer,
                    answer_token_id=answer_id, number_token_ids=number_ids,
                )
                method_results.append(result)
            method_results.extend(
                refine_transition(
                    adapter, ids, rankings[method], shape, method_results, full_logits,
                    example, answer_id, number_ids, method,
                )
            )
            rows.extend(result.as_dict() | {"phase": "per_example"} for result in method_results)
            for alpha in ALPHAS:
                feasible = smallest_feasible(method_results, alpha)
                if feasible is None:
                    continue
                key = f"{example.example_id}__{method}__a{str(alpha).replace('.', '_')}"
                indices = rankings[method][: feasible.retained].detach().cpu().numpy().astype(np.int32)
                saved_masks[key] = indices
                if alpha == 0.9:
                    mask = mask_from_ranking(rankings[method], feasible.retained, shape)
                    structure[key] = {
                        "retained": feasible.retained,
                        "layer_counts": mask.sum(dim=(1, 2)).int().cpu().tolist(),
                        "position_counts": mask.sum(dim=(0, 2)).int().cpu().tolist(),
                    }
        print(f"per-example {index}/{len(selected)} {example.example_id}", flush=True)

    # Frozen-mask transfer: each source mask is evaluated on a changed problem
    # from the same operation with a different correct answer.
    by_operation: dict[str, list[ArithmeticExample]] = defaultdict(list)
    for example in held_out:
        by_operation[example.operation].append(example)
    transfer_rows = []
    for example in selected:
        target = next(item for item in by_operation[example.operation] if item.answer != example.answer)
        target_ids = encode_prompt(tokenizer, target, args.device)
        with torch.no_grad():
            target_full = adapter.forward(target_ids).logits[0].float()
        for method in METHODS:
            source = [
                MaskEvaluation(**{key: value for key, value in row.items() if key != "phase"})
                for row in rows
                if row["phase"] == "per_example"
                and row["example_id"] == example.example_id
                and row["method"] == method
            ]
            feasible = smallest_feasible(source, 0.9)
            if feasible is None:
                continue
            ranking = rankings_by_example[example.example_id][method].to(
                args.device, dtype=torch.long
            )
            mask = mask_from_ranking(ranking, feasible.retained, shape)
            result = evaluate_mask(
                adapter, target_ids, mask, target_full,
                example_id=target.example_id, method=method,
                answer=target.answer, answer_token_id=number_ids[target.answer],
                number_token_ids=number_ids,
            )
            transfer_rows.append(
                result.as_dict()
                | {
                    "phase": "frozen_transfer",
                    "source_example_id": example.example_id,
                    "source_answer": example.answer,
                }
            )
    rows.extend(transfer_rows)
    print(f"frozen transfer evaluations={len(transfer_rows)}", flush=True)

    # Iterative pruning recomputes attribution after every removal step. Its
    # extra backward-pass cost is recorded rather than compared as if free.
    iterative_fractions = [1.0, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    for example in selected[:6]:
        ids = encode_prompt(tokenizer, example, args.device)
        answer_id = number_ids[example.answer]
        current = torch.ones(shape, device=args.device)
        full_logits = full_by_example[example.example_id].to(args.device)
        backward_count = 0
        iterative_seconds = 0.0
        for fraction in iterative_fractions:
            target_count = min(int(current.sum().item()), max(1, round(total * fraction)))
            if target_count < int(current.sum().item()):
                _, current_scores, seconds = importance_scores(
                    adapter, ids, answer_id, current
                )
                backward_count += 1
                iterative_seconds += seconds
                score = current_scores["gradient_activation"].clone()
                score[current == 0] = float("-inf")
                ranking = stable_descending_ranking(score)
                current = mask_from_ranking(ranking, target_count, shape)
            result = evaluate_mask(
                adapter, ids, current, full_logits,
                example_id=example.example_id, method="iterative_gradient_activation",
                answer=example.answer, answer_token_id=answer_id,
                number_token_ids=number_ids,
            )
            rows.append(result.as_dict() | {"phase": "iterative_pruning"})
        extraction_costs.append({
            "example_id": example.example_id,
            "method": "iterative_gradient_activation",
            "backward_evaluations": backward_count,
            "seconds": iterative_seconds,
        })
    print("iterative pruning complete", flush=True)

    # Task-family circuit: average normalized attribution scores across sources.
    family_ranking = stable_descending_ranking(score_accumulator / len(selected))
    family_eval = select_balanced(held_out, args.family_eval, rng)
    for retained in counts:
        mask = mask_from_ranking(family_ranking, retained, shape)
        for example in family_eval:
            ids = encode_prompt(tokenizer, example, args.device)
            with torch.no_grad():
                full_logits = adapter.forward(ids).logits[0].float()
            result = evaluate_mask(
                adapter, ids, mask, full_logits,
                example_id=example.example_id, method="family_gradient_activation",
                answer=example.answer, answer_token_id=number_ids[example.answer],
                number_token_ids=number_ids,
            )
            rows.append(result.as_dict() | {"phase": "task_family"})
    print(f"task-family evaluations={len(family_eval) * len(counts)}", flush=True)

    # Necessity is a separately optimized top-attribution deletion sweep.
    deletion_fractions = [0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    for example in selected[:12]:
        ids = encode_prompt(tokenizer, example, args.device)
        ranking = rankings_by_example[example.example_id]["gradient_activation"].to(
            args.device, dtype=torch.long
        )
        full_logits = full_by_example[example.example_id].to(args.device)
        for removed in retained_counts(total, deletion_fractions):
            if removed == total:
                continue
            mask = mask_from_ranking(ranking, removed, shape, invert=True)
            result = evaluate_mask(
                adapter, ids, mask, full_logits,
                example_id=example.example_id, method="gradient_activation_deletion",
                answer=example.answer, answer_token_id=number_ids[example.answer],
                number_token_ids=number_ids,
            )
            row = result.as_dict() | {"phase": "necessity", "removed": removed, "removed_fraction": removed / total}
            rows.append(row)
    print("necessity sweeps complete", flush=True)

    # Baseline sensitivity for the primary method at the same binary masks.
    baseline = mean_baseline(adapter, tokenizer, selected, args.device)
    for example in selected[:12]:
        ids = encode_prompt(tokenizer, example, args.device)
        ranking = rankings_by_example[example.example_id]["gradient_activation"].to(
            args.device, dtype=torch.long
        )
        full_logits = full_by_example[example.example_id].to(args.device)
        for retained in counts:
            mask = mask_from_ranking(ranking, retained, shape)
            result = evaluate_mask(
                adapter, ids, mask, full_logits,
                example_id=example.example_id, method="gradient_activation",
                answer=example.answer, answer_token_id=number_ids[example.answer],
                number_token_ids=number_ids, baseline=baseline, baseline_name="mean",
            )
            rows.append(result.as_dict() | {"phase": "baseline_sensitivity"})
    print("mean-baseline sensitivity complete", flush=True)

    np.savez_compressed(args.output / "successful_masks.npz", **saved_masks)
    write_jsonl(args.output / "evaluations.jsonl", rows)
    write_jsonl(args.output / "extraction_costs.jsonl", extraction_costs)
    (args.output / "structures.json").write_text(json.dumps(structure, indent=2), encoding="utf-8")
    metadata = {
        "model": args.model,
        "seed": args.seed,
        "device": args.device,
        "dtype": str(dtype),
        "candidate_count": len(candidates),
        "intact_correct_count": len(correct),
        "intact_accuracy": len(correct) / len(candidates),
        "candidate_operation_counts": operation_counts(candidates),
        "selected_examples": [asdict(item) for item in selected],
        "selected_operation_counts": operation_counts(selected),
        "sequence_length": next(iter(lengths)),
        "mask_shape": shape,
        "total_masked_activations": total,
        "all_one_max_logit_error": all_one_error,
        "downstream_recompute_probe": recompute_probe,
        "downstream_recompute_max_activation_change": downstream_change,
        "fractions": FRACTIONS,
        "alphas": ALPHAS,
        "methods": METHODS,
        "zero_ablation": True,
        "mean_baseline_sensitivity_examples": min(12, len(selected)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {args.output} in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
