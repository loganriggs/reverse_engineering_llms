#!/usr/bin/env python3
"""Rank positive and negative arithmetic examples for recurrent sparse MLP units."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import statistics

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.hf_adapter import GatedMLPMaskAdapter


ROOT = Path("artifacts/arithmetic_counterfactuals")
OUTPUT = Path("artifacts/arithmetic_neuron_examples")
OPERATIONS = ("add", "subtract", "multiply")
WIDTH = 4864
SPARSE_BUDGET = 100
CONDITION = "exclude_final_four_layers_and_position"


def token_id(tokenizer, number):
    ids = tokenizer.encode(str(number), add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{number} is not a single token")
    return ids[0]


def crossovers(rows):
    output = {}
    for pair_id in {row["pair_id"] for row in rows}:
        success = [row["patched"] for row in rows if row["pair_id"] == pair_id and row["target_top1"]]
        if success:
            output[pair_id] = min(success)
    return output


def recurrent_units(metadata, values, arrays, operation=None):
    frequency = Counter()
    eligible = []
    for pair in metadata["selected_pairs"]:
        if operation is not None and pair["family"] != operation:
            continue
        if values[pair["pair_id"]] > SPARSE_BUDGET:
            continue
        eligible.append(pair["pair_id"])
        length = pair["sequence_length"]
        seen = set()
        for raw in arrays[f"{pair['pair_id']}__{CONDITION}__top1"]:
            index = int(raw)
            seen.add((index // (length * WIDTH), (index // WIDTH) % length, index % WIDTH))
        frequency.update(seen)
    return eligible, [
        {"layer": layer, "position": position, "channel": channel, "pairs": count}
        for (layer, position, channel), count in frequency.most_common(5)
    ]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (ROOT / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
    values = crossovers(rows)
    arrays = np.load(ROOT / "crossover_indices.npz")
    eligible, top_units = recurrent_units(metadata, values, arrays)
    by_operation = {}
    for operation in OPERATIONS:
        op_eligible, op_units = recurrent_units(metadata, values, arrays, operation)
        by_operation[operation] = {"eligible_pairs": op_eligible, "units": op_units}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(metadata["model"])
    model = AutoModelForCausalLM.from_pretrained(metadata["model"], dtype=dtype).to(device).eval()
    adapter = GatedMLPMaskAdapter(model)
    unit_examples = {f"L{u['layer']}/P{u['position']}/N{u['channel']}": [] for u in top_units}

    for number, pair in enumerate(metadata["selected_pairs"], 1):
        base = pair["base"]
        donor = pair["source"]
        base_ids = tokenizer(base["prompt"], return_tensors="pt").input_ids.to(device)
        donor_ids = tokenizer(donor["prompt"], return_tensors="pt").input_ids.to(device)
        base_answer = token_id(tokenizer, base["answer"])
        donor_answer = token_id(tokenizer, donor["answer"])
        with torch.no_grad():
            donor_trace = adapter.forward(donor_ids)
        donor_activations = torch.stack([item[0].float() for item in donor_trace.activations])
        base_trace = adapter.forward(base_ids, retain_activation_grads=True)
        objective = base_trace.logits[0, donor_answer] - base_trace.logits[0, base_answer]
        gradients = torch.autograd.grad(objective, base_trace.activations)
        base_activations = torch.stack([item[0].detach().float() for item in base_trace.activations])
        gradient = torch.stack([item[0].detach().float() for item in gradients])
        delta = donor_activations - base_activations
        scores = gradient * delta
        base_tokens = tokenizer.convert_ids_to_tokens(base_ids[0].tolist())
        donor_tokens = tokenizer.convert_ids_to_tokens(donor_ids[0].tolist())
        for unit in top_units:
            layer, position, channel = unit["layer"], unit["position"], unit["channel"]
            key = f"L{layer}/P{position}/N{channel}"
            unit_examples[key].append({
                "pair_id": pair["pair_id"],
                "operation": pair["family"],
                "base_prompt": base["prompt"].strip(),
                "donor_prompt": donor["prompt"].strip(),
                "base_answer": base["answer"],
                "donor_answer": donor["answer"],
                "base_token": base_tokens[position],
                "donor_token": donor_tokens[position],
                "gradient": float(gradient[layer, position, channel]),
                "base_activation": float(base_activations[layer, position, channel]),
                "donor_activation": float(donor_activations[layer, position, channel]),
                "activation_delta": float(delta[layer, position, channel]),
                "signed_causal_score": float(scores[layer, position, channel]),
            })
        print(f"scored arithmetic {number}/{len(metadata['selected_pairs'])}: {pair['pair_id']}", flush=True)

    all_examples = []
    neuron_nodes = []
    csv_rows = []
    for unit in top_units:
        key = f"L{unit['layer']}/P{unit['position']}/N{unit['channel']}"
        examples = unit_examples[key]
        positive = sorted((x for x in examples if x["signed_causal_score"] > 0), key=lambda x: x["signed_causal_score"], reverse=True)[:5]
        negative = sorted((x for x in examples if x["signed_causal_score"] < 0), key=lambda x: x["signed_causal_score"])[:5]
        neuron_nodes.append({**unit, "unit": key, "top_positive": positive, "top_negative": negative})
        all_examples.extend({**example, "unit": key, "recurrence_pairs": unit["pairs"]} for example in examples)
        for polarity, examples_at_sign in (("positive", positive), ("negative", negative)):
            for rank, example in enumerate(examples_at_sign, 1):
                csv_rows.append({"unit": key, "recurrence_pairs": unit["pairs"], "polarity": polarity, "rank": rank, **example})

    top_positive = sorted((x for x in all_examples if x["signed_causal_score"] > 0), key=lambda x: x["signed_causal_score"], reverse=True)[:5]
    top_negative = sorted((x for x in all_examples if x["signed_causal_score"] < 0), key=lambda x: x["signed_causal_score"])[:5]
    result = {
        "model": metadata["model"],
        "definition": "signed_causal_score = base-run gradient of donor-minus-original answer logit multiplied by donor-minus-base activation",
        "sparse_definition": f"no layers 20–23, no final token position, strict crossover <= {SPARSE_BUDGET}",
        "crossover_summary": {},
        "eligible_pairs": eligible,
        "top_recurrent_units": neuron_nodes,
        "top_positive_examples": top_positive,
        "top_negative_examples": top_negative,
        "by_operation": by_operation,
    }
    for operation in OPERATIONS:
        op_values = [values[pair["pair_id"]] for pair in metadata["selected_pairs"] if pair["family"] == operation]
        result["crossover_summary"][operation] = {
            "median": statistics.median(op_values), "range": [min(op_values), max(op_values)],
            "pairs_at_or_below_100": sum(value <= 100 for value in op_values), "pairs": len(op_values),
        }
    (OUTPUT / "examples.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUTPUT / "examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)

    lines = [
        "# Arithmetic sparse-neuron examples", "",
        "Counterfactuals preserve the operation and left operand while changing the right operand. Sparse means no layers 20–23, no final token position, and a strict crossover of at most 100 scalar activations.", "",
        "| Operation | Median crossover | Range | ≤100 activations |", "|---|---:|---:|---:|",
    ]
    for operation in OPERATIONS:
        node = result["crossover_summary"][operation]
        lines.append(f"| {operation.title()} | {node['median']:g} | {node['range'][0]:,}–{node['range'][1]:,} | {node['pairs_at_or_below_100']}/{node['pairs']} |")
    sparse_count = len(eligible)
    lines += ["", f"Most recurrent units across the {sparse_count} sparse pairs: " + ", ".join(f"{n['unit']} ({n['pairs']}/{sparse_count})" for n in neuron_nodes), ""]
    for polarity, examples in (("Positive", top_positive), ("Negative", top_negative)):
        lines += [f"## Top {polarity.lower()} neuron–example combinations", "", "| Rank | Unit | Operation | Counterfactual | Answers | Score | Activation change |", "|---:|---|---|---|---|---:|---:|"]
        for rank, row in enumerate(examples, 1):
            lines.append(f"| {rank} | {row['unit']} | {row['operation']} | `{row['base_prompt']}` → `{row['donor_prompt']}` | {row['base_answer']} → {row['donor_answer']} | {row['signed_causal_score']:.4g} | {row['activation_delta']:.4g} |")
        lines.append("")
    lines += ["Positive scores locally support the donor answer; negative scores locally resist it. Scores are first-order attributions, not single-unit patch effects."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
