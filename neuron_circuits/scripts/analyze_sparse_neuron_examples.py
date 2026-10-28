#!/usr/bin/env python3
"""Find examples where recurrent strict-sparse MLP units help or resist a counterfactual."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.hf_adapter import GatedMLPMaskAdapter
from neuron_circuits.language_tasks import language_counterfactual_pairs
from run_counterfactual_patching import one_token


BASE_ROOT = Path("artifacts/counterfactual_patching")
RESTRICTED_ROOT = Path("artifacts/restricted_counterfactuals")
OUTPUT = Path("artifacts/sparse_neuron_examples")
FAMILIES = ("ioi", "agreement", "brackets")
WIDTH = 4864
SPARSE_BUDGET = 100
CONDITION = "exclude_final_four_layers_and_position"


def top_sparse_units(metadata, restricted_summary, arrays):
    output = {}
    for family in FAMILIES:
        frequency = Counter()
        eligible_pairs = []
        values = restricted_summary["families"][family][CONDITION]["pair_values"]
        for pair in metadata["selected_pairs"]:
            if pair["family"] != family or values[pair["pair_id"]] > SPARSE_BUDGET:
                continue
            eligible_pairs.append(pair["pair_id"])
            length = pair["sequence_length"]
            seen = set()
            for raw in arrays[f"{pair['pair_id']}__{CONDITION}__top1"]:
                index = int(raw)
                seen.add((
                    index // (length * WIDTH),
                    (index // WIDTH) % length,
                    index % WIDTH,
                ))
            frequency.update(seen)
        output[family] = {
            "eligible_pairs": eligible_pairs,
            "units": [
                {"layer": layer, "position": position, "channel": channel, "pairs": count}
                for (layer, position, channel), count in frequency.most_common(5)
            ],
        }
    return output


def compact_prompt(family: str, prompt: str) -> str:
    if family == "brackets":
        return prompt.splitlines()[-1]
    return prompt


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE_ROOT / "metadata.json").read_text(encoding="utf-8"))
    restricted_summary = json.loads((RESTRICTED_ROOT / "summary.json").read_text(encoding="utf-8"))
    arrays = np.load(RESTRICTED_ROOT / "crossover_indices.npz")
    selected = top_sparse_units(metadata, restricted_summary, arrays)
    pair_lookup = {pair.pair_id: pair for pair in language_counterfactual_pairs()}

    model_name = metadata["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    adapter = GatedMLPMaskAdapter(model)

    pair_scores = {family: {f"L{u['layer']}/P{u['position']}/N{u['channel']}": [] for u in selected[family]["units"]} for family in FAMILIES}
    for pair_number, pair_meta in enumerate(metadata["selected_pairs"], 1):
        pair = pair_lookup[pair_meta["pair_id"]]
        base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(device)
        donor_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(device)
        base_answer_id = one_token(tokenizer, pair.base.answer_text)
        donor_answer_id = one_token(tokenizer, pair.source.answer_text)
        with torch.no_grad():
            donor_trace = adapter.forward(donor_ids)
        donor_activations = torch.stack([item[0].float() for item in donor_trace.activations])
        base_trace = adapter.forward(base_ids, retain_activation_grads=True)
        objective = base_trace.logits[0, donor_answer_id] - base_trace.logits[0, base_answer_id]
        gradients = torch.autograd.grad(objective, base_trace.activations)
        base_activations = torch.stack([item[0].detach().float() for item in base_trace.activations])
        gradient = torch.stack([item[0].detach().float() for item in gradients])
        delta = donor_activations - base_activations
        scores = gradient * delta
        base_tokens = tokenizer.convert_ids_to_tokens(base_ids[0].tolist())
        donor_tokens = tokenizer.convert_ids_to_tokens(donor_ids[0].tolist())
        for unit in selected[pair.family]["units"]:
            layer, position, channel = unit["layer"], unit["position"], unit["channel"]
            key = f"L{layer}/P{position}/N{channel}"
            pair_scores[pair.family][key].append({
                "pair_id": pair.pair_id,
                "base_prompt": pair.base.prompt,
                "donor_prompt": pair.source.prompt,
                "base_display": compact_prompt(pair.family, pair.base.prompt),
                "donor_display": compact_prompt(pair.family, pair.source.prompt),
                "base_answer": pair.base.answer_text.strip(),
                "donor_answer": pair.source.answer_text.strip(),
                "base_token": base_tokens[position],
                "donor_token": donor_tokens[position],
                "gradient": float(gradient[layer, position, channel]),
                "base_activation": float(base_activations[layer, position, channel]),
                "donor_activation": float(donor_activations[layer, position, channel]),
                "activation_delta": float(delta[layer, position, channel]),
                "signed_causal_score": float(scores[layer, position, channel]),
            })
        print(f"scored {pair_number}/{len(metadata['selected_pairs'])}: {pair.pair_id}", flush=True)

    result = {
        "model": model_name,
        "definition": "signed_causal_score = base-run gradient of donor-minus-original answer logit multiplied by donor-minus-base activation",
        "sparse_definition": f"strictest restriction ({CONDITION}) with crossover <= {SPARSE_BUDGET} activations",
        "families": {},
    }
    csv_rows = []
    for family in FAMILIES:
        result["families"][family] = {
            "eligible_pairs": selected[family]["eligible_pairs"],
            "neurons": [],
        }
        aggregate_examples = []
        for unit in selected[family]["units"]:
            key = f"L{unit['layer']}/P{unit['position']}/N{unit['channel']}"
            examples = pair_scores[family][key]
            positive = sorted((item for item in examples if item["signed_causal_score"] > 0), key=lambda item: item["signed_causal_score"], reverse=True)[:5]
            negative = sorted((item for item in examples if item["signed_causal_score"] < 0), key=lambda item: item["signed_causal_score"])[:5]
            node = {**unit, "unit": key, "top_positive": positive, "top_negative": negative}
            result["families"][family]["neurons"].append(node)
            aggregate_examples.extend({**row, "unit": key, "recurrence_pairs": unit["pairs"]} for row in examples)
            for polarity, rows in (("positive", positive), ("negative", negative)):
                for rank, row in enumerate(rows, 1):
                    csv_rows.append({
                        "family": family, "unit": key, "recurrence_pairs": unit["pairs"],
                        "polarity": polarity, "rank": rank, **row,
                    })
        result["families"][family]["top_positive_examples"] = sorted(
            (row for row in aggregate_examples if row["signed_causal_score"] > 0),
            key=lambda row: row["signed_causal_score"], reverse=True,
        )[:5]
        result["families"][family]["top_negative_examples"] = sorted(
            (row for row in aggregate_examples if row["signed_causal_score"] < 0),
            key=lambda row: row["signed_causal_score"],
        )[:5]
    (OUTPUT / "examples.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUTPUT / "examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Positive and negative examples for recurrent sparse neurons", "",
        f"Sparse means the strictest restriction—no layers 20–23 and no final token position—with a strict top-1 crossover of at most {SPARSE_BUDGET} scalar activations.", "",
        "A positive signed causal score means copying the donor activation locally pushes toward the donor answer. A negative score means the same donor activation locally pushes against that answer. This is a first-order score at the base run, not the result of patching the unit alone.", "",
    ]
    for family in FAMILIES:
        lines += [f"## {family.upper()}", ""]
        lines.append("Most recurrent units: " + ", ".join(
            f"{node['unit']} ({node['pairs']}/{len(selected[family]['eligible_pairs'])} sparse pairs)"
            for node in result["families"][family]["neurons"]
        ))
        for polarity, examples in (
            ("Positive", result["families"][family]["top_positive_examples"]),
            ("Negative", result["families"][family]["top_negative_examples"]),
        ):
            lines += ["", f"### Top {polarity.lower()} neuron–example combinations", "", "| Rank | Unit | Counterfactual | Answers | Score | Activation change |", "|---:|---|---|---|---:|---:|"]
            if examples:
                for rank, row in enumerate(examples, 1):
                    prompt_change = f"`{row['base_display']}` → `{row['donor_display']}`"
                    lines.append(f"| {rank} | {row['unit']} | {prompt_change} | {row['base_answer']} → {row['donor_answer']} | {row['signed_causal_score']:.4g} | {row['activation_delta']:.4g} |")
            else:
                lines.append("| — | — | No examples with this sign | — | — | — |")
    lines += ["", "Full top-five positive and negative lists for all five recurrent units per family are in `examples.json` and `examples.csv`."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
