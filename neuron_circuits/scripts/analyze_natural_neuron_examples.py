#!/usr/bin/env python3
"""Find natural prompts that maximally activate or suppress sparse-circuit neurons."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.arithmetic import arithmetic_examples
from neuron_circuits.hf_adapter import GatedMLPMaskAdapter
from neuron_circuits.language_tasks import agreement_examples, bracket_examples, ioi_examples


OUTPUT = Path("artifacts/natural_neuron_examples")
LANGUAGE_SOURCE = Path("artifacts/sparse_neuron_examples/examples.json")
ARITHMETIC_SOURCE = Path("artifacts/arithmetic_neuron_examples/examples.json")


def compact_prompt(family: str, prompt: str) -> str:
    if family == "brackets":
        return prompt.splitlines()[-1]
    return prompt.strip()


def selected_units() -> dict[str, list[dict]]:
    language = json.loads(LANGUAGE_SOURCE.read_text(encoding="utf-8"))
    arithmetic = json.loads(ARITHMETIC_SOURCE.read_text(encoding="utf-8"))
    units = {
        family: [
            {key: neuron[key] for key in ("unit", "layer", "position", "channel", "pairs")}
            for neuron in language["families"][family]["neurons"]
        ]
        for family in ("ioi", "agreement", "brackets")
    }
    units["arithmetic"] = [
        {key: neuron[key] for key in ("unit", "layer", "position", "channel", "pairs")}
        for neuron in arithmetic["top_recurrent_units"]
    ]
    return units


def datasets() -> dict[str, list[dict]]:
    language = {
        "ioi": ioi_examples(),
        "agreement": agreement_examples(),
        "brackets": bracket_examples(),
    }
    output = {}
    for family, examples in language.items():
        output[family] = [
            {
                **asdict(example),
                "answer": example.answer_text.strip(),
                "display": compact_prompt(family, example.prompt),
            }
            for example in examples
        ]
    output["arithmetic"] = [
        {
            **asdict(example),
            "family": "arithmetic",
            "answer": str(example.answer),
            "display": example.prompt.strip(),
        }
        for example in arithmetic_examples()
    ]
    return output


def one_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"expected one token for {text!r}, got {ids}")
    return ids[0]


def score_family(model, adapter, tokenizer, device, family, examples, units):
    encoded = [(example, tokenizer(example["prompt"], return_tensors="pt").input_ids[0]) for example in examples]
    by_length = defaultdict(list)
    for example, ids in encoded:
        by_length[len(ids)].append((example, ids))
    records = {unit["unit"]: [] for unit in units}
    for length, group in sorted(by_length.items()):
        for start in range(0, len(group), 64):
            batch = group[start : start + 64]
            input_ids = torch.stack([ids for _, ids in batch]).to(device)
            with torch.no_grad():
                trace = adapter.forward(input_ids)
            predictions = trace.logits.argmax(dim=-1)
            tokens = [tokenizer.convert_ids_to_tokens(ids.tolist()) for _, ids in batch]
            for unit in units:
                layer, position, channel = unit["layer"], unit["position"], unit["channel"]
                if position >= length:
                    continue
                values = trace.activations[layer][:, position, channel].detach().float().cpu()
                for row, ((example, _), value) in enumerate(zip(batch, values)):
                    answer_text = str(example["answer"])
                    answer_for_tokenizer = answer_text if family == "arithmetic" else " " + answer_text
                    answer_id = one_token(tokenizer, answer_for_tokenizer)
                    predicted_id = int(predictions[row])
                    records[unit["unit"]].append({
                        "example_id": example["example_id"],
                        "family": family,
                        "operation": example.get("operation"),
                        "prompt": example["prompt"],
                        "display": example["display"],
                        "answer": answer_text,
                        "predicted_token": tokenizer.decode([predicted_id]).strip(),
                        "model_correct": predicted_id == answer_id,
                        "position_token": tokens[row][position],
                        "activation": float(value),
                    })
    return records


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units = selected_units()
    data = datasets()
    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
    adapter = GatedMLPMaskAdapter(model)

    result = {
        "model": model_name,
        "definition": "Positive and negative examples are ranked by the raw intact post-gating MLP activation, with no patching or gradients.",
        "families": {},
    }
    csv_rows = []
    for family in ("ioi", "agreement", "brackets", "arithmetic"):
        records = score_family(model, adapter, tokenizer, device, family, data[family], units[family])
        nodes = []
        for unit in units[family]:
            examples = records[unit["unit"]]
            positive = sorted(
                (row for row in examples if row["activation"] > 0),
                key=lambda row: row["activation"], reverse=True,
            )[:5]
            negative = sorted(
                (row for row in examples if row["activation"] < 0),
                key=lambda row: row["activation"],
            )[:5]
            node = {
                **unit,
                "examples_scored": len(examples),
                "positive_count": sum(row["activation"] > 0 for row in examples),
                "negative_count": sum(row["activation"] < 0 for row in examples),
                "top_positive": positive,
                "top_negative": negative,
            }
            nodes.append(node)
            for polarity, selected in (("positive", positive), ("negative", negative)):
                for rank, row in enumerate(selected, 1):
                    csv_rows.append({
                        "family": family, "unit": unit["unit"],
                        "recurrence_pairs": unit["pairs"], "polarity": polarity,
                        "rank": rank, **row,
                    })
        result["families"][family] = {
            "dataset_examples": len(data[family]),
            "neurons": nodes,
        }
        print(f"scored {len(data[family])} natural {family} examples", flush=True)

    (OUTPUT / "examples.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUTPUT / "examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Natural positive and negative examples for recurrent sparse neurons", "",
        "These are intact, unpatched dataset examples. Positive and negative mean the raw sign of the post-gating MLP activation—not counterfactual influence, gradient attribution, or answer support.", "",
    ]
    for family in ("ioi", "agreement", "brackets", "arithmetic"):
        lines += [f"## {family.upper()}", "", f"Scanned {len(data[family])} natural prompts.", ""]
        for node in result["families"][family]["neurons"]:
            lines += [
                f"### {node['unit']}", "",
                f"Recurred in {node['pairs']} sparse counterfactual circuits. At this coordinate, {node['positive_count']}/{node['examples_scored']} prompts had positive activation and {node['negative_count']}/{node['examples_scored']} had negative activation.", "",
                "| Sign/rank | Natural prompt | Answer | Position token | Activation |", "|---|---|---:|---|---:|",
            ]
            for polarity, selected in (("+", node["top_positive"]), ("−", node["top_negative"])):
                for rank, row in enumerate(selected, 1):
                    display = row["display"].replace("|", "\\|")
                    lines.append(f"| {polarity}{rank} | `{display}` | {row['answer']} | `{row['position_token']}` | {row['activation']:.5g} |")
            lines.append("")
    lines.append("The full prompt, model prediction, correctness, operation, and activation are retained in `examples.json` and `examples.csv`.")
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
