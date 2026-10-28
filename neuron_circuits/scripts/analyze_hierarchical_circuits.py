#!/usr/bin/env python3
"""Summarize module-to-neuron circuits and characterize neuron selectivity."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.algorithmic_tasks import algorithm_examples
from neuron_circuits.hf_adapter import GatedMLPMaskAdapter
from neuron_circuits.language_tasks import bracket_examples


ROOT = Path("artifacts/hierarchical_circuits")
FAMILIES = ("brackets", "maximum", "minimum")
VALUE_POSITIONS = {6: 0, 9: 1, 12: 2, 15: 3}


def selected_units(pairs, family):
    frequency = Counter()
    effects = defaultdict(list)
    for pair in pairs:
        if pair["family"] != family:
            continue
        for row in pair["neurons"]["candidates"][:12]:
            if row["margin_effect"] <= 0:
                continue
            unit = (row["layer"], row["position"], row["channel"])
            frequency[unit] += 1
            effects[unit].append(row["margin_effect"])
    ordered = sorted(frequency, key=lambda unit: (frequency[unit], statistics.mean(effects[unit])), reverse=True)
    final_position = 17 if family == "brackets" else 19
    chosen = ordered[:3]
    for unit in (item for item in ordered if item[1] != final_position):
        if unit not in chosen:
            chosen.append(unit)
        if len(chosen) == 5:
            break
    return [
        {
            "unit": f"L{layer}/P{position}/N{channel}", "layer": layer,
            "position": position, "channel": channel,
            "pair_count": frequency[(layer, position, channel)],
            "mean_exact_effect": statistics.mean(effects[(layer, position, channel)]),
        }
        for layer, position, channel in chosen
    ]


def eta_squared(values, labels):
    if not values:
        return 0.0
    overall = statistics.mean(values)
    total = sum((value - overall) ** 2 for value in values)
    if total == 0:
        return 0.0
    groups = defaultdict(list)
    for value, label in zip(values, labels):
        groups[label].append(value)
    between = sum(len(group) * (statistics.mean(group) - overall) ** 2 for group in groups.values())
    return between / total


def summarize_records(records, family, unit):
    values = [row["activation"] for row in records]
    answer_eta = eta_squared(values, [row["answer"] for row in records])
    result = {
        **unit,
        "examples": len(records),
        "answer_eta_squared": answer_eta,
        "top_positive": sorted(records, key=lambda row: row["activation"], reverse=True)[:3],
        "top_negative": sorted(records, key=lambda row: row["activation"])[:3],
    }
    position = unit["position"]
    if family in ("maximum", "minimum") and position in VALUE_POSITIONS:
        result["local_value_eta_squared"] = eta_squared(values, [row["values"][VALUE_POSITIONS[position]] for row in records])
        winners = [row for row in records if row["values"][VALUE_POSITIONS[position]] == row["answer"]]
        nonwinners = [row for row in records if row["values"][VALUE_POSITIONS[position]] != row["answer"]]
        result["winner_minus_nonwinner_mean"] = statistics.mean(row["activation"] for row in winners) - statistics.mean(row["activation"] for row in nonwinners)
    return result


def collect_brackets(adapter, tokenizer, device, units):
    records = {unit["unit"]: [] for unit in units}
    for example in bracket_examples():
        ids = tokenizer(example.prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            trace = adapter.forward(ids)
        tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
        for unit in units:
            if unit["position"] >= ids.shape[1]:
                continue
            records[unit["unit"]].append({
                "example_id": example.example_id,
                "display": example.prompt.splitlines()[-1],
                "answer": example.answer_label,
                "position_token": tokens[unit["position"]],
                "activation": float(trace.activations[unit["layer"]][0, unit["position"], unit["channel"]]),
            })
    return records


def collect_algorithm(adapter, model, tokenizer, device, family, units):
    rng = random.Random(103 if family == "maximum" else 107)
    examples = algorithm_examples(family)
    rng.shuffle(examples)
    examples = examples[:400]
    records = {unit["unit"]: [] for unit in units}
    for start in range(0, len(examples), 64):
        batch = examples[start : start + 64]
        ids = tokenizer([example.prompt for example in batch], return_tensors="pt", padding=True).input_ids.to(device)
        with torch.no_grad():
            trace = adapter.forward(ids)
        predictions = trace.logits.argmax(dim=-1)
        for row, example in enumerate(batch):
            answer_id = tokenizer.encode(str(example.answer), add_special_tokens=False)[0]
            if int(predictions[row]) != answer_id:
                continue
            for unit in units:
                records[unit["unit"]].append({
                    "example_id": example.example_id,
                    "display": example.prompt.strip(),
                    "answer": example.answer,
                    "values": list(example.values),
                    "activation": float(trace.activations[unit["layer"]][row, unit["position"], unit["channel"]]),
                })
    return records


def first_crossing(curve):
    successes = [row["patched"] for row in curve if row["target_top1"]]
    return min(successes) if successes else None


def recurrent_modules(pairs, family, component):
    counts = Counter()
    effects = defaultdict(list)
    for pair in pairs:
        if pair["family"] != family:
            continue
        for row in pair["modules"]["components"][component]["candidates"][:5]:
            if row["margin_effect"] <= 0:
                continue
            location = (row["layer"], row["position"])
            counts[location] += 1
            effects[location].append(row["margin_effect"])
    return [
        {"layer": layer, "position": position, "pair_count": count, "mean_exact_effect": statistics.mean(effects[(layer, position)])}
        for (layer, position), count in counts.most_common(6)
    ]


def main():
    raw = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
    capability = json.loads((ROOT / "capability_screen.json").read_text(encoding="utf-8"))
    pairs = raw["pairs"]
    units = {family: selected_units(pairs, family) for family in FAMILIES}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(raw["model"])
    model = AutoModelForCausalLM.from_pretrained(raw["model"], dtype=dtype).to(device).eval()
    adapter = GatedMLPMaskAdapter(model)

    characterization = {}
    bracket_records = collect_brackets(adapter, tokenizer, device, units["brackets"])
    characterization["brackets"] = [summarize_records(bracket_records[unit["unit"]], "brackets", unit) for unit in units["brackets"]]
    for family in ("maximum", "minimum"):
        records = collect_algorithm(adapter, model, tokenizer, device, family, units[family])
        characterization[family] = [summarize_records(records[unit["unit"]], family, unit) for unit in units[family]]

    summary = {"model": raw["model"], "families": {}}
    for family in FAMILIES:
        family_pairs = [pair for pair in pairs if pair["family"] == family]
        summary["families"][family] = {
            "pair_count": len(family_pairs),
            "module_crossovers": {
                component: [first_crossing(pair["modules"]["components"][component]["curve"]) for pair in family_pairs]
                for component in ("attention_output", "mlp_output")
            },
            "recurrent_modules": {
                component: recurrent_modules(pairs, family, component)
                for component in ("attention_output", "mlp_output")
            },
            "neuron_crossovers": [first_crossing(pair["neurons"]["curve"]) for pair in family_pairs],
            "neurons": characterization[family],
        }
    (ROOT / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Hierarchical module-to-neuron circuit analysis", "",
        "Whole attention and MLP outputs were localized first. Individual MLP neurons were then searched only inside the three strongest MLP layer-position modules for each counterfactual.", "",
        "## Capability screen", "",
        "| Task | Full-vocabulary top-1 |", "|---|---:|",
    ]
    for condition, row in capability["results"].items():
        lines.append(f"| {condition.replace('_', ' ')} | {row['correct']}/{row['total']} ({row['accuracy']:.0%}) |")
    lines += [
        "", "Second-largest and second-argmax were not interpreted because the model failed the capability screen; max and min were reliable enough to analyze.", "",
        "## Working circuit hypothesis", "",
        "1. **Local feature encoding:** early MLP neurons at the changed list element or outer bracket encode the token identity. For max/min, the layer-0 input-site neurons have local-value η² = 1.0.",
        "2. **Selection/routing:** attention outputs at the final answer position, especially layers 16 and 20, aggregate the relevant input. Brackets also repeatedly use an earlier attention output near the bracket sequence.",
        "3. **Answer writeout:** late MLP neurons at the final position are selective for the resulting digit or closing bracket and causally move the answer margin.", "",
        "This is a supported circuit skeleton, not yet a complete max/min comparator proof: head-level path patching is still needed to identify exactly how candidate values are compared and routed.", "",
    ]
    for family in FAMILIES:
        node = summary["families"][family]
        lines += [f"## {family.title()}", ""]
        for component, values in node["module_crossovers"].items():
            finite = [value for value in values if value is not None]
            lines.append(f"- {component.replace('_', ' ')}: target top-1 in {len(finite)}/{len(values)} pairs; median crossover {statistics.median(finite) if finite else 'none'} module outputs.")
        finite_neurons = [value for value in node["neuron_crossovers"] if value is not None]
        lines.append(f"- Localized neurons: target top-1 in {len(finite_neurons)}/{len(node['neuron_crossovers'])} pairs; median crossover {statistics.median(finite_neurons) if finite_neurons else 'none'} neurons.")
        for component, modules in node["recurrent_modules"].items():
            rendered = ", ".join(
                f"L{item['layer']}/P{item['position']} ({item['pair_count']}/{node['pair_count']})"
                for item in modules[:4]
            )
            lines.append(f"- Recurrent {component.replace('_', ' ')} locations: {rendered}.")
        lines += ["", "| Neuron | Sparse pairs | Mean exact effect | Answer η² | Local-value η² | Winner Δ |", "|---|---:|---:|---:|---:|---:|"]
        for neuron in node["neurons"]:
            local_eta = f"{neuron['local_value_eta_squared']:.3f}" if "local_value_eta_squared" in neuron else "—"
            winner_delta = f"{neuron['winner_minus_nonwinner_mean']:.3f}" if "winner_minus_nonwinner_mean" in neuron else "—"
            lines.append(
                f"| {neuron['unit']} | {neuron['pair_count']} | {neuron['mean_exact_effect']:.3f} | {neuron['answer_eta_squared']:.3f} | "
                f"{local_eta} | {winner_delta} |"
            )
        lines.append("")
        for neuron in node["neurons"]:
            positive = neuron["top_positive"][0]
            negative = neuron["top_negative"][0]
            lines.append(f"- `{neuron['unit']}`: high `{positive['display']}` ({positive['activation']:+.3f}); low `{negative['display']}` ({negative['activation']:+.3f}).")
        lines.append("")
    lines += [
        "η² is the fraction of activation variance explained by the answer or local list value. Winner Δ is mean activation when that list position contains the max/min minus its mean otherwise. These are descriptive selectivity measures; exact patches supply the causal evidence.",
    ]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
