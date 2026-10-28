#!/usr/bin/env python3
"""Localize algorithmic circuits by modules, then decompose important MLPs into neurons."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neuron_circuits.algorithmic_tasks import algorithm_counterfactual_pairs
from neuron_circuits.hf_adapter import GatedMLPMaskAdapter
from neuron_circuits.language_tasks import language_counterfactual_pairs
from neuron_circuits.module_patching import (
    ModulePatch,
    QwenModulePatchAdapter,
    module_counterfactual_scores,
    module_mask,
)


COMPONENTS = ("attention_output", "mlp_output")
MODULE_PREFILTER = 16
LOCALIZED_MLPS = 3
NEURON_PREFILTER = 24


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hierarchical_circuits"))
    parser.add_argument("--pairs-per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"expected one answer token for {text!r}, got {ids}")
    return ids[0]


def answer_text(example):
    return example.answer_text if hasattr(example, "answer_text") else str(example.answer)


def select_pairs(model, tokenizer, args):
    rng = random.Random(args.seed)
    bracket_lookup = {pair.pair_id: pair for pair in language_counterfactual_pairs() if pair.family == "brackets"}
    prior = json.loads(Path("artifacts/counterfactual_patching/metadata.json").read_text(encoding="utf-8"))
    bracket_ids = [item["pair_id"] for item in prior["selected_pairs"] if item["family"] == "brackets"]
    selected = [bracket_lookup[pair_id] for pair_id in bracket_ids[: args.pairs_per_family]]
    screening = {}
    for family in ("maximum", "minimum"):
        candidates = algorithm_counterfactual_pairs(family)
        rng.shuffle(candidates)
        groups = {"up": [], "down": []}
        for pair in candidates:
            direction = "up" if pair.source.answer > pair.base.answer else "down"
            groups[direction].append(pair)
        family_selected = []
        used = set()
        screened = 0
        for direction in ("up", "down"):
            for pair in groups[direction]:
                unordered = tuple(sorted((pair.base.example_id, pair.source.example_id)))
                if unordered in used:
                    continue
                base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(args.device)
                donor_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(args.device)
                with torch.no_grad():
                    predictions = model(torch.cat((base_ids, donor_ids)), use_cache=False).logits[:, -1].argmax(dim=-1).tolist()
                screened += 1
                expected = [token_id(tokenizer, str(pair.base.answer)), token_id(tokenizer, str(pair.source.answer))]
                if predictions != expected:
                    continue
                family_selected.append(pair)
                used.add(unordered)
                if sum((candidate.source.answer > candidate.base.answer) == (direction == "up") for candidate in family_selected) == args.pairs_per_family // 2:
                    break
        if len(family_selected) != args.pairs_per_family:
            raise RuntimeError(f"could not find {args.pairs_per_family} correct {family} counterfactuals")
        selected.extend(family_selected)
        screening[family] = {"candidates_checked": screened, "selected": len(family_selected)}
    return selected, screening


def patch_metrics(logits, base_margin, original_id, target_id):
    margin = float(logits[target_id] - logits[original_id])
    return {
        "margin": margin,
        "margin_effect": margin - base_margin,
        "pairwise_flip": margin >= 0,
        "target_top1": int(logits.argmax()) == target_id,
    }


def module_analysis(adapter, base_ids, donor_ids, original_id, target_id):
    with torch.no_grad():
        donor = adapter.forward(donor_ids)
    base = adapter.forward(base_ids, retain_grads=True)
    base_logits = base.logits[0].detach().float()
    base_margin = float(base_logits[target_id] - base_logits[original_id])
    scores = module_counterfactual_scores(base, donor, target_id, original_id)
    shape = adapter.unit_shape(base_ids.shape[1])
    result = {"base_margin": base_margin, "components": {}}
    donor_components = {component: tuple(x.detach() for x in donor.component(component)) for component in COMPONENTS}
    for component in COMPONENTS:
        ranking = torch.argsort(scores[component].flatten(), descending=True, stable=True)
        candidates = []
        for rank, flat in enumerate(ranking[:MODULE_PREFILTER].tolist(), 1):
            layer, position = divmod(flat, shape[1])
            patch = ModulePatch(component, module_mask(shape, torch.tensor([flat], device=base_ids.device)), donor_components[component])
            with torch.no_grad():
                logits = adapter.forward(base_ids, patch).logits[0].float()
            candidates.append({
                "rank": rank, "layer": layer, "position": position,
                "first_order_score": float(scores[component][layer, position]),
                **patch_metrics(logits, base_margin, original_id, target_id),
            })
        curve = []
        total = ranking.numel()
        for count in sorted({0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, total}):
            patch = ModulePatch(component, module_mask(shape, ranking[:count]), donor_components[component])
            with torch.no_grad():
                logits = adapter.forward(base_ids, patch).logits[0].float()
            curve.append({"patched": count, **patch_metrics(logits, base_margin, original_id, target_id)})
        result["components"][component] = {"candidates": candidates, "curve": curve}
    return result


def localized_mlp_positions(module_result):
    candidates = module_result["components"]["mlp_output"]["candidates"]
    ordered = sorted(candidates, key=lambda row: (row["margin_effect"], row["first_order_score"]), reverse=True)
    positive = [(row["layer"], row["position"]) for row in ordered if row["margin_effect"] > 0]
    chosen = positive[:LOCALIZED_MLPS]
    if len(chosen) < LOCALIZED_MLPS:
        for row in ordered:
            location = (row["layer"], row["position"])
            if location not in chosen:
                chosen.append(location)
            if len(chosen) == LOCALIZED_MLPS:
                break
    return chosen


def neuron_analysis(adapter, base_ids, donor_ids, original_id, target_id, locations):
    with torch.no_grad():
        donor_trace = adapter.forward(donor_ids)
    donor = torch.stack([item[0].float() for item in donor_trace.activations])
    base_trace = adapter.forward(base_ids, retain_activation_grads=True)
    base_logits = base_trace.logits[0].detach().float()
    base_margin = float(base_logits[target_id] - base_logits[original_id])
    gradients = torch.autograd.grad(base_trace.logits[0, target_id] - base_trace.logits[0, original_id], base_trace.activations)
    base = torch.stack([item[0].detach().float() for item in base_trace.activations])
    gradient = torch.stack([item[0].detach().float() for item in gradients])
    scores = gradient * (donor - base)
    width = adapter.neuron_width
    localized_scores = torch.stack([scores[layer, position] for layer, position in locations])
    ranking = torch.argsort(localized_scores.flatten(), descending=True, stable=True)

    def coordinate(raw):
        location_index, channel = divmod(int(raw), width)
        layer, position = locations[location_index]
        return layer, position, channel

    candidates = []
    shape = adapter.mask_shape(base_ids.shape[1])
    for rank, raw in enumerate(ranking[:NEURON_PREFILTER].tolist(), 1):
        layer, position, channel = coordinate(raw)
        mask = torch.ones(shape, device=base_ids.device)
        mask[layer, position, channel] = 0
        with torch.no_grad():
            logits = adapter.forward(base_ids, mask, baseline=donor).logits[0].float()
        candidates.append({
            "rank": rank, "layer": layer, "position": position, "channel": channel,
            "base_activation": float(base[layer, position, channel]),
            "donor_activation": float(donor[layer, position, channel]),
            "activation_delta": float(donor[layer, position, channel] - base[layer, position, channel]),
            "first_order_score": float(scores[layer, position, channel]),
            **patch_metrics(logits, base_margin, original_id, target_id),
        })
    curve = []
    total = ranking.numel()
    counts = sorted({0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, total})
    for count in counts:
        mask = torch.ones(shape, device=base_ids.device)
        for raw in ranking[:count].tolist():
            layer, position, channel = coordinate(raw)
            mask[layer, position, channel] = 0
        with torch.no_grad():
            logits = adapter.forward(base_ids, mask, baseline=donor).logits[0].float()
        curve.append({"patched": count, **patch_metrics(logits, base_margin, original_id, target_id)})
    return {"localized_mlp_positions": [list(location) for location in locations], "candidates": candidates, "curve": curve}


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
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    module_adapter = QwenModulePatchAdapter(model)
    neuron_adapter = GatedMLPMaskAdapter(model)
    pairs, screening = select_pairs(model, tokenizer, args)
    results = []
    for index, pair in enumerate(pairs, 1):
        base_ids = tokenizer(pair.base.prompt, return_tensors="pt").input_ids.to(args.device)
        donor_ids = tokenizer(pair.source.prompt, return_tensors="pt").input_ids.to(args.device)
        original_id = token_id(tokenizer, answer_text(pair.base))
        target_id = token_id(tokenizer, answer_text(pair.source))
        modules = module_analysis(module_adapter, base_ids, donor_ids, original_id, target_id)
        locations = localized_mlp_positions(modules)
        neurons = neuron_analysis(neuron_adapter, base_ids, donor_ids, original_id, target_id, locations)
        results.append({
            "pair_id": pair.pair_id, "family": pair.family,
            "base": asdict(pair.base), "source": asdict(pair.source),
            "sequence_length": int(base_ids.shape[1]),
            "modules": modules, "neurons": neurons,
        })
        print(f"hierarchical {index}/{len(pairs)} {pair.family}: {pair.pair_id}", flush=True)
    output = {
        "model": args.model, "seed": args.seed, "device": args.device, "dtype": str(dtype),
        "screening": screening,
        "method": "module-output counterfactual patching followed by neuron patches restricted to three causally strongest MLP layer-position modules",
        "pairs": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} pairs in {output['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
