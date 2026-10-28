#!/usr/bin/env python3
"""Audit pairwise causal and information synergy across grammar and memory depth sweeps."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from neuron_circuits.causal_synergy import (
    factorial_statistics, pair_complementary_information, pair_information_synergy,
)
from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import GrammarTransformer
from neuron_circuits.self_bilinear import DeepSelfBilinear


GRAMMAR_BASE = Path("artifacts/grammar_regimes")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/causal_synergy"))
    parser.add_argument("--panel-size", type=int, default=32)
    parser.add_argument("--top-neurons", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_memory_experiment():
    path = Path(__file__).with_name("run_self_bilinear_compositional_memory.py")
    spec = importlib.util.spec_from_file_location("self_bilinear_experiment", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def grammar_root(kind: str, depth: int) -> Path:
    if kind == "swiglu":
        return GRAMMAR_BASE / ("synergy_swiglu_depth3" if depth == 3 else "easy_pilot")
    return GRAMMAR_BASE / ("synergy_bilinear_missing" if depth in (1, 3) else "bilinear_pilot")


def load_grammar_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GrammarTransformer(
        VOCAB.size, checkpoint["width"], checkpoint["heads"], checkpoint["depth"],
        mlp_kind=checkpoint["mlp_kind"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def masks_for_pairs(pairs: np.ndarray, total: int, mode: str, device: str):
    masks = torch.ones(len(pairs) * 4, total, device=device) if mode == "removal" else torch.zeros(len(pairs) * 4, total, device=device)
    for pair_index, (first, second) in enumerate(pairs):
        rows = slice(4 * pair_index, 4 * pair_index + 4)
        if mode == "removal":
            # 11, 10, 01, 00: the pair states while all other neurons remain on.
            masks[4 * pair_index + 1, second] = 0
            masks[4 * pair_index + 2, first] = 0
            masks[4 * pair_index + 3, [first, second]] = 0
        else:
            masks[4 * pair_index, [first, second]] = 1
            masks[4 * pair_index + 1, first] = 1
            masks[4 * pair_index + 2, second] = 1
    return masks


def random_pairs(total: int, count: int, rng: np.random.Generator):
    output = set()
    while len(output) < count:
        pair = tuple(sorted(rng.choice(total, 2, replace=False).tolist()))
        output.add(pair)
    return np.asarray(sorted(output), dtype=np.int64)


def permute_within_context(targets: np.ndarray, context: np.ndarray | None, rng: np.random.Generator):
    output = targets.copy()
    if context is None:
        rng.shuffle(output); return output
    for value in np.unique(context):
        indices = np.flatnonzero(context == value)
        output[indices] = rng.permutation(output[indices])
    return output


def information_audit(activations, targets, context, global_pairs, permutations, rng):
    binary = activations > np.median(activations, axis=0, keepdims=True)
    rows = []
    for first, second in global_pairs:
        observed = pair_information_synergy(targets, binary[:, first], binary[:, second], context)
        complementary = pair_complementary_information(targets, binary[:, first], binary[:, second], context)
        null, complementary_null = [], []
        for _ in range(permutations):
            shuffled = permute_within_context(targets, context, rng)
            null.append(pair_information_synergy(shuffled, binary[:, first], binary[:, second], context))
            complementary_null.append(pair_complementary_information(
                shuffled, binary[:, first], binary[:, second], context,
            ))
        rows.append({
            "first": int(first), "second": int(second), "observed_bits": float(observed),
            "null_mean_bits": float(np.mean(null)), "adjusted_bits": float(observed - np.mean(null)),
            "null_upper_95_bits": float(np.quantile(null, .95)),
            "complementary_bits": float(complementary),
            "complementary_null_mean_bits": float(np.mean(complementary_null)),
            "adjusted_complementary_bits": float(complementary - np.mean(complementary_null)),
        })
    adjusted = np.asarray([row["adjusted_bits"] for row in rows])
    complementary_adjusted = np.asarray([row["adjusted_complementary_bits"] for row in rows])
    return {
        "pairs": rows, "median_adjusted_bits": float(np.median(adjusted)),
        "maximum_adjusted_bits": float(np.max(adjusted)),
        "fraction_positive_adjusted": float(np.mean(adjusted > 0)),
        "median_adjusted_complementary_bits": float(np.median(complementary_adjusted)),
        "maximum_adjusted_complementary_bits": float(np.max(complementary_adjusted)),
        "fraction_positive_adjusted_complementary": float(np.mean(complementary_adjusted > 0)),
    }


def summarize_factorial(per_example):
    keys = ("targeted_removal", "random_removal", "targeted_construction", "random_construction")
    output = {}
    for key in keys:
        medians = np.asarray([row[key]["median_fraction"] for row in per_example])
        q75s = np.asarray([row[key]["q75_fraction"] for row in per_example])
        flips = np.asarray([row[key]["sign_flip_fraction"] for row in per_example])
        output[key] = {
            "median_of_example_medians": float(np.median(medians)),
            "mean_of_example_medians": float(np.mean(medians)),
            "median_of_example_q75": float(np.median(q75s)),
            "mean_sign_flip_fraction": float(np.mean(flips)),
        }
    output["removal_targeted_minus_random"] = float(np.median([
        row["targeted_removal"]["median_fraction"] - row["random_removal"]["median_fraction"]
        for row in per_example
    ]))
    output["construction_targeted_minus_random"] = float(np.median([
        row["targeted_construction"]["median_fraction"] - row["random_construction"]["median_fraction"]
        for row in per_example
    ]))
    return output


def rows_from_statistics(statistics):
    fraction = statistics["interaction_fraction"].detach().cpu().numpy()
    interaction = statistics["interaction"].detach().cpu().numpy()
    flips = statistics["sign_flip"].detach().cpu().numpy()
    return {
        "median_fraction": float(np.median(fraction)),
        "q75_fraction": float(np.quantile(fraction, .75)),
        "maximum_fraction": float(np.max(fraction)),
        "mean_absolute_interaction": float(np.mean(np.abs(interaction))),
        "sign_flip_fraction": float(np.mean(flips)),
        "fractions": fraction.tolist(), "interactions": interaction.tolist(),
    }


def causal_audit(scores, forward_masks, total, top_neurons, rng, transfer_context):
    per_example, largest = [], None
    global_top = np.argsort(np.mean(scores, axis=0))[-top_neurons:][::-1]
    global_pairs = np.asarray(list(itertools.combinations(global_top.tolist(), 2)), dtype=np.int64)
    global_interactions = [[] for _ in global_pairs]
    for example_index in range(len(scores)):
        selected = np.argsort(scores[example_index])[-top_neurons:][::-1]
        targeted = np.asarray(list(itertools.combinations(selected.tolist(), 2)), dtype=np.int64)
        controls = random_pairs(total, len(targeted), rng)
        pairs = np.concatenate((targeted, controls), axis=0)
        record = {"example": example_index, "targeted_pairs": targeted.tolist()}
        for mode in ("removal", "construction"):
            margins = forward_masks(example_index, masks_for_pairs(pairs, total, mode, forward_masks.device))
            statistics = factorial_statistics(margins.reshape(len(pairs), 4))
            split = len(targeted)
            target_stats = {key: value[:split] for key, value in statistics.items()}
            random_stats = {key: value[split:] for key, value in statistics.items()}
            record[f"targeted_{mode}"] = rows_from_statistics(target_stats)
            record[f"random_{mode}"] = rows_from_statistics(random_stats)
            fractions = target_stats["interaction_fraction"]
            maximum = int(fractions.argmax())
            candidate = {
                "example": example_index, "mode": mode,
                "pair": targeted[maximum].tolist(), "interaction_fraction": float(fractions[maximum]),
                "interaction": float(target_stats["interaction"][maximum]),
                "margins_11_10_01_00": margins.reshape(len(pairs), 4)[maximum].detach().cpu().tolist(),
                "sign_flip": bool(target_stats["sign_flip"][maximum]),
            }
            if largest is None or candidate["interaction_fraction"] > largest["interaction_fraction"]:
                largest = candidate
        global_margins = forward_masks(
            example_index, masks_for_pairs(global_pairs, total, "removal", forward_masks.device),
        )
        global_stats = factorial_statistics(global_margins.reshape(len(global_pairs), 4))
        for pair_index, interaction in enumerate(global_stats["interaction"].detach().cpu().tolist()):
            global_interactions[pair_index].append(interaction)
        per_example.append(record)
    consistency = []
    for pair, values in zip(global_pairs, global_interactions):
        values = np.asarray(values)
        material = np.abs(values) > .05
        conditioned_parts = []
        for context_value in np.unique(transfer_context):
            keep = material & (transfer_context == context_value)
            if keep.sum() < 2: continue
            positive = np.mean(values[keep] > 0)
            conditioned_parts.extend([max(positive, 1 - positive)] * int(keep.sum()))
        if material.any():
            positive = np.mean(values[material] > 0)
            consistency.append({"pair": pair.tolist(), "material_examples": int(material.sum()),
                                "sign_consistency": float(max(positive, 1 - positive)),
                                "answer_conditioned_sign_consistency": (
                                    None if not conditioned_parts else float(np.mean(conditioned_parts))
                                ),
                                "median_interaction": float(np.median(values))})
    conditioned = [row["answer_conditioned_sign_consistency"] for row in consistency
                   if row["answer_conditioned_sign_consistency"] is not None]
    return {
        "summary": summarize_factorial(per_example), "largest_interaction": largest,
        "global_top_neurons": global_top.tolist(), "global_pairs": global_pairs.tolist(),
        "global_pair_sign_consistency": {
            "eligible_pairs": len(consistency),
            "mean": None if not consistency else float(np.mean([row["sign_consistency"] for row in consistency])),
            "median": None if not consistency else float(np.median([row["sign_consistency"] for row in consistency])),
            "answer_conditioned_mean": None if not conditioned else float(np.mean(conditioned)),
            "answer_conditioned_median": None if not conditioned else float(np.median(conditioned)),
            "pairs": consistency,
        },
        "per_example": per_example,
    }, global_pairs


class GrammarForwardMasks:
    def __init__(self, model, tokens, targets, competitors):
        self.model, self.tokens, self.targets, self.competitors = model, tokens, targets, competitors
        self.device = str(tokens.device)

    @torch.inference_mode()
    def __call__(self, example_index, flat_masks):
        count = len(flat_masks); tokens = self.tokens[example_index:example_index + 1].expand(count, -1)
        masks = {layer: flat_masks[:, layer * self.model.channels:(layer + 1) * self.model.channels]
                 for layer in range(self.model.layers_count)}
        logits = self.model(tokens, channel_masks=masks)[:, -1, VOCAB.answers]
        return logits[:, self.targets[example_index]] - logits[:, self.competitors[example_index]]


class MemoryForwardMasks:
    def __init__(self, model, inputs, targets):
        self.model, self.inputs, self.targets = model, inputs, targets
        self.device = str(inputs.device)

    @torch.inference_mode()
    def __call__(self, example_index, flat_masks):
        count = len(flat_masks); inputs = self.inputs[example_index:example_index + 1].expand(count, -1)
        masks = list(flat_masks.split(self.model.neurons, dim=1))
        logits = self.model(inputs, masks)
        target = int(self.targets[example_index]); return logits[:, target] - logits[:, 1 - target]


def grammar_activations_and_scores(model, tokens, targets):
    model.zero_grad(set_to_none=True)
    logits, activations = model(tokens, return_full_activations=True)
    answers = logits[:, -1, VOCAB.answers]
    rivals = answers.detach().clone(); rivals[torch.arange(len(targets), device=tokens.device), targets] = -torch.inf
    competitors = rivals.argmax(1)
    margins = answers[torch.arange(len(targets), device=tokens.device), targets] - answers[torch.arange(len(targets), device=tokens.device), competitors]
    gradients = torch.autograd.grad(margins.sum(), activations)
    scores = torch.cat([(activation * gradient).sum(1).abs() for activation, gradient in zip(activations, gradients)], 1)
    scalar = torch.cat([activation[:, -1] for activation in activations], 1)
    return margins.detach(), competitors.detach(), scores.detach().cpu().numpy(), scalar.detach().cpu().numpy()


def memory_activations_and_scores(model, inputs, targets):
    model.zero_grad(set_to_none=True)
    logits, activations = model(inputs, return_activations=True)
    margins = logits[torch.arange(len(targets), device=inputs.device), targets] - logits[torch.arange(len(targets), device=inputs.device), 1 - targets]
    gradients = torch.autograd.grad(margins.sum(), activations)
    scores = torch.cat([(activation * gradient).abs() for activation, gradient in zip(activations, gradients)], 1)
    scalar = torch.cat(activations, 1)
    return margins.detach(), scores.detach().cpu().numpy(), scalar.detach().cpu().numpy()


@torch.inference_mode()
def grammar_scalar_activations(model, tokens):
    _, activations = model(tokens, return_full_activations=True)
    return torch.cat([activation[:, -1] for activation in activations], 1).cpu().numpy()


@torch.inference_mode()
def memory_scalar_activations(model, inputs):
    _, activations = model(inputs, return_activations=True)
    return torch.cat(activations, 1).cpu().numpy()


def audit_grammar(args, rng):
    rows = []
    for kind in ("swiglu", "bilinear"):
        for depth in (1, 2, 3, 4):
            root = grammar_root(kind, depth)
            manifest = load_manifest(root / "manifest.npz")
            facts = np.asarray(manifest["train_facts"])
            model, checkpoint = load_grammar_model(root / "rho_0" / f"depth_{depth}" / "checkpoint.pt", args.device)
            for rho in (0.0, .5, 1.0):
                destination = root / f"rho_{rho:g}" / f"depth_{depth}"
                model, checkpoint = load_grammar_model(destination / "checkpoint.pt", args.device)
                labels, selected, changed = labels_for_rho(manifest, "train", rho)
                prompts = encode_prompts(manifest, facts, 6)
                with torch.inference_mode():
                    token_tensor = torch.from_numpy(prompts.astype(np.int64)).to(args.device)
                    answer_logits = model(token_tensor)[:, -1, VOCAB.answers]
                    predictions = answer_logits.argmax(1).cpu().numpy()
                correct = predictions == labels
                definitions = ({"structured": np.ones(len(facts), bool)} if rho == 0 else
                               {"structured": ~selected, "exception": changed} if rho == .5 else
                               {"arbitrary": changed})
                for group_index, (group, membership) in enumerate(definitions.items()):
                    eligible = np.flatnonzero(correct & membership)
                    if not len(eligible): continue
                    panel = rng.choice(eligible, min(args.panel_size, len(eligible)), replace=False)
                    tokens = torch.from_numpy(prompts[panel].astype(np.int64)).to(args.device)
                    targets = torch.from_numpy(labels[panel].astype(np.int64)).to(args.device)
                    margins, competitors, scores, activations = grammar_activations_and_scores(model, tokens, targets)
                    forward = GrammarForwardMasks(model, tokens, targets, competitors)
                    causal, global_pairs = causal_audit(
                        scores, forward, scores.shape[1], args.top_neurons, rng,
                        labels[panel],
                    )
                    information_panel = rng.choice(eligible, min(512, len(eligible)), replace=False)
                    information_tokens = torch.from_numpy(prompts[information_panel].astype(np.int64)).to(args.device)
                    information_activations = grammar_scalar_activations(model, information_tokens)
                    information = information_audit(
                        information_activations, labels[information_panel], facts[information_panel, 2], global_pairs,
                        args.permutations, rng,
                    )
                    largest = causal["largest_interaction"]
                    largest["fact"] = facts[panel[largest["example"]]].tolist()
                    largest["target"] = int(labels[panel[largest["example"]]])
                    rows.append({
                        "family": "grammar", "architecture": kind, "depth": depth, "rho": rho,
                        "group": group, "seed": checkpoint["seed"], "panel_size": len(panel),
                        "information_panel_size": len(information_panel),
                        "eligible_correct": len(eligible), "group_total": int(membership.sum()),
                        "group_accuracy": float(correct[membership].mean()),
                        "mean_intact_margin": float(margins.mean()), "total_neurons": scores.shape[1],
                        "causal": causal, "information": information,
                    })
                    print(f"grammar {kind} depth={depth} rho={rho:g} {group}", flush=True)
    return rows


def audit_memory(args, rng):
    experiment = load_memory_experiment(); rows = []
    root = args.output / "memory_models"
    for task in ("lookup", "lookup_parity2"):
        for depth in (1, 2, 3, 4):
            for seed in (7, 19, 31):
                destination = root / task / f"depth_{depth}" / f"seed_{seed}"
                checkpoint = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=False)
                config = checkpoint["config"]
                model = DeepSelfBilinear(
                    config["facts"] + len(experiment.ARITIES), config["neurons"], depth,
                    residual_stream=True, decoder_bias=True,
                ).to(args.device).eval()
                model.load_state_dict(checkpoint["state_dict"])
                run_args = SimpleNamespace(
                    facts=config["facts"], neurons=config["neurons"], computation_tasks=[18432],
                    test_tasks_per_arity=1024, device=args.device,
                )
                values = checkpoint["values"].to(args.device)
                bank = experiment.build_bank(run_args, seed)
                lookup = experiment.lookup_data(run_args, values)
                datasets = {"lookup": lookup}
                if task == "lookup_parity2":
                    data = experiment.encoded_bank(run_args, bank, values, config["tasks_per_arity"])
                    datasets["parity2"] = data["test"][2]
                for group_index, (group, dataset) in enumerate(datasets.items()):
                    inputs, targets = dataset
                    with torch.inference_mode(): predictions = model(inputs).argmax(1)
                    correct = (predictions == targets).cpu().numpy()
                    eligible = np.flatnonzero(correct)
                    panel = rng.choice(eligible, min(args.panel_size, len(eligible)), replace=False)
                    panel_inputs, panel_targets = inputs[panel], targets[panel]
                    margins, scores, activations = memory_activations_and_scores(model, panel_inputs, panel_targets)
                    forward = MemoryForwardMasks(model, panel_inputs, panel_targets)
                    causal, global_pairs = causal_audit(
                        scores, forward, scores.shape[1], args.top_neurons, rng,
                        panel_targets.cpu().numpy(),
                    )
                    information_panel = rng.choice(eligible, min(512, len(eligible)), replace=False)
                    information_inputs, information_targets = inputs[information_panel], targets[information_panel]
                    information_activations = memory_scalar_activations(model, information_inputs)
                    information = information_audit(
                        information_activations, information_targets.cpu().numpy(), None, global_pairs,
                        args.permutations, rng,
                    )
                    largest = causal["largest_interaction"]
                    largest["dataset_index"] = int(panel[largest["example"]])
                    largest["target"] = int(panel_targets[largest["example"]])
                    rows.append({
                        "family": "memory", "architecture": "self_bilinear", "task": task,
                        "depth": depth, "group": group, "seed": seed, "panel_size": len(panel),
                        "information_panel_size": len(information_panel),
                        "eligible_correct": len(eligible), "group_total": len(inputs),
                        "group_accuracy": float(correct.mean()), "converged": json.loads((destination / "results.json").read_text())["converged"],
                        "mean_intact_margin": float(margins.mean()), "total_neurons": scores.shape[1],
                        "causal": causal, "information": information,
                    })
                    print(f"memory {task} depth={depth} seed={seed} {group}", flush=True)
    return rows


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "results.json"
    if output_path.exists() and not args.force:
        print(f"SKIP {output_path}"); return
    rng = np.random.default_rng(99017)
    grammar = audit_grammar(args, rng)
    memory = audit_memory(args, rng)
    payload = {
        "panel_size": args.panel_size, "top_neurons": args.top_neurons,
        "permutations": args.permutations,
        "interaction_definition": "J=m11-m10-m01+m00; normalized by |A alone|+|B alone|+|J|",
        "rows": grammar + memory,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
