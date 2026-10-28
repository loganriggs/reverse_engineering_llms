#!/usr/bin/env python3
"""Measure exact per-example MLP circuit size, axis privilege, and reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import GrammarTransformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--rho", type=float, default=.5)
    parser.add_argument("--panel-per-group", type=int, default=12)
    parser.add_argument("--rotation-seed", type=int, default=73001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GrammarTransformer(
        VOCAB.size, checkpoint["width"], checkpoint["heads"], checkpoint["depth"],
        mlp_kind=checkpoint["mlp_kind"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def answer_margin(logits: torch.Tensor, target: int, competitor: int | None = None):
    answers = logits[..., VOCAB.answers]
    if competitor is None:
        rivals = answers.detach().clone()
        rivals[..., target] = -torch.inf
        competitor = int(rivals.argmax(-1).item())
    return answers[..., target] - answers[..., competitor], competitor


def native_masks(model, indices: np.ndarray, keep: bool, device: str):
    masks = {layer: torch.zeros(model.channels, device=device) for layer in range(model.layers_count)}
    if not keep:
        for mask in masks.values(): mask.fill_(1)
    for flat in indices:
        layer, channel = divmod(int(flat), model.channels)
        masks[layer][channel] = float(keep)
    return masks


def rotated_projectors(model, indices: np.ndarray, rotations: list[torch.Tensor], keep: bool, device: str):
    selected = [torch.zeros(model.channels, dtype=torch.bool, device=device) for _ in range(model.layers_count)]
    for flat in indices:
        layer, channel = divmod(int(flat), model.channels)
        selected[layer][channel] = True
    projectors = {}
    for layer, rotation in enumerate(rotations):
        basis = rotation[:, selected[layer]]
        projector = basis @ basis.T
        if not keep: projector = torch.eye(model.channels, device=device) - projector
        projectors[layer] = projector
    return projectors


def exact_margin(model, tokens, target, competitor, indices, keep, rotations=None):
    with torch.inference_mode():
        if rotations is None:
            logits = model(tokens, channel_masks=native_masks(model, indices, keep, tokens.device))
        else:
            logits = model(tokens, channel_projectors=rotated_projectors(
                model, indices, rotations, keep, tokens.device,
            ))
    margin, _ = answer_margin(logits[0, -1], target, competitor)
    return float(margin)


def candidate_counts(total: int):
    counts = set(range(min(64, total) + 1))
    counts.update(int(round(value)) for value in np.linspace(0, total, 65))
    counts.update({0, total})
    value = 1
    while value < total:
        counts.add(value); value *= 2
    return sorted(counts)


def threshold_count(evaluate, full: float, zero: float, mode: str, threshold: float = .9):
    denominator = full - zero
    cache = {}

    def score(count):
        if count not in cache:
            margin = evaluate(count)
            recovery = ((margin - zero) / denominator if mode == "keep"
                        else (full - margin) / denominator)
            cache[count] = {"margin": margin, "fraction": recovery}
        return cache[count]["fraction"]

    grid = candidate_counts(evaluate.total)
    crossing = None
    for count in grid:
        if score(count) >= threshold:
            crossing = count if crossing is None else min(crossing, count)
    return crossing, {str(key): cache[key] for key in sorted(cache)}


class PrefixEvaluation:
    def __init__(self, model, tokens, target, competitor, order, keep, rotations):
        self.model, self.tokens = model, tokens
        self.target, self.competitor = target, competitor
        self.order, self.keep, self.rotations = order, keep, rotations
        self.total = len(order)

    def __call__(self, count):
        return exact_margin(
            self.model, self.tokens, self.target, self.competitor,
            self.order[:count], self.keep, self.rotations,
        )


def rotations_for(model, seed: int, device: str):
    output = []
    for layer in range(model.layers_count):
        generator = torch.Generator(device=device).manual_seed(seed + layer)
        matrix = torch.randn(model.channels, model.channels, generator=generator, device=device)
        output.append(torch.linalg.qr(matrix).Q)
    return output


def audit_example(model, tokens, target: int, rotations):
    model.zero_grad(set_to_none=True)
    logits, activations = model(tokens, return_full_activations=True)
    margin, competitor = answer_margin(logits[0, -1], target)
    gradients = torch.autograd.grad(margin, activations)
    native_contributions = torch.cat([
        (activation[0] * gradient[0]).sum(0) for activation, gradient in zip(activations, gradients)
    ])
    rotated_contributions = torch.cat([
        ((activation[0] @ rotation) * (gradient[0] @ rotation)).sum(0)
        for activation, gradient, rotation in zip(activations, gradients, rotations)
    ])
    native_order = torch.argsort(native_contributions.abs(), descending=True).cpu().numpy()
    rotated_order = torch.argsort(rotated_contributions.abs(), descending=True).cpu().numpy()
    full = float(margin.detach())
    zero = exact_margin(model, tokens, target, competitor, np.array([], dtype=int), True)
    denominator = full - zero
    result = {
        "target": target, "competitor": competitor, "full_margin": full,
        "zero_mlp_margin": zero, "mlp_added_margin": denominator,
        "valid": bool(full > 0 and denominator > 1e-6),
    }
    if not result["valid"]: return result, None
    for basis, order, basis_rotations in (
        ("native", native_order, None), ("rotated", rotated_order, rotations),
    ):
        keep = PrefixEvaluation(model, tokens, target, competitor, order, True, basis_rotations)
        remove = PrefixEvaluation(model, tokens, target, competitor, order, False, basis_rotations)
        s90, sufficient_curve = threshold_count(keep, full, zero, "keep")
        n90, necessary_curve = threshold_count(remove, full, zero, "remove")
        result[basis] = {
            "s90": s90, "n90": n90,
            "s90_fraction": None if s90 is None else s90 / len(order),
            "n90_fraction": None if n90 is None else n90 / len(order),
            "sufficient_curve": sufficient_curve, "necessary_curve": necessary_curve,
        }
        if s90 is not None:
            result[basis]["s90_layer_counts"] = np.bincount(
                order[:s90] // model.channels, minlength=model.layers_count,
            ).tolist()
    result["native"]["top_channels"] = native_order[:16].tolist()
    result["native"]["top_signed_first_order_contributions"] = (
        native_contributions[native_order[:16]].detach().cpu().tolist()
    )
    result["axis_advantage"] = None
    if result["native"]["s90"] is not None and result["rotated"]["s90"] is not None:
        result["axis_advantage"] = float(np.log2(
            (result["rotated"]["s90"] + 1) / (result["native"]["s90"] + 1)
        ))
    return result, native_order[:result["native"]["s90"]] if result["native"]["s90"] is not None else None


def choose_panel(indices, count, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(indices, min(count, len(indices)), replace=False) if len(indices) else np.array([], dtype=int)


def summarize(rows):
    valid = [row for row in rows if row["valid"]]
    def values(path):
        output = []
        for row in valid:
            value = row
            for key in path: value = value[key]
            if value is not None: output.append(value)
        return np.asarray(output, dtype=float)
    result = {"examples": len(rows), "valid_examples": len(valid)}
    for basis, curve_name in (("native", "sufficient_curve"), ("native", "necessary_curve"),
                              ("rotated", "sufficient_curve")):
        reversals = []
        for row in valid:
            curve = row[basis][curve_name]
            curve_values = np.asarray([curve[key]["fraction"] for key in sorted(curve, key=int)])
            reversals.append(bool(len(curve_values) > 1 and np.any(np.diff(curve_values) < -.05)))
        result[f"{basis}_{curve_name}_nonmonotonic_fraction"] = (
            None if not reversals else float(np.mean(reversals))
        )
    for name, path in {
        "native_s90": ("native", "s90"), "native_n90": ("native", "n90"),
        "native_s90_fraction": ("native", "s90_fraction"),
        "native_n90_fraction": ("native", "n90_fraction"),
        "rotated_s90": ("rotated", "s90"), "axis_advantage": ("axis_advantage",),
    }.items():
        array = values(path)
        result[name] = None if not len(array) else {
            "median": float(np.median(array)), "mean": float(np.mean(array)),
            "q25": float(np.quantile(array, .25)), "q75": float(np.quantile(array, .75)),
            "count": len(array),
        }
    return result


def reuse_scores(model, records, facts, labels, prompts, eligible, group: str, device: str):
    valid = [(record, mask) for record, mask in records if mask is not None]
    scores = []
    for donor_position, (donor, mask) in enumerate(valid):
        donor_index = donor["index"]
        if group in {"train_unselected", "heldout_unselected"}:
            match = ((facts[:, 2] == facts[donor_index, 2]) &
                     (labels == labels[donor_index]))
        else:
            # Hold the output class fixed but change the semantic relation.
            match = ((labels == labels[donor_index]) &
                     (facts[:, 2] != facts[donor_index, 2]))
        candidates = np.flatnonzero(eligible & match & (np.arange(len(facts)) != donor_index))
        if not len(candidates): continue
        recipient_index = int(candidates[donor_position % len(candidates)])
        tokens = torch.from_numpy(prompts[recipient_index:recipient_index + 1].astype(np.int64)).to(device)
        with torch.inference_mode(): intact_logits = model(tokens)
        intact_margin, competitor = answer_margin(intact_logits[0, -1], int(labels[recipient_index]))
        full = float(intact_margin)
        zero = exact_margin(model, tokens, int(labels[recipient_index]), competitor,
                            np.array([], dtype=int), True)
        denominator = full - zero
        if full <= 0 or denominator <= 1e-6: continue
        margin = exact_margin(
            model, tokens, int(labels[recipient_index]), competitor, mask, True,
        )
        fraction = (margin - zero) / denominator
        scores.append(float(fraction))
    return {
        "pairs": len(scores), "mean_recovery": None if not scores else float(np.mean(scores)),
        "median_recovery": None if not scores else float(np.median(scores)),
        "fraction_at_least_90_percent": None if not scores else float(np.mean(np.asarray(scores) >= .9)),
        "recoveries": scores,
    }


def main():
    args = parse_args()
    destination = args.root / f"rho_{args.rho:g}" / f"depth_{args.depth}"
    output_path = destination / "causal_circuit_audit.json"
    if output_path.exists() and not args.force:
        print(f"SKIP {output_path}"); return
    manifest = load_manifest(args.root / "manifest.npz")
    model, checkpoint = load_model(destination / "checkpoint.pt", args.device)
    rotations = rotations_for(model, args.rotation_seed + checkpoint["seed"] * 100, args.device)
    groups = {}
    for split in ("train", "heldout_pair"):
        facts = np.asarray(manifest[f"{split}_facts"])
        labels, selected, changed = labels_for_rho(manifest, split, args.rho)
        prompts = encode_prompts(manifest, facts, 6)
        with np.load(destination / "predictions.npz") as stored:
            prediction = stored[f"{split}_template6_prediction"]
        correct = prediction == labels
        if split == "train":
            definitions = {"train_changed": changed, "train_unselected": ~selected}
        else:
            definitions = {"heldout_unselected": ~selected}
        for group_index, (name, membership) in enumerate(definitions.items()):
            panel = choose_panel(np.flatnonzero(correct & membership), args.panel_per_group,
                                 81000 + checkpoint["seed"] * 100 + group_index)
            rows, records = [], []
            for index in panel:
                tokens = torch.from_numpy(prompts[index:index + 1].astype(np.int64)).to(args.device)
                row, mask = audit_example(model, tokens, int(labels[index]), rotations)
                row.update({"index": int(index), "fact": facts[index].tolist(),
                            "selected": bool(selected[index]), "changed": bool(changed[index])})
                rows.append(row)
                reusable = dict(row)
                reusable["tokens"] = prompts[index]
                records.append((reusable, mask))
            groups[name] = {
                "summary": summarize(rows), "examples": rows,
                "reuse": reuse_scores(
                    model, records, facts, labels, prompts, correct & membership, name, args.device,
                ),
            }
    payload = {
        "root": str(args.root), "width": checkpoint["width"], "depth": checkpoint["depth"],
        "mlp_kind": checkpoint["mlp_kind"], "seed": checkpoint["seed"], "rho": args.rho,
        "channels_per_layer": model.channels, "total_neurons": model.channels * model.layers_count,
        "panel_per_group": args.panel_per_group, "rotation_seed": args.rotation_seed,
        "reuse_matching": "structured: relation+answer; exception control: answer+different-relation",
        "metric_note": "Smallest crossing on a fixed dense ranked-prefix grid; a resolution-limited upper-bound estimate, not a global minimum.",
        "groups": groups,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DONE {output_path}")


if __name__ == "__main__":
    main()
