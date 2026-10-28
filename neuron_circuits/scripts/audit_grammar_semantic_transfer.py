#!/usr/bin/env python3
"""Causally test whether grammar transformations transfer across entities.

For each direct relation and value transition, estimate a post-product MLP
activation delta on a discovery set.  Add that frozen delta to unrelated base
prompts and ask whether it reproduces the counterfactual answer change.  The
same delta is restricted to native channels, random native channels, or an
equally-sized random rotated basis to test neuron-axis privilege.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, load_manifest
from neuron_circuits.grammar_model import ActivationPatch, GrammarTransformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--examples", type=int, default=192,
                        help="Examples per relation/value transition before a 50/50 split.")
    parser.add_argument("--channels", type=int, nargs="+", default=(8, 32))
    parser.add_argument("--templates", type=int, nargs="+", default=(0, 6))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-underfit", action="store_true",
                        help="Run exploratory audits even when the preregistered fidelity gate fails.")
    return parser.parse_args()


def load_model(path: Path, device: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GrammarTransformer(
        VOCAB.size, checkpoint["width"], checkpoint["heads"], checkpoint["depth"],
        mlp_kind=checkpoint["mlp_kind"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def entity_from_digits(digits: np.ndarray) -> np.ndarray:
    return (digits[:, 0] * 64 + digits[:, 1] * 8 + digits[:, 2]).astype(np.int32)


def make_transition_examples(rng: np.random.Generator, relation: int, before: int,
                             after: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return base/counterfactual facts differing in the queried direct digit."""
    subject = rng.integers(0, 8, size=(count, 3), dtype=np.int32)
    obj = rng.integers(0, 8, size=(count, 3), dtype=np.int32)
    role = "subject" if relation < 2 else "object"
    position = relation if relation < 2 else relation - 2
    base_subject, changed_subject = subject.copy(), subject.copy()
    base_object, changed_object = obj.copy(), obj.copy()
    if role == "subject":
        base_subject[:, position] = before; changed_subject[:, position] = after
    else:
        base_object[:, position] = before; changed_object[:, position] = after
    base = np.column_stack((entity_from_digits(base_subject), entity_from_digits(base_object),
                            np.full(count, relation, dtype=np.int32)))
    changed = np.column_stack((entity_from_digits(changed_subject), entity_from_digits(changed_object),
                               np.full(count, relation, dtype=np.int32)))
    return base, changed


@torch.inference_mode()
def forward(model, prompts: np.ndarray, device: str, patch: ActivationPatch | None = None):
    tokens = torch.from_numpy(prompts.astype(np.int64)).to(device)
    logits, activations = model(tokens, patch=patch, return_activations=True)
    return logits[:, -1, list(VOCAB.answers)].float(), activations


def evaluate_patch(model, prompts, values, layer, old_answers, new_answers, device):
    patched, _ = forward(model, prompts, device, ActivationPatch(layer, values, additive=True))
    rows = torch.arange(len(patched), device=patched.device)
    margins = patched[rows, new_answers] - patched[rows, old_answers]
    return patched.argmax(1), margins


def summarize_patch(name, prediction, margin, base_margin, natural_margin, target):
    delta = (margin - base_margin).detach().cpu().numpy()
    denominator = (natural_margin - base_margin).detach().cpu().numpy()
    valid = np.abs(denominator) > 1e-6
    recovery = delta[valid] / denominator[valid]
    return {
        "method": name,
        "count": int(len(target)),
        "target_accuracy": float((prediction == target).float().mean()),
        "mean_margin_change": float(delta.mean()),
        "median_fraction_natural_margin_recovered": float(np.median(recovery)),
        "mean_fraction_natural_margin_recovered": float(np.mean(recovery)),
    }


def audit_condition(root: Path, run: dict, manifest: dict, args) -> dict:
    destination = root / f"rho_{run['rho']:g}" / f"depth_{run['depth']}"
    model = load_model(destination / "checkpoint.pt", args.device)
    permutations = np.asarray(manifest["permutations"])
    rng = np.random.default_rng(72000 + run["depth"] * 100 + round(run["rho"] * 10))
    transitions = []
    for relation in range(4):
        for before in (0, 2, 4, 6):
            after = before + 1
            base_facts, changed_facts = make_transition_examples(rng, relation, before, after, args.examples)
            old_answer = int(permutations[relation, before])
            new_answer = int(permutations[relation, after])
            split = args.examples // 2
            for template in args.templates:
                base_prompts = encode_prompts(manifest, base_facts, template)
                changed_prompts = encode_prompts(manifest, changed_facts, template)
                base_logits, base_acts = forward(model, base_prompts, args.device)
                changed_logits, changed_acts = forward(model, changed_prompts, args.device)
                eligible = ((base_logits.argmax(1) == old_answer) &
                            (changed_logits.argmax(1) == new_answer))
                eval_keep = torch.arange(args.examples, device=args.device) >= split
                keep = eligible & eval_keep
                record = {
                    "relation": relation, "before": before, "after": after,
                    "template": template, "old_answer": old_answer, "new_answer": new_answer,
                    "natural_base_accuracy": float((base_logits.argmax(1) == old_answer).float().mean()),
                    "natural_changed_accuracy": float((changed_logits.argmax(1) == new_answer).float().mean()),
                    "eligible_evaluation_count": int(keep.sum()), "layers": [],
                }
                if keep.sum() < 16:
                    transitions.append(record); continue
                old = torch.full((int(keep.sum()),), old_answer, device=args.device)
                new = torch.full((int(keep.sum()),), new_answer, device=args.device)
                rows = torch.arange(args.examples, device=args.device)
                base_margin_all = base_logits[rows, new_answer] - base_logits[rows, old_answer]
                natural_margin_all = changed_logits[rows, new_answer] - changed_logits[rows, old_answer]
                for layer in range(model.layers_count):
                    # Selection is frozen on the first half; intervention is evaluated
                    # on disjoint identities in the second half.
                    discovery_delta = (changed_acts[layer][:split] - base_acts[layer][:split]).mean(0)
                    full_delta = discovery_delta[None, :].expand(int(keep.sum()), -1)
                    generator = torch.Generator(device="cpu").manual_seed(
                        73000 + layer * 1000 + relation * 100 + before
                    )
                    rotation = torch.linalg.qr(torch.randn(model.channels, model.channels, generator=generator)).Q.to(args.device)
                    rotated_delta = discovery_delta @ rotation
                    base_eval = base_prompts[keep.cpu().numpy()]
                    base_margin = base_margin_all[keep]
                    natural_margin = natural_margin_all[keep]
                    methods = []
                    prediction, margin = evaluate_patch(model, base_eval, full_delta, layer, old, new, args.device)
                    methods.append(summarize_patch("full", prediction, margin, base_margin, natural_margin, new))
                    for channel_count in args.channels:
                        k = min(channel_count, model.channels)
                        native_indices = torch.topk(discovery_delta.abs(), k).indices
                        native_values = torch.zeros_like(full_delta)
                        native_values[:, native_indices] = discovery_delta[native_indices]
                        prediction, margin = evaluate_patch(model, base_eval, native_values, layer, old, new, args.device)
                        methods.append(summarize_patch(f"native_top_{k}", prediction, margin, base_margin, natural_margin, new))

                        random_indices = torch.randperm(model.channels, generator=generator)[:k].to(args.device)
                        random_values = torch.zeros_like(full_delta)
                        random_values[:, random_indices] = discovery_delta[random_indices]
                        prediction, margin = evaluate_patch(model, base_eval, random_values, layer, old, new, args.device)
                        methods.append(summarize_patch(f"native_random_{k}", prediction, margin, base_margin, natural_margin, new))

                        rotated_indices = torch.topk(rotated_delta.abs(), k).indices
                        retained_rotated = torch.zeros_like(rotated_delta)
                        retained_rotated[rotated_indices] = rotated_delta[rotated_indices]
                        rotated_values = (retained_rotated @ rotation.T)[None, :].expand_as(full_delta)
                        prediction, margin = evaluate_patch(model, base_eval, rotated_values, layer, old, new, args.device)
                        methods.append(summarize_patch(f"rotated_top_{k}", prediction, margin, base_margin, natural_margin, new))
                    record["layers"].append({"layer": layer, "methods": methods})
                transitions.append(record)
    return {
        "rho": run["rho"], "depth": run["depth"], "mlp_kind": run["mlp_kind"],
        "fidelity_gate_passed": bool(run["best"]["fixed_accuracy"] >= .97 and run["best"]["transfer_accuracy"] >= .95),
        "examples_per_transition": args.examples, "transitions": transitions,
    }


def main():
    args = parse_args()
    manifest = load_manifest(args.root / "manifest.npz")
    runs = json.loads((args.root / "pilot_results.json").read_text())["runs"]
    for run in runs:
        destination = args.root / f"rho_{run['rho']:g}" / f"depth_{run['depth']}" / "semantic_transfer.json"
        if destination.exists() and not args.force:
            print(f"SKIP {destination}", flush=True); continue
        if (not args.include_underfit and
                (run["best"]["fixed_accuracy"] < .97 or run["best"]["transfer_accuracy"] < .95)):
            print(f"UNDERFIT rho={run['rho']:g} depth={run['depth']}", flush=True); continue
        print(f"START rho={run['rho']:g} depth={run['depth']}", flush=True)
        result = audit_condition(args.root, run, manifest, args)
        destination.write_text(json.dumps(result), encoding="utf-8")
        eligible = sum(row["eligible_evaluation_count"] for row in result["transitions"])
        print(f"DONE rho={run['rho']:g} depth={run['depth']} eligible={eligible}", flush=True)


if __name__ == "__main__":
    main()
