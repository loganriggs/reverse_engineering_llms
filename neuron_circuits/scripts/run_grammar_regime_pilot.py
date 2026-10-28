#!/usr/bin/env python3
"""Train few-layer transformers on the compositional mini-language."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import (
    HELDOUT_TEMPLATES, TRAIN_TEMPLATES, VOCAB, build_manifest, encode_prompts,
    encode_sequences, labels_for_rho, language_model_weights, manifest_checks,
    load_manifest, restrict_train_rank, save_manifest,
)
from neuron_circuits.grammar_model import ActivationPatch, GrammarTransformer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/grammar_regimes/pilot"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-per-stratum", type=int, default=128)
    parser.add_argument("--manifest", type=Path,
                        help="Optional immutable parent manifest reused across load conditions.")
    parser.add_argument("--train-rank-limit", type=int,
                        help="Use a nested prefix of the parent manifest's training ranks.")
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--depths", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--rhos", type=float, nargs="+", default=(0.0, 0.5, 1.0))
    parser.add_argument("--mlp-kind", choices=("swiglu", "bilinear"), default="swiglu")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--answer-weight", type=float, default=4.0)
    parser.add_argument(
        "--loss-kind", choices=("language", "answer_only"), default="language",
        help="Train the full weighted next-token objective or only the final answer logits.",
    )
    parser.add_argument("--initial-steps", type=int, default=2000)
    parser.add_argument("--maximum-steps", type=int, default=20000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()


def build_cache(manifest: dict) -> dict[str, list[np.ndarray]]:
    return {
        split: [encode_prompts(manifest, manifest[f"{split}_facts"], template) for template in range(8)]
        for split in ("train", "heldout_pair", "heldout_entity")
    }


def weighted_language_loss(logits: torch.Tensor, sequence: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    target = sequence[:, 1:]
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    flat_weight = weights.reshape(-1)
    return (losses * flat_weight).sum() / flat_weight.sum().clamp_min(1)


@torch.inference_mode()
def predict_answers(model, prompts: np.ndarray, device: str, batch_size: int = 2048):
    probabilities = []
    answer_ids = torch.tensor(VOCAB.answers, device=device)
    for start in range(0, len(prompts), batch_size):
        tokens = torch.from_numpy(prompts[start:start + batch_size].astype(np.int64)).to(device)
        with autocast(device): logits = model(tokens)[:, -1]
        probabilities.append(logits[:, answer_ids].softmax(-1).cpu())
    probability = torch.cat(probabilities).numpy()
    return probability.argmax(1).astype(np.uint8), probability


@torch.inference_mode()
def evaluate(model, cache, manifest, rho: float, device: str) -> tuple[dict, dict]:
    output, arrays = {}, {}
    for split in ("train", "heldout_pair", "heldout_entity"):
        labels, selected, changed = labels_for_rho(manifest, split, rho)
        templates = (0, 6, 7) if split == "train" else (0, 6)
        output[split] = {}
        arrays[f"{split}_targets"] = labels
        arrays[f"{split}_selected"] = selected
        arrays[f"{split}_changed"] = changed
        for template in templates:
            prediction, probability = predict_answers(model, cache[split][template], device)
            groups = {}
            for name, keep in {
                "all": np.ones(len(labels), bool), "selected": selected,
                "unselected": ~selected, "changed": changed,
            }.items():
                if np.any(keep):
                    groups[name] = {
                        "count": int(keep.sum()), "accuracy": float(np.mean(prediction[keep] == labels[keep])),
                        "answer_ce": float(np.mean(-np.log(np.clip(probability[keep, labels[keep]], 1e-12, 1)))),
                    }
            output[split][str(template)] = groups
            arrays[f"{split}_template{template}_prediction"] = prediction
    return output, arrays


def score(model, cache, labels, indices, device):
    scores = []
    for template in (0, 6):
        prediction, probability = predict_answers(model, cache["train"][template][indices], device)
        scores.append((float(np.mean(prediction == labels[indices])),
                       float(np.mean(-np.log(np.clip(probability[np.arange(len(indices)), labels[indices]], 1e-12, 1))))))
    return {"fixed_accuracy": scores[0][0], "transfer_accuracy": scores[1][0],
            "mean_accuracy": float(np.mean([x[0] for x in scores])),
            "mean_answer_ce": float(np.mean([x[1] for x in scores]))}


def make_optimizer(args, model):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(.9, .95), weight_decay=.01)
    return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=.9, weight_decay=.0001)


def tiny_overfit(manifest, device: str, mlp_kind: str) -> dict:
    seed_all(991)
    facts = manifest["train_facts"][:64]
    labels, _, _ = labels_for_rho(manifest, "train", 1)
    sequence = encode_sequences(manifest, facts, labels[:64], 0)
    tokens = torch.from_numpy(sequence[:, :-1].astype(np.int64)).to(device)
    targets = torch.from_numpy(labels[:64].astype(np.int64)).to(device)
    model = GrammarTransformer(VOCAB.size, width=48, heads=3, layers=2, mlp_kind=mlp_kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    answer_ids = torch.tensor(VOCAB.answers, device=device)
    for step in range(1, 2001):
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)[:, -1, answer_ids]
        loss = F.cross_entropy(logits, targets); loss.backward(); optimizer.step()
        with torch.no_grad(): accuracy = float((logits.argmax(1) == targets).float().mean())
        if accuracy == 1: break
    return {"passed": accuracy == 1, "steps": step, "accuracy": accuracy}


def train_one(args, manifest, cache, rho: float, depth: int, destination: Path) -> dict:
    seed_all(args.seed + depth * 1000)
    labels, _, _ = labels_for_rho(manifest, "train", rho)
    model = GrammarTransformer(VOCAB.size, args.width, args.heads, depth, mlp_kind=args.mlp_kind).to(args.device)
    optimizer = make_optimizer(args, model)
    rng = np.random.default_rng(args.seed + depth * 10000 + round(rho * 100))
    validation = rng.choice(len(labels), min(4096, len(labels)), replace=False)
    history, best, best_state, recent = [], None, None, []
    started = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.maximum_steps + 1):
        chosen = rng.integers(0, len(labels), size=args.batch_size)
        templates = rng.choice(TRAIN_TEMPLATES, size=args.batch_size)
        sequences = np.empty((args.batch_size, 14), dtype=np.uint8)
        for template in TRAIN_TEMPLATES:
            keep = templates == template
            if np.any(keep):
                facts = manifest["train_facts"][chosen[keep]]
                sequences[keep] = encode_sequences(manifest, facts, labels[chosen[keep]], template)
        sequence = torch.from_numpy(sequences.astype(np.int64)).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.device):
            logits = model(sequence[:, :-1])
            if args.loss_kind == "language":
                weights = torch.from_numpy(language_model_weights(sequences, args.answer_weight)).to(args.device)
                loss = weighted_language_loss(logits, sequence, weights)
            else:
                answer_ids = torch.tensor(VOCAB.answers, device=args.device)
                loss = F.cross_entropy(logits[:, -1, answer_ids],
                                       torch.from_numpy(labels[chosen].astype(np.int64)).to(args.device))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        recent.append(float(loss.detach()))
        if step % args.validation_interval == 0 or step == args.initial_steps:
            current = score(model, cache, labels, validation, args.device)
            row = {"step": step, "train_loss": float(np.mean(recent)), **current}; recent = []
            history.append(row)
            if best is None or (current["mean_accuracy"], -current["mean_answer_ce"]) > (best["mean_accuracy"], -best["mean_answer_ce"]):
                best = {"step": step, **current}
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if step >= args.initial_steps and current["fixed_accuracy"] >= .97 and current["transfer_accuracy"] >= .95:
                full = score(model, cache, labels, np.arange(len(labels)), args.device)
                row["full_gate"] = full
                if full["fixed_accuracy"] >= .97 and full["transfer_accuracy"] >= .95:
                    best = {"step": step, **current, "full_gate": full}
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    break
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    evaluation, arrays = evaluate(model, cache, manifest, rho, args.device)

    probe = torch.from_numpy(cache["train"][0][:8].astype(np.int64)).to(args.device)
    with torch.inference_mode():
        intact, activations = model(probe, return_activations=True)
        same = model(probe, patch=ActivationPatch(0, activations[0]))
        changed = model(probe, patch=ActivationPatch(0, torch.zeros_like(activations[0])))
    checks = {"identity_patch_exact": bool(torch.equal(intact, same)),
              "zero_patch_changes_logits": bool(not torch.equal(intact, changed))}

    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = {"model": best_state, "width": args.width, "heads": args.heads,
                  "depth": depth, "mlp_kind": args.mlp_kind, "rho": rho, "seed": args.seed}
    torch.save(checkpoint, destination / "checkpoint.pt")
    np.savez_compressed(destination / "predictions.npz", **arrays)
    result = {
        "seed": args.seed, "rho": rho, "depth": depth, "width": args.width,
        "heads": args.heads, "mlp_kind": args.mlp_kind, "optimizer": args.optimizer,
        "loss_kind": args.loss_kind,
        "train_rank_limit": args.train_rank_limit,
        "learning_rate": args.learning_rate, "steps": step, "best": best, "history": history,
        "elapsed_seconds": elapsed, "examples_seen": step * args.batch_size,
        "exposures_per_fact": step * args.batch_size / len(labels),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "parameter_counts": model.component_counts(), "hook_checks": checks, "evaluation": evaluation,
        "manifest_hash": manifest["hash"],
    }
    dump(destination / "results.json", result)
    return result


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest) if args.manifest else build_manifest(args.seed, args.train_per_stratum)
    if args.train_rank_limit is not None:
        manifest = restrict_train_rank(manifest, args.train_rank_limit)
    save_manifest(args.output / "manifest.npz", manifest)
    checks = manifest_checks(manifest)
    for rho in args.rhos:
        labels, selected, _ = labels_for_rho(manifest, "train", rho)
        checks[f"rho_{rho:g}_balanced"] = all(
            np.ptp(np.bincount(labels[manifest["train_facts"][:, 2] == relation], minlength=8)) == 0
            for relation in range(8)
        )
        checks[f"rho_{rho:g}_selected_fraction"] = float(selected.mean())
    checks["tiny_overfit"] = tiny_overfit(manifest, args.device, args.mlp_kind)
    dump(args.output / "stage0_checks.json", checks)
    booleans = [value for value in checks.values() if isinstance(value, bool)] + [checks["tiny_overfit"]["passed"]]
    if not all(booleans): raise RuntimeError(checks)
    cache = build_cache(manifest)
    runs = []
    for rho in args.rhos:
        for depth in args.depths:
            destination = args.output / f"rho_{rho:g}" / f"depth_{depth}"
            if (destination / "results.json").exists() and not args.force:
                result = json.loads((destination / "results.json").read_text()); print(f"SKIP {destination}", flush=True)
            else:
                print(f"START rho={rho:g} depth={depth}", flush=True)
                result = train_one(args, manifest, cache, rho, depth, destination)
                print(f"DONE rho={rho:g} depth={depth} step={result['steps']} acc={result['best']['mean_accuracy']:.4f} elapsed={result['elapsed_seconds']:.1f}s", flush=True)
            runs.append(result)
    dump(args.output / "pilot_results.json", {"stage0": checks, "runs": runs})


if __name__ == "__main__":
    main()
