#!/usr/bin/env python3
"""Train and evaluate the amended factual-capacity Stage 1 pilot."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.factual_data import (
    ATTRIBUTES, HELDOUT_TEMPLATES, TRAIN_TEMPLATES, VOCAB, build_manifest,
    encode_prompts, labels_for_rho, manifest_checks, save_manifest,
)
from neuron_circuits.factual_model import FactualTransformer, Intervention


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/factual_capacity_pilot"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--widths", type=int, nargs="+", default=(32, 64, 128, 256))
    parser.add_argument("--rhos", type=float, nargs="+", default=(0.0, 0.5, 1.0))
    parser.add_argument("--initial-steps", type=int, default=2000)
    parser.add_argument("--maximum-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an existing checkpoint with fresh optimizer state")
    return parser.parse_args()


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()


def pair_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask.reshape(-1))


def build_prompt_cache(manifest: dict) -> list[np.ndarray]:
    entities = np.repeat(np.arange(len(manifest["traits"])), ATTRIBUTES)
    attributes = np.tile(np.arange(ATTRIBUTES), len(manifest["traits"])).astype(np.uint8)
    return [encode_prompts(manifest["ids"][entities], manifest["traits"][entities], attributes, template) for template in range(8)]


def tiny_overfit_check(manifest, cache, device: str) -> dict:
    """Verify that the training path can store a small fixed random table."""
    seed_everything(4242)
    labels, _, _ = labels_for_rho(manifest, 1.0)
    indices = pair_indices(manifest["train_pairs"])[:128]
    tokens = torch.from_numpy(cache[0][indices].astype(np.int64)).to(device)
    targets = torch.from_numpy(labels.reshape(-1)[indices].astype(np.int64)).to(device)
    model = FactualTransformer(VOCAB.size, 32, 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    for step in range(1, 2001):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device): loss = F.cross_entropy(model(tokens), targets)
        loss.backward(); optimizer.step()
        with torch.inference_mode(), autocast(device): accuracy = float((model(tokens).argmax(1) == targets).float().mean())
        if accuracy == 1.0: break
    return {"passed": accuracy == 1.0, "accuracy": accuracy, "steps": step, "examples": len(indices)}


@torch.inference_mode()
def predict(model, prompts: np.ndarray, device: str, batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    predictions, losses = [], []
    # Targets are deliberately handled by the caller so this function can be
    # reused by the mask checks.
    for start in range(0, len(prompts), batch_size):
        tokens = torch.from_numpy(prompts[start:start + batch_size].astype(np.int64)).to(device)
        with autocast(device): logits = model(tokens)
        predictions.append(logits.softmax(-1).cpu().numpy())
    probabilities = np.concatenate(predictions)
    return probabilities.argmax(1).astype(np.uint8), probabilities


def summarize_predictions(targets, predictions, probabilities, selected, changed) -> dict:
    rows = {}
    groups = {
        "all": np.ones(len(targets), dtype=bool), "selected": selected,
        "unselected": ~selected, "changed": changed, "coincidence": selected & ~changed,
    }
    for name, keep in groups.items():
        if not np.any(keep): continue
        target = targets[keep]
        probability = probabilities[keep]
        rows[name] = {
            "count": int(keep.sum()), "accuracy": float(np.mean(predictions[keep] == target)),
            "answer_kl": float(np.mean(-np.log(np.clip(probability[np.arange(len(target)), target], 1e-12, 1)))),
        }
    return rows


def evaluate_split(model, cache, indices, labels, selected, changed, templates, device):
    targets = labels.reshape(-1)[indices]
    result = {}
    stored = {}
    for template in templates:
        prediction, probability = predict(model, cache[template][indices], device)
        summary = summarize_predictions(
            targets, prediction, probability, selected.reshape(-1)[indices], changed.reshape(-1)[indices],
        )
        summary["per_attribute"] = {
            str(attribute): summarize_predictions(
                targets[indices % ATTRIBUTES == attribute], prediction[indices % ATTRIBUTES == attribute],
                probability[indices % ATTRIBUTES == attribute], selected.reshape(-1)[indices][indices % ATTRIBUTES == attribute],
                changed.reshape(-1)[indices][indices % ATTRIBUTES == attribute],
            )
            for attribute in range(ATTRIBUTES)
        }
        result[str(template)] = summary
        stored[str(template)] = {"prediction": prediction, "probability": probability.astype(np.float16)}
    return result, stored


def checkpoint_score(model, cache, validation_indices, labels, device):
    targets = labels.reshape(-1)[validation_indices]
    accuracies = []
    losses = []
    for template in (0, 6):
        prediction, probability = predict(model, cache[template][validation_indices], device)
        accuracies.append(float(np.mean(prediction == targets)))
        losses.append(float(np.mean(-np.log(np.clip(probability[np.arange(len(targets)), targets], 1e-12, 1)))))
    return {"fixed_accuracy": accuracies[0], "transfer_accuracy": accuracies[1], "mean_accuracy": float(np.mean(accuracies)), "mean_loss": float(np.mean(losses))}


def train_one(args, manifest, cache, rho: float, width: int, heads: int, destination: Path):
    seed_everything(args.seed)
    labels, selected, changed = labels_for_rho(manifest, rho)
    train_indices = pair_indices(manifest["train_pairs"])
    validation_rng = np.random.default_rng(args.seed + 9000)
    validation_indices = validation_rng.choice(train_indices, min(8192, len(train_indices)), replace=False)
    model = FactualTransformer(VOCAB.size, width, heads).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)
    start_step = 0
    history, best, best_state = [], None, None
    previous_elapsed = 0.0
    if args.resume and (destination / "checkpoint.pt").exists():
        checkpoint = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        previous = json.loads((destination / "results.json").read_text())
        start_step = int(previous["steps"]); history = list(previous["history"])
        best = dict(previous["best"]); best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        previous_elapsed = float(previous["elapsed_seconds"])
    generator = np.random.default_rng(args.seed + 10000 + start_step)
    started = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    step = start_step
    recent = []
    while step < args.maximum_steps:
        step += 1
        chosen = generator.choice(train_indices, args.batch_size, replace=True)
        template_ids = generator.choice(TRAIN_TEMPLATES, args.batch_size, replace=True)
        batch = np.empty((args.batch_size, cache[0].shape[1]), dtype=np.uint8)
        for template in TRAIN_TEMPLATES:
            keep = template_ids == template
            if np.any(keep): batch[keep] = cache[template][chosen[keep]]
        tokens = torch.from_numpy(batch.astype(np.int64)).to(args.device)
        targets = torch.from_numpy(labels.reshape(-1)[chosen].astype(np.int64)).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.device):
            logits = model(tokens)
            loss = F.cross_entropy(logits, targets)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        recent.append(float(loss.detach()))
        if step % 250 == 0 or step == args.initial_steps:
            score = checkpoint_score(model, cache, validation_indices, labels, args.device)
            entry = {"step": step, "train_batch_loss": float(np.mean(recent)), **score}
            high_fidelity = score["fixed_accuracy"] >= 0.98 and score["transfer_accuracy"] >= 0.95
            if high_fidelity:
                full_score = checkpoint_score(model, cache, train_indices, labels, args.device)
                entry["full_panel_gate"] = full_score
                high_fidelity = full_score["fixed_accuracy"] >= 0.98 and full_score["transfer_accuracy"] >= 0.95
            history.append(entry); recent = []
            if best is None or (score["mean_accuracy"], -score["mean_loss"]) > (best["mean_accuracy"], -best["mean_loss"]):
                best = {"step": step, **score}
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if step >= args.initial_steps:
                # Every condition continues until the same fidelity gate or the
                # disclosed maximum; otherwise optimization speed would be
                # mistaken for representational capacity.
                if high_fidelity:
                    best = {"step": step, **score, "full_panel_gate": full_score}
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    break
    elapsed = previous_elapsed + time.perf_counter() - started
    assert best_state is not None
    model.load_state_dict(best_state)

    # Exact intervention contract on a deterministic mini-batch.
    probe = torch.from_numpy(cache[0][train_indices[:16]].astype(np.int64)).to(args.device)
    with torch.inference_mode(), autocast(args.device):
        intact = model(probe)
        masks = [torch.ones(model.channels, device=args.device) for _ in model.blocks]
        all_one_equivalent = model(probe, channel_masks=masks)
        masks[0][0] = 0
        ablated = model(probe, channel_masks=masks)
    hook_checks = {
        "no_intervention_reproduces_logits": bool(torch.equal(intact, all_one_equivalent)),
        "single_channel_changes_logits": bool(not torch.equal(intact, ablated)),
    }

    split_masks = {
        "seen_fact": manifest["train_pairs"], "heldout_pair": manifest["heldout_pairs"],
        "heldout_entity": np.repeat(manifest["heldout_entities"][:, None], ATTRIBUTES, axis=1),
    }
    evaluation, prediction_payload = {}, {}
    for split, mask in split_masks.items():
        indices = pair_indices(mask)
        templates = (0, 6, 7) if split == "seen_fact" else (0, 6)
        evaluation[split], prediction_payload[split] = evaluate_split(model, cache, indices, labels, selected, changed, templates, args.device)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "width": width, "heads": heads, "rho": rho, "seed": args.seed}, destination / "checkpoint.pt")
    # Per-example compact output: indices, target, metadata, and predictions.
    arrays = {}
    for split, mask in split_masks.items():
        indices = pair_indices(mask)
        arrays[f"{split}_indices"] = indices.astype(np.int32)
        arrays[f"{split}_targets"] = labels.reshape(-1)[indices]
        arrays[f"{split}_selected"] = selected.reshape(-1)[indices]
        arrays[f"{split}_changed"] = changed.reshape(-1)[indices]
        for template, values in prediction_payload[split].items():
            arrays[f"{split}_template{template}_prediction"] = values["prediction"]
    np.savez_compressed(destination / "predictions.npz", **arrays)
    metadata = {
        "rho": rho, "width": width, "heads": heads, "seed": args.seed,
        "learning_rate": args.learning_rate,
        "manifest_hash": manifest["hash"], "parameter_counts": model.component_counts(),
        "best": best, "history": history, "elapsed_seconds": elapsed,
        "steps": step, "examples_seen": step * args.batch_size,
        "exposures_per_train_fact": step * args.batch_size / len(train_indices),
        "training_tokens": step * args.batch_size * cache[0].shape[1],
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "hook_checks": hook_checks, "evaluation": evaluation,
    }
    json_dump(destination / "results.json", metadata)
    return metadata


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.seed, args.replicas)
    save_manifest(args.output / "manifest.npz", manifest)
    checks = manifest_checks(manifest)
    for rho in args.rhos:
        labels, selected, _ = labels_for_rho(manifest, rho)
        checks[f"rho_{rho:g}_answer_balanced"] = all(np.ptp(np.bincount(labels[:, a], minlength=8)) == 0 for a in range(ATTRIBUTES))
        checks[f"rho_{rho:g}_selected_fraction"] = float(selected.mean())
    cache = build_prompt_cache(manifest)
    checks["tiny_random_overfit"] = tiny_overfit_check(manifest, cache, args.device)
    json_dump(args.output / "stage0_checks.json", checks)
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)] + [checks["tiny_random_overfit"]["passed"]]
    if not all(boolean_checks): raise RuntimeError(f"Stage 0 checks failed: {checks}")
    all_results = []
    heads_for_width = {32: 1, 64: 2, 128: 4, 256: 8}
    for rho in args.rhos:
        for width in args.widths:
            destination = args.output / f"rho_{rho:g}" / f"width_{width}"
            if (destination / "results.json").exists() and not args.force and not args.resume:
                print(f"SKIP rho={rho:g} width={width}", flush=True)
                all_results.append(json.loads((destination / "results.json").read_text()))
                continue
            print(f"START rho={rho:g} width={width}", flush=True)
            result = train_one(args, manifest, cache, rho, width, heads_for_width[width], destination)
            all_results.append(result)
            print(f"DONE rho={rho:g} width={width} step={result['steps']} mean_acc={result['best']['mean_accuracy']:.4f} elapsed={result['elapsed_seconds']:.1f}s", flush=True)
    json_dump(args.output / "pilot_results.json", {"stage0": checks, "runs": all_results})


if __name__ == "__main__": main()
