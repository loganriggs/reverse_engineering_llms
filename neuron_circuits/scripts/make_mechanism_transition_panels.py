#!/usr/bin/env python3
"""Create fact- and competitor-matched panels for the transition factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from audit_grammar_constraints import answer_logits, load_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/mechanism_transition/factorial"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def condition_root(root: Path, seed: int, optimizer: str, load: int) -> Path:
    return root / f"seed{seed}_{optimizer}_load{load}"


@torch.inference_mode()
def probabilities(model, prompts: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    output = []
    for start in range(0, len(prompts), batch_size):
        tokens = torch.from_numpy(prompts[start:start + batch_size].astype(np.int64)).to(device)
        output.append(answer_logits(model, tokens).softmax(-1).cpu().numpy())
    return np.concatenate(output)


def make_panel(root: Path, seed: int, device: str) -> dict:
    conditions = [(optimizer, load) for optimizer in ("adamw", "sgd") for load in (64, 128)]
    manifests, eligible_maps, probability_maps = {}, {}, {}
    target_by_fact = {}
    common = None
    for optimizer, load in conditions:
        condition = condition_root(root, seed, optimizer, load)
        manifest = load_manifest(condition / "manifest.npz")
        labels, _, _ = labels_for_rho(manifest, "train", 1)
        facts = np.asarray(manifest["train_facts"])
        with np.load(condition / "rho_1" / "depth_4" / "predictions.npz") as stored:
            eligible = ((stored["train_template0_prediction"] == labels)
                        & (stored["train_template6_prediction"] == labels))
        eligible_facts = {tuple(map(int, fact)) for fact in facts[eligible]}
        common = eligible_facts if common is None else common & eligible_facts
        manifests[(optimizer, load)] = manifest
        eligible_maps[(optimizer, load)] = {tuple(map(int, fact)): index for index, fact in enumerate(facts)}
        for fact, label in zip(facts, labels):
            key = tuple(map(int, fact))
            if key in target_by_fact and target_by_fact[key] != int(label):
                raise AssertionError(f"unstable target for {key}")
            target_by_fact[key] = int(label)

    assert common is not None
    rng = np.random.default_rng(91000 + seed)
    selected = []
    for relation in range(8):
        for answer in range(8):
            candidates = sorted(fact for fact in common if fact[2] == relation and target_by_fact[fact] == answer)
            if not candidates:
                raise RuntimeError(f"no common eligible fact for seed={seed}, relation={relation}, answer={answer}")
            selected.append(candidates[int(rng.integers(len(candidates)))])

    probability_rows = []
    for optimizer, load in conditions:
        condition = condition_root(root, seed, optimizer, load)
        manifest = manifests[(optimizer, load)]
        local = eligible_maps[(optimizer, load)]
        indices = np.asarray([local[fact] for fact in selected])
        prompts = encode_prompts(manifest, np.asarray(manifest["train_facts"])[indices], 0)
        model = load_model(condition / "rho_1" / "depth_4" / "checkpoint.pt", device)
        probability_rows.append(probabilities(model, prompts, device))
        del model
    mean_probability = np.mean(probability_rows, axis=0)
    targets = np.asarray([target_by_fact[fact] for fact in selected], dtype=np.uint8)
    mean_probability[np.arange(len(targets)), targets] = -np.inf
    competitors = mean_probability.argmax(1).astype(np.uint8)
    return {
        "seed": seed,
        "conditions": [{"optimizer": optimizer, "load": load} for optimizer, load in conditions],
        "selection": "one common-correct fact per relation x arbitrary-answer stratum",
        "competitor": "strongest wrong answer under mean softmax probability across four matched models",
        "facts": [list(fact) for fact in selected],
        "targets": targets.tolist(),
        "competitors": competitors.tolist(),
    }


def main():
    args = parse_args()
    for seed in args.seeds:
        panel = make_panel(args.root, seed, args.device)
        path = args.root / f"seed{seed}_matched_panel.json"
        path.write_text(json.dumps(panel, indent=2), encoding="utf-8")
        print(f"WROTE {path} facts={len(panel['facts'])}", flush=True)


if __name__ == "__main__":
    main()
