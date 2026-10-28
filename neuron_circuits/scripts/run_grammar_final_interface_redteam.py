#!/usr/bin/env python3
"""Distinguish reusable grammar rules from a generic final answer interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch, GrammarTransformer
from audit_grammar_semantic_transfer import make_transition_examples


CONFIG = Path("configs/grammar_final_interface_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/final_interface_redteam.json")
REPORT = Path("artifacts/grammar_regimes/FINAL_INTERFACE_REDTEAM.md")


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GrammarTransformer(
        VOCAB.size, checkpoint["width"], checkpoint["heads"], checkpoint["depth"],
        mlp_kind=checkpoint["mlp_kind"],
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.inference_mode()
def forward(model, prompts, device, patch=None):
    tokens = torch.as_tensor(prompts.astype(np.int64), device=device)
    logits, activations = model(tokens, patch=patch, return_activations=True)
    return logits[:, -1, list(VOCAB.answers)].float(), activations[-1].float()


def effect_similarity(base_logits, target, predicted):
    probability = F.softmax(base_logits.float(), -1)
    target = target - (probability * target).sum(-1, keepdim=True)
    predicted = predicted - (probability * predicted).sum(-1, keepdim=True)
    target_energy = (probability * target.square()).sum(-1).clamp_min(1e-30)
    predicted_energy = (probability * predicted.square()).sum(-1).clamp_min(1e-30)
    cosine = (probability * target * predicted).sum(-1) / torch.sqrt(
        target_energy * predicted_energy).clamp_min(1e-30)
    fidelity = 1 - (probability * (predicted - target).square()).sum(-1) / target_energy
    return float(cosine.median()), float(fidelity.median())


def restricted(delta, method, channels, seed):
    if method == "full":
        return delta
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if method == "native":
        indices = delta.abs().topk(channels).indices
        result = torch.zeros_like(delta); result[indices] = delta[indices]
        return result
    if method == "random":
        indices = torch.randperm(len(delta), generator=generator)[:channels].to(delta.device)
        result = torch.zeros_like(delta); result[indices] = delta[indices]
        return result
    if method == "rotated":
        rotation = torch.linalg.qr(torch.randn(len(delta), len(delta), generator=generator)).Q.to(delta.device)
        coordinates = delta @ rotation
        indices = coordinates.abs().topk(channels).indices
        kept = torch.zeros_like(coordinates); kept[indices] = coordinates[indices]
        return kept @ rotation.T
    raise ValueError(method)


@torch.inference_mode()
def patch_metrics(model, prompts, delta, target, base_logits, device, channels, seed):
    result = {}
    for number, method in enumerate(("full", "native", "random", "rotated")):
        values = restricted(delta, method, channels, seed + number)
        changed, _ = forward(model, prompts, device,
                             ActivationPatch(model.layers_count - 1, values, additive=True))
        rows = torch.arange(len(changed), device=device)
        result[method] = {
            "accuracy": float((changed.argmax(-1) == target).float().mean()),
            "target_margin": float((changed[rows, target]
                                    - changed.masked_fill(F.one_hot(target, 8).bool(), -torch.inf).max(-1).values).mean()),
            "effect": changed - base_logits,
            "indices": values.abs().topk(min(channels, len(values))).indices.cpu().tolist(),
        }
    return result


def summarize(rows, keys):
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


@torch.inference_mode()
def audit_run(spec, config, device):
    root = Path(spec["root"])
    rho = spec["rho"]
    manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    labels, _, _ = labels_for_rho(manifest, "train", rho)
    facts = np.asarray(manifest["train_facts"])
    prompt0 = encode_prompts(manifest, facts, config["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, config["evaluation_template"])
    logits0, acts0 = forward(model, prompt0, device)
    logits6, _ = forward(model, prompt6, device)
    labels_t = torch.as_tensor(labels, device=device)
    eligible = ((logits0.argmax(-1) == labels_t) & (logits6.argmax(-1) == labels_t)).cpu().numpy()
    rng = np.random.default_rng(config["seed"] + spec["seed"] * 100 + round(rho * 10))
    prototypes = {}
    evaluation = {}
    for answer in range(8):
        indices = np.flatnonzero(eligible & (labels == answer))
        rng.shuffle(indices)
        need = config["prototype_per_class"] + config["evaluation_per_class"]
        if len(indices) < need:
            raise RuntimeError(f"{spec['name']} answer {answer} has {len(indices)} eligible, need {need}")
        discovery = torch.as_tensor(indices[:config["prototype_per_class"]], device=device)
        prototypes[answer] = acts0[discovery].mean(0)
        evaluation[answer] = indices[config["prototype_per_class"]:need]

    answer_rows = []
    for old in range(8):
        new = (old + 1) % 8
        indices = evaluation[old]
        prompts = prompt6[indices]
        base = logits6[torch.as_tensor(indices, device=device)]
        target = torch.full((len(indices),), new, device=device, dtype=torch.long)
        delta = prototypes[new] - prototypes[old]
        methods = patch_metrics(model, prompts, delta, target, base, device,
                                config["channels"], config["seed"] + old * 100 + spec["seed"])
        answer_rows.append({
            "old": old, "new": new,
            "methods": {name: {key: value for key, value in row.items() if key != "effect"}
                        for name, row in methods.items()},
        })

    semantic_rows = []
    permutations = np.asarray(manifest["permutations"])
    semantic_rng = np.random.default_rng(config["seed"] + 10000 + spec["seed"] * 100)
    count = config["semantic_examples"]
    split = count // 2
    for relation in range(4):
        for before in (0, 2, 4, 6):
            after = before + 1
            base_facts, changed_facts = make_transition_examples(
                semantic_rng, relation, before, after, count,
            )
            old, new = int(permutations[relation, before]), int(permutations[relation, after])
            base0, changed0 = (encode_prompts(manifest, rows, config["discovery_template"])
                               for rows in (base_facts, changed_facts))
            base6, changed6 = (encode_prompts(manifest, rows, config["evaluation_template"])
                               for rows in (base_facts, changed_facts))
            base0_logits, base0_acts = forward(model, base0, device)
            changed0_logits, changed0_acts = forward(model, changed0, device)
            base6_logits, _ = forward(model, base6, device)
            changed6_logits, _ = forward(model, changed6, device)
            discovery_ok = ((base0_logits[:split].argmax(-1) == old)
                            & (changed0_logits[:split].argmax(-1) == new))
            keep = ((base6_logits[split:].argmax(-1) == old)
                    & (changed6_logits[split:].argmax(-1) == new))
            if int(discovery_ok.sum()) < 16 or int(keep.sum()) < 16:
                continue
            semantic_delta = (changed0_acts[:split][discovery_ok]
                              - base0_acts[:split][discovery_ok]).mean(0)
            answer_delta = prototypes[new] - prototypes[old]
            prompts = base6[split:][keep.cpu().numpy()]
            base = base6_logits[split:][keep]
            target = torch.full((int(keep.sum()),), new, device=device, dtype=torch.long)
            semantic_methods = patch_metrics(
                model, prompts, semantic_delta, target, base, device, config["channels"],
                config["seed"] + relation * 1000 + before * 10 + spec["seed"],
            )
            answer_methods = patch_metrics(
                model, prompts, answer_delta, target, base, device, config["channels"],
                config["seed"] + 50000 + relation * 1000 + before * 10 + spec["seed"],
            )
            cosine, fidelity = effect_similarity(
                base, semantic_methods["full"]["effect"], answer_methods["full"]["effect"],
            )
            semantic_indices = set(semantic_methods["native"]["indices"])
            answer_indices = set(answer_methods["native"]["indices"])
            semantic_rows.append({
                "relation": relation, "before": before, "after": after,
                "old": old, "new": new, "eligible": int(keep.sum()),
                "semantic_accuracy": semantic_methods["full"]["accuracy"],
                "semantic_native_accuracy": semantic_methods["native"]["accuracy"],
                "answer_accuracy": answer_methods["full"]["accuracy"],
                "answer_native_accuracy": answer_methods["native"]["accuracy"],
                "effect_cosine": cosine, "effect_fidelity": fidelity,
                "top_support_overlap": len(semantic_indices & answer_indices) / config["channels"],
            })

    def method_mean(name, key):
        return float(np.mean([row["methods"][name][key] for row in answer_rows]))

    answer_summary = {name: {key: method_mean(name, key) for key in ("accuracy", "target_margin")}
                      for name in ("full", "native", "random", "rotated")}
    semantic_summary = None
    if semantic_rows:
        semantic_summary = summarize(semantic_rows, (
            "semantic_accuracy", "semantic_native_accuracy", "answer_accuracy",
            "answer_native_accuracy", "effect_cosine", "effect_fidelity",
            "top_support_overlap",
        ))
        semantic_summary["transitions"] = len(semantic_rows)
    return {"name": spec["name"], "rho": rho, "seed": spec["seed"],
            "answer_summary": answer_summary, "semantic_summary": semantic_summary,
            "answer_rows": answer_rows, "semantic_rows": semantic_rows}


def main():
    config = json.loads(CONFIG.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for spec in config["runs"]:
        print("audit", spec["name"], flush=True)
        runs.append(audit_run(spec, config, device))
    by_rho = {}
    for rho in (0.0, 0.5, 1.0):
        selected = [row for row in runs if row["rho"] == rho]
        by_rho[str(rho)] = {
            "runs": len(selected),
            "answer_full_accuracy": float(np.mean([r["answer_summary"]["full"]["accuracy"] for r in selected])),
            "answer_native_accuracy": float(np.mean([r["answer_summary"]["native"]["accuracy"] for r in selected])),
            "answer_rotated_accuracy": float(np.mean([r["answer_summary"]["rotated"]["accuracy"] for r in selected])),
            "answer_random_accuracy": float(np.mean([r["answer_summary"]["random"]["accuracy"] for r in selected])),
        }
        semantic = [r["semantic_summary"] for r in selected if r["semantic_summary"]]
        if semantic:
            for key in ("semantic_accuracy", "semantic_native_accuracy", "answer_accuracy",
                        "answer_native_accuracy", "effect_cosine", "effect_fidelity",
                        "top_support_overlap"):
                by_rho[str(rho)][key] = float(np.mean([row[key] for row in semantic]))

    a0 = all(r["answer_summary"]["full"]["accuracy"] >= .60 for r in runs)
    a1 = all(r["answer_summary"]["native"]["accuracy"]
             >= r["answer_summary"]["full"]["accuracy"] - .10
             and r["answer_summary"]["native"]["accuracy"]
             >= r["answer_summary"]["rotated"]["accuracy"] + .30 for r in runs)
    pure = by_rho["0.0"]
    a2 = (pure.get("semantic_accuracy", -1) >= pure.get("answer_accuracy", 1) + .10
          and pure.get("effect_cosine", 1) <= .80)
    a3 = (abs(pure.get("semantic_accuracy", 0) - pure.get("answer_accuracy", 1)) <= .05
          and pure.get("effect_cosine", 0) >= .90
          and pure.get("top_support_overlap", 0) >= .50)
    mixed = by_rho["0.5"]
    a4 = (mixed["answer_full_accuracy"] >= .60
          and mixed.get("semantic_accuracy", 1) <= mixed.get("answer_accuracy", 0) - .20)
    hypotheses = {
        "A0_generic_answer_interface_exists_in_all_regimes": {"supported": a0},
        "A1_answer_interface_is_native_axis_privileged": {"supported": a1},
        "A2_pure_semantic_patch_contains_rule_specific_computation": {"supported": a2},
        "A3_pure_semantic_headline_is_explained_by_output_interface": {"supported": a3},
        "A4_mixed_models_keep_output_interface_after_semantic_transfer_fails": {"supported": a4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "summary": by_rho, "runs": runs}, indent=2))
    lines = ["# Final-layer grammar rule versus generic answer interface", "",
             "Class-prototype activation deltas learned from unrelated facts are compared with paired semantic-rule deltas on disjoint identities and an unseen template.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| rho | Runs | Answer full/native/rotated/random accuracy | Semantic full/native | Matched answer full/native | Effect cosine | Support overlap |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for rho in ("0.0", "0.5", "1.0"):
        row = by_rho[rho]
        semantic = (f"{row['semantic_accuracy']:.3f}/{row['semantic_native_accuracy']:.3f}"
                    if "semantic_accuracy" in row else "n/a")
        matched = (f"{row['answer_accuracy']:.3f}/{row['answer_native_accuracy']:.3f}"
                   if "answer_accuracy" in row else "n/a")
        cosine = f"{row['effect_cosine']:.3f}" if "effect_cosine" in row else "n/a"
        overlap = f"{row['top_support_overlap']:.3f}" if "top_support_overlap" in row else "n/a"
        lines.append(f"| {rho} | {row['runs']} | {row['answer_full_accuracy']:.3f}/{row['answer_native_accuracy']:.3f}/{row['answer_rotated_accuracy']:.3f}/{row['answer_random_accuracy']:.3f} | {semantic} | {matched} | {cosine} | {overlap} |")
    lines += ["", "A generic final answer interface is an interpretable and useful mechanism, but it is downstream of the computation that determined the answer. If it explains the semantic patch, earlier layers—not the final patched channels—must carry the rule-specific evidence."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
