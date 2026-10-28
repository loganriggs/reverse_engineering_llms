#!/usr/bin/env python3
"""Trace semantic-rule and generic-answer activation operators through depth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from audit_grammar_semantic_transfer import make_transition_examples
from run_grammar_final_interface_redteam import effect_similarity, load_model, restricted


CONFIG = Path("configs/grammar_interface_depth_trace.json")
OUTPUT = Path("artifacts/grammar_regimes/interface_depth_trace.json")
REPORT = Path("artifacts/grammar_regimes/INTERFACE_DEPTH_TRACE.md")


@torch.inference_mode()
def forward(model, prompts, device, patch=None):
    tokens = torch.as_tensor(prompts.astype(np.int64), device=device)
    logits, activations = model(tokens, patch=patch, return_activations=True)
    return logits[:, -1, list(VOCAB.answers)].float(), [row.float() for row in activations]


@torch.inference_mode()
def patched(model, prompts, layer, delta, target, base, device):
    changed, _ = forward(model, prompts, device, ActivationPatch(layer, delta, additive=True))
    return changed - base, float((changed.argmax(-1) == target).float().mean())


@torch.inference_mode()
def audit_run(spec, parent, config, device):
    root = Path(spec["root"]); rho = spec["rho"]
    manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    labels, _, _ = labels_for_rho(manifest, "train", rho)
    facts = np.asarray(manifest["train_facts"])
    prompt0 = encode_prompts(manifest, facts, parent["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, parent["evaluation_template"])
    logits0, acts0 = forward(model, prompt0, device)
    logits6, _ = forward(model, prompt6, device)
    labels_t = torch.as_tensor(labels, device=device)
    eligible = ((logits0.argmax(-1) == labels_t) & (logits6.argmax(-1) == labels_t)).cpu().numpy()
    rng = np.random.default_rng(parent["seed"] + spec["seed"] * 100 + round(rho * 10))
    prototypes = [{answer: None for answer in range(8)} for _ in range(model.layers_count)]
    evaluations = {}
    for answer in range(8):
        indices = np.flatnonzero(eligible & (labels == answer)); rng.shuffle(indices)
        discovery = torch.as_tensor(indices[:parent["prototype_per_class"]], device=device)
        for layer in range(model.layers_count):
            prototypes[layer][answer] = acts0[layer][discovery].mean(0)
        evaluations[answer] = indices[parent["prototype_per_class"]:
                                      parent["prototype_per_class"] + parent["evaluation_per_class"]]

    generic_layers = []
    for layer in range(model.layers_count):
        rows = []
        for old in range(8):
            new = (old + 1) % 8; indices = evaluations[old]
            prompts = prompt6[indices]; base = logits6[torch.as_tensor(indices, device=device)]
            target = torch.full((len(indices),), new, device=device, dtype=torch.long)
            delta = prototypes[layer][new] - prototypes[layer][old]
            full_effect, full_accuracy = patched(model, prompts, layer, delta, target, base, device)
            native_delta = restricted(delta, "native", config["channels"], config["seed"] + layer * 100 + old)
            rotated_delta = restricted(delta, "rotated", config["channels"], config["seed"] + 5000 + layer * 100 + old)
            _, native_accuracy = patched(model, prompts, layer, native_delta, target, base, device)
            _, rotated_accuracy = patched(model, prompts, layer, rotated_delta, target, base, device)
            rows.append({"accuracy": full_accuracy, "native_accuracy": native_accuracy,
                         "rotated_accuracy": rotated_accuracy})
        generic_layers.append({key: float(np.mean([row[key] for row in rows]))
                               for key in rows[0]})

    semantic_layers = [[] for _ in range(model.layers_count)]
    permutations = np.asarray(manifest["permutations"])
    semantic_rng = np.random.default_rng(parent["seed"] + 10000 + spec["seed"] * 100)
    count, split = config["semantic_examples"], config["semantic_examples"] // 2
    for relation in range(4):
        for before in (0, 2, 4, 6):
            after = before + 1
            base_facts, changed_facts = make_transition_examples(semantic_rng, relation, before, after, count)
            old, new = int(permutations[relation, before]), int(permutations[relation, after])
            base0, changed0 = (encode_prompts(manifest, values, parent["discovery_template"])
                               for values in (base_facts, changed_facts))
            base6, changed6 = (encode_prompts(manifest, values, parent["evaluation_template"])
                               for values in (base_facts, changed_facts))
            base0_logits, base0_acts = forward(model, base0, device)
            changed0_logits, changed0_acts = forward(model, changed0, device)
            base6_logits, _ = forward(model, base6, device)
            changed6_logits, _ = forward(model, changed6, device)
            discover = ((base0_logits[:split].argmax(-1) == old)
                        & (changed0_logits[:split].argmax(-1) == new))
            keep = ((base6_logits[split:].argmax(-1) == old)
                    & (changed6_logits[split:].argmax(-1) == new))
            if int(discover.sum()) < 16 or int(keep.sum()) < 16:
                continue
            prompts = base6[split:][keep.cpu().numpy()]
            base = base6_logits[split:][keep]
            target = torch.full((int(keep.sum()),), new, device=device, dtype=torch.long)
            for layer in range(model.layers_count):
                semantic_delta = (changed0_acts[layer][:split][discover]
                                  - base0_acts[layer][:split][discover]).mean(0)
                answer_delta = prototypes[layer][new] - prototypes[layer][old]
                semantic_effect, semantic_accuracy = patched(
                    model, prompts, layer, semantic_delta, target, base, device,
                )
                answer_effect, answer_accuracy = patched(
                    model, prompts, layer, answer_delta, target, base, device,
                )
                semantic_native = restricted(semantic_delta, "native", config["channels"],
                                             config["seed"] + relation * 1000 + before * 10 + layer)
                semantic_rotated = restricted(semantic_delta, "rotated", config["channels"],
                                              config["seed"] + 50000 + relation * 1000 + before * 10 + layer)
                _, semantic_native_accuracy = patched(
                    model, prompts, layer, semantic_native, target, base, device,
                )
                _, semantic_rotated_accuracy = patched(
                    model, prompts, layer, semantic_rotated, target, base, device,
                )
                cosine, fidelity = effect_similarity(base, semantic_effect, answer_effect)
                semantic_indices = set(semantic_delta.abs().topk(config["channels"]).indices.tolist())
                answer_indices = set(answer_delta.abs().topk(config["channels"]).indices.tolist())
                semantic_layers[layer].append({
                    "semantic_accuracy": semantic_accuracy, "answer_accuracy": answer_accuracy,
                    "semantic_native_accuracy": semantic_native_accuracy,
                    "semantic_rotated_accuracy": semantic_rotated_accuracy,
                    "effect_cosine": cosine, "effect_fidelity": fidelity,
                    "support_overlap": len(semantic_indices & answer_indices) / config["channels"],
                })
    semantic_summary = []
    for rows in semantic_layers:
        semantic_summary.append(None if not rows else {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        } | {"transitions": len(rows)})
    return {"name": spec["name"], "rho": rho, "seed": spec["seed"],
            "generic_layers": generic_layers, "semantic_layers": semantic_summary}


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for spec in parent["runs"]:
        print("audit", spec["name"], flush=True)
        runs.append(audit_run(spec, parent, config, device))
    summary = {}
    for rho in (0.0, 0.5, 1.0):
        selected = [row for row in runs if row["rho"] == rho]
        layers = []
        for layer in range(4):
            row = {f"generic_{key}": float(np.mean([run["generic_layers"][layer][key] for run in selected]))
                   for key in selected[0]["generic_layers"][layer]}
            semantic = [run["semantic_layers"][layer] for run in selected
                        if run["semantic_layers"][layer] is not None]
            if semantic:
                for key in semantic[0]:
                    if key != "transitions":
                        row[f"semantic_{key}"] = float(np.mean([value[key] for value in semantic]))
            layers.append(row)
        summary[str(rho)] = {"runs": len(selected), "layers": layers}
    pure = summary["0.0"]["layers"]
    d0 = any(row["semantic_effect_cosine"] <= .80
             and row["semantic_semantic_accuracy"] >= row["semantic_answer_accuracy"] + .10
             for row in pure[:3])
    d1 = (pure[3]["semantic_effect_cosine"] >= .90
          and pure[3]["semantic_effect_cosine"] >= pure[0]["semantic_effect_cosine"] + .10)
    d2 = all(summary[str(rho)]["layers"][3]["generic_accuracy"]
             >= summary[str(rho)]["layers"][0]["generic_accuracy"] + .20
             for rho in (0.0, 0.5, 1.0))
    d3 = all(summary[str(rho)]["layers"][layer]["generic_accuracy"]
             >= summary["0.0"]["layers"][layer]["generic_accuracy"] + .15
             for rho in (0.5, 1.0) for layer in (0, 1))
    pure_advantage = max(row["semantic_semantic_native_accuracy"]
                         - row["semantic_semantic_rotated_accuracy"] for row in pure)
    mixed_generic_advantage = max(row["generic_native_accuracy"] - row["generic_rotated_accuracy"]
                                  for row in summary["0.5"]["layers"])
    random_generic_advantage = max(row["generic_native_accuracy"] - row["generic_rotated_accuracy"]
                                   for row in summary["1.0"]["layers"])
    d4 = pure_advantage >= .30 and not (mixed_generic_advantage >= .30 and random_generic_advantage >= .30)
    hypotheses = {
        "D0_pure_rule_direction_is_specific_upstream": {"supported": d0},
        "D1_pure_rule_converges_to_output_interface": {"supported": d1},
        "D2_generic_answer_interface_strengthens_with_depth": {"supported": d2},
        "D3_mixed_arbitrary_answer_interface_emerges_earlier": {"supported": d3},
        "D4_native_axis_privilege_is_regime_and_depth_specific": {
            "supported": d4, "pure_semantic_max_advantage": pure_advantage,
            "mixed_generic_max_advantage": mixed_generic_advantage,
            "random_generic_max_advantage": random_generic_advantage,
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "summary": summary, "runs": runs}, indent=2))
    lines = ["# Depth trace from rule-specific computation to answer interface", "",
             "At every MLP layer, paired semantic-rule deltas and unrelated answer-class prototype deltas are patched into the same held-out recipients.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| rho | Layer | Generic full/native/rotated acc | Semantic full/native/rotated acc | Matched answer acc | Effect cosine/fidelity | Support overlap |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for rho in ("0.0", "0.5", "1.0"):
        for layer, row in enumerate(summary[rho]["layers"]):
            if "semantic_semantic_accuracy" in row:
                semantic = f"{row['semantic_semantic_accuracy']:.3f}/{row['semantic_semantic_native_accuracy']:.3f}/{row['semantic_semantic_rotated_accuracy']:.3f}"
                matched = f"{row['semantic_answer_accuracy']:.3f}"
                effect = f"{row['semantic_effect_cosine']:.3f}/{row['semantic_effect_fidelity']:.3f}"
                overlap = f"{row['semantic_support_overlap']:.3f}"
            else:
                semantic = matched = effect = overlap = "n/a"
            lines.append(f"| {rho} | {layer} | {row['generic_accuracy']:.3f}/{row['generic_native_accuracy']:.3f}/{row['generic_rotated_accuracy']:.3f} | {semantic} | {matched} | {effect} | {overlap} |")
    lines += ["", "Effect cosine asks whether the two patches implement the same downstream verb; accuracy and magnitude determine whether that verb is driven strongly enough. A final shared direction does not locate the upstream rule computation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
