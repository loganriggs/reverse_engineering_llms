#!/usr/bin/env python3
"""Test whether grammar edits causally transfer across independently trained models."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from audit_grammar_semantic_transfer import make_transition_examples
from run_grammar_final_interface_redteam import effect_similarity, load_model
from run_grammar_interface_depth_trace import forward


CONFIG = Path("configs/grammar_cross_seed_semantic_invariance.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_cross_seed_semantic_invariance.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_CROSS_SEED_SEMANTIC_INVARIANCE.md")


def random_facts(rng: np.random.Generator, count: int) -> np.ndarray:
    return np.column_stack((
        rng.integers(0, 512, count, dtype=np.int32),
        rng.integers(0, 512, count, dtype=np.int32),
        rng.integers(0, 8, count, dtype=np.int32),
    ))


@torch.inference_mode()
def activations_for_templates(model, manifest, facts, templates, device):
    by_layer = [[] for _ in range(model.layers_count)]
    for template in templates:
        prompts = encode_prompts(manifest, facts, template)
        _, activations = forward(model, prompts, device)
        for layer, values in enumerate(activations):
            by_layer[layer].append(values.float().cpu())
    return [torch.cat(values, 0) for values in by_layer]


def fit_ridge(source: torch.Tensor, target: torch.Tensor, relative_penalty: float):
    source = source.double(); target = target.double()
    source_mean = source.mean(0); target_mean = target.mean(0)
    x = source - source_mean; y = target - target_mean
    gram = x.T @ x
    penalty = relative_penalty * float(torch.trace(gram)) / gram.shape[0]
    weight = torch.linalg.solve(
        gram + penalty * torch.eye(gram.shape[0], dtype=gram.dtype), x.T @ y,
    )
    return source_mean.float(), target_mean.float(), weight.float()


def alignment_metrics(source, target, source_mean, target_mean, weight):
    predicted = (source - source_mean) @ weight + target_mean
    residual = (predicted - target).square().sum()
    total = (target - target.mean(0)).square().sum().clamp_min(1e-30)
    r2 = 1 - residual / total
    centered_prediction = predicted - predicted.mean(0)
    centered_target = target - target.mean(0)
    cosine = torch.nn.functional.cosine_similarity(
        centered_prediction.flatten(), centered_target.flatten(), dim=0,
    )
    return float(r2), float(cosine)


@torch.inference_mode()
def patch_effect(model, prompts, layer, delta, base, target, device):
    changed, _ = forward(model, prompts, device, ActivationPatch(layer, delta, additive=True))
    return changed - base, float((changed.argmax(-1) == target).float().mean())


@torch.inference_mode()
def answer_prototypes(model, manifest, rho, config, device):
    labels, _, _ = labels_for_rho(manifest, "train", rho)
    facts = np.asarray(manifest["train_facts"])
    prompts = encode_prompts(manifest, facts, 0)
    logits, activations = forward(model, prompts, device)
    correct = (logits.argmax(-1).cpu().numpy() == labels)
    output = [{answer: None for answer in range(8)} for _ in range(model.layers_count)]
    rng = np.random.default_rng(config["seed"] + int(manifest["seed"]) * 100)
    for answer in range(8):
        indices = np.flatnonzero(correct & (labels == answer)); rng.shuffle(indices)
        chosen = torch.as_tensor(indices[:128], device=device)
        for layer in range(model.layers_count):
            output[layer][answer] = activations[layer][chosen].mean(0)
    return output


@torch.inference_mode()
def semantic_deltas(model, manifest, config, device):
    rng = np.random.default_rng(config["seed"] + 10000 + int(manifest["seed"]) * 100)
    perms = np.asarray(manifest["permutations"])
    output = {}
    count = config["semantic_examples"]
    split = count // 2
    for relation in range(4):
        for before in (0, 2, 4, 6):
            after = before + 1
            base_facts, changed_facts = make_transition_examples(
                rng, relation, before, after, count,
            )
            base0 = encode_prompts(manifest, base_facts, 0)
            changed0 = encode_prompts(manifest, changed_facts, 0)
            base_logits, base_acts = forward(model, base0, device)
            changed_logits, changed_acts = forward(model, changed0, device)
            old, new = int(perms[relation, before]), int(perms[relation, after])
            keep = ((base_logits[:split].argmax(-1) == old)
                    & (changed_logits[:split].argmax(-1) == new))
            if int(keep.sum()) < 16:
                continue
            output[(relation, before)] = {
                "old": old, "new": new,
                "layers": [(changed_acts[layer][:split][keep]
                            - base_acts[layer][:split][keep]).mean(0)
                           for layer in range(model.layers_count)],
                "evaluation_facts": (base_facts[split:], changed_facts[split:]),
            }
    return output


def summarize(rows, key):
    return float(np.mean([row[key] for row in rows]))


@torch.inference_mode()
def audit_pair(source_spec, target_spec, config, device):
    source_root, target_root = Path(source_spec["root"]), Path(target_spec["root"])
    source_manifest = load_manifest(source_root / "manifest.npz")
    target_manifest = load_manifest(target_root / "manifest.npz")
    source_model = load_model(source_root / "rho_0" / "depth_4" / "checkpoint.pt", device)
    target_model = load_model(target_root / "rho_0" / "depth_4" / "checkpoint.pt", device)
    rng = np.random.default_rng(config["seed"] + source_spec["seed"] * 1000
                                + target_spec["seed"] * 10)
    train_facts = random_facts(rng, config["anchor_train_examples"])
    eval_facts = random_facts(rng, config["anchor_evaluation_examples"])
    source_train = activations_for_templates(
        source_model, source_manifest, train_facts, config["anchor_train_templates"], device,
    )
    target_train = activations_for_templates(
        target_model, target_manifest, train_facts, config["anchor_train_templates"], device,
    )
    source_eval = activations_for_templates(
        source_model, source_manifest, eval_facts, [config["anchor_evaluation_template"]], device,
    )
    target_eval = activations_for_templates(
        target_model, target_manifest, eval_facts, [config["anchor_evaluation_template"]], device,
    )
    maps, alignment = [], []
    for layer in range(source_model.layers_count):
        fitted = fit_ridge(source_train[layer], target_train[layer],
                           config["ridge_relative_penalty"])
        maps.append(fitted)
        r2, cosine = alignment_metrics(source_eval[layer], target_eval[layer], *fitted)
        alignment.append({"r2": r2, "cosine": cosine})

    source_semantic = semantic_deltas(source_model, source_manifest, config, device)
    target_semantic = semantic_deltas(target_model, target_manifest, config, device)
    source_prototypes = answer_prototypes(source_model, source_manifest, 0.0, config, device)
    common = sorted(set(source_semantic) & set(target_semantic))
    rows = {layer: [] for layer in (1, 2, 3)}
    for index, transition in enumerate(common):
        relation, before = transition
        target_item = target_semantic[transition]
        source_item = source_semantic[transition]
        wrong_transition = common[(index + 5) % len(common)]
        wrong_item = source_semantic[wrong_transition]
        base_facts, changed_facts = target_item["evaluation_facts"]
        base_prompts = encode_prompts(target_manifest, base_facts, config["anchor_evaluation_template"])
        changed_prompts = encode_prompts(target_manifest, changed_facts, config["anchor_evaluation_template"])
        base_logits, _ = forward(target_model, base_prompts, device)
        changed_logits, _ = forward(target_model, changed_prompts, device)
        old, new = target_item["old"], target_item["new"]
        keep = ((base_logits.argmax(-1) == old) & (changed_logits.argmax(-1) == new))
        if int(keep.sum()) < 16:
            continue
        prompts = base_prompts[keep.cpu().numpy()]
        base = base_logits[keep]
        target = torch.full((int(keep.sum()),), new, dtype=torch.long, device=device)
        for layer in (1, 2, 3):
            _, _, weight = maps[layer]
            weight = weight.to(device)
            own_delta = target_item["layers"][layer]
            mapped = source_item["layers"][layer] @ weight
            mapped_wrong = wrong_item["layers"][layer] @ weight
            mapped_wrong = mapped_wrong * (mapped.norm() / mapped_wrong.norm().clamp_min(1e-30))
            source_answer = (source_prototypes[layer][source_item["new"]]
                             - source_prototypes[layer][source_item["old"]])
            mapped_answer = source_answer @ weight
            own_effect, own_accuracy = patch_effect(
                target_model, prompts, layer, own_delta, base, target, device,
            )
            mapped_effect, mapped_accuracy = patch_effect(
                target_model, prompts, layer, mapped, base, target, device,
            )
            wrong_effect, wrong_accuracy = patch_effect(
                target_model, prompts, layer, mapped_wrong, base, target, device,
            )
            answer_effect, answer_accuracy = patch_effect(
                target_model, prompts, layer, mapped_answer, base, target, device,
            )
            mapped_cosine, mapped_fidelity = effect_similarity(base, own_effect, mapped_effect)
            wrong_cosine, wrong_fidelity = effect_similarity(base, own_effect, wrong_effect)
            answer_cosine, answer_fidelity = effect_similarity(base, own_effect, answer_effect)
            rows[layer].append({
                "relation": relation, "before": before,
                "own_accuracy": own_accuracy,
                "mapped_accuracy": mapped_accuracy,
                "wrong_accuracy": wrong_accuracy,
                "answer_accuracy": answer_accuracy,
                "mapped_cosine": mapped_cosine,
                "mapped_fidelity": mapped_fidelity,
                "wrong_cosine": wrong_cosine,
                "wrong_fidelity": wrong_fidelity,
                "answer_cosine": answer_cosine,
                "answer_fidelity": answer_fidelity,
            })
    layers = {}
    for layer, values in rows.items():
        keys = [key for key in values[0] if key not in {"relation", "before"}]
        layers[str(layer)] = {key: summarize(values, key) for key in keys}
        layers[str(layer)]["transitions"] = len(values)
    return {
        "source": source_spec["name"], "target": target_spec["name"],
        "alignment": alignment, "layers": layers,
    }


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    pure = [run for run in parent["runs"] if run["rho"] == 0.0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for source, target in permutations(pure, 2):
        print("audit", source["name"], "->", target["name"], flush=True)
        results.append(audit_pair(source, target, config, device))

    summary = {}
    valid = {}
    for layer in (1, 2, 3):
        alignment_r2 = [row["alignment"][layer]["r2"] for row in results]
        valid[str(layer)] = min(alignment_r2) >= .50
        layer_rows = [row["layers"][str(layer)] for row in results]
        keys = [key for key in layer_rows[0] if key != "transitions"]
        summary[str(layer)] = {key: float(np.mean([row[key] for row in layer_rows]))
                               for key in keys}
        summary[str(layer)]["alignment_r2_mean"] = float(np.mean(alignment_r2))
        summary[str(layer)]["alignment_r2_min"] = float(np.min(alignment_r2))

    adjudicative = [layer for layer in (1, 2, 3) if valid[str(layer)]]
    early = [layer for layer in (1, 2) if valid[str(layer)]]
    x0 = bool(adjudicative)
    x1 = any(summary[str(layer)]["mapped_fidelity"] >= .50
             and summary[str(layer)]["mapped_accuracy"]
             >= summary[str(layer)]["own_accuracy"] - .15 for layer in early)
    x2 = any(summary[str(layer)]["mapped_fidelity"]
             >= summary[str(layer)]["wrong_fidelity"] + .20
             or summary[str(layer)]["mapped_accuracy"]
             >= summary[str(layer)]["wrong_accuracy"] + .15 for layer in early)
    x3 = any(summary[str(layer)]["mapped_fidelity"]
             >= summary[str(layer)]["answer_fidelity"] + .10 for layer in early)
    final = summary["3"]
    x4 = (valid["3"] and final["mapped_fidelity"] >= .70
          and final["answer_fidelity"] >= final["mapped_fidelity"] - .10)
    hypotheses = {
        "X0_alignment_is_adjudicative": {"supported": x0, "valid_layers": adjudicative},
        "X1_semantic_computation_is_cross_seed_reusable": {"supported": x1},
        "X2_same_rule_beats_wrong_rule": {"supported": x2},
        "X3_semantics_adds_more_than_source_answer_identity": {"supported": x3},
        "X4_final_layer_is_a_shared_output_interface": {"supported": x4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "valid_layers": valid, "summary": summary,
                                  "pairs": results}, indent=2))
    lines = ["# Cross-seed causal invariance of grammar computations", "",
             "A global ridge map is fitted on unrelated shared prompts, frozen, and used to translate semantic activation edits between independently trained pure-rule models with different answer permutations.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Alignment R2 mean/min | Own/mapped/wrong/answer accuracy | Mapped/wrong/answer effect fidelity | Mapped/wrong/answer cosine | Adjudicative |",
              "|---:|---:|---:|---:|---:|:---:|"]
    for layer in (1, 2, 3):
        row = summary[str(layer)]
        lines.append(
            f"| {layer} | {row['alignment_r2_mean']:.3f}/{row['alignment_r2_min']:.3f} | "
            f"{row['own_accuracy']:.3f}/{row['mapped_accuracy']:.3f}/{row['wrong_accuracy']:.3f}/{row['answer_accuracy']:.3f} | "
            f"{row['mapped_fidelity']:.3f}/{row['wrong_fidelity']:.3f}/{row['answer_fidelity']:.3f} | "
            f"{row['mapped_cosine']:.3f}/{row['wrong_cosine']:.3f}/{row['answer_cosine']:.3f} | "
            f"{'yes' if valid[str(layer)] else 'no'} |"
        )
    lines += ["", "A failed causal transfer is treated as evidence against global-linear invariance only where held-out state alignment passes. Wrong-rule vectors are norm matched after mapping; this prevents simple edit magnitude from winning the control."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
