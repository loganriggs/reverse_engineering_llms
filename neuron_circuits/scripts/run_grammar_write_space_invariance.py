#!/usr/bin/env python3
"""Align causally relevant MLP residual writes rather than bilinear channels."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, load_manifest
from run_grammar_cross_seed_leave_relation_out import relation_facts
from run_grammar_cross_seed_semantic_invariance import (
    activations_for_templates, alignment_metrics, answer_prototypes, fit_ridge,
    patch_effect, semantic_deltas,
)
from run_grammar_final_interface_redteam import effect_similarity, load_model
from run_grammar_interface_depth_trace import forward


CONFIG = Path("configs/grammar_write_space_invariance.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_write_space_invariance.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_WRITE_SPACE_INVARIANCE.md")
RAW = Path("artifacts/grammar_regimes/grammar_same_codebook_leave_relation_out.json")


def to_writes(model, activations):
    return [values @ model.blocks[layer].down.weight.detach().float().cpu().T
            for layer, values in enumerate(activations)]


@torch.inference_mode()
def audit_pair(source_spec, target_spec, config, loro, base, device):
    source_root, target_root = Path(source_spec["root"]), Path(target_spec["root"])
    source_manifest = load_manifest(source_root / "manifest.npz")
    target_manifest = load_manifest(target_root / "manifest.npz")
    source_model = load_model(source_root / "rho_0" / "depth_4" / "checkpoint.pt", device)
    target_model = load_model(target_root / "rho_0" / "depth_4" / "checkpoint.pt", device)
    source_semantic = semantic_deltas(source_model, source_manifest, base, device)
    target_semantic = semantic_deltas(target_model, target_manifest, base, device)
    source_proto = answer_prototypes(source_model, source_manifest, 0.0, base, device)
    target_proto = answer_prototypes(target_model, target_manifest, 0.0, base, device)
    source_down = [block.down.weight.detach().float() for block in source_model.blocks]
    target_down = [block.down.weight.detach().float() for block in target_model.blocks]
    target_inverse = [torch.linalg.pinv(weight.T) for weight in target_down]
    common = sorted(set(source_semantic) & set(target_semantic))
    rows = {layer: [] for layer in (1, 2, 3)}
    alignment = {relation: [] for relation in range(4)}
    max_inverse_error = 0.0
    for relation in range(4):
        rng = np.random.default_rng(config["seed"] + source_spec["seed"] * 10000
                                    + target_spec["seed"] * 100 + relation)
        train_facts = relation_facts(
            rng, loro["anchor_train_examples"], [value for value in range(8) if value != relation],
        )
        eval_facts = relation_facts(rng, loro["anchor_evaluation_examples"], [relation])
        source_train = to_writes(source_model, activations_for_templates(
            source_model, source_manifest, train_facts, base["anchor_train_templates"], device,
        ))
        target_train = to_writes(target_model, activations_for_templates(
            target_model, target_manifest, train_facts, base["anchor_train_templates"], device,
        ))
        source_eval = to_writes(source_model, activations_for_templates(
            source_model, source_manifest, eval_facts, [base["anchor_evaluation_template"]], device,
        ))
        target_eval = to_writes(target_model, activations_for_templates(
            target_model, target_manifest, eval_facts, [base["anchor_evaluation_template"]], device,
        ))
        maps = []
        for layer in range(source_model.layers_count):
            fitted = fit_ridge(source_train[layer], target_train[layer],
                               base["ridge_relative_penalty"])
            maps.append(fitted)
            r2, cosine = alignment_metrics(source_eval[layer], target_eval[layer], *fitted)
            alignment[relation].append({"r2": r2, "cosine": cosine})

        transitions = [key for key in common if key[0] == relation]
        for index, transition in enumerate(transitions):
            target_item, source_item = target_semantic[transition], source_semantic[transition]
            wrong_keys = [key for key in common if key[0] != relation]
            wrong_item = source_semantic[wrong_keys[(index + relation) % len(wrong_keys)]]
            base_facts, changed_facts = target_item["evaluation_facts"]
            template = base["anchor_evaluation_template"]
            base_prompts = encode_prompts(target_manifest, base_facts, template)
            changed_prompts = encode_prompts(target_manifest, changed_facts, template)
            base_logits, _ = forward(target_model, base_prompts, device)
            changed_logits, _ = forward(target_model, changed_prompts, device)
            old, new = target_item["old"], target_item["new"]
            keep = ((base_logits.argmax(-1) == old) & (changed_logits.argmax(-1) == new))
            if int(keep.sum()) < 16:
                continue
            prompts = base_prompts[keep.cpu().numpy()]
            base_logits = base_logits[keep]
            target = torch.full((int(keep.sum()),), new, dtype=torch.long, device=device)
            for layer in (1, 2, 3):
                weight = maps[layer][2].to(device)
                own = target_item["layers"][layer]
                source_write = source_item["layers"][layer] @ source_down[layer].T
                mapped_write = source_write @ weight
                mapped = mapped_write @ target_inverse[layer]
                reconstructed = mapped @ target_down[layer].T
                relative_error = ((reconstructed - mapped_write).norm()
                                  / mapped_write.norm().clamp_min(1e-30))
                max_inverse_error = max(max_inverse_error, float(relative_error))
                wrong_write = wrong_item["layers"][layer] @ source_down[layer].T @ weight
                wrong_write = wrong_write * (mapped_write.norm()
                                             / wrong_write.norm().clamp_min(1e-30))
                wrong = wrong_write @ target_inverse[layer]
                source_answer_write = ((source_proto[layer][source_item["new"]]
                                       - source_proto[layer][source_item["old"]])
                                      @ source_down[layer].T @ weight)
                source_answer = source_answer_write @ target_inverse[layer]
                target_answer = target_proto[layer][new] - target_proto[layer][old]
                deltas = {"own": own, "mapped": mapped, "wrong": wrong,
                          "source_answer": source_answer, "target_answer": target_answer}
                effects, accuracies = {}, {}
                for name, delta in deltas.items():
                    effects[name], accuracies[name] = patch_effect(
                        target_model, prompts, layer, delta, base_logits, target, device,
                    )
                record = {"relation": relation, "before": transition[1]}
                for name in ("mapped", "wrong", "source_answer", "target_answer"):
                    cosine, fidelity = effect_similarity(base_logits, effects["own"], effects[name])
                    record[f"{name}_cosine"] = cosine
                    record[f"{name}_fidelity"] = fidelity
                for name, accuracy in accuracies.items():
                    record[f"{name}_accuracy"] = accuracy
                rows[layer].append(record)
    layers = {}
    for layer, values in rows.items():
        keys = [key for key in values[0] if key not in {"relation", "before"}]
        layers[str(layer)] = {key: float(np.mean([row[key] for row in values])) for key in keys}
        layers[str(layer)]["transitions"] = len(values)
    return {"source": source_spec["name"], "target": target_spec["name"],
            "alignment": alignment, "layers": layers,
            "max_inverse_relative_error": max_inverse_error}


def main():
    config = json.loads(CONFIG.read_text())
    same = json.loads(Path(config["parent_config"]).read_text())
    loro = json.loads(Path(same["parent_config"]).read_text())
    base = json.loads(Path(loro["parent_config"]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for source, target in permutations(same["runs"], 2):
        print("audit", source["name"], "->", target["name"], flush=True)
        results.append(audit_pair(source, target, config, loro, base, device))
    raw = json.loads(RAW.read_text())["summary"]
    summary, valid = {}, {}
    for layer in (1, 2, 3):
        alignment_r2 = [pair["alignment"][relation][layer]["r2"]
                        for pair in results for relation in range(4)]
        valid[str(layer)] = min(alignment_r2) >= config["alignment_r2_gate"]
        values = [pair["layers"][str(layer)] for pair in results]
        keys = [key for key in values[0] if key != "transitions"]
        summary[str(layer)] = {key: float(np.mean([row[key] for row in values])) for key in keys}
        summary[str(layer)]["alignment_r2_mean"] = float(np.mean(alignment_r2))
        summary[str(layer)]["alignment_r2_min"] = float(np.min(alignment_r2))
    maximum_error = max(row["max_inverse_relative_error"] for row in results)
    adjudicative = [layer for layer in (1, 2, 3) if valid[str(layer)]]
    early = [layer for layer in (1, 2) if valid[str(layer)]]
    w0 = bool(adjudicative)
    w1 = any(summary[str(layer)]["mapped_fidelity"] >= .50
             and summary[str(layer)]["mapped_accuracy"]
             >= summary[str(layer)]["own_accuracy"] - .15 for layer in early)
    w2 = any(summary[str(layer)]["mapped_fidelity"]
             >= summary[str(layer)]["wrong_fidelity"] + .20
             or summary[str(layer)]["mapped_accuracy"]
             >= summary[str(layer)]["wrong_accuracy"] + .15 for layer in early)
    w3 = any(summary[str(layer)]["alignment_r2_mean"]
             >= raw[str(layer)]["alignment_r2_mean"] + .30 for layer in adjudicative)
    final = summary["3"]
    w4 = (valid["3"] and final["mapped_fidelity"] >= .70
          and final["target_answer_fidelity"] >= .70
          and final["target_answer_fidelity"] >= final["mapped_fidelity"] - .10)
    w5 = maximum_error <= 1e-5
    hypotheses = {
        "W0_write_alignment_is_adjudicative": {"supported": w0, "valid_layers": adjudicative},
        "W1_semantic_write_is_initialization_invariant": {"supported": w1},
        "W2_same_rule_write_beats_wrong_rule": {"supported": w2},
        "W3_write_space_is_more_canonical_than_channels": {"supported": w3},
        "W4_final_write_matches_target_answer_interface": {"supported": w4},
        "W5_activation_inverse_is_exact": {"supported": w5, "maximum_error": maximum_error},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "valid_layers": valid, "summary": summary,
                                  "raw_activation_summary": raw, "pairs": results}, indent=2))
    lines = ["# Symmetry-aware cross-initialization MLP-write invariance", "",
             "The alignment and transferred semantic edits are expressed in each MLP's residual write space, quotienting out bilinear channel permutation, scaling, and null directions.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum write realization relative error: `{maximum_error:.2e}`.", "",
              "| Layer | Write R2 mean/min | Raw-channel R2 mean/min | Own/mapped/wrong accuracy | Mapped/wrong fidelity | Target-answer fidelity | Adjudicative |",
              "|---:|---:|---:|---:|---:|---:|:---:|"]
    for layer in (1, 2, 3):
        row, old = summary[str(layer)], raw[str(layer)]
        lines.append(
            f"| {layer} | {row['alignment_r2_mean']:.3f}/{row['alignment_r2_min']:.3f} | "
            f"{old['alignment_r2_mean']:.3f}/{old['alignment_r2_min']:.3f} | "
            f"{row['own_accuracy']:.3f}/{row['mapped_accuracy']:.3f}/{row['wrong_accuracy']:.3f} | "
            f"{row['mapped_fidelity']:.3f}/{row['wrong_fidelity']:.3f} | "
            f"{row['target_answer_fidelity']:.3f} | {'yes' if valid[str(layer)] else 'no'} |"
        )
    lines += ["", "A write-space success would locate the reusable object after the channel basis but before residual-stream transport. A failure with an exact inverse cannot be blamed on MLP channel null directions."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
