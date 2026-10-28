#!/usr/bin/env python3
"""Leave one grammar relation out of cross-model activation alignment."""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, load_manifest
from audit_grammar_semantic_transfer import make_transition_examples
from run_grammar_cross_seed_semantic_invariance import (
    activations_for_templates, alignment_metrics, answer_prototypes, fit_ridge,
    patch_effect, semantic_deltas,
)
from run_grammar_final_interface_redteam import effect_similarity, load_model
from run_grammar_interface_depth_trace import forward


CONFIG = Path("configs/grammar_cross_seed_leave_relation_out.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_cross_seed_leave_relation_out.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_CROSS_SEED_LEAVE_RELATION_OUT.md")


def relation_facts(rng, count, allowed):
    allowed = np.asarray(allowed, dtype=np.int32)
    return np.column_stack((
        rng.integers(0, 512, count, dtype=np.int32),
        rng.integers(0, 512, count, dtype=np.int32),
        rng.choice(allowed, count).astype(np.int32),
    ))


@torch.inference_mode()
def audit_pair(source_spec, target_spec, config, base_config, device):
    source_root, target_root = Path(source_spec["root"]), Path(target_spec["root"])
    source_manifest = load_manifest(source_root / "manifest.npz")
    target_manifest = load_manifest(target_root / "manifest.npz")
    source_model = load_model(source_root / "rho_0" / "depth_4" / "checkpoint.pt", device)
    target_model = load_model(target_root / "rho_0" / "depth_4" / "checkpoint.pt", device)
    source_semantic = semantic_deltas(source_model, source_manifest, base_config, device)
    target_semantic = semantic_deltas(target_model, target_manifest, base_config, device)
    source_proto = answer_prototypes(source_model, source_manifest, 0.0, base_config, device)
    target_proto = answer_prototypes(target_model, target_manifest, 0.0, base_config, device)
    common = sorted(set(source_semantic) & set(target_semantic))
    rows = {layer: [] for layer in (1, 2, 3)}
    alignment = {relation: [] for relation in range(4)}
    for relation in range(4):
        rng = np.random.default_rng(config["seed"] + source_spec["seed"] * 10000
                                    + target_spec["seed"] * 100 + relation)
        train_facts = relation_facts(
            rng, config["anchor_train_examples"], [value for value in range(8) if value != relation],
        )
        eval_facts = relation_facts(rng, config["anchor_evaluation_examples"], [relation])
        source_train = activations_for_templates(
            source_model, source_manifest, train_facts,
            base_config["anchor_train_templates"], device,
        )
        target_train = activations_for_templates(
            target_model, target_manifest, train_facts,
            base_config["anchor_train_templates"], device,
        )
        source_eval = activations_for_templates(
            source_model, source_manifest, eval_facts,
            [base_config["anchor_evaluation_template"]], device,
        )
        target_eval = activations_for_templates(
            target_model, target_manifest, eval_facts,
            [base_config["anchor_evaluation_template"]], device,
        )
        maps = []
        for layer in range(source_model.layers_count):
            fitted = fit_ridge(source_train[layer], target_train[layer],
                               base_config["ridge_relative_penalty"])
            maps.append(fitted)
            r2, cosine = alignment_metrics(source_eval[layer], target_eval[layer], *fitted)
            alignment[relation].append({"r2": r2, "cosine": cosine})

        transitions = [key for key in common if key[0] == relation]
        for index, transition in enumerate(transitions):
            target_item, source_item = target_semantic[transition], source_semantic[transition]
            wrong_candidates = [key for key in common if key[0] != relation]
            wrong_item = source_semantic[wrong_candidates[(index + relation) % len(wrong_candidates)]]
            base_facts, changed_facts = target_item["evaluation_facts"]
            template = base_config["anchor_evaluation_template"]
            base_prompts = encode_prompts(target_manifest, base_facts, template)
            changed_prompts = encode_prompts(target_manifest, changed_facts, template)
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
                weight = maps[layer][2].to(device)
                own = target_item["layers"][layer]
                mapped = source_item["layers"][layer] @ weight
                wrong = wrong_item["layers"][layer] @ weight
                wrong = wrong * (mapped.norm() / wrong.norm().clamp_min(1e-30))
                source_answer = (source_proto[layer][source_item["new"]]
                                 - source_proto[layer][source_item["old"]]) @ weight
                target_answer = target_proto[layer][new] - target_proto[layer][old]
                effects = {}
                accuracies = {}
                for name, delta in (("own", own), ("mapped", mapped), ("wrong", wrong),
                                    ("source_answer", source_answer),
                                    ("target_answer", target_answer)):
                    effects[name], accuracies[name] = patch_effect(
                        target_model, prompts, layer, delta, base, target, device,
                    )
                record = {"relation": relation, "before": transition[1]}
                for name in ("mapped", "wrong", "source_answer", "target_answer"):
                    cosine, fidelity = effect_similarity(base, effects["own"], effects[name])
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
            "alignment": alignment, "layers": layers}


def main():
    config = json.loads(CONFIG.read_text())
    base_config = json.loads(Path(config["parent_config"]).read_text())
    parent = json.loads(Path(base_config["parent_config"]).read_text())
    pure = [run for run in parent["runs"] if run["rho"] == 0.0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for source, target in permutations(pure, 2):
        print("audit", source["name"], "->", target["name"], flush=True)
        results.append(audit_pair(source, target, config, base_config, device))

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
    adjudicative = [layer for layer in (1, 2, 3) if valid[str(layer)]]
    early = [layer for layer in (1, 2) if valid[str(layer)]]
    l0 = bool(adjudicative)
    l1 = any(summary[str(layer)]["mapped_fidelity"] >= .50
             and summary[str(layer)]["mapped_accuracy"]
             >= summary[str(layer)]["own_accuracy"] - .15 for layer in early)
    l2 = any(summary[str(layer)]["mapped_fidelity"]
             >= summary[str(layer)]["wrong_fidelity"] + .20
             or summary[str(layer)]["mapped_accuracy"]
             >= summary[str(layer)]["wrong_accuracy"] + .15 for layer in early)
    final = summary["3"]
    l3 = (valid["3"] and final["mapped_fidelity"] >= .70
          and final["target_answer_fidelity"] >= .70
          and final["target_answer_fidelity"] >= final["mapped_fidelity"] - .10)
    l4 = any(summary[str(layer)]["mapped_fidelity"]
             >= summary[str(layer)]["source_answer_fidelity"] + .20
             for layer in adjudicative)
    hypotheses = {
        "L0_unseen_relation_alignment_is_adjudicative": {"supported": l0, "valid_layers": adjudicative},
        "L1_rule_edit_generalizes_beyond_alignment_relations": {"supported": l1},
        "L2_same_rule_beats_norm_matched_wrong_rule": {"supported": l2},
        "L3_final_effect_matches_target_answer_interface": {"supported": l3},
        "L4_semantic_correspondence_exceeds_source_label_identity": {"supported": l4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "valid_layers": valid, "summary": summary,
                                  "pairs": results}, indent=2))
    lines = ["# Leave-one-relation-out cross-seed causal invariance", "",
             "Each source-to-target activation map is fit without examples from the relation whose semantic edit is later translated and patched.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Alignment R2 mean/min | Own/mapped/wrong accuracy | Mapped/wrong fidelity | Source/target-answer fidelity | Mapped/target-answer cosine | Adjudicative |",
              "|---:|---:|---:|---:|---:|---:|:---:|"]
    for layer in (1, 2, 3):
        row = summary[str(layer)]
        lines.append(
            f"| {layer} | {row['alignment_r2_mean']:.3f}/{row['alignment_r2_min']:.3f} | "
            f"{row['own_accuracy']:.3f}/{row['mapped_accuracy']:.3f}/{row['wrong_accuracy']:.3f} | "
            f"{row['mapped_fidelity']:.3f}/{row['wrong_fidelity']:.3f} | "
            f"{row['source_answer_fidelity']:.3f}/{row['target_answer_fidelity']:.3f} | "
            f"{row['mapped_cosine']:.3f}/{row['target_answer_cosine']:.3f} | "
            f"{'yes' if valid[str(layer)] else 'no'} |"
        )
    lines += ["", "The alignment gate is a minimum over every ordered seed pair and omitted relation. Passing therefore cannot be driven by one easy pair. The target-answer control, unlike mapped source label identity, uses the target model's own output codebook."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
