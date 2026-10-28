#!/usr/bin/env python3
"""Calibrate generic answer operators to full semantic Fisher effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from audit_grammar_semantic_transfer import make_transition_examples
from run_grammar_final_interface_redteam import effect_similarity, load_model
from run_grammar_interface_depth_scale_redteam import patch
from run_grammar_interface_depth_trace import forward


CONFIG = Path("configs/grammar_interface_fisher_scale_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/interface_fisher_scale_redteam.json")
REPORT = Path("artifacts/grammar_regimes/INTERFACE_FISHER_SCALE_REDTEAM.md")


@torch.inference_mode()
def audit_run(spec, base_config, config, device):
    root = Path(spec["root"]); rho = spec["rho"]
    manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    labels, _, _ = labels_for_rho(manifest, "train", rho)
    facts = np.asarray(manifest["train_facts"])
    prompt0 = encode_prompts(manifest, facts, base_config["discovery_template"])
    prompt6 = encode_prompts(manifest, facts, base_config["evaluation_template"])
    logits0, acts0 = forward(model, prompt0, device)
    logits6, _ = forward(model, prompt6, device)
    labels_t = torch.as_tensor(labels, device=device)
    eligible = ((logits0.argmax(-1) == labels_t) & (logits6.argmax(-1) == labels_t)).cpu().numpy()
    rng = np.random.default_rng(base_config["seed"] + spec["seed"] * 100 + round(rho * 10))
    prototypes = [{answer: None for answer in range(8)} for _ in range(4)]
    for answer in range(8):
        indices = np.flatnonzero(eligible & (labels == answer)); rng.shuffle(indices)
        discovery = torch.as_tensor(indices[:base_config["prototype_per_class"]], device=device)
        for layer in range(4): prototypes[layer][answer] = acts0[layer][discovery].mean(0)

    rows = {layer: [] for layer in (1, 2, 3)}
    permutations = np.asarray(manifest["permutations"])
    semantic_rng = np.random.default_rng(base_config["seed"] + 10000 + spec["seed"] * 100)
    count, split = base_config["semantic_examples"], base_config["semantic_examples"] // 2
    for relation in range(4):
        for before in (0, 2, 4, 6):
            after = before + 1
            base_facts, changed_facts = make_transition_examples(semantic_rng, relation, before, after, count)
            old, new = int(permutations[relation, before]), int(permutations[relation, after])
            base0, changed0 = (encode_prompts(manifest, values, base_config["discovery_template"])
                               for values in (base_facts, changed_facts))
            base6, changed6 = (encode_prompts(manifest, values, base_config["evaluation_template"])
                               for values in (base_facts, changed_facts))
            base0_logits, base0_acts = forward(model, base0, device)
            changed0_logits, changed0_acts = forward(model, changed0, device)
            base6_logits, _ = forward(model, base6, device)
            changed6_logits, _ = forward(model, changed6, device)
            discover = ((base0_logits[:split].argmax(-1) == old)
                        & (changed0_logits[:split].argmax(-1) == new))
            keep = ((base6_logits[split:].argmax(-1) == old)
                    & (changed6_logits[split:].argmax(-1) == new))
            if int(discover.sum()) < 16 or int(keep.sum()) < 16: continue
            discovery_prompts = base0[:split][discover.cpu().numpy()]
            discovery_base = base0_logits[:split][discover]
            eval_prompts = base6[split:][keep.cpu().numpy()]
            eval_base = base6_logits[split:][keep]
            for layer in (1, 2, 3):
                semantic_delta = (changed0_acts[layer][:split][discover]
                                  - base0_acts[layer][:split][discover]).mean(0)
                answer_delta = prototypes[layer][new] - prototypes[layer][old]
                discovery_semantic, _, _, _ = patch(
                    model, discovery_prompts, layer, semantic_delta,
                    discovery_base, old, new, device,
                )
                candidates = []
                for scale in config["scale_grid"]:
                    answer_effect, _, _, _ = patch(
                        model, discovery_prompts, layer, answer_delta * scale,
                        discovery_base, old, new, device,
                    )
                    cosine, fidelity = effect_similarity(
                        discovery_base, discovery_semantic, answer_effect,
                    )
                    candidates.append((fidelity, cosine, scale))
                discovery_fidelity, discovery_cosine, scale = max(candidates)
                semantic_effect, semantic_accuracy, _, _ = patch(
                    model, eval_prompts, layer, semantic_delta,
                    eval_base, old, new, device,
                )
                answer_effect, answer_accuracy, _, _ = patch(
                    model, eval_prompts, layer, answer_delta * scale,
                    eval_base, old, new, device,
                )
                heldout_cosine, heldout_fidelity = effect_similarity(
                    eval_base, semantic_effect, answer_effect,
                )
                rows[layer].append({
                    "semantic_accuracy": semantic_accuracy,
                    "answer_accuracy": answer_accuracy, "scale": scale,
                    "discovery_cosine": discovery_cosine,
                    "discovery_fidelity": discovery_fidelity,
                    "heldout_cosine": heldout_cosine,
                    "heldout_fidelity": heldout_fidelity,
                })
    layers = {str(layer): {key: float(np.mean([row[key] for row in values]))
                           for key in values[0]} | {"transitions": len(values)}
              for layer, values in rows.items()}
    return {"name": spec["name"], "rho": rho, "seed": spec["seed"], "layers": layers}


def main():
    config = json.loads(CONFIG.read_text())
    depth_scale = json.loads(Path(config["parent_config"]).read_text())
    depth = json.loads(Path(depth_scale["parent_config"]).read_text())
    base = json.loads(Path(depth["parent_config"]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for spec in base["runs"]:
        if spec["rho"] != 0.0: continue
        print("audit", spec["name"], flush=True)
        runs.append(audit_run(spec, base, config, device))
    layers = {}
    for layer in (1, 2, 3):
        layers[str(layer)] = {key: float(np.mean([run["layers"][str(layer)][key]
                                                for run in runs]))
                              for key in runs[0]["layers"][str(layer)] if key != "transitions"}
    def passes(layer, fidelity):
        row = layers[str(layer)]
        return (row["answer_accuracy"] >= row["semantic_accuracy"] - .10
                and row["heldout_fidelity"] >= fidelity)
    f0, f1, f2 = passes(1, .80), passes(2, .80), passes(3, .90)
    f3 = all(layers[str(layer)]["heldout_fidelity"] >= .70
             and layers[str(layer)]["heldout_fidelity"]
             >= layers[str(layer)]["discovery_fidelity"] - .20 for layer in (1, 2, 3))
    hypotheses = {
        "F0_layer1_full_effect_is_a_scaled_answer_verb": {"supported": f0},
        "F1_layer2_full_effect_is_a_scaled_answer_verb": {"supported": f1},
        "F2_layer3_full_effect_is_a_scaled_answer_verb": {"supported": f2},
        "F3_fisher_scale_generalizes": {"supported": f3},
        "F4_rule_specific_effect_remains_upstream_of_boundary": {"supported": (not f0 or not f1) and f2},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "summary": layers, "runs": runs}, indent=2))
    lines = ["# Full-effect calibration of the grammar answer verb", "",
             "One scalar per transition/layer is chosen on discovery identities to maximize complete softmax-Fisher fidelity, then frozen for held-out identities and template 6.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Semantic/answer accuracy | Discovery cosine/fidelity | Held-out cosine/fidelity | Scale |",
              "|---:|---:|---:|---:|---:|"]
    for layer in ("1", "2", "3"):
        row = layers[layer]
        lines.append(f"| {layer} | {row['semantic_accuracy']:.3f}/{row['answer_accuracy']:.3f} | {row['discovery_cosine']:.3f}/{row['discovery_fidelity']:.3f} | {row['heldout_cosine']:.3f}/{row['heldout_fidelity']:.3f} | {row['scale']:.2f} |")
    lines += ["", "Full-effect calibration is stricter than target-class calibration but still only tests a one-dimensional effect ray. Failure can reflect state-dependent scale or a genuinely different direction; the discovery/held-out gap distinguishes them partially."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__": main()
