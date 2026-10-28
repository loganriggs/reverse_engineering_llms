#!/usr/bin/env python3
"""Locate the depth where semantic effects become scaled answer-interface effects."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from audit_grammar_semantic_transfer import make_transition_examples
from run_grammar_final_interface_redteam import effect_similarity, load_model
from run_grammar_interface_depth_trace import forward


CONFIG = Path("configs/grammar_interface_depth_scale_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/interface_depth_scale_redteam.json")
REPORT = Path("artifacts/grammar_regimes/INTERFACE_DEPTH_SCALE_REDTEAM.md")


@torch.inference_mode()
def patch(model, prompts, layer, delta, base, old, new, device):
    changed, _ = forward(model, prompts, device, ActivationPatch(layer, delta, additive=True))
    rows = torch.arange(len(changed), device=device)
    target = torch.full((len(changed),), new, device=device, dtype=torch.long)
    margin = ((changed[rows, new] - changed[rows, old])
              - (base[rows, new] - base[rows, old]))
    effect = changed - base
    probability = F.softmax(base, -1)
    centered = effect - (probability * effect).sum(-1, keepdim=True)
    energy = float((probability * centered.square()).sum(-1).median())
    return effect, float((changed.argmax(-1) == target).float().mean()), float(margin.mean()), energy


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
    prototypes = [{answer: None for answer in range(8)} for _ in range(4)]
    for answer in range(8):
        indices = np.flatnonzero(eligible & (labels == answer)); rng.shuffle(indices)
        discovery = torch.as_tensor(indices[:parent["prototype_per_class"]], device=device)
        for layer in range(4):
            prototypes[layer][answer] = acts0[layer][discovery].mean(0)

    layer_rows = {layer: [] for layer in (1, 2, 3)}
    permutations = np.asarray(manifest["permutations"])
    semantic_rng = np.random.default_rng(parent["seed"] + 10000 + spec["seed"] * 100)
    count, split = parent["semantic_examples"], parent["semantic_examples"] // 2
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
            discovery_prompts = base0[:split][discover.cpu().numpy()]
            discovery_base = base0_logits[:split][discover]
            eval_prompts = base6[split:][keep.cpu().numpy()]
            eval_base = base6_logits[split:][keep]
            for layer in (1, 2, 3):
                semantic_delta = (changed0_acts[layer][:split][discover]
                                  - base0_acts[layer][:split][discover]).mean(0)
                answer_delta = prototypes[layer][new] - prototypes[layer][old]
                _, _, semantic_margin, _ = patch(
                    model, discovery_prompts, layer, semantic_delta,
                    discovery_base, old, new, device,
                )
                candidates = []
                for scale in config["scale_grid"]:
                    _, _, margin, _ = patch(model, discovery_prompts, layer,
                                            answer_delta * scale, discovery_base,
                                            old, new, device)
                    candidates.append((abs(margin - semantic_margin), scale))
                calibrated_scale = min(candidates)[1]
                norm_scale = float(semantic_delta.norm() / answer_delta.norm().clamp_min(1e-30))
                semantic_effect, semantic_accuracy, _, semantic_energy = patch(
                    model, eval_prompts, layer, semantic_delta, eval_base, old, new, device,
                )
                norm_effect, norm_accuracy, _, _ = patch(
                    model, eval_prompts, layer, answer_delta * norm_scale,
                    eval_base, old, new, device,
                )
                calibrated_effect, calibrated_accuracy, _, _ = patch(
                    model, eval_prompts, layer, answer_delta * calibrated_scale,
                    eval_base, old, new, device,
                )
                norm_cosine, norm_fidelity = effect_similarity(eval_base, semantic_effect, norm_effect)
                calibrated_cosine, calibrated_fidelity = effect_similarity(
                    eval_base, semantic_effect, calibrated_effect,
                )
                layer_rows[layer].append({
                    "semantic_accuracy": semantic_accuracy, "semantic_energy": semantic_energy,
                    "norm_accuracy": norm_accuracy, "calibrated_accuracy": calibrated_accuracy,
                    "norm_scale": norm_scale, "calibrated_scale": calibrated_scale,
                    "norm_cosine": norm_cosine, "norm_fidelity": norm_fidelity,
                    "calibrated_cosine": calibrated_cosine,
                    "calibrated_fidelity": calibrated_fidelity,
                })
    layers = {}
    for layer, rows in layer_rows.items():
        layers[str(layer)] = {key: float(np.mean([row[key] for row in rows]))
                              for key in rows[0]} | {"transitions": len(rows)}
    return {"name": spec["name"], "rho": rho, "seed": spec["seed"], "layers": layers}


def main():
    config = json.loads(CONFIG.read_text())
    depth_config = json.loads(Path(config["parent_config"]).read_text())
    parent = json.loads(Path(depth_config["parent_config"]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for spec in parent["runs"]:
        if spec["rho"] == 1.0:
            continue
        print("audit", spec["name"], flush=True)
        runs.append(audit_run(spec, parent, config, device))
    summary = {}
    for rho in (0.0, 0.5):
        selected = [run for run in runs if run["rho"] == rho]
        layers = {}
        for layer in (1, 2, 3):
            layers[str(layer)] = {key: float(np.mean([run["layers"][str(layer)][key]
                                                     for run in selected]))
                                  for key in selected[0]["layers"][str(layer)]
                                  if key != "transitions"}
        summary[str(rho)] = {"runs": len(selected), "layers": layers}
    pure = summary["0.0"]["layers"]
    best1_accuracy = max(pure["1"]["norm_accuracy"], pure["1"]["calibrated_accuracy"])
    best1_fidelity = max(pure["1"]["norm_fidelity"], pure["1"]["calibrated_fidelity"])
    b0 = (best1_accuracy <= pure["1"]["semantic_accuracy"] - .10 or best1_fidelity < .80)
    b1 = (pure["2"]["calibrated_accuracy"] >= pure["2"]["semantic_accuracy"] - .10
          and pure["2"]["calibrated_fidelity"] >= .80)
    b2 = ((pure["3"]["norm_accuracy"] >= pure["3"]["semantic_accuracy"] - .10
           and pure["3"]["norm_fidelity"] >= .80)
          or (pure["3"]["calibrated_accuracy"] >= pure["3"]["semantic_accuracy"] - .10
              and pure["3"]["calibrated_fidelity"] >= .80))
    b3 = (pure["1"]["calibrated_fidelity"] <= pure["2"]["calibrated_fidelity"]
          <= pure["3"]["calibrated_fidelity"])
    mixed_depth = json.loads(Path(depth_config.get("parent_results", "artifacts/grammar_regimes/interface_depth_trace.json")).read_text()) if depth_config.get("parent_results") else json.loads(Path("artifacts/grammar_regimes/interface_depth_trace.json").read_text())
    mixed_trace = mixed_depth["summary"]["0.5"]["layers"]
    b4 = (mixed_trace[1]["semantic_semantic_accuracy"] <= .20
          and mixed_trace[2]["semantic_semantic_accuracy"] <= .20
          and mixed_trace[3]["generic_accuracy"] >= .70)
    hypotheses = {
        "B0_layer1_contains_rule_specific_effect": {"supported": b0},
        "B1_layer2_is_already_a_scaled_answer_interface": {"supported": b1},
        "B2_layer3_is_a_scaled_answer_interface": {"supported": b2},
        "B3_interface_equivalence_increases_through_depth": {"supported": b3},
        "B4_mixed_models_lack_the_upstream_rule_effect": {"supported": b4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "summary": summary, "runs": runs}, indent=2))
    lines = ["# Calibrated depth boundary of the grammar answer interface", "",
             "At Layers 1–3, generic answer deltas are norm matched or discovery-margin calibrated before held-out semantic evaluation.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| rho | Layer | Semantic acc | Norm/calibrated acc | Norm/calibrated cosine | Norm/calibrated fidelity | Semantic effect energy | Norm/calibrated scale |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rho in ("0.0", "0.5"):
        for layer in ("1", "2", "3"):
            row = summary[rho]["layers"][layer]
            lines.append(f"| {rho} | {layer} | {row['semantic_accuracy']:.3f} | {row['norm_accuracy']:.3f}/{row['calibrated_accuracy']:.3f} | {row['norm_cosine']:.3f}/{row['calibrated_cosine']:.3f} | {row['norm_fidelity']:.3f}/{row['calibrated_fidelity']:.3f} | {row['semantic_energy']:.2e} | {row['norm_scale']:.2f}/{row['calibrated_scale']:.2f} |")
    lines += ["", "Layer 0 is omitted from relative fidelity claims because its semantic patch effect is behaviorally negligible. The boundary identifies effect equivalence, not the complete rule circuit upstream of the patched activation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
