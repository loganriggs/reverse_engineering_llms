#!/usr/bin/env python3
"""Shapley-decompose semantic activation patches into answer verb and residual."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from audit_grammar_semantic_transfer import make_transition_examples
from run_grammar_final_interface_redteam import effect_similarity, load_model
from run_grammar_interface_depth_trace import forward


CONFIG = Path("configs/grammar_interface_residual_shapley.json")
OUTPUT = Path("artifacts/grammar_regimes/interface_residual_shapley.json")
REPORT = Path("artifacts/grammar_regimes/INTERFACE_RESIDUAL_SHAPLEY.md")


@torch.inference_mode()
def changed_logits(model, prompts, layer, delta, device):
    logits, _ = forward(model, prompts, device, ActivationPatch(layer, delta, additive=True))
    return logits


def centered_inner(base, left, right):
    probability = F.softmax(base, -1)
    left = left - (probability * left).sum(-1, keepdim=True)
    right = right - (probability * right).sum(-1, keepdim=True)
    return (probability * left * right).sum(-1)


@torch.inference_mode()
def audit_run(spec, base_config, fisher_config, device):
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
    rng = np.random.default_rng(base_config["seed"] + spec["seed"] * 100)
    prototypes = [{answer: None for answer in range(8)} for _ in range(4)]
    for answer in range(8):
        indices = np.flatnonzero(eligible & (labels == answer)); rng.shuffle(indices)
        discovery = torch.as_tensor(indices[:base_config["prototype_per_class"]], device=device)
        for layer in range(4): prototypes[layer][answer] = acts0[layer][discovery].mean(0)

    rows = {layer: [] for layer in (1, 2, 3)}
    permutations = np.asarray(manifest["permutations"])
    semantic_rng = np.random.default_rng(base_config["seed"] + 10000 + spec["seed"] * 100)
    count, split = base_config["semantic_examples"], base_config["semantic_examples"] // 2
    max_error = 0.0
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
            discover_prompts = base0[:split][discover.cpu().numpy()]
            discover_base = base0_logits[:split][discover]
            eval_prompts = base6[split:][keep.cpu().numpy()]
            eval_base = base6_logits[split:][keep]
            target = torch.full((int(keep.sum()),), new, device=device, dtype=torch.long)
            for layer in (1, 2, 3):
                semantic_delta = (changed0_acts[layer][:split][discover]
                                  - base0_acts[layer][:split][discover]).mean(0)
                answer_delta = prototypes[layer][new] - prototypes[layer][old]
                discovery_semantic = changed_logits(
                    model, discover_prompts, layer, semantic_delta, device) - discover_base
                candidates = []
                for scale in fisher_config["scale_grid"]:
                    effect = changed_logits(
                        model, discover_prompts, layer, answer_delta * scale, device) - discover_base
                    cosine, fidelity = effect_similarity(discover_base, discovery_semantic, effect)
                    candidates.append((fidelity, cosine, scale))
                _, _, scale = max(candidates)
                answer = answer_delta * scale
                residual = semantic_delta - answer
                logits_answer = changed_logits(model, eval_prompts, layer, answer, device)
                logits_residual = changed_logits(model, eval_prompts, layer, residual, device)
                logits_both = changed_logits(model, eval_prompts, layer, answer + residual, device)
                effect_answer = logits_answer - eval_base
                effect_residual = logits_residual - eval_base
                effect_both = logits_both - eval_base
                interaction = effect_both - effect_answer - effect_residual
                shapley_answer = effect_answer + .5 * interaction
                shapley_residual = effect_residual + .5 * interaction
                max_error = max(max_error, float(
                    (shapley_answer + shapley_residual - effect_both).abs().max()))
                energy = centered_inner(eval_base, effect_both, effect_both).clamp_min(1e-30)
                projection_answer = centered_inner(eval_base, shapley_answer, effect_both) / energy
                projection_residual = centered_inner(eval_base, shapley_residual, effect_both) / energy
                interaction_relative = torch.sqrt(
                    centered_inner(eval_base, interaction, interaction).clamp_min(0) / energy)
                denominator = projection_answer.abs() + projection_residual.abs()
                residual_share = projection_residual.abs() / denominator.clamp_min(1e-30)
                rows[layer].append({
                    "answer_projection": float(projection_answer.median()),
                    "residual_projection": float(projection_residual.median()),
                    "residual_absolute_share": float(residual_share.median()),
                    "interaction_relative_norm": float(interaction_relative.median()),
                    "semantic_accuracy": float((logits_both.argmax(-1) == target).float().mean()),
                    "answer_accuracy": float((logits_answer.argmax(-1) == target).float().mean()),
                    "residual_accuracy": float((logits_residual.argmax(-1) == target).float().mean()),
                    "scale": scale,
                })
    layers = {str(layer): {key: float(np.mean([row[key] for row in values]))
                           for key in values[0]} | {"transitions": len(values)}
              for layer, values in rows.items()}
    return {"name": spec["name"], "seed": spec["seed"], "layers": layers,
            "max_recomposition_error": max_error}


def main():
    config = json.loads(CONFIG.read_text())
    fisher_config = json.loads(Path(config["parent_config"]).read_text())
    depth_scale = json.loads(Path(fisher_config["parent_config"]).read_text())
    depth = json.loads(Path(depth_scale["parent_config"]).read_text())
    base = json.loads(Path(depth["parent_config"]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for spec in base["runs"]:
        if spec["rho"] != 0.0: continue
        print("audit", spec["name"], flush=True)
        runs.append(audit_run(spec, base, fisher_config, device))
    layers = {str(layer): {key: float(np.mean([run["layers"][str(layer)][key]
                                              for run in runs]))
                           for key in runs[0]["layers"][str(layer)] if key != "transitions"}
              for layer in (1, 2, 3)}
    max_error = max(run["max_recomposition_error"] for run in runs)
    final = layers["3"]
    c0 = max_error <= 1e-5
    c1 = (final["answer_projection"] >= .80
          and final["residual_absolute_share"] <= .20
          and final["interaction_relative_norm"] <= .20)
    c2 = any(layers[str(layer)]["residual_absolute_share"] >= .25 for layer in (1, 2))
    c3 = any(layers[str(layer)]["interaction_relative_norm"]
             >= final["interaction_relative_norm"] + .10 for layer in (1, 2))
    c4 = all(layers[str(layer)]["residual_accuracy"]
             <= layers[str(layer)]["semantic_accuracy"] - .20 for layer in (1, 2, 3))
    hypotheses = {
        "C0_shapley_recomposes_semantic_effect": {"supported": c0},
        "C1_final_effect_is_answer_verb_dominated": {"supported": c1},
        "C2_rule_specific_residual_matters_upstream": {"supported": c2},
        "C3_interaction_is_more_important_upstream": {"supported": c3},
        "C4_residual_alone_is_not_the_rule": {"supported": c4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "max_recomposition_error": max_error,
                                  "summary": layers, "runs": runs}, indent=2))
    lines = ["# Shapley split of generic answer verb and semantic residual", "",
             "The exact semantic activation patch is split into the discovery-Fisher-scaled answer ray and its residual; all four finite coalitions are evaluated on held-out identities/template 6.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items(): lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum vector recomposition error: `{max_error:.2e}`.", "",
              "| Layer | Answer/residual target projection | Residual absolute share | Interaction relative norm | Semantic/answer/residual accuracy | Scale |",
              "|---:|---:|---:|---:|---:|---:|"]
    for layer in ("1", "2", "3"):
        row = layers[layer]
        lines.append(f"| {layer} | {row['answer_projection']:.3f}/{row['residual_projection']:.3f} | {row['residual_absolute_share']:.3f} | {row['interaction_relative_norm']:.3f} | {row['semantic_accuracy']:.3f}/{row['answer_accuracy']:.3f}/{row['residual_accuracy']:.3f} | {row['scale']:.2f} |")
    lines += ["", "Shapley allocation is exact for this declared two-component activation game. Changing the answer ray or calibration changes the residual and therefore the allocation; causal dominance is conditional, not a unique storage percentage."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__": main()
