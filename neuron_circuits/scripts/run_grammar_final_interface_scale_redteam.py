#!/usr/bin/env python3
"""Test whether final semantic patches are scaled generic output verbs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from audit_grammar_semantic_transfer import make_transition_examples
from run_grammar_final_interface_redteam import (
    effect_similarity,
    forward,
    load_model,
)


CONFIG = Path("configs/grammar_final_interface_scale_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/final_interface_scale_redteam.json")
REPORT = Path("artifacts/grammar_regimes/FINAL_INTERFACE_SCALE_REDTEAM.md")


@torch.inference_mode()
def patch(model, prompts, delta, base_logits, old, new, device):
    changed, _ = forward(model, prompts, device,
                         ActivationPatch(model.layers_count - 1, delta, additive=True))
    target = torch.full((len(changed),), new, device=device, dtype=torch.long)
    rows = torch.arange(len(changed), device=device)
    margin_change = ((changed[rows, new] - changed[rows, old])
                     - (base_logits[rows, new] - base_logits[rows, old]))
    return changed - base_logits, float((changed.argmax(-1) == target).float().mean()), float(margin_change.mean())


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
    prototypes = {}
    for answer in range(8):
        indices = np.flatnonzero(eligible & (labels == answer)); rng.shuffle(indices)
        discovery = torch.as_tensor(indices[:parent["prototype_per_class"]], device=device)
        prototypes[answer] = acts0[discovery].mean(0)

    rows = []
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
            semantic_delta = (changed0_acts[:split][discover] - base0_acts[:split][discover]).mean(0)
            answer_delta = prototypes[new] - prototypes[old]
            discovery_prompts = base0[:split][discover.cpu().numpy()]
            discovery_base = base0_logits[:split][discover]
            _, _, semantic_discovery_margin = patch(
                model, discovery_prompts, semantic_delta, discovery_base, old, new, device,
            )
            scale_rows = []
            for scale in config["scale_grid"]:
                _, _, margin = patch(model, discovery_prompts, answer_delta * scale,
                                     discovery_base, old, new, device)
                scale_rows.append((abs(margin - semantic_discovery_margin), scale, margin))
            _, calibrated_scale, _ = min(scale_rows)
            norm_scale = float(semantic_delta.norm() / answer_delta.norm().clamp_min(1e-30))

            eval_prompts = base6[split:][keep.cpu().numpy()]
            eval_base = base6_logits[split:][keep]
            semantic_effect, semantic_accuracy, _ = patch(
                model, eval_prompts, semantic_delta, eval_base, old, new, device,
            )
            unscaled_effect, unscaled_accuracy, _ = patch(
                model, eval_prompts, answer_delta, eval_base, old, new, device,
            )
            norm_effect, norm_accuracy, _ = patch(
                model, eval_prompts, answer_delta * norm_scale, eval_base, old, new, device,
            )
            calibrated_effect, calibrated_accuracy, _ = patch(
                model, eval_prompts, answer_delta * calibrated_scale, eval_base, old, new, device,
            )
            unscaled_cosine, unscaled_fidelity = effect_similarity(eval_base, semantic_effect, unscaled_effect)
            norm_cosine, norm_fidelity = effect_similarity(eval_base, semantic_effect, norm_effect)
            calibrated_cosine, calibrated_fidelity = effect_similarity(eval_base, semantic_effect, calibrated_effect)
            rows.append({
                "relation": relation, "before": before, "old": old, "new": new,
                "eligible": int(keep.sum()), "semantic_accuracy": semantic_accuracy,
                "unscaled_accuracy": unscaled_accuracy, "norm_accuracy": norm_accuracy,
                "calibrated_accuracy": calibrated_accuracy,
                "norm_scale": norm_scale, "calibrated_scale": calibrated_scale,
                "unscaled_cosine": unscaled_cosine, "unscaled_fidelity": unscaled_fidelity,
                "norm_cosine": norm_cosine, "norm_fidelity": norm_fidelity,
                "calibrated_cosine": calibrated_cosine,
                "calibrated_fidelity": calibrated_fidelity,
            })
    summary = {}
    if rows:
        for key in ("semantic_accuracy", "unscaled_accuracy", "norm_accuracy",
                    "calibrated_accuracy", "norm_scale", "calibrated_scale",
                    "unscaled_cosine", "unscaled_fidelity", "norm_cosine",
                    "norm_fidelity", "calibrated_cosine", "calibrated_fidelity"):
            summary[key] = float(np.mean([row[key] for row in rows]))
        summary["transitions"] = len(rows)
    return {"name": spec["name"], "rho": rho, "seed": spec["seed"],
            "summary": summary or None, "rows": rows}


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for spec in parent["runs"]:
        if spec["rho"] == 1.0:
            continue
        print("audit", spec["name"], flush=True)
        runs.append(audit_run(spec, parent, config, device))
    summary = {}
    for rho in (0.0, 0.5):
        selected = [r["summary"] for r in runs if r["rho"] == rho and r["summary"]]
        summary[str(rho)] = {key: float(np.mean([row[key] for row in selected]))
                             for key in selected[0] if key != "transitions"}
        summary[str(rho)]["runs"] = len(selected)
    pure, mixed = summary["0.0"], summary["0.5"]
    s0 = pure["unscaled_cosine"] >= .90
    s1 = (pure["calibrated_accuracy"] >= pure["semantic_accuracy"] - .05
          and pure["calibrated_fidelity"] >= .90)
    s2 = (pure["norm_accuracy"] >= pure["semantic_accuracy"] - .10
          and pure["norm_fidelity"] >= .80)
    s4 = mixed["calibrated_accuracy"] >= mixed["semantic_accuracy"] + .15
    hypotheses = {
        "S0_pure_semantic_and_answer_effect_directions_agree": {"supported": s0},
        "S1_discovery_calibrated_answer_delta_recovers_semantic_patch": {"supported": s1},
        "S2_norm_matching_is_sufficient": {"supported": s2},
        "S3_final_semantic_patch_is_output_verb_plus_amplitude": {"supported": s0 and s1},
        "S4_mixed_semantic_failure_is_not_fixed_by_scalar_calibration": {"supported": s4},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "summary": summary, "runs": runs}, indent=2))
    lines = ["# Direction versus amplitude in the final grammar interface", "",
             "Generic answer deltas are activation-norm matched or scalar-calibrated on discovery examples, then compared with semantic-rule patches on held-out identities and template 6.", "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", "| rho | Runs | Semantic acc | Answer unscaled/norm/calibrated acc | Effect cosine unscaled/calibrated | Fidelity unscaled/norm/calibrated | Norm/calibrated scale |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for rho in ("0.0", "0.5"):
        row = summary[rho]
        lines.append(f"| {rho} | {row['runs']} | {row['semantic_accuracy']:.3f} | {row['unscaled_accuracy']:.3f}/{row['norm_accuracy']:.3f}/{row['calibrated_accuracy']:.3f} | {row['unscaled_cosine']:.3f}/{row['calibrated_cosine']:.3f} | {row['unscaled_fidelity']:.3f}/{row['norm_fidelity']:.3f}/{row['calibrated_fidelity']:.3f} | {row['norm_scale']:.2f}/{row['calibrated_scale']:.2f} |")
    lines += ["", "Calibration uses only discovery identities/template. If it succeeds, the final-layer rule patch is a generic effect direction with rule-conditioned amplitude, not the location of the upstream rule computation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
