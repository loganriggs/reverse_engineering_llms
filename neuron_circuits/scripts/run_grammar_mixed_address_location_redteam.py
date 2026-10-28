#!/usr/bin/env python3
"""Locate whether attention assembles the quadratic arbitrary-fact address."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from run_grammar_final_interface_redteam import load_model
from run_grammar_mixed_address_router import (collect, fit_pca, fit_ridge, predict_ridge,
                                               prototypes, apply_prototypes, quadratic,
                                               score, select_panel)


CONFIG = Path("configs/grammar_mixed_address_location_redteam.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_mixed_address_location_redteam.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_MIXED_ADDRESS_LOCATION_REDTEAM.md")


def cross_features(boundary, attention, rank):
    b, a = boundary[:, :rank], attention[:, :rank]; upper = torch.triu_indices(rank, rank)
    return torch.cat((b, a, b[:, upper[0]] * b[:, upper[1]],
                      a[:, upper[0]] * a[:, upper[1]],
                      (b[:, :, None] * a[:, None, :]).flatten(1)), 1)


def analyze(spec, config, device):
    root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
    model = load_model(root / f"rho_{config['rho']:g}/depth_4/checkpoint.pt", device)
    facts = np.asarray(manifest["train_facts"]); labels, selected, changed = labels_for_rho(manifest, "train", config["rho"])
    eligible = np.ones(len(facts), bool)
    for template in (config["discovery_template"], config["evaluation_template"]):
        prompts = encode_prompts(manifest, facts, template)
        for start in range(0, len(prompts), 1024):
            tokens = torch.as_tensor(prompts[start:start + 1024].astype(np.int64), device=device)
            prediction = model(tokens)[:, -1, list(VOCAB.answers)].argmax(-1).cpu().numpy()
            eligible[start:start + len(prediction)] &= prediction == labels[start:start + len(prediction)]
    panel = select_panel(labels, selected, changed, eligible, facts, config, config["seed"] + spec["seed"])
    indices = {name: panel[True][name] for name in ("discovery", "validation", "evaluation")}
    train, _, _ = collect(model, encode_prompts(manifest, facts[indices["discovery"]], config["discovery_template"]), device)
    validation, _, _ = collect(model, encode_prompts(manifest, facts[indices["validation"]], config["discovery_template"]), device)
    evaluation, forward_error, tensor_error = collect(model, encode_prompts(manifest, facts[indices["evaluation"]], config["evaluation_template"]), device)
    sources = ("boundary2_full", "boundary3_final", "attn_write", "premlp")
    pca = {source: fit_pca(train[source], [validation[source], evaluation[source]], config["pca_rank"])
           for source in sources}
    features = {}
    for source in sources:
        for rank in config["quadratic_ranks"]:
            features[f"{source}_quadratic_{rank}"] = tuple(quadratic(values, rank) for values in pca[source])
    cross_rank = config["cross_rank"]
    features[f"boundary_attention_cross_{cross_rank}"] = tuple(
        cross_features(pca["boundary3_final"][position], pca["attn_write"][position], cross_rank)
        for position in range(3)
    )
    generator = torch.Generator().manual_seed(config["seed"] + spec["seed"])
    shuffled = train["write"][torch.randperm(len(train["write"]), generator=generator)]
    fitted = {name: fit_ridge(values[0], train["write"], values[1], validation["write"], config["ridge_relative"])
              for name, values in features.items()}
    fitted["premlp_quadratic_32_shuffled"] = fit_ridge(features["premlp_quadratic_32"][0], shuffled,
                                                        features["premlp_quadratic_32"][1], validation["write"], config["ridge_relative"])
    predictions = {name: predict_ridge(values[2], fitted[name]) for name, values in features.items()}
    predictions["premlp_quadratic_32_shuffled"] = predict_ridge(features["premlp_quadratic_32"][2], fitted["premlp_quadratic_32_shuffled"])
    block = model.blocks[3]
    predictions["exact_bilinear_tensor"] = ((evaluation["premlp"] @ block.left.weight.detach().cpu().T)
                                             * (evaluation["premlp"] @ block.right.weight.detach().cpu().T)) @ block.down.weight.detach().cpu().T
    train_relation, train_labels = facts[indices["discovery"], 2], labels[indices["discovery"]]
    eval_relation, eval_labels = facts[indices["evaluation"], 2], labels[indices["evaluation"]]
    prototype_table = prototypes(train["write"], train_relation, train_labels, True)
    prototype = apply_prototypes(prototype_table, eval_relation, eval_labels, True)
    methods = {name: score(values, evaluation["write"], evaluation["after"], evaluation["zero_logits"],
                           evaluation["logits"], eval_labels, prototype, model,
                           config["minimum_fisher_effect_norm"])
               for name, values in predictions.items()}
    return {"seed": spec["seed"], "maximum_forward_error": forward_error, "maximum_tensor_error": tensor_error,
            "evaluation_facts": len(indices["evaluation"]), "methods": methods}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; runs = []
    for spec in config["models"]:
        print("redteam", spec["seed"], flush=True); runs.append(analyze(spec, config, device))
    def value(run, name): return run["methods"][name]["median_effect_fidelity"]
    l0 = all(value(run, "boundary3_final_quadratic_32") >= .7 for run in runs)
    l1 = all(value(run, "boundary2_full_quadratic_32") >= .7 for run in runs)
    l2 = all(value(run, "attn_write_quadratic_32") >= .5 for run in runs)
    l3 = all(value(run, "premlp_quadratic_32") >= value(run, "boundary3_final_quadratic_32") + .15 for run in runs)
    l4 = all(value(run, f"boundary_attention_cross_{config['cross_rank']}") >= .7 and
             value(run, f"boundary_attention_cross_{config['cross_rank']}") >= value(run, "premlp_quadratic_32") - .1 for run in runs)
    l5 = all(value(run, "premlp_quadratic_32") >= .75 and value(run, "premlp_quadratic_32") >= value(run, "premlp_quadratic_8") + .2 for run in runs)
    l6 = all(value(run, "premlp_quadratic_48") < value(run, "premlp_quadratic_32") + .05 for run in runs)
    l7 = all(value(run, "premlp_quadratic_32") >= value(run, "premlp_quadratic_32_shuffled") + .2 for run in runs)
    l8 = all(value(run, "exact_bilinear_tensor") >= .9999 and run["methods"]["exact_bilinear_tensor"]["effectful_examples"] >= 64 and
             run["maximum_forward_error"] <= 1e-5 and run["maximum_tensor_error"] <= 1e-5 for run in runs)
    hypotheses = {name: {"supported": bool(result)} for name, result in zip(config["hypotheses"], (l0,l1,l2,l3,l4,l5,l6,l7,l8))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Where is the mixed arbitrary-fact quadratic address assembled?", "",
             "Matched ridge budgets predict the exact final-block residual write from quadratic codes at successive boundaries. All evaluation facts are learned randomized assignments, disjoint from fitting, and encoded with held-out template 6. The explicit boundary×attention model contains rank-16 self terms for each input plus all 256 cross terms.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    names = ("boundary2_full_quadratic_32", "boundary3_final_quadratic_32", "attn_write_quadratic_32", f"boundary_attention_cross_{config['cross_rank']}", "premlp_quadratic_8", "premlp_quadratic_16", "premlp_quadratic_24", "premlp_quadratic_32", "premlp_quadratic_48", "premlp_quadratic_32_shuffled", "exact_bilinear_tensor")
    lines += ["", "| Seed | " + " | ".join(names) + " |", "|---:|" + "---:|" * len(names)]
    for run in runs:
        lines.append(f"| {run['seed']} | " + " | ".join(f"{value(run, name):.3f}" for name in names) + " |")
    lines += ["", "A successful earlier-boundary quadratic predictor means the address is already present in a multiplexed upstream state. A large pre-MLP advantage means final attention assembles or rotates the address before the bilinear decoder. Cross-term success identifies cooperation between the incoming residual and attention write rather than either source alone."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
