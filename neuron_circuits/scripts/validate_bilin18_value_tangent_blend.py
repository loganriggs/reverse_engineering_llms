#!/usr/bin/env python3
"""Disjoint validation of the selected value–tangent blend coefficient."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evaluate_bilin18_causal_cp import load_cp
from evaluate_bilin18_decomposition_metrics import (
    behavioral_evaluation,
    causal_evaluation,
    json_ready,
)
from explore_bilin18_value_tangent_blend import BilinearBlend
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT


CONFIG = Path("configs/bilin18_value_tangent_blend_validation.json")
OOD = Path("artifacts/bilin18_ood_corpora")
OUTPUT = ROOT / "value_tangent_blend_validation.json"
REPORT = ROOT / "VALUE_TANGENT_BLEND_VALIDATION.md"


def load_rows(config):
    sources = {
        "fineweb": np.load(DEFAULT_CORPUS, mmap_mode="r"),
        "wikitext103": np.load(OOD / "wikitext103_1200x129.npy", mmap_mode="r"),
        "tinystories": np.load(OOD / "tinystories_1200x129.npy", mmap_mode="r"),
    }
    return {name: source[config["corpora"][name]["start"]:
                                config["corpora"][name]["start"] + config["corpora"][name]["rows"]]
            for name, source in sources.items()}


def main():
    config = json.loads(CONFIG.read_text())
    device = "cuda"
    language = load_cp(ROOT / "cp2048_causal_r2.pt", device)
    global_model = load_cp(ROOT / "cp_r2048.pt", device)
    lambdas = (0.0, config["selected_lambda"], config["overshoot_control_lambda"])
    candidates = {f"lambda_{weight:.1f}": (BilinearBlend(language, global_model, weight).eval(), False)
                  for weight in lambdas}
    model = load_full_model(device)
    corpora = load_rows(config)
    result = {"config": config, "behavior": {}, "causal": {}}
    for corpus, rows in corpora.items():
        print("validate behavior", corpus, flush=True)
        result["behavior"][corpus] = behavioral_evaluation(
            model, candidates, rows, config, device)
        print("validate causal", corpus, flush=True)
        result["causal"][corpus] = causal_evaluation(
            model, candidates, rows[:config["causal_rows"]], config["interventions"],
            config, device)

    base, selected, overshoot = "lambda_0.0", "lambda_0.2", "lambda_0.4"
    kl = lambda corpus, name: result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
    causal = lambda corpus, name, condition: result["causal"][corpus][name][condition]["fisher_effect_fidelity"]
    behavior_deltas = {corpus: kl(corpus, selected) - kl(corpus, base) for corpus in corpora}
    causal_deltas = {f"{corpus}/{condition}": causal(corpus, selected, condition) - causal(corpus, base, condition)
                     for corpus in corpora for condition in config["interventions"]}
    selected_mean_causal = np.mean([causal(corpus, selected, condition)
                                    for corpus in corpora for condition in config["interventions"]])
    overshoot_mean_causal = np.mean([causal(corpus, overshoot, condition)
                                     for corpus in corpora for condition in config["interventions"]])
    result["hypotheses"] = {
        "H1_behavior": {"supported": behavior_deltas["fineweb"] < 0
                        and behavior_deltas["tinystories"] < 0
                        and behavior_deltas["wikitext103"] <= 0.002,
                        "kl_deltas": behavior_deltas},
        "H2_causal": {"supported": all(value > 0 for value in causal_deltas.values()),
                      "fidelity_deltas": causal_deltas},
        "H3_interior": {"supported": kl("fineweb", selected) < kl("fineweb", overshoot)
                        and selected_mean_causal > overshoot_mean_causal,
                        "selected_mean_causal": float(selected_mean_causal),
                        "overshoot_mean_causal": float(overshoot_mean_causal)},
    }
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))

    lines = ["# Value–tangent blend disjoint validation", "", "## Outcome", ""]
    for name, row in result["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Behavior", "",
              "| Corpus | λ=0 KL | λ=.20 KL | λ=.40 KL |", "|---|---:|---:|---:|"]
    for corpus in corpora:
        lines.append(f"| {corpus} | {kl(corpus, base):.4f} | {kl(corpus, selected):.4f} | {kl(corpus, overshoot):.4f} |")
    lines += ["", "## Causal fidelity", "",
              "| Corpus/intervention | λ=0 | λ=.20 | λ=.40 |", "|---|---:|---:|---:|"]
    for corpus in corpora:
        for condition in config["interventions"]:
            lines.append(f"| {corpus}/{condition} | {causal(corpus, base, condition):.3f} "
                         f"| {causal(corpus, selected, condition):.3f} "
                         f"| {causal(corpus, overshoot, condition):.3f} |")
    lines += ["", "The λ=.20 coefficient and row ranges were frozen before this run. The blend remains a rank-4096 diagnostic unless recompressed.", "",
              "Raw results: `value_tangent_blend_validation.json`."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
