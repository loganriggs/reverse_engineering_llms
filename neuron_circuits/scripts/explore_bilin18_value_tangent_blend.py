#!/usr/bin/env python3
"""Explore the Pareto frontier between language-value and global-tangent CP fits."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from evaluate_bilin18_causal_cp import load_cp
from evaluate_bilin18_decomposition_metrics import (
    behavioral_evaluation,
    causal_evaluation,
    corpus_rows,
    json_ready,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_tensor_decomposition import ROOT


OUTPUT = ROOT / "value_tangent_blend_results.json"
REPORT = ROOT / "VALUE_TANGENT_BLEND_REPORT.md"
FIGURE = ROOT / "value_tangent_blend.png"
LAMBDAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0)


class BilinearBlend(nn.Module):
    def __init__(self, language, global_model, weight):
        super().__init__()
        self.language = language
        self.global_model = global_model
        self.weight = weight

    def forward(self, x):
        return (1 - self.weight) * self.language(x) + self.weight * self.global_model(x)

    def bilinear(self, x, y):
        return ((1 - self.weight) * self.language.bilinear(x, y)
                + self.weight * self.global_model.bilinear(x, y))


def write_report(result):
    lines = [
        "# Value–tangent blend experiment", "",
        "This exploratory experiment linearly blends the causally distilled language-manifold CP with the direct tensor CP:", "",
        "`f_lambda = (1 - lambda) f_language + lambda f_global`.", "",
        "It tests whether the two notions of fidelity can be combined into a better end-to-end replacement.", "",
        "![Value–tangent blend](value_tangent_blend.png)", "",
        "| Global weight λ | FineWeb KL | WikiText KL | Gaussian fidelity | Natural-mix fidelity |", 
        "|---:|---:|---:|---:|---:|",
    ]
    for key, row in result["models"].items():
        lines.append(f"| {row['lambda']:.3f} | {row['behavior']['fineweb_kl']:.4f} "
                     f"| {row['behavior']['wikitext103_kl']:.4f} "
                     f"| {row['causal']['gaussian_fidelity']:.3f} "
                     f"| {row['causal']['natural_fidelity']:.3f} |")
    best_g = max(result["models"].values(), key=lambda row: row["causal"]["gaussian_fidelity"])
    best_n = max(result["models"].values(), key=lambda row: row["causal"]["natural_fidelity"])
    best_kl = min(result["models"].values(), key=lambda row: row["behavior"]["fineweb_kl"])
    lines += ["", "## Outcome", "",
              f"- Best FineWeb behavior occurs at λ={best_kl['lambda']:.3f} (KL {best_kl['behavior']['fineweb_kl']:.4f}).",
              f"- Best Gaussian intervention fidelity occurs at λ={best_g['lambda']:.3f} ({best_g['causal']['gaussian_fidelity']:.3f}).",
              f"- Best natural-mixture fidelity occurs at λ={best_n['lambda']:.3f} ({best_n['causal']['natural_fidelity']:.3f}).",
              "", "A useful interior point would show that global and manifold structure are complementary. Monotonic degradation would show that direct tensor fidelity is not useful after downstream composition, despite its better layer-local isotropic tangent score.",
              "", "This blend doubles stored factors if implemented literally; it is a diagnostic Pareto construction, not a compression result.",
              "", "Raw results: `value_tangent_blend_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    device = "cuda"
    config = json.load(open("configs/bilin18_decomposition_evaluation.json"))
    config["bootstrap_resamples"] = 1000
    language = load_cp(ROOT / "cp2048_causal_r2.pt", device)
    global_model = load_cp(ROOT / "cp_r2048.pt", device)
    blends = {f"lambda_{weight:.3f}": (BilinearBlend(language, global_model, weight).eval(), False)
              for weight in LAMBDAS}
    model = load_full_model(device)
    corpora = corpus_rows(config)
    behavior = {}
    for corpus, rows in corpora.items():
        print("behavior", corpus, flush=True)
        behavior[corpus] = behavioral_evaluation(
            model, blends, rows[:64], config, device)
    causal = causal_evaluation(
        model, blends, corpora["fineweb"][:24], config["interventions"], config, device)
    result = {"status": "exploratory_pareto_test", "models": {}}
    for key, (blend, _) in blends.items():
        row = {
            "lambda": blend.weight,
            "behavior": {
                f"{corpus}_kl": behavior[corpus]["models"][key]["mean_kl"]["mean"]
                for corpus in behavior},
            "causal": {
                "gaussian_fidelity": causal[key]["gaussian_rms_0.10"]["fisher_effect_fidelity"],
                "gaussian_cosine": causal[key]["gaussian_rms_0.10"]["fisher_effect_cosine"],
                "natural_fidelity": causal[key]["natural_mix_0.25"]["fisher_effect_fidelity"],
                "natural_cosine": causal[key]["natural_mix_0.25"]["fisher_effect_cosine"],
            },
        }
        result["models"][key] = row
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))

    lambdas = [row["lambda"] for row in result["models"].values()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for corpus in ("fineweb", "wikitext103", "tinystories"):
        axes[0].plot(lambdas, [row["behavior"][f"{corpus}_kl"]
                              for row in result["models"].values()], marker="o", label=corpus)
    axes[0].set_xlabel("Global tensor weight λ"); axes[0].set_ylabel("KL")
    axes[0].set_title("Behavioral fidelity"); axes[0].legend()
    axes[1].plot(lambdas, [row["causal"]["gaussian_fidelity"]
                           for row in result["models"].values()], marker="o")
    axes[1].set_xlabel("Global tensor weight λ"); axes[1].set_ylabel("Fisher effect fidelity")
    axes[1].set_title("Gaussian RMS .10")
    axes[2].plot(lambdas, [row["causal"]["natural_fidelity"]
                           for row in result["models"].values()], marker="o")
    axes[2].set_xlabel("Global tensor weight λ"); axes[2].set_ylabel("Fisher effect fidelity")
    axes[2].set_title("Natural mixture .25")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    write_report(result)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
