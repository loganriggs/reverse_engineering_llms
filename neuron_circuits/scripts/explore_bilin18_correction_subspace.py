#!/usr/bin/env python3
"""Basis-invariant output-subspace rank curves for the validated CP blend."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_bilin18_causal_cp import load_cp
from evaluate_bilin18_decomposition_metrics import behavioral_evaluation, causal_evaluation, json_ready
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_tensor_decomposition import ROOT
from validate_bilin18_value_tangent_blend import CONFIG as EVAL_CONFIG, load_rows


CONFIG = Path("configs/bilin18_correction_subspace.json")
DATA = Path("artifacts/bilin18_sparse_codes/layer1_inputs_large.npz")
OUTPUT = ROOT / "correction_subspace_results.json"
REPORT = ROOT / "CORRECTION_SUBSPACE_REPORT.md"
FIGURE = ROOT / "correction_subspace.png"


def cp_tangent(model, x, v):
    left, right, output, gain = model.factors()
    feature = ((F.linear(v, left) * F.linear(x, right)
                + F.linear(x, left) * F.linear(v, right)) * gain)
    return F.linear(feature, output)


@torch.inference_mode()
def fit_covariances(language, direct, x_cpu, examples, seed):
    covariances = {name: torch.zeros(1152, 1152, device="cuda", dtype=torch.float64)
                   for name in ("value", "gaussian_tangent", "natural_tangent")}
    energies = {name: 0.0 for name in covariances}
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for start in range(0, examples, 128):
        x = x_cpu[start:start + 128].cuda()
        donor = x.roll(1, 0)
        gaussian = torch.randn(x.shape, generator=generator, device="cuda")
        gaussian = gaussian / gaussian.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        rows = {
            "value": direct(x) - language(x),
            "gaussian_tangent": cp_tangent(direct, x, gaussian)
                                - cp_tangent(language, x, gaussian),
            "natural_tangent": cp_tangent(direct, x, donor - x)
                               - cp_tangent(language, x, donor - x),
        }
        for name, values in rows.items():
            values = values.double()
            covariances[name].addmm_(values.T, values)
            energies[name] += float(values.square().sum())
        if (start // 128 + 1) % 32 == 0:
            print("covariance", start + len(x), "/", examples, flush=True)
    normalized = {name: matrix / max(energies[name], 1e-30)
                  for name, matrix in covariances.items()}
    return covariances, normalized, energies


@torch.inference_mode()
def basis_from_covariance(covariance):
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = eigenvalues.argsort(descending=True)
    return eigenvectors[:, order].float(), eigenvalues[order].clamp_min(0).float()


class ProjectedBlend(nn.Module):
    def __init__(self, language, direct, basis, rank, weight):
        super().__init__()
        self.language = language
        self.direct = direct
        self.register_buffer("basis", basis[:, :rank].contiguous())
        self.weight = weight

    def forward(self, x):
        base = self.language(x)
        if self.basis.shape[1] == 0:
            return base
        correction = self.direct(x) - base
        projected = (correction @ self.basis) @ self.basis.T
        return base + self.weight * projected


def fraction(base, selected, full, higher):
    numerator = selected - base if higher else base - selected
    denominator = full - base if higher else base - full
    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("nan")


def main():
    config = json.loads(CONFIG.read_text())
    eval_config = json.loads(EVAL_CONFIG.read_text())
    eval_config["bootstrap_resamples"] = 1000
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    direct = load_cp(ROOT / "cp_r2048.pt", "cuda")
    for model in (language, direct):
        model.requires_grad_(False)
    archive = np.load(DATA)
    x = torch.from_numpy(archive["fineweb_train__input1"].astype(np.float32))
    raw_covariances, normalized, energies = fit_covariances(
        language, direct, x, config["fit_examples"], config["seed"])
    value_basis, value_eigenvalues = basis_from_covariance(raw_covariances["value"])
    hybrid_covariance = sum(normalized.values())
    hybrid_basis, hybrid_eigenvalues = basis_from_covariance(hybrid_covariance)
    bases = {"value": value_basis, "hybrid": hybrid_basis}
    candidates = {}
    for basis_name, basis in bases.items():
        for rank in config["ranks"]:
            candidates[f"{basis_name}_{rank}"] = (
                ProjectedBlend(language, direct, basis, rank,
                               config["blend_lambda"]).eval(), False)

    model = load_full_model("cuda")
    corpora = load_rows(eval_config)
    result = {"config": config, "fit": {
        "component_energy": energies,
        "value_explained": (value_eigenvalues.cumsum(0) / value_eigenvalues.sum()).cpu().tolist(),
        "hybrid_explained": (hybrid_eigenvalues.cumsum(0) / hybrid_eigenvalues.sum()).cpu().tolist(),
    }, "behavior": {}, "causal": {}}
    for corpus, rows in corpora.items():
        print("subspace behavior", corpus, flush=True)
        result["behavior"][corpus] = behavioral_evaluation(
            model, candidates, rows, eval_config, "cuda")
        print("subspace causal", corpus, flush=True)
        result["causal"][corpus] = causal_evaluation(
            model, candidates, rows[:eval_config["causal_rows"]],
            eval_config["interventions"], eval_config, "cuda")

    kl = lambda corpus, name: result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
    causal = lambda corpus, name, condition: result["causal"][corpus][name][condition]["fisher_effect_fidelity"]
    base, full = "hybrid_0", "hybrid_1152"
    h1_fraction = fraction(kl("fineweb", base), kl("fineweb", "value_128"),
                           kl("fineweb", full), False)
    h2_fractions = {condition: fraction(
        causal("fineweb", base, condition), causal("fineweb", "hybrid_256", condition),
        causal("fineweb", full, condition), True)
        for condition in eval_config["interventions"]}
    transfer = {corpus: (kl(corpus, "hybrid_256") < kl(corpus, base)
                         and all(causal(corpus, "hybrid_256", condition)
                                 > causal(corpus, base, condition)
                                 for condition in eval_config["interventions"]))
                for corpus in corpora}
    mean_kl = lambda name: float(np.mean([kl(corpus, name) for corpus in corpora]))
    mean_causal = lambda name: float(np.mean([
        causal(corpus, name, condition) for corpus in corpora
        for condition in eval_config["interventions"]]))
    result["hypotheses"] = {
        "H1_value_compact": {"supported": h1_fraction >= .8,
                             "recovered_fraction": h1_fraction},
        "H2_hybrid_causal_compact": {"supported": all(v >= .7 for v in h2_fractions.values())
                                      and all(causal("fineweb", "hybrid_256", condition)
                                              >= causal("fineweb", base, condition)
                                              for condition in eval_config["interventions"]),
                                     "recovered_fractions": h2_fractions},
        "H3_hybrid_transfer": {"supported": sum(transfer.values()) >= 2,
                               "corpus_wins": transfer},
        "H4_hybrid_tradeoff": {"supported": mean_causal("hybrid_256")
                                > mean_causal("value_256")
                                and mean_kl("hybrid_256") <= mean_kl("value_256") + .0005,
                               "hybrid_mean_causal": mean_causal("hybrid_256"),
                               "value_mean_causal": mean_causal("value_256"),
                               "hybrid_mean_kl": mean_kl("hybrid_256"),
                               "value_mean_kl": mean_kl("value_256")},
    }
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))

    lines = ["# Basis-invariant correction output subspaces", "", "## Hypotheses", ""]
    for name, row in result["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Correction subspace curves](correction_subspace.png)", "",
              "| Basis / rank | FineWeb KL | FW Gaussian | FW natural | Value variance | Hybrid variance |",
              "|---|---:|---:|---:|---:|---:|"]
    for rank in config["ranks"]:
        for basis_name in ("value", "hybrid"):
            name = f"{basis_name}_{rank}"
            value_variance = 0 if rank == 0 else result["fit"]["value_explained"][rank - 1]
            hybrid_variance = 0 if rank == 0 else result["fit"]["hybrid_explained"][rank - 1]
            lines.append(f"| {basis_name} / {rank} | {kl('fineweb', name):.4f} "
                         f"| {causal('fineweb', name, 'gaussian_rms_0.10'):.3f} "
                         f"| {causal('fineweb', name, 'natural_mix_0.25'):.3f} "
                         f"| {value_variance:.3f} | {hybrid_variance:.3f} |")
    lines += ["", "The subspace is in the 1,152-dimensional layer-write space and is invariant to permutation/rescaling of CP terms. It is a descriptive projection: both parent CP models are still evaluated, so this is not yet a runtime compression result.", "",
              "Value PCA fits natural correction writes. Hybrid PCA equally weights normalized covariance from values, Gaussian tangents, and natural tangents.", "",
              "Raw results: `correction_subspace_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    xs = np.asarray(config["ranks"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for basis_name in bases:
        axes[0].plot(xs, [kl("fineweb", f"{basis_name}_{rank}") for rank in xs], marker="o", label=basis_name)
        axes[1].plot(xs, [causal("fineweb", f"{basis_name}_{rank}", "gaussian_rms_0.10") for rank in xs], marker="o")
        axes[2].plot(xs, [causal("fineweb", f"{basis_name}_{rank}", "natural_mix_0.25") for rank in xs], marker="o")
    axes[0].set_ylabel("FineWeb KL"); axes[0].legend()
    axes[1].set_ylabel("Gaussian causal fidelity")
    axes[2].set_ylabel("Natural causal fidelity")
    for axis in axes:
        axis.set_xlabel("Correction output-subspace rank")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
