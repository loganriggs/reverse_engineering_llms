#!/usr/bin/env python3
"""Measure how much extra CP rank recovers the validated blend's gains."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_bilin18_causal_cp import load_cp
from evaluate_bilin18_decomposition_metrics import behavioral_evaluation, causal_evaluation, json_ready
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_tensor_decomposition import ROOT
from validate_bilin18_value_tangent_blend import CONFIG, load_rows


DATA = "artifacts/bilin18_sparse_codes/layer1_inputs_large.npz"
OUTPUT = ROOT / "blend_rank_curve_results.json"
REPORT = ROOT / "BLEND_RANK_CURVE_REPORT.md"
FIGURE = ROOT / "blend_rank_curve.png"
COUNTS = (0, 128, 256, 512, 1024, 2048)


class SelectiveBlend(nn.Module):
    """Language model plus selected lambda*(direct-language) term corrections."""

    def __init__(self, language, direct, indices, weight=0.2):
        super().__init__()
        self.language = language
        self.direct = direct
        self.register_buffer("indices", indices)
        self.weight = weight

    @staticmethod
    def selected_write(model, x, indices):
        if not len(indices):
            return torch.zeros_like(x)
        left, right, output, gain = model.factors()
        left, right = left[indices], right[indices]
        feature = F.linear(x, left) * F.linear(x, right) * gain[indices]
        return F.linear(feature, output[:, indices])

    def forward(self, x):
        base = self.language(x)
        correction = (self.selected_write(self.direct, x, self.indices)
                      - self.selected_write(self.language, x, self.indices))
        return base + self.weight * correction


@torch.inference_mode()
def correction_ranking(language, direct, x_cpu, examples=16384):
    language_features, direct_features = [], []
    for start in range(0, examples, 256):
        x = x_cpu[start:start + 256].cuda()
        for model, rows in ((language, language_features), (direct, direct_features)):
            left, right, _, gain = model.factors()
            rows.append((F.linear(x, left) * F.linear(x, right) * gain).cpu())
    language_features = torch.cat(language_features)
    direct_features = torch.cat(direct_features)
    language_output = language.factors()[2].cpu()
    direct_output = direct.factors()[2].cpu()
    output_dot = (language_output * direct_output).sum(0)
    # Per-pair expected squared norm of direct_i(x)-language_i(x).
    score = (direct_features.square().mean(0) + language_features.square().mean(0)
             - 2 * (direct_features * language_features).mean(0) * output_dot)
    return score.argsort(descending=True), score


def main():
    config = json.loads(CONFIG.read_text())
    config["bootstrap_resamples"] = 1000
    device = "cuda"
    language = load_cp(ROOT / "cp2048_causal_r2.pt", device)
    direct = load_cp(ROOT / "cp_r2048.pt", device)
    archive = np.load(DATA)
    train_x = torch.from_numpy(archive["fineweb_train__input1"].astype(np.float32))
    ranking, scores = correction_ranking(language, direct, train_x)
    candidates = {}
    for count in COUNTS:
        indices = ranking[:count].to(device)
        candidates[f"top_{count}"] = (SelectiveBlend(language, direct, indices).eval(), False)
    generator = torch.Generator().manual_seed(20261029)
    random_indices = torch.randperm(2048, generator=generator)[:512].to(device)
    candidates["random_512"] = (SelectiveBlend(language, direct, random_indices).eval(), False)

    model = load_full_model(device)
    corpora = load_rows(config)
    result = {"status": "exploratory_rank_curve", "selection": {
        "examples": 16384, "score": "individual expected correction-write energy",
        "top_score_fraction": {str(count): float(scores[ranking[:count]].sum() / scores.sum())
                               for count in COUNTS if count}},
        "behavior": {}, "causal": {}}
    for corpus, rows in corpora.items():
        print("rank curve behavior", corpus, flush=True)
        result["behavior"][corpus] = behavioral_evaluation(
            model, candidates, rows, config, device)
        print("rank curve causal", corpus, flush=True)
        result["causal"][corpus] = causal_evaluation(
            model, candidates, rows[:config["causal_rows"]], config["interventions"],
            config, device)
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))

    kl = lambda corpus, name: result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
    causal = lambda corpus, name, condition: result["causal"][corpus][name][condition]["fisher_effect_fidelity"]
    lines = ["# Rank curve for direct-tensor corrections", "",
             "Each selected component replaces one language CP term by a two-term 0.8/0.2 blend, so the literal upper-bound rank is `2048 + selected_count`.", "",
             "![Blend rank curve](blend_rank_curve.png)", "",
             "| Selected corrections | Rank upper bound | Score energy | FineWeb KL | FW Gaussian | FW natural |", 
             "|---:|---:|---:|---:|---:|---:|"]
    for count in COUNTS:
        name = f"top_{count}"
        energy = 0.0 if count == 0 else result["selection"]["top_score_fraction"][str(count)]
        lines.append(f"| {count} | {2048 + count} | {energy:.3f} | {kl('fineweb', name):.4f} "
                     f"| {causal('fineweb', name, 'gaussian_rms_0.10'):.3f} "
                     f"| {causal('fineweb', name, 'natural_mix_0.25'):.3f} |")
    lines += ["", "## Random-selection control", "",
              f"At 512 corrections, energy-ranked selection gives FineWeb KL {kl('fineweb', 'top_512'):.4f}, versus {kl('fineweb', 'random_512'):.4f} for random selection.", "",
              "A steep early curve supports a sparse high-sensitivity correction. A near-linear curve supports broadly distributed complementary rank. Selection used FineWeb training activations; cross-corpus results in the JSON are transfer tests.", "",
              "Raw results: `blend_rank_curve_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    xs = np.asarray(COUNTS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(xs, [kl("fineweb", f"top_{count}") for count in COUNTS], marker="o", label="FineWeb")
    axes[0].plot(xs, [kl("wikitext103", f"top_{count}") for count in COUNTS], marker="o", label="WikiText")
    axes[0].plot(xs, [kl("tinystories", f"top_{count}") for count in COUNTS], marker="o", label="TinyStories")
    axes[0].set_xlabel("Selected corrections"); axes[0].set_ylabel("KL"); axes[0].legend()
    for corpus in ("fineweb", "wikitext103", "tinystories"):
        axes[1].plot(xs, [causal(corpus, f"top_{count}", "gaussian_rms_0.10") for count in COUNTS], marker="o", label=corpus)
        axes[2].plot(xs, [causal(corpus, f"top_{count}", "natural_mix_0.25") for count in COUNTS], marker="o", label=corpus)
    axes[1].set_title("Gaussian causal fidelity"); axes[2].set_title("Natural causal fidelity")
    for axis in axes[1:]: axis.set_xlabel("Selected corrections"); axis.set_ylabel("Fidelity")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
