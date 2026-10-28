#!/usr/bin/env python3
"""Evaluate trained downstream-weighted low-rank correction quotients."""

from __future__ import annotations

import json

import numpy as np
import torch

from evaluate_bilin18_causal_cp import load_cp
from evaluate_bilin18_decomposition_metrics import behavioral_evaluation, causal_evaluation, json_ready
from explore_bilin18_correction_subspace import ProjectedBlend
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_tensor_decomposition import ROOT
from train_bilin18_downstream_correction_quotient import CONFIG, LowRankCorrection
from validate_bilin18_value_tangent_blend import CONFIG as EVAL_CONFIG, load_rows


SUBSPACE = ROOT / "correction_subspace_results.json"
OUTPUT = ROOT / "downstream_correction_quotient_results.json"
REPORT = ROOT / "DOWNSTREAM_CORRECTION_QUOTIENT_REPORT.md"


def load_student(language, direct, rank, weight):
    payload = torch.load(ROOT / f"downstream_correction_rank{rank}.pt",
                         map_location="cpu", weights_only=True)
    # The constructor basis is overwritten immediately.
    dummy = torch.zeros(1152, rank, device="cuda")
    model = LowRankCorrection(language, direct, dummy, rank, weight).cuda().eval()
    with torch.no_grad():
        model.encoder.copy_(payload["encoder"].cuda())
        model.decoder.copy_(payload["decoder"].cuda())
    return model


def fraction(base, selected, full, higher):
    numerator = selected - base if higher else base - selected
    denominator = full - base if higher else base - full
    return float(numerator / denominator) if abs(denominator) > 1e-12 else float("nan")


def main():
    config = json.loads(CONFIG.read_text())
    eval_config = json.loads(EVAL_CONFIG.read_text())
    eval_config["bootstrap_resamples"] = 1000
    subspace = json.loads(SUBSPACE.read_text())
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    direct = load_cp(ROOT / "cp_r2048.pt", "cuda")
    candidates = {f"learned_{rank}": (load_student(
        language, direct, rank, config["blend_lambda"]), False)
        for rank in config["ranks"]}
    # Reconstruct frozen PCA comparators from saved bases by refitting is avoided:
    # their already-disjoint evaluation is copied below for comparison.
    model = load_full_model("cuda")
    corpora = load_rows(eval_config)
    result = {"config": config, "behavior": {}, "causal": {},
              "pca_comparison_source": str(SUBSPACE)}
    for corpus, rows in corpora.items():
        print("quotient behavior", corpus, flush=True)
        result["behavior"][corpus] = behavioral_evaluation(
            model, candidates, rows, eval_config, "cuda")
        print("quotient causal", corpus, flush=True)
        result["causal"][corpus] = causal_evaluation(
            model, candidates, rows[:eval_config["causal_rows"]],
            eval_config["interventions"], eval_config, "cuda")

    kl = lambda corpus, name: result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
    causal = lambda corpus, name, condition: result["causal"][corpus][name][condition]["fisher_effect_fidelity"]
    pca_kl = lambda corpus, name: subspace["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
    pca_causal = lambda corpus, name, condition: subspace["causal"][corpus][name][condition]["fisher_effect_fidelity"]
    base_name, full_name, selected = "hybrid_0", "hybrid_1152", "learned_256"
    base_kl = lambda corpus: pca_kl(corpus, base_name)
    full_kl = lambda corpus: pca_kl(corpus, full_name)
    base_causal = lambda corpus, condition: pca_causal(corpus, base_name, condition)
    full_causal = lambda corpus, condition: pca_causal(corpus, full_name, condition)
    behavior_fraction = fraction(base_kl("fineweb"), kl("fineweb", selected),
                                 full_kl("fineweb"), False)
    causal_fractions = {condition: fraction(
        base_causal("fineweb", condition), causal("fineweb", selected, condition),
        full_causal("fineweb", condition), True)
        for condition in eval_config["interventions"]}
    transfer = {corpus: (kl(corpus, selected) < base_kl(corpus)
                         and all(causal(corpus, selected, condition)
                                 > base_causal(corpus, condition)
                                 for condition in eval_config["interventions"]))
                for corpus in corpora}
    learned_mean_kl = float(np.mean([kl(corpus, selected) for corpus in corpora]))
    pca_mean_kl = float(np.mean([pca_kl(corpus, "hybrid_256") for corpus in corpora]))
    learned_mean_causal = float(np.mean([
        causal(corpus, selected, condition) for corpus in corpora
        for condition in eval_config["interventions"]]))
    pca_mean_causal = float(np.mean([
        pca_causal(corpus, "hybrid_256", condition) for corpus in corpora
        for condition in eval_config["interventions"]]))
    result["hypotheses"] = {
        "H1_behavior": {"supported": behavior_fraction >= .7,
                        "recovered_fraction": behavior_fraction},
        "H2_causal": {"supported": all(v >= .7 for v in causal_fractions.values())
                       and all(causal("fineweb", selected, condition)
                               >= base_causal("fineweb", condition)
                               for condition in eval_config["interventions"]),
                      "recovered_fractions": causal_fractions},
        "H3_transfer": {"supported": sum(transfer.values()) >= 2,
                        "corpus_wins": transfer},
        "H4_beats_pca": {"supported": learned_mean_kl < pca_mean_kl
                         and learned_mean_causal > pca_mean_causal,
                         "learned_mean_kl": learned_mean_kl,
                         "pca_mean_kl": pca_mean_kl,
                         "learned_mean_causal": learned_mean_causal,
                         "pca_mean_causal": pca_mean_causal},
    }
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))
    lines = ["# Downstream-weighted correction quotient", "", "## Hypotheses", ""]
    for name, row in result["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Model | FineWeb KL | FW Gaussian | FW natural | Wiki KL | Tiny KL |",
              "|---|---:|---:|---:|---:|---:|"]
    for rank in config["ranks"]:
        name = f"learned_{rank}"
        lines.append(f"| Learned rank {rank} | {kl('fineweb', name):.4f} "
                     f"| {causal('fineweb', name, 'gaussian_rms_0.10'):.3f} "
                     f"| {causal('fineweb', name, 'natural_mix_0.25'):.3f} "
                     f"| {kl('wikitext103', name):.4f} | {kl('tinystories', name):.4f} |")
    lines += ["", "The learned object is a rank-constrained linear map applied to the exact direct-minus-language correction. It is basis-invariant with respect to CP term order but still evaluates both parent CP models, so it is a causal-description test rather than a runtime compression.", "",
              "Raw results: `downstream_correction_quotient_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
