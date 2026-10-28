#!/usr/bin/env python3
"""Evaluate intervention-specialized rank-256 correction quotients."""

from __future__ import annotations

import json

import numpy as np
import torch

from evaluate_bilin18_causal_cp import load_cp
from evaluate_bilin18_decomposition_metrics import behavioral_evaluation, causal_evaluation, json_ready
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_tensor_decomposition import ROOT
from train_bilin18_downstream_correction_quotient import LowRankCorrection
from train_bilin18_correction_quotient_specialists import CONFIG
from validate_bilin18_value_tangent_blend import CONFIG as EVAL_CONFIG, load_rows


BASE_RESULTS = ROOT / "correction_subspace_results.json"
UNIVERSAL_RESULTS = ROOT / "downstream_correction_quotient_results.json"
OUTPUT = ROOT / "correction_quotient_specialist_results.json"
REPORT = ROOT / "CORRECTION_QUOTIENT_SPECIALIST_REPORT.md"


def load_variant(language, direct, filename):
    payload = torch.load(ROOT / filename, map_location="cpu", weights_only=True)
    dummy = torch.zeros(1152, 256, device="cuda")
    model = LowRankCorrection(language, direct, dummy, 256, .2).cuda().eval()
    with torch.no_grad():
        model.encoder.copy_(payload["encoder"].cuda()); model.decoder.copy_(payload["decoder"].cuda())
    return model


def fraction(base, value, full):
    return float((value - base) / (full - base)) if abs(full - base) > 1e-12 else float("nan")


def main():
    config = json.loads(CONFIG.read_text())
    eval_config = json.loads(EVAL_CONFIG.read_text()); eval_config["bootstrap_resamples"] = 1000
    baseline = json.loads(BASE_RESULTS.read_text())
    universal = json.loads(UNIVERSAL_RESULTS.read_text())
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    direct = load_cp(ROOT / "cp_r2048.pt", "cuda")
    candidates = {
        "gaussian": (load_variant(language, direct, "correction_quotient_gaussian.pt"), False),
        "natural": (load_variant(language, direct, "correction_quotient_natural.pt"), False),
    }
    model = load_full_model("cuda"); corpora = load_rows(eval_config)
    result = {"config": config, "behavior": {}, "causal": {}}
    for corpus, rows in corpora.items():
        print("specialist behavior", corpus, flush=True)
        result["behavior"][corpus] = behavioral_evaluation(model, candidates, rows, eval_config, "cuda")
        print("specialist causal", corpus, flush=True)
        result["causal"][corpus] = causal_evaluation(
            model, candidates, rows[:eval_config["causal_rows"]],
            eval_config["interventions"], eval_config, "cuda")
    kl = lambda corpus, name: result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
    causal = lambda corpus, name, condition: result["causal"][corpus][name][condition]["fisher_effect_fidelity"]
    bcausal = lambda name, condition: baseline["causal"]["fineweb"][name][condition]["fisher_effect_fidelity"]
    ucausal = lambda condition: universal["causal"]["fineweb"]["learned_256"][condition]["fisher_effect_fidelity"]
    ukl = universal["behavior"]["fineweb"]["models"]["learned_256"]["mean_kl"]["mean"]
    gcond, ncond = "gaussian_rms_0.10", "natural_mix_0.25"
    gfraction = fraction(bcausal("hybrid_0", gcond), causal("fineweb", "gaussian", gcond),
                         bcausal("hybrid_1152", gcond))
    nfraction = fraction(bcausal("hybrid_0", ncond), causal("fineweb", "natural", ncond),
                         bcausal("hybrid_1152", ncond))
    result["hypotheses"] = {
        "H1_gaussian_specialist": {"supported": causal("fineweb", "gaussian", gcond) > ucausal(gcond)
                                    and gfraction >= .7, "recovered_fraction": gfraction},
        "H2_natural_specialist": {"supported": causal("fineweb", "natural", ncond) > ucausal(ncond)
                                   and nfraction >= .7, "recovered_fraction": nfraction},
        "H3_cross_compatible": {"supported": causal("fineweb", "gaussian", ncond) >= bcausal("hybrid_0", ncond)
                                and causal("fineweb", "natural", gcond) >= bcausal("hybrid_0", gcond)
                                and kl("fineweb", "gaussian") <= ukl + .0005
                                and kl("fineweb", "natural") <= ukl + .0005},
    }
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))
    lines = ["# Intervention-specialized correction quotients", "", "## Hypotheses", ""]
    for name, row in result["hypotheses"].items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Specialist | FineWeb KL | FW Gaussian | FW natural | Wiki Gaussian | Tiny Gaussian |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in candidates:
        lines.append(f"| {name} | {kl('fineweb', name):.4f} | {causal('fineweb', name, gcond):.3f} "
                     f"| {causal('fineweb', name, ncond):.3f} | {causal('wikitext103', name, gcond):.3f} "
                     f"| {causal('tinystories', name, gcond):.3f} |")
    lines += ["", "Specialization is a red-team of the universal-subspace failure. Success only on the trained intervention would support use-case-specific quotients; failure even in-distribution supports a broad correction at rank 256.", "",
              "Raw results: `correction_quotient_specialist_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__": main()
