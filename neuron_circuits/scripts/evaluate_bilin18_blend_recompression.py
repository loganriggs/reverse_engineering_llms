#!/usr/bin/env python3
"""Evaluate whether CP-2048 recompression preserves the validated blend."""

from __future__ import annotations

import json

import numpy as np
import torch

from analyze_bilin18_tangent_ontology import Agreement
from evaluate_bilin18_causal_cp import load_cp
from evaluate_bilin18_decomposition_metrics import behavioral_evaluation, causal_evaluation, json_ready
from explore_bilin18_value_tangent_blend import BilinearBlend
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_tensor_decomposition import ROOT
from validate_bilin18_value_tangent_blend import CONFIG, load_rows


DATA = "artifacts/bilin18_sparse_codes/layer1_inputs_large.npz"
OUTPUT = ROOT / "blend_recompression_results.json"
REPORT = ROOT / "BLEND_RECOMPRESSION_REPORT.md"
DISPLAY = {
    "language_baseline": "Language CP-2048",
    "exact_blend": "Exact rank≤4096 blend",
    "recompress_value": "Value-recompressed CP-2048",
    "recompress_tangent": "Tangent-recompressed CP-2048",
    "recompress_downstream": "Downstream-recompressed CP-2048",
}


@torch.inference_mode()
def local_agreement(target, candidates, archive, examples=8192):
    generator = torch.Generator(device="cuda").manual_seed(20261024)
    result = {}
    for split in ("fineweb_test", "wikitext103", "tinystories"):
        accumulators = {name: {kind: Agreement() for kind in
                              ("value", "gaussian_tangent", "natural_tangent", "isotropic_bilinear")}
                        for name in candidates}
        x_cpu = torch.from_numpy(archive[f"{split}__input1"][:examples].astype(np.float32))
        for start in range(0, len(x_cpu), 256):
            x = x_cpu[start:start + 256].cuda()
            gaussian = torch.randn(x.shape, generator=generator, device="cuda")
            gaussian = gaussian / gaussian.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)
            natural = x.roll(1, 0) - x
            y = torch.randn(x.shape, generator=generator, device="cuda")
            targets = {
                "value": target(x),
                "gaussian_tangent": 2 * target.bilinear(x, gaussian),
                "natural_tangent": 2 * target.bilinear(x, natural),
                "isotropic_bilinear": target.bilinear(gaussian, y),
            }
            for name, candidate in candidates.items():
                predictions = {
                    "value": candidate(x),
                    "gaussian_tangent": 2 * candidate.bilinear(x, gaussian),
                    "natural_tangent": 2 * candidate.bilinear(x, natural),
                    "isotropic_bilinear": candidate.bilinear(gaussian, y),
                }
                for kind in targets:
                    accumulators[name][kind].add(targets[kind], predictions[kind])
        result[split] = {name: {kind: metric.result() for kind, metric in rows.items()}
                         for name, rows in accumulators.items()}
    return result


def main():
    config = json.loads(CONFIG.read_text())
    device = "cuda"
    language = load_cp(ROOT / "cp2048_causal_r2.pt", device)
    direct = load_cp(ROOT / "cp_r2048.pt", device)
    exact = BilinearBlend(language, direct, 0.2).cuda().eval()
    candidates = {
        "language_baseline": language,
        "exact_blend": exact,
        "recompress_value": load_cp(ROOT / "blend_recompress_value.pt", device),
        "recompress_tangent": load_cp(ROOT / "blend_recompress_tangent.pt", device),
        "recompress_downstream": load_cp(ROOT / "blend_recompress_downstream.pt", device),
    }
    wrapped = {name: (candidate, False) for name, candidate in candidates.items()}
    model = load_full_model(device)
    corpora = load_rows(config)
    result = {"status": "exploratory_recompression_evaluation", "behavior": {}, "causal": {}}
    for corpus, rows in corpora.items():
        print("recompression behavior", corpus, flush=True)
        result["behavior"][corpus] = behavioral_evaluation(model, wrapped, rows, config, device)
        print("recompression causal", corpus, flush=True)
        result["causal"][corpus] = causal_evaluation(
            model, wrapped, rows[:config["causal_rows"]], config["interventions"], config, device)
    result["local_agreement_to_exact_blend"] = local_agreement(
        exact, {name: candidate for name, candidate in candidates.items() if name != "exact_blend"},
        np.load(DATA))
    OUTPUT.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False))

    kl = lambda corpus, name: result["behavior"][corpus]["models"][name]["mean_kl"]["mean"]
    causal = lambda corpus, name, condition: result["causal"][corpus][name][condition]["fisher_effect_fidelity"]
    lines = ["# CP-2048 recompression of the value–tangent blend", "",
             "## Full-model behavior", "",
             "| Model | FineWeb KL | WikiText KL | TinyStories KL |", "|---|---:|---:|---:|"]
    for name in candidates:
        lines.append(f"| {DISPLAY[name]} | {kl('fineweb', name):.4f} | {kl('wikitext103', name):.4f} | {kl('tinystories', name):.4f} |")
    lines += ["", "## Full-model causal fidelity", "",
              "| Model | FW Gaussian | FW natural | Wiki Gaussian | Wiki natural | Tiny Gaussian | Tiny natural |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in candidates:
        values = []
        for corpus in ("fineweb", "wikitext103", "tinystories"):
            values += [causal(corpus, name, "gaussian_rms_0.10"),
                       causal(corpus, name, "natural_mix_0.25")]
        lines.append(f"| {DISPLAY[name]} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "## Layer-local agreement with the exact blend", "",
              "| Model | Value | Gaussian tangent | Natural tangent | Isotropic bilinear |",
              "|---|---:|---:|---:|---:|"]
    local = result["local_agreement_to_exact_blend"]["fineweb_test"]
    for name in ("language_baseline", "recompress_value", "recompress_tangent",
                 "recompress_downstream"):
        lines.append(f"| {DISPLAY[name]} | " + " | ".join(
            f"{local[name][kind]['aggregate_fidelity']:.4f}" for kind in
            ("value", "gaussian_tangent", "natural_tangent", "isotropic_bilinear")) + " |")
    lines += ["", "## Interpretation", "",
              "If recompressed models track the exact blend, the apparent rank-4096 construction reflects redundant/aligned terms rather than two independent rank-2048 mechanisms. If value recompression preserves behavior but loses tangent fidelity, value and causal descriptions require different fitting objectives.", "",
              "This remains an inherited-basis compression result; it does not show independent recovery of the factors.", "",
              "Raw results: `blend_recompression_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", OUTPUT, REPORT, flush=True)


if __name__ == "__main__":
    main()
