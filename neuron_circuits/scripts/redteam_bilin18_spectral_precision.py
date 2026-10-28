#!/usr/bin/env python3
"""Float32 precision red-team for spectral intervention scale curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_atlas import ARRAYS as ATLAS_ARRAYS, OUTPUT as ATLAS_OUTPUT
from run_bilin18_spectral_spike_pilot import SPARSE, TENSOR_ROOT, load_feature_directions
from redteam_bilin18_spectral_downstream_ranking import (
    OUTPUT as BFLOAT_OUTPUT,
    aggregate,
    validation_sweep,
)


CONFIG = Path("configs/bilin18_spectral_precision_redteam.json")
OUTPUT = TENSOR_ROOT / "spectral_precision_redteam.json"
REPORT = TENSOR_ROOT / "SPECTRAL_PRECISION_REDTEAM.md"
FIGURE = TENSOR_ROOT / "spectral_precision_redteam.png"


def compact_sites(previous):
    output = {}
    for name, row in previous["validation"].items():
        output[name] = [{key: site[key] for key in ("row", "position", "score", "token_id")}
                        for site in row["sites"]]
    return output


def main():
    config = json.loads(CONFIG.read_text())
    previous = json.loads(BFLOAT_OUTPUT.read_text())
    atlas = json.loads(ATLAS_OUTPUT.read_text())
    arrays = np.load(ATLAS_ARRAYS)
    inverse_square = torch.from_numpy(arrays["covariance_inverse_square"]).cuda()
    all_categories = atlas["config"]["reference_categories"] + atlas["config"]["prospective_categories"]
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    down = state["transformer.h.1.mlp.Down.weight"].float().cuda()
    del state
    archive = np.load(SPARSE / "natural_large.npz")
    h_std = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32)).std(0).clamp_min(1e-4)
    del archive
    directions, metadata = load_feature_directions(
        {"strong_categories": all_categories, "sae_seeds": atlas["config"]["feature_seeds"],
         "seed": atlas["config"]["seed"]}, h_std, down)
    names = list(previous["validation"])
    direction_rows = {name: {"metadata": metadata[name], "coefficients": directions[name]}
                      for name in names}
    discovery = {name: {"orders": previous["discovery"][name]["orders"]} for name in names}
    sites = compact_sites(previous)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    panel = previous["config"]["validation"]
    all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    rows = all_rows[panel["start"]:panel["start"] + panel["rows"]]
    run_config = {"candidate_modes": config["candidate_modes"], "ranks": config["ranks"],
                  "scales": config["scales"]}
    validation = validation_sweep(model, rows, sites, direction_rows, arrays,
                                  inverse_square, discovery, run_config, tokenizer)
    rankings = ["magnitude", "value_greedy", "fisher_greedy"]
    combined = aggregate(validation, rankings, config["ranks"], config["scales"])

    full_differences = {}
    for ranking in rankings:
        fp32 = combined[ranking]["4"]["1.0"]["median_fidelity"]
        bf16 = previous["aggregate"][ranking]["4"]["1.0"]["median_fidelity"]
        full_differences[ranking] = {"float32": fp32, "bfloat16": bf16,
                                     "absolute_difference": abs(fp32 - bf16)}

    local_scales = [0.01, 0.02, 0.05, 0.1]
    coefficients_of_variation = []
    for row in validation.values():
        for site in row["sites"]:
            energies = np.asarray([
                site["rankings"]["magnitude"]["1"]["scales"][str(scale)]["target_effect_energy"]
                / (scale * scale) for scale in local_scales])
            coefficients_of_variation.append(float(energies.std() / max(energies.mean(), 1e-30)))
    local_cv = float(np.median(coefficients_of_variation))

    category_local_differences = []
    for row in validation.values():
        summary = row["summary"]["magnitude"]["1"]
        category_local_differences.append(abs(
            summary["0.01"]["median_fidelity"] - summary["0.02"]["median_fidelity"]))
    local_fidelity_difference = float(np.median(category_local_differences))

    reordered = {}
    for ranking in ("value_greedy", "fisher_greedy"):
        row = combined[ranking]["4"]["1.0"]
        wins = sum(a > b for a, b in zip(row["category_fidelities"],
                                         row["category_complement_fidelities"]))
        reordered[ranking] = {"median_fidelity": row["median_fidelity"],
                              "category_wins": wins}
    hypotheses = {
        "P1_full_scale_robustness": {"supported": max(
            row["absolute_difference"] for row in full_differences.values()) <= .10,
            "rank4": full_differences},
        "P2_local_quadratic_scaling": {"supported": local_cv <= .20,
                                       "median_coefficient_of_variation": local_cv},
        "P3_local_fidelity_limit": {"supported": local_fidelity_difference <= .10,
                                    "median_absolute_difference": local_fidelity_difference,
                                    "category_differences": category_local_differences},
        "P4_reordered_rank4_survives": {"supported": all(
            row["median_fidelity"] >= .30 and row["category_wins"] >= 2
            for row in reordered.values()), **reordered},
    }
    result = {"config": config, "precision": "float32 weights and computation",
              "validation": validation, "aggregate": combined,
              "local_energy_cv": coefficients_of_variation,
              "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))

    lines = ["# Float32 spectral-intervention precision red-team", "",
             "This reruns the frozen validation sites and frozen mode orders after converting the identical checkpoint weights to float32. It tests whether the bfloat16 scale sweep measured model curvature or numerical quantization.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Precision red-team](spectral_precision_redteam.png)", "",
              "| Ranking | BF16 rank-4 scale-1 | FP32 rank-4 scale-1 | Absolute difference |",
              "|---|---:|---:|---:|"]
    for ranking, row in full_differences.items():
        lines.append(f"| {ranking} | {row['bfloat16']:.3f} | {row['float32']:.3f} | {row['absolute_difference']:.3f} |")
    lines += ["", f"Median local energy/scaling coefficient of variation: **{local_cv:.3f}**.",
              "", "The float32 model is a numerical extension of the stored checkpoint weights, not the exact deployed bfloat16 computation. Full-scale agreement establishes whether conclusions transfer between those two use cases."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ranking in rankings:
        axes[0].plot(config["scales"], [combined[ranking]["4"][str(scale)]["median_fidelity"]
                                       for scale in config["scales"]], marker="o", label=ranking)
    axes[0].set_xscale("log"); axes[0].set_xlabel("intervention scale")
    axes[0].set_ylabel("median causal fidelity"); axes[0].set_title("Float32 rank-4 sweep")
    axes[0].legend(fontsize=8)
    x = np.arange(len(rankings))
    axes[1].bar(x - .18, [full_differences[r]["bfloat16"] for r in rankings], .36, label="bfloat16")
    axes[1].bar(x + .18, [full_differences[r]["float32"] for r in rankings], .36, label="float32")
    axes[1].set_xticks(x, rankings, rotation=20, ha="right")
    axes[1].set_ylabel("rank-4 scale-1 fidelity"); axes[1].set_title("Full-scale precision robustness")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
