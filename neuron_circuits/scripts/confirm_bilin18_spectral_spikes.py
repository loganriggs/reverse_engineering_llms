#!/usr/bin/env python3
"""Independent-feature-seed and OOD confirmation for spectral spike circuits."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_spike_pilot import (
    SPARSE,
    TENSOR_ROOT,
    load_feature_directions,
    safe_name,
)
from redteam_bilin18_spectral_spikes import causal_curves, select_transfer_sites


CONFIG = Path("configs/bilin18_spectral_spike_confirmation.json")
PILOT = TENSOR_ROOT / "spectral_spike_pilot.json"
ARRAYS = TENSOR_ROOT / "spectral_spike_pilot.npz"
OUTPUT = TENSOR_ROOT / "spectral_spike_confirmation.json"
REPORT = TENSOR_ROOT / "SPECTRAL_SPIKE_CONFIRMATION.md"
FIGURE = TENSOR_ROOT / "spectral_spike_confirmation.png"


def rank1_stability(arrays, categories):
    matrix = np.zeros((len(categories), len(categories)))
    for i, first in enumerate(categories):
        left = arrays[f"{safe_name(f'semantic/seed101/{first}_whitened')}_eigenvectors"][:, 0]
        for j, second in enumerate(categories):
            right = arrays[f"{safe_name(f'semantic/seed102/{second}_whitened')}_eigenvectors"][:, 0]
            matrix[i, j] = float(np.dot(left, right) ** 2)
    observed = float(np.trace(matrix) / len(categories))
    null = [float(np.mean([matrix[i, permutation[i]] for i in range(len(categories))]))
            for permutation in itertools.permutations(range(len(categories)))]
    return {
        "matrix": matrix.tolist(),
        "observed_diagonal_mean": observed,
        "permutation_p_value": float(np.mean(np.asarray(null) >= observed)),
        "diagonal_top_matches": int(sum(np.argmax(matrix[i]) == i for i in range(len(categories)))),
        "permutations": len(null),
    }


def median_metric(causal, names, key, field):
    values = [causal[name]["cleanings"]["whitened"]["summary"][key][field]
              for name in names if name in causal]
    return float(np.median(values)) if values else None


def main():
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(PILOT.read_text())
    arrays = np.load(ARRAYS)
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    down = state["transformer.h.1.mlp.Down.weight"].float().cuda()
    del state
    archive = np.load(SPARSE / "natural_large.npz")
    h_std = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32)).std(0).clamp_min(1e-4)
    del archive
    directions, metadata = load_feature_directions(pilot["config"], h_std, down)
    seed = config["feature_seed"]
    direction_rows = {name: {"metadata": metadata[name], "coefficients": value}
                      for name, value in directions.items() if metadata[name]["seed"] == seed}
    inverse_square = torch.from_numpy(arrays["covariance_inverse_square"]).cuda()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")

    corpora = {}
    for corpus, specification in config["corpora"].items():
        path = DEFAULT_CORPUS if specification["path"] == "default_fineweb" else Path(specification["path"])
        all_rows = np.load(path, mmap_mode="r")
        rows = all_rows[specification["start"]:specification["start"] + specification["rows"]]
        sites = select_transfer_sites(model, rows, direction_rows, pilot, arrays,
                                      inverse_square,
                                      {"causal_transfer": {"positions": config["positions"]},
                                       "causal_sites_per_direction": config["causal_sites_per_direction"]},
                                      tokenizer)
        eligible = []
        excluded = {}
        for category in config["strong_categories"]:
            name = f"semantic/seed{seed}/{category}"
            if len(sites[name]) >= specification["minimum_positive_sites"]:
                eligible.append(category)
            else:
                excluded[category] = len(sites[name])
        selected_names = ({f"semantic/seed{seed}/{category}" for category in eligible}
                          | {f"control/seed{seed}/{category}" for category in eligible})
        chosen_rows = {name: direction_rows[name] for name in selected_names}
        chosen_sites = {name: sites[name] for name in selected_names}
        causal = causal_curves(model, rows, chosen_sites, chosen_rows, pilot, arrays,
                               inverse_square, {"ranks": config["ranks"]}, tokenizer)
        corpora[corpus] = {"eligible_categories": eligible, "excluded": excluded,
                           "causal": causal}

    summaries = {}
    for corpus, row in corpora.items():
        semantics = [f"semantic/seed{seed}/{category}" for category in row["eligible_categories"]]
        controls = [f"control/seed{seed}/{category}" for category in row["eligible_categories"]]
        spike = median_metric(row["causal"], semantics, "1", "median_spike_fidelity")
        bulk = median_metric(row["causal"], semantics, "1", "median_bulk_fidelity")
        control = median_metric(row["causal"], controls, "1", "median_spike_fidelity")
        significant = median_metric(row["causal"], semantics, "significant", "median_spike_fidelity")
        control_significant = median_metric(row["causal"], controls, "significant", "median_spike_fidelity")
        wins = sum(row["causal"][name]["cleanings"]["whitened"]["summary"]["1"]["median_spike_fidelity"]
                   > row["causal"][name]["cleanings"]["whitened"]["summary"]["1"]["median_bulk_fidelity"]
                   for name in semantics)
        summaries[corpus] = {
            "semantic_top1_median": spike, "semantic_bulk_median": bulk,
            "control_top1_median": control,
            "semantic_minus_control": spike - control,
            "semantic_significant_median": significant,
            "control_significant_median": control_significant,
            "significant_advantage": significant - control_significant,
            "spike_beats_bulk": wins, "eligible_count": len(semantics),
        }

    stability = rank1_stability(arrays, config["strong_categories"])
    fineweb = summaries["fineweb"]
    ood_supported = []
    for corpus in ("wikitext103", "tinystories"):
        row = summaries[corpus]
        ood_supported.append(row["semantic_top1_median"] >= .20
                             and row["spike_beats_bulk"] >= int(np.ceil(.75 * row["eligible_count"])))
    control_wins = sum(row["semantic_minus_control"] >= .10 for row in summaries.values())
    significant_wins = sum(row["significant_advantage"] >= .10 for row in summaries.values())
    hypotheses = {
        "C1_independent_feature_seed": {"supported": bool(fineweb["semantic_top1_median"] >= .25
            and fineweb["spike_beats_bulk"] >= 4), **fineweb},
        "C2_ood_transfer": {"supported": bool(all(ood_supported)),
                            "per_corpus_supported": dict(zip(("wikitext103", "tinystories"), ood_supported))},
        "C3_semantic_control_advantage": {"supported": bool(control_wins >= 2),
                                           "corpus_wins": control_wins},
        "C4_null_significance": {"supported": bool(significant_wins >= 2),
                                  "corpus_wins": significant_wins},
        "C5_rank1_category_stability": {"supported": bool(stability["permutation_p_value"] <= .05
            and stability["diagonal_top_matches"] >= 4), **stability},
    }
    result = {"config": config, "summaries": summaries, "corpora": corpora,
              "rank1_stability": stability, "hypotheses": hypotheses,
              "selection_rule": "new feature seed; fresh FineWeb and OOD panels; semantic true positives ranked only by full centered form"}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))

    lines = ["# Spectral-spike independent confirmation", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Confirmation](spectral_spike_confirmation.png)", "",
              "| Corpus | Eligible semantic categories | Top-1 spike | Bulk | Matched control | Semantic−control | Significant-only |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in summaries.items():
        lines.append(f"| {corpus} | {row['eligible_count']} | {row['semantic_top1_median']:.3f} | "
                     f"{row['semantic_bulk_median']:.3f} | {row['control_top1_median']:.3f} | "
                     f"{row['semantic_minus_control']:.3f} | {row['semantic_significant_median']:.3f} |")
    lines += ["", "This run changes the learned SAE feature basis from seed 101 to seed 102, uses FineWeb rows unseen by both earlier causal analyses, and repeats the causal test on WikiText-103 and TinyStories. Categories with fewer than eight true-positive sites are reported and excluded symmetrically with their controls.", "", "Raw results: `spectral_spike_confirmation.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = list(summaries)
    x = np.arange(len(labels))
    width = .25
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(x - width, [summaries[k]["semantic_top1_median"] for k in labels], width, label="semantic spike")
    axes[0].bar(x, [summaries[k]["semantic_bulk_median"] for k in labels], width, label="complementary bulk")
    axes[0].bar(x + width, [summaries[k]["control_top1_median"] for k in labels], width, label="matched control")
    axes[0].set_xticks(x, labels); axes[0].set_ylabel("median causal fidelity"); axes[0].legend(fontsize=8)
    matrix = np.asarray(stability["matrix"])
    image = axes[1].imshow(matrix, vmin=0, vmax=max(.5, matrix.max()), cmap="viridis")
    axes[1].set_title("Cross-seed leading-mode overlap")
    axes[1].set_xticks(range(len(config["strong_categories"])), config["strong_categories"], rotation=45, ha="right")
    axes[1].set_yticks(range(len(config["strong_categories"])), config["strong_categories"])
    fig.colorbar(image, ax=axes[1], fraction=.046)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
