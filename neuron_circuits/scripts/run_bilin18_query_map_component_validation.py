#!/usr/bin/env python3
"""Separate query-map spectral strength, natural variance, and score fidelity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_query_conditioned_interpreter import (
    ARRAYS as MAP_ARRAYS,
    CONFIG as INTERPRETER_CONFIG,
    collect_states_and_writes,
    score_metrics,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_query_map_component_validation.json")
OUTPUT = TENSOR_ROOT / "query_map_component_validation.json"
REPORT = TENSOR_ROOT / "QUERY_MAP_COMPONENT_VALIDATION.md"
FIGURE = TENSOR_ROOT / "query_map_component_validation.png"


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    a, b = ranks(first), ranks(second)
    return float(np.corrcoef(a, b)[0, 1])


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    saved = np.load(MAP_ARRAYS)
    count = config["components"]
    singular = saved["singular_values"][:count]
    output_components = torch.from_numpy(saved["output_components"][:, :count]).cuda()
    matrix_components = torch.from_numpy(saved["matrix_components"][:count]).cuda()
    axes = torch.from_numpy(np.load(ATLAS_ARRAYS)["layer17_axes"]).cuda()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()

    results = {}
    for corpus_name, corpus_config in interpreter_config["evaluation"].items():
        states, writes, _ = collect_states_and_writes(
            model, corpus_config, interpreter_config["positions"], interpreter_config["batch_size"]
        )
        z = torch.from_numpy(states).cuda() @ axes
        target = torch.from_numpy(writes).cuda() @ output_components
        prediction = torch.einsum(
            "ni,kij,nj,k->nk",
            z,
            matrix_components,
            z,
            torch.from_numpy(singular).cuda(),
        )
        target_np = target.cpu().numpy()
        prediction_np = prediction.cpu().numpy()
        correlation, r2 = score_metrics(target_np, prediction_np)
        variance = np.var(target_np, axis=0)
        groups = {}
        for name, (start, end) in config["groups"].items():
            groups[name] = {
                "median_correlation": float(np.median(correlation[start:end])),
                "median_centered_r2": float(np.median(r2[start:end])),
                "median_target_variance": float(np.median(variance[start:end])),
            }
        results[corpus_name] = {
            "groups": groups,
            "spearman_singular_variance": spearman(singular, variance),
            "spearman_singular_correlation": spearman(singular, correlation),
            "components": [
                {
                    "rank": index + 1,
                    "singular_value": float(singular[index]),
                    "target_variance": float(variance[index]),
                    "correlation": float(correlation[index]),
                    "centered_r2": float(r2[index]),
                }
                for index in range(count)
            ],
        }

    hypotheses = {
        "C1_spectral_strength_tracks_variance": {
            "supported": all(
                row["groups"]["top32"]["median_target_variance"]
                >= 2 * row["groups"]["middle32"]["median_target_variance"]
                for row in results.values()
            ),
            "ratios": {
                corpus: row["groups"]["top32"]["median_target_variance"]
                / row["groups"]["middle32"]["median_target_variance"]
                for corpus, row in results.items()
            },
        },
        "C2_top_components_are_faithful": {
            "supported": all(row["groups"]["top32"]["median_correlation"] >= 0.60 for row in results.values()),
            "per_corpus": {corpus: row["groups"]["top32"]["median_correlation"] for corpus, row in results.items()},
        },
        "C3_global_strength_variance_relation": {
            "supported": all(row["spearman_singular_variance"] >= 0.50 for row in results.values()),
            "per_corpus": {corpus: row["spearman_singular_variance"] for corpus, row in results.items()},
        },
        "C4_global_strength_fidelity_relation": {
            "supported": all(row["spearman_singular_correlation"] >= 0.30 for row in results.values()),
            "per_corpus": {corpus: row["spearman_singular_correlation"] for corpus, row in results.items()},
        },
        "C5_top4_cleaner_than_top32": {
            "supported": all(
                row["groups"]["top4"]["median_correlation"] >= row["groups"]["top32"]["median_correlation"]
                for row in results.values()
            ),
            "per_corpus": {
                corpus: {
                    "top4": row["groups"]["top4"]["median_correlation"],
                    "top32": row["groups"]["top32"]["median_correlation"],
                }
                for corpus, row in results.items()
            },
        },
    }
    output = {"config": config, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Natural-state validation of query-map spectral components",
        "",
        "For each of the leading 256 query-map singular components, the full MLP scalar score is compared with its 32D interaction-matrix prediction on three corpora. Spectral strength, target variance, and predictive fidelity are reported separately.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Component validation](query_map_component_validation.png)",
        "",
        "| Corpus | Top-4 corr | Top-32 corr | Middle-32 corr | Top/middle variance | Strength↔variance | Strength↔fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        groups = row["groups"]
        lines.append(
            f"| {corpus} | {groups['top4']['median_correlation']:.3f} | "
            f"{groups['top32']['median_correlation']:.3f} | {groups['middle32']['median_correlation']:.3f} | "
            f"{groups['top32']['median_target_variance'] / groups['middle32']['median_target_variance']:.2f} | "
            f"{row['spearman_singular_variance']:.3f} | {row['spearman_singular_correlation']:.3f} |"
        )
    lines += [
        "",
        "A strong weight singular component is a candidate operation. It is only a validated circuit direction if its natural score is also predicted by the restricted state and its proposed semantic/causal role survives independent tests.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    for corpus, row in results.items():
        components = row["components"]
        plots[0].scatter(
            [item["singular_value"] for item in components],
            [item["target_variance"] for item in components],
            s=10,
            alpha=0.5,
            label=corpus,
        )
        plots[1].plot(
            [item["rank"] for item in components],
            [item["correlation"] for item in components],
            alpha=0.75,
            label=corpus,
        )
    plots[0].set_xscale("log")
    plots[0].set_yscale("log")
    plots[0].set_xlabel("query-map singular value")
    plots[0].set_ylabel("natural score variance")
    plots[0].legend()
    plots[1].set_xlabel("query-map component rank")
    plots[1].set_ylabel("natural-score correlation")
    plots[1].set_ylim(-0.2, 1.02)
    plots[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
