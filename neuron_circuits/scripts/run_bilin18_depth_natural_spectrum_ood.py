#!/usr/bin/env python3
"""OOD replication of empirical natural-write spectral regimes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_natural_spectrum import covariance_summary, eigensystem
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_depth_natural_spectrum_ood.json")
OUTPUT = TENSOR_ROOT / "depth_natural_spectrum_ood.json"
REPORT = TENSOR_ROOT / "DEPTH_NATURAL_SPECTRUM_OOD.md"
FIGURE = TENSOR_ROOT / "depth_natural_spectrum_ood.png"


def write_covariance(write: torch.Tensor) -> torch.Tensor:
    centered = write - write.mean(0, keepdim=True)
    return centered.T @ centered / len(centered)


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    results = {}
    for layer in config["layers"]:
        print("layer", layer, "FineWeb discovery", flush=True)
        _, discovery_write = collect(model, layer, config["fineweb_discovery"], config)
        discovery_covariance = write_covariance(discovery_write)
        discovery_values, discovery_vectors = eigensystem(discovery_covariance)
        discovery_top = discovery_vectors[:, : config["top_components"]]
        layer_rows = {}
        for corpus, corpus_config in config["corpora"].items():
            print(" layer", layer, corpus, flush=True)
            _, write = collect(model, layer, corpus_config, config)
            covariance = write_covariance(write)
            values, vectors = eigensystem(covariance)
            summary = covariance_summary(values, config["top_components"])
            captured = torch.trace(discovery_top.T @ covariance @ discovery_top)
            optimal = values[: config["top_components"]].sum()
            canonical = torch.linalg.svdvals(
                discovery_top.T @ vectors[:, : config["top_components"]]
            )
            layer_rows[corpus] = {
                **summary,
                "fineweb_top4_efficiency": float(captured / optimal),
                "fineweb_top4_canonical_cosines": [float(value) for value in canonical],
                "fineweb_top4_mean_canonical_cosine": float(canonical.mean()),
            }
        results[str(layer)] = layer_rows

    corpora = list(config["corpora"])
    hypotheses = {
        "O1_late_concentration_transfers": {
            "supported": all(results["17"][corpus]["top4_fraction"] >= 0.75 for corpus in corpora)
        },
        "O2_middle_distribution_transfers": {
            "supported": all(results["8"][corpus]["top4_fraction"] <= 0.15 for corpus in corpora)
        },
        "O3_early_intermediate_transfers": {
            "supported": all(0.20 <= results["0"][corpus]["top4_fraction"] <= 0.50 for corpus in corpora)
        },
        "O4_regime_order_is_stable": {
            "supported": all(
                results["17"][corpus]["effective_rank"]
                < results["0"][corpus]["effective_rank"]
                < results["8"][corpus]["effective_rank"]
                for corpus in corpora
            )
        },
        "O5_fineweb_subspace_transfers": {
            "supported": all(
                results[layer][corpus]["fineweb_top4_efficiency"] >= 0.80
                for layer in results for corpus in corpora
            )
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "results": results, "hypotheses": hypotheses}, indent=2))

    lines = [
        "# OOD replication of depth-dependent natural-write spectra",
        "",
        "Centered MLP-write covariance is measured independently on held-out FineWeb, WikiText-103, and TinyStories. The top-four output subspace is fit only on the FineWeb discovery panel, then frozen for transfer tests.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![OOD natural spectra](depth_natural_spectrum_ood.png)",
        "",
        "| Layer | Corpus | Top-four variance | Effective rank | FineWeb-subspace efficiency | Mean canonical cosine |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for layer, rows in results.items():
        for corpus, row in rows.items():
            lines.append(
                f"| {layer} | {corpus} | {row['top4_fraction']:.3f} | {row['effective_rank']:.1f} | "
                f"{row['fineweb_top4_efficiency']:.3f} | {row['fineweb_top4_mean_canonical_cosine']:.3f} |"
            )
    lines += [
        "",
        "These are output-variance regimes, not claims about semantic monosemanticity. A low-rank late write can combine multiple contexts, and a high-rank middle write can still contain individually clean conditional operations.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(corpora))
    width = 0.25
    for index, layer in enumerate(("0", "8", "17")):
        axes[0].bar(x + (index - 1) * width, [results[layer][c]["top4_fraction"] for c in corpora], width, label=f"L{layer}")
        axes[1].bar(x + (index - 1) * width, [results[layer][c]["effective_rank"] for c in corpora], width, label=f"L{layer}")
    for axis, title, ylabel in (
        (axes[0], "Top-four natural-write variance", "variance fraction"),
        (axes[1], "Natural-write effective rank", "participation rank"),
    ):
        axis.set_xticks(x, corpora, rotation=15)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
