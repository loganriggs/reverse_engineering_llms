#!/usr/bin/env python3
"""Scale red-team for the layer-8 direct/attention path decomposition."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_layer8_path_decomposition import evaluate_corpus
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_path_scale_redteam.json")
OUTPUT = TENSOR_ROOT / "layer8_path_scale_redteam.json"
REPORT = TENSOR_ROOT / "LAYER8_PATH_SCALE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer8_path_scale_redteam.png"


def at(results, scale, corpus, route, path, metric):
    return results[str(scale)][corpus]["summary"]["within_route"][route][path][metric]


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    discovery_config = {**config, "positions": config["discovery_positions"]}
    discovery_x, discovery_write = collect(
        model, config["layer"], config["discovery"], discovery_config,
    )
    _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = state_vh.T[:, :config["dimension"]]
    centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
    _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
    directions = write_vh.T[:, :config["queries"]]
    block = model.blocks[config["layer"]]
    left = block.mlp.Left.weight.float() @ basis
    right = block.mlp.Right.weight.float() @ basis
    coefficients = block.mlp.Down.weight.float().T @ directions
    matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
    matrices = 0.5 * (matrices + matrices.transpose(1, 2))
    values, vectors = ordered_eigensystem(matrices)

    results = {}
    for scale in config["edit_norm_fractions"]:
        scale_config = {**config, "edit_norm_fraction": scale}
        scale_rows = {}
        for corpus, corpus_config in config["evaluation"].items():
            print("scale", scale, corpus, flush=True)
            scale_rows[corpus] = evaluate_corpus(
                model, scale_config, corpus_config, basis, values, vectors,
            )
        results[str(scale)] = scale_rows

    corpora = list(config["evaluation"])
    routes = config["routes"]
    hypotheses = {
        "S1_local_additivity": {
            "supported": all(
                at(results, scale, corpus, route, "additive", "median_downstream_fisher_fidelity_to_full")
                >= (0.98 if scale == 0.01 else 0.95)
                for scale in (0.01, 0.05) for corpus in corpora for route in routes
            ),
        },
        "S2_additivity_survives_ten_percent": {
            "supported": all(
                at(results, 0.1, corpus, route, "additive", "median_downstream_fisher_fidelity_to_full") >= 0.80
                for corpus in corpora for route in routes
            ),
        },
        "S3_direct_path_survives_ten_percent": {
            "supported": all(
                at(results, scale, corpus, route, "direct", "median_downstream_fisher_fidelity_to_full") >= 0.80
                for scale in (0.01, 0.05, 0.1) for corpus in corpora for route in routes
            ),
        },
        "S4_twenty_percent_exposes_nonlinearity": {
            "supported": any(
                at(results, 0.2, corpus, route, "additive", "median_downstream_fisher_fidelity_to_full") < 0.95
                or results["0.2"][corpus]["summary"]["within_route"][route]["median_layer9_interaction_energy_fraction"] > 0.05
                for corpus in corpora for route in routes
            ),
        },
        "S5_attention_route_match_is_scale_stable": {
            "supported": all(
                results[str(scale)][corpus]["summary"]["route_to_oracle"]["hierarchical128"]["attention"]["median_fisher_fidelity"] >= 0.85
                for scale in config["edit_norm_fractions"] for corpus in corpora
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
    }, indent=2))

    lines = [
        "# Scale red-team of the Layer-8 → Layer-9 path split",
        "",
        "The same direct/attention hybrids are evaluated at input-edit norms of 1%, 5%, 10%, and 20%. Sites and routes are reselected deterministically from the same frozen evaluation panels at every scale. Two high-score sites per each of 16 queries are used per corpus.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Scale red-team](layer8_path_scale_redteam.png)",
        "",
        "| Scale | Corpus | Route | Direct fidelity | Additive fidelity | Layer-9 interaction energy | Attention route→oracle | Full route→oracle |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for scale in config["edit_norm_fractions"]:
        for corpus in corpora:
            summary = results[str(scale)][corpus]["summary"]
            for route in routes:
                within = summary["within_route"][route]
                if route == "oracle":
                    attention_match = full_match = 1.0
                else:
                    comparison = summary["route_to_oracle"][route]
                    attention_match = comparison["attention"]["median_fisher_fidelity"]
                    full_match = comparison["full"]["median_fisher_fidelity"]
                lines.append(
                    f"| {scale:.2f} | {corpus} | {route} | "
                    f"{within['direct']['median_downstream_fisher_fidelity_to_full']:.3f} | "
                    f"{within['additive']['median_downstream_fisher_fidelity_to_full']:.3f} | "
                    f"{within['median_layer9_interaction_energy_fraction']:.3e} | "
                    f"{attention_match:.3f} | {full_match:.3f} |"
                )
    lines += [
        "",
        "At larger scales, additive failure can arise both from the bilinear cross term inside layer 9 and from nonlinear propagation through later normalization, attention, MLPs, and the bounded-logit transform. This experiment does not assign those later interactions to a unique layer.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for corpus in corpora:
        for route in routes:
            label = f"{corpus}/{route}"
            scales = config["edit_norm_fractions"]
            axes[0].plot(scales, [
                at(results, scale, corpus, route, "additive", "median_downstream_fisher_fidelity_to_full")
                for scale in scales
            ], marker="o", label=label)
            axes[1].plot(scales, [
                results[str(scale)][corpus]["summary"]["within_route"][route]["median_layer9_interaction_energy_fraction"]
                for scale in scales
            ], marker="o", label=label)
    axes[0].set_title("Downstream additive fidelity")
    axes[0].set_ylabel("median Fisher fidelity")
    axes[1].set_title("Immediate bilinear interaction")
    axes[1].set_ylabel("median interaction/full energy")
    for axis in axes:
        axis.set_xlabel("input edit norm fraction")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
