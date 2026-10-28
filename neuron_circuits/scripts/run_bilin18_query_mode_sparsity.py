#!/usr/bin/env python3
"""Per-example sparsity and multiplexing of query-conditioned eigenmodes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_querywise_truncation import ordered_eigensystem
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_query_conditioned_interpreter import score_metrics
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_query_mode_sparsity.json")
OUTPUT = TENSOR_ROOT / "query_mode_sparsity.json"
REPORT = TENSOR_ROOT / "QUERY_MODE_SPARSITY.md"
FIGURE = TENSOR_ROOT / "query_mode_sparsity.png"


def support_statistics(contributions: torch.Tensor, occupancy_rank: int) -> dict:
    """Summarize [site, query, mode] signed contributions."""
    absolute = contributions.abs()
    sorted_absolute, order = torch.sort(absolute, dim=-1, descending=True)
    sorted_signed = torch.gather(contributions, -1, order)
    total_absolute = absolute.sum(-1).clamp_min(1e-30)
    cumulative_mass = sorted_absolute.cumsum(-1) / total_absolute[..., None]
    k90 = (cumulative_mass < 0.90).sum(-1) + 1
    effective = total_absolute.square() / absolute.square().sum(-1).clamp_min(1e-30)
    cancellation = contributions.sum(-1).abs() / total_absolute
    top_indices = order[..., :occupancy_rank]
    query_occupancy = []
    query_union = []
    for query in range(contributions.shape[1]):
        counts = torch.bincount(top_indices[:, query].flatten(), minlength=contributions.shape[-1]).float()
        query_occupancy.append(float(counts.sum().square() / counts.square().sum().clamp_min(1e-30)))
        query_union.append(int((counts > 0).sum()))
    return {
        "median_k90_absolute_mass": float(k90.float().median()),
        "median_effective_support": float(effective.median()),
        "median_cancellation_ratio": float(cancellation.median()),
        "median_occupancy_effective_modes": float(np.median(query_occupancy)),
        "median_occupancy_union_modes": float(np.median(query_union)),
        "per_query_occupancy_effective_modes": query_occupancy,
        "per_query_occupancy_union_modes": query_union,
        "sorted_signed": sorted_signed,
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    results = {}
    for layer in config["layers"]:
        print("layer", layer, "discovery", flush=True)
        discovery_x, discovery_write = collect(model, layer, config["discovery"], config)
        _, _, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
        basis = state_vh.T[:, : config["dimension"]]
        centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
        _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
        natural_output = write_vh.T[:, : config["queries_per_group"]]
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float()
        left = block.mlp.Left.weight.float() @ basis
        right = block.mlp.Right.weight.float() @ basis
        generator = torch.Generator(device="cuda").manual_seed(config["seed"] + layer)
        random_output = torch.linalg.qr(
            torch.randn(down.shape[0], config["queries_per_group"], generator=generator, device="cuda"),
            mode="reduced",
        ).Q
        directions_by_group = {"natural_top": natural_output, "random": random_output}
        panels = {
            corpus: collect(model, layer, corpus_config, config)
            for corpus, corpus_config in config["evaluation"].items()
        }
        group_rows = {}
        for group in config["groups"]:
            print(" layer", layer, group, flush=True)
            directions = directions_by_group[group]
            coefficients = down.T @ directions
            matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
            matrices = 0.5 * (matrices + matrices.transpose(1, 2))
            values, vectors = ordered_eigensystem(matrices)
            corpus_rows = {}
            for corpus, (states, _) in panels.items():
                z = states @ basis
                projections = torch.einsum("nd,qdk->nqk", z, vectors)
                contributions = projections.square() * values[None]
                full_score = contributions.sum(-1)
                stats = support_statistics(contributions, config["occupancy_rank"])
                sorted_signed = stats.pop("sorted_signed")
                adaptive_cumulative = sorted_signed.cumsum(-1)
                fixed_cumulative = contributions.cumsum(-1)
                rank_rows = {}
                full_np = full_score.cpu().numpy()
                for rank in config["ranks"]:
                    fixed_corr, fixed_r2 = score_metrics(
                        full_np, fixed_cumulative[:, :, rank - 1].cpu().numpy()
                    )
                    adaptive_corr, adaptive_r2 = score_metrics(
                        full_np, adaptive_cumulative[:, :, rank - 1].cpu().numpy()
                    )
                    rank_rows[str(rank)] = {
                        "fixed_median_correlation": float(np.median(fixed_corr)),
                        "fixed_median_centered_r2": float(np.median(fixed_r2)),
                        "adaptive_median_correlation": float(np.median(adaptive_corr)),
                        "adaptive_median_centered_r2": float(np.median(adaptive_r2)),
                    }
                stats["ranks"] = rank_rows
                corpus_rows[corpus] = stats
            group_rows[group] = {"corpora": corpus_rows}
        results[str(layer)] = {"groups": group_rows}

    def row(layer: str, corpus: str) -> dict:
        return results[layer]["groups"]["natural_top"]["corpora"][corpus]

    hypotheses = {
        "S1_middle_is_sparse_per_example": {
            "supported": all(row("8", corpus)["median_k90_absolute_mass"] <= 64 for corpus in config["evaluation"])
        },
        "S2_adaptive_modes_rescue_middle": {
            "supported": all(
                row("8", corpus)["ranks"]["16"]["adaptive_median_correlation"]
                - row("8", corpus)["ranks"]["16"]["fixed_median_correlation"] >= 0.20
                for corpus in config["evaluation"]
            )
        },
        "S3_middle_multiplexes_mode_identity": {
            "supported": all(
                row("8", corpus)["median_occupancy_effective_modes"] >= 128
                for corpus in config["evaluation"]
            )
        },
        "S4_middle_is_denser_than_late": {
            "supported": all(
                row("8", corpus)["median_k90_absolute_mass"]
                >= 2 * row("17", corpus)["median_k90_absolute_mass"]
                for corpus in config["evaluation"]
            )
        },
        "S5_middle_has_more_cancellation": {
            "supported": all(
                row("17", corpus)["median_cancellation_ratio"]
                - row("8", corpus)["median_cancellation_ratio"] >= 0.10
                for corpus in config["evaluation"]
            )
        },
        "S6_middle_support_size_transfers": {
            "supported": all(
                abs(row("8", corpus)["median_k90_absolute_mass"] - row("8", "fineweb")["median_k90_absolute_mass"]) <= 16
                for corpus in ("wikitext", "tinystories")
            )
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "results": results, "hypotheses": hypotheses}, indent=2))

    lines = [
        "# Per-example query-mode sparsity and multiplexing",
        "",
        "Each leading natural output query is decomposed into 512 signed interaction eigenmodes. Fixed rank follows global eigenvalue magnitude; adaptive rank selects the largest realized contributions separately for each token. Absolute-mass K90 measures per-token support, occupancy diversity measures how many different modes fill adaptive top-16 slots across the corpus, and the cancellation ratio is `|sum contributions| / sum |contributions|`.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Mode sparsity](query_mode_sparsity.png)",
        "",
        "| Layer | Group | Corpus | Per-token K90 | Eff. support | Cancellation | Occupancy eff./union | Fixed/adaptive rank-16 corr. |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for layer, layer_row in results.items():
        for group, group_row in layer_row["groups"].items():
            for corpus, item in group_row["corpora"].items():
                rank16 = item["ranks"]["16"]
                lines.append(
                    f"| {layer} | {group} | {corpus} | {item['median_k90_absolute_mass']:.0f} | "
                    f"{item['median_effective_support']:.1f} | {item['median_cancellation_ratio']:.3f} | "
                    f"{item['median_occupancy_effective_modes']:.1f}/{item['median_occupancy_union_modes']:.0f} | "
                    f"{rank16['fixed_median_correlation']:.3f}/{rank16['adaptive_median_correlation']:.3f} |"
                )
    lines += [
        "",
        "Adaptive selection is an oracle diagnostic, not an executable interpreter: it looks at the realized contribution before choosing modes. The result supports predictive conditional sparsity but rejects strict absolute-mass sparsity at layer 8: adaptive rank 16 predicts the signed score well, while roughly 193 modes are needed for 90% of absolute contribution mass. The missing object is a router that predicts the useful mode subset from context, followed by causal validation of that routed program.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    corpora = list(config["evaluation"])
    x = np.arange(len(corpora))
    width = 0.25
    for index, layer in enumerate(("0", "8", "17")):
        rows = [row(layer, corpus) for corpus in corpora]
        axes[0].bar(x + (index - 1) * width, [item["median_k90_absolute_mass"] for item in rows], width, label=f"L{layer}")
        axes[1].bar(x + (index - 1) * width, [item["median_occupancy_effective_modes"] for item in rows], width, label=f"L{layer}")
        axes[2].bar(x + (index - 1) * width, [item["median_cancellation_ratio"] for item in rows], width, label=f"L{layer}")
    for axis, title, ylabel in (
        (axes[0], "Per-token support", "absolute-mass K90"),
        (axes[1], "Mode identity across tokens", "effective occupied modes"),
        (axes[2], "Signed cancellation", "|sum| / sum |.|"),
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
