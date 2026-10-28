#!/usr/bin/env python3
"""Natural-score rank curves for queried interaction forms across depth."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_query_conditioned_interpreter import score_metrics
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_depth_querywise_truncation.json")
OUTPUT = TENSOR_ROOT / "depth_querywise_truncation.json"
REPORT = TENSOR_ROOT / "DEPTH_QUERYWISE_TRUNCATION.md"
FIGURE = TENSOR_ROOT / "depth_querywise_truncation.png"


def ordered_eigensystem(matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(matrices)
    order = torch.argsort(values.abs(), dim=-1, descending=True)
    values = torch.gather(values, -1, order)
    vectors = torch.gather(vectors, -1, order[:, None, :].expand_as(vectors))
    return values, vectors


def first_passing_rank(curves: dict, ranks: list[int]) -> int | None:
    for rank in ranks:
        row = curves[str(rank)]
        if row["median_correlation"] >= 0.90 and row["median_centered_r2"] >= 0.80:
            return rank
    return None


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
        write_v = write_vh.T
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float()
        left = block.mlp.Left.weight.float() @ basis
        right = block.mlp.Right.weight.float() @ basis
        _, tensor_output = output_map_eigensystem(
            down, block.mlp.Left.weight.float(), block.mlp.Right.weight.float()
        )
        count = config["queries_per_group"]
        generator = torch.Generator(device="cuda").manual_seed(config["seed"] + layer)
        random_output = torch.linalg.qr(
            torch.randn(down.shape[0], count, generator=generator, device="cuda"), mode="reduced"
        ).Q
        groups = {
            "natural_top": write_v[:, :count],
            "natural_middle": write_v[:, config["middle_start"] : config["middle_start"] + count],
            "tensor_top": tensor_output[:, :count],
            "random": random_output,
        }
        panels = {
            corpus: collect(model, layer, corpus_config, config)
            for corpus, corpus_config in config["evaluation"].items()
        }
        group_rows = {}
        for group, directions in groups.items():
            print(" layer", layer, group, flush=True)
            coefficients = down.T @ directions
            matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
            matrices = 0.5 * (matrices + matrices.transpose(1, 2))
            values, vectors = ordered_eigensystem(matrices)
            corpus_rows = {}
            for corpus, (states, writes) in panels.items():
                z = states @ basis
                target = writes @ directions
                projections = torch.einsum("nd,qdk->nqk", z, vectors)
                contributions = projections.square() * values[None]
                cumulative = contributions.cumsum(-1)
                rank_rows = {}
                for rank in config["ranks"]:
                    prediction = cumulative[:, :, rank - 1]
                    correlations, r2 = score_metrics(target.cpu().numpy(), prediction.cpu().numpy())
                    rank_rows[str(rank)] = {
                        "median_correlation": float(np.median(correlations)),
                        "median_centered_r2": float(np.median(r2)),
                        "per_query_correlation": correlations.tolist(),
                        "per_query_centered_r2": r2.tolist(),
                    }
                corpus_rows[corpus] = {
                    "ranks": rank_rows,
                    "first_passing_rank": first_passing_rank(rank_rows, config["ranks"]),
                }
            group_rows[group] = {"corpora": corpus_rows}
        results[str(layer)] = {"groups": group_rows}

    layer8_rank = results["8"]["groups"]["natural_top"]["corpora"]["fineweb"]["first_passing_rank"]
    layer17_rank = results["17"]["groups"]["natural_top"]["corpora"]["fineweb"]["first_passing_rank"]
    hypotheses = {
        "B1_rank64_natural_direction_fidelity": {
            "supported": all(
                results[layer]["groups"]["natural_top"]["corpora"][corpus]["ranks"]["64"]["median_correlation"] >= 0.90
                for layer in results for corpus in config["evaluation"]
            )
        },
        "B2_rank64_natural_magnitude_fidelity": {
            "supported": all(
                results[layer]["groups"]["natural_top"]["corpora"][corpus]["ranks"]["64"]["median_centered_r2"] >= 0.80
                for layer in results for corpus in config["evaluation"]
            )
        },
        "B3_late_needs_fewer_behavioral_modes": {
            "supported": layer8_rank is not None and layer17_rank is not None and layer17_rank <= layer8_rank / 2,
            "layer8": layer8_rank,
            "layer17": layer17_rank,
        },
        "B4_natural_head_beats_random": {
            "supported": all(
                results[layer]["groups"]["natural_top"]["corpora"]["fineweb"]["ranks"]["16"]["median_correlation"]
                - results[layer]["groups"]["random"]["corpora"]["fineweb"]["ranks"]["16"]["median_correlation"] >= 0.10
                for layer in results
            )
        },
        "B5_late_tensor_spikes_have_compact_scores": {
            "supported": all(
                results["17"]["groups"]["tensor_top"]["corpora"][corpus]["ranks"]["16"]["median_correlation"] >= 0.80
                for corpus in config["evaluation"]
            )
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "results": results, "hypotheses": hypotheses}, indent=2))

    lines = [
        "# Natural-score truncation of queried forms across depth",
        "",
        "Weight-energy rank90 was about 200 modes even for leading natural queries. This red-team asks how many signed eigenmodes are needed to predict the query's actual score on held-out language states. Queries and the 512D FineWeb state basis are frozen before evaluation on three corpora.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Query truncation](depth_querywise_truncation.png)",
        "",
        "| Layer | Query family | Corpus | First rank corr≥.90 & R²≥.80 | Rank-16 corr/R² | Rank-64 corr/R² | Rank-512 corr/R² |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for layer, layer_row in results.items():
        for group, group_row in layer_row["groups"].items():
            for corpus, corpus_row in group_row["corpora"].items():
                ranks = corpus_row["ranks"]
                first = corpus_row["first_passing_rank"]
                lines.append(
                    f"| {layer} | {group} | {corpus} | {first if first is not None else '—'} | "
                    f"{ranks['16']['median_correlation']:.3f}/{ranks['16']['median_centered_r2']:.3f} | "
                    f"{ranks['64']['median_correlation']:.3f}/{ranks['64']['median_centered_r2']:.3f} | "
                    f"{ranks['512']['median_correlation']:.3f}/{ranks['512']['median_centered_r2']:.3f} |"
                )
    lines += [
        "",
        "This is observational score sufficiency inside a fixed 512D state basis. It does not establish that the same truncated modes predict finite edits, because tangent effects include context cross terms and can require a larger causal quotient.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for axis, layer in zip(axes, ("0", "8", "17")):
        for group in ("natural_top", "natural_middle", "tensor_top", "random"):
            rank_rows = results[layer]["groups"][group]["corpora"]["fineweb"]["ranks"]
            axis.plot(config["ranks"], [rank_rows[str(rank)]["median_correlation"] for rank in config["ranks"]], marker="o", label=group)
        axis.set_xscale("log", base=2)
        axis.set_ylim(-0.1, 1.02)
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel("interaction eigenmodes")
        axis.set_ylabel("median FineWeb score correlation")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
