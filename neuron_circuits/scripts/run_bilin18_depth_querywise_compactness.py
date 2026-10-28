#!/usr/bin/env python3
"""Test whether globally broad middle layers contain compact queried operations."""

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


CONFIG = Path("configs/bilin18_depth_querywise_compactness.json")
OUTPUT = TENSOR_ROOT / "depth_querywise_compactness.json"
REPORT = TENSOR_ROOT / "DEPTH_QUERYWISE_COMPACTNESS.md"
FIGURE = TENSOR_ROOT / "depth_querywise_compactness.png"


def form_summaries(matrices: torch.Tensor, config: dict) -> dict:
    values = torch.sort(torch.linalg.eigvalsh(matrices).abs(), dim=-1, descending=True).values
    energy = values.square()
    total = energy.sum(-1).clamp_min(1e-30)
    cumulative = energy.cumsum(-1) / total[:, None]
    rank90 = (cumulative < 0.90).sum(-1) + 1
    effective = total.square() / energy.square().sum(-1).clamp_min(1e-30)
    return {
        "per_query_top4_energy": (energy[:, : config["top_mode_count"]].sum(-1) / total).cpu().tolist(),
        "per_query_top16_energy": (energy[:, : config["extended_mode_count"]].sum(-1) / total).cpu().tolist(),
        "per_query_rank90": rank90.cpu().tolist(),
        "per_query_effective_rank": effective.cpu().tolist(),
        "median_top4_energy": float((energy[:, : config["top_mode_count"]].sum(-1) / total).median()),
        "median_top16_energy": float((energy[:, : config["extended_mode_count"]].sum(-1) / total).median()),
        "median_rank90": float(rank90.float().median()),
        "median_effective_rank": float(effective.median()),
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
        state_v = state_vh.T
        centered_write = discovery_write - discovery_write.mean(0, keepdim=True)
        _, _, write_vh = torch.linalg.svd(centered_write, full_matrices=False)
        write_v = write_vh.T
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float()
        left_weight = block.mlp.Left.weight.float()
        right_weight = block.mlp.Right.weight.float()
        _, tensor_output = output_map_eigensystem(down, left_weight, right_weight)
        count = config["queries_per_group"]
        generator = torch.Generator(device="cuda").manual_seed(config["seed"] + layer)
        random_output = torch.linalg.qr(
            torch.randn(down.shape[0], count, generator=generator, device="cuda"),
            mode="reduced",
        ).Q
        groups = {
            "natural_top": write_v[:, :count],
            "natural_middle": write_v[:, config["middle_start"] : config["middle_start"] + count],
            "tensor_top": tensor_output[:, :count],
            "random": random_output,
        }
        panels = {}
        for corpus, corpus_config in config["evaluation"].items():
            print(" layer", layer, corpus, flush=True)
            panels[corpus] = collect(model, layer, corpus_config, config)

        dimension_rows = {}
        for dimension in config["dimensions"]:
            print(" layer", layer, "dimension", dimension, flush=True)
            basis = state_v[:, :dimension]
            left = left_weight @ basis
            right = right_weight @ basis
            group_rows = {}
            for group, directions in groups.items():
                coefficients = down.T @ directions
                matrices = torch.einsum("hq,hi,hj->qij", coefficients, left, right)
                matrices = 0.5 * (matrices + matrices.transpose(1, 2))
                row = form_summaries(matrices, config)
                corpus_rows = {}
                for corpus, (states, writes) in panels.items():
                    z = states @ basis
                    target = writes @ directions
                    prediction = torch.einsum("ni,qij,nj->nq", z, matrices, z)
                    correlations, r2 = score_metrics(target.cpu().numpy(), prediction.cpu().numpy())
                    variances = target.var(0, unbiased=False).cpu().numpy()
                    corpus_rows[corpus] = {
                        "median_correlation": float(np.median(correlations)),
                        "median_centered_r2": float(np.median(r2)),
                        "median_target_variance": float(np.median(variances)),
                        "per_query_correlation": correlations.tolist(),
                        "per_query_centered_r2": r2.tolist(),
                        "per_query_target_variance": variances.tolist(),
                    }
                row["corpora"] = corpus_rows
                group_rows[group] = row
            dimension_rows[str(dimension)] = group_rows
        results[str(layer)] = {"dimensions": dimension_rows}

    final_dimension = str(config["dimensions"][-1])
    natural_rank = {
        layer: results[layer]["dimensions"][final_dimension]["natural_top"]["median_rank90"]
        for layer in results
    }
    hypotheses = {
        "Q1_middle_queries_are_individually_compact": {
            "supported": natural_rank["8"] <= 32,
            "layer8_rank90": natural_rank["8"],
        },
        "Q2_natural_queries_beat_random_compactness": {
            "supported": all(
                results[layer]["dimensions"][final_dimension]["natural_top"]["median_top16_energy"]
                - results[layer]["dimensions"][final_dimension]["random"]["median_top16_energy"] >= 0.10
                for layer in results
            )
        },
        "Q3_natural_query_scores_are_faithful": {
            "supported": all(
                results[layer]["dimensions"][final_dimension]["natural_top"]["corpora"][corpus]["median_correlation"] >= 0.90
                for layer in results for corpus in config["evaluation"]
            )
        },
        "Q4_tensor_top_queries_are_naturally_used": {
            "supported": all(
                results[layer]["dimensions"][final_dimension]["tensor_top"]["corpora"]["fineweb"]["median_target_variance"]
                > results[layer]["dimensions"][final_dimension]["natural_middle"]["corpora"]["fineweb"]["median_target_variance"]
                for layer in results
            )
        },
        "Q5_middle_queries_are_relatively_distributed": {
            "supported": natural_rank["8"] >= 2 * natural_rank["17"]
            and natural_rank["8"] >= 1.5 * natural_rank["0"],
            "per_layer": natural_rank,
        },
        "Q6_query_interpreter_transfers_ood": {
            "supported": all(
                abs(
                    results[layer]["dimensions"][final_dimension]["natural_top"]["corpora"][corpus]["median_correlation"]
                    - results[layer]["dimensions"][final_dimension]["natural_top"]["corpora"]["fineweb"]["median_correlation"]
                ) <= 0.10
                for layer in results for corpus in ("wikitext", "tinystories")
            )
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "results": results, "hypotheses": hypotheses}, indent=2))

    lines = [
        "# Query-wise compactness across depth",
        "",
        "This asks whether a globally broad layer is nevertheless a library of individually compact operations. Output queries are frozen from FineWeb discovery writes (`natural_top`, `natural_middle`), the complete raw tensor (`tensor_top`), or a random orthonormal control. Each query is converted exactly into an interaction form inside state-SVD input bases and evaluated on three corpora.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Query-wise compactness](depth_querywise_compactness.png)",
        "",
        "### Dimension 512",
        "",
        "| Layer | Query family | Top-4 energy | Top-16 energy | Rank90 | Eff. rank | Fine/Wiki/Tiny corr. | FineWeb variance |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, layer_row in results.items():
        for group, row in layer_row["dimensions"][final_dimension].items():
            corpora = row["corpora"]
            lines.append(
                f"| {layer} | {group} | {row['median_top4_energy']:.3f} | {row['median_top16_energy']:.3f} | "
                f"{row['median_rank90']:.0f} | {row['median_effective_rank']:.1f} | "
                f"{corpora['fineweb']['median_correlation']:.3f}/{corpora['wikitext']['median_correlation']:.3f}/{corpora['tinystories']['median_correlation']:.3f} | "
                f"{corpora['fineweb']['median_target_variance']:.2e} |"
            )
    lines += [
        "",
        "Query-wise low rank is conditional simplicity, not a neuron count. Different queries may reuse overlapping modes, and the 512D basis still omits part of layers 0 and 8. Random and middle-spectrum queries expose how much apparent compactness follows generically from restricting the input manifold.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    groups = ("natural_top", "natural_middle", "tensor_top", "random")
    x = np.arange(len(groups))
    width = 0.25
    for index, layer in enumerate(("0", "8", "17")):
        rows = results[layer]["dimensions"][final_dimension]
        axes[0].bar(x + (index - 1) * width, [rows[g]["median_rank90"] for g in groups], width, label=f"L{layer}")
        axes[1].bar(x + (index - 1) * width, [rows[g]["median_top16_energy"] for g in groups], width, label=f"L{layer}")
        axes[2].bar(x + (index - 1) * width, [rows[g]["corpora"]["fineweb"]["median_correlation"] for g in groups], width, label=f"L{layer}")
    for axis, title, ylabel in (
        (axes[0], "Conditional form rank", "median rank90"),
        (axes[1], "Leading-mode concentration", "median top-16 energy"),
        (axes[2], "FineWeb score fidelity", "median correlation"),
    ):
        axis.set_xticks(x, groups, rotation=18)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
