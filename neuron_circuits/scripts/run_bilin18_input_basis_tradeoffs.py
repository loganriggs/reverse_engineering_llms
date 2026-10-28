#!/usr/bin/env python3
"""Compare semantic, state-variance, hybrid, and random layer-17 input bases."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_joint_axis_residual_roles import random_orthogonal
from run_bilin18_query_conditioned_interpreter import (
    ARRAYS as MAP_ARRAYS,
    CONFIG as INTERPRETER_CONFIG,
    collect_states_and_writes,
    normalize,
    score_metrics,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_input_basis_tradeoffs.json")
OUTPUT = TENSOR_ROOT / "input_basis_tradeoffs.json"
ARRAYS = TENSOR_ROOT / "input_basis_tradeoffs.npz"
REPORT = TENSOR_ROOT / "INPUT_BASIS_TRADEOFFS.md"
FIGURE = TENSOR_ROOT / "input_basis_tradeoffs.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    saved_map = np.load(MAP_ARRAYS)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    semantic_basis = torch.from_numpy(np.load(ATLAS_ARRAYS)["layer17_axes"]).cuda()
    discovery_states, discovery_writes, _ = collect_states_and_writes(
        model,
        interpreter_config["discovery"],
        interpreter_config["positions"],
        interpreter_config["batch_size"],
    )
    discovery_x = torch.from_numpy(discovery_states).cuda()
    _, _, state_v = torch.pca_lowrank(discovery_x, q=64, center=False, niter=4)
    state32, state64 = state_v[:, :32], state_v[:, :64]
    hybrid_raw = torch.cat((semantic_basis, state32), dim=1)
    hybrid64 = torch.linalg.qr(hybrid_raw, mode="reduced").Q
    random32 = random_orthogonal(1152, 32, config["seed"]).cuda()
    random64 = random_orthogonal(1152, 64, config["seed"] + 1).cuda()
    bases = {
        "semantic32": semantic_basis,
        "state32": state32,
        "state64": state64,
        "hybrid64": hybrid64,
        "random32": random32,
        "random64": random64,
    }

    unembedding = model.lm_head.weight.float()[: tokenizer.vocab_size]
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    semantic_queries = []
    for concept in interpreter_config["semantic_concepts"]:
        positive = torch.as_tensor(flags[:, CATEGORIES.index(concept)].astype(bool), device="cuda")
        semantic_queries.append(unembedding[positive].mean(0) - unembedding[~positive].mean(0))
    semantic_queries = normalize(torch.stack(semantic_queries))
    frequent_ids = torch.from_numpy(saved_map["frequent_ids"]).cuda()
    frequent_queries = normalize(unembedding[frequent_ids])
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    gaussian_queries = normalize(torch.randn(64, 1152, generator=generator, device="cuda"))
    discovery_write = torch.from_numpy(discovery_writes).cuda()
    discovery_write -= discovery_write.mean(0, keepdim=True)
    _, _, natural_v = torch.pca_lowrank(discovery_write, q=64, center=False, niter=4)
    map_outliers = torch.from_numpy(saved_map["output_components"][:, :4].T).cuda()
    queries = {
        "semantic": semantic_queries,
        "frequent_token": frequent_queries,
        "gaussian": gaussian_queries,
        "natural_write_pca": natural_v.T,
        "map_outlier": map_outliers,
    }

    results = {}
    for corpus_name, corpus_config in interpreter_config["evaluation"].items():
        states, writes, _ = collect_states_and_writes(
            model, corpus_config, interpreter_config["positions"], interpreter_config["batch_size"]
        )
        x = torch.from_numpy(states).cuda()
        target_write = torch.from_numpy(writes).cuda()
        centered_target = target_write - target_write.mean(0, keepdim=True)
        target_energy = centered_target.square().sum().clamp_min(1e-30)
        results[corpus_name] = {}
        targets = {family: (target_write @ query.T).cpu().numpy() for family, query in queries.items()}
        for basis_name, basis in bases.items():
            projected_x = (x @ basis) @ basis.T
            projected_product = block.mlp.Left(projected_x) * block.mlp.Right(projected_x)
            projected_write = block.mlp.Down(projected_product)
            centered_prediction = projected_write - projected_write.mean(0, keepdim=True)
            write_r2 = float(1 - (centered_prediction - centered_target).square().sum() / target_energy)
            family_results = {}
            for family, query in queries.items():
                prediction = (projected_write @ query.T).cpu().numpy()
                correlation, r2 = score_metrics(targets[family], prediction)
                family_results[family] = {
                    "median_correlation": float(np.median(correlation)),
                    "median_centered_r2": float(np.median(r2)),
                }
            results[corpus_name][basis_name] = {
                "whole_write_centered_r2": write_r2,
                "queries": family_results,
            }

    def corr(corpus: str, basis: str, family: str) -> float:
        return results[corpus][basis]["queries"][family]["median_correlation"]

    hypotheses = {
        "B1_state_basis_recovers_natural_pcs": {
            "supported": all(corr(c, "state32", "natural_write_pca") - corr(c, "semantic32", "natural_write_pca") >= 0.15 for c in results),
            "gaps": {c: corr(c, "state32", "natural_write_pca") - corr(c, "semantic32", "natural_write_pca") for c in results},
        },
        "B2_semantic_basis_recovers_map_outliers": {
            "supported": all(corr(c, "semantic32", "map_outlier") - corr(c, "state32", "map_outlier") >= 0.10 for c in results),
            "gaps": {c: corr(c, "semantic32", "map_outlier") - corr(c, "state32", "map_outlier") for c in results},
        },
        "B3_hybrid_serves_both": {
            "supported": all(
                corr(c, "hybrid64", family) >= max(corr(c, "semantic32", family), corr(c, "state64", family)) - 0.05
                for c in results for family in ("semantic", "natural_write_pca", "map_outlier")
            ),
            "values": {
                c: {
                    family: {
                        "hybrid64": corr(c, "hybrid64", family),
                        "best_specialist": max(corr(c, "semantic32", family), corr(c, "state64", family)),
                    }
                    for family in ("semantic", "natural_write_pca", "map_outlier")
                }
                for c in results
            },
        },
        "B4_hybrid_beats_random": {
            "supported": all(
                corr(c, "hybrid64", family) > corr(c, "random64", family)
                for c in results for family in queries
            ),
            "minimum_gap": min(
                corr(c, "hybrid64", family) - corr(c, "random64", family)
                for c in results for family in queries
            ),
        },
        "B5_objective_tradeoff": {
            "supported": all(
                corr(c, "semantic32", "map_outlier") > corr(c, "state32", "map_outlier")
                and results[c]["state32"]["whole_write_centered_r2"] > results[c]["semantic32"]["whole_write_centered_r2"]
                for c in results
            ),
            "values": {
                c: {
                    "map_outlier_semantic32": corr(c, "semantic32", "map_outlier"),
                    "map_outlier_state32": corr(c, "state32", "map_outlier"),
                    "write_r2_semantic32": results[c]["semantic32"]["whole_write_centered_r2"],
                    "write_r2_state32": results[c]["state32"]["whole_write_centered_r2"],
                }
                for c in results
            },
        },
    }
    output = {"config": config, "results": results, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        **{name: basis.cpu().numpy() for name, basis in bases.items()},
        natural_write_queries=natural_v.T.cpu().numpy(),
    )

    lines = [
        "# Input-basis tradeoffs for the query interpreter",
        "",
        "The same layer-17 bilinear MLP is projected through semantic/circuit, uncentered state-SVD, hybrid, and random input bases. This asks whether interpretability queries and whole-write reconstruction select the same equivalence class.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Basis tradeoffs](input_basis_tradeoffs.png)",
        "",
        "| Corpus | Basis | Semantic | Natural-write PCA | Map top-4 | Whole-write R² |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, basis_rows in results.items():
        for basis_name, row in basis_rows.items():
            lines.append(
                f"| {corpus} | {basis_name} | {row['queries']['semantic']['median_correlation']:.3f} | "
                f"{row['queries']['natural_write_pca']['median_correlation']:.3f} | "
                f"{row['queries']['map_outlier']['median_correlation']:.3f} | "
                f"{row['whole_write_centered_r2']:.3f} |"
            )
    lines += [
        "",
        "A basis is always relative to a loss or readout family. Semantic score fidelity, high-variance write reconstruction, and full-vector reconstruction are distinct objectives even when they share some directions.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 3, figsize=(14, 4.2))
    basis_names = list(bases)
    for column, family in enumerate(("semantic", "natural_write_pca", "map_outlier")):
        x_positions = np.arange(len(basis_names))
        for corpus in results:
            plots[column].plot(
                x_positions,
                [results[corpus][basis]["queries"][family]["median_correlation"] for basis in basis_names],
                marker="o",
                label=corpus,
            )
        plots[column].set_xticks(x_positions, basis_names, rotation=30, ha="right")
        plots[column].set_ylim(-0.2, 1.02)
        plots[column].set_title(family)
        plots[column].set_ylabel("median score correlation")
        plots[column].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
