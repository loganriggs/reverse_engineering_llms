#!/usr/bin/env python3
"""Locate whether layer-17 category specificity is lost before or after RMSNorm."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin18_layer17_output_spectra import (
    ARRAYS as SPECTRAL_ARRAYS,
    OUTPUT as SPECTRAL_OUTPUT,
    class_contrast,
    final_logits,
    prefix_to_layer17,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_direct_vs_final.json")
OUTPUT = TENSOR_ROOT / "layer17_direct_vs_final.json"
REPORT = TENSOR_ROOT / "LAYER17_DIRECT_VS_FINAL.md"
FIGURE = TENSOR_ROOT / "layer17_direct_vs_final.png"


def assignment_test(matrix: np.ndarray) -> dict:
    size = len(matrix)
    diagonal = float(np.trace(matrix))
    scores = np.asarray([
        sum(matrix[row, column] for row, column in enumerate(permutation))
        for permutation in itertools.permutations(range(size))
    ])
    return {
        "diagonal_score": diagonal,
        "best_score": float(scores.max()),
        "exact_p": float(np.mean(scores >= diagonal - 1e-12)),
        "diagonal_is_best": bool(diagonal >= scores.max() - 1e-12),
        "row_wins": int(sum(matrix[i, i] >= matrix[i].max() - 1e-12 for i in range(size))),
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(SPECTRAL_OUTPUT.read_text())
    arrays = np.load(SPECTRAL_ARRAYS)
    categories = config["categories"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    positive_masks = {
        category: torch.as_tensor(
            flags[:, CATEGORIES.index(category)].astype(bool), device="cuda"
        )
        for category in categories
    }
    output_directions = {
        category: torch.from_numpy(arrays[f"{category}_output_direction"]).cuda()
        for category in categories
    }
    vectors = {
        category: torch.from_numpy(arrays[f"{category}_eigenvectors"]).cuda()
        for category in categories
    }
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    corpus_specs = pilot["config"]["corpora"]
    records = []

    for corpus, corpus_config in corpus_specs.items():
        all_rows = np.load(corpus_config["path"], mmap_mode="r")
        rows = all_rows[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        for target in categories:
            sites = [
                site for site in pilot["site_pools"][corpus][target]
                if site["stratum"] == config["stratum"]
            ]
            by_row = {}
            for site in sites:
                by_row.setdefault(site["row"], []).append(site)
            for row_index, row_sites in by_row.items():
                raw = np.asarray(rows[row_index : row_index + 1, :129], dtype=np.int64)
                ids = torch.as_tensor(raw[:, :128], device="cuda")
                pre, normalized = prefix_to_layer17(model, ids)
                for site in row_sites:
                    position = site["position"]
                    x = normalized[0, position].float()
                    pre_site = pre[0, position].float()
                    native_product = block.mlp.Left(x) * block.mlp.Right(x)
                    native_write = block.mlp.Down(native_product) + block.mlp.Down_bias
                    native_logits = final_logits(model, pre_site, x)
                    native_contrast = float(
                        class_contrast(native_logits, positive_masks[target])
                    )
                    rank_results = {}
                    for rank in config["ranks"]:
                        sources = {}
                        for source in categories:
                            source_vectors = vectors[source][:, :rank]
                            delta = -(x @ source_vectors) @ source_vectors.T
                            delta *= (
                                config["input_norm_fraction"] * x.norm()
                                / delta.norm().clamp_min(1e-20)
                            )
                            changed_x = x + delta
                            changed_product = block.mlp.Left(changed_x) * block.mlp.Right(changed_x)
                            changed_write = block.mlp.Down(changed_product) + block.mlp.Down_bias
                            direct_effect = float(
                                output_directions[target] @ (changed_write - native_write)
                            )
                            changed_logits = final_logits(model, pre_site, changed_x)
                            final_effect = float(
                                class_contrast(changed_logits, positive_masks[target])
                                - native_contrast
                            )
                            sources[source] = {
                                "direct_absolute": abs(direct_effect),
                                "direct_signed": direct_effect,
                                "final_absolute": abs(final_effect),
                                "final_signed": final_effect,
                            }
                        rank_results[str(rank)] = sources
                    records.append(
                        {
                            "corpus": corpus,
                            "target": target,
                            **site,
                            "ranks": rank_results,
                        }
                    )

    summaries = {}
    tests = {}
    for corpus in corpus_specs:
        eligible = [category for category in categories if pilot["site_pools"][corpus][category]]
        summaries[corpus] = {"eligible": eligible, "ranks": {}}
        tests[corpus] = {}
        for rank in config["ranks"]:
            matrices = {}
            for metric in ("direct_absolute", "final_absolute"):
                matrix = np.zeros((len(eligible), len(eligible)))
                for i, target in enumerate(eligible):
                    target_rows = [
                        row for row in records
                        if row["corpus"] == corpus and row["target"] == target
                    ]
                    for j, source in enumerate(eligible):
                        matrix[i, j] = np.median([
                            row["ranks"][str(rank)][source][metric]
                            for row in target_rows
                        ])
                matrices[metric] = matrix.tolist()
            summaries[corpus]["ranks"][str(rank)] = matrices
            tests[corpus][str(rank)] = {
                metric: assignment_test(np.asarray(matrix))
                for metric, matrix in matrices.items()
            }

    better_direct_rank = {
        corpus: min(
            config["ranks"],
            key=lambda rank: tests[corpus][str(rank)]["direct_absolute"]["exact_p"],
        )
        for corpus in corpus_specs
    }
    hypotheses = {
        "P1_direct_rank1_assignment": {
            "supported": all(
                tests[c]["1"]["direct_absolute"]["exact_p"] <= 0.05
                and tests[c]["1"]["direct_absolute"]["diagonal_is_best"]
                for c in corpus_specs
            ),
            "per_corpus": {c: tests[c]["1"]["direct_absolute"] for c in corpus_specs},
        },
        "P2_direct_rank4_assignment": {
            "supported": all(
                tests[c]["4"]["direct_absolute"]["exact_p"] <= 0.05
                and tests[c]["4"]["direct_absolute"]["diagonal_is_best"]
                for c in corpus_specs
            ),
            "per_corpus": {c: tests[c]["4"]["direct_absolute"] for c in corpus_specs},
        },
        "P3_direct_improves_final": {
            "supported": all(
                tests[c][str(rank)]["direct_absolute"]["exact_p"]
                < tests[c][str(rank)]["final_absolute"]["exact_p"]
                for c in corpus_specs for rank in config["ranks"]
            ),
            "per_corpus": {
                c: {
                    str(rank): {
                        "direct": tests[c][str(rank)]["direct_absolute"]["exact_p"],
                        "final": tests[c][str(rank)]["final_absolute"]["exact_p"],
                    }
                    for rank in config["ranks"]
                }
                for c in corpus_specs
            },
        },
        "P4_direct_row_specificity": {
            "supported": all(
                tests[c][str(better_direct_rank[c])]["direct_absolute"]["row_wins"] >= 4
                for c in corpus_specs
            ),
            "best_rank_and_wins": {
                c: {
                    "rank": better_direct_rank[c],
                    "wins": tests[c][str(better_direct_rank[c])]["direct_absolute"]["row_wins"],
                }
                for c in corpus_specs
            },
        },
    }
    output = {
        "config": config,
        "summaries": summaries,
        "tests": tests,
        "hypotheses": hypotheses,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Layer-17 direct-write versus end-to-end specificity",
        "",
        "The output-conditioned quadratic form is exact for the MLP write before residual addition and final RMSNorm. This test compares cross-category assignment at that direct interface with the actual final logits for identical edits.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Direct versus final](layer17_direct_vs_final.png)",
        "",
        "| Corpus | Rank | Direct assignment p | Direct row wins | Final assignment p | Final row wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpus_specs:
        for rank in config["ranks"]:
            direct = tests[corpus][str(rank)]["direct_absolute"]
            final = tests[corpus][str(rank)]["final_absolute"]
            lines.append(
                f"| {corpus} | {rank} | {direct['exact_p']:.4f} | {direct['row_wins']} | "
                f"{final['exact_p']:.4f} | {final['row_wins']} |"
            )
    lines += ["", "Full matrices and per-site effects are in `layer17_direct_vs_final.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(corpus_specs)
    x = np.arange(len(corpora))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for rank in config["ranks"]:
        ax.plot(
            x,
            [tests[c][str(rank)]["direct_absolute"]["exact_p"] for c in corpora],
            marker="o",
            label=f"direct rank {rank}",
        )
        ax.plot(
            x,
            [tests[c][str(rank)]["final_absolute"]["exact_p"] for c in corpora],
            marker="x",
            linestyle="--",
            label=f"final rank {rank}",
        )
    ax.axhline(0.05, color="black", linewidth=1)
    ax.set_xticks(x, corpora)
    ax.set_ylabel("exact assignment p")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
