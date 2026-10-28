#!/usr/bin/env python3
"""Rank sweep for category specificity of final-layer output modes."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


CONFIG = Path("configs/bilin18_layer17_rank_specificity.json")
OUTPUT = TENSOR_ROOT / "layer17_rank_specificity.json"
REPORT = TENSOR_ROOT / "LAYER17_RANK_SPECIFICITY.md"
FIGURE = TENSOR_ROOT / "layer17_rank_specificity.png"


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
    all_vectors = {
        category: torch.from_numpy(arrays[f"{category}_eigenvectors"]).cuda()
        for category in categories
    }
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
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
                if site["stratum"] in config["strata"]
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
                    native_logits = final_logits(model, pre_site, x)
                    positive = positive_masks[target]
                    native_contrast = float(class_contrast(native_logits, positive))
                    rank_results = {}
                    for rank in config["ranks"]:
                        sources = {}
                        for source in categories:
                            vectors = all_vectors[source][:, :rank]
                            raw_delta = -(x @ vectors) @ vectors.T
                            natural_logits = final_logits(model, pre_site, x + raw_delta)
                            fixed_delta = raw_delta * (
                                config["input_norm_fraction"] * x.norm()
                                / raw_delta.norm().clamp_min(1e-20)
                            )
                            fixed_logits = final_logits(model, pre_site, x + fixed_delta)
                            sources[source] = {
                                "fixed": abs(float(class_contrast(fixed_logits, positive)) - native_contrast),
                                "natural": abs(float(class_contrast(natural_logits, positive)) - native_contrast),
                                "projection_fraction": float(
                                    raw_delta.norm() / x.norm().clamp_min(1e-20)
                                ),
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
        summaries[corpus] = {}
        tests[corpus] = {}
        for stratum in config["strata"]:
            summaries[corpus][stratum] = {}
            tests[corpus][stratum] = {}
            for rank in config["ranks"]:
                matrices = {}
                for metric in ("fixed", "natural"):
                    matrix = np.zeros((len(eligible), len(eligible)))
                    for i, target in enumerate(eligible):
                        rows_for_target = [
                            row for row in records
                            if row["corpus"] == corpus
                            and row["stratum"] == stratum
                            and row["target"] == target
                        ]
                        for j, source in enumerate(eligible):
                            matrix[i, j] = np.median([
                                row["ranks"][str(rank)][source][metric]
                                for row in rows_for_target
                            ])
                    matrices[metric] = matrix.tolist()
                summaries[corpus][stratum][str(rank)] = {
                    "eligible": eligible,
                    "matrices": matrices,
                }
                tests[corpus][stratum][str(rank)] = {
                    metric: assignment_test(np.asarray(matrix))
                    for metric, matrix in matrices.items()
                }

    hypotheses = {
        "RS1_rank1_assignment": {
            "supported": all(
                tests[c]["high_positive"]["1"]["fixed"]["exact_p"] <= 0.05
                and tests[c]["high_positive"]["1"]["fixed"]["diagonal_is_best"]
                for c in corpus_specs
            ),
            "per_corpus": {c: tests[c]["high_positive"]["1"]["fixed"] for c in corpus_specs},
        },
        "RS2_rank1_row_wins": {
            "supported": all(
                tests[c]["high_positive"]["1"]["fixed"]["row_wins"] >= 4
                for c in corpus_specs
            ),
            "per_corpus": {c: tests[c]["high_positive"]["1"]["fixed"]["row_wins"] for c in corpus_specs},
        },
        "RS3_rank1_improves_rank4": {
            "supported": all(
                tests[c]["high_positive"]["1"]["fixed"]["exact_p"]
                <= tests[c]["high_positive"]["4"]["fixed"]["exact_p"]
                for c in corpus_specs
            ),
            "per_corpus": {
                c: {
                    "rank1": tests[c]["high_positive"]["1"]["fixed"]["exact_p"],
                    "rank4": tests[c]["high_positive"]["4"]["fixed"]["exact_p"],
                }
                for c in corpus_specs
            },
        },
        "RS4_random_positive_rank1": {
            "supported": all(
                tests[c]["random_positive"]["1"]["fixed"]["exact_p"] <= 0.05
                and tests[c]["random_positive"]["1"]["fixed"]["diagonal_is_best"]
                for c in corpus_specs
            ),
            "per_corpus": {c: tests[c]["random_positive"]["1"]["fixed"] for c in corpus_specs},
        },
        "RS5_natural_rank1": {
            "supported": all(
                tests[c]["high_positive"]["1"]["natural"]["exact_p"] <= 0.05
                and tests[c]["high_positive"]["1"]["natural"]["diagonal_is_best"]
                for c in corpus_specs
            ),
            "per_corpus": {c: tests[c]["high_positive"]["1"]["natural"] for c in corpus_specs},
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
        "# Layer-17 rank-specificity sweep",
        "",
        "The rank-4 category assignment failed despite strong top-versus-low effects. This sweep tests whether the leading individual mode is category-specific while additional high-magnitude modes form a shared output-control subspace.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Rank specificity](layer17_rank_specificity.png)",
        "",
        "| Corpus | Rank | High fixed p | High row wins | High natural p | Random fixed p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpus_specs:
        for rank in config["ranks"]:
            high = tests[corpus]["high_positive"][str(rank)]
            random = tests[corpus]["random_positive"][str(rank)]
            lines.append(
                f"| {corpus} | {rank} | {high['fixed']['exact_p']:.4f} | "
                f"{high['fixed']['row_wins']} | {high['natural']['exact_p']:.4f} | "
                f"{random['fixed']['exact_p']:.4f} |"
            )
    lines += ["", "All matrices and per-site rank sweeps are in `layer17_rank_specificity.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(corpus_specs)
    fig, axes = plt.subplots(1, len(corpora), figsize=(5 * len(corpora), 4.2))
    for axis, corpus in zip(axes, corpora):
        axis.plot(
            config["ranks"],
            [tests[corpus]["high_positive"][str(rank)]["fixed"]["exact_p"] for rank in config["ranks"]],
            marker="o",
            label="high fixed",
        )
        axis.plot(
            config["ranks"],
            [tests[corpus]["random_positive"][str(rank)]["fixed"]["exact_p"] for rank in config["ranks"]],
            marker="o",
            label="random fixed",
        )
        axis.axhline(0.05, color="black", linewidth=1)
        axis.set_xlabel("spectral rank")
        axis.set_ylabel("exact assignment p")
        axis.set_title(corpus)
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
