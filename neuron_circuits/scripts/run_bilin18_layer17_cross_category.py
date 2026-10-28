#!/usr/bin/env python3
"""Cross-category specificity red-team for layer-17 spectral routes."""

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
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer17_cross_category.json")
OUTPUT = TENSOR_ROOT / "layer17_cross_category.json"
REPORT = TENSOR_ROOT / "LAYER17_CROSS_CATEGORY.md"
FIGURE = TENSOR_ROOT / "layer17_cross_category.png"


def assignment_test(matrix: np.ndarray) -> dict:
    size = matrix.shape[0]
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
    vectors = {
        category: torch.from_numpy(
            arrays[f"{category}_eigenvectors"][:, : config["rank"]]
        ).cuda()
        for category in categories
    }

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    records = []
    corpus_specs = pilot["config"]["corpora"]

    for corpus, corpus_config in corpus_specs.items():
        all_rows = np.load(corpus_config["path"], mmap_mode="r")
        rows = all_rows[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        for target_category in categories:
            selected_sites = [
                site for site in pilot["site_pools"][corpus][target_category]
                if site["stratum"] in config["strata"]
            ]
            by_row = {}
            for site in selected_sites:
                by_row.setdefault(site["row"], []).append(site)
            for row_index, sites in by_row.items():
                raw = np.asarray(rows[row_index : row_index + 1, :129], dtype=np.int64)
                ids = torch.as_tensor(raw[:, :128], device="cuda")
                pre, normalized = prefix_to_layer17(model, ids)
                for site in sites:
                    position = site["position"]
                    x = normalized[0, position].float()
                    pre_site = pre[0, position].float()
                    native_logits = final_logits(model, pre_site, x)
                    positive = positive_masks[target_category]
                    native_contrast = float(class_contrast(native_logits, positive))
                    source_results = {}
                    for source_category in categories:
                        raw_delta = -(x @ vectors[source_category]) @ vectors[source_category].T
                        natural_logits = final_logits(model, pre_site, x + raw_delta)
                        natural_effect = natural_logits - native_logits
                        natural_change = float(
                            class_contrast(natural_logits, positive) - native_contrast
                        )
                        fixed_delta = raw_delta * (
                            config["input_norm_fraction"] * x.norm()
                            / raw_delta.norm().clamp_min(1e-20)
                        )
                        fixed_logits = final_logits(model, pre_site, x + fixed_delta)
                        fixed_effect = fixed_logits - native_logits
                        fixed_change = float(
                            class_contrast(fixed_logits, positive) - native_contrast
                        )
                        _, _, energy = fisher_fidelity(
                            native_logits, fixed_effect, torch.zeros_like(fixed_effect)
                        )
                        source_results[source_category] = {
                            "fixed_absolute_class_effect": abs(fixed_change),
                            "fixed_signed_class_effect": fixed_change,
                            "fixed_fisher_energy": energy,
                            "fixed_selectivity": abs(fixed_change) / max(energy ** 0.5, 1e-20),
                            "natural_absolute_class_effect": abs(natural_change),
                            "natural_projection_fraction": float(
                                raw_delta.norm() / x.norm().clamp_min(1e-20)
                            ),
                        }
                    records.append(
                        {
                            "corpus": corpus,
                            "target_category": target_category,
                            **site,
                            "sources": source_results,
                        }
                    )

    summaries = {}
    tests = {}
    for corpus in corpus_specs:
        summaries[corpus] = {}
        tests[corpus] = {}
        eligible = [
            category for category in categories
            if pilot["site_pools"][corpus][category]
        ]
        for stratum in config["strata"]:
            matrices = {}
            for metric in (
                "fixed_absolute_class_effect",
                "fixed_selectivity",
                "natural_absolute_class_effect",
            ):
                matrix = np.zeros((len(eligible), len(eligible)))
                for target_index, target in enumerate(eligible):
                    target_rows = [
                        row for row in records
                        if row["corpus"] == corpus
                        and row["stratum"] == stratum
                        and row["target_category"] == target
                    ]
                    for source_index, source in enumerate(eligible):
                        matrix[target_index, source_index] = np.median([
                            row["sources"][source][metric] for row in target_rows
                        ])
                matrices[metric] = matrix.tolist()
            summaries[corpus][stratum] = {
                "eligible_categories": eligible,
                "matrices": matrices,
            }
            tests[corpus][stratum] = {
                metric: assignment_test(np.asarray(matrix))
                for metric, matrix in matrices.items()
            }
            fixed = np.asarray(matrices["fixed_absolute_class_effect"])
            tests[corpus][stratum]["fixed_row_wins"] = int(
                sum(fixed[index, index] >= fixed[index].max() - 1e-12 for index in range(len(eligible)))
            )

    hypotheses = {
        "X1_fixed_norm_assignment": {
            "supported": all(
                tests[corpus]["high_positive"]["fixed_absolute_class_effect"]["exact_p"] <= 0.05
                and tests[corpus]["high_positive"]["fixed_absolute_class_effect"]["diagonal_is_best"]
                for corpus in corpus_specs
            ),
            "per_corpus": {
                corpus: tests[corpus]["high_positive"]["fixed_absolute_class_effect"]
                for corpus in corpus_specs
            },
        },
        "X2_fixed_norm_row_wins": {
            "supported": all(
                tests[corpus]["high_positive"]["fixed_row_wins"] >= 4
                for corpus in corpus_specs
            ),
            "per_corpus": {
                corpus: tests[corpus]["high_positive"]["fixed_row_wins"]
                for corpus in corpus_specs
            },
        },
        "X3_selectivity_assignment": {
            "supported": all(
                tests[corpus]["high_positive"]["fixed_selectivity"]["exact_p"] <= 0.05
                and tests[corpus]["high_positive"]["fixed_selectivity"]["diagonal_is_best"]
                for corpus in corpus_specs
            ),
            "per_corpus": {
                corpus: tests[corpus]["high_positive"]["fixed_selectivity"]
                for corpus in corpus_specs
            },
        },
        "X4_natural_ablation_assignment": {
            "supported": all(
                tests[corpus]["high_positive"]["natural_absolute_class_effect"]["exact_p"] <= 0.05
                and tests[corpus]["high_positive"]["natural_absolute_class_effect"]["diagonal_is_best"]
                for corpus in corpus_specs
            ),
            "per_corpus": {
                corpus: tests[corpus]["high_positive"]["natural_absolute_class_effect"]
                for corpus in corpus_specs
            },
        },
        "X5_random_positive_assignment": {
            "supported": all(
                tests[corpus]["random_positive"]["fixed_absolute_class_effect"]["exact_p"] <= 0.05
                and tests[corpus]["random_positive"]["fixed_absolute_class_effect"]["diagonal_is_best"]
                for corpus in corpus_specs
            ),
            "per_corpus": {
                corpus: tests[corpus]["random_positive"]["fixed_absolute_class_effect"]
                for corpus in corpus_specs
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
        "# Layer-17 cross-category spectral red-team",
        "",
        "Each target class is tested against every category's rank-4 output-conditioned subspace. Fixed-norm edits are exactly 10% of the local MLP-input norm; natural ablations remove only the projection already present. Assignment tests ask whether the diagonal pairing is globally exceptional.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Cross-category matrices](layer17_cross_category.png)",
        "",
        "| Corpus | Eligible | High fixed p | Row wins | High selectivity p | Natural-ablation p | Random-positive p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpus_specs:
        high = tests[corpus]["high_positive"]
        random = tests[corpus]["random_positive"]
        lines.append(
            f"| {corpus} | {len(summaries[corpus]['high_positive']['eligible_categories'])} | "
            f"{high['fixed_absolute_class_effect']['exact_p']:.4f} | {high['fixed_row_wins']} | "
            f"{high['fixed_selectivity']['exact_p']:.4f} | "
            f"{high['natural_absolute_class_effect']['exact_p']:.4f} | "
            f"{random['fixed_absolute_class_effect']['exact_p']:.4f} |"
        )
    lines += ["", "Full matrices and per-site interventions are in `layer17_cross_category.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(corpus_specs)
    fig, axes = plt.subplots(1, len(corpora), figsize=(5 * len(corpora), 4.3))
    for axis, corpus in zip(axes, corpora):
        matrix = np.asarray(
            summaries[corpus]["high_positive"]["matrices"]["fixed_absolute_class_effect"]
        )
        image = axis.imshow(matrix, cmap="magma")
        labels = summaries[corpus]["high_positive"]["eligible_categories"]
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("edited source subspace")
        axis.set_ylabel("measured target contrast")
        axis.set_title(corpus)
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
