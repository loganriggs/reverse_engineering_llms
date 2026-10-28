#!/usr/bin/env python3
"""Separate semantic readout transfer from numerical mechanism reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_layer17_joint_forms import ARRAYS as JOINT_ARRAYS
from run_bilin18_layer17_output_spectra import OUTPUT as SPECTRAL_OUTPUT, prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_score_fidelity.json")
OUTPUT = TENSOR_ROOT / "layer17_score_fidelity.json"
REPORT = TENSOR_ROOT / "LAYER17_SCORE_FIDELITY.md"
FIGURE = TENSOR_ROOT / "layer17_score_fidelity.png"


def oriented_auc(labels, scores) -> float | None:
    value = auc(labels, scores)
    if not np.isfinite(value):
        return None
    return float(max(value, 1 - value))


def score_metrics(target, prediction) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target_centered = target - target.mean()
    prediction_centered = prediction - prediction.mean()
    energy = max(np.square(target_centered).sum(), 1e-30)
    correlation = float(
        np.dot(target_centered, prediction_centered)
        / max(np.linalg.norm(target_centered) * np.linalg.norm(prediction_centered), 1e-30)
    )
    centered_r2 = float(1 - np.square(prediction_centered - target_centered).sum() / energy)
    return {
        "correlation": correlation,
        "affine_r2": correlation ** 2,
        "centered_r2": centered_r2,
        "relative_centered_error": float(
            np.linalg.norm(prediction_centered - target_centered)
            / max(np.linalg.norm(target_centered), 1e-30)
        ),
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(SPECTRAL_OUTPUT.read_text())
    categories = config["original_categories"] + config["heldout_categories"]
    groups = {
        category: "original" if category in config["original_categories"] else "heldout"
        for category in categories
    }
    joint = np.load(JOINT_ARRAYS)
    shared = torch.from_numpy(
        joint["shared_vectors"][:, : config["shared_dimension"]]
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    unembedding = model.lm_head.weight.float()[: tokenizer.vocab_size]
    channel_coefficients = {}
    restricted_forms = {}
    for category in categories:
        positive = torch.as_tensor(
            flags[:, CATEGORIES.index(category)].astype(bool), device="cuda"
        )
        direction = unembedding[positive].mean(0) - unembedding[~positive].mean(0)
        direction /= direction.norm().clamp_min(1e-20)
        coefficients = down.T @ direction
        form = left.T @ (coefficients[:, None] * right)
        form = 0.5 * (form + form.T)
        channel_coefficients[category] = coefficients
        restricted_forms[category] = shared.T @ form @ shared

    positions = pilot["config"]["positions"]
    position_tensor = torch.as_tensor(positions, device="cuda")
    results = {}
    for corpus, corpus_config in pilot["config"]["corpora"].items():
        all_rows = np.load(corpus_config["path"], mmap_mode="r")
        rows = all_rows[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        values = {
            category: {"labels": [], "full": [], "shared": [], "diagonal": []}
            for category in categories
        }
        for start in range(0, len(rows), 4):
            raw = np.asarray(rows[start : start + 4, :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            _, normalized = prefix_to_layer17(model, ids)
            x = normalized[:, position_tensor].float()
            z = x @ shared
            products = block.mlp.Left(x) * block.mlp.Right(x)
            next_ids = raw[:, np.asarray(positions) + 1]
            for category in categories:
                form = restricted_forms[category]
                full = products @ channel_coefficients[category]
                shared_score = torch.einsum("...i,ij,...j->...", z, form, z)
                diagonal = z.square() @ form.diagonal()
                values[category]["labels"].extend(
                    flags[next_ids, CATEGORIES.index(category)].astype(bool).reshape(-1).tolist()
                )
                values[category]["full"].extend(full.flatten().cpu().tolist())
                values[category]["shared"].extend(shared_score.flatten().cpu().tolist())
                values[category]["diagonal"].extend(diagonal.flatten().cpu().tolist())
        results[corpus] = {}
        for category in categories:
            row = values[category]
            positive_count = int(np.sum(row["labels"]))
            full_auc = oriented_auc(row["labels"], row["full"])
            shared_auc = oriented_auc(row["labels"], row["shared"])
            diagonal_auc = oriented_auc(row["labels"], row["diagonal"])
            results[corpus][category] = {
                "group": groups[category],
                "positive_examples": positive_count,
                "eligible": positive_count >= config["minimum_positive_examples"],
                "full_oriented_auc": full_auc,
                "shared_oriented_auc": shared_auc,
                "diagonal_oriented_auc": diagonal_auc,
                "shared_score": score_metrics(row["full"], row["shared"]),
                "diagonal_score": score_metrics(row["full"], row["diagonal"]),
            }

    summary = {}
    for corpus, corpus_rows in results.items():
        summary[corpus] = {}
        for group in ("original", "heldout"):
            selected = [
                row for row in corpus_rows.values()
                if row["group"] == group and row["eligible"]
            ]
            retained = sum(
                row["shared_oriented_auc"] >= row["full_oriented_auc"] - 0.05
                for row in selected
            )
            summary[corpus][group] = {
                "eligible_concepts": len(selected),
                "median_full_oriented_auc": float(np.median([row["full_oriented_auc"] for row in selected])),
                "median_shared_oriented_auc": float(np.median([row["shared_oriented_auc"] for row in selected])),
                "median_diagonal_oriented_auc": float(np.median([row["diagonal_oriented_auc"] for row in selected])),
                "median_shared_correlation": float(np.median([row["shared_score"]["correlation"] for row in selected])),
                "median_diagonal_correlation": float(np.median([row["diagonal_score"]["correlation"] for row in selected])),
                "median_shared_centered_r2": float(np.median([row["shared_score"]["centered_r2"] for row in selected])),
                "semantic_retention_fraction": retained / max(len(selected), 1),
            }

    hypotheses = {
        "Y1_original_score_correlation": {
            "supported": all(row["original"]["median_shared_correlation"] >= 0.70 for row in summary.values()),
            "per_corpus": {c: row["original"]["median_shared_correlation"] for c, row in summary.items()},
        },
        "Y2_heldout_score_correlation": {
            "supported": all(row["heldout"]["median_shared_correlation"] >= 0.50 for row in summary.values()),
            "per_corpus": {c: row["heldout"]["median_shared_correlation"] for c, row in summary.items()},
        },
        "Y3_original_executable_fidelity": {
            "supported": all(row["original"]["median_shared_centered_r2"] >= 0.50 for row in summary.values()),
            "per_corpus": {c: row["original"]["median_shared_centered_r2"] for c, row in summary.items()},
        },
        "Y4_per_concept_semantic_retention": {
            "supported": all(row["heldout"]["semantic_retention_fraction"] >= 0.75 for row in summary.values()),
            "per_corpus": {c: row["heldout"]["semantic_retention_fraction"] for c, row in summary.items()},
        },
        "Y5_cross_term_fidelity_gain": {
            "supported": all(
                group_row["median_shared_correlation"] - group_row["median_diagonal_correlation"] >= 0.20
                for corpus_row in summary.values() for group_row in corpus_row.values()
            ),
            "per_corpus_group_gain": {
                c: {
                    group: row[group]["median_shared_correlation"] - row[group]["median_diagonal_correlation"]
                    for group in ("original", "heldout")
                }
                for c, row in summary.items()
            },
        },
    }
    output = {
        "config": config,
        "summary": summary,
        "results": results,
        "hypotheses": hypotheses,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Layer-17 shared-score fidelity red-team",
        "",
        "This separates semantic prediction from reconstruction. Oriented AUROC treats coherent suppression and promotion symmetrically; score correlation and centered R-squared ask whether the frozen 32D quadratic quotient numerically reproduces the full output-conditioned MLP contribution on language states.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Score fidelity](layer17_score_fidelity.png)",
        "",
        "| Corpus | Group | Full oriented AUROC | Shared AUROC | Score correlation | Centered R² | Semantic retention |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_rows in summary.items():
        for group, row in corpus_rows.items():
            lines.append(
                f"| {corpus} | {group} | {row['median_full_oriented_auc']:.3f} | "
                f"{row['median_shared_oriented_auc']:.3f} | {row['median_shared_correlation']:.3f} | "
                f"{row['median_shared_centered_r2']:.3f} | {row['semantic_retention_fraction']:.1%} |"
            )
    lines += ["", "Per-concept fidelity and semantic results are in `layer17_score_fidelity.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(summary)
    x = np.arange(len(corpora))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for group in ("original", "heldout"):
        axes[0].plot(x, [summary[c][group]["median_shared_correlation"] for c in corpora], marker="o", label=group)
        axes[1].plot(x, [summary[c][group]["semantic_retention_fraction"] for c in corpora], marker="o", label=group)
    axes[0].set_xticks(x, corpora)
    axes[0].set_ylabel("median shared/full score correlation")
    axes[1].set_xticks(x, corpora)
    axes[1].set_ylabel("fraction concepts retaining AUROC")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
