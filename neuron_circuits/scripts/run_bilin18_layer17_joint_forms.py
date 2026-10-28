#!/usr/bin/env python3
"""Analyze final-layer output classes as a joint family of quadratic forms."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_layer17_output_spectra import OUTPUT as SPECTRAL_OUTPUT, prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_joint_forms.json")
OUTPUT = TENSOR_ROOT / "layer17_joint_forms.json"
ARRAYS = TENSOR_ROOT / "layer17_joint_forms.npz"
REPORT = TENSOR_ROOT / "LAYER17_JOINT_FORMS.md"
FIGURE = TENSOR_ROOT / "layer17_joint_forms.png"


def effective_rank(eigenvalues: np.ndarray) -> float:
    eigenvalues = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0, None)
    return float(eigenvalues.sum() ** 2 / max(np.square(eigenvalues).sum(), 1e-30))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(SPECTRAL_OUTPUT.read_text())
    categories = config["categories"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    flags = np.stack([token_flags(text) for text in texts])

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    unembedding = model.lm_head.weight.float()[: tokenizer.vocab_size]

    forms = []
    channel_coefficients = []
    for category in categories:
        positive = torch.as_tensor(
            flags[:, CATEGORIES.index(category)].astype(bool), device="cuda"
        )
        direction = unembedding[positive].mean(0) - unembedding[~positive].mean(0)
        direction /= direction.norm().clamp_min(1e-20)
        coefficients = down.T @ direction
        form = left.T @ (coefficients[:, None] * right)
        form = 0.5 * (form + form.T)
        forms.append(form)
        channel_coefficients.append(coefficients)
    form_norms = torch.stack([form.square().sum().sqrt() for form in forms])
    normalized_forms = [form / norm for form, norm in zip(forms, form_norms)]

    gram = torch.stack([
        torch.stack([(first * second).sum() for second in normalized_forms])
        for first in normalized_forms
    ])
    gram_eigenvalues = torch.linalg.eigvalsh(gram).cpu().numpy()[::-1]
    form_effective_rank = effective_rank(gram_eigenvalues)
    form_cosine = gram.cpu().numpy()

    commutator = np.zeros((len(categories), len(categories)))
    for i, first in enumerate(normalized_forms):
        for j, second in enumerate(normalized_forms):
            if i < j:
                value = float((first @ second - second @ first).norm())
                commutator[i, j] = commutator[j, i] = value
    off_diagonal = ~np.eye(len(categories), dtype=bool)
    median_commutator = float(np.median(commutator[off_diagonal]))

    individual_vectors = []
    for form in forms:
        eigenvalues, eigenvectors = torch.linalg.eigh(form)
        order = torch.argsort(eigenvalues.abs(), descending=True)
        individual_vectors.append(eigenvectors[:, order])
    rank4_overlap = np.zeros((len(categories), len(categories)))
    for i, first in enumerate(individual_vectors):
        for j, second in enumerate(individual_vectors):
            rank4_overlap[i, j] = float(
                (first[:, :4].T @ second[:, :4]).square().sum() / 4
            )
    median_rank4_overlap = float(np.median(rank4_overlap[off_diagonal]))
    random_overlap = 4 / forms[0].shape[0]

    aggregate = torch.zeros_like(forms[0])
    for form in normalized_forms:
        aggregate += form @ form
    aggregate_eigenvalues, aggregate_vectors = torch.linalg.eigh(aggregate)
    order = torch.argsort(aggregate_eigenvalues, descending=True)
    aggregate_eigenvalues = aggregate_eigenvalues[order]
    aggregate_vectors = aggregate_vectors[:, order]
    maximum_dimension = max(config["shared_dimensions"])
    shared = aggregate_vectors[:, :maximum_dimension]
    restricted = {
        category: shared.T @ form @ shared
        for category, form in zip(categories, forms)
    }
    subspace_metrics = {}
    for dimension in config["shared_dimensions"]:
        rows = {}
        for category, form, norm in zip(categories, forms, form_norms):
            coefficient = restricted[category][:dimension, :dimension]
            restricted_energy = float(coefficient.square().sum() / norm.square())
            diagonal_energy = float(
                coefficient.diagonal().square().sum()
                / coefficient.square().sum().clamp_min(1e-30)
            )
            rows[category] = {
                "full_form_energy_fraction": restricted_energy,
                "diagonal_fraction_in_shared_basis": diagonal_energy,
                "cross_term_fraction_in_shared_basis": 1 - diagonal_energy,
            }
        subspace_metrics[str(dimension)] = rows

    semantic = {}
    positions = pilot["config"]["positions"]
    position_tensor = torch.as_tensor(positions, device="cuda")
    for corpus, corpus_config in pilot["config"]["corpora"].items():
        all_rows = np.load(corpus_config["path"], mmap_mode="r")
        rows = all_rows[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        values = {
            category: {"labels": [], "full": [], **{str(d): [] for d in config["shared_dimensions"]}}
            for category in categories
        }
        for start in range(0, len(rows), 4):
            raw = np.asarray(rows[start : start + 4, :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            _, normalized = prefix_to_layer17(model, ids)
            x = normalized[:, position_tensor].float()
            z = x @ shared
            next_ids = raw[:, np.asarray(positions) + 1]
            products = block.mlp.Left(x) * block.mlp.Right(x)
            for category, coefficients in zip(categories, channel_coefficients):
                values[category]["labels"].extend(
                    flags[next_ids, CATEGORIES.index(category)].astype(bool).reshape(-1).tolist()
                )
                values[category]["full"].extend((products @ coefficients).flatten().cpu().tolist())
                for dimension in config["shared_dimensions"]:
                    current_z = z[..., :dimension]
                    current_form = restricted[category][:dimension, :dimension]
                    score = torch.einsum("...i,ij,...j->...", current_z, current_form, current_z)
                    values[category][str(dimension)].extend(score.flatten().cpu().tolist())
        semantic[corpus] = {}
        for category in categories:
            labels = values[category]["labels"]
            semantic[corpus][category] = {
                "full_auc": auc(labels, values[category]["full"]),
                **{
                    f"shared_{dimension}_auc": auc(labels, values[category][str(dimension)])
                    for dimension in config["shared_dimensions"]
                },
            }

    # Keep only cells with enough positive next-token examples for stable AUROC.
    semantic_summary = {}
    for corpus in semantic:
        eligible = [category for category in categories if pilot["scans"][corpus][category]["eligible"]]
        semantic_summary[corpus] = {
            "full": float(np.median([semantic[corpus][category]["full_auc"] for category in eligible])),
            **{
                f"shared_{dimension}": float(np.median([
                    semantic[corpus][category][f"shared_{dimension}_auc"]
                    for category in eligible
                ]))
                for dimension in config["shared_dimensions"]
            },
        }

    energy32 = float(np.median([
        row["full_form_energy_fraction"] for row in subspace_metrics["32"].values()
    ]))
    cross32 = float(np.median([
        row["cross_term_fraction_in_shared_basis"] for row in subspace_metrics["32"].values()
    ]))
    hypotheses = {
        "J1_form_independence": {
            "supported": form_effective_rank >= 4,
            "effective_rank": form_effective_rank,
            "gram_eigenvalues": gram_eigenvalues.tolist(),
        },
        "J2_shared_input_geometry": {
            "supported": median_rank4_overlap >= 10 * random_overlap,
            "median_overlap": median_rank4_overlap,
            "random_expectation": random_overlap,
            "multiple": median_rank4_overlap / random_overlap,
        },
        "J3_noncommuting_forms": {
            "supported": median_commutator >= 0.05,
            "median_normalized_commutator": median_commutator,
        },
        "J4_compact_shared_subspace": {
            "supported": energy32 >= 0.20,
            "median_energy_fraction": energy32,
        },
        "J5_semantic_retention": {
            "supported": all(
                row["shared_32"] >= row["full"] - 0.05
                for row in semantic_summary.values()
            ),
            "per_corpus": semantic_summary,
        },
        "J6_cross_terms_required": {
            "supported": cross32 >= 0.25,
            "median_cross_term_fraction": cross32,
        },
    }
    output = {
        "config": config,
        "form_cosine": form_cosine.tolist(),
        "form_gram_eigenvalues": gram_eigenvalues.tolist(),
        "form_effective_rank": form_effective_rank,
        "normalized_commutator": commutator.tolist(),
        "rank4_subspace_overlap": rank4_overlap.tolist(),
        "subspace_metrics": subspace_metrics,
        "semantic": semantic,
        "semantic_summary": semantic_summary,
        "hypotheses": hypotheses,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        shared_vectors=shared.cpu().numpy(),
        aggregate_eigenvalues=aggregate_eigenvalues[:maximum_dimension].cpu().numpy(),
        **{f"{category}_restricted_form": restricted[category].cpu().numpy() for category in categories},
    )

    lines = [
        "# Layer-17 joint quadratic-form analysis",
        "",
        "The five output classes are treated as a family of quadratic forms rather than as owners of separate eigenspaces. A shared basis is learned from the leading eigenspace of the sum of normalized squared forms; every class retains its own coefficient matrix inside that basis.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Joint forms](layer17_joint_forms.png)",
        "",
        f"Form Gram effective rank: **{form_effective_rank:.2f}**. Median normalized commutator: **{median_commutator:.3f}**. Rank-4 subspace overlap is **{median_rank4_overlap:.3f}**, versus random expectation **{random_overlap:.4f}**.",
        "",
        "| Shared dimension | Median form energy | Median cross-term fraction | FineWeb AUROC | WikiText AUROC | TinyStories AUROC |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for dimension in config["shared_dimensions"]:
        metric_rows = subspace_metrics[str(dimension)].values()
        energy = np.median([row["full_form_energy_fraction"] for row in metric_rows])
        cross = np.median([row["cross_term_fraction_in_shared_basis"] for row in subspace_metrics[str(dimension)].values()])
        lines.append(
            f"| {dimension} | {energy:.3f} | {cross:.3f} | "
            f"{semantic_summary['fineweb'][f'shared_{dimension}']:.3f} | "
            f"{semantic_summary['wikitext103'][f'shared_{dimension}']:.3f} | "
            f"{semantic_summary['tinystories'][f'shared_{dimension}']:.3f} |"
        )
    lines += ["", "Matrices, per-category energy, and AUROCs are in `layer17_joint_forms.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    dims = config["shared_dimensions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(
        dims,
        [np.median([row["full_form_energy_fraction"] for row in subspace_metrics[str(d)].values()]) for d in dims],
        marker="o",
        label="form energy",
    )
    axes[0].plot(
        dims,
        [np.median([row["cross_term_fraction_in_shared_basis"] for row in subspace_metrics[str(d)].values()]) for d in dims],
        marker="o",
        label="cross-term fraction",
    )
    axes[0].set_xlabel("shared input dimension")
    axes[0].legend()
    for corpus in semantic_summary:
        axes[1].plot(
            dims,
            [semantic_summary[corpus][f"shared_{d}"] for d in dims],
            marker="o",
            label=corpus,
        )
    axes[1].set_xlabel("shared input dimension")
    axes[1].set_ylabel("median next-token AUROC")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
