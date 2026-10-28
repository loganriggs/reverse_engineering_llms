#!/usr/bin/env python3
"""Compare embedding- and unembedding-conditioned joint quadratic codes at layer 0."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_layer17_output_spectra import OUTPUT as SPECTRAL_OUTPUT
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer0_joint_forms.json")
OUTPUT = TENSOR_ROOT / "layer0_joint_forms.json"
ARRAYS = TENSOR_ROOT / "layer0_joint_forms.npz"
REPORT = TENSOR_ROOT / "LAYER0_JOINT_FORMS.md"
FIGURE = TENSOR_ROOT / "layer0_joint_forms.png"


def score_metrics(target, prediction) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target -= target.mean()
    prediction -= prediction.mean()
    energy = max(np.square(target).sum(), 1e-30)
    correlation = float(
        np.dot(target, prediction)
        / max(np.linalg.norm(target) * np.linalg.norm(prediction), 1e-30)
    )
    return {
        "correlation": correlation,
        "centered_r2": float(1 - np.square(prediction - target).sum() / energy),
    }


@torch.inference_mode()
def layer0_input(model, ids: torch.Tensor) -> torch.Tensor:
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    block = model.blocks[0]
    mixed = block.lambdas[0] * x0 + block.lambdas[1] * x0
    attention, _ = block.attn(F.rms_norm(mixed, (dimension,)), None)
    return F.rms_norm(mixed + attention, (dimension,))


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    corpus_specs = json.loads(SPECTRAL_OUTPUT.read_text())["config"]["corpora"]
    categories = config["categories"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[0]
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    down = block.mlp.Down.weight.float()
    vocab_matrices = {
        "embedding": model.embedding.weight.float()[: tokenizer.vocab_size],
        "unembedding": model.lm_head.weight.float()[: tokenizer.vocab_size],
    }

    forms = {readout: {} for readout in config["readouts"]}
    channel_coefficients = {readout: {} for readout in config["readouts"]}
    for readout, vocabulary in vocab_matrices.items():
        for category in categories:
            positive = torch.as_tensor(
                flags[:, CATEGORIES.index(category)].astype(bool), device="cuda"
            )
            direction = vocabulary[positive].mean(0) - vocabulary[~positive].mean(0)
            direction /= direction.norm().clamp_min(1e-20)
            coefficients = down.T @ direction
            form = left.T @ (coefficients[:, None] * right)
            form = 0.5 * (form + form.T)
            forms[readout][category] = form
            channel_coefficients[readout][category] = coefficients

    geometry = {}
    shared_vectors = {}
    restricted_forms = {}
    for readout in config["readouts"]:
        norms = {category: forms[readout][category].norm() for category in categories}
        normalized = {category: forms[readout][category] / norms[category] for category in categories}
        gram = np.asarray([
            [float((normalized[first] * normalized[second]).sum()) for second in categories]
            for first in categories
        ])
        commutator = np.zeros_like(gram)
        overlap = np.zeros_like(gram)
        eigenvectors = {}
        for category in categories:
            eigenvalues, vectors = torch.linalg.eigh(forms[readout][category])
            eigenvectors[category] = vectors[:, torch.argsort(eigenvalues.abs(), descending=True)]
        for i, first in enumerate(categories):
            for j, second in enumerate(categories):
                if i < j:
                    value = float(
                        (normalized[first] @ normalized[second]
                         - normalized[second] @ normalized[first]).norm()
                    )
                    commutator[i, j] = commutator[j, i] = value
                overlap[i, j] = float(
                    (eigenvectors[first][:, :4].T @ eigenvectors[second][:, :4]).square().sum() / 4
                )
        aggregate = torch.zeros_like(next(iter(forms[readout].values())))
        for category in categories:
            aggregate += normalized[category] @ normalized[category]
        aggregate_values, aggregate_basis = torch.linalg.eigh(aggregate)
        shared = aggregate_basis[:, torch.argsort(aggregate_values, descending=True)[: config["shared_dimension"]]]
        shared_vectors[readout] = shared
        restricted_forms[readout] = {
            category: shared.T @ forms[readout][category] @ shared
            for category in categories
        }
        mask = ~np.eye(len(categories), dtype=bool)
        gram_eigenvalues = np.linalg.eigvalsh(gram)[::-1]
        geometry[readout] = {
            "form_cosine": gram.tolist(),
            "gram_eigenvalues": gram_eigenvalues.tolist(),
            "gram_effective_rank": float(
                gram_eigenvalues.sum() ** 2 / np.square(gram_eigenvalues).sum()
            ),
            "normalized_commutator": commutator.tolist(),
            "median_normalized_commutator": float(np.median(commutator[mask])),
            "rank4_overlap": overlap.tolist(),
            "median_rank4_overlap": float(np.median(overlap[mask])),
            "median_shared_energy_fraction": float(np.median([
                float(restricted_forms[readout][category].square().sum() / norms[category].square())
                for category in categories
            ])),
            "median_cross_term_fraction": float(np.median([
                float(
                    1 - restricted_forms[readout][category].diagonal().square().sum()
                    / restricted_forms[readout][category].square().sum().clamp_min(1e-30)
                )
                for category in categories
            ])),
        }

    same_category_form_cosines = {
        category: float(
            (forms["embedding"][category] * forms["unembedding"][category]).sum()
            / (forms["embedding"][category].norm() * forms["unembedding"][category].norm()).clamp_min(1e-30)
        )
        for category in categories
    }

    positions = json.loads(SPECTRAL_OUTPUT.read_text())["config"]["positions"]
    position_tensor = torch.as_tensor(positions, device="cuda")
    results = {}
    for corpus, corpus_config in corpus_specs.items():
        all_rows = np.load(corpus_config["path"], mmap_mode="r")
        rows = all_rows[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        values = {
            readout: {
                category: {"labels": [], "full": [], "shared": [], "diagonal": []}
                for category in categories
            }
            for readout in config["readouts"]
        }
        for start in range(0, len(rows), 8):
            raw = np.asarray(rows[start : start + 8, :128], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda")
            x = layer0_input(model, ids)[:, position_tensor].float()
            products = block.mlp.Left(x) * block.mlp.Right(x)
            current_ids = raw[:, positions]
            for readout in config["readouts"]:
                z = x @ shared_vectors[readout]
                for category in categories:
                    labels = flags[current_ids, CATEGORIES.index(category)].astype(bool)
                    full = products @ channel_coefficients[readout][category]
                    form = restricted_forms[readout][category]
                    shared_score = torch.einsum("...i,ij,...j->...", z, form, z)
                    diagonal = z.square() @ form.diagonal()
                    row = values[readout][category]
                    row["labels"].extend(labels.reshape(-1).tolist())
                    row["full"].extend(full.flatten().cpu().tolist())
                    row["shared"].extend(shared_score.flatten().cpu().tolist())
                    row["diagonal"].extend(diagonal.flatten().cpu().tolist())
        results[corpus] = {}
        for readout in config["readouts"]:
            results[corpus][readout] = {}
            for category in categories:
                row = values[readout][category]
                results[corpus][readout][category] = {
                    "positive_examples": int(np.sum(row["labels"])),
                    "full_auc": auc(row["labels"], row["full"]),
                    "shared_auc": auc(row["labels"], row["shared"]),
                    "diagonal_auc": auc(row["labels"], row["diagonal"]),
                    "shared_score": score_metrics(row["full"], row["shared"]),
                    "diagonal_score": score_metrics(row["full"], row["diagonal"]),
                }

    summary = {}
    for corpus, corpus_rows in results.items():
        summary[corpus] = {}
        for readout in config["readouts"]:
            selected = list(corpus_rows[readout].values())
            summary[corpus][readout] = {
                key: float(np.median([row[key] for row in selected]))
                for key in ("full_auc", "shared_auc", "diagonal_auc")
            }
            summary[corpus][readout].update({
                "shared_correlation": float(np.median([row["shared_score"]["correlation"] for row in selected])),
                "shared_centered_r2": float(np.median([row["shared_score"]["centered_r2"] for row in selected])),
                "diagonal_correlation": float(np.median([row["diagonal_score"]["correlation"] for row in selected])),
            })

    median_same_readout_cosine = float(np.median(list(same_category_form_cosines.values())))
    hypotheses = {
        "Z1_embedding_semantics": {
            "supported": all(row["embedding"]["full_auc"] >= 0.70 for row in summary.values()),
            "per_corpus": {c: row["embedding"]["full_auc"] for c, row in summary.items()},
        },
        "Z2_early_joint_diagonalization": {
            "supported": geometry["embedding"]["median_normalized_commutator"] < 0.04,
            "median_commutator": geometry["embedding"]["median_normalized_commutator"],
            "layer17_reference": 0.05909577012062073,
        },
        "Z3_embedding_diagonal_sufficiency": {
            "supported": all(
                row["embedding"]["diagonal_auc"] >= row["embedding"]["shared_auc"] - 0.05
                for row in summary.values()
            ),
            "per_corpus": {
                c: {"shared": row["embedding"]["shared_auc"], "diagonal": row["embedding"]["diagonal_auc"]}
                for c, row in summary.items()
            },
        },
        "Z4_embedding_score_fidelity": {
            "supported": all(row["embedding"]["shared_correlation"] >= 0.70 for row in summary.values()),
            "per_corpus": {c: row["embedding"]["shared_correlation"] for c, row in summary.items()},
        },
        "Z5_readout_dependence": {
            "supported": median_same_readout_cosine < 0.80,
            "median_same_category_form_cosine": median_same_readout_cosine,
            "per_category": same_category_form_cosines,
        },
    }
    output = {
        "config": config,
        "geometry": geometry,
        "same_category_form_cosines": same_category_form_cosines,
        "summary": summary,
        "results": results,
        "hypotheses": hypotheses,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        **{f"{readout}_shared_vectors": shared_vectors[readout].cpu().numpy() for readout in config["readouts"]},
        **{
            f"{readout}_{category}_restricted_form": restricted_forms[readout][category].cpu().numpy()
            for readout in config["readouts"] for category in categories
        },
    )

    lines = [
        "# Layer-0 joint quadratic forms under two readout definitions",
        "",
        "The exact layer-0 bilinear tensor is conditioned either on token-embedding class contrasts (an identity-facing residual query) or on unembedding contrasts (a logit-facing query). Current-token labels are evaluated on all three corpora.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Layer-0 joint forms](layer0_joint_forms.png)",
        "",
        "| Readout | Form effective rank | Median commutator | Rank-4 overlap | Shared-32 energy | Cross-term fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for readout, row in geometry.items():
        lines.append(
            f"| {readout} | {row['gram_effective_rank']:.2f} | "
            f"{row['median_normalized_commutator']:.3f} | {row['median_rank4_overlap']:.3f} | "
            f"{row['median_shared_energy_fraction']:.3f} | {row['median_cross_term_fraction']:.3f} |"
        )
    lines += [
        "",
        "| Corpus | Readout | Full AUROC | Shared-32 AUROC | Diagonal AUROC | Score correlation | Centered R² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_rows in summary.items():
        for readout, row in corpus_rows.items():
            lines.append(
                f"| {corpus} | {readout} | {row['full_auc']:.3f} | {row['shared_auc']:.3f} | "
                f"{row['diagonal_auc']:.3f} | {row['shared_correlation']:.3f} | {row['shared_centered_r2']:.3f} |"
            )
    lines += ["", "Per-category results and both form families are in `layer0_joint_forms.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(summary)
    x = np.arange(len(corpora))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for readout in config["readouts"]:
        axes[0].plot(x, [summary[c][readout]["full_auc"] for c in corpora], marker="o", label=f"{readout} full")
        axes[0].plot(x, [summary[c][readout]["diagonal_auc"] for c in corpora], marker="x", linestyle="--", label=f"{readout} diagonal")
        axes[1].plot(x, [summary[c][readout]["shared_correlation"] for c in corpora], marker="o", label=readout)
    axes[0].set_xticks(x, corpora)
    axes[0].set_ylabel("median current-token AUROC")
    axes[1].set_xticks(x, corpora)
    axes[1].set_ylabel("shared/full score correlation")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
