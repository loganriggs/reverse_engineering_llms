#!/usr/bin/env python3
"""Test whether joint cross terms matter on language states in the shared basis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer17_joint_forms import ARRAYS as JOINT_ARRAYS
from run_bilin18_layer17_output_spectra import OUTPUT as SPECTRAL_OUTPUT, prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from transformers import AutoTokenizer


CONFIG = Path("configs/bilin18_layer17_diagonal_redteam.json")
OUTPUT = TENSOR_ROOT / "layer17_diagonal_redteam.json"
REPORT = TENSOR_ROOT / "LAYER17_DIAGONAL_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer17_diagonal_redteam.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(SPECTRAL_OUTPUT.read_text())
    arrays = np.load(JOINT_ARRAYS)
    categories = config["categories"]
    shared = torch.from_numpy(arrays["shared_vectors"]).cuda()
    forms = {
        category: torch.from_numpy(arrays[f"{category}_restricted_form"]).cuda()
        for category in categories
    }
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    positions = pilot["config"]["positions"]
    position_tensor = torch.as_tensor(positions, device="cuda")
    results = {}

    for corpus, corpus_config in pilot["config"]["corpora"].items():
        all_rows = np.load(corpus_config["path"], mmap_mode="r")
        rows = all_rows[
            corpus_config["start"] : corpus_config["start"] + corpus_config["rows"]
        ]
        values = {
            category: {
                "labels": [],
                **{
                    str(dimension): {"full": [], "diagonal": [], "cross": []}
                    for dimension in config["shared_dimensions"]
                },
            }
            for category in categories
        }
        for start in range(0, len(rows), 4):
            raw = np.asarray(rows[start : start + 4, :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            _, normalized = prefix_to_layer17(model, ids)
            z = normalized[:, position_tensor].float() @ shared
            next_ids = raw[:, np.asarray(positions) + 1]
            for category in categories:
                values[category]["labels"].extend(
                    flags[next_ids, CATEGORIES.index(category)].astype(bool).reshape(-1).tolist()
                )
                for dimension in config["shared_dimensions"]:
                    current_z = z[..., :dimension]
                    form = forms[category][:dimension, :dimension]
                    full = torch.einsum("...i,ij,...j->...", current_z, form, current_z)
                    diagonal = current_z.square() @ form.diagonal()
                    values[category][str(dimension)]["full"].extend(full.flatten().cpu().tolist())
                    values[category][str(dimension)]["diagonal"].extend(diagonal.flatten().cpu().tolist())
                    values[category][str(dimension)]["cross"].extend((full - diagonal).flatten().cpu().tolist())
        results[corpus] = {}
        for category in categories:
            labels = values[category]["labels"]
            results[corpus][category] = {}
            for dimension in config["shared_dimensions"]:
                row = values[category][str(dimension)]
                full_values = np.asarray(row["full"])
                diagonal_values = np.asarray(row["diagonal"])
                cross_values = np.asarray(row["cross"])
                results[corpus][category][str(dimension)] = {
                    "full_auc": auc(labels, full_values),
                    "diagonal_auc": auc(labels, diagonal_values),
                    "cross_auc": auc(labels, cross_values),
                    "cross_variance_fraction": float(
                        np.var(cross_values) / max(np.var(full_values), 1e-30)
                    ),
                    "diagonal_full_correlation": float(
                        np.corrcoef(diagonal_values, full_values)[0, 1]
                    ),
                }

    summary = {}
    for corpus in results:
        eligible = [category for category in categories if pilot["scans"][corpus][category]["eligible"]]
        summary[corpus] = {}
        for dimension in config["shared_dimensions"]:
            selected = [results[corpus][category][str(dimension)] for category in eligible]
            summary[corpus][str(dimension)] = {
                key: float(np.median([row[key] for row in selected]))
                for key in (
                    "full_auc",
                    "diagonal_auc",
                    "cross_auc",
                    "cross_variance_fraction",
                    "diagonal_full_correlation",
                )
            }

    hypotheses = {
        "Q1_diagonal4_sufficiency": {
            "supported": all(
                row["4"]["diagonal_auc"] >= row["4"]["full_auc"] - 0.05
                for row in summary.values()
            ),
            "per_corpus": {c: row["4"] for c, row in summary.items()},
        },
        "Q2_diagonal32_sufficiency": {
            "supported": all(
                row["32"]["diagonal_auc"] >= row["32"]["full_auc"] - 0.05
                for row in summary.values()
            ),
            "per_corpus": {c: row["32"] for c, row in summary.items()},
        },
        "Q3_cross_term_state_variance": {
            "supported": all(
                row["32"]["cross_variance_fraction"] >= 0.25
                for row in summary.values()
            ),
            "per_corpus": {
                c: row["32"]["cross_variance_fraction"] for c, row in summary.items()
            },
        },
        "Q4_cross_term_semantic_gain": {
            "supported": sum(
                row["32"]["full_auc"] - row["32"]["diagonal_auc"] >= 0.02
                for row in summary.values()
            ) >= 2,
            "per_corpus_gain": {
                c: row["32"]["full_auc"] - row["32"]["diagonal_auc"]
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
        "# Layer-17 diagonal-only shared-basis red-team",
        "",
        "The shared joint-form basis is frozen. This test deletes every off-diagonal coefficient, leaving only squared scalar coordinates, and asks whether language-state semantic prediction survives.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Diagonal red-team](layer17_diagonal_redteam.png)",
        "",
        "| Corpus | Dim | Full AUROC | Diagonal AUROC | Cross-only AUROC | Cross variance / full | Diagonal/full correlation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in summary:
        for dimension in config["shared_dimensions"]:
            row = summary[corpus][str(dimension)]
            lines.append(
                f"| {corpus} | {dimension} | {row['full_auc']:.3f} | "
                f"{row['diagonal_auc']:.3f} | {row['cross_auc']:.3f} | "
                f"{row['cross_variance_fraction']:.3f} | "
                f"{row['diagonal_full_correlation']:.3f} |"
            )
    lines += ["", "Per-category results are in `layer17_diagonal_redteam.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    dims = config["shared_dimensions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for corpus in summary:
        axes[0].plot(dims, [summary[corpus][str(d)]["full_auc"] for d in dims], marker="o", label=f"{corpus} full")
        axes[0].plot(dims, [summary[corpus][str(d)]["diagonal_auc"] for d in dims], marker="x", linestyle="--", label=f"{corpus} diagonal")
        axes[1].plot(dims, [summary[corpus][str(d)]["cross_variance_fraction"] for d in dims], marker="o", label=corpus)
    axes[0].set_xlabel("shared dimension")
    axes[0].set_ylabel("median AUROC")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("shared dimension")
    axes[1].set_ylabel("cross-term variance / full variance")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
