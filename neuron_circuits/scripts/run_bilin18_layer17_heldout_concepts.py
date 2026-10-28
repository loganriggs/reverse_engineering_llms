#!/usr/bin/env python3
"""Test whether the layer-17 joint quadratic basis transfers to unseen concepts."""

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


CONFIG = Path("configs/bilin18_layer17_heldout_concepts.json")
OUTPUT = TENSOR_ROOT / "layer17_heldout_concepts.json"
ARRAYS = TENSOR_ROOT / "layer17_heldout_concepts.npz"
REPORT = TENSOR_ROOT / "LAYER17_HELDOUT_CONCEPTS.md"
FIGURE = TENSOR_ROOT / "layer17_heldout_concepts.png"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(SPECTRAL_OUTPUT.read_text())
    joint = np.load(JOINT_ARRAYS)
    categories = config["heldout_categories"]
    frozen_shared = torch.from_numpy(joint["shared_vectors"]).cuda()
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

    forms = {}
    channel_coefficients = {}
    form_norms = {}
    for category in categories:
        positive = torch.as_tensor(
            flags[:, CATEGORIES.index(category)].astype(bool), device="cuda"
        )
        direction = unembedding[positive].mean(0) - unembedding[~positive].mean(0)
        direction /= direction.norm().clamp_min(1e-20)
        coefficients = down.T @ direction
        form = left.T @ (coefficients[:, None] * right)
        form = 0.5 * (form + form.T)
        forms[category] = form
        channel_coefficients[category] = coefficients
        form_norms[category] = form.norm()

    aggregate = torch.zeros_like(next(iter(forms.values())))
    for category in categories:
        normalized = forms[category] / form_norms[category]
        aggregate += normalized @ normalized
    eigenvalues, eigenvectors = torch.linalg.eigh(aggregate)
    heldout_shared = eigenvectors[:, torch.argsort(eigenvalues, descending=True)[:64]]

    frozen_restricted = {
        category: frozen_shared.T @ form @ frozen_shared
        for category, form in forms.items()
    }
    heldout_restricted = {
        category: heldout_shared.T @ form @ heldout_shared
        for category, form in forms.items()
    }
    weight_metrics = {}
    for category in categories:
        weight_metrics[category] = {}
        for dimension in config["shared_dimensions"]:
            frozen_form = frozen_restricted[category][:dimension, :dimension]
            heldout_form = heldout_restricted[category][:dimension, :dimension]
            weight_metrics[category][str(dimension)] = {
                "frozen_energy_fraction": float(
                    frozen_form.square().sum() / form_norms[category].square()
                ),
                "heldout_basis_energy_fraction": float(
                    heldout_form.square().sum() / form_norms[category].square()
                ),
                "frozen_cross_term_fraction": float(
                    1 - frozen_form.diagonal().square().sum()
                    / frozen_form.square().sum().clamp_min(1e-30)
                ),
            }

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
                "full": [],
                **{
                    str(dimension): {
                        "frozen": [],
                        "frozen_diagonal": [],
                        "heldout_basis": [],
                    }
                    for dimension in config["shared_dimensions"]
                },
            }
            for category in categories
        }
        for start in range(0, len(rows), 4):
            raw = np.asarray(rows[start : start + 4, :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            _, normalized = prefix_to_layer17(model, ids)
            x = normalized[:, position_tensor].float()
            frozen_z = x @ frozen_shared
            heldout_z = x @ heldout_shared
            products = block.mlp.Left(x) * block.mlp.Right(x)
            next_ids = raw[:, np.asarray(positions) + 1]
            for category in categories:
                values[category]["labels"].extend(
                    flags[next_ids, CATEGORIES.index(category)].astype(bool).reshape(-1).tolist()
                )
                values[category]["full"].extend(
                    (products @ channel_coefficients[category]).flatten().cpu().tolist()
                )
                for dimension in config["shared_dimensions"]:
                    fz = frozen_z[..., :dimension]
                    hz = heldout_z[..., :dimension]
                    frozen_form = frozen_restricted[category][:dimension, :dimension]
                    heldout_form = heldout_restricted[category][:dimension, :dimension]
                    frozen_score = torch.einsum("...i,ij,...j->...", fz, frozen_form, fz)
                    heldout_score = torch.einsum("...i,ij,...j->...", hz, heldout_form, hz)
                    diagonal_score = fz.square() @ frozen_form.diagonal()
                    values[category][str(dimension)]["frozen"].extend(
                        frozen_score.flatten().cpu().tolist()
                    )
                    values[category][str(dimension)]["frozen_diagonal"].extend(
                        diagonal_score.flatten().cpu().tolist()
                    )
                    values[category][str(dimension)]["heldout_basis"].extend(
                        heldout_score.flatten().cpu().tolist()
                    )
        results[corpus] = {}
        for category in categories:
            labels = values[category]["labels"]
            positive_count = int(np.sum(labels))
            def measured_auc(scores):
                value = auc(labels, scores)
                return value if np.isfinite(value) else None
            results[corpus][category] = {
                "positive_examples": positive_count,
                "eligible": positive_count >= config["minimum_positive_examples"],
                "full_auc": measured_auc(values[category]["full"]),
            }
            for dimension in config["shared_dimensions"]:
                row = values[category][str(dimension)]
                results[corpus][category][str(dimension)] = {
                    "frozen_auc": measured_auc(row["frozen"]),
                    "frozen_diagonal_auc": measured_auc(row["frozen_diagonal"]),
                    "heldout_basis_auc": measured_auc(row["heldout_basis"]),
                }

    summary = {}
    for corpus in results:
        eligible = [category for category in categories if results[corpus][category]["eligible"]]
        summary[corpus] = {
            "eligible_categories": eligible,
            "full_auc": float(np.median([results[corpus][c]["full_auc"] for c in eligible])),
        }
        for dimension in config["shared_dimensions"]:
            summary[corpus][str(dimension)] = {
                key: float(np.median([results[corpus][c][str(dimension)][key] for c in eligible]))
                for key in ("frozen_auc", "frozen_diagonal_auc", "heldout_basis_auc")
            }

    eligible_any = sorted(set().union(*[set(row["eligible_categories"]) for row in summary.values()]))
    energy32 = float(np.median([
        weight_metrics[category]["32"]["frozen_energy_fraction"]
        for category in eligible_any
    ]))
    hypotheses = {
        "H1_frozen_shared4_transfer": {
            "supported": all(
                row["4"]["frozen_auc"] >= row["full_auc"] - 0.05
                for row in summary.values()
            ),
            "per_corpus": summary,
        },
        "H2_frozen_shared32_transfer": {
            "supported": all(
                row["32"]["frozen_auc"] >= row["full_auc"] - 0.05
                for row in summary.values()
            ),
            "per_corpus": {
                c: {"full": row["full_auc"], "frozen32": row["32"]["frozen_auc"]}
                for c, row in summary.items()
            },
        },
        "H3_frozen_weight_coverage": {
            "supported": energy32 >= 0.15,
            "median_energy_fraction": energy32,
        },
        "H4_heldout_cross_terms": {
            "supported": sum(
                row["32"]["frozen_auc"] - row["32"]["frozen_diagonal_auc"] >= 0.05
                for row in summary.values()
            ) >= 2,
            "per_corpus_gain": {
                c: row["32"]["frozen_auc"] - row["32"]["frozen_diagonal_auc"]
                for c, row in summary.items()
            },
        },
        "H5_basis_universality": {
            "supported": all(
                row["4"]["frozen_auc"] >= row["4"]["heldout_basis_auc"] - 0.05
                for row in summary.values()
            ),
            "per_corpus": {
                c: {
                    "frozen4": row["4"]["frozen_auc"],
                    "heldout_basis4": row["4"]["heldout_basis_auc"],
                }
                for c, row in summary.items()
            },
        },
    }
    output = {
        "config": config,
        "weight_metrics": weight_metrics,
        "results": results,
        "summary": summary,
        "hypotheses": hypotheses,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        heldout_shared_vectors=heldout_shared.cpu().numpy(),
        **{
            f"{category}_frozen_restricted": frozen_restricted[category].cpu().numpy()
            for category in categories
        },
    )

    lines = [
        "# Layer-17 held-out-concept generalization",
        "",
        "The shared basis is frozen from newline, three suffix classes, and all-caps. Nine untouched output predicates are then evaluated. A separately constructed held-out-form basis is reported only as an oracle comparison; no corpus activations are used to construct either basis.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Held-out concepts](layer17_heldout_concepts.png)",
        "",
        "| Corpus | Eligible concepts | Full AUROC | Frozen 4D | Held-out-basis 4D | Frozen 32D | Diagonal-only 32D |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in summary.items():
        lines.append(
            f"| {corpus} | {len(row['eligible_categories'])} | {row['full_auc']:.3f} | "
            f"{row['4']['frozen_auc']:.3f} | {row['4']['heldout_basis_auc']:.3f} | "
            f"{row['32']['frozen_auc']:.3f} | {row['32']['frozen_diagonal_auc']:.3f} |"
        )
    lines += ["", "Per-concept counts, AUROCs, and weight-space coverage are in `layer17_heldout_concepts.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    dims = config["shared_dimensions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for corpus, row in summary.items():
        axes[0].plot(dims, [row[str(d)]["frozen_auc"] for d in dims], marker="o", label=corpus)
        axes[1].plot(dims, [row[str(d)]["frozen_diagonal_auc"] for d in dims], marker="o", label=corpus)
    axes[0].set_title("Frozen full quadratic forms")
    axes[1].set_title("Frozen diagonal-only forms")
    for axis in axes:
        axis.set_xlabel("shared dimension")
        axis.set_ylabel("median held-out-concept AUROC")
        axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
