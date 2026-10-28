#!/usr/bin/env python3
"""Weight-only approximate joint diagonalization of layer-0 and layer-17 form families."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_layer0_joint_forms import ARRAYS as LAYER0_ARRAYS, layer0_input
from run_bilin18_layer17_heldout_concepts import ARRAYS as HELDOUT_ARRAYS
from run_bilin18_layer17_joint_forms import ARRAYS as LAYER17_ARRAYS
from run_bilin18_layer17_output_spectra import OUTPUT as SPECTRAL_OUTPUT, prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_joint_diagonalization.json")
OUTPUT = TENSOR_ROOT / "joint_diagonalization.json"
ARRAYS = TENSOR_ROOT / "joint_diagonalization.npz"
REPORT = TENSOR_ROOT / "JOINT_DIAGONALIZATION.md"
FIGURE = TENSOR_ROOT / "joint_diagonalization.png"

ORIGINAL = ["newline", "suffix_ing", "suffix_ed", "suffix_ly", "all_caps"]
HELDOUT = ["opening_delimiter", "closing_delimiter", "quote", "code_operator", "url_web", "non_ascii", "word_continuation", "comma_period", "colon_semicolon"]


def score_metrics(target, prediction) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    target -= target.mean()
    prediction -= prediction.mean()
    correlation = float(
        np.dot(target, prediction)
        / max(np.linalg.norm(target) * np.linalg.norm(prediction), 1e-30)
    )
    return {
        "correlation": correlation,
        "affine_r2": correlation ** 2,
        "centered_r2": float(
            1 - np.square(prediction - target).sum()
            / max(np.square(target).sum(), 1e-30)
        ),
    }


def train_rotation(
    forms: torch.Tensor,
    config: dict,
    initial_raw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    forms = forms.double()
    forms = forms / forms.square().sum((1, 2), keepdim=True).sqrt().clamp_min(1e-30)
    dimension = forms.shape[-1]
    if initial_raw is None:
        initial_raw = torch.zeros(dimension, dimension, device="cuda", dtype=torch.float64)
    raw = initial_raw.to(device="cuda", dtype=torch.float64).clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=config["learning_rate"])
    identity = torch.eye(dimension, device="cuda", dtype=torch.float64)

    def rotation():
        skew = raw - raw.T
        return torch.linalg.solve(identity + skew, identity - skew)

    history = []
    with torch.no_grad():
        initial = float(
            (forms - torch.diag_embed(forms.diagonal(dim1=-2, dim2=-1))).square().sum()
            / forms.square().sum()
        )
    for step in range(1, config["steps"] + 1):
        q = rotation()
        rotated = q.T[None] @ forms @ q[None]
        off = rotated - torch.diag_embed(rotated.diagonal(dim1=-2, dim2=-1))
        loss = off.square().sum() / rotated.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            history.append({"step": step, "off_diagonal_fraction": float(loss.detach())})
    with torch.no_grad():
        q = rotation().float()
        rotated = q.T[None] @ forms.float() @ q[None]
        off = rotated - torch.diag_embed(rotated.diagonal(dim1=-2, dim2=-1))
        final = float(off.square().sum() / rotated.square().sum())
        orthogonality_error = float((q.T @ q - torch.eye(dimension, device="cuda")).abs().max())
    return q, {
        "initial_off_diagonal_fraction": initial,
        "final_off_diagonal_fraction": final,
        "relative_reduction": 1 - final / initial,
        "orthogonality_max_error": orthogonality_error,
        "history": history,
    }


def main() -> None:
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(SPECTRAL_OUTPUT.read_text())
    layer0 = np.load(LAYER0_ARRAYS)
    layer17 = np.load(LAYER17_ARRAYS)
    heldout = np.load(HELDOUT_ARRAYS)
    dimension = config["dimension"]

    layer0_shared = torch.from_numpy(layer0["embedding_shared_vectors"][:, :dimension]).cuda()
    layer17_shared = torch.from_numpy(layer17["shared_vectors"][:, :dimension]).cuda()
    layer0_forms = torch.stack([
        torch.from_numpy(layer0[f"embedding_{category}_restricted_form"][:dimension, :dimension]).cuda()
        for category in ORIGINAL
    ])
    layer17_forms = torch.stack([
        torch.from_numpy(layer17[f"{category}_restricted_form"][:dimension, :dimension]).cuda()
        for category in ORIGINAL
    ])
    heldout_forms = {
        category: torch.from_numpy(
            heldout[f"{category}_frozen_restricted"][:dimension, :dimension]
        ).cuda()
        for category in HELDOUT
    }

    torch.manual_seed(config["seed"])
    layer0_rotation, layer0_training = train_rotation(layer0_forms, config)
    layer17_rotation, layer17_training = train_rotation(layer17_forms, config)
    layer0_rotation = layer0_rotation.detach()
    layer17_rotation = layer17_rotation.detach()
    rotations = {"layer0": layer0_rotation, "layer17": layer17_rotation}
    training = {"layer0": layer0_training, "layer17": layer17_training}

    rotated_forms = {
        "layer0": {
            category: layer0_rotation.T @ layer0_forms[index] @ layer0_rotation
            for index, category in enumerate(ORIGINAL)
        },
        "layer17_original": {
            category: layer17_rotation.T @ layer17_forms[index] @ layer17_rotation
            for index, category in enumerate(ORIGINAL)
        },
        "layer17_heldout": {
            category: layer17_rotation.T @ form @ layer17_rotation
            for category, form in heldout_forms.items()
        },
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
        groups = {
            "layer0": {category: {"labels": [], "full": [], "before": [], "after": []} for category in ORIGINAL},
            "layer17_original": {category: {"labels": [], "full": [], "before": [], "after": []} for category in ORIGINAL},
            "layer17_heldout": {category: {"labels": [], "full": [], "before": [], "after": []} for category in HELDOUT},
        }
        for start in range(0, len(rows), 4):
            raw = np.asarray(rows[start : start + 4, :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            x0 = layer0_input(model, ids)[:, position_tensor].float()
            _, x17_all = prefix_to_layer17(model, ids)
            x17 = x17_all[:, position_tensor].float()
            z0 = x0 @ layer0_shared
            z17 = x17 @ layer17_shared
            rotated_z0 = z0 @ layer0_rotation
            rotated_z17 = z17 @ layer17_rotation
            current_ids = raw[:, positions]
            next_ids = raw[:, np.asarray(positions) + 1]

            for category in ORIGINAL:
                labels0 = flags[current_ids, CATEGORIES.index(category)].astype(bool)
                form0 = layer0_forms[ORIGINAL.index(category)]
                full0 = torch.einsum("...i,ij,...j->...", z0, form0, z0)
                before0 = z0.square() @ form0.diagonal()
                after0 = rotated_z0.square() @ rotated_forms["layer0"][category].diagonal()
                row0 = groups["layer0"][category]
                row0["labels"].extend(labels0.reshape(-1).tolist())
                row0["full"].extend(full0.flatten().cpu().tolist())
                row0["before"].extend(before0.flatten().cpu().tolist())
                row0["after"].extend(after0.flatten().cpu().tolist())

                labels17 = flags[next_ids, CATEGORIES.index(category)].astype(bool)
                form17 = layer17_forms[ORIGINAL.index(category)]
                full17 = torch.einsum("...i,ij,...j->...", z17, form17, z17)
                before17 = z17.square() @ form17.diagonal()
                after17 = rotated_z17.square() @ rotated_forms["layer17_original"][category].diagonal()
                row17 = groups["layer17_original"][category]
                row17["labels"].extend(labels17.reshape(-1).tolist())
                row17["full"].extend(full17.flatten().cpu().tolist())
                row17["before"].extend(before17.flatten().cpu().tolist())
                row17["after"].extend(after17.flatten().cpu().tolist())

            for category in HELDOUT:
                labels = flags[next_ids, CATEGORIES.index(category)].astype(bool)
                form = heldout_forms[category]
                full = torch.einsum("...i,ij,...j->...", z17, form, z17)
                before = z17.square() @ form.diagonal()
                after = rotated_z17.square() @ rotated_forms["layer17_heldout"][category].diagonal()
                row = groups["layer17_heldout"][category]
                row["labels"].extend(labels.reshape(-1).tolist())
                row["full"].extend(full.flatten().cpu().tolist())
                row["before"].extend(before.flatten().cpu().tolist())
                row["after"].extend(after.flatten().cpu().tolist())

        results[corpus] = {}
        for group, category_rows in groups.items():
            results[corpus][group] = {}
            for category, row in category_rows.items():
                positive_count = int(np.sum(row["labels"]))
                def measured_auc(scores):
                    value = auc(row["labels"], scores)
                    return value if np.isfinite(value) else None
                results[corpus][group][category] = {
                    "positive_examples": positive_count,
                    "eligible": positive_count >= config["minimum_positive_examples"],
                    "full_auc": measured_auc(row["full"]),
                    "before_auc": measured_auc(row["before"]),
                    "after_auc": measured_auc(row["after"]),
                    "before_score": score_metrics(row["full"], row["before"]),
                    "after_score": score_metrics(row["full"], row["after"]),
                }

    summary = {}
    for corpus, corpus_rows in results.items():
        summary[corpus] = {}
        for group, category_rows in corpus_rows.items():
            selected = [row for row in category_rows.values() if row["eligible"]]
            summary[corpus][group] = {
                "eligible_concepts": len(selected),
                "full_auc": float(np.median([row["full_auc"] for row in selected])),
                "before_auc": float(np.median([row["before_auc"] for row in selected])),
                "after_auc": float(np.median([row["after_auc"] for row in selected])),
                "before_correlation": float(np.median([row["before_score"]["correlation"] for row in selected])),
                "after_correlation": float(np.median([row["after_score"]["correlation"] for row in selected])),
            }

    hypotheses = {
        "A1_weight_diagonalization": {
            "supported": all(row["relative_reduction"] >= 0.50 for row in training.values()),
            "per_layer": training,
        },
        "A2_layer0_diagonal_semantics": {
            "supported": all(
                row["layer0"]["after_auc"] >= row["layer0"]["full_auc"] - 0.03
                for row in summary.values()
            ),
            "per_corpus": {c: row["layer0"] for c, row in summary.items()},
        },
        "A3_layer17_diagonal_semantics": {
            "supported": all(
                row["layer17_original"]["after_auc"] >= row["layer17_original"]["full_auc"] - 0.05
                for row in summary.values()
            ),
            "per_corpus": {c: row["layer17_original"] for c, row in summary.items()},
        },
        "A4_heldout_diagonal_transfer": {
            "supported": all(
                row["layer17_heldout"]["after_auc"] >= row["layer17_heldout"]["full_auc"] - 0.05
                for row in summary.values()
            ),
            "per_corpus": {c: row["layer17_heldout"] for c, row in summary.items()},
        },
        "A5_score_fidelity_gain": {
            "supported": all(
                row["layer17_original"]["after_correlation"]
                - row["layer17_original"]["before_correlation"] >= 0.20
                for row in summary.values()
            ),
            "per_corpus_gain": {
                c: row["layer17_original"]["after_correlation"]
                - row["layer17_original"]["before_correlation"]
                for c, row in summary.items()
            },
        },
    }
    output = {
        "config": config,
        "training": training,
        "summary": summary,
        "results": results,
        "hypotheses": hypotheses,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))
    np.savez_compressed(
        ARRAYS,
        layer0_rotation=layer0_rotation.cpu().numpy(),
        layer17_rotation=layer17_rotation.cpu().numpy(),
    )

    lines = [
        "# Weight-only approximate joint diagonalization",
        "",
        "Orthogonal rotations are optimized using only the frozen families of five weight-derived quadratic forms inside their existing 32D shared subspaces. No corpus activations enter the rotations. The layer-17 rotation is then applied unchanged to nine held-out concepts.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        status = "supported" if row["supported"] else "not supported"
        lines.append(f"- **{name}: {status}.**")
    lines += [
        "",
        "![Joint diagonalization](joint_diagonalization.png)",
        "",
        "| Family | Initial off-diagonal energy | Final | Relative reduction |",
        "|---|---:|---:|---:|",
    ]
    for name, row in training.items():
        lines.append(
            f"| {name} | {row['initial_off_diagonal_fraction']:.3f} | "
            f"{row['final_off_diagonal_fraction']:.3f} | {row['relative_reduction']:.1%} |"
        )
    lines += [
        "",
        "| Corpus | Family | Full AUROC | Diagonal before | Diagonal after | Correlation before | Correlation after |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_rows in summary.items():
        for family, row in corpus_rows.items():
            lines.append(
                f"| {corpus} | {family} | {row['full_auc']:.3f} | {row['before_auc']:.3f} | "
                f"{row['after_auc']:.3f} | {row['before_correlation']:.3f} | "
                f"{row['after_correlation']:.3f} |"
            )
    lines += ["", "Per-concept results and optimization traces are in `joint_diagonalization.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    families = ["layer0", "layer17_original", "layer17_heldout"]
    corpora = list(summary)
    x = np.arange(len(corpora))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for family in families:
        axes[0].plot(x, [summary[c][family]["before_auc"] for c in corpora], marker="x", linestyle="--", label=f"{family} before")
        axes[0].plot(x, [summary[c][family]["after_auc"] for c in corpora], marker="o", label=f"{family} after")
        axes[1].plot(x, [summary[c][family]["after_correlation"] for c in corpora], marker="o", label=family)
    axes[0].set_xticks(x, corpora)
    axes[0].set_ylabel("median diagonal-only AUROC")
    axes[1].set_xticks(x, corpora)
    axes[1].set_ylabel("diagonal/full score correlation after")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
