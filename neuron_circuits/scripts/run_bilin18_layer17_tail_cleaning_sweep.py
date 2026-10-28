#!/usr/bin/env python3
"""Resolve what covariance whitening and trace cleaning do to Layer-17 tail forms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_cleaning_sweep.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_cleaning_sweep.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_cleaning_sweep.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_CLEANING_SWEEP.md"
FIGURE = TENSOR_ROOT / "layer17_tail_cleaning_sweep.png"


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def covariance_transforms(
    x: torch.Tensor, floor_fraction: float
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    covariance = x.T @ x / len(x)
    values, vectors = torch.linalg.eigh(covariance)
    floor = values.median() * floor_fraction
    values = values.clamp_min(floor)
    square = (vectors * values.sqrt()[None]) @ vectors.T
    inverse = (vectors * values.rsqrt()[None]) @ vectors.T
    return square, inverse, {
        "minimum_eigenvalue": float(values.min()),
        "maximum_eigenvalue": float(values.max()),
        "condition_number": float(values.max() / values.min()),
    }


def trace_clean(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    isotropic = torch.trace(matrix) / len(matrix)
    return matrix - isotropic * torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype), isotropic


def sorted_eigensystem(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(matrix)
    order = values.abs().argsort(descending=True)
    return values[order], vectors[:, order]


def reconstruct_quadratic(
    coordinates: torch.Tensor,
    matrix: torch.Tensor,
    rank: int,
    preserve_trace: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    clean, isotropic = trace_clean(matrix)
    values, vectors = sorted_eigensystem(clean)
    projection = coordinates @ vectors[:, :rank]
    clean_full = torch.einsum("bi,ij,bj->b", coordinates, clean, coordinates)
    clean_top = (projection.square() * values[:rank][None]).sum(1)
    remainder = isotropic * coordinates.square().sum(1)
    native_full = clean_full + remainder
    native_top = clean_top + remainder if preserve_trace else clean_top
    energy = values.square()
    return native_full.cpu().numpy(), native_top.cpu().numpy(), {
        "trace": float(isotropic),
        "clean_top4_energy": float(energy[:rank].sum() / energy.sum().clamp_min(1e-30)),
        "clean_effective_rank": float(
            energy.sum().square() / energy.square().sum().clamp_min(1e-30)
        ),
        "max_abs_eigenvalue": float(values.abs().max()),
        "clean_score_correlation": correlation(clean_full.cpu().numpy(), clean_top.cpu().numpy()),
        "native_score_correlation": correlation(native_full.cpu().numpy(), native_top.cpu().numpy()),
        "native_without_isotropic_correlation": correlation(native_full.cpu().numpy(), clean_top.cpu().numpy()),
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    saved = np.load(SOURCE_ARRAYS)
    axes = torch.from_numpy(saved["tail_output_basis"][:, : config["components"]]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()

    print("discovery covariance", flush=True)
    discovery_x, _ = collect(model, config["layer"], config["discovery"], config)
    square, inverse, covariance = covariance_transforms(
        discovery_x, config["covariance_floor_fraction"]
    )
    discovery_z = discovery_x @ inverse
    forms = []
    arrays = {
        "square": square.cpu().numpy().astype(np.float32),
        "inverse_square": inverse.cpu().numpy().astype(np.float32),
    }
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    null_signs = torch.where(
        torch.rand(
            config["components"], config["null_sign_flips"], down.shape[1],
            generator=generator, device="cuda",
        ) < 0.5,
        -torch.ones((), device="cuda"),
        torch.ones((), device="cuda"),
    )
    for component in range(config["components"]):
        print("form", component + 1, "/", config["components"], flush=True)
        coefficients = down.T @ axes[:, component]
        raw = left.T @ (coefficients[:, None] * right)
        raw = 0.5 * (raw + raw.T)
        weighted = square @ raw @ square
        raw_full, raw_top, raw_stats = reconstruct_quadratic(
            discovery_x, raw, config["rank"], preserve_trace=True
        )
        weighted_full, weighted_top, weighted_stats = reconstruct_quadratic(
            discovery_z, weighted, config["rank"], preserve_trace=True
        )
        null_maxima = []
        for replicate in range(config["null_sign_flips"]):
            null_coefficients = coefficients * null_signs[component, replicate]
            null = left.T @ (null_coefficients[:, None] * right)
            null = 0.5 * (null + null.T)
            null_clean, _ = trace_clean(square @ null @ square)
            null_maxima.append(float(torch.linalg.eigvalsh(null_clean).abs().max()))
        threshold = float(np.quantile(null_maxima, config["null_quantile"]))
        weighted_clean, weighted_trace = trace_clean(weighted)
        weighted_values, weighted_vectors = sorted_eigensystem(weighted_clean)
        raw_clean, raw_trace = trace_clean(raw)
        raw_values, raw_vectors = sorted_eigensystem(raw_clean)
        forms.append({
            "raw": raw,
            "weighted": weighted,
            "raw_values": raw_values,
            "raw_vectors": raw_vectors,
            "weighted_values": weighted_values,
            "weighted_vectors": weighted_vectors,
            "raw_trace": raw_trace,
            "weighted_trace": weighted_trace,
            "raw_stats": raw_stats,
            "weighted_stats": weighted_stats,
            "sign_null_maxima": null_maxima,
            "sign_null_threshold": threshold,
            "outlier_ratio": weighted_stats["max_abs_eigenvalue"] / max(threshold, 1e-30),
        })
        arrays[f"component{component}_raw_eigenvalues"] = raw_values[:128].cpu().numpy()
        arrays[f"component{component}_weighted_eigenvalues"] = weighted_values[:128].cpu().numpy()
        arrays[f"component{component}_weighted_eigenvectors"] = weighted_vectors[:, :32].cpu().numpy()

    evaluation = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluation", corpus, flush=True)
        x, _ = collect(model, config["layer"], spec, config)
        z = x @ inverse
        rows = {}
        for component, form in enumerate(forms):
            raw_native = torch.einsum("bi,ij,bj->b", x, form["raw"], x).cpu().numpy()
            raw_projection = x @ form["raw_vectors"][:, : config["rank"]]
            raw_top = (
                raw_projection.square() * form["raw_values"][: config["rank"]][None]
            ).sum(1) + form["raw_trace"] * x.square().sum(1)
            weighted_native = torch.einsum("bi,ij,bj->b", z, form["weighted"], z).cpu().numpy()
            weighted_projection = z @ form["weighted_vectors"][:, : config["rank"]]
            weighted_clean_top = (
                weighted_projection.square() * form["weighted_values"][: config["rank"]][None]
            ).sum(1)
            isotropic = form["weighted_trace"] * z.square().sum(1)
            weighted_top = weighted_clean_top + isotropic
            weighted_clean_full = torch.einsum(
                "bi,ij,bj->b", z, form["weighted"] - form["weighted_trace"] * torch.eye(len(z.T), device="cuda"), z
            ).cpu().numpy()
            rows[str(component)] = {
                "raw_native_score_correlation": correlation(raw_native, raw_top.cpu().numpy()),
                "weighted_native_score_correlation": correlation(weighted_native, weighted_top.cpu().numpy()),
                "weighted_native_without_isotropic_correlation": correlation(
                    weighted_native, weighted_clean_top.cpu().numpy()
                ),
                "weighted_clean_score_correlation": correlation(
                    weighted_clean_full, weighted_clean_top.cpu().numpy()
                ),
                "raw_weighted_native_exact_correlation": correlation(raw_native, weighted_native),
            }
        evaluation[corpus] = rows

    raw_energy = [form["raw_stats"]["clean_top4_energy"] for form in forms]
    weighted_energy = [form["weighted_stats"]["clean_top4_energy"] for form in forms]
    hypotheses = {
        "C1_covariance_metric_concentrates_tail_forms": {
            "supported": float(np.median(weighted_energy) - np.median(raw_energy)) >= 0.20,
            "raw_median": float(np.median(raw_energy)),
            "weighted_median": float(np.median(weighted_energy)),
        },
        "C2_trace_cleaned_tail_axes_exceed_sign_nulls": {
            "supported": all(form["outlier_ratio"] > 1 for form in forms),
            "ratios": [form["outlier_ratio"] for form in forms],
        },
        "C3_clean_top4_reconstructs_clean_scores": {
            "supported": all(
                row["weighted_clean_score_correlation"] >= 0.90
                for corpus_rows in evaluation.values() for row in corpus_rows.values()
            ),
        },
        "C4_cleaning_improves_native_score_reconstruction": {
            "supported": all(
                np.median([row["weighted_native_score_correlation"] for row in corpus_rows.values()])
                - np.median([row["raw_native_score_correlation"] for row in corpus_rows.values()]) >= 0.10
                for corpus_rows in evaluation.values()
            ),
        },
        "C5_isotropic_remainder_is_not_the_explanation": {
            "supported": all(
                np.median([row["weighted_native_score_correlation"] for row in corpus_rows.values()])
                - np.median([row["weighted_native_without_isotropic_correlation"] for row in corpus_rows.values()]) <= 0.10
                for corpus_rows in evaluation.values()
            ),
        },
        "C6_clean_rank4_transfers_ood": {
            "supported": all(
                np.median([row["weighted_clean_score_correlation"] for row in evaluation[corpus].values()]) >= 0.90
                for corpus in ("wikitext", "tinystories")
            ),
        },
    }
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "covariance": covariance,
        "forms": [{
            "component": index,
            "raw": form["raw_stats"],
            "weighted": form["weighted_stats"],
            "sign_null_threshold": form["sign_null_threshold"],
            "sign_null_median": float(np.median(form["sign_null_maxima"])),
            "outlier_ratio": form["outlier_ratio"],
        } for index, form in enumerate(forms)],
        "evaluation": evaluation,
    }, indent=2, allow_nan=False))

    lines = [
        "# Layer-17 residual-tail input-metric cleaning sweep",
        "",
        "Each frozen residual-tail output axis defines an exact 1,152×1,152 bilinear input form. `raw` uses the Euclidean parameter metric. `covariance weighted` rewrites the same scalar operation in FineWeb second-moment coordinates (`x = S z`). `trace cleaned` separates the isotropic quadratic term before ranking modes. The isotropic term is then restored when native-score fidelity is assessed, so cleaning is not allowed to silently change the function.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Tail cleaning sweep](layer17_tail_cleaning_sweep.png)",
        "",
        "| Tail axis | Raw top-4 energy | Weighted clean top-4 | Weighted eff. rank | Sign-null outlier ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, form in enumerate(forms):
        lines.append(
            f"| {index + 1} | {form['raw_stats']['clean_top4_energy']:.3f} | "
            f"{form['weighted_stats']['clean_top4_energy']:.3f} | {form['weighted_stats']['clean_effective_rank']:.1f} | "
            f"{form['outlier_ratio']:.2f}× |"
        )
    lines += [
        "",
        "| Corpus | Raw rank-4 native corr. | Weighted rank-4 native corr. | Weighted clean corr. | Without isotropic remainder |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, rows in evaluation.items():
        lines.append(
            f"| {corpus} | {np.median([r['raw_native_score_correlation'] for r in rows.values()]):.3f} | "
            f"{np.median([r['weighted_native_score_correlation'] for r in rows.values()]):.3f} | "
            f"{np.median([r['weighted_clean_score_correlation'] for r in rows.values()]):.3f} | "
            f"{np.median([r['weighted_native_without_isotropic_correlation'] for r in rows.values()]):.3f} |"
        )
    lines += [
        "",
        "Covariance weighting is an activation-conditioned choice of metric. It can reveal operations that are compact on natural states but diffuse in raw Frobenius norm; it cannot establish a distribution-free tensor circuit. Channel-sign nulls preserve coefficient magnitudes but do not correct for the fact that the four tail axes were selected from FineWeb write covariance.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes_plot = plt.subplots(1, 3, figsize=(14, 4.2))
    indices = np.arange(config["components"])
    width = 0.36
    axes_plot[0].bar(indices - width / 2, raw_energy, width, label="raw")
    axes_plot[0].bar(indices + width / 2, weighted_energy, width, label="weighted clean")
    axes_plot[0].set_xticks(indices, [f"T{i + 1}" for i in indices])
    axes_plot[0].set_ylabel("top-four spectral energy")
    axes_plot[0].set_title("Metric changes apparent rank")
    axes_plot[0].legend()
    ratios = [form["outlier_ratio"] for form in forms]
    axes_plot[1].bar(indices, ratios)
    axes_plot[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes_plot[1].set_xticks(indices, [f"T{i + 1}" for i in indices])
    axes_plot[1].set_ylabel("observed / sign-null 95%")
    axes_plot[1].set_title("Cleaned spike test")
    corpora = list(evaluation)
    axes_plot[2].plot(corpora, [np.median([r["raw_native_score_correlation"] for r in evaluation[c].values()]) for c in corpora], marker="o", label="raw")
    axes_plot[2].plot(corpora, [np.median([r["weighted_native_score_correlation"] for r in evaluation[c].values()]) for c in corpora], marker="o", label="weighted clean + isotropic")
    axes_plot[2].set_ylim(0, 1.02)
    axes_plot[2].set_ylabel("rank-four native-score correlation")
    axes_plot[2].set_title("Held-out reconstruction")
    axes_plot[2].legend(fontsize=8)
    for axis in axes_plot:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
