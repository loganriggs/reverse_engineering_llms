#!/usr/bin/env python3
"""Ask whether Layer-17's stable residual tail contains weight-readable circuits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer17_bottleneck_tail_ontology import collect_writes, corpus_rows
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_query_map_spectral_redteam import semantic_profiles
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_circuit_redteam.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_circuit_redteam.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_circuit_redteam.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_CIRCUIT_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer17_tail_circuit_redteam.png"


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def cumulative_subspace_coverage(
    target: np.ndarray, ordered_basis: np.ndarray, ranks: list[int]
) -> dict[str, float]:
    return {
        str(rank): float(np.square(target.T @ ordered_basis[:, :rank]).sum() / target.shape[1])
        for rank in ranks
    }


def axis_alignment_statistic(scores: np.ndarray, flags: np.ndarray) -> float:
    """Mean, across axes, of the strongest absolute standardized class gap."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = (scores - scores.mean(0)) / scores.std(0).clip(1e-12)
    gaps = []
    for category in range(flags.shape[1]):
        positive = flags[:, category].astype(bool)
        if not positive.any() or positive.all():
            continue
        gaps.append(np.abs(scores[positive].mean(0) - scores[~positive].mean(0)))
    return float(np.stack(gaps).max(0).mean())


def rotation_null(
    scores: np.ndarray, flags: np.ndarray, replicates: int, seed: int
) -> tuple[float, list[float], float]:
    observed = axis_alignment_statistic(scores, flags)
    generator = np.random.default_rng(seed)
    null = []
    for _ in range(replicates):
        rotation, _ = np.linalg.qr(generator.normal(size=(scores.shape[1], scores.shape[1])))
        null.append(axis_alignment_statistic(scores @ rotation, flags))
    p_value = (1 + sum(value >= observed for value in null)) / (replicates + 1)
    return observed, null, float(p_value)


def summarize_modes(records: list[dict], component: int, mode: str) -> dict[str, float]:
    selected = [row for row in records if row["component"] == component and row["mode"] == mode]
    actual = np.asarray([row["actual_scalar_change"] for row in selected])
    predicted = np.asarray([row["predicted_scalar_change"] for row in selected])
    return {
        "median_absolute_scalar_change": float(np.median(np.abs(actual))),
        "median_kl": float(np.median([row["kl"] for row in selected])),
        "prediction_correlation": correlation(actual, predicted),
    }


@torch.inference_mode()
def collect_interventions(
    model,
    config: dict,
    corpus: str,
    spec: dict,
    tail_axes: torch.Tensor,
    component_data: list[dict],
    write_mean: np.ndarray,
) -> list[dict]:
    rows = corpus_rows(spec)
    writes, _ = collect_writes(model, spec, config)
    centered = writes - write_mean[None]
    scores = centered @ tail_axes.cpu().numpy()
    positions = config["positions"]
    selected = []
    for component in range(config["components"]):
        order = np.argsort(np.abs(scores[:, component] - np.median(scores[:, component])))[::-1]
        selected.append(order[: config["sites_per_component"]])

    block = model.blocks[config["layer"]]
    dimension = model.config.n_embd
    records = []
    for component, sites in enumerate(selected):
        axis = tail_axes[:, component]
        data = component_data[component]
        for site in sites:
            row_index = int(site // len(positions))
            position = positions[int(site % len(positions))]
            ids = torch.as_tensor(
                np.asarray(rows[row_index : row_index + 1, :128], dtype=np.int64), device="cuda"
            )
            pre, normalized = prefix_to_layer17(model, ids)
            x = normalized[0, position].float()
            pre_site = pre[0, position].float()
            native_write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x)) + block.mlp.Down_bias
            native_logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(pre_site + native_write, (dimension,))).float() / 30
            )
            native_logprob = native_logits.log_softmax(-1)
            native_probability = native_logprob.exp()
            native_score = float(native_write @ axis)
            for mode in ("top", "low", "random"):
                vectors = data[mode]
                delta = -(x @ vectors) @ vectors.T
                delta *= (
                    config["edit_norm_fraction"] * x.norm()
                    / delta.norm().clamp_min(1e-20)
                )
                changed_x = x + delta
                changed_write = block.mlp.Down(
                    block.mlp.Left(changed_x) * block.mlp.Right(changed_x)
                ) + block.mlp.Down_bias
                changed_logits = 30 * torch.tanh(
                    model.lm_head(F.rms_norm(pre_site + changed_write, (dimension,))).float() / 30
                )
                changed_logprob = changed_logits.log_softmax(-1)
                actual = float(changed_write @ axis) - native_score
                matrix = data["matrix"]
                predicted = float(
                    2 * torch.einsum("i,ij,j->", delta, matrix, x)
                    + torch.einsum("i,ij,j->", delta, matrix, delta)
                )
                records.append({
                    "corpus": corpus,
                    "component": component,
                    "row": row_index,
                    "position": position,
                    "mode": mode,
                    "actual_scalar_change": actual,
                    "predicted_scalar_change": predicted,
                    "kl": float((native_probability * (native_logprob - changed_logprob)).sum()),
                    "delta_norm": float(delta.norm()),
                })
    return records


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    saved = np.load(SOURCE_ARRAYS)
    tail_axes_np = saved["tail_output_basis"][:, : config["components"]].astype(np.float32)
    write_mean = saved["write_mean"].astype(np.float32)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    tail_axes = torch.from_numpy(tail_axes_np).cuda()

    print("raw weight spectrum", flush=True)
    raw_singular, raw_output = output_map_eigensystem(down, left, right)
    raw_coverage = cumulative_subspace_coverage(
        tail_axes_np, raw_output.cpu().numpy(), config["weight_ranks"]
    )
    raw_best = []
    overlap = np.abs(tail_axes_np.T @ raw_output.cpu().numpy())
    for component in range(config["components"]):
        rank = int(np.argmax(overlap[component]))
        raw_best.append({"rank": rank + 1, "cosine": float(overlap[component, rank])})

    print("natural input covariance and weighted spectrum", flush=True)
    discovery_x, _ = collect(model, config["layer"], config["discovery"], config)
    _, state_singular, state_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    weighted_basis = state_vh.T * (state_singular / np.sqrt(len(discovery_x)))
    weighted_singular, weighted_output = output_map_eigensystem(
        down, left @ weighted_basis, right @ weighted_basis
    )
    weighted_coverage = cumulative_subspace_coverage(
        tail_axes_np, weighted_output.cpu().numpy(), config["weight_ranks"]
    )

    print("tail forms", flush=True)
    arrays = {
        "tail_output_axes": tail_axes_np,
        "raw_singular_values": raw_singular[:256].cpu().numpy(),
        "weighted_singular_values": weighted_singular[:256].cpu().numpy(),
        "raw_tail_overlap": overlap[:, :256],
    }
    component_data = []
    for component in range(config["components"]):
        axis = tail_axes[:, component]
        coefficients = down.T @ axis
        matrix = left.T @ (coefficients[:, None] * right)
        matrix = 0.5 * (matrix + matrix.T)
        values, vectors = torch.linalg.eigh(matrix)
        order = torch.argsort(values.abs(), descending=True)
        values, vectors = values[order], vectors[:, order]
        energy = values.square()
        top = vectors[:, : config["input_eigenspace_rank"]]
        low = vectors[:, -config["input_eigenspace_rank"] :]
        generator = torch.Generator(device="cuda").manual_seed(config["seed"] + component)
        random = torch.randn(
            left.shape[1], config["input_eigenspace_rank"], generator=generator, device="cuda"
        )
        random -= top @ (top.T @ random)
        random = torch.linalg.qr(random, mode="reduced").Q
        component_data.append({
            "matrix": matrix,
            "top": top,
            "low": low,
            "random": random,
            "top4_form_energy": float(energy[: config["input_eigenspace_rank"]].sum() / energy.sum()),
        })
        arrays[f"component{component}_eigenvalues"] = values[:64].cpu().numpy()
        arrays[f"component{component}_eigenvectors"] = vectors[:, :32].cpu().numpy()

    profiles, _ = semantic_profiles(
        tail_axes,
        model.lm_head.weight.float()[: tokenizer.vocab_size],
        flags,
        {"semantic_auc_threshold": 0.70},
    )
    for profile in profiles:
        profile["top_tokens"] = [tokenizer.decode([index]) for index in profile.pop("top_token_ids")]
        profile["bottom_tokens"] = [tokenizer.decode([index]) for index in profile.pop("bottom_token_ids")]

    vocab_scores = (model.lm_head.weight.float()[: tokenizer.vocab_size] @ tail_axes).cpu().numpy()
    observed_alignment, null_alignment, rotation_p = rotation_null(
        vocab_scores, flags, config["rotation_null_replicates"], config["seed"]
    )

    records = []
    summaries = {}
    for corpus, spec in config["evaluation"].items():
        print("finite interventions", corpus, flush=True)
        corpus_records = collect_interventions(
            model, config, corpus, spec, tail_axes, component_data, write_mean
        )
        records.extend(corpus_records)
        summaries[corpus] = {}
        for component in range(config["components"]):
            summaries[corpus][str(component)] = {
                "top4_form_energy": component_data[component]["top4_form_energy"],
                "output_profile": profiles[component],
                "modes": {
                    mode: summarize_modes(corpus_records, component, mode)
                    for mode in ("top", "low", "random")
                },
            }

    raw_rank64 = raw_coverage["64"]
    hypotheses = {
        "R1_raw_top64_recovers_tail": {
            "supported": raw_rank64 >= 0.75,
            "coverage": raw_rank64,
        },
        "R2_natural_weighting_improves_early_recovery": {
            "supported": weighted_coverage["16"] - raw_coverage["16"] >= 0.20,
            "raw_rank16": raw_coverage["16"],
            "weighted_rank16": weighted_coverage["16"],
        },
        "R3_tail_axes_have_compact_input_forms": {
            "supported": all(row["top4_form_energy"] >= 0.50 for row in component_data),
            "per_component": [row["top4_form_energy"] for row in component_data],
        },
        "R4_weight_formula_predicts_tail_edits": {
            "supported": all(
                row["modes"]["top"]["prediction_correlation"] >= 0.99
                for corpus_rows in summaries.values() for row in corpus_rows.values()
            ),
        },
        "R5_tail_top_modes_are_scalar_specific": {
            "supported": all(
                row["modes"]["top"]["median_absolute_scalar_change"]
                >= 4 * row["modes"]["low"]["median_absolute_scalar_change"]
                for corpus_rows in summaries.values() for row in corpus_rows.values()
            ),
        },
        "R6_tail_top_modes_reach_behavior": {
            "supported": all(
                sum(
                    row["modes"]["top"]["median_kl"]
                    >= 4 * row["modes"]["low"]["median_kl"]
                    for row in corpus_rows.values()
                ) >= 3
                for corpus_rows in summaries.values()
            ),
        },
        "R7_pca_axes_are_semantically_privileged": {
            "supported": rotation_p <= 0.05,
            "observed": observed_alignment,
            "null_median": float(np.median(null_alignment)),
            "p_value": rotation_p,
        },
        "R8_tail_axes_have_simple_output_roles": {
            "supported": all(row["best_oriented_auc"] >= 0.70 for row in profiles),
            "per_component": [row["best_oriented_auc"] for row in profiles],
        },
    }
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "weight_discovery": {
            "raw_coverage": raw_coverage,
            "weighted_coverage": weighted_coverage,
            "raw_best_component_matches": raw_best,
        },
        "rotation_null": {
            "observed": observed_alignment,
            "replicates": null_alignment,
            "p_value": rotation_p,
        },
        "profiles": profiles,
        "summaries": summaries,
        "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Layer-17 residual-tail circuit red-team",
        "",
        "The four axes are the leading FineWeb residual-write PCs after projecting away the strict full-width weight outliers. They are frozen before this test. We ask whether raw or natural-covariance-weighted tensor spectra discover them, whether each axis has a compact exact input form, and whether those forms survive finite interventions on disjoint FineWeb and OOD panels.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Tail circuit red-team](layer17_tail_circuit_redteam.png)",
        "",
        "| Weight rank | Raw tail-4 coverage | Natural-weighted tail-4 coverage |",
        "|---:|---:|---:|",
    ]
    for rank in config["weight_ranks"]:
        lines.append(f"| {rank} | {raw_coverage[str(rank)]:.3f} | {weighted_coverage[str(rank)]:.3f} |")
    lines += [
        "",
        "| Tail axis | Best raw weight rank / cosine | Output role / AUROC | Input top-4 energy |",
        "|---:|---:|---|---:|",
    ]
    for component in range(config["components"]):
        profile = profiles[component]
        lines.append(
            f"| {component + 1} | {raw_best[component]['rank']} / {raw_best[component]['cosine']:.3f} | "
            f"{profile['best_concept']} / {profile['best_oriented_auc']:.3f} | "
            f"{component_data[component]['top4_form_energy']:.3f} |"
        )
    lines += [
        "",
        "| Corpus | Median top/low scalar ratio | Components with ≥4× top/low KL | Median exact prediction corr. |",
        "|---|---:|---:|---:|",
    ]
    for corpus, corpus_rows in summaries.items():
        scalar_ratios, correlations = [], []
        behavioral = 0
        for row in corpus_rows.values():
            top, low = row["modes"]["top"], row["modes"]["low"]
            scalar_ratios.append(top["median_absolute_scalar_change"] / max(low["median_absolute_scalar_change"], 1e-30))
            correlations.append(top["prediction_correlation"])
            behavioral += top["median_kl"] >= 4 * low["median_kl"]
        lines.append(
            f"| {corpus} | {np.median(scalar_ratios):.1f}× | {behavioral}/4 | {np.median(correlations):.3f} |"
        )
    lines += [
        "",
        f"PCA-axis structural alignment: **{observed_alignment:.3f}**; random-rotation median **{np.median(null_alignment):.3f}**; empirical `p={rotation_p:.4f}`.",
        "",
        "The rotation null preserves the entire tail-4 subspace and changes only its coordinate axes. Passing it would justify interpreting individual PCA coordinates; failing it would leave the subspace interpretable while making the four-axis naming noncanonical. Natural weighting uses the FineWeb input covariance and is therefore a distribution-conditioned cleaning, not a distribution-free RMT spectrum.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ranks = np.asarray(config["weight_ranks"])
    axes[0].plot(ranks, [raw_coverage[str(rank)] for rank in ranks], marker="o", label="raw")
    axes[0].plot(ranks, [weighted_coverage[str(rank)] for rank in ranks], marker="o", label="natural weighted")
    axes[0].axhline(0.75, color="black", linestyle="--", linewidth=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("weight spectral rank")
    axes[0].set_ylabel("tail-4 subspace coverage")
    axes[0].set_title("Weight discoverability")
    axes[0].legend()
    axes[1].bar(np.arange(4), [row["top4_form_energy"] for row in component_data])
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(np.arange(4), [f"T{i + 1}" for i in range(4)])
    axes[1].set_ylabel("interaction-form energy")
    axes[1].set_title("Four leading input modes")
    axes[2].hist(null_alignment, bins=16, alpha=0.75, label="rotated axes")
    axes[2].axvline(observed_alignment, color="tab:red", linewidth=2, label="PCA axes")
    axes[2].set_xlabel("structural alignment")
    axes[2].set_title("Coordinate red-team")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
