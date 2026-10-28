#!/usr/bin/env python3
"""Test power-law bulk, spectral outliers, and semantics of the query map."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_joint_axis_atlas import ARRAYS as ATLAS_ARRAYS
from run_bilin18_query_conditioned_interpreter import CONFIG as INTERPRETER_CONFIG, collect_states_and_writes
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_query_map_spectral_redteam.json")
OUTPUT = TENSOR_ROOT / "query_map_spectral_redteam.json"
REPORT = TENSOR_ROOT / "QUERY_MAP_SPECTRAL_REDTEAM.md"
FIGURE = TENSOR_ROOT / "query_map_spectral_redteam.png"


def spectrum_fit(values: np.ndarray, config: dict) -> dict:
    start, end = config["bulk_start"], config["bulk_end"]
    ranks = np.arange(start + 1, end + 1)
    log_values = np.log(np.maximum(values[start:end], 1e-30))
    slope, intercept = np.polyfit(np.log(ranks), log_values, 1)
    fitted = intercept + slope * np.log(ranks)
    r2 = 1 - np.square(log_values - fitted).sum() / np.square(log_values - log_values.mean()).sum()
    top_ranks = np.arange(1, config["outlier_count"] + 1)
    top_fit = intercept + slope * np.log(top_ranks)
    excess = np.log(np.maximum(values[: config["outlier_count"]], 1e-30)) - top_fit
    return {
        "exponent": float(-slope),
        "loglog_r2": float(r2),
        "mean_top_log_excess": float(excess.mean()),
        "top_excess_ratio": np.exp(excess).tolist(),
    }


def semantic_profiles(components: torch.Tensor, unembedding: torch.Tensor, flags: np.ndarray, config: dict):
    vocab_scores = (unembedding @ components).cpu().numpy()
    rows = []
    for index in range(components.shape[1]):
        best = None
        for concept in json.loads(INTERPRETER_CONFIG.read_text())["semantic_concepts"]:
            labels = flags[:, CATEGORIES.index(concept)].astype(bool)
            raw_auc = auc(labels, vocab_scores[:, index])
            oriented = max(raw_auc, 1 - raw_auc)
            candidate = (oriented, concept, 1 if raw_auc >= 0.5 else -1)
            if best is None or candidate[0] > best[0]:
                best = candidate
        order = np.argsort(vocab_scores[:, index])
        rows.append({
            "component": index,
            "best_concept": best[1],
            "best_oriented_auc": float(best[0]),
            "orientation": best[2],
            "top_token_ids": order[-8:][::-1].tolist(),
            "bottom_token_ids": order[:8].tolist(),
        })
    count = sum(row["best_oriented_auc"] >= config["semantic_auc_threshold"] for row in rows)
    return rows, count


def build_map(down: torch.Tensor, left_axes: torch.Tensor, right_axes: torch.Tensor):
    tensor = torch.einsum("dh,hi,hj->dij", down, left_axes, right_axes)
    tensor = 0.5 * (tensor + tensor.transpose(1, 2))
    return tensor.flatten(1)


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    interpreter_config = json.loads(INTERPRETER_CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[17]
    axes = torch.from_numpy(np.load(ATLAS_ARRAYS)["layer17_axes"]).cuda()
    left_axes = block.mlp.Left.weight.float() @ axes
    right_axes = block.mlp.Right.weight.float() @ axes
    down = block.mlp.Down.weight.float()
    unembedding = model.lm_head.weight.float()[: tokenizer.vocab_size]

    observed_map = build_map(down, left_axes, right_axes)
    observed_u, observed_s, _ = torch.linalg.svd(observed_map, full_matrices=False)
    observed_fit = spectrum_fit(observed_s.cpu().numpy(), config)
    observed_profiles, observed_semantic_count = semantic_profiles(
        observed_u[:, : config["semantic_components"]], unembedding, flags, config
    )
    for row in observed_profiles:
        row["top_tokens"] = [tokenizer.decode([index]) for index in row.pop("top_token_ids")]
        row["bottom_tokens"] = [tokenizer.decode([index]) for index in row.pop("bottom_token_ids")]

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    nulls = {"sign": [], "pairing": []}
    null_spectra = {"sign": [], "pairing": []}
    for replicate in range(config["null_replicates"]):
        sign = torch.where(
            torch.rand(down.shape[1], generator=generator, device="cuda") < 0.5,
            -torch.ones(down.shape[1], device="cuda"),
            torch.ones(down.shape[1], device="cuda"),
        )
        permutation = torch.randperm(down.shape[1], generator=generator, device="cuda")
        for family, null_down in (
            ("sign", down * sign[None, :]),
            ("pairing", down[:, permutation]),
        ):
            null_map = build_map(null_down, left_axes, right_axes)
            null_u, null_s, _ = torch.linalg.svd(null_map, full_matrices=False)
            fit = spectrum_fit(null_s.cpu().numpy(), config)
            _, semantic_count = semantic_profiles(
                null_u[:, : config["semantic_components"]], unembedding, flags, config
            )
            fit["semantic_components"] = semantic_count
            nulls[family].append(fit)
            null_spectra[family].append(null_s.cpu().numpy())
            print(family, replicate + 1, "/", config["null_replicates"], flush=True)

    _, discovery_writes, _ = collect_states_and_writes(
        model,
        interpreter_config["discovery"],
        interpreter_config["positions"],
        interpreter_config["batch_size"],
    )
    write = torch.from_numpy(discovery_writes).cuda()
    write -= write.mean(0, keepdim=True)
    _, _, natural_v = torch.pca_lowrank(write, q=64, center=False, niter=4)
    overlaps = {}
    for rank in (4, 16, 64):
        canonical = torch.linalg.svdvals(observed_u[:, :rank].T @ natural_v[:, :rank])
        overlaps[str(rank)] = {
            "mean": float(canonical.mean()),
            "minimum": float(canonical.min()),
            "maximum": float(canonical.max()),
        }

    null_summary = {
        family: {
            key: float(np.median([row[key] for row in rows]))
            for key in ("exponent", "loglog_r2", "mean_top_log_excess", "semantic_components")
        }
        for family, rows in nulls.items()
    }
    exponent_ok = all(
        abs(null_summary[family]["exponent"] - observed_fit["exponent"]) <= 0.20
        and null_summary[family]["loglog_r2"] >= 0.90
        for family in null_summary
    )
    all_null_rows = nulls["sign"] + nulls["pairing"]
    hypotheses = {
        "N1_generic_power_law_bulk": {
            "supported": exponent_ok,
            "observed": observed_fit,
            "null_medians": null_summary,
        },
        "N2_observed_outlier_excess": {
            "supported": observed_fit["mean_top_log_excess"]
            > max(row["mean_top_log_excess"] for row in all_null_rows),
            "observed": observed_fit["mean_top_log_excess"],
            "maximum_null": max(row["mean_top_log_excess"] for row in all_null_rows),
        },
        "N3_semantic_outlier_components": {
            "supported": observed_semantic_count >= 8,
            "count": observed_semantic_count,
        },
        "N4_semantics_exceed_null": {
            "supported": observed_semantic_count
            > max(row["semantic_components"] for row in all_null_rows),
            "observed": observed_semantic_count,
            "maximum_null": max(row["semantic_components"] for row in all_null_rows),
        },
        "N5_not_just_write_pca": {
            "supported": overlaps["16"]["mean"] < 0.50,
            "mean_overlap": overlaps["16"]["mean"],
        },
    }
    output = {
        "config": config,
        "observed_fit": observed_fit,
        "observed_semantic_components": observed_semantic_count,
        "semantic_profiles": observed_profiles,
        "nulls": nulls,
        "null_summary": null_summary,
        "natural_write_overlap": overlaps,
        "hypotheses": hypotheses,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = [
        "# Spectral red-team of the query-conditioned map",
        "",
        "The 1,152→32×32 readout-to-interaction map is compared with six channel-sign nulls and six Down-to-factor pairing nulls. The bulk fit is frozen to ranks 33–512; outlier excess extrapolates that fit to ranks 1–4. Leading output singular vectors are independently tested against 14 token predicates.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Query-map nulls](query_map_spectral_redteam.png)",
        "",
        "| Spectrum | Exponent | Bulk R² | Top-four log excess | Semantic top-32 |",
        "|---|---:|---:|---:|---:|",
        f"| observed | {observed_fit['exponent']:.3f} | {observed_fit['loglog_r2']:.3f} | {observed_fit['mean_top_log_excess']:.3f} | {observed_semantic_count} |",
    ]
    for family, row in null_summary.items():
        lines.append(
            f"| {family} null median | {row['exponent']:.3f} | {row['loglog_r2']:.3f} | "
            f"{row['mean_top_log_excess']:.3f} | {row['semantic_components']:.1f} |"
        )
    lines += [
        "",
        f"Mean canonical overlap with natural-write PCA is {overlaps['4']['mean']:.3f} at rank 4, {overlaps['16']['mean']:.3f} at rank 16, and {overlaps['64']['mean']:.3f} at rank 64.",
        "",
        "The observed top-four log deviation is negative: the leading singular values are stronger than every tested null relative to their bulks, but remain below a naive extrapolation of the observed rank-33–512 power law. This is comparative top-spectrum structure, not yet a strict above-power-law outlier.",
        "",
        "The semantic-count test also fails as an independent circuit criterion: sign-null output directions are drawn from the same learned Down/unembedding geometry and score equally well under the broad predicate vocabulary.",
        "",
        "A power-law fit and an outlier are structural clues, not circuit validation. Semantic alignment is tested separately, and the nulls preserve parameter norms while disrupting different kinds of learned coordination.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, plots = plt.subplots(1, 2, figsize=(11, 4.2))
    ranks = np.arange(1, len(observed_s) + 1)
    plots[0].loglog(ranks, observed_s.cpu().numpy(), label="observed", linewidth=2)
    for family, color in (("sign", "tab:orange"), ("pairing", "tab:green")):
        median_spectrum = np.median(np.stack(null_spectra[family]), axis=0)
        plots[0].loglog(ranks, median_spectrum, label=f"{family} null", color=color)
    plots[0].set_xlabel("rank")
    plots[0].set_ylabel("singular value")
    plots[0].legend()
    plots[1].bar(
        ["observed", "sign null", "pairing null"],
        [observed_semantic_count, null_summary["sign"]["semantic_components"], null_summary["pairing"]["semantic_components"]],
    )
    plots[1].axhline(8, color="black", linestyle="--", linewidth=1)
    plots[1].set_ylabel("top-32 components with AUROC ≥ 0.70")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
