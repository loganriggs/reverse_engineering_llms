#!/usr/bin/env python3
"""Replicate the final-layer spectral-bottleneck/tail regime on a 12L model.

The raw output directions are selected from weights alone. A large FineWeb
discovery panel supplies a residual-tail PCA basis and an unconstrained natural
PCA control. Frozen bases are then evaluated on disjoint FineWeb, WikiText, and
TinyStories states and with exact final-layer write replacements.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer11_bottleneck_tail_replication.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer11_bottleneck_tail_replication.json"
ARRAYS = TENSOR_ROOT / "bilin12_layer11_bottleneck_tail_replication.npz"
REPORT = TENSOR_ROOT / "BILIN12_LAYER11_BOTTLENECK_TAIL_REPLICATION.md"
FIGURE = TENSOR_ROOT / "bilin12_layer11_bottleneck_tail_replication.png"


def project_complement(values: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kept = (values @ basis) @ basis.T
    return kept, values - kept


def covariance_eigensystem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    covariance = values.T @ values / max(len(values), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return np.maximum(eigenvalues[order], 0), eigenvectors[:, order]


def effective_rank(values: np.ndarray) -> float:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0)
    return float(values.sum() ** 2 / max(np.square(values).sum(), 1e-30))


def variance_capture(centered: np.ndarray, basis: np.ndarray) -> float:
    return float(np.square(centered @ basis).sum() / max(np.square(centered).sum(), 1e-30))


def exact_kl(reference: torch.Tensor, changed: torch.Tensor) -> torch.Tensor:
    probability = F.softmax(reference, -1)
    return (probability * (F.log_softmax(reference, -1) - F.log_softmax(changed, -1))).sum(-1)


@torch.inference_mode()
def final_layer_variant_logits(
    model,
    ids: torch.Tensor,
    layer: int,
    post_bias_mean: torch.Tensor,
    bases: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute exact final-position logits while replacing the final MLP write."""
    assert layer == len(model.blocks) - 1
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x = x0
    value_bus = None
    for block in model.blocks[:layer]:
        x, value_bus = block(x, value_bus, x0)
    block = model.blocks[layer]
    mixed = block.lambdas[0] * x + block.lambdas[1] * x0
    attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    pre = mixed + attention
    normalized = F.rms_norm(pre, (dimension,))
    write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    native = write[:, -1].float()

    def logits(selected_write: torch.Tensor) -> torch.Tensor:
        raw = model.lm_head(F.rms_norm(pre[:, -1].float() + selected_write, (dimension,))).float()
        return 30 * torch.tanh(raw / 30)

    reference = logits(native)
    centered = native - post_bias_mean
    changed = {}
    for name, basis in bases.items():
        projected = post_bias_mean + (centered @ basis) @ basis.T
        changed[name] = logits(projected)
    return reference, changed


def load_rows(spec: dict) -> np.ndarray:
    source = np.load(spec["path"], mmap_mode="r")
    return np.asarray(source[spec["start"] : spec["start"] + spec["rows"]])


@torch.inference_mode()
def behavior_metrics(
    model,
    layer: int,
    spec: dict,
    batch_size: int,
    post_bias_mean: np.ndarray,
    bases: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    rows = load_rows(spec)
    mean = torch.as_tensor(post_bias_mean, device="cuda", dtype=torch.float32)
    device_bases = {
        name: torch.as_tensor(basis, device="cuda", dtype=torch.float32)
        for name, basis in bases.items()
    }
    records = {
        name: {key: [] for key in ("kl", "ce_delta", "top1")}
        for name in bases
    }
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start : start + batch_size, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        target = torch.as_tensor(
            np.asarray(rows[start : start + len(raw), 128], dtype=np.int64),
            device="cuda",
        )
        reference, changed = final_layer_variant_logits(model, ids, layer, mean, device_bases)
        reference_logprob = F.log_softmax(reference, -1)
        reference_ce = -reference_logprob.gather(1, target[:, None]).squeeze(1)
        reference_top1 = reference.argmax(-1)
        for name, logits in changed.items():
            changed_logprob = F.log_softmax(logits, -1)
            changed_ce = -changed_logprob.gather(1, target[:, None]).squeeze(1)
            records[name]["kl"].extend(exact_kl(reference, logits).cpu().tolist())
            records[name]["ce_delta"].extend((changed_ce - reference_ce).cpu().tolist())
            records[name]["top1"].extend((logits.argmax(-1) == reference_top1).cpu().tolist())
    output = {}
    for name, row in records.items():
        kl = np.asarray(row["kl"])
        output[name] = {
            "median_kl": float(np.median(kl)),
            "median_absolute_kl": float(np.median(np.abs(kl))),
            "mean_kl": float(np.mean(kl)),
            "p90_kl": float(np.quantile(kl, 0.9)),
            "maximum_absolute_kl": float(np.max(np.abs(kl))),
            "mean_ce_delta": float(np.mean(row["ce_delta"])),
            "top1_agreement": float(np.mean(row["top1"])),
        }
    return output


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    layer = config["layer"]
    block = model.blocks[layer]
    _, raw_vectors = output_map_eigensystem(
        block.mlp.Down.weight.float(),
        block.mlp.Left.weight.float(),
        block.mlp.Right.weight.float(),
    )
    raw4 = raw_vectors[:, : config["outlier_rank"]].cpu().numpy().astype(np.float64)

    print("collecting Layer-11 FineWeb discovery writes", flush=True)
    _, discovery_write_t = collect_states_writes(
        model, layer, config["discovery"], config["positions"], config["batch_size"]
    )
    discovery_write = discovery_write_t.cpu().numpy().astype(np.float64)
    del discovery_write_t
    pre_bias_mean = discovery_write.mean(0)
    post_bias_mean = pre_bias_mean + block.mlp.Down_bias.float().cpu().numpy()
    centered_discovery = discovery_write - pre_bias_mean
    raw_part, discovery_tail = project_complement(centered_discovery, raw4)
    natural_values, natural_basis = covariance_eigensystem(centered_discovery)
    tail_values, tail_basis = covariance_eigensystem(discovery_tail)

    max_tail_rank = max(config["tail_ranks"])
    generator = np.random.default_rng(config["seed"])
    random_matrix = generator.normal(size=(model.config.n_embd, max_tail_rank))
    random_matrix -= raw4 @ (raw4.T @ random_matrix)
    random_tail, _ = np.linalg.qr(random_matrix)

    bases: dict[str, np.ndarray] = {
        "raw4": raw4,
        "natural4": natural_basis[:, :4],
        "full": np.eye(model.config.n_embd),
    }
    for tail_rank in config["tail_ranks"]:
        total_rank = config["outlier_rank"] + tail_rank
        bases[f"raw4_tail{tail_rank}"] = np.concatenate((raw4, tail_basis[:, :tail_rank]), axis=1)
        bases[f"raw4_random{tail_rank}"] = np.concatenate((raw4, random_tail[:, :tail_rank]), axis=1)
        bases[f"natural{total_rank}"] = natural_basis[:, :total_rank]

    arrays = {
        "raw_outlier_basis": raw4.astype(np.float32),
        "tail_basis": tail_basis[:, :max_tail_rank].astype(np.float32),
        "natural_basis": natural_basis[:, : config["outlier_rank"] + max_tail_rank].astype(np.float32),
        "pre_bias_write_mean": pre_bias_mean.astype(np.float32),
        "post_bias_write_mean": post_bias_mean.astype(np.float32),
        "natural_eigenvalues": natural_values[:256].astype(np.float32),
        "tail_eigenvalues": tail_values[:256].astype(np.float32),
    }

    results = {}
    for corpus, spec in config["evaluation"].items():
        print("state evaluation", corpus, flush=True)
        _, write_t = collect_states_writes(model, layer, spec, config["positions"], config["batch_size"])
        centered = write_t.cpu().numpy().astype(np.float64) - pre_bias_mean
        del write_t
        local_values, local_basis = covariance_eigensystem(centered)
        _, local_tail = project_complement(centered, raw4)
        local_tail_values, local_tail_basis = covariance_eigensystem(local_tail)
        covariance = centered.T @ centered / len(centered)
        tail_covariance = local_tail.T @ local_tail / len(local_tail)
        captured4 = float(np.trace(raw4.T @ covariance @ raw4))
        total_variance = float(np.trace(covariance))
        rank_rows = {}
        for tail_rank in config["tail_ranks"]:
            total_rank = config["outlier_rank"] + tail_rank
            frozen_tail = tail_basis[:, :tail_rank]
            canonical = np.linalg.svd(frozen_tail.T @ local_tail_basis[:, :tail_rank], compute_uv=False)
            frozen_tail_variance = float(np.trace(frozen_tail.T @ tail_covariance @ frozen_tail))
            optimal_tail_variance = float(local_tail_values[:tail_rank].sum())
            combined = bases[f"raw4_tail{tail_rank}"]
            combined_variance = float(np.trace(combined.T @ covariance @ combined))
            rank_rows[str(tail_rank)] = {
                "total_rank": total_rank,
                "combined_total_variance_fraction": combined_variance / max(total_variance, 1e-30),
                "combined_optimal_variance_efficiency": combined_variance / max(float(local_values[:total_rank].sum()), 1e-30),
                "tail_variance_efficiency": frozen_tail_variance / max(optimal_tail_variance, 1e-30),
                "tail_mean_canonical_cosine": float(canonical.mean()),
                "tail_minimum_canonical_cosine": float(canonical.min()),
            }
        results[corpus] = {
            "sites": len(centered),
            "raw4_total_variance_fraction": captured4 / max(total_variance, 1e-30),
            "raw4_optimal_variance_efficiency": captured4 / max(float(local_values[:4].sum()), 1e-30),
            "tail_total_variance_fraction": float(np.square(local_tail).sum() / max(np.square(centered).sum(), 1e-30)),
            "tail_effective_rank": effective_rank(local_tail_values),
            "tail_top16_fraction": float(local_tail_values[:16].sum() / max(local_tail_values.sum(), 1e-30)),
            "rank_rows": rank_rows,
        }

    for corpus, spec in config["behavior"].items():
        print("causal evaluation", corpus, flush=True)
        results[corpus]["behavior"] = behavior_metrics(
            model, layer, spec, config["batch_size"], post_bias_mean, bases
        )

    hypotheses = {
        "T1_raw4_is_cross_corpus_natural_bottleneck": {
            "supported": all(row["raw4_optimal_variance_efficiency"] >= 0.75 for row in results.values()),
            "per_corpus": {key: row["raw4_optimal_variance_efficiency"] for key, row in results.items()},
        },
        "T2_raw4_alone_is_behaviorally_sufficient": {
            "supported": all(row["behavior"]["raw4"]["median_kl"] <= 0.02 for row in results.values()),
            "per_corpus": {key: row["behavior"]["raw4"]["median_kl"] for key, row in results.items()},
        },
        "T3_raw4_plus_tail16_is_near_sufficient": {
            "supported": all(row["behavior"]["raw4_tail16"]["median_kl"] <= 0.02 for row in results.values()),
            "per_corpus": {key: row["behavior"]["raw4_tail16"]["median_kl"] for key, row in results.items()},
        },
        "T4_raw4_plus_tail64_is_high_fidelity": {
            "supported": all(row["behavior"]["raw4_tail64"]["median_kl"] <= 0.005 for row in results.values()),
            "per_corpus": {key: row["behavior"]["raw4_tail64"]["median_kl"] for key, row in results.items()},
        },
        "T5_tail16_is_stable_and_compact": {
            "supported": all(
                row["rank_rows"]["16"]["tail_mean_canonical_cosine"] >= 0.70
                and row["rank_rows"]["16"]["tail_variance_efficiency"] >= 0.75
                for row in results.values()
            ),
        },
        "T6_learned_tail_beats_random_tail": {
            "supported": all(
                row["behavior"]["raw4_tail16"]["median_kl"]
                <= 0.5 * row["behavior"]["raw4_random16"]["median_kl"]
                for row in results.values()
            ),
        },
        "T7_weight_anchored_program_beats_unconstrained_pca": {
            "supported": all(
                row["behavior"]["raw4_tail16"]["median_kl"]
                <= row["behavior"]["natural20"]["median_kl"]
                for row in results.values()
            ),
        },
        "T8_full_projection_is_identity": {
            "supported": all(row["behavior"]["full"]["median_absolute_kl"] <= 1e-7 for row in results.values()),
            "per_corpus": {key: row["behavior"]["full"]["median_absolute_kl"] for key, row in results.items()},
        },
    }

    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "discovery": {
            "sites": len(centered_discovery),
            "raw4_variance_fraction": float(np.square(raw_part).sum() / np.square(centered_discovery).sum()),
            "tail_effective_rank": effective_rank(tail_values),
            "tail_top16_fraction": float(tail_values[:16].sum() / max(tail_values.sum(), 1e-30)),
        },
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Independent Layer-11 spectral bottleneck and residual-tail replication",
        "",
        "Four output directions are frozen from the complete bilinear weight tensor. An 8,192-state FineWeb panel supplies the post-write mean, an orthogonal residual-tail PCA basis, and an unconstrained natural PCA control. Disjoint FineWeb, WikiText, and TinyStories panels test transfer. Causal projections use the full post-bias MLP write and recompute final RMSNorm and unembedding exactly.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-11 bottleneck and tail](bilin12_layer11_bottleneck_tail_replication.png)",
        "",
        "| Corpus | Raw-4 variance / optimal efficiency | Tail effective rank | Tail top-16 fraction | Tail-16 transfer cosine / efficiency |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        tail16 = row["rank_rows"]["16"]
        lines.append(
            f"| {corpus} | {row['raw4_total_variance_fraction']:.3f} / {row['raw4_optimal_variance_efficiency']:.3f} | "
            f"{row['tail_effective_rank']:.1f} | {row['tail_top16_fraction']:.3f} | "
            f"{tail16['tail_mean_canonical_cosine']:.3f} / {tail16['tail_variance_efficiency']:.3f} |"
        )
    lines += [
        "",
        "| Corpus | Raw-4 KL | + tail-16 | + tail-64 | Natural-20 | Raw-4 + random-16 | Full median abs. KL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        behavior = row["behavior"]
        lines.append(
            f"| {corpus} | {behavior['raw4']['median_kl']:.5f} | {behavior['raw4_tail16']['median_kl']:.5f} | "
            f"{behavior['raw4_tail64']['median_kl']:.5f} | {behavior['natural20']['median_kl']:.5f} | "
            f"{behavior['raw4_random16']['median_kl']:.5f} | {behavior['full']['median_absolute_kl']:.2e} |"
        )
    lines += [
        "",
        "The decisive distinction is between a direction being unusually strong, naturally used, necessary, and sufficient. A low-dimensional outlier space can be real and causally important while remaining far from a sufficient program. The learned residual-tail comparison asks whether the remaining computation is compact and transferable; the natural-PCA and random-tail controls test whether weight anchoring adds anything beyond ordinary write covariance.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(results)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(len(corpora))
    axes[0].bar(x - 0.18, [results[c]["raw4_total_variance_fraction"] for c in corpora], 0.36, label="raw-4")
    axes[0].bar(x + 0.18, [results[c]["raw4_optimal_variance_efficiency"] for c in corpora], 0.36, label="efficiency")
    axes[0].set_xticks(x, corpora)
    axes[0].set_title("Natural write alignment")
    axes[0].set_ylabel("fraction")
    axes[0].legend()
    ranks = config["tail_ranks"]
    for corpus in corpora:
        axes[1].plot(ranks, [results[corpus]["rank_rows"][str(rank)]["combined_total_variance_fraction"] for rank in ranks], marker="o", label=corpus)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(ranks, ranks)
    axes[1].set_title("Raw-4 + tail variance")
    axes[1].set_xlabel("tail rank")
    axes[1].set_ylabel("variance fraction")
    axes[1].legend()
    for corpus in corpora:
        axes[2].plot(ranks, [results[corpus]["behavior"][f"raw4_tail{rank}"]["median_kl"] for rank in ranks], marker="o", label=corpus)
    axes[2].axhline(0.02, color="black", linestyle="--", linewidth=1, label="near-sufficient gate")
    axes[2].axhline(0.005, color="gray", linestyle=":", linewidth=1, label="high-fidelity gate")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_xticks(ranks, ranks)
    axes[2].set_title("Exact final-distribution frontier")
    axes[2].set_xlabel("tail rank")
    axes[2].set_ylabel("median KL")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
