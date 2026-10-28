#!/usr/bin/env python3
"""Separate the global Layer-17 spectral bottleneck from its contextual tail.

The four output directions are selected only from the complete bilinear weight
tensor.  A disjoint FineWeb panel supplies the write mean and residual-tail
principal directions.  Frozen objects are then evaluated on disjoint FineWeb,
WikiText, and TinyStories for variance, structural-token decodability, and exact
final-layer interventions.  The random causal control is matched per site for
the norm of the removed tail-16 write; it is not presented as an on-manifold
ablation.
"""

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
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_bottleneck_tail_ontology.json")
OUTPUT = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.json"
ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
REPORT = TENSOR_ROOT / "LAYER17_BOTTLENECK_TAIL_ONTOLOGY.md"
FIGURE = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.png"


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


def fit_lda_readout(features: np.ndarray, labels: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    mean = features.mean(0)
    scale = features.std(0).clip(1e-8)
    standardized = (features - mean) / scale
    positive = standardized[labels]
    negative = standardized[~labels]
    difference = positive.mean(0) - negative.mean(0)
    covariance = (
        (positive - positive.mean(0)).T @ (positive - positive.mean(0))
        + (negative - negative.mean(0)).T @ (negative - negative.mean(0))
    ) / max(len(features) - 2, 1)
    weight = np.linalg.solve(covariance + ridge * np.eye(features.shape[1]), difference)
    if float(((standardized @ weight)[labels]).mean()) < float(((standardized @ weight)[~labels]).mean()):
        weight = -weight
    return {"mean": mean, "scale": scale, "weight": weight}


def make_site_labels(
    rows: np.ndarray,
    positions: list[int],
    flags: np.ndarray,
    eos_id: int,
    category_names: list[str] | tuple[str, ...] = CATEGORIES,
) -> dict[str, np.ndarray]:
    positions_array = np.asarray(positions)
    current = rows[:, positions_array].reshape(-1)
    following = rows[:, positions_array + 1].reshape(-1)
    labels: dict[str, np.ndarray] = {}
    for prefix, ids in (("current", current), ("next", following)):
        selected = flags[ids]
        for index, name in enumerate(category_names):
            labels[f"{prefix}/{name}"] = selected[:, index].astype(bool)
        labels[f"{prefix}/eos"] = ids == eos_id
    return labels


def corpus_rows(spec: dict) -> np.ndarray:
    source = np.load(spec["path"], mmap_mode="r")
    return np.asarray(source[spec["start"] : spec["start"] + spec["rows"]])


@torch.inference_mode()
def collect_writes(model, spec: dict, config: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = corpus_rows(spec)
    positions = config["positions"]
    result = np.empty((len(rows) * len(positions), model.config.n_embd), dtype=np.float32)
    block = model.blocks[config["layer"]]
    offset = 0
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, normalized = prefix_to_layer17(model, ids)
        selected = normalized[:, positions].float().flatten(0, 1)
        write = block.mlp.Down(block.mlp.Left(selected) * block.mlp.Right(selected))
        count = len(write)
        result[offset : offset + count] = write.cpu().numpy()
        offset += count
    return result, {"rows": rows}


def fit_semantic_decoders(
    representations: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    config: dict,
) -> dict[str, dict[str, dict]]:
    result = {}
    minimum = config["minimum_positive_examples"]
    for representation_name, features in representations.items():
        readouts = {}
        for label_name, target in labels.items():
            positive = int(target.sum())
            negative = len(target) - positive
            if min(positive, negative) < minimum:
                continue
            decoder = fit_lda_readout(features, target, config["lda_ridge"])
            readouts[label_name] = {
                "mean": decoder["mean"].tolist(),
                "scale": decoder["scale"].tolist(),
                "weight": decoder["weight"].tolist(),
                "discovery_positive": positive,
            }
        result[representation_name] = readouts
    return result


def evaluate_semantic_decoders(
    representations: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    decoders: dict[str, dict[str, dict]],
    minimum: int,
) -> dict[str, dict]:
    output = {}
    for representation_name, readouts in decoders.items():
        rows = {}
        for label_name, readout in readouts.items():
            target = labels[label_name]
            positive = int(target.sum())
            if min(positive, len(target) - positive) < minimum:
                continue
            features = representations[representation_name]
            score = (
                (features - np.asarray(readout["mean"])) / np.asarray(readout["scale"])
            ) @ np.asarray(readout["weight"])
            rows[label_name] = {"auc": auc(target, score), "positive": positive}
        aucs = [row["auc"] for row in rows.values()]
        output[representation_name] = {
            "features": rows,
            "median_auc": float(np.median(aucs)) if aucs else None,
            "strong_count": sum(row["auc"] >= 0.70 for row in rows.values()),
        }
    return output


def variant_metrics(records: dict[str, list[float]]) -> dict[str, float]:
    return {
        "median_kl": float(np.median(records["kl"])),
        "mean_kl": float(np.mean(records["kl"])),
        "ce_delta": float(np.mean(records["ce_delta"])),
        "top1_agreement": float(np.mean(records["top1"])),
        "median_relative_edit_norm": float(np.median(records["relative_edit_norm"])),
    }


@torch.inference_mode()
def causal_variants(
    model,
    spec: dict,
    config: dict,
    write_mean: np.ndarray,
    outliers: np.ndarray,
    tail16: np.ndarray,
    random16: np.ndarray,
) -> dict[str, dict[str, float]]:
    rows = corpus_rows(spec)
    positions = config["positions"]
    block = model.blocks[config["layer"]]
    dimension = model.config.n_embd
    mean = torch.as_tensor(write_mean, device="cuda")
    outlier = torch.as_tensor(outliers, device="cuda")
    tail = torch.as_tensor(tail16, device="cuda")
    random = torch.as_tensor(random16, device="cuda")
    bias = block.mlp.Down_bias.float()
    names = (
        "outlier_only", "tail_only", "outlier_plus_tail16",
        "remove_tail16", "norm_matched_random16", "mean_only",
    )
    records = {name: {key: [] for key in ("kl", "ce_delta", "top1", "relative_edit_norm")} for name in names}
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1)
        normalized = normalized[:, positions].float().flatten(0, 1)
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        raw_write = block.mlp.Down(product).float()
        native_write = raw_write + bias
        centered = raw_write - mean
        outlier_part = (centered @ outlier) @ outlier.T
        tail_part = centered - outlier_part
        tail16_part = (tail_part @ tail) @ tail.T
        random_part = (tail_part @ random) @ random.T
        random_part = random_part * (
            tail16_part.norm(dim=1, keepdim=True)
            / random_part.norm(dim=1, keepdim=True).clamp_min(1e-20)
        )
        variants = {
            "outlier_only": mean + outlier_part + bias,
            "tail_only": mean + tail_part + bias,
            "outlier_plus_tail16": mean + outlier_part + tail16_part + bias,
            "remove_tail16": native_write - tail16_part,
            "norm_matched_random16": native_write - random_part,
            "mean_only": mean.expand_as(centered) + bias,
        }
        native_logits = 30 * torch.tanh(
            model.lm_head(F.rms_norm(pre + native_write, (dimension,))).float() / 30
        )
        native_logprob = native_logits.log_softmax(-1)
        native_probability = native_logprob.exp()
        target = torch.as_tensor(
            rows[start : start + len(raw), np.asarray(positions) + 1].reshape(-1),
            device="cuda",
            dtype=torch.long,
        )
        native_ce = -native_logprob.gather(1, target[:, None]).squeeze(1)
        native_top1 = native_logits.argmax(-1)
        denominator = native_write.norm(dim=1).clamp_min(1e-20)
        for name, changed_write in variants.items():
            changed_logits = 30 * torch.tanh(
                model.lm_head(F.rms_norm(pre + changed_write, (dimension,))).float() / 30
            )
            changed_logprob = changed_logits.log_softmax(-1)
            kl = (native_probability * (native_logprob - changed_logprob)).sum(-1)
            changed_ce = -changed_logprob.gather(1, target[:, None]).squeeze(1)
            records[name]["kl"].extend(kl.cpu().tolist())
            records[name]["ce_delta"].extend((changed_ce - native_ce).cpu().tolist())
            records[name]["top1"].extend((changed_logits.argmax(-1) == native_top1).cpu().tolist())
            records[name]["relative_edit_norm"].extend(
                ((changed_write - native_write).norm(dim=1) / denominator).cpu().tolist()
            )
    return {name: variant_metrics(row) for name, row in records.items()}


def next_feature_recurrence(semantics: dict[str, dict], representation: str) -> list[str]:
    common = None
    for corpus_rows in semantics.values():
        features = corpus_rows[representation]["features"]
        strong = {name for name, row in features.items() if name.startswith("next/") and row["auc"] >= 0.70}
        common = strong if common is None else common & strong
    return sorted(common or set())


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    token_flags_array = np.stack([token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    eos_id = tokenizer.eos_token_id

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[config["layer"]]
    _, output_vectors = output_map_eigensystem(
        block.mlp.Down.weight.float(),
        block.mlp.Left.weight.float(),
        block.mlp.Right.weight.float(),
    )
    outliers = output_vectors[:, : config["outlier_rank"]].cpu().numpy().astype(np.float64)

    print("FineWeb discovery writes", flush=True)
    discovery_write, discovery_meta = collect_writes(model, config["discovery"], config)
    write_mean = discovery_write.mean(0).astype(np.float64)
    discovery_centered = discovery_write.astype(np.float64) - write_mean
    discovery_outlier, discovery_tail = project_complement(discovery_centered, outliers)
    tail_values, tail_vectors = covariance_eigensystem(discovery_tail)
    full_values, full_vectors = covariance_eigensystem(discovery_centered)
    max_tail_rank = max(config["tail_ranks"])
    tail_basis = tail_vectors[:, :max_tail_rank]
    natural_basis = full_vectors[:, : config["semantic_control_rank"]]
    generator = np.random.default_rng(config["seed"])
    random_matrix = generator.normal(size=(model.config.n_embd, 16))
    random_matrix -= outliers @ (outliers.T @ random_matrix)
    random16, _ = np.linalg.qr(random_matrix)

    discovery_labels = make_site_labels(
        discovery_meta["rows"], config["positions"], token_flags_array, eos_id
    )
    discovery_representations = {
        "outlier4": discovery_centered @ outliers,
        "tail4": discovery_tail @ tail_basis[:, :4],
        "tail16": discovery_tail @ tail_basis[:, :16],
        "natural64": discovery_centered @ natural_basis,
    }
    decoders = fit_semantic_decoders(discovery_representations, discovery_labels, config)

    arrays = {
        "outlier_output_basis": outliers.astype(np.float32),
        "tail_output_basis": tail_basis.astype(np.float32),
        "natural_output_basis": natural_basis.astype(np.float32),
        "write_mean": write_mean.astype(np.float32),
        "tail_eigenvalues": tail_values[:256].astype(np.float32),
        "full_eigenvalues": full_values[:256].astype(np.float32),
    }
    results = {}
    semantics = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluation", corpus, flush=True)
        write, metadata = collect_writes(model, spec, config)
        centered = write.astype(np.float64) - write_mean
        outlier_part, tail = project_complement(centered, outliers)
        covariance = centered.T @ centered / len(centered)
        local_values, local_vectors = covariance_eigensystem(centered - (centered @ outliers) @ outliers.T)
        optimal4 = np.linalg.eigvalsh(covariance)[-config["outlier_rank"] :].sum()
        captured4 = np.trace(outliers.T @ covariance @ outliers)
        tail_rows = {}
        for rank in config["tail_ranks"]:
            frozen = tail_basis[:, :rank]
            local = local_vectors[:, :rank]
            canonical = np.linalg.svd(frozen.T @ local, compute_uv=False)
            captured = np.trace(frozen.T @ (tail.T @ tail / len(tail)) @ frozen)
            optimal = local_values[:rank].sum()
            tail_rows[str(rank)] = {
                "mean_canonical_cosine": float(canonical.mean()),
                "minimum_canonical_cosine": float(canonical.min()),
                "variance_efficiency": float(captured / max(optimal, 1e-30)),
            }
        total_variance = float(np.trace(covariance))
        tail_variance = float(np.square(tail).sum() / len(tail))
        result = {
            "sites": len(centered),
            "outlier4_optimal_variance_efficiency": float(captured4 / max(optimal4, 1e-30)),
            "outlier4_total_variance_fraction": float(captured4 / max(total_variance, 1e-30)),
            "tail_total_variance_fraction": float(tail_variance / max(total_variance, 1e-30)),
            "tail_effective_rank": effective_rank(local_values),
            "tail_top4_fraction": float(local_values[:4].sum() / max(local_values.sum(), 1e-30)),
            "tail_top16_fraction": float(local_values[:16].sum() / max(local_values.sum(), 1e-30)),
            "tail_transfer": tail_rows,
        }
        labels = make_site_labels(metadata["rows"], config["positions"], token_flags_array, eos_id)
        representations = {
            "outlier4": centered @ outliers,
            "tail4": tail @ tail_basis[:, :4],
            "tail16": tail @ tail_basis[:, :16],
            "natural64": centered @ natural_basis,
        }
        semantics[corpus] = evaluate_semantic_decoders(
            representations, labels, decoders, config["minimum_positive_examples"]
        )
        print(" causal", corpus, flush=True)
        result["causal"] = causal_variants(
            model, spec, config, write_mean.astype(np.float32), outliers.astype(np.float32),
            tail_basis[:, :16].astype(np.float32), random16.astype(np.float32),
        )
        results[corpus] = result

    outlier_recurrent = next_feature_recurrence(semantics, "outlier4")
    tail_recurrent = next_feature_recurrence(semantics, "tail16")
    hypotheses = {
        "B1_weight_outliers_match_natural_bottleneck": {
            "supported": all(row["outlier4_optimal_variance_efficiency"] >= 0.85 for row in results.values()),
            "per_corpus": {key: row["outlier4_optimal_variance_efficiency"] for key, row in results.items()},
        },
        "B2_outlier_only_is_behaviorally_sufficient": {
            "supported": all(row["causal"]["outlier_only"]["median_kl"] <= 0.02 for row in results.values()),
            "per_corpus": {key: row["causal"]["outlier_only"]["median_kl"] for key, row in results.items()},
        },
        "B3_outliers_plus_tail16_are_near_sufficient": {
            "supported": all(row["causal"]["outlier_plus_tail16"]["median_kl"] <= 0.005 for row in results.values()),
            "per_corpus": {key: row["causal"]["outlier_plus_tail16"]["median_kl"] for key, row in results.items()},
        },
        "B4_outlier_removal_is_more_damaging_than_tail_removal": {
            "supported": all(
                row["causal"]["tail_only"]["median_kl"]
                >= 2 * row["causal"]["outlier_only"]["median_kl"]
                for row in results.values()
            ),
        },
        "B5_tail16_is_cross_corpus_stable": {
            "supported": all(
                row["tail_transfer"]["16"]["mean_canonical_cosine"] >= 0.70
                and row["tail_transfer"]["16"]["variance_efficiency"] >= 0.75
                for row in results.values()
            ),
        },
        "B6_tail16_beats_norm_matched_random_edits": {
            "supported": all(
                row["causal"]["remove_tail16"]["median_kl"]
                >= 2 * row["causal"]["norm_matched_random16"]["median_kl"]
                for row in results.values()
            ),
        },
        "B7_outlier_subspace_has_recurrent_next_token_roles": {
            "supported": len(outlier_recurrent) >= 3,
            "features": outlier_recurrent,
        },
        "B8_tail_subspace_has_recurrent_next_token_roles": {
            "supported": len(tail_recurrent) >= 3,
            "features": tail_recurrent,
        },
    }
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "discovery": {
            "sites": len(discovery_centered),
            "outlier_variance_fraction": float(np.square(discovery_outlier).sum() / np.square(discovery_centered).sum()),
            "tail_variance_fraction": float(np.square(discovery_tail).sum() / np.square(discovery_centered).sum()),
            "tail_effective_rank": effective_rank(tail_values),
            "tail_top16_fraction": float(tail_values[:16].sum() / tail_values.sum()),
        },
        "results": results,
        "semantic_decoders": decoders,
        "semantics": semantics,
    }, indent=2, allow_nan=False))

    lines = [
        "# Layer-17 spectral bottleneck and contextual-tail ontology",
        "",
        "The four output directions are frozen from the complete bilinear weight tensor. A 16,000-site FineWeb panel supplies only the write mean and residual-tail basis. Disjoint FineWeb, WikiText, and TinyStories panels test variance, simple structural-token readouts, and exact post-MLP interventions through final RMSNorm and the unembedding.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Bottleneck and tail](layer17_bottleneck_tail_ontology.png)",
        "",
        "| Corpus | Outlier-4 variance efficiency | Total variance in outlier / tail | Tail eff. rank | Tail-16 transfer cosine / efficiency |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        transfer = row["tail_transfer"]["16"]
        lines.append(
            f"| {corpus} | {row['outlier4_optimal_variance_efficiency']:.3f} | "
            f"{row['outlier4_total_variance_fraction']:.3f} / {row['tail_total_variance_fraction']:.3f} | "
            f"{row['tail_effective_rank']:.1f} | {transfer['mean_canonical_cosine']:.3f} / {transfer['variance_efficiency']:.3f} |"
        )
    lines += [
        "",
        "| Corpus | Keep outlier-4 KL | Keep outlier-4 + tail-16 KL | Remove outlier-4 KL | Remove tail-16 KL | Norm-matched random KL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        causal = row["causal"]
        lines.append(
            f"| {corpus} | {causal['outlier_only']['median_kl']:.5f} | "
            f"{causal['outlier_plus_tail16']['median_kl']:.5f} | {causal['tail_only']['median_kl']:.5f} | "
            f"{causal['remove_tail16']['median_kl']:.5f} | {causal['norm_matched_random16']['median_kl']:.5f} |"
        )
    lines += [
        "",
        "| Corpus | Outlier-4 median structural AUROC | Tail-4 | Tail-16 | Natural-64 |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, row in semantics.items():
        lines.append(
            f"| {corpus} | {row['outlier4']['median_auc']:.3f} | {row['tail4']['median_auc']:.3f} | "
            f"{row['tail16']['median_auc']:.3f} | {row['natural64']['median_auc']:.3f} |"
        )
    lines += [
        "",
        f"Recurrent next-token roles decoded from outlier-4: **{', '.join(outlier_recurrent) or 'none'}**.",
        f"Recurrent next-token roles decoded from tail-16: **{', '.join(tail_recurrent) or 'none'}**.",
        "",
        "`outlier_only` and `outlier_plus_tail16` are output-write replacements around the frozen FineWeb mean, not input-side replacements. `tail_only` removes the four-dimensional outlier component. The random-16 edit is matched per site to the removed tail-16 output norm, but is off-manifold; it controls output sensitivity rather than serving as another native circuit. Variance, semantic decodability, causal necessity, and behavioral sufficiency are therefore reported as separate definitions of importance.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    corpora = list(results)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].bar(corpora, [results[c]["outlier4_total_variance_fraction"] for c in corpora], label="outlier-4")
    axes[0].bar(corpora, [results[c]["tail_total_variance_fraction"] for c in corpora], bottom=[results[c]["outlier4_total_variance_fraction"] for c in corpora], label="tail")
    axes[0].set_title("Centered write variance")
    axes[0].set_ylabel("fraction")
    axes[0].legend()
    x = np.arange(len(corpora))
    width = 0.24
    for offset, (name, label) in enumerate((("outlier_only", "keep outlier-4"), ("outlier_plus_tail16", "+ tail-16"), ("tail_only", "remove outlier-4"))):
        axes[1].bar(x + (offset - 1) * width, [results[c]["causal"][name]["median_kl"] for c in corpora], width, label=label)
    axes[1].set_xticks(x, corpora)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median KL")
    axes[1].set_title("Behavioral decomposition")
    axes[1].legend(fontsize=8)
    for name in ("outlier4", "tail4", "tail16", "natural64"):
        axes[2].plot(corpora, [semantics[c][name]["median_auc"] for c in corpora], marker="o", label=name)
    axes[2].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[2].set_ylim(0.45, 1.0)
    axes[2].set_ylabel("median held-out AUROC")
    axes[2].set_title("Structural-token readouts")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
