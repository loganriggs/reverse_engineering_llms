#!/usr/bin/env python3
"""Decode the basis-invariant state-conditioned Layer-8 behavioral bottleneck."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER, classify_token, CLASSES
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_direct_tensor_jacobian import (
    analytic_suffix,
    bilinear_mlp_jvp,
    output_fisher_response,
)
from run_bilin18_layer8_full_tensor_gram_rank import gram_eigensystem
from run_bilin18_layer8_router_causal_validation import prefix_to_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_behavioral_bottleneck.json")
OUTPUT = TENSOR_ROOT / "layer8_behavioral_bottleneck.json"
REPORT = TENSOR_ROOT / "LAYER8_BEHAVIORAL_BOTTLENECK.md"
FIGURE = TENSOR_ROOT / "layer8_behavioral_bottleneck.png"


def rank_fractions(values: torch.Tensor, ranks: list[int]) -> dict[str, float]:
    total = values.sum().clamp_min(1e-30)
    return {str(rank): float(values[:rank].sum() / total) for rank in ranks}


def normalized_rows(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-30)


def orthonormal_row_space(values: torch.Tensor) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(values.float(), full_matrices=False)
    return vh[: values.shape[0]]


def subspace_overlap(first: np.ndarray, second: np.ndarray) -> float:
    singular = np.linalg.svd(first @ second.T, compute_uv=False)
    return float(np.square(singular).mean())


def best_axis_alignment(first: np.ndarray, second: np.ndarray) -> tuple[float, tuple[int, ...]]:
    similarity = np.abs(first @ second.T)
    best_score, best_permutation = -1.0, tuple(range(len(first)))
    for permutation in itertools.permutations(range(len(first))):
        score = float(np.mean([similarity[index, permutation[index]] for index in range(len(first))]))
        if score > best_score:
            best_score, best_permutation = score, permutation
    return best_score, best_permutation


def median(values) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def pure_softmax_lower_bound(probability: torch.Tensor, ranks: list[int], tokens: int):
    selected = torch.topk(probability, min(tokens, len(probability))).values.double()
    principal = torch.diag(selected) - selected[:, None] * selected[None, :]
    values = torch.linalg.eigvalsh(principal).flip(0).clamp_min(0)
    full_trace = (1 - probability.double().square().sum()).clamp_min(1e-30)
    return {str(rank): float(values[:rank].sum() / full_trace) for rank in ranks}


def decode_axes(tokenizer, logits, logit_tangents, gram_vectors, response, count, top_tokens):
    probability = F.softmax(logits.float(), -1)
    centered = logit_tangents - (logit_tangents * probability[None]).sum(-1, keepdim=True)
    logit_axes = gram_vectors[:, :count].T @ centered
    fisher_axes = normalized_rows((response @ gram_vectors[:, :count]).T)
    rows = []
    for axis in range(count):
        logit_axis = logit_axes[axis]
        probability_delta = probability * logit_axis
        dominant = int(probability_delta.abs().argmax())
        if probability_delta[dominant] < 0:
            logit_axis = -logit_axis
            probability_delta = -probability_delta
            fisher_axes[axis] = -fisher_axes[axis]
        positive = torch.topk(probability_delta, top_tokens).indices.tolist()
        negative = torch.topk(-probability_delta, top_tokens).indices.tolist()
        rows.append({
            "axis": axis,
            "promoted": [tokenizer.decode([index]) for index in positive],
            "promoted_ids": positive,
            "promoted_probability_derivatives": [float(probability_delta[index]) for index in positive],
            "suppressed": [tokenizer.decode([index]) for index in negative],
            "suppressed_ids": negative,
            "suppressed_probability_derivatives": [float(probability_delta[index]) for index in negative],
        })
    logit_subspace = orthonormal_row_space(logit_axes).cpu().numpy()
    return rows, fisher_axes.cpu().numpy(), logit_subspace


def pair_summary(records):
    overlaps, logit_overlaps, input_overlaps, alignments, jaccards = [], [], [], [], []
    same_class, different_class = [], []
    same_class_logit, different_class_logit = [], []
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            first, second = records[first_index], records[second_index]
            overlap = subspace_overlap(first["_fisher_axes"], second["_fisher_axes"])
            logit_overlap = subspace_overlap(first["_logit_subspace"], second["_logit_subspace"])
            input_overlap = subspace_overlap(first["_input_axes"], second["_input_axes"])
            alignment, permutation = best_axis_alignment(first["_fisher_axes"], second["_fisher_axes"])
            token_scores = []
            for axis, target in enumerate(permutation):
                first_tokens = set(first["_token_sets"][axis])
                second_tokens = set(second["_token_sets"][target])
                token_scores.append(len(first_tokens & second_tokens) / max(len(first_tokens | second_tokens), 1))
            overlaps.append(overlap)
            logit_overlaps.append(logit_overlap)
            input_overlaps.append(input_overlap)
            alignments.append(alignment)
            jaccards.append(float(np.mean(token_scores)))
            (same_class if first["prediction_class"] == second["prediction_class"] else different_class).append(overlap)
            (same_class_logit if first["prediction_class"] == second["prediction_class"] else different_class_logit).append(logit_overlap)
    return {
        "median_subspace_overlap": median(overlaps),
        "median_unweighted_logit_subspace_overlap": median(logit_overlaps),
        "median_input_subspace_overlap": median(input_overlaps),
        "median_best_axis_cosine": median(alignments),
        "median_top_token_jaccard": median(jaccards),
        "same_prediction_class_overlap": median(same_class),
        "different_prediction_class_overlap": median(different_class),
        "prediction_class_advantage": median(same_class) - median(different_class),
        "same_prediction_class_logit_overlap": median(same_class_logit),
        "different_prediction_class_logit_overlap": median(different_class_logit),
        "prediction_class_logit_advantage": median(same_class_logit) - median(different_class_logit),
        "pairs": len(overlaps),
        "same_class_pairs": len(same_class),
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    tensor_config = {**source_config, "positions": source_config["discovery_positions"]}
    discovery_x, _ = collect(model, 8, source_config["tensor_discovery"], tensor_config)
    _, _, tensor_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = tensor_vh.T[:, :source_config["dimension"]]
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    rotation = torch.linalg.qr(torch.randn(
        source_config["dimension"], source_config["dimension"],
        generator=generator, device="cuda",
    )).Q
    rotated_basis = basis @ rotation
    block8 = model.blocks[8]
    position = source_config["causal_position"]
    results, private_records = {}, {}
    rotation_checks = []

    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(config["seed"] + corpus_index)
        site_indices = rng.choice(len(rows), size=config["sites_per_corpus"], replace=False)
        corpus_records = []
        for site_number, row_index in enumerate(site_indices):
            raw = np.asarray(rows[row_index, :position + 2], dtype=np.int64)
            ids = torch.as_tensor(raw[:position + 1], device="cuda")[None]
            x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
            native_write = block8.mlp.Down(
                block8.mlp.Left(normalized) * block8.mlp.Right(normalized)
            ) + block8.mlp.Down_bias
            x8 = pre + native_write
            x = normalized[0, position].float()
            write_tangents = bilinear_mlp_jvp(block8, x, basis.T)
            logits, logit_tangents, _ = analytic_suffix(
                model, x8, x0, value_bus, position, write_tangents, include_attention=True,
            )
            response = output_fisher_response(logits, logit_tangents)
            gram_values, gram_vectors = gram_eigensystem(response)
            fractions = rank_fractions(gram_values, config["ranks"])
            probability = F.softmax(logits.float(), -1)
            decoded, fisher_axes, logit_subspace = decode_axes(
                tokenizer, logits, logit_tangents, gram_vectors, response,
                config["decoded_axes"], config["top_tokens"],
            )
            native_top = torch.topk(probability, config["top_tokens"])
            record = {
                "row": int(row_index),
                "context": tokenizer.decode(raw[max(0, position - 24):position + 1].tolist()),
                "target": tokenizer.decode([int(raw[position + 1])]),
                "native_top": [tokenizer.decode([int(index)]) for index in native_top.indices],
                "native_top_probabilities": [float(value) for value in native_top.values],
                "prediction_class": CLASSES[classify_token(tokenizer.decode([int(native_top.indices[0])]))],
                "target_class": CLASSES[classify_token(tokenizer.decode([int(raw[position + 1])]))],
                "entropy_nats": float(-(probability * probability.clamp_min(1e-30).log()).sum()),
                "effective_probability_support": float(1 / probability.square().sum()),
                "rank_energy": fractions,
                "softmax_fisher_lower_bound": pure_softmax_lower_bound(
                    probability, config["ranks"], config["softmax_principal_tokens"],
                ),
                "axes": decoded,
                "_fisher_axes": fisher_axes,
                "_logit_subspace": logit_subspace,
                "_input_axes": gram_vectors[:, :config["decoded_axes"]].T.cpu().numpy(),
                "_token_sets": [row["promoted_ids"] + row["suppressed_ids"] for row in decoded],
            }
            if site_number < config["rotation_check_sites"]:
                rotated_tangents = bilinear_mlp_jvp(block8, x, rotated_basis.T)
                _, rotated_logits_tangents, _ = analytic_suffix(
                    model, x8, x0, value_bus, position, rotated_tangents, include_attention=True,
                )
                rotated_response = output_fisher_response(logits, rotated_logits_tangents)
                rotated_values, rotated_vectors = gram_eigensystem(rotated_response)
                rotated_axes = normalized_rows(
                    (rotated_response @ rotated_vectors[:, :config["decoded_axes"]]).T
                ).cpu().numpy()
                rotation_checks.append({
                    "corpus": corpus,
                    "row": int(row_index),
                    "maximum_rank_energy_difference": max(
                        abs(fractions[str(rank)] - rank_fractions(rotated_values, config["ranks"])[str(rank)])
                        for rank in config["ranks"]
                    ),
                    "leading_subspace_overlap": subspace_overlap(fisher_axes, rotated_axes),
                })
            corpus_records.append(record)
            print(corpus, site_number + 1, "/", len(site_indices), flush=True)
        private_records[corpus] = corpus_records
        summary = {
            "rank_energy": {
                str(rank): median([row["rank_energy"][str(rank)] for row in corpus_records])
                for rank in config["ranks"]
            },
            "softmax_fisher_lower_bound": {
                str(rank): median([row["softmax_fisher_lower_bound"][str(rank)] for row in corpus_records])
                for rank in config["ranks"]
            },
            "entropy_nats": median([row["entropy_nats"] for row in corpus_records]),
            "effective_probability_support": median([
                row["effective_probability_support"] for row in corpus_records
            ]),
            "recurrence": pair_summary(corpus_records),
        }
        public_records = []
        for record in corpus_records:
            public_records.append({key: value for key, value in record.items() if not key.startswith("_")})
        results[corpus] = {"summary": summary, "records": public_records}

    cross_pairs = []
    for first in private_records["fineweb"]:
        for second in private_records["wikitext"]:
            cross_pairs.append(subspace_overlap(first["_fisher_axes"], second["_fisher_axes"]))
    cross_corpus_overlap = median(cross_pairs)
    max_rotation_difference = max(row["maximum_rank_energy_difference"] for row in rotation_checks)
    min_rotation_overlap = min(row["leading_subspace_overlap"] for row in rotation_checks)
    hypotheses = {
        "C1_unit_input_rank16_is_compact": {"supported": all(results[c]["summary"]["rank_energy"]["16"] >= 0.90 for c in results)},
        "C2_unit_input_rank4_is_compact": {"supported": all(results[c]["summary"]["rank_energy"]["4"] >= 0.75 for c in results)},
        "C3_basis_rotation_invariance": {"supported": max_rotation_difference <= 1e-4 and min_rotation_overlap >= 0.999},
        "C4_axes_recur_globally": {"supported": all(results[c]["summary"]["recurrence"]["median_subspace_overlap"] >= 0.25 for c in results)},
        "C5_prediction_class_explains_recurrence": {"supported": all(results[c]["summary"]["recurrence"]["prediction_class_advantage"] >= 0.05 for c in results)},
        "C6_token_labels_recur": {"supported": all(results[c]["summary"]["recurrence"]["median_top_token_jaccard"] >= 0.10 for c in results)},
        "C7_softmax_geometry_can_support_rank4": {"supported": all(
            results[c]["summary"]["softmax_fisher_lower_bound"]["4"]
            >= results[c]["summary"]["rank_energy"]["4"] - 0.10 for c in results
        )},
        "C8_unweighted_logit_axes_recur": {"supported": all(
            results[c]["summary"]["recurrence"]["median_unweighted_logit_subspace_overlap"] >= 0.25
            for c in results
        )},
        "C9_input_verbs_recur": {"supported": all(
            results[c]["summary"]["recurrence"]["median_input_subspace_overlap"] >= 0.25
            for c in results
        )},
    }
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "rotation_checks": rotation_checks,
        "cross_corpus_subspace_overlap": cross_corpus_overlap,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Basis-invariant Layer-8 behavioral bottleneck",
        "",
        "The previous query-conditioned Gram used a complete 512-direction eigenbasis, so readout changes mostly changed coordinates rather than the underlying tangent span. This correction applies unit perturbations in a fixed natural-state basis, propagates the exact bilinear MLP and tensor-attention Jacobian, and eigendecomposes the resulting full-vocabulary Fisher response. Orthogonal basis rotations are checked explicitly.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Behavioral bottleneck](layer8_behavioral_bottleneck.png)",
        "",
        "| Corpus | Rank-4 / rank-16 energy | Pure-softmax rank-4 lower bound | Fisher/logit/input site overlap | Matched-axis cosine | Token Jaccard | Same-class advantage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        summary, recurrence = result["summary"], result["summary"]["recurrence"]
        lines.append(
            f"| {corpus} | {summary['rank_energy']['4']:.3f} / {summary['rank_energy']['16']:.3f} | "
            f"{summary['softmax_fisher_lower_bound']['4']:.3f} | {recurrence['median_subspace_overlap']:.3f} / {recurrence['median_unweighted_logit_subspace_overlap']:.3f} / {recurrence['median_input_subspace_overlap']:.3f} | "
            f"{recurrence['median_best_axis_cosine']:.3f} | {recurrence['median_top_token_jaccard']:.3f} | "
            f"{recurrence['prediction_class_advantage']:.3f} |"
        )
    lines += [
        "",
        f"The median cross-corpus rank-4 subspace overlap is **{cross_corpus_overlap:.3f}**. The worst rotation energy discrepancy is **{max_rotation_difference:.2e}**, and the worst rotated-basis output-subspace overlap is **{min_rotation_overlap:.6f}**.",
        "",
        "## Concrete decoded axes",
        "",
    ]
    for corpus in results:
        for record in results[corpus]["records"][:2]:
            lines += [
                f"### {corpus}, row {record['row']}",
                "",
                f"Context: `{record['context'].replace('`', "'")}`",
                "",
                f"Native prediction: `{record['native_top'][0]}` ({record['native_top_probabilities'][0]:.3f}); corpus target: `{record['target']}`.",
                "",
            ]
            for axis in record["axes"]:
                promoted = ", ".join(repr(token) for token in axis["promoted"][:5])
                suppressed = ", ".join(repr(token) for token in axis["suppressed"][:5])
                lines.append(f"- Axis {axis['axis']}: promotes {promoted}; suppresses {suppressed}.")
            lines.append("")
    lines += [
        "## Interpretation",
        "",
        "Compactness and semantic recurrence are separate. A low-rank state-conditioned output space can be a short behavioral interface even when its axes rotate with the local next-token alternatives. Stable semantics require recurrence or conditional alignment beyond this rank result.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ranks = config["ranks"]
    for corpus, result in results.items():
        axes[0].plot(ranks, [result["summary"]["rank_energy"][str(rank)] for rank in ranks], marker="o", label=f"{corpus}: model")
        axes[0].plot(ranks, [result["summary"]["softmax_fisher_lower_bound"][str(rank)] for rank in ranks], marker="x", linestyle="--", label=f"{corpus}: softmax lower bound")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel("median energy fraction")
    axes[0].set_title("State-conditioned rank curve")
    axes[0].legend(fontsize=8)
    corpora = list(results)
    x = np.arange(len(corpora))
    width = 0.25
    axes[1].bar(x - width, [results[c]["summary"]["recurrence"]["median_subspace_overlap"] for c in corpora], width, label="subspace")
    axes[1].bar(x, [results[c]["summary"]["recurrence"]["median_best_axis_cosine"] for c in corpora], width, label="axis cosine")
    axes[1].bar(x + width, [results[c]["summary"]["recurrence"]["median_top_token_jaccard"] for c in corpora], width, label="token Jaccard")
    axes[1].set_xticks(x, corpora)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Cross-site recurrence")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
