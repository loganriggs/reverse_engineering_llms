#!/usr/bin/env python3
"""Find split-stable source-role axes left unexplained by human token categories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_routing_primitive_factorization import decode_coordinate
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, token_categories


CONFIG = Path("configs/bilin18_unknown_source_feature_axes.json")
PROFILE_SOURCE = TENSOR_ROOT / "bilin18_causal_source_profiles.pt"
OUTPUT = TENSOR_ROOT / "bilin18_unknown_source_feature_axes.json"
REPORT = TENSOR_ROOT / "BILIN18_UNKNOWN_SOURCE_FEATURE_AXES.md"


def category_centroids(matrix, categories, weights, count):
    centroids = torch.zeros(count, matrix.shape[1], device=matrix.device)
    mass = torch.zeros(count, device=matrix.device)
    centroids.index_add_(0, categories, matrix * weights[:, None])
    mass.index_add_(0, categories, weights)
    return centroids / mass[:, None].clamp_min(1e-30)


def weighted_axes(residual, weights, count):
    valid = weights > 0
    weighted = residual[valid] * weights[valid].sqrt()[:, None]
    _, _, right = torch.linalg.svd(weighted, full_matrices=False)
    return right[:count]


def greedy_axis_alignment(first, second):
    similarity = F.normalize(first, dim=1) @ F.normalize(second, dim=1).T
    available_first = set(range(len(first))); available_second = set(range(len(second))); pairs = []
    while available_first:
        _, first_index, second_index = max(
            ((abs(float(similarity[i, j])), i, j) for i in available_first for j in available_second),
            key=lambda row: row[0],
        )
        signed = float(similarity[first_index, second_index])
        pairs.append((abs(signed), first_index, second_index, 1 if signed >= 0 else -1))
        available_first.remove(first_index); available_second.remove(second_index)
    pairs.sort(key=lambda row: row[1])
    return pairs


def weighted_correlation(first, second, weights):
    total = weights.sum().clamp_min(1e-30)
    first_mean = (first * weights).sum() / total; second_mean = (second * weights).sum() / total
    first_centered = first - first_mean; second_centered = second - second_mean
    covariance = (weights * first_centered * second_centered).sum()
    denominator = ((weights * first_centered.square()).sum() * (weights * second_centered.square()).sum()).sqrt().clamp_min(1e-30)
    return float(covariance / denominator)


def heldout_recovery(target, base, source_residual, axes, weights):
    prediction = base + (source_residual @ axes.T) @ axes
    baseline = (weights[:, None] * (target - base).square()).sum()
    error = (weights[:, None] * (target - prediction).square()).sum()
    return float(1 - error / baseline.clamp_min(1e-30))


def token_contexts(token_ids, corpus, tokenizer, limit):
    contexts = []
    for token in token_ids:
        locations = np.argwhere(corpus == token)
        if not len(locations): continue
        row, column = locations[0]
        start = max(0, int(column) - 8); stop = min(corpus.shape[1], int(column) + 9)
        contexts.append({"token": tokenizer.decode([int(token)]), "context": tokenizer.decode(corpus[row, start:stop].tolist()).replace("\n", "\\n")})
        if len(contexts) >= limit: break
    return contexts


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]).astype(np.float32)
    human = token_categories(flags)
    profiles = torch.load(PROFILE_SOURCE, map_location="cpu", weights_only=True)
    matrix_a = profiles["a"]["routing"].float().to("cuda"); matrix_b = profiles["b"]["routing"].float().to("cuda")
    count_a = profiles["a"]["counts"].float().to("cuda"); count_b = profiles["b"]["counts"].float().to("cuda")
    human_tensor = torch.as_tensor(human, device="cuda")
    centroid_a = category_centroids(matrix_a, human_tensor, count_a, len(CATEGORIES))
    centroid_b = category_centroids(matrix_b, human_tensor, count_b, len(CATEGORIES))
    residual_a = matrix_a - centroid_a[human_tensor]; residual_b = matrix_b - centroid_b[human_tensor]
    axes_a = weighted_axes(residual_a, count_a, config["candidate_axes"])
    axes_b = weighted_axes(residual_b, count_b, config["candidate_axes"])
    pairs = greedy_axis_alignment(axes_a, axes_b)
    aligned_b = torch.stack([axes_b[row[2]] * row[3] for row in pairs])
    axis_cosines = [row[0] for row in pairs]
    common_weight = (count_a * count_b).sqrt()
    scores_a = residual_a @ axes_a.T; scores_b = residual_b @ aligned_b.T
    score_correlations = [weighted_correlation(scores_a[:, index], scores_b[:, index], common_weight) for index in range(config["candidate_axes"])]
    feature_correlations = []
    flag_tensor = torch.as_tensor(flags, device="cuda")
    for axis in range(config["candidate_axes"]):
        correlations = [weighted_correlation(scores_a[:, axis], flag_tensor[:, feature], count_a) for feature in range(len(FEATURES))]
        best = int(np.argmax(np.abs(correlations)))
        feature_correlations.append({"feature": FEATURES[best], "correlation": correlations[best], "maximum_absolute": abs(correlations[best])})
    base_prediction = centroid_a[human_tensor]
    recoveries = {str(rank): heldout_recovery(matrix_b, base_prediction, residual_a, axes_a[:rank], count_b) for rank in config["ranks"]}
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random16 = []
    for _ in range(32):
        random = torch.randn(16, matrix_a.shape[1], generator=generator, device="cuda")
        random = torch.linalg.qr(random.T).Q.T
        random16.append(heldout_recovery(matrix_b, base_prediction, residual_a, random, count_b))
    unknown = [index for index in range(config["candidate_axes"]) if axis_cosines[index] >= .70 and score_correlations[index] >= .60 and feature_correlations[index]["maximum_absolute"] < .50]
    corpus = np.load(config["corpus"], mmap_mode="r")
    context_raw = np.asarray(corpus[config["context_corpus_start"]:config["context_corpus_start"] + config["context_corpus_rows"], :], dtype=np.int64)
    total_count = (count_a + count_b).cpu().numpy(); eligible = total_count >= config["minimum_token_count"]
    candidates = []
    for axis in range(config["candidate_axes"]):
        score = scores_a[:, axis].cpu().numpy(); positive = np.flatnonzero(eligible)[np.argsort(score[eligible])[::-1]]; negative = np.flatnonzero(eligible)[np.argsort(score[eligible])]
        top_functions = axes_a[axis].abs().argsort(descending=True)[:12].cpu().numpy()
        candidates.append({
            "axis": axis, "split_cosine": axis_cosines[axis], "token_score_correlation": score_correlations[axis], "closest_known_feature": feature_correlations[axis],
            "top_positive_tokens": [tokenizer.decode([int(index)]).replace("\n", "\\n").replace("\t", "\\t") for index in positive[:50]],
            "top_negative_tokens": [tokenizer.decode([int(index)]).replace("\n", "\\n").replace("\t", "\\t") for index in negative[:50]],
            "top_function_coordinates": [{"coordinate": decode_coordinate(int(index), config), "loading": float(axes_a[axis, index])} for index in top_functions],
            "positive_examples": token_contexts(positive, context_raw, tokenizer, config["examples_per_pole"]),
            "negative_examples": token_contexts(negative, context_raw, tokenizer, config["examples_per_pole"]),
        })
    hypotheses = {
        "U1_human_categories_leave_structure": {"supported": recoveries["16"] >= .20, "rank16_recovery": recoveries["16"]},
        "U2_residual_axes_beat_random": {"supported": recoveries["16"] >= float(np.quantile(random16, .95)) + .10, "selected": recoveries["16"], "random_p95": float(np.quantile(random16, .95))},
        "U3_multiple_residual_axes_are_stable": {"supported": sum(value >= .70 for value in axis_cosines) >= 8, "stable_axis_count": sum(value >= .70 for value in axis_cosines)},
        "U4_unknown_axes_exist": {"supported": len(unknown) >= 3, "candidate_count": len(unknown), "candidate_axes": unknown},
        "U5_residual_recovery_scales_with_rank": {"supported": all(recoveries[str(a)] < recoveries[str(b)] for a, b in zip(config["ranks"], config["ranks"][1:]))},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "recoveries": recoveries, "random16": random16, "axis_cosines": axis_cosines, "score_correlations": score_correlations, "unknown_axes": unknown, "candidates": candidates}, indent=2, allow_nan=False))
    lines = ["# Unknown source-feature axes beyond the human vocabulary", "", "Human category centroids are subtracted from each token's six-layer routing profile. Weighted residual axes are fit on one FineWeb split, aligned to an independent split, and used prospectively to predict the second split. Each candidate saves 50 positive and 50 negative tokens plus up to 50 corpus contexts per pole.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "Residual recovery by rank: " + ", ".join(f"{rank}: {recoveries[str(rank)]:.3f}" for rank in config["ranks"]) + ".", "", "## Strongest candidate unknown axes", ""]
    ranked = sorted(unknown, key=lambda index: axis_cosines[index] * max(score_correlations[index], 0), reverse=True)
    for axis in ranked[:10]:
        row = candidates[axis]; positive = ", ".join(repr(value) for value in row["top_positive_tokens"][:10]); negative = ", ".join(repr(value) for value in row["top_negative_tokens"][:10]); functions = ", ".join(value["coordinate"] for value in row["top_function_coordinates"][:5])
        lines.append(f"- **U{axis}**: split axis cosine {row['split_cosine']:.3f}, token-score correlation {row['token_score_correlation']:.3f}, closest known `{row['closest_known_feature']['feature']}` ({row['closest_known_feature']['correlation']:+.3f}). Positive: {positive}. Negative: {negative}. Functions: {functions}.")
    lines += ["", "## Guardrails", "", "These are token-level routing-role axes, not yet semantic concepts. The source profile uses magnitude and discards sign. Prediction transfers across two FineWeb halves but uses the same token identities, so it tests role stability rather than unseen-token generalization. Existing-feature correlations cover the current 50-label audit only. A low correlation means unknown to that vocabulary, not necessarily novel to a human. Context examples are discovery aids and are not used in the locked quantitative gates."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "recoveries": recoveries, "unknown": unknown}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
