#!/usr/bin/env python3
"""Test whether held-out semantic suffix gradients lie in a reusable subspace."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_gradient_subspace.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_gradient_subspace.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_GRADIENT_SUBSPACE.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_gradient_subspace.png"


def row_cosine(left, right):
    return np.sum(left * right, axis=1) / np.maximum(np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1), 1e-30)


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["source_arrays"])
    discovery = arrays["discovery_gradients"].astype(np.float32); discovery_features = arrays["discovery_features"]
    evaluation = arrays["evaluation_gradients"].astype(np.float32); deltas = arrays["evaluation_deltas"].astype(np.float32); evaluation_features = arrays["evaluation_features"]
    _, singular, vh = np.linalg.svd(discovery, full_matrices=False)
    feature_means = {feature: discovery[discovery_features == feature].mean(0) for feature in np.unique(discovery_features)}
    mean_prediction = np.stack([feature_means[feature] for feature in evaluation_features])
    mean_cosine = row_cosine(evaluation, mean_prediction)
    results = {}; saved_rank32 = None
    exact_contribution = evaluation * deltas
    exact_support = np.argpartition(exact_contribution, -config["support_rank"], axis=1)[:, -config["support_rank"]:]
    exact_shift = np.take_along_axis(exact_contribution, exact_support, axis=1).sum(1)
    rng = np.random.default_rng(config["seed"])
    random_basis, _ = np.linalg.qr(rng.standard_normal((evaluation.shape[1], config["ranks"][-1])).astype(np.float32))
    random_prediction = evaluation @ random_basis @ random_basis.T
    random_cosine = row_cosine(evaluation, random_prediction)
    for rank in config["ranks"]:
        basis = vh[:rank].T
        prediction = evaluation @ basis @ basis.T
        contribution = prediction * deltas
        support = np.argpartition(contribution, -config["support_rank"], axis=1)[:, -config["support_rank"]:]
        selected_true = np.take_along_axis(exact_contribution, support, axis=1).sum(1)
        jaccard = []
        for left, right in zip(exact_support, support):
            left_set, right_set = set(left.tolist()), set(right.tolist()); jaccard.append(len(left_set & right_set) / len(left_set | right_set))
        results[str(rank)] = {"gradient_cosine_median": float(np.median(row_cosine(evaluation, prediction))), "gradient_cosine_q10": float(np.quantile(row_cosine(evaluation, prediction), .1)), "contribution_cosine_median": float(np.median(row_cosine(exact_contribution, contribution))), "support_jaccard_median": float(np.median(jaccard)), "true_first_order_leverage_ratio_median": float(np.median(selected_true / np.maximum(exact_shift, 1e-30)))}
        if rank == 32: saved_rank32 = prediction
    summary = {"discovery_singular_values": singular.tolist(), "feature_mean_gradient_cosine_median": float(np.median(mean_cosine)), "feature_mean_gradient_cosine_q10": float(np.quantile(mean_cosine, .1)), "random_rank32_gradient_cosine_median": float(np.median(random_cosine)), "results": results}
    rank16, rank32 = results["16"], results["32"]
    hypotheses = {
        "G0_rank16_gradient_transfer": {"supported": rank16["gradient_cosine_median"] >= .70},
        "G1_rank32_gradient_transfer": {"supported": rank32["gradient_cosine_median"] >= .85},
        "G2_rank32_contribution_transfer": {"supported": rank32["contribution_cosine_median"] >= .70},
        "G3_rank32_support_transfer": {"supported": rank32["support_jaccard_median"] >= .20},
        "G4_rank32_preserves_first_order_leverage": {"supported": rank32["true_first_order_leverage_ratio_median"] >= .70},
        "G5_subspace_beats_feature_mean": {"supported": rank32["gradient_cosine_median"] - summary["feature_mean_gradient_cosine_median"] >= .10},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary}, indent=2, allow_nan=False))
    lines = ["# Reusable subspace for Layer-1 semantic suffix gradients", "", "Thirty-two exact gradients from disjoint discovery contexts (four per semantic feature) define an uncentered SVD basis. Held-out gradients, their elementwise products with held-out activation deltas, and the resulting top-64 positive supports are evaluated without refitting.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Suffix-gradient subspace](bilin18_layer1_suffix_gradient_subspace.png)", "", f"Feature-mean gradient cosine: **{summary['feature_mean_gradient_cosine_median']:.3f}**; random rank-32 projection: **{summary['random_rank32_gradient_cosine_median']:.3f}**.", "", "| Rank | Gradient cosine | Q10 | Contribution cosine | Top-64 Jaccard | First-order leverage ratio |", "|---:|---:|---:|---:|---:|---:|"]
    for rank in config["ranks"]:
        row = results[str(rank)]; lines.append(f"| {rank} | {row['gradient_cosine_median']:.3f} | {row['gradient_cosine_q10']:.3f} | {row['contribution_cosine_median']:.3f} | {row['support_jaccard_median']:.3f} | {row['true_first_order_leverage_ratio_median']:.3f} |")
    lines += ["", "## Guardrail", "", "Projection coefficients are computed from the held-out exact gradient. This tests whether a reusable basis exists, not whether a cheap router can infer its coefficients from the forward state. Finite intervention must also be rerun before support transfer counts as causal compression."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    ranks = config["ranks"]; axes[0].plot(ranks, [results[str(rank)]["gradient_cosine_median"] for rank in ranks], marker="o", label="gradient"); axes[0].plot(ranks, [results[str(rank)]["contribution_cosine_median"] for rank in ranks], marker="o", label="delta × gradient"); axes[0].axhline(summary["feature_mean_gradient_cosine_median"], color="black", linestyle="--", label="feature mean"); axes[0].set(xlabel="Discovery gradient rank", ylabel="Median cosine", title="Held-out sensitivity geometry"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(ranks, [results[str(rank)]["support_jaccard_median"] for rank in ranks], marker="o", label="support Jaccard"); axes[1].plot(ranks, [results[str(rank)]["true_first_order_leverage_ratio_median"] for rank in ranks], marker="o", label="leverage ratio"); axes[1].set(xlabel="Discovery gradient rank", ylabel="Median", title="Does the basis preserve the dynamic coalition?"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": {key: value for key, value in summary.items() if key != "discovery_singular_values"}}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
