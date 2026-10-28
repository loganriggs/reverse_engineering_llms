#!/usr/bin/env python3
"""Validate the pulled-back final-logit Fisher across finite tail-edit doses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer17_bottleneck_tail_ontology import corpus_rows
from run_bilin18_layer17_output_spectra import prefix_to_layer17
from run_bilin18_layer17_tail_fisher_basis import (
    final_logit_fisher,
    fisher_quadratic,
    prediction_for_selection,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_fisher_finite_redteam.json")
FISHER_OUTPUT = TENSOR_ROOT / "layer17_tail_fisher_basis.json"
FISHER_ARRAYS = TENSOR_ROOT / "layer17_tail_fisher_basis.npz"
TAIL_ARRAYS = TENSOR_ROOT / "layer17_tail_tensor_canonicalization.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_fisher_finite_redteam.json"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_FISHER_FINITE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer17_tail_fisher_finite_redteam.png"


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64) - np.mean(first)
    second = np.asarray(second, dtype=np.float64) - np.mean(second)
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def summarize(records: dict[str, list[float]], dose: float) -> dict[str, float]:
    exact = np.maximum(np.asarray(records["exact_kl"], dtype=np.float64), 0)
    predicted = np.maximum(np.asarray(records["predicted_kl"], dtype=np.float64), 1e-30)
    positive = predicted >= np.quantile(predicted, 0.01)
    return {
        "mean_exact_kl": float(exact.mean()),
        "median_exact_kl": float(np.median(exact)),
        "mean_predicted_kl": float(predicted.mean()),
        "median_predicted_kl": float(np.median(predicted)),
        "prediction_correlation": correlation(exact[positive], predicted[positive]),
        "calibration_ratio": float(np.median(exact[positive] / predicted[positive])),
        "exact_kl_over_dose_squared": float(exact.mean() / (dose * dose)),
    }


@torch.inference_mode()
def evaluate(
    model,
    spec: dict,
    config: dict,
    output_basis: torch.Tensor,
    inverse: torch.Tensor,
    candidate_basis: torch.Tensor,
    forms: torch.Tensor,
    traces: torch.Tensor,
    variants: dict[str, list[int]],
) -> dict[str, dict[str, dict[str, float]]]:
    rows = corpus_rows(spec)
    positions = config["positions"]
    block = model.blocks[config["layer"]]
    dimension = model.config.n_embd
    records = {
        name: {
            str(dose): {"exact_kl": [], "predicted_kl": []}
            for dose in config["dose_fractions"]
        }
        for name in variants
    }
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start : start + config["batch_size"], :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        pre, normalized = prefix_to_layer17(model, ids)
        pre = pre[:, positions].float().flatten(0, 1)
        x = normalized[:, positions].float().flatten(0, 1)
        raw_write = block.mlp.Down(block.mlp.Left(x) * block.mlp.Right(x)).float()
        native_write = raw_write + block.mlp.Down_bias.float()
        residual = pre + native_write
        native_score = raw_write @ output_basis
        z = x @ inverse
        coordinates = z @ candidate_basis
        isotropic = z.square().sum(1, keepdim=True) * traces[None]
        predictions = {
            name: prediction_for_selection(coordinates, forms, axes, isotropic)
            for name, axes in variants.items()
        }
        metric = final_logit_fisher(model, residual, output_basis)
        native_logits = 30 * torch.tanh(
            model.lm_head(F.rms_norm(residual, (dimension,))).float() / 30
        )
        native_logprob = native_logits.log_softmax(-1)
        native_probability = native_logprob.exp()
        for name, prediction in predictions.items():
            error = prediction - native_score
            base_predicted_kl = 0.5 * fisher_quadratic(error, metric)
            for dose in config["dose_fractions"]:
                changed_write = native_write + dose * error @ output_basis.T
                logits = 30 * torch.tanh(
                    model.lm_head(F.rms_norm(pre + changed_write, (dimension,))).float() / 30
                )
                logprob = logits.log_softmax(-1)
                exact_kl = (native_probability * (native_logprob - logprob)).sum(1)
                records[name][str(dose)]["exact_kl"].extend(exact_kl.cpu().tolist())
                records[name][str(dose)]["predicted_kl"].extend(
                    (dose * dose * base_predicted_kl).cpu().tolist()
                )
    return {
        name: {
            str(dose): summarize(dose_rows, dose)
            for dose, dose_rows in ((value, rows_by_dose[str(value)]) for value in config["dose_fractions"])
        }
        for name, rows_by_dose in records.items()
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    fisher_result = json.loads(FISHER_OUTPUT.read_text())
    fisher_arrays = np.load(FISHER_ARRAYS)
    tail_arrays = np.load(TAIL_ARRAYS)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    output_basis = torch.from_numpy(tail_arrays["tail_output_basis"]).cuda().float()
    inverse = torch.from_numpy(tail_arrays["inverse_square"]).cuda().float()
    candidate_basis = torch.from_numpy(fisher_arrays["candidate_basis"]).cuda().float()
    forms = torch.from_numpy(fisher_arrays["restricted_forms"]).cuda().float()
    traces = torch.from_numpy(fisher_arrays["traces"]).cuda().float()
    selections = fisher_result["selections"]
    variants = {
        "fisher_rank4": selections["fisher"][:4],
        "frobenius_rank4": selections["frobenius"][:4],
        "fisher_rank16": selections["fisher"][:16],
    }
    assert set(variants) == set(config["variants"])

    results = {}
    for corpus, spec in config["evaluation"].items():
        print("finite Fisher", corpus, flush=True)
        results[corpus] = evaluate(
            model, spec, config, output_basis, inverse, candidate_basis,
            forms, traces, variants,
        )

    local = "0.1"
    full = "1.0"
    hypotheses = {
        "L1_fisher_predicts_local_kl_direction": {
            "supported": all(
                row[variant][local]["prediction_correlation"] >= 0.90
                for row in results.values() for variant in variants
            ),
        },
        "L2_fisher_locally_calibrated": {
            "supported": all(
                0.5 <= row[variant][local]["calibration_ratio"] <= 2.0
                for row in results.values() for variant in variants
            ),
        },
        "L3_fisher_predicts_full_replacement": {
            "supported": all(
                row[variant][full]["prediction_correlation"] >= 0.80
                for row in results.values() for variant in variants
            ),
        },
        "L4_fisher_rank4_beats_frobenius_locally": {
            "supported": all(
                row["fisher_rank4"][local]["mean_exact_kl"]
                < row["frobenius_rank4"][local]["mean_exact_kl"]
                for row in results.values()
            ),
        },
        "L5_local_kl_is_quadratic": {
            "supported": all(
                0.5 <= row[variant]["0.1"]["exact_kl_over_dose_squared"]
                / max(row[variant]["0.25"]["exact_kl_over_dose_squared"], 1e-30) <= 2.0
                for row in results.values() for variant in variants
            ),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "variants": variants,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Finite-dose red-team of the Layer-17 tail Fisher metric",
        "",
        "The native-to-compact tail-score error is applied at 10%, 25%, 50%, and 100% dose. The local prediction is `0.5 * dose² * eᵀGe`, where `G` is the exact native softmax Fisher pulled back through final RMSNorm, unembedding, and logit tanh. Exact KL recomputes the full final distribution.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Finite Fisher red-team](layer17_tail_fisher_finite_redteam.png)",
        "",
        "| Corpus | Variant | 10% corr. / calibration | 100% corr. / calibration | 100% median KL |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        for variant in variants:
            local_row, full_row = row[variant][local], row[variant][full]
            lines.append(
                f"| {corpus} | {variant} | {local_row['prediction_correlation']:.3f} / {local_row['calibration_ratio']:.2f} | "
                f"{full_row['prediction_correlation']:.3f} / {full_row['calibration_ratio']:.2f} | "
                f"{full_row['median_exact_kl']:.5f} |"
            )
    lines += [
        "",
        "If direction is predicted locally but calibration or ordering fails at full dose, Fisher remains a tangent explanation rather than a replacement objective. If Fisher-selected rank four loses even at 10% dose on held-out corpora, the discovery gain is sampling/metric overfit rather than merely finite-step curvature.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, plots = plt.subplots(1, 3, figsize=(14, 4.2))
    doses = config["dose_fractions"]
    for corpus, row in results.items():
        plots[0].plot(doses, [row["fisher_rank16"][str(d)]["prediction_correlation"] for d in doses], marker="o", label=corpus)
        plots[1].plot(doses, [row["fisher_rank16"][str(d)]["calibration_ratio"] for d in doses], marker="o", label=corpus)
    plots[0].set_ylim(0, 1.02)
    plots[0].set_xlabel("edit dose")
    plots[0].set_ylabel("predicted vs exact KL corr.")
    plots[0].set_title("Fisher direction")
    plots[0].legend(fontsize=8)
    plots[1].axhline(1, color="black", linestyle="--", linewidth=1)
    plots[1].set_xlabel("edit dose")
    plots[1].set_ylabel("median exact / predicted KL")
    plots[1].set_title("Fisher calibration")
    plots[1].legend(fontsize=8)
    x = np.arange(len(results))
    width = 0.35
    plots[2].bar(x - width / 2, [results[c]["fisher_rank4"][local]["mean_exact_kl"] for c in results], width, label="Fisher rank-4")
    plots[2].bar(x + width / 2, [results[c]["frobenius_rank4"][local]["mean_exact_kl"] for c in results], width, label="Frobenius rank-4")
    plots[2].set_xticks(x, list(results), rotation=15)
    plots[2].set_ylabel("10% mean exact KL")
    plots[2].set_title("Held-out local selection")
    plots[2].legend(fontsize=8)
    for plot in plots:
        plot.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
