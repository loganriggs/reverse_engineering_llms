#!/usr/bin/env python3
"""Full-width power-law and coordination-null spectra across tensor depth."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_spectral_redteam import symmetric_channel_gram
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_depth_powerlaw_nulls.json")
OUTPUT = TENSOR_ROOT / "depth_powerlaw_nulls.json"
ARRAYS = TENSOR_ROOT / "depth_powerlaw_nulls.npz"
REPORT = TENSOR_ROOT / "DEPTH_POWERLAW_NULLS.md"
FIGURE = TENSOR_ROOT / "depth_powerlaw_nulls.png"


def spectrum_from_gram(down: torch.Tensor, channel_gram: torch.Tensor) -> torch.Tensor:
    gram = down @ (channel_gram @ down.T)
    gram = 0.5 * (gram + gram.T)
    return torch.linalg.eigvalsh(gram).flip(0).clamp_min(0).sqrt()


def fit_powerlaw(values: np.ndarray, start: int, stop: int, outliers: int) -> dict:
    values = np.sort(np.abs(np.asarray(values, dtype=np.float64)))[::-1]
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    keep = slice(start - 1, min(stop, len(values)))
    x = np.log(ranks[keep])
    y = np.log(values[keep].clip(1e-30))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    r2 = 1 - np.square(y - predicted).sum() / max(np.square(y - y.mean()).sum(), 1e-30)
    top_ranks = ranks[:outliers]
    top_excess = np.log(values[:outliers].clip(1e-30)) - (intercept + slope * np.log(top_ranks))
    energy = np.square(values)
    return {
        "alpha": float(-slope),
        "r_squared": float(r2),
        "intercept": float(intercept),
        "top_log_excess": [float(value) for value in top_excess],
        "mean_top_log_excess": float(top_excess.mean()),
        "top4_energy_fraction": float(energy[:outliers].sum() / energy.sum()),
        "effective_rank": float(energy.sum() ** 2 / np.square(energy).sum()),
    }


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    results = {}
    arrays = {}
    for layer in config["layers"]:
        print("layer", layer, flush=True)
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float()
        left = block.mlp.Left.weight.float()
        right = block.mlp.Right.weight.float()
        channel_gram = symmetric_channel_gram(left, right)
        observed_tensor = spectrum_from_gram(down, channel_gram)
        observed = observed_tensor.cpu().numpy()
        arrays[f"layer{layer}_observed"] = observed
        sign_rows = []
        pairing_rows = []
        for replicate in range(config["null_replicates"]):
            signs = torch.randint(0, 2, (down.shape[1],), generator=generator, device="cuda") * 2 - 1
            sign_spectrum = spectrum_from_gram(down * signs, channel_gram).cpu().numpy()
            permutation = torch.randperm(down.shape[1], generator=generator, device="cuda")
            pairing_spectrum = spectrum_from_gram(down[:, permutation], channel_gram).cpu().numpy()
            arrays[f"layer{layer}_sign{replicate}"] = sign_spectrum
            arrays[f"layer{layer}_pairing{replicate}"] = pairing_spectrum
            sign_rows.append(fit_powerlaw(sign_spectrum, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"]))
            pairing_rows.append(fit_powerlaw(pairing_spectrum, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"]))
        observed_fit = fit_powerlaw(observed, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"])
        results[str(layer)] = {
            "observed": observed_fit,
            "sign_nulls": sign_rows,
            "pairing_nulls": pairing_rows,
            "sign_summary": {
                key: float(np.median([row[key] for row in sign_rows]))
                for key in ("alpha", "r_squared", "mean_top_log_excess", "top4_energy_fraction")
            },
            "pairing_summary": {
                key: float(np.median([row[key] for row in pairing_rows]))
                for key in ("alpha", "r_squared", "mean_top_log_excess", "top4_energy_fraction")
            },
        }
    np.savez_compressed(ARRAYS, **arrays)

    hypotheses = {
        "T1_powerlaw_bulk_all_layers": {
            "supported": all(results[layer]["observed"]["r_squared"] >= 0.90 for layer in results)
        },
        "T2_nulls_retain_bulk": {
            "supported": all(
                abs(results[layer][family]["alpha"] - results[layer]["observed"]["alpha"]) <= 0.15
                and results[layer][family]["r_squared"] >= 0.90
                for layer in results for family in ("sign_summary", "pairing_summary")
            )
        },
        "T3_observed_exceeds_null_median": {
            "supported": all(
                results[layer]["observed"]["mean_top_log_excess"]
                > max(results[layer]["sign_summary"]["mean_top_log_excess"],
                      results[layer]["pairing_summary"]["mean_top_log_excess"])
                for layer in results
            )
        },
        "T4_strict_positive_outliers": {
            "supported": all(results[layer]["observed"]["top_log_excess"][0] > 0 for layer in results)
        },
        "T5_late_full_width_concentration": {
            "supported": results["17"]["observed"]["top4_energy_fraction"]
            - max(results["0"]["observed"]["top4_energy_fraction"],
                  results["8"]["observed"]["top4_energy_fraction"]) >= 0.005
        },
        "T6_late_exceeds_all_nulls": {
            "supported": results["17"]["observed"]["mean_top_log_excess"]
            > max(
                row["mean_top_log_excess"]
                for family in ("sign_nulls", "pairing_nulls")
                for row in results["17"][family]
            )
        },
    }
    OUTPUT.write_text(json.dumps({"config": config, "results": results, "hypotheses": hypotheses}, indent=2))

    lines = [
        "# Full-width power-law spectra and coordination nulls across depth",
        "",
        "Each complete bilinear MLP tensor is treated as a map from symmetric input interactions to residual writes. Its output-side singular spectrum is computed exactly through the channel Gram matrix. Sign nulls preserve each channel form and Down column magnitude while changing cross-channel coordination; pairing nulls reassign Down columns to input channel forms.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Depth power-law nulls](depth_powerlaw_nulls.png)",
        "",
        "| Layer | Observed alpha / R² | Top-1 log excess | Mean top-4 excess | Top-4 energy | Sign-null excess | Pair-null excess |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, row in results.items():
        observed = row["observed"]
        lines.append(
            f"| {layer} | {observed['alpha']:.3f} / {observed['r_squared']:.3f} | "
            f"{observed['top_log_excess'][0]:.3f} | {observed['mean_top_log_excess']:.3f} | "
            f"{observed['top4_energy_fraction']:.3f} | {row['sign_summary']['mean_top_log_excess']:.3f} | "
            f"{row['pairing_summary']['mean_top_log_excess']:.3f} |"
        )
    lines += [
        "",
        "A positive log excess is the strict above-extrapolated-power-law criterion. Exceeding a coordination null instead shows learned cross-channel structure relative to that null; neither criterion by itself establishes semantics or causal sufficiency.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ranks = np.arange(1, len(next(iter(arrays.values()))) + 1)
    for axis, layer in zip(axes, ("0", "8", "17")):
        observed = arrays[f"layer{layer}_observed"]
        axis.loglog(ranks, observed, label="observed", linewidth=2)
        sign_stack = np.stack([arrays[f"layer{layer}_sign{i}"] for i in range(config["null_replicates"])])
        pair_stack = np.stack([arrays[f"layer{layer}_pairing{i}"] for i in range(config["null_replicates"])])
        axis.loglog(ranks, np.median(sign_stack, axis=0), label="sign null", alpha=0.8)
        axis.loglog(ranks, np.median(pair_stack, axis=0), label="pair null", alpha=0.8)
        axis.axvspan(config["bulk_rank_start"], config["bulk_rank_stop"], color="grey", alpha=0.1)
        axis.set_title(f"Layer {layer}")
        axis.set_xlabel("spectral rank")
        axis.set_ylabel("singular value")
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
