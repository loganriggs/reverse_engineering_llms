#!/usr/bin/env python3
"""Test the power-law-plus-spikes picture on the Layer-3 full bilinear tensor."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_powerlaw_nulls import fit_powerlaw, spectrum_from_gram
from run_bilin18_depth_spectral_redteam import symmetric_channel_gram
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_powerlaw_spikes.json")
OUTPUT = TENSOR_ROOT / "layer3_powerlaw_spikes.json"
ARRAYS = TENSOR_ROOT / "layer3_powerlaw_spikes.npz"
REPORT = TENSOR_ROOT / "LAYER3_POWERLAW_SPIKES.md"
FIGURE = TENSOR_ROOT / "layer3_powerlaw_spikes.png"


def median_summary(rows):
    return {
        key: float(np.median([row[key] for row in rows]))
        for key in ("alpha", "r_squared", "mean_top_log_excess", "top4_energy_fraction")
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float()
    left = block.mlp.Left.weight.float()
    right = block.mlp.Right.weight.float()
    channel_gram = symmetric_channel_gram(left, right)
    observed = spectrum_from_gram(down, channel_gram).cpu().numpy()
    observed_fit = fit_powerlaw(
        observed, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"],
    )
    arrays = {"observed": observed}
    sign_nulls, pairing_nulls = [], []
    for replicate in range(config["null_replicates"]):
        signs = torch.randint(0, 2, (down.shape[1],), generator=generator, device="cuda") * 2 - 1
        sign_spectrum = spectrum_from_gram(down * signs, channel_gram).cpu().numpy()
        permutation = torch.randperm(down.shape[1], generator=generator, device="cuda")
        pairing_spectrum = spectrum_from_gram(down[:, permutation], channel_gram).cpu().numpy()
        arrays[f"sign{replicate}"] = sign_spectrum
        arrays[f"pairing{replicate}"] = pairing_spectrum
        sign_nulls.append(fit_powerlaw(
            sign_spectrum, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"],
        ))
        pairing_nulls.append(fit_powerlaw(
            pairing_spectrum, config["bulk_rank_start"], config["bulk_rank_stop"], config["outlier_count"],
        ))
        print("null", replicate + 1, "/", config["null_replicates"], flush=True)
    np.savez_compressed(ARRAYS, **arrays)
    sign_summary, pairing_summary = median_summary(sign_nulls), median_summary(pairing_nulls)
    all_nulls = sign_nulls + pairing_nulls
    hypotheses = {
        "S1_layer3_has_powerlaw_like_bulk": {
            "supported": observed_fit["r_squared"] >= 0.90,
        },
        "S2_coordination_nulls_retain_bulk": {
            "supported": all(
                row["r_squared"] >= 0.90
                and abs(row["alpha"] - observed_fit["alpha"]) <= 0.15
                for row in (sign_summary, pairing_summary)
            ),
        },
        "S3_layer3_has_leading_strict_spike": {
            "supported": observed_fit["top_log_excess"][0] > 0,
        },
        "S4_layer3_has_four_strict_spikes": {
            "supported": all(value > 0 for value in observed_fit["top_log_excess"]),
        },
        "S5_spikes_exceed_all_coordination_nulls": {
            "supported": observed_fit["mean_top_log_excess"]
            > max(row["mean_top_log_excess"] for row in all_nulls),
            "null_maximum": max(row["mean_top_log_excess"] for row in all_nulls),
        },
        "S6_top4_are_globally_concentrated": {
            "supported": observed_fit["top4_energy_fraction"] >= 0.03,
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "observed": observed_fit,
        "sign_summary": sign_summary,
        "pairing_summary": pairing_summary,
        "sign_nulls": sign_nulls,
        "pairing_nulls": pairing_nulls,
    }, indent=2))

    lines = [
        "# Layer-3 full-tensor power law and spike audit",
        "",
        "The complete Layer-3 bilinear MLP tensor is treated as a symmetric-interaction-to-residual-write operator. Its output-side singular spectrum is computed exactly. Sign nulls preserve channel forms and Down magnitudes but disrupt signed coordination; pairing nulls reassign Down columns to interaction forms.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-3 power-law spikes](layer3_powerlaw_spikes.png)",
        "",
        "| Spectrum | Alpha / R² | Top-1 excess | Mean top-4 excess | Top-4 energy |",
        "|---|---:|---:|---:|---:|",
        f"| observed | {observed_fit['alpha']:.3f} / {observed_fit['r_squared']:.3f} | {observed_fit['top_log_excess'][0]:.3f} | {observed_fit['mean_top_log_excess']:.3f} | {observed_fit['top4_energy_fraction']:.3f} |",
        f"| sign-null median | {sign_summary['alpha']:.3f} / {sign_summary['r_squared']:.3f} | — | {sign_summary['mean_top_log_excess']:.3f} | {sign_summary['top4_energy_fraction']:.3f} |",
        f"| pairing-null median | {pairing_summary['alpha']:.3f} / {pairing_summary['r_squared']:.3f} | — | {pairing_summary['mean_top_log_excess']:.3f} | {pairing_summary['top4_energy_fraction']:.3f} |",
        "",
        "Individual observed top-four log excesses: "
        + ", ".join(f"{value:.3f}" for value in observed_fit["top_log_excess"]) + ".",
        "",
        "A positive extrapolation excess is a strict spike under this frozen bulk fit, not circuit validation. If the native family-address result coexists with no strict spike, reusable computations are embedded in the tensor bulk. If spikes exist, they become prospectively selected candidates for semantic and causal validation.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    ranks = np.arange(1, len(observed) + 1)
    sign_stack = np.stack([arrays[f"sign{i}"] for i in range(config["null_replicates"])])
    pairing_stack = np.stack([arrays[f"pairing{i}"] for i in range(config["null_replicates"])])
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.loglog(ranks, observed, label="observed", linewidth=2)
    axis.loglog(ranks, np.median(sign_stack, axis=0), label="sign null")
    axis.loglog(ranks, np.median(pairing_stack, axis=0), label="pairing null")
    axis.axvspan(config["bulk_rank_start"], config["bulk_rank_stop"], color="grey", alpha=.12)
    axis.set_xlabel("spectral rank")
    axis.set_ylabel("singular value")
    axis.set_title("Layer 3 full bilinear tensor")
    axis.legend()
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
