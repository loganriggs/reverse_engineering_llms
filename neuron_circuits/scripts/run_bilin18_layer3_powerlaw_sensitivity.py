#!/usr/bin/env python3
"""Red-team Layer-3 power-law/spike claims for fit and null sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_depth_powerlaw_nulls import fit_powerlaw
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_powerlaw_sensitivity.json")
OUTPUT = TENSOR_ROOT / "layer3_powerlaw_sensitivity.json"
REPORT = TENSOR_ROOT / "LAYER3_POWERLAW_SENSITIVITY.md"
FIGURE = TENSOR_ROOT / "layer3_powerlaw_sensitivity.png"


def heldout_model_comparison(values, start, stop):
    values = np.sort(np.abs(np.asarray(values, dtype=np.float64)))[::-1]
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    indices = np.arange(start - 1, min(stop, len(values)))
    train, heldout = indices[::2], indices[1::2]
    target = np.log(values.clip(1e-30))
    power = np.polyfit(np.log(ranks[train]), target[train], 1)
    exponential = np.polyfit(ranks[train], target[train], 1)
    power_error = float(np.sqrt(np.mean(np.square(
        target[heldout] - np.polyval(power, np.log(ranks[heldout]))
    ))))
    exponential_error = float(np.sqrt(np.mean(np.square(
        target[heldout] - np.polyval(exponential, ranks[heldout])
    ))))
    return {
        "powerlaw_heldout_log_rmse": power_error,
        "exponential_heldout_log_rmse": exponential_error,
        "powerlaw_relative_improvement": (exponential_error - power_error)
        / max(exponential_error, 1e-30),
    }


def main():
    config = json.loads(CONFIG.read_text())
    arrays = np.load(config["source_arrays"])
    observed = arrays["observed"]
    sign_names = sorted((key for key in arrays.files if key.startswith("sign")), key=lambda x: int(x[4:]))
    pairing_names = sorted((key for key in arrays.files if key.startswith("pairing")), key=lambda x: int(x[7:]))
    null_names = sign_names + pairing_names
    nulls = [arrays[name] for name in null_names]
    windows = []
    for start in config["bulk_starts"]:
        for stop in config["bulk_stops"]:
            if start >= stop:
                continue
            observed_fit = fit_powerlaw(observed, start, stop, config["outlier_count"])
            null_fits = [fit_powerlaw(values, start, stop, config["outlier_count"]) for values in nulls]
            windows.append({
                "start": start,
                "stop": stop,
                "observed_top1_excess": observed_fit["top_log_excess"][0],
                "observed_mean_top4_excess": observed_fit["mean_top_log_excess"],
                "null_max_mean_top4_excess": max(row["mean_top_log_excess"] for row in null_fits),
                "beats_all_nulls": observed_fit["mean_top_log_excess"]
                > max(row["mean_top_log_excess"] for row in null_fits),
                "strict_top1": observed_fit["top_log_excess"][0] > 0,
                "r_squared": observed_fit["r_squared"],
            })
    no_spike_fraction = float(np.mean([not row["strict_top1"] for row in windows]))
    null_win_fraction = float(np.mean([row["beats_all_nulls"] for row in windows]))
    null_stack = np.stack(nulls)
    empirical_null = {
        "observed_top4": observed[:4].tolist(),
        "null_max_top4": null_stack[:, :4].max(0).tolist(),
        "ratios_to_null_max": (observed[:4] / null_stack[:, :4].max(0)).tolist(),
        "observed_top4_energy": float(np.square(observed[:4]).sum() / np.square(observed).sum()),
        "null_max_top4_energy": float(max(
            np.square(values[:4]).sum() / np.square(values).sum() for values in nulls
        )),
    }
    comparison = heldout_model_comparison(
        observed, config["model_comparison_start"], config["model_comparison_stop"],
    )
    hypotheses = {
        "R1_no_strict_spike_is_window_robust": {
            "supported": no_spike_fraction >= 0.80,
            "fraction": no_spike_fraction,
        },
        "R2_relative_null_advantage_is_window_robust": {
            "supported": null_win_fraction >= 0.75,
            "fraction": null_win_fraction,
        },
        "R3_top_singular_value_is_empirical_null_outlier": {
            "supported": empirical_null["ratios_to_null_max"][0] > 1,
            "ratio": empirical_null["ratios_to_null_max"][0],
        },
        "R4_all_top4_are_empirical_null_outliers": {
            "supported": all(value > 1 for value in empirical_null["ratios_to_null_max"]),
            "ratios": empirical_null["ratios_to_null_max"],
        },
        "R5_powerlaw_predicts_tail_better_than_exponential": {
            "supported": comparison["powerlaw_relative_improvement"] >= 0.10,
            **comparison,
        },
        "R6_top4_energy_exceeds_every_null": {
            "supported": empirical_null["observed_top4_energy"] > empirical_null["null_max_top4_energy"],
            "observed": empirical_null["observed_top4_energy"],
            "null_maximum": empirical_null["null_max_top4_energy"],
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "windows": windows,
        "empirical_null": empirical_null,
        "model_comparison": comparison,
    }, indent=2))

    lines = [
        "# Layer-3 power-law and spike sensitivity red team",
        "",
        "The frozen spectrum is re-fit over fifteen bulk windows. Spike strength is also compared directly with all 32 coordination-null spectra, avoiding bulk extrapolation. Finally, power-law and exponential-rank models are fit on alternating ranks and scored on held-out ranks 33--1024.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-3 spectrum sensitivity](layer3_powerlaw_sensitivity.png)",
        "",
        f"No-strict-spike fraction across windows: **{no_spike_fraction:.1%}**.",
        f"Observed relative-bulk top four beat every null in **{null_win_fraction:.1%}** of windows.",
        "",
        "| Rank | Observed / maximum null singular value |",
        "|---:|---:|",
    ]
    for rank, ratio in enumerate(empirical_null["ratios_to_null_max"], 1):
        lines.append(f"| {rank} | {ratio:.4f} |")
    lines += [
        "",
        f"Observed/null-maximum top-four energy: **{empirical_null['observed_top4_energy']:.4f} / {empirical_null['null_max_top4_energy']:.4f}**.",
        f"Held-out log-spectrum RMSE is **{comparison['powerlaw_heldout_log_rmse']:.4f}** for a power law versus **{comparison['exponential_heldout_log_rmse']:.4f}** for an exponential-rank curve ({comparison['powerlaw_relative_improvement']:.1%} relative improvement).",
        "",
        "A robust absence of positive extrapolation spikes together with direct null exceedance would mean the observed top spectrum is coordinated but remains part of a continuous heavy tail. Failure of both tests would make the apparent advantage at the original fit window a fitting artifact.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for stop in config["bulk_stops"]:
        chosen = [row for row in windows if row["stop"] == stop]
        axes[0].plot(
            [row["start"] for row in chosen], [row["observed_top1_excess"] for row in chosen],
            marker="o", label=f"stop {stop}",
        )
    axes[0].axhline(0, color="black", linewidth=.8)
    axes[0].set_xlabel("bulk fit start rank")
    axes[0].set_ylabel("top-1 log excess")
    axes[0].set_title("Strict-spike sensitivity")
    axes[0].legend()
    axes[1].bar(np.arange(1, 5), empirical_null["ratios_to_null_max"])
    axes[1].axhline(1, color="black", linewidth=.8)
    axes[1].set_xticks(np.arange(1, 5))
    axes[1].set_xlabel("spectral rank")
    axes[1].set_ylabel("observed / maximum null")
    axes[1].set_title("Direct empirical-null test")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
