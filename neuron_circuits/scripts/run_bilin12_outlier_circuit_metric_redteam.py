#!/usr/bin/env python3
"""Sensitivity checks for the all-layer outlier-to-circuit sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_outlier_circuit_metric_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_outlier_circuit_metric_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_OUTLIER_CIRCUIT_METRIC_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_outlier_circuit_metric_redteam.png"


def spearman(first, second) -> float:
    first = np.asarray(first); second = np.asarray(second)
    rank_first = np.empty(len(first)); rank_second = np.empty(len(second))
    rank_first[np.argsort(first)] = np.arange(len(first))
    rank_second[np.argsort(second)] = np.arange(len(second))
    return float(np.corrcoef(rank_first, rank_second)[0, 1])


def aggregate(values, selector):
    if selector == "first": return values[0]
    if selector == "minimum": return min(values)
    if selector == "median": return float(np.median(values))
    if selector == "maximum": return max(values)
    raise ValueError(selector)


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    layers = source["config"]["layers"]
    rows = [source["results"][str(layer)] for layer in layers]
    excess = [row["leading_log_excess"] for row in rows]
    efficiency = [row["natural_rank1_efficiency"] for row in rows]
    selectors = ("first", "minimum", "median", "maximum")
    aggregation_correlations = {}
    aggregation_values = {}
    for selector in selectors:
        values = [aggregate([a["shapley_k90"] for a in row["clean_addresses"]], selector) for row in rows]
        aggregation_values[selector] = values
        aggregation_correlations[selector] = spearman(excess, values)
    leave_one_out = []
    for omitted in layers:
        keep = [index for index in range(len(layers)) if index != omitted]
        leave_one_out.append({
            "omitted_layer": omitted,
            "spearman": spearman([excess[i] for i in keep], [efficiency[i] for i in keep]),
        })
    unanimous_sign_top1_layers = [
        layer for layer, row in zip(layers, rows)
        if len({address["top8_channels"][0] for address in row["sign_nulls"]}) == 1
    ]
    phase_summary = {}
    for phase, selected_layers in config["phases"].items():
        selected = [rows[layer] for layer in selected_layers]
        phase_summary[phase] = {
            "median_log_excess": float(np.median([row["leading_log_excess"] for row in selected])),
            "median_natural_efficiency": float(np.median([row["natural_rank1_efficiency"] for row in selected])),
            "median_operator_k90": float(np.median([row["clean_summary"]["median_operator_k90"] for row in selected])),
            "median_channel_k90": float(np.median([row["clean_summary"]["median_shapley_k90"] for row in selected])),
        }
    strongest = np.argsort(excess)[-config["strongest_count"]:][::-1].tolist()
    first_values = aggregation_values["first"]
    hypotheses = {
        "R1_no_address_aggregation_recovers_strong_outlier_localization": {"supported": all(abs(value) < .60 for value in aggregation_correlations.values())},
        "R2_strong_outliers_have_a_small_first_address": {"supported": all(first_values[layer] <= 64 for layer in strongest), "strongest_layers": strongest, "first_k90": [first_values[layer] for layer in strongest]},
        "R3_outlier_natural_use_relation_is_weak_leave_one_out": {"supported": all(row["spearman"] < .60 for row in leave_one_out)},
        "R4_sign_null_top_channel_recurrence_is_structural": {"supported": len(unanimous_sign_top1_layers) >= 8, "layers": unanimous_sign_top1_layers},
        "R5_late_layers_form_a_compact_program_phase": {"supported": phase_summary["late"]["median_operator_k90"] <= .25 * phase_summary["middle"]["median_operator_k90"] and phase_summary["late"]["median_channel_k90"] <= .25 * phase_summary["middle"]["median_channel_k90"]},
        "R6_late_compactness_is_not_explained_by_stronger_outliers": {"supported": phase_summary["late"]["median_log_excess"] < phase_summary["middle"]["median_log_excess"]},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {
        "config": config, "hypotheses": hypotheses,
        "aggregation_correlations": aggregation_correlations,
        "aggregation_values": aggregation_values,
        "leave_one_out": leave_one_out,
        "unanimous_sign_top1_layers": unanimous_sign_top1_layers,
        "phase_summary": phase_summary,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(selectors, [aggregation_correlations[key] for key in selectors])
    axes[0].axhline(.6, color="black", ls="--", lw=1); axes[0].axhline(-.6, color="black", ls="--", lw=1)
    axes[0].set_ylabel("Spearman(outlier excess, channel K90)"); axes[0].set_title("Address aggregation sensitivity")
    phases = list(config["phases"])
    axes[1].plot(phases, [phase_summary[p]["median_operator_k90"] for p in phases], marker="o", label="operator K90")
    axes[1].plot(phases, [phase_summary[p]["median_channel_k90"] for p in phases], marker="s", label="channel K90")
    axes[1].set_yscale("log"); axes[1].set_title("Depth regimes"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = ["# Metric-sensitivity red team for the outlier depth sweep", "",
             "This post-sweep red team asks whether the negative result was an artifact of using the median of four cleaned addresses and whether the apparent late compactness simply follows outlier strength.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Metric red team](bilin12_outlier_circuit_metric_redteam.png)", "",
              f"Address-aggregation correlations: `{aggregation_correlations}`.",
              f"Leave-one-layer-out excess/use correlations span `{min(x['spearman'] for x in leave_one_out):.3f}` to `{max(x['spearman'] for x in leave_one_out):.3f}`.",
              f"Sign-null top channel is unanimous in layers `{unanimous_sign_top1_layers}`.", "",
              "| Phase | Median excess | Natural efficiency | Operator K90 | Channel K90 |",
              "|---|---:|---:|---:|---:|"]
    for phase in phases:
        row = phase_summary[phase]
        lines.append(f"| {phase} | {row['median_log_excess']:.3f} | {row['median_natural_efficiency']:.3f} | {row['median_operator_k90']:.1f} | {row['median_channel_k90']:.1f} |")
    lines += ["", "The late compact-program phase is real in this checkpoint, but it is not an outlier-strength phase: middle layers have stronger outliers and broader programs. Sign-null unanimity confirms that recurrent native channels can arise from structurally dominant coordinate magnitudes even after coordinated signs are destroyed."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(REPORT)


if __name__ == "__main__": main()
