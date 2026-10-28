#!/usr/bin/env python3
"""Include native in the prospective suffix-Fisher chart competition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_bilin12_layer01_dynamic_alignment_replication import median, pair_cluster_interval
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_all_chart_router_analysis.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_all_chart_router_analysis.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_ALL_CHART_ROUTER_ANALYSIS.md"


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_results"]).read_text())
    summaries = {}
    records = {}
    for layer_index, (layer, layer_rows) in enumerate(source["records"].items()):
        records[layer] = []
        for row in layer_rows:
            rotated_score = row["predicted_fidelity"][row["fisher_choice"]]
            native_score = row["predicted_fidelity"]["native"]
            native_choice = native_score >= rotated_score
            effect = row["actual"]["native"] if native_choice else row["actual"]["fisher_router"]
            oracle = max(row["actual"]["native"], row["actual"]["oracle"])
            records[layer].append({**row, "all_native_choice": native_choice, "all_effect": effect, "all_oracle": oracle})
        summaries[layer] = {}
        for corpus_index, corpus in enumerate(sorted({row["corpus"] for row in records[layer]})):
            selected = [row for row in records[layer] if row["corpus"] == corpus]
            advantage, interval = pair_cluster_interval(
                selected, lambda row: row["all_effect"] - row["actual"]["native"],
                config["bootstrap_samples"], config["bootstrap_seed"] + layer_index * 100 + corpus_index,
            )
            summaries[layer][corpus] = {
                "native_choice_fraction": float(np.mean([row["all_native_choice"] for row in selected])),
                "native": median([row["actual"]["native"] for row in selected]),
                "all_router": median([row["all_effect"] for row in selected]),
                "all_oracle": median([row["all_oracle"] for row in selected]),
                "all_minus_native_pair_median": advantage,
                "all_minus_native_pair_ci": interval,
            }
    def cell(layer, corpus): return summaries[str(layer)][corpus]
    corpora = list(summaries["0"])
    g0 = all(cell(0, c)["native_choice_fraction"] >= .5 for c in corpora)
    g1 = all(cell(1, c)["native_choice_fraction"] < .5 for c in corpora)
    g2 = all(cell(l, c)["all_oracle"] - cell(l, c)["all_router"] <= .03 for l in (0, 1) for c in corpora)
    g3 = all(cell(l, c)["all_minus_native_pair_median"] >= -.02 for l in (0, 1) for c in corpora)
    g4 = all(cell(1, c)["all_minus_native_pair_median"] >= .03 and cell(1, c)["all_minus_native_pair_ci"][0] > 0 for c in corpora)
    g5 = all((1 - cell(1, c)["native_choice_fraction"]) - (1 - cell(0, c)["native_choice_fraction"]) >= .2 for c in corpora)
    hypotheses = {
        "G0_layer0_prefers_the_native_chart": {"supported": g0},
        "G1_layer1_prefers_rotated_charts": {"supported": g1},
        "G2_all_chart_router_matches_the_all_chart_oracle": {"supported": g2},
        "G3_adding_chart_options_does_not_harm_typical_behavior": {"supported": g3},
        "G4_layer1_adaptive_charts_improve_on_native": {"supported": g4},
        "G5_depth_changes_chart_preference": {"supported": g5},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Native-plus-rotated prospective chart router", "", "The same suffix-Fisher score now chooses between the native rank-256 midpoint program and seven rotated programs. No new model evaluations or labels are introduced; chart choice was already computed prospectively, and finite effects remain held-out outcomes.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Corpus | Native-choice fraction | Native / all-chart router / all-chart oracle | Router−native pair median (95% CI) |", "|---:|---|---:|---:|---:|"]
    for layer in (0, 1):
        for corpus in corpora:
            row = cell(layer, corpus); ci = row["all_minus_native_pair_ci"]
            lines.append(f"| {layer} | {corpus} | {row['native_choice_fraction']:.3f} | {row['native']:.3f}/{row['all_router']:.3f}/{row['all_oracle']:.3f} | {row['all_minus_native_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) |")
    lines += ["", "The all-chart router is an effect-space compiler, not a claim that rotated coordinates are native causal parts. A native choice favors independently ablatable neuron coordinates; a rotated choice favors a shorter output-conditioned approximation that becomes dense when mapped back to native products."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
