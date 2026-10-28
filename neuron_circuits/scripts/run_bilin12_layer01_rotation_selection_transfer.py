#!/usr/bin/env python3
"""Fair transfer test after the per-example best-of-seven rotation red-team."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_bilin12_layer01_dynamic_alignment_replication import median
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_rotation_selection_transfer.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_rotation_selection_transfer.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_ROTATION_SELECTION_TRANSFER.md"


def clustered_interval(pair_values, samples, seed):
    values = np.asarray(list(pair_values.values()), dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.asarray([
        np.median(generator.choice(values, len(values), replace=True)) for _ in range(samples)
    ])
    return float(np.median(values)), [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def pair_means(records, getter):
    return {
        pair: median([getter(row) for row in records if row["pair"] == pair])
        for pair in sorted({row["pair"] for row in records})
    }


def best_seed(records, seeds, excluded_pair=None):
    training = [row for row in records if row["pair"] != excluded_pair]
    return max(
        seeds,
        key=lambda seed: median([row["rotations"][str(seed)] for row in training]),
    )


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_results"]).read_text())
    seeds = source["config"]["rotation_seeds"]
    layers = sorted(int(layer) for layer in source["records"])
    corpora = sorted({row["corpus"] for rows in source["records"].values() for row in rows})
    summaries = {str(layer): {} for layer in layers}
    global_choices = {str(layer): {} for layer in layers}

    for layer in layers:
        for corpus_index, corpus in enumerate(corpora):
            records = [row for row in source["records"][str(layer)] if row["corpus"] == corpus]
            fixed_medians = {
                str(seed): median([row["rotations"][str(seed)] for row in records]) for seed in seeds
            }
            global_choice = max(seeds, key=lambda seed: fixed_medians[str(seed)])
            global_choices[str(layer)][corpus] = global_choice
            pairs = sorted({row["pair"] for row in records})
            lopo_choice = {pair: best_seed(records, seeds, excluded_pair=pair) for pair in pairs}
            lopo_advantage = {
                pair: median([
                    row["native"] - row["rotations"][str(lopo_choice[pair])]
                    for row in records if row["pair"] == pair
                ]) for pair in pairs
            }
            oracle_lopo_gap = {
                pair: median([
                    row["best_rotation"] - row["rotations"][str(lopo_choice[pair])]
                    for row in records if row["pair"] == pair
                ]) for pair in pairs
            }
            advantage_center, advantage_ci = clustered_interval(
                lopo_advantage, config["bootstrap_samples"], config["bootstrap_seed"] + layer * 100 + corpus_index,
            )
            oracle_center, oracle_ci = clustered_interval(
                oracle_lopo_gap, config["bootstrap_samples"], config["bootstrap_seed"] + 1000 + layer * 100 + corpus_index,
            )
            summaries[str(layer)][corpus] = {
                "native_median": median([row["native"] for row in records]),
                "fixed_rotation_medians": fixed_medians,
                "global_best_seed": global_choice,
                "global_best_median": fixed_medians[str(global_choice)],
                "lopo_choices": lopo_choice,
                "native_minus_lopo_pair_median": advantage_center,
                "native_minus_lopo_pair_ci": advantage_ci,
                "oracle_minus_lopo_pair_median": oracle_center,
                "oracle_minus_lopo_pair_ci": oracle_ci,
            }

    # Select a single fixed basis on FineWeb, then test the same seed on TinyStories.
    ood = {}
    for layer in layers:
        selected_seed = global_choices[str(layer)]["fineweb"]
        records = [row for row in source["records"][str(layer)] if row["corpus"] == "tinystories"]
        pair_values = pair_means(
            records, lambda row, seed=selected_seed: row["native"] - row["rotations"][str(seed)],
        )
        center, interval = clustered_interval(
            pair_values, config["bootstrap_samples"], config["bootstrap_seed"] + 2000 + layer,
        )
        ood[str(layer)] = {"selected_seed": selected_seed, "pair_median": center, "pair_ci": interval}

    fixed_weaker = all(
        row["native_median"] > max(row["fixed_rotation_medians"].values())
        for rows in summaries.values() for row in rows.values()
    )
    lopo_pass = all(
        row["native_minus_lopo_pair_median"] >= 0.03 and row["native_minus_lopo_pair_ci"][0] > 0
        for rows in summaries.values() for row in rows.values()
    )
    ood_pass = all(
        row["pair_median"] >= 0.03 and row["pair_ci"][0] > 0 for row in ood.values()
    )
    oracle_stronger = all(
        row["oracle_minus_lopo_pair_median"] >= 0.03 for rows in summaries.values() for row in rows.values()
    )
    unstable_choice = any(
        global_choices[str(layer)]["fineweb"] != global_choices[str(layer)]["tinystories"]
        for layer in layers
    )
    hypotheses = {
        "T0_every_fixed_rotation_is_weaker_in_the_median": {"supported": fixed_weaker},
        "T1_leave_one_pair_out_rotation_selection_does_not_erase_native_privilege": {"supported": lopo_pass},
        "T2_fineweb_selected_rotation_does_not_erase_ood_native_privilege": {"supported": ood_pass},
        "T3_per_example_oracle_is_strictly_more_powerful_than_frozen_selection": {"supported": oracle_stronger},
        "T4_rotation_choice_is_not_cross_corpus_stable": {"supported": unstable_choice, "choices": global_choices},
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses, "summaries": summaries, "ood": ood,
    }, indent=2, allow_nan=False))

    lines = [
        "# Transfer test for selecting among random rotated bases", "",
        "The earlier best-of-seven control chose a different rotated basis after seeing each held-out example's downstream fidelity. Here one rotation must be frozen using other lexical pairs, or selected on FineWeb and transferred unchanged to TinyStories. Native and rotated programs retain the same rank-256 execution budget; only the basis-selection information changes.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Corpus | Native / best fixed rotation | Native−LOPO rotation median (95% CI) | Oracle-best−LOPO median (95% CI) | Best fixed seed |", "|---:|---|---:|---:|---:|---:|"]
    for layer in layers:
        for corpus in corpora:
            row = summaries[str(layer)][corpus]
            ci = row["native_minus_lopo_pair_ci"]; oci = row["oracle_minus_lopo_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {row['native_median']:.3f}/{row['global_best_median']:.3f} | "
                f"{row['native_minus_lopo_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) | "
                f"{row['oracle_minus_lopo_pair_median']:+.3f} ({oci[0]:+.3f}, {oci[1]:+.3f}) | {row['global_best_seed']} |"
            )
    lines += ["", "FineWeb-selected basis transferred to TinyStories:", ""]
    for layer, row in ood.items():
        ci = row["pair_ci"]
        lines.append(f"- Layer {layer}: seed {row['selected_seed']}, native advantage {row['pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}).")
    lines += [
        "", "A per-example best-of-seven result answers whether a larger adaptive dictionary contains a coordinate system well suited to each edit. The frozen-selection result answers whether one rotated coordinate system is a reusable alternative to the native neuron basis. These are different definitions of basis privilege and should not be merged.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
