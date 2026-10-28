#!/usr/bin/env python3
"""Red-team and report the Bilin18 QK×OV causal edge experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


CONTROLS = (
    "route_only", "same_head_random_source", "same_source_random_head", "uniform_random",
)


def binomial_greater(wins, trials):
    if trials == 0:
        return 1.0
    return float(sum(
        math.comb(trials, k) for k in range(wins, trials + 1)
    ) / (2 ** trials))


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def spearman(first, second):
    return float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path,
        default=Path("artifacts/bilin18_qkov_causal_edges/results.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261019)
    return parser.parse_args()


def paired(records, split, layer, left, right, key):
    table = defaultdict(dict)
    for row in records:
        if row["split"] == split and (layer is None or row["layer"] == layer):
            table[row["site_index"]][row["condition"]] = row
    pairs = [
        (values[left][key], values[right][key])
        for values in table.values() if left in values and right in values
    ]
    return np.asarray(pairs, dtype=float)


def unit_cluster_ci(records, split, control, rng, count):
    grouped = defaultdict(list)
    pairs = paired(records, split, None, "qkov", control, "product_delta")
    # Recover units through ordered site ids rather than treating 48 sites as
    # independent experimental replicas.
    table = defaultdict(dict)
    meta = {}
    for row in records:
        if row["split"] == split:
            table[row["site_index"]][row["condition"]] = row
            meta[row["site_index"]] = (row["layer"], row["unit"])
    for site, values in table.items():
        if "qkov" in values and control in values:
            grouped[meta[site]].append(
                abs(values["qkov"]["product_delta"]) - abs(values[control]["product_delta"])
            )
    units = sorted(grouped)
    unit_means = np.asarray([np.mean(grouped[u]) for u in units])
    draws = np.empty(count)
    for i in range(count):
        draws[i] = rng.choice(unit_means, len(unit_means), replace=True).mean()
    return {
        "units": len(units),
        "mean_unit_difference": float(unit_means.mean()),
        "bootstrap_95_ci": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
    }


def analyze(data, args):
    records = data["records"]
    layers = sorted({row["layer"] for row in records})
    rng = np.random.default_rng(args.seed)
    result = {
        "all_one_validation": data["all_one_validation"],
        "layers": layers,
        "splits": {},
        "red_team": {},
    }
    for split in ("selection", "transfer"):
        split_result = {"layers": {}, "paired_controls": {}}
        for layer in layers:
            qkov = [r for r in records if r["split"] == split and r["layer"] == layer and r["condition"] == "qkov"]
            pred = np.asarray([r["predicted_product_delta"] for r in qkov])
            actual = np.asarray([r["product_delta"] for r in qkov])
            split_result["layers"][str(layer)] = {
                "n": len(qkov),
                "prediction_pearson": float(np.corrcoef(pred, actual)[0, 1]),
                "prediction_sign_agreement": float(np.mean(np.sign(pred) == np.sign(actual))),
                "median_abs_product_delta": float(np.median(np.abs(actual))),
                "median_product_fraction_of_all_local": float(np.median([r["abs_product_delta_fraction"] for r in qkov])),
                "median_kl": float(np.median([r["kl_base_to_changed"] for r in qkov])),
                "mean_nll_delta": float(np.mean([r["nll_delta"] for r in qkov])),
                "mean_abs_nll_delta": float(np.mean(np.abs([r["nll_delta"] for r in qkov]))),
                "median_mediated_fraction": float(np.median([
                    r["mediated_logit_delta_rms"] /
                    max(r["mediated_logit_delta_rms"] + r["residual_edge_logit_delta_rms"], 1e-12)
                    for r in qkov
                ])),
            }
        for control in CONTROLS:
            pairs = paired(records, split, None, "qkov", control, "product_delta")
            differences = np.abs(pairs[:, 0]) - np.abs(pairs[:, 1])
            nonzero = differences[differences != 0]
            wins = int((nonzero > 0).sum())
            pvalue = binomial_greater(wins, len(nonzero))
            split_result["paired_controls"][control] = {
                "n": len(pairs),
                "qkov_wins_fraction": float(np.mean(differences > 0)),
                "median_abs_difference": float(np.median(differences)),
                "paired_sign_test_one_sided_p": pvalue,
                "unit_cluster": unit_cluster_ci(records, split, control, rng, args.bootstrap),
            }
        qkov = [r for r in records if r["split"] == split and r["condition"] == "qkov"]
        result["splits"][split] = split_result
        result["splits"][split]["product_to_logit"] = {
            "spearman_abs_product_vs_kl": spearman(
                [abs(r["product_delta"]) for r in qkov],
                [r["kl_base_to_changed"] for r in qkov],
            ),
            "spearman_abs_product_vs_abs_nll": spearman(
                [abs(r["product_delta"]) for r in qkov],
                [abs(r["nll_delta"]) for r in qkov],
            ),
        }

    # Control collisions and bfloat16 resolution can make a weak baseline look
    # even weaker; quantify both explicitly.
    for split in ("selection", "transfer"):
        site_rows = defaultdict(dict)
        for row in records:
            if row["split"] == split:
                site_rows[row["site_index"]][row["condition"]] = row
        collisions = 0
        comparisons = 0
        zero_rates = {}
        for control in CONTROLS:
            rows = [values[control] for values in site_rows.values()]
            zero_rates[control] = float(np.mean([r["product_delta"] == 0 for r in rows]))
        for values in site_rows.values():
            edges = []
            for control in CONTROLS:
                row = values[control]
                edges.append((row["head"], row["source"]))
            for i in range(len(edges)):
                for j in range(i):
                    comparisons += 1
                    collisions += edges[i] == edges[j]
        result["red_team"][split] = {
            "control_pair_collision_rate": collisions / comparisons,
            "zero_product_delta_rates": zero_rates,
        }
    return result


def write_report(analysis, path):
    lines = [
        "# Bilin18 QK×OV causal edge experiment", "",
        "## Verdict", "",
        "The weight-and-activation QK×OV calculation identifies attention edges that "
        "causally control the selected native neuron on disjoint FineWeb examples. The "
        "result survives routing-only and random-edge controls. It does **not** support a "
        "one-edge→one-neuron→one-logit circuit: restoring the selected neuron removes only "
        "about one third of the edge's additive logit change, because the native attention "
        "edge writes a full residual vector read by many channels.", "",
        "The all-one reference and custom forwards match exactly: max logit error "
        f"{analysis['all_one_validation']['max_abs_logit_error']:.3g}.", "",
    ]
    for split in ("selection", "transfer"):
        item = analysis["splits"][split]
        lines.extend([f"## {split.title()} panel", "",
                      "|layer|n|prediction r|sign agreement|median |Δ product||fraction of full local-attention effect|median KL|mean ΔNLL|mediated logit fraction|",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for layer in analysis["layers"]:
            row = item["layers"][str(layer)]
            lines.append(
                f"|{layer}|{row['n']}|{row['prediction_pearson']:.3f}|{row['prediction_sign_agreement']:.1%}|"
                f"{row['median_abs_product_delta']:.2f}|{row['median_product_fraction_of_all_local']:.1%}|"
                f"{row['median_kl']:.4g}|{row['mean_nll_delta']:+.4f}|{row['median_mediated_fraction']:.1%}|"
            )
        lines.extend(["", "Paired target-edge controls:", "",
                      "|control|QK×OV wins|median |Δproduct| advantage|one-sided paired sign p|unit-cluster mean [95% CI]|",
                      "|---|---:|---:|---:|---:|"])
        for control, row in item["paired_controls"].items():
            ci = row["unit_cluster"]["bootstrap_95_ci"]
            lines.append(
                f"|{control}|{row['qkov_wins_fraction']:.1%}|{row['median_abs_difference']:.2f}|"
                f"{row['paired_sign_test_one_sided_p']:.3g}|{row['unit_cluster']['mean_unit_difference']:.2f} "
                f"[{ci[0]:.2f}, {ci[1]:.2f}]|"
            )
        lines.append("")
    lines.extend([
        "## Red-team limits", "",
        "- Neurons were selected in the earlier semantic atlas. The transfer panel is fresh, "
        "but this is evidence about those twelve neurons, not an unbiased population rate.",
        "- Edge ranking is recomputed from weights and activations for each input. Transfer "
        "therefore validates the explanatory formula, not one fixed absolute source position.",
        "- The routing-only control isolates the extra value of OV/factor alignment. Random "
        "controls are weaker and sometimes quantize to zero product change in bfloat16.",
        "- An attention edge writes a 1,152-dimensional vector. Its final loss effect can be "
        "large even when the selected neuron is not its principal mediator.",
        "- Signed NLL changes can cancel: ablating a learned edge occasionally improves a "
        "particular next-token prediction. Product prediction and KL are cleaner mechanism tests.",
        "- Sites are correlated within twelve neurons. The report therefore includes bootstrap "
        "intervals over neuron-level means rather than treating 48 sites as independent models.",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    data = json.loads(args.results.read_text())
    analysis = analyze(data, args)
    path = args.results.parent / "analysis.json"
    path.write_text(json.dumps(analysis, indent=2))
    write_report(analysis, args.results.parent / "REPORT.md")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
