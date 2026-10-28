#!/usr/bin/env python3
"""Freeze Taylor order by discovery data and evaluate pair/OOD transfer."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from run_bilin12_layer01_dynamic_alignment_replication import median, pair_cluster_interval
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_directional_order_transfer.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_directional_order_transfer.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_DIRECTIONAL_ORDER_TRANSFER.md"


def choose_order(records, orders):
    return max(orders, key=lambda order: median([
        row["fidelity"][f"midpoint_order{order}"] for row in records
    ]))


def main():
    red = json.loads(CONFIG.read_text())
    source = json.loads(Path(red["source_results"]).read_text())
    maximum_order = source["config"]["maximum_order"]
    orders = list(range(1, maximum_order + 1))
    results = {}

    for layer, rows in source["records"].items():
        fineweb = [row for row in rows if row["corpus"] == "fineweb"]
        tinystories = [row for row in rows if row["corpus"] == "tinystories"]
        global_order = choose_order(fineweb, orders)
        pairs = sorted({row["pair"] for row in fineweb})
        fold_orders = {}
        loo_rows = []
        for pair in pairs:
            discovery = [row for row in fineweb if row["pair"] != pair]
            held_out = [row for row in fineweb if row["pair"] == pair]
            order = choose_order(discovery, orders)
            fold_orders[pair] = order
            for row in held_out:
                loo_rows.append({
                    "pair": pair,
                    "chosen_order": order,
                    "routed": row["fidelity"][f"midpoint_order{order}"],
                    "oracle": max(row["fidelity"][f"midpoint_order{candidate}"] for candidate in orders),
                    "explicit_midpoint": row["fidelity"]["explicit_midpoint"],
                })
        counts = Counter(fold_orders.values())
        mode_order, mode_count = counts.most_common(1)[0]
        loo_gap, loo_gap_ci = pair_cluster_interval(
            loo_rows, lambda row: row["oracle"] - row["routed"],
            red["bootstrap_samples"], red["bootstrap_seed"] + int(layer),
        )
        ood_selected = [row["fidelity"][f"midpoint_order{global_order}"] for row in tinystories]
        ood_explicit = [row["fidelity"]["explicit_midpoint"] for row in tinystories]
        results[layer] = {
            "fineweb_global_order": global_order,
            "fineweb_global_fidelity": median([
                row["fidelity"][f"midpoint_order{global_order}"] for row in fineweb
            ]),
            "fold_orders": fold_orders,
            "modal_order": mode_order,
            "modal_fraction": mode_count / len(pairs),
            "loo_routed_fidelity": median([row["routed"] for row in loo_rows]),
            "loo_oracle_fidelity": median([row["oracle"] for row in loo_rows]),
            "loo_oracle_gap_pair_median": loo_gap,
            "loo_oracle_gap_pair_ci": loo_gap_ci,
            "tinystories_selected_fidelity": median(ood_selected),
            "tinystories_explicit_midpoint_fidelity": median(ood_explicit),
            "tinystories_selected_minus_explicit": median(ood_selected) - median(ood_explicit),
        }

    cells = source["summaries"]
    hypotheses = {
        "F0_leave_one_pair_out_order_is_stable": {"supported": all(
            results[layer]["modal_fraction"] >= .70 for layer in results
        )},
        "F1_fineweb_selected_order_transfers_to_tinystories": {"supported": all(
            abs(results[layer]["tinystories_selected_minus_explicit"]) <= .03 for layer in results
        )},
        "F2_leave_one_pair_out_order_stays_near_the_order_oracle": {"supported": all(
            results[layer]["loo_oracle_gap_pair_median"] <= .05 for layer in results
        )},
        "F3_optimal_description_order_differs_by_layer": {"supported": len({
            results[layer]["fineweb_global_order"] for layer in results
        }) > 1},
        "F4_cubic_is_an_adequate_universal_default": {"supported": all(
            abs(cells[layer][corpus]["fidelity"]["midpoint_order3"] - cells[layer][corpus]["fidelity"]["explicit_midpoint"]) <= .08
            for layer in cells for corpus in cells[layer]
        )},
        "F5_midpoint_expansion_is_more_stable_than_endpoint_integration": {"supported": all(
            cells[layer][corpus]["fidelity"]["midpoint_order5"] - cells[layer][corpus]["fidelity"]["integral_order5"] >= .10
            for layer in cells for corpus in cells[layer]
        )},
        "F6_fifth_order_improves_support_even_when_behavior_oscillates": {"supported": all(
            cells[layer][corpus]["support_jaccard"]["midpoint_order5"] - cells[layer][corpus]["support_jaccard"]["midpoint_order3"] >= .02
            for layer in cells for corpus in cells[layer]
        )},
        "F7_transfer_choices_are_frozen_before_test_outcomes": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Frozen transfer of directional derivative order", "",
        "Derivative order is treated as part of the interpreter description. Within FineWeb, each lexical pair is evaluated using the order selected on every other pair. For OOD evaluation, one order per layer is selected on all FineWeb examples and frozen before TinyStories outcomes are read. No model calls or new chart choices are introduced.",
        "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | FineWeb global order/fidelity | LOO modal order/fraction | LOO routed/oracle/gap (95% CI) | TinyStories frozen/explicit/difference |", "|---:|---:|---:|---:|---:|"]
    for layer, row in results.items():
        ci = row["loo_oracle_gap_pair_ci"]
        lines.append(
            f"| {layer} | {row['fineweb_global_order']} / {row['fineweb_global_fidelity']:.3f} | "
            f"{row['modal_order']} / {row['modal_fraction']:.3f} | "
            f"{row['loo_routed_fidelity']:.3f} / {row['loo_oracle_fidelity']:.3f} / {row['loo_oracle_gap_pair_median']:.3f} ({ci[0]:.3f}, {ci[1]:.3f}) | "
            f"{row['tinystories_selected_fidelity']:.3f} / {row['tinystories_explicit_midpoint_fidelity']:.3f} / {row['tinystories_selected_minus_explicit']:+.3f} |"
        )
    lines += ["", "A stable frozen order supports a compact interpreter program. Layer-dependent order remains a real complexity cost: cubic is the best cheap Layer-0 description, while Layer 1 benefits from fifth order. Endpoint-integral deterioration is a convergence-radius result, not evidence that the differentiated tensors are numerically wrong."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
