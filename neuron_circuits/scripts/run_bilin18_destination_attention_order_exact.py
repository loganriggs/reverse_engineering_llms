#!/usr/bin/env python3
"""Exact all-order red team for the destination attention path expansion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_destination_attention_order import first_passing
from run_bilin18_destination_path_predictor import (
    analytic_candidate_paths,
    fidelity,
    geometry_correlation,
    median_angle_partner,
    response,
)
from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import build_function_bank
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_destination_attention_order_exact.json")
OUTPUT = TENSOR_ROOT / "bilin18_destination_attention_order_exact.json"
REPORT = TENSOR_ROOT / "BILIN18_DESTINATION_ATTENTION_ORDER_EXACT.md"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_config"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    bank, raw_gram, norms = build_function_bank(model, source["candidate_layers"])
    labels = [row["label"] for row in bank]
    corpus = np.load(source["corpus"], mmap_mode="r")
    rng = np.random.default_rng(source["evaluation_random_seed"])
    all_indices = np.sort(rng.choice(
        np.arange(source["evaluation_index_low"], source["evaluation_index_high"]),
        source["evaluation_rows"], replace=False,
    ))
    # Preserve both halves of the prospective panel while reducing this costly
    # invariant audit to the first four already-fixed rows from each half.
    half_size = source["evaluation_rows"] // 2
    count = config["evaluation_rows"] // 2
    indices = np.concatenate((all_indices[:count], all_indices[half_size:half_size + count]))
    raw = np.asarray(corpus[indices, :source["sequence_length"]], dtype=np.int64)
    halves = {"a": raw[:count], "b": raw[count:]}
    positions = torch.as_tensor(source["positions"], device="cuda")
    targets = {}
    maximum_forward_error = 0.0
    maximum_recomposition_error = 0.0

    for target_layer, target_head in source["targets"]:
        target_label = f"L{target_layer}H{target_head}"
        target = labels.index(target_label)
        partner = median_angle_partner(raw_gram, norms, target)
        partner_scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        max_order = len(model.blocks) - target_layer - 1
        half_rows = {}
        for half, panel in halves.items():
            print("exact attention order", target_label, half, flush=True)
            cumulative = {
                order: {"prediction2": 0.0, "dot": 0.0, "gram": None}
                for order in range(max_order + 1)
            }
            full2 = 0.0
            full_gram = None
            row_errors = []
            for row_index, row in enumerate(panel):
                ids = torch.as_tensor(row, device="cuda")[None]
                analytic = analytic_candidate_paths(
                    model, ids, bank[target], bank[partner], partner_scale,
                    controls, bank, positions, max_order=max_order,
                )
                row_errors.append(analytic["order_recomposition_relative_error"])
                maximum_recomposition_error = max(maximum_recomposition_error, row_errors[-1])
                maximum_forward_error = max(maximum_forward_error, analytic["attention_forward_error"])
                _, full_response, site_full_gram = response(
                    model, analytic["final_state"], analytic["full"],
                )
                full2 += float(full_response.square().sum())
                if full_gram is None:
                    full_gram = np.zeros_like(site_full_gram.cpu().numpy())
                full_gram += site_full_gram.cpu().numpy()
                running = torch.zeros_like(analytic["orders"][:, 0])
                for order in range(max_order + 1):
                    running = running + analytic["orders"][:, order]
                    _, predicted_response, predicted_gram = response(
                        model, analytic["final_state"], running,
                    )
                    item = cumulative[order]
                    item["prediction2"] += float(predicted_response.square().sum())
                    item["dot"] += float((predicted_response * full_response).sum())
                    if item["gram"] is None:
                        item["gram"] = np.zeros_like(predicted_gram.cpu().numpy())
                    item["gram"] += predicted_gram.cpu().numpy()
                print(" ", row_index + 1, "/", len(panel), flush=True)
            orders = {}
            for order, item in cumulative.items():
                orders[str(order)] = {
                    "response_fidelity": fidelity(full2, item["prediction2"], item["dot"]),
                    "fisher_geometry_correlation": geometry_correlation(full_gram, item["gram"]),
                }
            half_rows[half] = {
                "orders": orders,
                "first_passing_order": first_passing(orders, .90),
                "maximum_state_recomposition_error": max(row_errors),
            }
        targets[target_label] = {
            "partner": labels[partner],
            "max_order": max_order,
            "halves": half_rows,
        }

    r1 = maximum_recomposition_error <= 1e-4
    r2 = all(
        row["orders"][str(target["max_order"])]["response_fidelity"] >= .999
        and row["orders"][str(target["max_order"])]["fisher_geometry_correlation"] >= .999
        for target in targets.values() for row in target["halves"].values()
    )
    r3 = all(
        targets["L0H0"]["halves"][half]["orders"]["6"]["response_fidelity"] < 0
        and targets["L0H0"]["halves"][half]["orders"][str(targets["L0H0"]["max_order"])]["response_fidelity"] >= .999
        for half in halves
    )
    r4 = all(
        targets["L0H0"]["halves"][half]["first_passing_order"]
        > targets["L5H3"]["halves"][half]["first_passing_order"]
        for half in halves
    )
    hypotheses = {
        "R1_full_order_expansion_recomposes_exact_jvp": {"supported": r1, "maximum_relative_error": maximum_recomposition_error},
        "R2_full_order_expansion_recomposes_fisher_response": {"supported": r2},
        "R3_low_order_explosion_is_resolved_by_cancellation": {"supported": r3},
        "R4_earlier_target_needs_higher_attention_order": {"supported": r4},
        "attention_jvp_forward_invariant": {"supported": maximum_forward_error <= 1e-5, "maximum_relative_error": maximum_forward_error},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({
        "config": config,
        "evaluation_indices": indices.tolist(),
        "hypotheses": hypotheses,
        "targets": targets,
    }, indent=2, allow_nan=False))

    lines = [
        "# Exact all-order attention-path recomposition", "",
        "This audit was locked after the order-6 expansion exploded for early targets. It includes every possible downstream attention order, so its sum must equal the independently propagated full Jacobian. Four already-fixed rows from each prospective half are used; no examples were reselected.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Full frontier", ""]
    for target, target_row in targets.items():
        lines += [f"### {target}", "", "| Half | First ≥.90 | Final fidelity / geometry | Order-6 fidelity | State error |", "|---|---:|---:|---:|---:|"]
        for half, row in target_row["halves"].items():
            final = row["orders"][str(target_row["max_order"])]
            order6 = row["orders"][str(min(6, target_row["max_order"]))]["response_fidelity"]
            lines.append(f"| {half} | {row['first_passing_order']} | {final['response_fidelity']:.6f} / {final['fisher_geometry_correlation']:.6f} | {order6:.3f} | {row['maximum_state_recomposition_error']:.2e} |")
        lines.append("")
    lines += [
        "## Guardrails", "",
        "Exact recomposition validates the algebra, not a human-readable ontology. Large individual orders followed by cancellation mean that truncating by apparent path energy can be actively misleading. The first-passing order is a property of this local expansion and corpus panel; it is not the number of discrete reasoning steps taken by the model.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
