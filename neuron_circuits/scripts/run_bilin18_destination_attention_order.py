#!/usr/bin/env python3
"""Measure how many composed downstream attention operations explain compatibility."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

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
from run_bilin18_variable_function_atlas import spearman


CONFIG = Path("configs/bilin18_destination_attention_order.json")
OUTPUT = TENSOR_ROOT / "bilin18_destination_attention_order.json"
REPORT = TENSOR_ROOT / "BILIN18_DESTINATION_ATTENTION_ORDER.md"


def first_passing(rows, threshold):
    for order in sorted(rows, key=int):
        if rows[order]["response_fidelity"] >= threshold:
            return int(order)
    return None


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_config"]).read_text())
    max_order = config["max_attention_order"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    bank, raw_gram, norms = build_function_bank(model, source["candidate_layers"])
    labels = [row["label"] for row in bank]
    corpus = np.load(source["corpus"], mmap_mode="r")
    rng = np.random.default_rng(source["evaluation_random_seed"])
    indices = np.sort(rng.choice(
        np.arange(source["evaluation_index_low"], source["evaluation_index_high"]),
        source["evaluation_rows"], replace=False,
    ))
    raw = np.asarray(corpus[indices, :source["sequence_length"]], dtype=np.int64)
    halves = {"a": raw[:len(raw) // 2], "b": raw[len(raw) // 2:]}
    positions = torch.as_tensor(source["positions"], device="cuda")
    targets = {}
    maximum_forward_error = 0.0

    for target_layer, target_head in source["targets"]:
        target_label = f"L{target_layer}H{target_head}"
        target = labels.index(target_label)
        partner = median_angle_partner(raw_gram, norms, target)
        partner_scale, controls, _ = complete_controls(raw_gram, norms, target, partner)
        halves_result = {}
        for half, panel in halves.items():
            print("attention order", target_label, half, flush=True)
            cumulative = {
                order: {"prediction2": 0.0, "dot": 0.0, "gram": None}
                for order in range(max_order + 1)
            }
            full2 = 0.0
            order_energy = np.zeros(max_order + 1)
            full_gram = None
            for row_index, row in enumerate(panel):
                ids = torch.as_tensor(row, device="cuda")[None]
                analytic = analytic_candidate_paths(
                    model, ids, bank[target], bank[partner], partner_scale,
                    controls, bank, positions, max_order=max_order,
                )
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
                    _, exact_order_response, _ = response(
                        model, analytic["final_state"], analytic["orders"][:, order],
                    )
                    order_energy[order] += float(exact_order_response.square().sum())
                    running = running + analytic["orders"][:, order]
                    _, predicted_response, predicted_gram = response(
                        model, analytic["final_state"], running,
                    )
                    cumulative[order]["prediction2"] += float(predicted_response.square().sum())
                    cumulative[order]["dot"] += float((predicted_response * full_response).sum())
                    if cumulative[order]["gram"] is None:
                        cumulative[order]["gram"] = np.zeros_like(predicted_gram.cpu().numpy())
                    cumulative[order]["gram"] += predicted_gram.cpu().numpy()
                print(" ", row_index + 1, "/", len(panel), flush=True)
            order_rows = {}
            for order, item in cumulative.items():
                order_rows[str(order)] = {
                    "response_fidelity": fidelity(full2, item["prediction2"], item["dot"]),
                    "fisher_geometry_correlation": geometry_correlation(full_gram, item["gram"]),
                }
            normalized_energy = order_energy / max(order_energy.sum(), 1e-30)
            halves_result[half] = {
                "orders": order_rows,
                "exact_order_energy_fraction": normalized_energy.tolist(),
                "first_passing_order": first_passing(order_rows, config["fidelity_threshold"]),
            }
        stability = spearman(
            halves_result["a"]["exact_order_energy_fraction"],
            halves_result["b"]["exact_order_energy_fraction"],
        )
        targets[target_label] = {
            "partner": labels[partner],
            "order_energy_split_stability": stability,
            "halves": halves_result,
        }

    all_halves = [row for target in targets.values() for row in target["halves"].values()]
    o1 = all(
        row["orders"]["2"]["response_fidelity"]
        >= row["orders"]["1"]["response_fidelity"] + .10
        for row in all_halves
    )
    o2 = all(
        row["orders"][str(max_order)]["response_fidelity"] >= .90
        and row["orders"][str(max_order)]["fisher_geometry_correlation"] >= .95
        for row in all_halves
    )
    ordered_targets = [f"L{layer}H{head}" for layer, head in source["targets"]]
    o3 = True
    for half in halves:
        values = [targets[target]["halves"][half]["first_passing_order"] for target in ordered_targets]
        o3 = o3 and all(value is not None for value in values) and all(
            values[index] >= values[index + 1] for index in range(len(values) - 1)
        )
    o4 = all(
        row["first_passing_order"] is None or row["first_passing_order"] >= 2
        for row in all_halves
    )
    o5 = all(target["order_energy_split_stability"] >= .80 for target in targets.values())
    hypotheses = {
        "O1_second_order_materially_improves_all_targets": {"supported": o1},
        "O2_order_six_is_sufficient": {"supported": o2},
        "O3_earlier_destinations_require_higher_order": {"supported": o3},
        "O4_multi_attention_composition_is_necessary": {"supported": o4},
        "O5_order_energy_profile_is_split_stable": {"supported": o5},
        "attention_jvp_forward_invariant": {
            "supported": maximum_forward_error <= 1e-5,
            "maximum_relative_error": maximum_forward_error,
        },
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
        "# Attention-composition order in destination compatibility", "",
        "After the preregistered single-attention program failed, this follow-up groups the exact analytic tensor expansion by the number of downstream attention Jacobians in a path. Order 0 is the attention-free residual/MLP route; order 1 sums every path with exactly one downstream attention use; higher orders include increasingly long attention-mediated chains. No finite outcomes are used in this decomposition.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Cumulative order frontier", "", "| Target / half | First ≥.90 | " + " | ".join(f"order {order}" for order in range(max_order + 1)) + " |", "|---|---:" + "|---:" * (max_order + 1) + "|"]
    for target, target_row in targets.items():
        for half, row in target_row["halves"].items():
            values = " | ".join(f"{row['orders'][str(order)]['response_fidelity']:.3f}" for order in range(max_order + 1))
            lines.append(f"| {target} / {half} | {row['first_passing_order']} | {values} |")
        lines.append(f"| {target} energy split stability | {target_row['order_energy_split_stability']:.3f} | " + " | ".join("—" for _ in range(max_order + 1)) + " |")
    lines += [
        "", "## Guardrails", "",
        "Orders are algebraic path groups in the local Jacobian, not claims that the model executes a discrete sequence of symbolic calls. Cumulative fidelity can be non-monotonic because different path orders cancel. An early layer has more possible downstream attention operations than a late layer, so depth and learned organization remain partially confounded. The split test addresses sampling stability, not that structural confound.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
