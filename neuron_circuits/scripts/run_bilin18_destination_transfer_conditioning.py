#!/usr/bin/env python3
"""Audit the conditioning of the compact layerwise tangent recurrence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_destination_path_predictor import (
    analytic_candidate_paths,
    median_angle_partner,
)
from run_bilin18_full_operator_control_census import complete_controls
from run_bilin18_full_operator_transplant import build_function_bank
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_variable_function_atlas import spearman


CONFIG = Path("configs/bilin18_destination_transfer_conditioning.json")
OUTPUT = TENSOR_ROOT / "bilin18_destination_transfer_conditioning.json"
REPORT = TENSOR_ROOT / "BILIN18_DESTINATION_TRANSFER_CONDITIONING.md"


def summarize(records):
    layers = sorted({row["layer"] for record in records for row in record})
    result = []
    for layer in layers:
        rows = [row for record in records for row in record if row["layer"] == layer]
        result.append({
            "layer": layer,
            **{
                key: float(np.median([row[key] for row in rows]))
                for key in (
                    "gain", "attention_to_identity", "branch_cosine",
                    "cancellation_condition",
                )
            },
        })
    return result


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
        half_rows = {}
        for half, panel in halves.items():
            print("transfer conditioning", target_label, half, flush=True)
            records = []
            for row_index, row in enumerate(panel):
                ids = torch.as_tensor(row, device="cuda")[None]
                analytic = analytic_candidate_paths(
                    model, ids, bank[target], bank[partner], partner_scale,
                    controls, bank, positions, max_order=1,
                )
                maximum_forward_error = max(maximum_forward_error, analytic["attention_forward_error"])
                records.append(analytic["transfer_stats"])
                print(" ", row_index + 1, "/", len(panel), flush=True)
            profile = summarize(records)
            conditions = np.asarray([row["cancellation_condition"] for row in profile])
            top = np.argsort(conditions)[::-1][:3]
            half_rows[half] = {
                "profile": profile,
                "strong_cancellation_layers": int((conditions >= 1.5).sum()),
                "median_attention_to_identity": float(np.median([
                    row["attention_to_identity"] for row in profile
                ])),
                "log_condition_sum": float(np.log(conditions).sum()),
                "top_three_layers": [profile[index]["layer"] for index in top],
            }
        by_half = {
            half: {row["layer"]: row for row in value["profile"]}
            for half, value in half_rows.items()
        }
        common = sorted(set(by_half["a"]) & set(by_half["b"]))
        stability = spearman(
            [by_half["a"][layer]["cancellation_condition"] for layer in common],
            [by_half["b"][layer]["cancellation_condition"] for layer in common],
        )
        overlap = len(set(half_rows["a"]["top_three_layers"]) & set(half_rows["b"]["top_three_layers"]))
        targets[target_label] = {
            "partner": labels[partner],
            "condition_profile_split_stability": stability,
            "top_three_layer_overlap": overlap,
            "halves": half_rows,
        }

    c1 = all(targets["L0H0"]["halves"][half]["strong_cancellation_layers"] >= 3 for half in halves)
    c2 = all(target["condition_profile_split_stability"] >= .80 for target in targets.values())
    c3 = all(row["median_attention_to_identity"] >= .30 for target in targets.values() for row in target["halves"].values())
    order = [f"L{layer}H{head}" for layer, head in source["targets"]]
    c4 = all(
        all(targets[order[index]]["halves"][half]["log_condition_sum"]
            >= targets[order[index + 1]]["halves"][half]["log_condition_sum"]
            for index in range(len(order) - 1))
        for half in halves
    )
    c5 = all(target["top_three_layer_overlap"] >= 2 for target in targets.values())
    hypotheses = {
        "C1_early_transfer_contains_strong_local_cancellations": {"supported": c1},
        "C2_transfer_conditioning_profile_is_split_stable": {"supported": c2},
        "C3_attention_is_not_a_small_correction": {"supported": c3},
        "C4_earlier_targets_accumulate_more_conditioning": {"supported": c4},
        "C5_same_layers_are_cancellation_bottlenecks": {"supported": c5},
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
        "# Conditioning of the destination transfer recurrence", "",
        "The exact Jacobian has a short compositional program even when its expanded paths are unusable. At each downstream layer the incoming candidate bundle is split into the residual/MLP branch and the dynamic-attention branch, then recombined. `Cancellation condition = (||identity|| + ||attention||) / ||sum||`; one means aligned/non-cancelling and larger values mean the additive path basis is ill-conditioned.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Summary", "", "| Target / half | Strong-cancel layers | Median attention/identity | Sum log condition | Top-3 layers |", "|---|---:|---:|---:|---|"]
    for target, target_row in targets.items():
        for half, row in target_row["halves"].items():
            lines.append(f"| {target} / {half} | {row['strong_cancellation_layers']} | {row['median_attention_to_identity']:.3f} | {row['log_condition_sum']:.3f} | {row['top_three_layers']} |")
        lines.append(f"| {target} split stability / top-3 overlap | {target_row['condition_profile_split_stability']:.3f} | — | — | {target_row['top_three_layer_overlap']} |")
    lines += [
        "", "## Guardrails", "",
        "These norms use the checkpoint's residual coordinates and aggregate candidates, tokens, and examples. They diagnose numerical/mechanistic conditioning of one replacement family, not semantic importance. A well-conditioned local split can still participate in a badly conditioned long product; conversely, large local cancellation can be harmless when represented as one composed transfer operator rather than expanded paths.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
