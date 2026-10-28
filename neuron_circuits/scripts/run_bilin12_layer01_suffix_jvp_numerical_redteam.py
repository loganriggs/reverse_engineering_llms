#!/usr/bin/env python3
"""AD-duality and finite-radius audit for the midpoint suffix JVP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import cosine, midpoint_product_jvp
from run_bilin12_layer01_suffix_fisher_chart_router import (
    suffix_jvp_batch,
    suffix_logits,
    trace_from_x0,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_suffix_jvp_numerical_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_suffix_jvp_numerical_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_SUFFIX_JVP_NUMERICAL_REDTEAM.md"


def relative_scalar_error(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-30))


def main():
    red = json.loads(CONFIG.read_text())
    fisher = json.loads(Path(red["source_config"]).read_text())
    router = json.loads(Path(fisher["source_config"]).read_text())
    parent = json.loads(Path(router["source_config"]).read_text())
    config = json.loads(Path(parent["source_config"]).read_text())
    pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]] for corpus, spec in config["evaluation"].items()}
    edit_position, position = config["edit_position"], config["prediction_position"]
    generator = torch.Generator(device="cuda").manual_seed(red["duality_seed"])
    records = []
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family, family_pairs in pairs["new_families"].items():
            for source_token, target_token in family_pairs:
                row_index, start = select_windows(rows, source_token, edit_position, position, red["contexts_per_pair"], rng)[0]
                raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy()
                changed = raw.copy(); changed[edit_position] = target_token
                base_ids = torch.as_tensor(raw, device="cuda")[None]; changed_ids = torch.as_tensor(changed, device="cuda")[None]
                midpoint_trace = None
                for layer in config["layers"]:
                    decoder = model.blocks[layer].mlp.Down.weight.float()
                    tangent, point, _ = midpoint_product_jvp(model, base_ids, changed_ids, layer, position, router["path_point"])
                    if midpoint_trace is None:
                        with torch.no_grad(): midpoint_trace = trace_from_x0(model, point, max(config["layers"]))
                    write = F.linear(tangent, decoder)
                    batch_logits, batch_derivative, _, _, _ = suffix_jvp_batch(model, layer, midpoint_trace[layer], [write, write], position)
                    single_logits, single_derivative, primal, output_tangent, function = suffix_jvp_batch(model, layer, midpoint_trace[layer], [write], position)
                    batch_single_cosine = cosine(batch_derivative[0], single_derivative[0])
                    batch_single_error = float((batch_derivative[0] - single_derivative[0]).norm() / single_derivative[0].norm().clamp_min(1e-30))

                    primal_single = primal[:1]
                    tangent_single = output_tangent[:1]
                    x0 = midpoint_trace[layer]["x0"]
                    bus = midpoint_trace[layer]["value_bus"]
                    single_function = lambda value: suffix_logits(model, layer, x0, bus, value, position)
                    with torch.enable_grad():
                        logits, vjp_function = torch.func.vjp(single_function, primal_single)
                    duality_errors = []
                    for _ in range(red["duality_probes"]):
                        cotangent = torch.randn(logits.shape, generator=generator, device="cuda")
                        input_cotangent = vjp_function(cotangent)[0]
                        forward_dot = float((single_derivative * cotangent).sum())
                        reverse_dot = float((tangent_single * input_cotangent).sum())
                        duality_errors.append(relative_scalar_error(forward_dot, reverse_dot))

                    finite = {}
                    with torch.no_grad():
                        for epsilon in red["epsilons"]:
                            positive = single_function(primal_single + epsilon * tangent_single)
                            negative = single_function(primal_single - epsilon * tangent_single)
                            derivative = (positive - negative) / (2 * epsilon)
                            finite[str(epsilon)] = {
                                "cosine": cosine(single_derivative, derivative),
                                "relative_error": float((single_derivative - derivative).norm() / derivative.norm().clamp_min(1e-30)),
                            }
                    records.append({"corpus": corpus, "family": family, "pair": f"{source_token}->{target_token}", "layer": layer, "batch_single_cosine": batch_single_cosine, "batch_single_error": batch_single_error, "duality_errors": duality_errors, "finite": finite})
                print("suffix JVP numeric", corpus, family, source_token, target_token, flush=True)

    summaries = {}
    for layer in config["layers"]:
        summaries[str(layer)] = {}
        for corpus in config["evaluation"]:
            selected = [row for row in records if row["layer"] == layer and row["corpus"] == corpus]
            summaries[str(layer)][corpus] = {
                "batch_single_minimum_cosine": min(row["batch_single_cosine"] for row in selected),
                "batch_single_maximum_error": max(row["batch_single_error"] for row in selected),
                "duality_maximum_error": max(error for row in selected for error in row["duality_errors"]),
                "epsilons": {str(epsilon): {
                    "minimum_cosine": min(row["finite"][str(epsilon)]["cosine"] for row in selected),
                    "median_relative_error": float(np.median([row["finite"][str(epsilon)]["relative_error"] for row in selected])),
                    "maximum_relative_error": max(row["finite"][str(epsilon)]["relative_error"] for row in selected),
                } for epsilon in red["epsilons"]},
            }
    def cell(layer, corpus): return summaries[str(layer)][corpus]
    batch_match = all(cell(l, c)["batch_single_minimum_cosine"] >= .999999 and cell(l, c)["batch_single_maximum_error"] <= 1e-5 for l in config["layers"] for c in config["evaluation"])
    duality = all(cell(l, c)["duality_maximum_error"] <= 1e-4 for l in config["layers"] for c in config["evaluation"])
    passing_epsilons = [epsilon for epsilon in red["epsilons"] if all(cell(l, c)["epsilons"][str(epsilon)]["minimum_cosine"] >= .999 and cell(l, c)["epsilons"][str(epsilon)]["maximum_relative_error"] <= .02 for l in config["layers"] for c in config["evaluation"])]
    curvature = max(cell(l, c)["epsilons"]["0.01"]["maximum_relative_error"] for l in config["layers"] for c in config["evaluation"]) >= .05 and duality
    depth = all(cell(1, c)["epsilons"][str(e)]["median_relative_error"] > cell(0, c)["epsilons"][str(e)]["median_relative_error"] for c in config["evaluation"] for e in red["epsilons"])
    hypotheses = {
        "D0_batched_and_single_jvps_match": {"supported": batch_match},
        "D1_forward_reverse_ad_duality_holds": {"supported": duality},
        "D2_a_smaller_finite_step_validates_the_derivative": {"supported": bool(passing_epsilons), "passing_epsilons": passing_epsilons},
        "D3_the_original_step_measures_finite_curvature": {"supported": curvature},
        "D4_suffix_curvature_increases_with_depth": {"supported": depth},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Numerical red-team of the midpoint suffix JVP", "", "The router's batched forward-mode derivative is compared with a single-example JVP, reverse-mode VJP duality under random output cotangents, and a finite-radius sweep. AD agreement tests implementation correctness without assuming that a finite 1% step is locally linear.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Corpus | Batch min cosine / max error | Duality max error | epsilon: min cosine / median error / max error |", "|---:|---|---:|---:|---|"]
    for layer in config["layers"]:
        for corpus in config["evaluation"]:
            row = cell(layer, corpus); values = "; ".join(f"{epsilon:g}: {row['epsilons'][str(epsilon)]['minimum_cosine']:.6f}/{row['epsilons'][str(epsilon)]['median_relative_error']:.4f}/{row['epsilons'][str(epsilon)]['maximum_relative_error']:.4f}" for epsilon in red["epsilons"])
            lines.append(f"| {layer} | {corpus} | {row['batch_single_minimum_cosine']:.7f}/{row['batch_single_maximum_error']:.2e} | {row['duality_maximum_error']:.2e} | {values} |")
    lines += ["", "If batch/single equality and JVP/VJP duality pass while finite steps fail, the analytic derivative is valid and the failure measures suffix curvature at that intervention radius. The original finite-difference gate remains failed."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
