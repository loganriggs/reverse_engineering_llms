#!/usr/bin/env python3
"""Cauchy-normalized JVP/VJP duality audit robust to near-zero dot products."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import midpoint_product_jvp
from run_bilin12_layer01_suffix_fisher_chart_router import suffix_jvp_batch, suffix_logits, trace_from_x0
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_suffix_jvp_duality_cancellation_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_suffix_jvp_duality_cancellation_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_SUFFIX_JVP_DUALITY_CANCELLATION_REDTEAM.md"


def duality_metrics(jvp, cotangent, tangent, vjp):
    forward = float((jvp * cotangent).sum())
    reverse = float((tangent * vjp).sum())
    cauchy = max(float(jvp.norm() * cotangent.norm()), float(tangent.norm() * vjp.norm()), 1e-30)
    discrepancy = abs(forward - reverse)
    return {
        "forward": forward,
        "reverse": reverse,
        "raw_relative_error": discrepancy / max(abs(forward), abs(reverse), 1e-30),
        "cauchy_normalized_error": discrepancy / cauchy,
        "dot_alignment": max(abs(forward), abs(reverse)) / cauchy,
    }


def main():
    red = json.loads(CONFIG.read_text())
    numerical = json.loads(Path(red["source_config"]).read_text())
    fisher = json.loads(Path(numerical["source_config"]).read_text())
    router = json.loads(Path(fisher["source_config"]).read_text())
    parent = json.loads(Path(router["source_config"]).read_text())
    config = json.loads(Path(parent["source_config"]).read_text())
    pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]] for corpus, spec in config["evaluation"].items()}
    generator = torch.Generator(device="cuda").manual_seed(red["seed"])
    edit_position, position = config["edit_position"], config["prediction_position"]
    records = []
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family, family_pairs in pairs["new_families"].items():
            for source_token, target_token in family_pairs:
                row_index, start = select_windows(rows, source_token, edit_position, position, red["contexts_per_pair"], rng)[0]
                raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy(); changed = raw.copy(); changed[edit_position] = target_token
                base_ids = torch.as_tensor(raw, device="cuda")[None]; changed_ids = torch.as_tensor(changed, device="cuda")[None]
                midpoint_trace = None
                for layer in config["layers"]:
                    decoder = model.blocks[layer].mlp.Down.weight.float()
                    product_tangent, point, _ = midpoint_product_jvp(model, base_ids, changed_ids, layer, position, router["path_point"])
                    if midpoint_trace is None:
                        with torch.no_grad(): midpoint_trace = trace_from_x0(model, point, max(config["layers"]))
                    write = F.linear(product_tangent, decoder)
                    _, derivative, primal, tangent, _ = suffix_jvp_batch(model, layer, midpoint_trace[layer], [write], position)
                    record = midpoint_trace[layer]; single_function = lambda value: suffix_logits(model, layer, record["x0"], record["value_bus"], value, position)
                    with torch.enable_grad(): logits, vjp_function = torch.func.vjp(single_function, primal)
                    probes = []
                    for _ in range(red["probes"]):
                        cotangent = torch.randn(logits.shape, generator=generator, device="cuda")
                        vjp = vjp_function(cotangent)[0]
                        probes.append(duality_metrics(derivative, cotangent, tangent, vjp))
                    records.append({"corpus": corpus, "pair": f"{source_token}->{target_token}", "layer": layer, "probes": probes})
                print("duality cancellation", corpus, family, source_token, target_token, flush=True)

    all_probes = [probe for row in records for probe in row["probes"]]
    raw = np.asarray([probe["raw_relative_error"] for probe in all_probes])
    large = [probe for probe in all_probes if probe["raw_relative_error"] > 1e-4]
    summaries = {}
    for layer in config["layers"]:
        summaries[str(layer)] = {}
        for corpus in config["evaluation"]:
            probes = [probe for row in records if row["layer"] == layer and row["corpus"] == corpus for probe in row["probes"]]
            summaries[str(layer)][corpus] = {
                "maximum_cauchy_error": max(probe["cauchy_normalized_error"] for probe in probes),
                "raw_relative_q99": float(np.quantile([probe["raw_relative_error"] for probe in probes], .99)),
                "minimum_dot_alignment": min(probe["dot_alignment"] for probe in probes),
            }
    cauchy = max(probe["cauchy_normalized_error"] for probe in all_probes) <= 1e-6
    cancellation = all(probe["dot_alignment"] < .01 for probe in large)
    typical = float(np.quantile(raw, .99)) <= 1e-4
    uniform = all(row["maximum_cauchy_error"] <= 1e-6 for cells in summaries.values() for row in cells.values())
    hypotheses = {
        "C0_cauchy_normalized_duality_holds": {"supported": cauchy, "maximum": max(probe["cauchy_normalized_error"] for probe in all_probes)},
        "C1_large_raw_relative_errors_are_cancellation_cases": {"supported": cancellation, "count": len(large)},
        "C2_typical_raw_relative_duality_is_precise": {"supported": typical, "q99": float(np.quantile(raw, .99))},
        "C3_duality_validation_is_uniform": {"supported": uniform},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Cancellation-robust forward/reverse AD duality red-team", "", "Raw relative error is unstable when the JVP and random output cotangent are nearly orthogonal. This audit also divides the bilinear-form discrepancy by its Cauchy bound, which remains well scaled at a zero dot product.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Corpus | Maximum Cauchy-normalized error | Raw-relative q99 | Minimum dot alignment |", "|---:|---|---:|---:|---:|"]
    for layer in config["layers"]:
        for corpus in config["evaluation"]:
            row = summaries[str(layer)][corpus]; lines.append(f"| {layer} | {corpus} | {row['maximum_cauchy_error']:.2e} | {row['raw_relative_q99']:.2e} | {row['minimum_dot_alignment']:.2e} |")
    lines += ["", "This does not turn a finite-radius failure into a pass. It determines whether the analytic tangent used for chart routing is the correct derivative of the implemented tensor network."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
