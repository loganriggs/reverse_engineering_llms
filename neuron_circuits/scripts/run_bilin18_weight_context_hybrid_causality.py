#!/usr/bin/env python3
"""Finite validation of fixed-rank weight/context route allocations."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_ce_adjoint_residual_atlas import fit_basis
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, intervene, native_routes, pearson, update_metric
from run_bilin18_weight_context_route_hybrid import hybrid_basis, project


CONFIG = Path("configs/bilin18_weight_context_hybrid_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_weight_context_hybrid_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_WEIGHT_CONTEXT_HYBRID_CAUSALITY.md"


def semantic_reconstruction(routes, centers, threshold):
    if centers is None: return torch.zeros_like(routes)
    norm = routes.norm(dim=-1, keepdim=True); unit = routes / norm.clamp_min(1e-30)
    similarity = unit @ centers.T; assignment = similarity.argmax(-1); selected = similarity.max(-1).values >= threshold
    return torch.where(selected[..., None], norm * centers[assignment], 0)


def main():
    config = json.loads(CONFIG.read_text()); weight_source = json.loads(Path(config["weight_source"]).read_text())
    weights = np.load(config["weight_bases"]); residual_arrays = np.load(config["residual_arrays"])
    semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text()); geometry = json.loads(Path(config["hybrid_geometry"]).read_text())
    weight_rows = {row["layer"]: row for row in weight_source["layers"]}; geometry_rows = {row["layer"]: row for row in geometry["layers"]}
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    semantic_by_layer = {layer: [] for layer in config["layers"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in semantic_by_layer: semantic_by_layer[row["layer"]].append(row["center"])
    bases = {}; centers = {}
    for layer in config["layers"]:
        discovery = residual_arrays[f"discovery_residual_routes_L{layer}"]; method = weight_rows[layer]["best_method"]; weight = weights[f"{method}_basis_L{layer}"]
        bases[layer] = {k: torch.as_tensor(hybrid_basis(weight, discovery, k, config["total_rank"]), device="cuda", dtype=torch.float32) for k in range(5)}
        centers[layer] = torch.as_tensor(semantic_by_layer[layer], device="cuda", dtype=torch.float32) if semantic_by_layer[layer] else None
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    methods = ["native", "deleted"] + [f"weight_{k}" for k in range(5)]
    metrics = {layer: {name: empty_metric() for name in methods} for layer in config["layers"]}; maximum_error = 0.0
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")
            targets = torch.as_tensor(np.asarray(corpus[start:start + len(raw), length], dtype=np.int64), device="cuda"); reference = full_logits(model, ids)
            for layer in config["layers"]:
                route = native_routes(model, ids, layer); semantic_part = semantic_reconstruction(route, centers[layer], config["semantic_mode_cosine_threshold"]); residual = route - semantic_part
                conditions = {"native": route, "deleted": torch.zeros_like(route)}
                for k in range(5): conditions[f"weight_{k}"] = semantic_part + project(residual, bases[layer][k])
                names, logits = intervene(model, ids, layer, conditions); output = {name: logits[index] for index, name in enumerate(names)}
                maximum_error = max(maximum_error, float((output["native"] - reference).abs().max()))
                for name in methods: update_metric(metrics[layer][name], output["native"], output["deleted"], output[name], targets)
            if start % 32 == 0: print(f"hybrid causal d={destination} rows {start}/{len(corpus)}", flush=True)
    results = {str(layer): {name: finish(value) for name, value in rows.items()} for layer, rows in metrics.items()}
    route_values, causal_values = [], []
    for layer in config["layers"]:
        for k in range(5):
            route_values.append(geometry_rows[layer]["allocation"][str(k)]["replication_fidelity"]); causal_values.append(results[str(layer)][f"weight_{k}"]["causal_fidelity"])
    correlation = pearson(route_values, causal_values)
    context4_mean = float(np.mean([results[str(layer)]["weight_0"]["causal_fidelity"] for layer in config["layers"]]))
    L15_gap = results["15"]["weight_0"]["causal_fidelity"] - results["15"]["weight_4"]["causal_fidelity"]
    L11_gap = results["11"]["weight_0"]["causal_fidelity"] - results["11"]["weight_3"]["causal_fidelity"]
    early_gaps = {layer: results[str(layer)]["weight_0"]["causal_fidelity"] - results[str(layer)]["weight_4"]["causal_fidelity"] for layer in (1, 4)}
    two_context_wins = [layer for layer in config["layers"] if results[str(layer)]["weight_2"]["causal_fidelity"] > results[str(layer)]["weight_4"]["causal_fidelity"]]
    hypotheses = {
        "F0_native_path_is_exact": {"supported": maximum_error <= .02, "maximum_logit_error": maximum_error},
        "F1_context_rank4_is_causally_strong": {"supported": context4_mean >= .80, "mean": context4_mean},
        "F2_route_geometry_predicts_finite_causality": {"supported": correlation >= .70, "correlation": correlation, "cells": len(route_values)},
        "F3_L15_is_a_static_weight_regime": {"supported": L15_gap <= .05, "context_minus_weight": L15_gap},
        "F4_L11_needs_only_one_context_coordinate": {"supported": L11_gap <= .05, "context_minus_hybrid": L11_gap},
        "F5_early_layers_need_context_conditioning": {"supported": all(value >= .10 for value in early_gaps.values()), "gaps": early_gaps},
        "F6_two_context_coordinates_improve_the_weight_prior": {"supported": len(two_context_wins) >= 3, "layers": two_context_wins},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite causality of fixed-rank weight/context route hybrids", "", "At four prospectively selected depth regimes, known semantic modes are retained and the remaining native destination routes are projected into five rank-four allocations. `k=0` is four context/CE-fitted coordinates; `k=4` is the static weight basis. The target-layer MLP and full suffix recompute on fresh FineWeb.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | Full-logit fidelity k=0/1/2/3/4 | CE-effect fidelity k=0/1/2/3/4 |", "|---:|---:|---:|"]
    for layer in config["layers"]:
        logit = "/".join(f"{results[str(layer)][f'weight_{k}']['causal_fidelity']:.3f}" for k in range(5)); ce = "/".join(f"{results[str(layer)][f'weight_{k}']['ce_effect_fidelity']:.3f}" for k in range(5))
        lines.append(f"| {layer} | {logit} | {ce} |")
    lines += ["", "## Guardrail", "", "These context directions are fixed across the new panel; this validates a corpus/loss-conditioned chart, not a per-example dynamic basis. The selected layers represent regimes discovered earlier, so the layerwise claims are prospective for text but not an unbiased estimate over depth."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
