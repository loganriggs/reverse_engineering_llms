#!/usr/bin/env python3
"""Finite matched-random test for static weight and CE-conditioned route bases."""

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
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, intervene, native_routes, update_metric
from run_bilin18_weight_only_ce_route_subspace import random_basis
from run_bilin18_weight_context_route_hybrid import project
from run_bilin18_weight_context_hybrid_causality import semantic_reconstruction


CONFIG = Path("configs/bilin18_weight_context_finite_null.json")
OUTPUT = TENSOR_ROOT / "bilin18_weight_context_finite_null.json"
REPORT = TENSOR_ROOT / "BILIN18_WEIGHT_CONTEXT_FINITE_NULL.md"


def main():
    config = json.loads(CONFIG.read_text()); weight_source = json.loads(Path(config["weight_source"]).read_text())
    weights = np.load(config["weight_bases"]); residual_arrays = np.load(config["residual_arrays"])
    semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    weight_rows = {row["layer"]: row for row in weight_source["layers"]}
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    semantic_by_layer = {layer: [] for layer in config["layers"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in semantic_by_layer: semantic_by_layer[row["layer"]].append(row["center"])
    rng = np.random.default_rng(config["seed"]); bases = {}; centers = {}
    for layer in config["layers"]:
        discovery = residual_arrays[f"discovery_residual_routes_L{layer}"]; method = weight_rows[layer]["best_method"]
        bases[layer] = {
            "weight": torch.as_tensor(weights[f"{method}_basis_L{layer}"], device="cuda", dtype=torch.float32),
            "context": torch.as_tensor(fit_basis(discovery, config["rank"]), device="cuda", dtype=torch.float32),
            "random": [torch.as_tensor(random_basis(9, config["rank"], rng), device="cuda", dtype=torch.float32) for _ in range(config["null_replicates"])],
        }
        centers[layer] = torch.as_tensor(semantic_by_layer[layer], device="cuda", dtype=torch.float32) if semantic_by_layer[layer] else None
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    methods = ["native", "deleted", "weight", "context"] + [f"random_{index}" for index in range(config["null_replicates"])]
    metrics = {layer: {name: empty_metric() for name in methods} for layer in config["layers"]}; maximum_error = 0.0
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")
            targets = torch.as_tensor(np.asarray(corpus[start:start + len(raw), length], dtype=np.int64), device="cuda"); reference = full_logits(model, ids)
            for layer in config["layers"]:
                route = native_routes(model, ids, layer); semantic_part = semantic_reconstruction(route, centers[layer], config["semantic_mode_cosine_threshold"]); residual = route - semantic_part
                conditions = {"native": route, "deleted": torch.zeros_like(route), "weight": semantic_part + project(residual, bases[layer]["weight"]), "context": semantic_part + project(residual, bases[layer]["context"])}
                for index, basis in enumerate(bases[layer]["random"]): conditions[f"random_{index}"] = semantic_part + project(residual, basis)
                names, logits = intervene(model, ids, layer, conditions); output = {name: logits[index] for index, name in enumerate(names)}
                maximum_error = max(maximum_error, float((output["native"] - reference).abs().max()))
                for name in methods: update_metric(metrics[layer][name], output["native"], output["deleted"], output[name], targets)
            if start % 32 == 0: print(f"finite null d={destination} rows {start}/{len(corpus)}", flush=True)
    results = {str(layer): {name: finish(value) for name, value in rows.items()} for layer, rows in metrics.items()}; summary = {}
    for layer in config["layers"]:
        random_logit = np.asarray([results[str(layer)][f"random_{index}"]["causal_fidelity"] for index in range(config["null_replicates"])]); random_ce = np.asarray([results[str(layer)][f"random_{index}"]["ce_effect_fidelity"] for index in range(config["null_replicates"])])
        summary[str(layer)] = {
            "weight_logit": results[str(layer)]["weight"]["causal_fidelity"], "context_logit": results[str(layer)]["context"]["causal_fidelity"],
            "weight_ce": results[str(layer)]["weight"]["ce_effect_fidelity"], "context_ce": results[str(layer)]["context"]["ce_effect_fidelity"],
            "random_logit_mean": float(random_logit.mean()), "random_logit_p95": float(np.quantile(random_logit, .95)),
            "random_ce_mean": float(random_ce.mean()), "random_ce_p95": float(np.quantile(random_ce, .95)),
        }
    weight_logit_pass = [int(layer) for layer, row in summary.items() if row["weight_logit"] > row["random_logit_p95"]]
    context_logit_pass = [int(layer) for layer, row in summary.items() if row["context_logit"] > row["random_logit_p95"]]
    weight_ce_pass = [int(layer) for layer, row in summary.items() if row["weight_ce"] > row["random_ce_p95"]]
    context_ce_pass = [int(layer) for layer, row in summary.items() if row["context_ce"] > row["random_ce_p95"]]
    hypotheses = {
        "R0_native_path_is_exact": {"supported": maximum_error <= .02, "maximum_logit_error": maximum_error},
        "R1_weight_bases_are_finitely_specific": {"supported": len(weight_logit_pass) >= 3, "layers": weight_logit_pass},
        "R2_context_bases_are_finitely_specific": {"supported": len(context_logit_pass) >= 3, "layers": context_logit_pass},
        "R3_L15_static_regime_beats_random": {"supported": 15 in weight_logit_pass},
        "R4_L1_static_regime_beats_random": {"supported": 1 in weight_logit_pass},
        "R5_ce_effect_specificity_agrees_with_logit_specificity": {"supported": len(weight_ce_pass) >= 3 and len(context_ce_pass) >= 3, "weight_layers": weight_ce_pass, "context_layers": context_ce_pass},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite random-basis red team for weight/context route codes", "", "Static weight and CE-conditioned rank-four bases are compared with 32 frozen Haar-random rank-four bases in the same nine-head coefficient space. Known semantic modes are retained equally in every condition; the residual route is projected, and the target-layer MLP plus suffix recompute on a fresh FineWeb panel.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | Weight/context logit | Random logit mean/p95 | Weight/context CE | Random CE mean/p95 |", "|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = summary[str(layer)]; lines.append(f"| {layer} | {row['weight_logit']:.3f}/{row['context_logit']:.3f} | {row['random_logit_mean']:.3f}/{row['random_logit_p95']:.3f} | {row['weight_ce']:.3f}/{row['context_ce']:.3f} | {row['random_ce_mean']:.3f}/{row['random_ce_p95']:.3f} |")
    lines += ["", "## Guardrail", "", "Random rank four is a demanding null because the head space has dimension nine. Passing establishes basis specificity under finite behavior; failing means high absolute fidelity is not enough to prefer that ontology. The four layers were selected from earlier regime results, so this tests those regimes rather than estimating prevalence across all depth."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
