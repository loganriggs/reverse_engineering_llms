#!/usr/bin/env python3
"""Finite causal test of CE-discovered route directions on fresh FineWeb."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from redteam_bilin18_ce_adjoint_residual_novelty import normalized_complement
from run_bilin18_ce_adjoint_residual_atlas import fit_basis
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, intervene, native_routes, update_metric


CONFIG = Path("configs/bilin18_ce_residual_incremental_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_ce_residual_incremental_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_CE_RESIDUAL_INCREMENTAL_CAUSALITY.md"


def project(routes, basis):
    return (routes @ basis.T) @ basis


def semantic_reconstruction(routes, centers, threshold):
    if centers is None:
        return torch.zeros_like(routes)
    norm = routes.norm(dim=-1, keepdim=True); unit = routes / norm.clamp_min(1e-30)
    similarity = unit @ centers.T; assignment = similarity.argmax(-1)
    selected = similarity.max(-1).values >= threshold
    return torch.where(selected[..., None], norm * centers[assignment], 0)


def random_complement(generic_basis, rank, rng):
    dimension = generic_basis.shape[1]
    values = rng.standard_normal((rank, dimension))
    values = values - (values @ generic_basis.T) @ generic_basis
    q, _ = np.linalg.qr(values.T)
    return q[:, :rank].T


def main():
    config = json.loads(CONFIG.read_text()); semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    residual_arrays = np.load(config["residual_arrays"]); generic_arrays = np.load(config["generic_arrays"])
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    semantic_by_layer = {layer: [] for layer in config["layers"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in semantic_by_layer: semantic_by_layer[row["layer"]].append(row["center"])
    rng = np.random.default_rng(config["seed"])
    bases = {}
    for layer in config["layers"]:
        generic = fit_basis(generic_arrays[f"discovery_routes_L{layer}"], config["generic_rank"])
        discovery_residual = residual_arrays[f"discovery_residual_routes_L{layer}"]
        novel = fit_basis(normalized_complement(discovery_residual, generic), config["novel_rank"])
        ce = fit_basis(discovery_residual, config["ce_rank"])
        randoms = [random_complement(generic, config["novel_rank"], rng) for _ in range(config["null_replicates"])]
        bases[layer] = {
            "generic": torch.as_tensor(generic, device="cuda", dtype=torch.float32),
            "novel": torch.as_tensor(novel, device="cuda", dtype=torch.float32),
            "ce": torch.as_tensor(ce, device="cuda", dtype=torch.float32),
            "random": [torch.as_tensor(value, device="cuda", dtype=torch.float32) for value in randoms],
            "semantic": torch.as_tensor(semantic_by_layer[layer], device="cuda", dtype=torch.float32) if semantic_by_layer[layer] else None,
        }
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    methods = ["native", "deleted", "semantic_generic4", "semantic_generic4_novel2", "semantic_ce4"] + [f"random_{index}" for index in range(config["null_replicates"])]
    metrics = {layer: {method: empty_metric() for method in methods} for layer in config["layers"]}; maximum_error = 0.0
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda"); targets = torch.as_tensor(np.asarray(corpus[start:start + len(raw), length], dtype=np.int64), device="cuda")
            reference = full_logits(model, ids)
            for layer in config["layers"]:
                route = native_routes(model, ids, layer); basis = bases[layer]
                semantic_part = semantic_reconstruction(route, basis["semantic"], config["semantic_mode_cosine_threshold"])
                residual = route - semantic_part; generic_part = project(residual, basis["generic"]); complement = residual - generic_part
                conditions = {
                    "native": route, "deleted": torch.zeros_like(route),
                    "semantic_generic4": semantic_part + generic_part,
                    "semantic_generic4_novel2": semantic_part + generic_part + project(complement, basis["novel"]),
                    "semantic_ce4": semantic_part + project(residual, basis["ce"]),
                }
                for index, random_basis in enumerate(basis["random"]):
                    conditions[f"random_{index}"] = semantic_part + generic_part + project(complement, random_basis)
                names, logits = intervene(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}
                maximum_error = max(maximum_error, float((outputs["native"] - reference).abs().max()))
                for name in methods: update_metric(metrics[layer][name], outputs["native"], outputs["deleted"], outputs[name], targets)
            if start % 32 == 0: print(f"incremental d={destination} rows {start}/{len(corpus)}", flush=True)
    results = {str(layer): {name: finish(value) for name, value in rows.items()} for layer, rows in metrics.items()}
    logit_improvement = {layer: results[str(layer)]["semantic_generic4_novel2"]["causal_fidelity"] - results[str(layer)]["semantic_generic4"]["causal_fidelity"] for layer in config["layers"]}
    ce_improvement = {layer: results[str(layer)]["semantic_generic4_novel2"]["ce_effect_fidelity"] - results[str(layer)]["semantic_generic4"]["ce_effect_fidelity"] for layer in config["layers"]}
    null_means = [float(np.mean([results[str(layer)][f"random_{index}"]["causal_fidelity"] - results[str(layer)]["semantic_generic4"]["causal_fidelity"] for layer in config["layers"]])) for index in range(config["null_replicates"])]
    novel_mean = float(np.mean(list(logit_improvement.values()))); ce_mean = float(np.mean(list(ce_improvement.values())))
    ce4_mean = float(np.mean([results[str(layer)]["semantic_ce4"]["causal_fidelity"] for layer in config["layers"]]))
    hypotheses = {
        "I0_native_path_is_exact": {"supported": maximum_error <= .02, "maximum_logit_error": maximum_error},
        "I1_novel_axes_add_finite_logit_coverage": {"supported": sum(value > 0 for value in logit_improvement.values()) >= 3 and novel_mean >= .02, "by_layer": logit_improvement, "mean": novel_mean},
        "I2_novel_axes_add_prediction_relevant_coverage": {"supported": sum(value > 0 for value in ce_improvement.values()) >= 3 and ce_mean >= .02, "by_layer": ce_improvement, "mean": ce_mean},
        "I3_novel_axes_beat_equal_rank_random_complements": {"supported": novel_mean > max(null_means), "observed": novel_mean, "null_means": null_means},
        "I4_ce_selected_rank4_is_a_strong_compact_reference": {"supported": ce4_mean >= .80, "mean": ce4_mean, "by_layer": {str(layer): results[str(layer)]["semantic_ce4"]["causal_fidelity"] for layer in config["layers"]}},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Incremental finite causality of novel CE-residual route axes", "", "On fresh FineWeb, the known semantic modes plus a generic rank-four route projection form the baseline. The newly discovered rank-two generic-complement axes, equal-rank random complement controls, and a CE-selected residual rank-four reference are added before recomputing the target-layer MLP and complete suffix.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | Base logit/CE-effect fidelity | +novel2 logit/CE-effect | Improvement logit/CE | CE-rank4 logit/CE | Best random logit improvement |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; base = row["semantic_generic4"]; novel = row["semantic_generic4_novel2"]; ce = row["semantic_ce4"]
        best_random = max(row[f"random_{index}"]["causal_fidelity"] - base["causal_fidelity"] for index in range(config["null_replicates"]))
        lines.append(f"| {layer} | {base['causal_fidelity']:.3f}/{base['ce_effect_fidelity']:.3f} | {novel['causal_fidelity']:.3f}/{novel['ce_effect_fidelity']:.3f} | {logit_improvement[layer]:+.3f}/{ce_improvement[layer]:+.3f} | {ce['causal_fidelity']:.3f}/{ce['ce_effect_fidelity']:.3f} | {best_random:+.3f} |")
    lines += ["", "## Guardrail", "", "The new axes were chosen after an earlier novelty screen, while this text panel is fresh. Improvement shows incremental causal utility for this frozen coordinate code; it does not turn each PCA pole into a discrete semantic circuit. Random controls are orthogonal to the generic basis but are not optimized on discovery CE, so the comparison tests selection value rather than parameter-count-only reconstruction."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
