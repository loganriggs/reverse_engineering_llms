#!/usr/bin/env python3
"""Oracle O(2) pairing red team for route-read/head-write hybrids."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_ce_adjoint_residual_atlas import semantic_reconstruction
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_stratified_chart_tensor_factorial import factorized_transform
from run_bilin18_stratified_natural_route_census import all_strata, token_class_from_flags, weighted_basis
from run_bilin18_stratified_route_finite_causality import runtime_mask
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, intervene, native_routes, update_metric


CONFIG = Path("configs/bilin18_stratified_chart_factor_symmetry_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_stratified_chart_factor_symmetry_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_STRATIFIED_CHART_FACTOR_SYMMETRY_REDTEAM.md"


def orthogonal_grid(count):
    rows = []
    for reflection in (1, -1):
        for index in range(count):
            angle = 2 * np.pi * index / count; c, s = np.cos(angle), np.sin(angle)
            rows.append((f"r{reflection}_{index}", np.asarray([[c, -s], [s, c]]) @ np.diag([1, reflection])))
    return rows


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"])
    semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    centers = {target["layer"]: [] for target in config["targets"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in centers: centers[row["layer"]].append(row["center"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    token_classes = token_class_from_flags(flags); grid = orthogonal_grid(config["angles"]); prepared = {}
    for target in config["targets"]:
        layer = target["layer"]
        panel = {key: arrays[f"discovery_{key}_L{layer}"] for key in ("residuals", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss")}
        panel["source_class"] = token_classes[panel["source_id"]]; panel["next_class"] = token_classes[panel["next_id"]]
        masks = all_strata(panel, panel["loss"]); effects = panel["full_effect"] - panel["semantic_effect"]
        matching = weighted_basis(panel["residuals"][masks[target["target"]]], effects[masks[target["target"]]], config["rank"])
        alternative = weighted_basis(panel["residuals"][masks[target["alternative"]]], effects[masks[target["alternative"]]], config["rank"])
        prepared[layer] = {"matching": torch.as_tensor(matching, device="cuda", dtype=torch.float32), "alternatives": [(name, torch.as_tensor(rotation @ alternative, device="cuda", dtype=torch.float32)) for name, rotation in grid], "semantic": torch.as_tensor(centers[layer], device="cuda", dtype=torch.float32) if centers[layer] else None, "loss": panel["loss"]}
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    hybrid_names = [f"MA_{name}" for name, _ in grid] + [f"AM_{name}" for name, _ in grid]
    methods = ["native", "deleted", "MM", "AA"] + hybrid_names
    metrics = {target["layer"]: {name: empty_metric() for name in methods} for target in config["targets"]}; max_error = 0.0
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length + 1], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :length], device="cuda"); targets = torch.as_tensor(raw[:, length], device="cuda")
            reference = full_logits(model, ids); losses = F.cross_entropy(reference, targets, reduction="none").cpu().numpy()
            for target in config["targets"]:
                layer = target["layer"]; basis = prepared[layer]
                selected_np = runtime_mask(target["target"], raw, destination, losses, token_classes, basis["loss"]); selected_examples = selected_np.any(1)
                if not selected_examples.any(): continue
                selected = torch.as_tensor(selected_np, device="cuda"); routes = native_routes(model, ids, layer)
                semantic_part, _ = semantic_reconstruction(routes, basis["semantic"], config["semantic_mode_cosine_threshold"]); residual = routes - semantic_part; m = basis["matching"]
                first_alt = basis["alternatives"][0][1]
                conditions = {"native": routes, "deleted": torch.where(selected[..., None], semantic_part, routes), "MM": torch.where(selected[..., None], semantic_part + factorized_transform(residual, m, m), routes), "AA": torch.where(selected[..., None], semantic_part + factorized_transform(residual, first_alt, first_alt), routes)}
                for name, alt in basis["alternatives"]:
                    conditions[f"MA_{name}"] = torch.where(selected[..., None], semantic_part + factorized_transform(residual, m, alt), routes)
                    conditions[f"AM_{name}"] = torch.where(selected[..., None], semantic_part + factorized_transform(residual, alt, m), routes)
                names, logits = intervene(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}
                max_error = max(max_error, float((outputs["native"] - reference).abs().max())); example_mask = torch.as_tensor(selected_examples, device="cuda")
                for name in methods: update_metric(metrics[layer][name], outputs["native"][example_mask], outputs["deleted"][example_mask], outputs[name][example_mask], targets[example_mask])
        print(f"factor-symmetry destination={destination}", flush=True)
    results = {}; variation = []; both_gap = []; close_both = []
    for target in config["targets"]:
        layer = target["layer"]; rows = {name: finish(value) for name, value in metrics[layer].items()}; hybrids = [rows[name] | {"name": name} for name in hybrid_names]
        logit_values = [row["causal_fidelity"] for row in hybrids]; ce_values = [row["ce_effect_fidelity"] for row in hybrids]
        best_logit = max(hybrids, key=lambda row: row["causal_fidelity"]); best_ce = max(hybrids, key=lambda row: row["ce_effect_fidelity"])
        joint_close = [row for row in hybrids if rows["MM"]["causal_fidelity"] - row["causal_fidelity"] <= .02 and rows["MM"]["ce_effect_fidelity"] - row["ce_effect_fidelity"] <= .02]
        results[target["target"]] = {"layer": layer, "metrics": {name: rows[name] for name in ("MM", "AA")}, "best_logit_hybrid": best_logit, "best_ce_hybrid": best_ce, "logit_hybrid_range": [float(min(logit_values)), float(max(logit_values))], "ce_hybrid_range": [float(min(ce_values)), float(max(ce_values))], "oracle_logit_gap": rows["MM"]["causal_fidelity"] - best_logit["causal_fidelity"], "oracle_ce_gap": rows["MM"]["ce_effect_fidelity"] - best_ce["ce_effect_fidelity"], "jointly_close_hybrids": [row["name"] for row in joint_close]}
        variation.append(max(logit_values) - min(logit_values) >= .05 or max(ce_values) - min(ce_values) >= .05)
        both_gap.append(results[target["target"]]["oracle_logit_gap"] >= .05 and results[target["target"]]["oracle_ce_gap"] >= .05)
        close_both.append(bool(joint_close))
    hypotheses = {
        "R0_native_replay_is_exact": {"supported": max_error <= .02, "maximum_logit_error": max_error},
        "R1_procrustes_conclusion_is_not_assumed_invariant": {"supported": any(variation), "by_target": variation},
        "R2_full_matching_chart_beats_oracle_hybrids": {"supported": any(both_gap), "by_target": both_gap},
        "R3_no_single_factor_explanation_is_universal": {"supported": not all(close_both), "jointly_close_by_target": close_both},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Orthogonal-pairing red team for chart tensor factors", "", "Every rotation and reflection in a 32-point O(2) grid is allowed to pair the matching and alternative rank-two axes. Best hybrid performance is selected on evaluation itself, making this an intentionally generous oracle test of the one-matching-factor explanation.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | MM logit/CE | AA logit/CE | Hybrid logit range / best gap | Hybrid CE range / best gap | Jointly close hybrids |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        mm = row["metrics"]["MM"]; aa = row["metrics"]["AA"]
        lines.append(f"| L{row['layer']} {name} | {mm['causal_fidelity']:.3f}/{mm['ce_effect_fidelity']:.3f} | {aa['causal_fidelity']:.3f}/{aa['ce_effect_fidelity']:.3f} | {row['logit_hybrid_range'][0]:.3f}..{row['logit_hybrid_range'][1]:.3f} / {row['oracle_logit_gap']:+.3f} | {row['ce_hybrid_range'][0]:.3f}..{row['ce_hybrid_range'][1]:.3f} / {row['oracle_ce_gap']:+.3f} | {len(row['jointly_close_hybrids'])} |")
    lines += ["", "## Guardrail", "", "The grid is an evaluation-selected oracle and therefore cannot validate a deployable axis pairing. It can only falsify claims that interaction is unavoidable under every internal chart symmetry. A surviving gap is strong evidence for paired read/write structure; a rescued hybrid means the previous Procrustes interaction was coordinate-dependent."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
