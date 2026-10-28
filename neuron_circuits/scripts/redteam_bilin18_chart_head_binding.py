#!/usr/bin/env python3
"""Finite coefficient-matched head-binding red team for natural charts."""

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


CONFIG = Path("configs/bilin18_chart_head_binding_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_chart_head_binding_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_CHART_HEAD_BINDING_REDTEAM.md"


def unique_permutations(dimension, count, rng):
    identity = tuple(range(dimension)); values = set()
    while len(values) < count:
        candidate = tuple(rng.permutation(dimension).tolist())
        if candidate != identity: values.add(candidate)
    return [np.asarray(value, dtype=int) for value in sorted(values)]


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"]); rng = np.random.default_rng(config["seed"])
    semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    centers = {target["layer"]: [] for target in config["targets"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in centers: centers[row["layer"]].append(row["center"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    token_classes = token_class_from_flags(flags); permutations = unique_permutations(9, config["null_replicates"], rng); prepared = {}
    for target in config["targets"]:
        layer = target["layer"]
        panel = {key: arrays[f"discovery_{key}_L{layer}"] for key in ("residuals", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss")}
        panel["source_class"] = token_classes[panel["source_id"]]; panel["next_class"] = token_classes[panel["next_id"]]
        mask = all_strata(panel, panel["loss"])[target["target"]]; effects = panel["full_effect"] - panel["semantic_effect"]
        basis = weighted_basis(panel["residuals"][mask], effects[mask], config["rank"])
        prepared[layer] = {"basis": torch.as_tensor(basis, device="cuda", dtype=torch.float32), "permuted": [torch.as_tensor(basis[:, permutation], device="cuda", dtype=torch.float32) for permutation in permutations], "semantic": torch.as_tensor(centers[layer], device="cuda", dtype=torch.float32) if centers[layer] else None, "loss": panel["loss"]}
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    null_names = [f"joint_{i}" for i in range(config["null_replicates"])] + [f"read_{i}" for i in range(config["null_replicates"])] + [f"write_{i}" for i in range(config["null_replicates"])]
    methods = ["native", "deleted", "matching"] + null_names
    metrics = {target["layer"]: {name: empty_metric() for name in methods} for target in config["targets"]}; max_error = 0.0
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length + 1], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :length], device="cuda"); targets = torch.as_tensor(raw[:, length], device="cuda")
            reference = full_logits(model, ids); losses = F.cross_entropy(reference, targets, reduction="none").cpu().numpy()
            for target in config["targets"]:
                layer = target["layer"]; item = prepared[layer]
                selected_np = runtime_mask(target["target"], raw, destination, losses, token_classes, item["loss"]); selected_examples = selected_np.any(1)
                if not selected_examples.any(): continue
                selected = torch.as_tensor(selected_np, device="cuda"); routes = native_routes(model, ids, layer)
                semantic_part, _ = semantic_reconstruction(routes, item["semantic"], config["semantic_mode_cosine_threshold"]); residual = routes - semantic_part; basis = item["basis"]
                conditions = {"native": routes, "deleted": torch.where(selected[..., None], semantic_part, routes), "matching": torch.where(selected[..., None], semantic_part + factorized_transform(residual, basis, basis), routes)}
                for index, permuted in enumerate(item["permuted"]):
                    conditions[f"joint_{index}"] = torch.where(selected[..., None], semantic_part + factorized_transform(residual, permuted, permuted), routes)
                    conditions[f"read_{index}"] = torch.where(selected[..., None], semantic_part + factorized_transform(residual, permuted, basis), routes)
                    conditions[f"write_{index}"] = torch.where(selected[..., None], semantic_part + factorized_transform(residual, basis, permuted), routes)
                names, logits = intervene(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}
                max_error = max(max_error, float((outputs["native"] - reference).abs().max())); example_mask = torch.as_tensor(selected_examples, device="cuda")
                for name in methods: update_metric(metrics[layer][name], outputs["native"][example_mask], outputs["deleted"][example_mask], outputs[name][example_mask], targets[example_mask])
        print(f"head-binding destination={destination}", flush=True)
    results = {}; gates = []
    for target in config["targets"]:
        layer = target["layer"]; rows = {name: finish(value) for name, value in metrics[layer].items()}; matching = rows["matching"]
        summary = {"layer": layer, "target": target["target"], "matching": matching, "nulls": {}}
        target_gates = {}
        for family in ("joint", "read", "write"):
            logit = [rows[f"{family}_{index}"]["causal_fidelity"] for index in range(config["null_replicates"])]
            ce = [rows[f"{family}_{index}"]["ce_effect_fidelity"] for index in range(config["null_replicates"])]
            summary["nulls"][family] = {"logit_mean": float(np.mean(logit)), "logit_p95": float(np.quantile(logit, .95)), "ce_mean": float(np.mean(ce)), "ce_p95": float(np.quantile(ce, .95)), "matching_beats_logit_p95": bool(matching["causal_fidelity"] > np.quantile(logit, .95)), "matching_beats_ce_p95": bool(matching["ce_effect_fidelity"] > np.quantile(ce, .95))}
            target_gates[family] = summary["nulls"][family]["matching_beats_logit_p95"] or summary["nulls"][family]["matching_beats_ce_p95"]
        results[target["target"]] = summary; gates.append(target_gates)
    hypotheses = {
        "P0_native_replay_is_exact": {"supported": max_error <= .02, "maximum_logit_error": max_error},
        "P1_native_chart_beats_joint_head_permutation": {"supported": all(row["joint"] and results[target["target"]]["nulls"]["joint"]["matching_beats_logit_p95"] and results[target["target"]]["nulls"]["joint"]["matching_beats_ce_p95"] for row, target in zip(gates, config["targets"])), "by_target": [row["joint"] for row in gates]},
        "P2_native_read_binding_is_specific": {"supported": all(row["read"] for row in gates), "by_target": [row["read"] for row in gates]},
        "P3_native_write_binding_is_specific": {"supported": all(row["write"] for row in gates), "by_target": [row["write"] for row in gates]},
        "P4_tensor_head_binding_survives_geometry_redteam": {"supported": any(all(row.values()) for row in gates), "by_target": [all(row.values()) for row in gates]},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Coefficient-matched head-binding red team", "", "The matching chart's head coefficients are preserved exactly but reassigned to random native head identities. Joint, read-only, and write-only permutations are tested with full suffix recomputation on fresh FineWeb.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | Matching logit/CE | Joint null p95 logit/CE | Read null p95 logit/CE | Write null p95 logit/CE |", "|---|---:|---:|---:|---:|"]
    for name, row in results.items():
        m = row["matching"]; j = row["nulls"]["joint"]; r = row["nulls"]["read"]; w = row["nulls"]["write"]
        lines.append(f"| L{row['layer']} {name} | {m['causal_fidelity']:.3f}/{m['ce_effect_fidelity']:.3f} | {j['logit_p95']:.3f}/{j['ce_p95']:.3f} | {r['logit_p95']:.3f}/{r['ce_p95']:.3f} | {w['logit_p95']:.3f}/{w['ce_p95']:.3f} |")
    lines += ["", "## Interpretation boundary", "", "Column permutation preserves rank, orthonormality, coefficient values, and factor budget while destroying native head identity. A native advantage supports learned binding to the model's head-indexed QK routes and contextual OV writes. It does not isolate static QK weights from the input-conditioned route values or static OV from the value state."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
