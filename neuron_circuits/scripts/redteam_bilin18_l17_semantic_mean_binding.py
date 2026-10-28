#!/usr/bin/env python3
"""Separate the shared L17 mean write from correctly bound semantic variation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_known_ontology_residual_atlas import collect_panel, design_matrices, multivariate_r2
from run_bilin18_known_ontology_residual_finite import fit_fixed_ridge, make_lookup, ridge_predict
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_taskblind_mlp_subspace_finite import intervene_write, target_product_and_write
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, update_metric


CONFIG = Path("configs/bilin18_l17_semantic_mean_binding_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_l17_semantic_mean_binding_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_L17_SEMANTIC_MEAN_BINDING_REDTEAM.md"


def rank_order(values):
    return sorted(values, key=values.get)


def main():
    config = json.loads(CONFIG.read_text()); known = json.loads(Path(config["known_source"]).read_text()); rng = np.random.default_rng(config["seed"]); layer = config["layer"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); corpus = np.load(config["corpus"], mmap_mode="r")
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    panels = {}
    for name, offset, rows, positions in (("discovery", config["discovery_start"], config["discovery_rows"], config["discovery_positions"]), ("validation", config["validation_start"], config["validation_rows"], config["validation_positions"])):
        values, metadata = collect_panel(model, corpus[offset:offset + rows], offset, positions, [layer], config["collection_batch_size"]); designs = design_matrices(metadata, flags, config["discovery_positions"]); panels[name] = {"values": values[layer], "metadata": metadata, "designs": designs}
    discovery = panels["discovery"]["values"]; validation = panels["validation"]["values"]; mean_write = discovery.mean(0); predictions = {"mean": np.broadcast_to(mean_write, validation.shape).copy()}
    for name in ("current50", "next50", "declared"):
        alpha = known["results"][str(layer)]["designs"][name]["alpha"]; fit = fit_fixed_ridge(panels["discovery"]["designs"][name], discovery, alpha); predictions[{"declared": "semantic"}.get(name, name)] = ridge_predict(fit, panels["validation"]["designs"][name])
    permutations = [rng.permutation(len(validation)) for _ in range(config["permutation_replicates"])]
    for index, permutation in enumerate(permutations): predictions[f"permuted_{index}"] = predictions["semantic"][permutation]
    observational = {name: multivariate_r2(validation, value, mean_write) for name, value in predictions.items()}; tensors = {name: torch.as_tensor(value, device="cuda", dtype=torch.float32) for name, value in predictions.items()}; lookup = make_lookup(panels["validation"]["metadata"])
    methods = ["native", "deleted"] + list(predictions); metrics = {name: empty_metric() for name in methods}; maximum_error = 0.0; validation_corpus = corpus[config["validation_start"]:config["validation_start"] + config["validation_rows"]]
    for position in config["validation_positions"]:
        length = position + 1
        for local_row in range(len(validation_corpus)):
            raw = np.asarray(validation_corpus[local_row:local_row + 1, :length + 1], dtype=np.int64); ids = torch.as_tensor(raw[:, :length], device="cuda"); target = torch.as_tensor(raw[:, length], device="cuda"); reference = full_logits(model, ids); _, native = target_product_and_write(model, ids, layer); index = lookup[(config["validation_start"] + local_row, position)]; conditions = {"native": native, "deleted": torch.zeros_like(native)}
            for name in predictions: conditions[name] = tensors[name][index:index + 1]
            names, logits = intervene_write(model, ids, layer, conditions); outputs = {name: logits[i] for i, name in enumerate(names)}; maximum_error = max(maximum_error, float((outputs["native"] - reference).abs().max()))
            for name in methods: update_metric(metrics[name], outputs["native"], outputs["deleted"], outputs[name], target)
        print(f"L17 semantic binding position={position}", flush=True)
    results = {name: finish(value) for name, value in metrics.items()}; permuted_logit = [results[f"permuted_{index}"]["causal_fidelity"] for index in range(config["permutation_replicates"])]; permuted_ce = [results[f"permuted_{index}"]["ce_effect_fidelity"] for index in range(config["permutation_replicates"])]; logit_p95 = float(np.quantile(permuted_logit, .95)); ce_p95 = float(np.quantile(permuted_ce, .95)); semantic = results["semantic"]; mean = results["mean"]; relative = (semantic["causal_fidelity"] - mean["causal_fidelity"]) / max(1 - mean["causal_fidelity"], 1e-30)
    named = ("mean", "current50", "next50", "semantic"); observational_order = rank_order({name: observational[name] for name in named}); finite_order = rank_order({name: results[name]["causal_fidelity"] for name in named})
    hypotheses = {
        "B0_native_replay_is_exact": {"supported": maximum_error <= .02, "maximum_logit_error": maximum_error},
        "B1_semantics_beats_mean_write": {"supported": semantic["causal_fidelity"] >= mean["causal_fidelity"] + .05 and semantic["ce_effect_fidelity"] >= mean["ce_effect_fidelity"] + .05, "logit_advantage": semantic["causal_fidelity"] - mean["causal_fidelity"], "ce_advantage": semantic["ce_effect_fidelity"] - mean["ce_effect_fidelity"]},
        "B2_semantics_beats_permuted_binding": {"supported": semantic["causal_fidelity"] >= logit_p95 + .05 and semantic["ce_effect_fidelity"] >= ce_p95 + .05, "semantic_logit": semantic["causal_fidelity"], "permuted_logit_p95": logit_p95, "semantic_ce": semantic["ce_effect_fidelity"], "permuted_ce_p95": ce_p95},
        "B3_joint_semantics_beats_single_role_labels": {"supported": semantic["causal_fidelity"] >= max(results["current50"]["causal_fidelity"], results["next50"]["causal_fidelity"]) + .05, "advantage": semantic["causal_fidelity"] - max(results["current50"]["causal_fidelity"], results["next50"]["causal_fidelity"])},
        "B4_correct_binding_closes_material_remaining_gap": {"supported": relative >= .25, "fraction": relative},
        "B5_write_r2_and_finite_rank_controls_agree": {"supported": observational_order == finite_order, "observational_order": observational_order, "finite_order": finite_order},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "observational_write_r2": observational, "results": results}, indent=2, allow_nan=False))
    lines = ["# Layer-17 semantic mean/binding red team", "", "The late semantic write is compared with its discovery mean, current-feature-only and next-feature-only predictions, and 32 row-permuted semantic writes with the same empirical distribution but wrong context binding. The complete final normalization and unembedding are recomputed.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Method | Write R2 | Full-logit fidelity | CE-effect fidelity |", "|---|---:|---:|---:|"]
    for name in named: lines.append(f"| {name} | {observational[name]:.3f} | {results[name]['causal_fidelity']:.3f} | {results[name]['ce_effect_fidelity']:.3f} |")
    lines += [f"| permuted p95 | — | {logit_p95:.3f} | {ce_p95:.3f} |", "", "## Guardrail", "", "The semantic predictor uses the true next-token feature labels and remains a hindsight explanation. Beating the mean and row permutation establishes context-bound information beyond a shared output prior; it does not show that the model internally computes these exact fifty labels or that the semantic basis is unique."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
