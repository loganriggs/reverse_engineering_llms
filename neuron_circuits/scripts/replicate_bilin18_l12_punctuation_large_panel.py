#!/usr/bin/env python3
"""Large-panel replication of the L12 punctuation chart after a CE outlier."""

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


CONFIG = Path("configs/bilin18_l12_punctuation_large_replication.json")
OUTPUT = TENSOR_ROOT / "bilin18_l12_punctuation_large_replication.json"
REPORT = TENSOR_ROOT / "BILIN18_L12_PUNCTUATION_LARGE_REPLICATION.md"


def fidelity(target, approximation):
    return float(1 - np.square(target - approximation).sum() / max(np.square(target).sum(), 1e-30))


def cluster_bootstrap(rows, target, approximation, replicates, rng):
    unique = np.unique(rows); values = []
    grouped = {row: np.where(rows == row)[0] for row in unique}
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([grouped[row] for row in sampled])
        values.append(fidelity(target[indices], approximation[indices]))
    return values


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"]); layer = config["layer"]
    semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    centers = [row["center"] for row in semantic["results"] if row["name"] in passed and row["layer"] == layer]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    token_classes = token_class_from_flags(flags)
    panel = {key: arrays[f"discovery_{key}_L{layer}"] for key in ("residuals", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss")}
    panel["source_class"] = token_classes[panel["source_id"]]; panel["next_class"] = token_classes[panel["next_id"]]
    masks = all_strata(panel, panel["loss"]); effects = panel["full_effect"] - panel["semantic_effect"]
    matching = torch.as_tensor(weighted_basis(panel["residuals"][masks[config["target"]]], effects[masks[config["target"]]], config["rank"]), device="cuda", dtype=torch.float32)
    alternative = torch.as_tensor(weighted_basis(panel["residuals"][masks[config["alternative"]]], effects[masks[config["alternative"]]], config["rank"]), device="cuda", dtype=torch.float32)
    semantic_basis = torch.as_tensor(centers, device="cuda", dtype=torch.float32) if centers else None
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    methods = ("native", "deleted", "matching", "alternative"); metrics = {name: empty_metric() for name in methods}; max_error = 0.0
    per_example = {key: [] for key in ("row", "destination", "ce_target", "ce_matching", "ce_alternative")}
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length + 1], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :length], device="cuda"); targets = torch.as_tensor(raw[:, length], device="cuda")
            reference = full_logits(model, ids); losses = F.cross_entropy(reference, targets, reduction="none").cpu().numpy()
            selected_np = runtime_mask(config["target"], raw, destination, losses, token_classes, panel["loss"]); selected_examples = selected_np.any(1)
            if not selected_examples.any(): continue
            selected = torch.as_tensor(selected_np, device="cuda"); routes = native_routes(model, ids, layer)
            semantic_part, _ = semantic_reconstruction(routes, semantic_basis, config["semantic_mode_cosine_threshold"]); residual = routes - semantic_part
            conditions = {"native": routes, "deleted": torch.where(selected[..., None], semantic_part, routes), "matching": torch.where(selected[..., None], semantic_part + factorized_transform(residual, matching, matching), routes), "alternative": torch.where(selected[..., None], semantic_part + factorized_transform(residual, alternative, alternative), routes)}
            names, logits = intervene(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}
            max_error = max(max_error, float((outputs["native"] - reference).abs().max())); example_mask = torch.as_tensor(selected_examples, device="cuda")
            for name in methods: update_metric(metrics[name], outputs["native"][example_mask], outputs["deleted"][example_mask], outputs[name][example_mask], targets[example_mask])
            native_loss = F.cross_entropy(outputs["native"][example_mask], targets[example_mask], reduction="none").cpu().numpy(); deleted_loss = F.cross_entropy(outputs["deleted"][example_mask], targets[example_mask], reduction="none").cpu().numpy()
            matching_loss = F.cross_entropy(outputs["matching"][example_mask], targets[example_mask], reduction="none").cpu().numpy(); alternative_loss = F.cross_entropy(outputs["alternative"][example_mask], targets[example_mask], reduction="none").cpu().numpy()
            selected_indices = np.where(selected_examples)[0]
            per_example["row"].extend((config["start_row"] + start + selected_indices).tolist()); per_example["destination"].extend([destination] * len(selected_indices))
            per_example["ce_target"].extend((deleted_loss - native_loss).tolist()); per_example["ce_matching"].extend((deleted_loss - matching_loss).tolist()); per_example["ce_alternative"].extend((deleted_loss - alternative_loss).tolist())
        print(f"large-punctuation destination={destination}", flush=True)
    finished = {name: finish(value) for name, value in metrics.items()}; data = {key: np.asarray(value) for key, value in per_example.items()}; rng = np.random.default_rng(config["seed"])
    bootstrap = cluster_bootstrap(data["row"], data["ce_target"], data["ce_matching"], config["bootstrap_replicates"], rng)
    logit_advantage = finished["matching"]["causal_fidelity"] - finished["alternative"]["causal_fidelity"]; ce_advantage = finished["matching"]["ce_effect_fidelity"] - finished["alternative"]["ce_effect_fidelity"]
    hypotheses = {
        "V0_native_replay_is_exact": {"supported": max_error <= .02, "maximum_logit_error": max_error},
        "V1_matching_chart_has_stable_logit_coverage": {"supported": finished["matching"]["causal_fidelity"] >= .40, "fidelity": finished["matching"]["causal_fidelity"]},
        "V2_matching_chart_has_positive_ce_coverage": {"supported": finished["matching"]["ce_effect_fidelity"] >= .40, "fidelity": finished["matching"]["ce_effect_fidelity"]},
        "V3_ce_coverage_survives_row_cluster_uncertainty": {"supported": float(np.quantile(bootstrap, .05)) > 0, "bootstrap_mean": float(np.mean(bootstrap)), "p05": float(np.quantile(bootstrap, .05)), "p95": float(np.quantile(bootstrap, .95)), "rows": int(len(np.unique(data["row"]))), "examples": int(len(data["row"]))},
        "V4_punctuation_chart_beats_numeric_chart": {"supported": logit_advantage >= .05 and ce_advantage >= .05, "logit_advantage": logit_advantage, "ce_advantage": ce_advantage},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "metrics": finished, "per_example": {key: value.tolist() for key, value in data.items()}}, indent=2, allow_nan=False))
    lines = ["# Large-panel L12 punctuation replication", "", "After one 38-example panel produced negative signed CE-effect fidelity, the frozen punctuation and numeric charts are compared on 256 new FineWeb rows. Uncertainty resamples FineWeb rows rather than treating prediction positions as independent.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Method | Examples | Full-logit fidelity | CE-effect fidelity | CE cosine | CE norm ratio |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("matching", "alternative"):
        row = finished[name]; lines.append(f"| {name} | {row['examples']} | {row['causal_fidelity']:.3f} | {row['ce_effect_fidelity']:.3f} | {row['ce_effect_cosine']:.3f} | {row['ce_effect_norm_ratio']:.3f} |")
    lines += ["", "## Guardrail", "", "A larger aggregate can establish typical transfer while retaining the earlier negative panel as real heterogeneity. The row bootstrap measures sampling uncertainty over this panel; it does not erase distribution shift or prove that every punctuation context uses the chart."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
