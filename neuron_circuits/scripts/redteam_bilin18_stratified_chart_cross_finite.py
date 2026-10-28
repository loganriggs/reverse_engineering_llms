#!/usr/bin/env python3
"""Finite cross-stratum specificity test on a second fresh FineWeb panel."""

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
from run_bilin18_stratified_natural_route_census import all_strata, token_class_from_flags, weighted_basis
from run_bilin18_stratified_route_finite_causality import project, runtime_mask
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, intervene, native_routes, update_metric


CONFIG = Path("configs/bilin18_stratified_chart_cross_finite_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_stratified_chart_cross_finite_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_STRATIFIED_CHART_CROSS_FINITE_REDTEAM.md"


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"])
    semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    centers = {target["layer"]: [] for target in config["targets"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in centers: centers[row["layer"]].append(row["center"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    token_classes = token_class_from_flags(flags)
    prepared = {}
    for target in config["targets"]:
        layer = target["layer"]
        panel = {key: arrays[f"discovery_{key}_L{layer}"] for key in ("residuals", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss")}
        panel["source_class"] = token_classes[panel["source_id"]]; panel["next_class"] = token_classes[panel["next_id"]]
        masks = all_strata(panel, panel["loss"]); effects = panel["full_effect"] - panel["semantic_effect"]
        prepared[layer] = {
            "target": torch.as_tensor(weighted_basis(panel["residuals"][masks[target["target"]]], effects[masks[target["target"]]], config["rank"]), device="cuda", dtype=torch.float32),
            "alternative": torch.as_tensor(weighted_basis(panel["residuals"][masks[target["alternative"]]], effects[masks[target["alternative"]]], config["rank"]), device="cuda", dtype=torch.float32),
            "semantic": torch.as_tensor(centers[layer], device="cuda", dtype=torch.float32) if centers[layer] else None,
            "loss": panel["loss"],
        }
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    metrics = {target["layer"]: {name: empty_metric() for name in ("native", "deleted", "matching", "alternative")} for target in config["targets"]}
    max_error = 0.0
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
                semantic_part, _ = semantic_reconstruction(routes, basis["semantic"], config["semantic_mode_cosine_threshold"]); residual = routes - semantic_part
                conditions = {
                    "native": routes,
                    "deleted": torch.where(selected[..., None], semantic_part, routes),
                    "matching": torch.where(selected[..., None], semantic_part + project(residual, basis["target"]), routes),
                    "alternative": torch.where(selected[..., None], semantic_part + project(residual, basis["alternative"]), routes),
                }
                names, logits = intervene(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}
                max_error = max(max_error, float((outputs["native"] - reference).abs().max())); example_mask = torch.as_tensor(selected_examples, device="cuda")
                for name in conditions: update_metric(metrics[layer][name], outputs["native"][example_mask], outputs["deleted"][example_mask], outputs[name][example_mask], targets[example_mask])
        print(f"cross-finite destination={destination}", flush=True)
    results = {}; signs = []
    observational_signs = {"source:function": 1, "distance:mid17-64": -1, "next:punctuation": 1, "loss:hard": 1}
    for target in config["targets"]:
        rows = {name: finish(value) for name, value in metrics[target["layer"]].items()}
        logit_margin = rows["matching"]["causal_fidelity"] - rows["alternative"]["causal_fidelity"]
        ce_margin = rows["matching"]["ce_effect_fidelity"] - rows["alternative"]["ce_effect_fidelity"]
        results[target["target"]] = {"layer": target["layer"], "alternative": target["alternative"], "metrics": rows, "logit_margin": logit_margin, "ce_margin": ce_margin}
        signs.append(np.sign(logit_margin) == observational_signs[target["target"]])
    specific = [results[name]["logit_margin"] >= .05 for name in ("next:punctuation", "loss:hard")]
    shared = [results[name]["logit_margin"] < .05 for name in ("source:function", "distance:mid17-64")]
    hypotheses = {
        "Y0_native_replay_is_exact": {"supported": max_error <= .02, "maximum_logit_error": max_error},
        "Y1_observationally_specific_charts_are_finitely_specific": {"supported": all(specific), "margins": {name: results[name]["logit_margin"] for name in ("next:punctuation", "loss:hard")}},
        "Y2_shared_family_charts_are_not_falsely_discrete": {"supported": all(shared), "margins": {name: results[name]["logit_margin"] for name in ("source:function", "distance:mid17-64")}},
        "Y3_specificity_signs_transfer": {"supported": all(signs), "agreements": int(sum(signs)), "targets": len(signs)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite cross-stratum specificity red team", "", "Matching and strongest nonmatching discovery charts are compared on a second fresh FineWeb panel while all non-target route components remain native.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | Examples | Matching/alternative logit fidelity | Margin | Matching/alternative CE fidelity | Margin |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        match = row["metrics"]["matching"]; alt = row["metrics"]["alternative"]
        lines.append(f"| L{row['layer']} {name} vs {row['alternative']} | {match['examples']} | {match['causal_fidelity']:.3f}/{alt['causal_fidelity']:.3f} | {row['logit_margin']:+.3f} | {match['ce_effect_fidelity']:.3f}/{alt['ce_effect_fidelity']:.3f} | {row['ce_margin']:+.3f} |")
    lines += ["", "## Guardrail", "", "A matching-chart advantage establishes conditional coordinate specificity, not a human semantic interpretation. Conversely, transfer between adjacent distance charts is evidence for a continuous family, not evidence that distance is irrelevant."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
