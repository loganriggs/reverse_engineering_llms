#!/usr/bin/env python3
"""Cross route-read and head-write factors of specific natural charts."""

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
from run_bilin18_stratified_route_finite_causality import runtime_mask
from run_bilin18_unseeded_route_end_to_end_causality import empty_metric, finish, full_logits, intervene, native_routes, update_metric


CONFIG = Path("configs/bilin18_stratified_chart_tensor_factorial.json")
OUTPUT = TENSOR_ROOT / "bilin18_stratified_chart_tensor_factorial.json"
REPORT = TENSOR_ROOT / "BILIN18_STRATIFIED_CHART_TENSOR_FACTORIAL.md"


def align_basis(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Resolve the internal O(k) symmetry of an orthonormal chart basis."""
    u, _, vh = np.linalg.svd(reference @ other.T)
    return (u @ vh) @ other


def factorized_transform(routes: torch.Tensor, read: torch.Tensor, write: torch.Tensor) -> torch.Tensor:
    return (routes @ read.T) @ write


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"])
    semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    centers = {target["layer"]: [] for target in config["targets"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in centers: centers[row["layer"]].append(row["center"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    token_classes = token_class_from_flags(flags); prepared = {}
    for target in config["targets"]:
        layer = target["layer"]
        panel = {key: arrays[f"discovery_{key}_L{layer}"] for key in ("residuals", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss")}
        panel["source_class"] = token_classes[panel["source_id"]]; panel["next_class"] = token_classes[panel["next_id"]]
        masks = all_strata(panel, panel["loss"]); effects = panel["full_effect"] - panel["semantic_effect"]
        matching = weighted_basis(panel["residuals"][masks[target["target"]]], effects[masks[target["target"]]], config["rank"])
        alternative_raw = weighted_basis(panel["residuals"][masks[target["alternative"]]], effects[masks[target["alternative"]]], config["rank"])
        alternative = align_basis(matching, alternative_raw)
        prepared[layer] = {
            "matching": torch.as_tensor(matching, device="cuda", dtype=torch.float32),
            "alternative": torch.as_tensor(alternative, device="cuda", dtype=torch.float32),
            "semantic": torch.as_tensor(centers[layer], device="cuda", dtype=torch.float32) if centers[layer] else None,
            "loss": panel["loss"],
            "principal_cosines": np.linalg.svd(matching @ alternative.T, compute_uv=False).tolist(),
        }
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    methods = ("native", "deleted", "MM", "AA", "MA", "AM")
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
                semantic_part, _ = semantic_reconstruction(routes, basis["semantic"], config["semantic_mode_cosine_threshold"]); residual = routes - semantic_part
                m = basis["matching"]; a = basis["alternative"]
                reconstructed = {
                    "MM": semantic_part + factorized_transform(residual, m, m),
                    "AA": semantic_part + factorized_transform(residual, a, a),
                    "MA": semantic_part + factorized_transform(residual, m, a),
                    "AM": semantic_part + factorized_transform(residual, a, m),
                }
                conditions = {"native": routes, "deleted": torch.where(selected[..., None], semantic_part, routes)}
                conditions.update({name: torch.where(selected[..., None], value, routes) for name, value in reconstructed.items()})
                names, logits = intervene(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}
                max_error = max(max_error, float((outputs["native"] - reference).abs().max())); example_mask = torch.as_tensor(selected_examples, device="cuda")
                for name in methods: update_metric(metrics[layer][name], outputs["native"][example_mask], outputs["deleted"][example_mask], outputs[name][example_mask], targets[example_mask])
        print(f"tensor-factorial destination={destination}", flush=True)
    results = {}; distinguish = []; adds = []; irreducible = []; larger_roles = {"logit": [], "ce": []}
    for target in config["targets"]:
        layer = target["layer"]; rows = {name: finish(value) for name, value in metrics[layer].items()}
        summary = {"layer": layer, "target": target["target"], "alternative": target["alternative"], "principal_cosines": prepared[layer]["principal_cosines"], "metrics": rows}
        for metric_name, field in (("logit", "causal_fidelity"), ("ce", "ce_effect_fidelity")):
            read_gain = rows["MA"][field] - rows["AA"][field]
            write_gain = rows["AM"][field] - rows["AA"][field]
            full_gain = rows["MM"][field] - rows["AA"][field]
            interaction = full_gain - read_gain - write_gain
            summary[f"{metric_name}_read_gain"] = read_gain; summary[f"{metric_name}_write_gain"] = write_gain; summary[f"{metric_name}_joint_gain"] = full_gain; summary[f"{metric_name}_interaction"] = interaction
            larger_roles[metric_name].append("read" if read_gain > write_gain else "write")
        results[target["target"]] = summary
        distinguish.append(max(abs(summary["logit_read_gain"] - summary["logit_write_gain"]), abs(summary["ce_read_gain"] - summary["ce_write_gain"])) >= .05)
        adds.append(max(summary["logit_read_gain"], summary["logit_write_gain"], summary["ce_read_gain"], summary["ce_write_gain"]) >= .05)
        irreducible.append(max(rows["MM"]["causal_fidelity"] - max(rows["MA"]["causal_fidelity"], rows["AM"]["causal_fidelity"]), rows["MM"]["ce_effect_fidelity"] - max(rows["MA"]["ce_effect_fidelity"], rows["AM"]["ce_effect_fidelity"])) >= .05)
    role_differs = len(set(larger_roles["logit"])) > 1 or len(set(larger_roles["ce"])) > 1
    hypotheses = {
        "Q0_native_replay_is_exact": {"supported": max_error <= .02, "maximum_logit_error": max_error},
        "Q1_tensor_factors_are_behaviorally_distinguishable": {"supported": any(distinguish), "by_target": distinguish},
        "Q2_matching_factor_adds_to_the_wrong_chart": {"supported": all(adds), "by_target": adds},
        "Q3_full_matching_chart_is_not_reducible_to_one_matching_factor": {"supported": any(irreducible), "by_target": irreducible},
        "Q4_factor_roles_need_not_match_across_regimes": {"supported": role_differs, "larger_roles": larger_roles},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Route-read × head-write tensor factorial", "", "For each specific natural chart, the matching and strongest alternative rank-two bases are Procrustes-aligned and crossed. `MA` reads matching route coordinates but writes them through the alternative head/OV coalition; `AM` does the reverse. Only selected residual source edges are replaced.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | Principal cosines | Logit MM/AA/MA/AM | Read/write/joint/interaction gain | CE MM/AA/MA/AM | Read/write/joint/interaction gain |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        m = row["metrics"]
        lines.append(f"| L{row['layer']} {name} | {[round(value, 3) for value in row['principal_cosines']]} | {m['MM']['causal_fidelity']:.3f}/{m['AA']['causal_fidelity']:.3f}/{m['MA']['causal_fidelity']:.3f}/{m['AM']['causal_fidelity']:.3f} | {row['logit_read_gain']:+.3f}/{row['logit_write_gain']:+.3f}/{row['logit_joint_gain']:+.3f}/{row['logit_interaction']:+.3f} | {m['MM']['ce_effect_fidelity']:.3f}/{m['AA']['ce_effect_fidelity']:.3f}/{m['MA']['ce_effect_fidelity']:.3f}/{m['AM']['ce_effect_fidelity']:.3f} | {row['ce_read_gain']:+.3f}/{row['ce_write_gain']:+.3f}/{row['ce_joint_gain']:+.3f}/{row['ce_interaction']:+.3f} |")
    lines += ["", "## Interpretation boundary", "", "The read factor acts on realized head-route coefficients produced by QK; it is not a pure intervention on QK weights. The write factor remixes those coordinates into native heads whose contextual writes include OV and value state. The factorial therefore localizes specificity to the route-read versus head-payload interface, not to a single static matrix."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
