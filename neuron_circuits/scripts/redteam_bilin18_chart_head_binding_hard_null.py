#!/usr/bin/env python3
"""Discovery-selected hard head-permutation nulls for natural charts."""

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


CONFIG = Path("configs/bilin18_chart_head_binding_hard_null.json")
OUTPUT = TENSOR_ROOT / "bilin18_chart_head_binding_hard_null.json"
REPORT = TENSOR_ROOT / "BILIN18_CHART_HEAD_BINDING_HARD_NULL.md"


def sample_permutations(dimension, count, rng):
    identity = tuple(range(dimension)); values = set()
    while len(values) < count:
        value = tuple(rng.permutation(dimension).tolist())
        if value != identity: values.add(value)
    return [np.asarray(value, dtype=int) for value in values]


def local_fidelity(values, gradients, read, write):
    target = np.einsum("ni,ni->n", values, gradients)
    predicted = np.einsum("ni,ni->n", (values @ read.T) @ write, gradients)
    return float(1 - np.square(target - predicted).sum() / max(np.square(target).sum(), 1e-30))


def select_hard_permutations(values, gradients, basis, permutations, count):
    scored = {family: [] for family in ("joint", "read", "write")}
    for permutation in permutations:
        permuted = basis[:, permutation]
        scored["joint"].append((local_fidelity(values, gradients, permuted, permuted), permutation))
        scored["read"].append((local_fidelity(values, gradients, permuted, basis), permutation))
        scored["write"].append((local_fidelity(values, gradients, basis, permuted), permutation))
    return {family: sorted(rows, key=lambda item: item[0], reverse=True)[:count] for family, rows in scored.items()}


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"]); rng = np.random.default_rng(config["seed"])
    semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    centers = {target["layer"]: [] for target in config["targets"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in centers: centers[row["layer"]].append(row["center"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    token_classes = token_class_from_flags(flags); candidates = sample_permutations(9, config["permutation_candidates"], rng); prepared = {}
    for target in config["targets"]:
        layer = target["layer"]
        panel = {key: arrays[f"discovery_{key}_L{layer}"] for key in ("residuals", "gradients", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss")}
        panel["source_class"] = token_classes[panel["source_id"]]; panel["next_class"] = token_classes[panel["next_id"]]
        mask = all_strata(panel, panel["loss"])[target["target"]]; effects = panel["full_effect"] - panel["semantic_effect"]
        basis = weighted_basis(panel["residuals"][mask], effects[mask], config["rank"])
        selected = select_hard_permutations(panel["residuals"][mask], panel["gradients"][mask], basis, candidates, config["selected_per_family"])
        prepared[layer] = {"basis": torch.as_tensor(basis, device="cuda", dtype=torch.float32), "selected": {family: [(score, torch.as_tensor(basis[:, permutation], device="cuda", dtype=torch.float32)) for score, permutation in rows] for family, rows in selected.items()}, "semantic": torch.as_tensor(centers[layer], device="cuda", dtype=torch.float32) if centers[layer] else None, "loss": panel["loss"]}
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    null_names = [f"{family}_{i}" for family in ("joint", "read", "write") for i in range(config["selected_per_family"])]
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
                selected_mask = torch.as_tensor(selected_np, device="cuda"); routes = native_routes(model, ids, layer)
                semantic_part, _ = semantic_reconstruction(routes, item["semantic"], config["semantic_mode_cosine_threshold"]); residual = routes - semantic_part; basis = item["basis"]
                conditions = {"native": routes, "deleted": torch.where(selected_mask[..., None], semantic_part, routes), "matching": torch.where(selected_mask[..., None], semantic_part + factorized_transform(residual, basis, basis), routes)}
                for family in ("joint", "read", "write"):
                    for index, (_, permuted) in enumerate(item["selected"][family]):
                        read = permuted if family in ("joint", "read") else basis; write = permuted if family in ("joint", "write") else basis
                        conditions[f"{family}_{index}"] = torch.where(selected_mask[..., None], semantic_part + factorized_transform(residual, read, write), routes)
                names, logits = intervene(model, ids, layer, conditions); outputs = {name: logits[index] for index, name in enumerate(names)}
                max_error = max(max_error, float((outputs["native"] - reference).abs().max())); example_mask = torch.as_tensor(selected_examples, device="cuda")
                for name in methods: update_metric(metrics[layer][name], outputs["native"][example_mask], outputs["deleted"][example_mask], outputs[name][example_mask], targets[example_mask])
        print(f"hard-head-null destination={destination}", flush=True)
    results = {}; family_pass = {family: [] for family in ("joint", "read", "write")}; material = []
    for target in config["targets"]:
        layer = target["layer"]; rows = {name: finish(value) for name, value in metrics[layer].items()}; matching = rows["matching"]
        summary = {"layer": layer, "target": target["target"], "matching": matching, "families": {}}
        target_margins = []
        for family in ("joint", "read", "write"):
            logit = [rows[f"{family}_{index}"]["causal_fidelity"] for index in range(config["selected_per_family"])]
            ce = [rows[f"{family}_{index}"]["ce_effect_fidelity"] for index in range(config["selected_per_family"])]
            best_logit = max(logit); best_ce = max(ce); logit_margin = matching["causal_fidelity"] - best_logit; ce_margin = matching["ce_effect_fidelity"] - best_ce
            selected_rows = prepared[layer]["selected"][family]
            summary["families"][family] = {"best_finite_logit": best_logit, "best_finite_ce": best_ce, "logit_margin": logit_margin, "ce_margin": ce_margin, "discovery_score_range": [float(selected_rows[0][0]), float(selected_rows[-1][0])]}
            family_pass[family].append(logit_margin > 0 and ce_margin > 0); target_margins.extend((logit_margin, ce_margin))
        material.append(min(target_margins) >= .05); results[target["target"]] = summary
    hypotheses = {
        "U0_native_replay_is_exact": {"supported": max_error <= .02, "maximum_logit_error": max_error},
        "U1_joint_binding_beats_discovery_selected_nulls": {"supported": all(family_pass["joint"]), "by_target": family_pass["joint"]},
        "U2_read_binding_beats_discovery_selected_nulls": {"supported": all(family_pass["read"]), "by_target": family_pass["read"]},
        "U3_write_binding_beats_discovery_selected_nulls": {"supported": all(family_pass["write"]), "by_target": family_pass["write"]},
        "U4_native_binding_has_a_material_hard_null_margin": {"supported": any(material), "by_target": material},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Discovery-selected hard head-binding null", "", "For each target, 4,096 coefficient-preserving head permutations are screened on discovery frozen-route CE fidelity. The best 32 joint, read-only, and write-only permutations are frozen and evaluated with full suffix recomputation on new FineWeb.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | Matching logit/CE | Joint best + margins | Read best + margins | Write best + margins |", "|---|---:|---:|---:|---:|"]
    for name, row in results.items():
        m = row["matching"]
        cells = []
        for family in ("joint", "read", "write"):
            f = row["families"][family]; cells.append(f"{f['best_finite_logit']:.3f}/{f['best_finite_ce']:.3f} ({f['logit_margin']:+.3f}/{f['ce_margin']:+.3f})")
        lines.append(f"| L{row['layer']} {name} | {m['causal_fidelity']:.3f}/{m['ce_effect_fidelity']:.3f} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += ["", "## Guardrail", "", "The hard nulls are selected on a local frozen-route CE metric, not on finite validation logits. This is a strong multiple-comparison control for prediction-relevant binding but not an exhaustive search over all 9! permutations or all possible orthogonal head remappings."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
