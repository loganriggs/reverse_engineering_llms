#!/usr/bin/env python3
"""Finite validation of natural-stratum-specific route charts."""

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
from run_bilin18_stratified_natural_route_census import (
    FEATURES,
    all_strata,
    distance_class,
    random_basis,
    token_class_from_flags,
    weighted_basis,
)
from run_bilin18_unseeded_route_end_to_end_causality import (
    empty_metric,
    finish,
    full_logits,
    intervene,
    native_routes,
    update_metric,
)


CONFIG = Path("configs/bilin18_stratified_route_finite_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_stratified_route_finite_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_STRATIFIED_ROUTE_FINITE_CAUSALITY.md"


def project(routes: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (routes @ basis.T) @ basis


def runtime_mask(stratum: str, raw: np.ndarray, destination: int, losses: np.ndarray,
                 token_classes: np.ndarray, discovery_loss: np.ndarray) -> np.ndarray:
    batch = len(raw); sources = destination + 1
    family, label = stratum.split(":", 1)
    if family == "distance":
        labels = distance_class(destination - np.arange(sources))
        return np.broadcast_to(labels == label, (batch, sources)).copy()
    if family == "source":
        return token_classes[raw[:, :sources]] == label
    if family == "next":
        selected = token_classes[raw[:, destination + 1]] == label
        return np.broadcast_to(selected[:, None], (batch, sources)).copy()
    if family == "loss":
        boundaries = np.quantile(discovery_loss, [.25, .50, .75])
        labels = np.asarray(["easy", "medium-low", "medium-high", "hard"])[np.digitize(losses, boundaries)]
        return np.broadcast_to((labels == label)[:, None], (batch, sources)).copy()
    raise ValueError(stratum)


def main():
    config = json.loads(CONFIG.read_text())
    source_result = json.loads(Path(config["source"]).read_text())
    arrays = np.load(config["arrays"])
    semantic = json.loads(Path(config["semantic_source"]).read_text())
    validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {
        row["candidate"] for name, row in validation["hypotheses"].items()
        if name.startswith("S") and name[1:2].isdigit()
        and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]
    }
    raw_centers = {target["layer"]: [] for target in config["targets"]}
    for row in semantic["results"]:
        if row["name"] in passed and row["layer"] in raw_centers:
            raw_centers[row["layer"]].append(row["center"])

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    token_flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    token_classes = token_class_from_flags(token_flags)
    rng = np.random.default_rng(config["seed"])
    observational = {(row["layer"], row["stratum"]): row for row in source_result["results"] if row["rank"] == config["rank"]}
    prepared = {}
    for target in config["targets"]:
        layer = target["layer"]; stratum = target["stratum"]
        panel = {
            "residuals": arrays[f"discovery_residuals_L{layer}"],
            "full_effect": arrays[f"discovery_full_effect_L{layer}"],
            "semantic_effect": arrays[f"discovery_semantic_effect_L{layer}"],
            "distance": arrays[f"discovery_distance_L{layer}"],
            "source_id": arrays[f"discovery_source_id_L{layer}"],
            "next_id": arrays[f"discovery_next_id_L{layer}"],
            "loss": arrays[f"discovery_loss_L{layer}"],
        }
        panel["source_class"] = token_classes[panel["source_id"]]
        panel["next_class"] = token_classes[panel["next_id"]]
        masks = all_strata(panel, panel["loss"])
        effects = panel["full_effect"] - panel["semantic_effect"]
        global_basis = weighted_basis(panel["residuals"], effects, config["rank"])
        local_basis = weighted_basis(panel["residuals"][masks[stratum]], effects[masks[stratum]], config["rank"])
        prepared[(layer, stratum)] = {
            "global": torch.as_tensor(global_basis, device="cuda", dtype=torch.float32),
            "local": torch.as_tensor(local_basis, device="cuda", dtype=torch.float32),
            "random": [torch.as_tensor(random_basis(9, config["rank"], rng), device="cuda", dtype=torch.float32) for _ in range(config["null_replicates"])],
            "semantic": torch.as_tensor(raw_centers[layer], device="cuda", dtype=torch.float32) if raw_centers[layer] else None,
            "discovery_loss": panel["loss"],
        }

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"]]
    methods = ["native", "deleted", "global", "local"] + [f"random_{index}" for index in range(config["null_replicates"])]
    metrics = {(target["layer"], target["stratum"]): {name: empty_metric() for name in methods} for target in config["targets"]}
    maximum_error = 0.0
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length + 1], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :length], device="cuda")
            targets = torch.as_tensor(raw[:, length], device="cuda")
            reference = full_logits(model, ids)
            losses = F.cross_entropy(reference, targets, reduction="none").cpu().numpy()
            for target in config["targets"]:
                layer = target["layer"]; stratum = target["stratum"]
                basis = prepared[(layer, stratum)]
                selected_np = runtime_mask(stratum, raw, destination, losses, token_classes, basis["discovery_loss"])
                selected_examples = selected_np.any(1)
                if not selected_examples.any():
                    continue
                selected = torch.as_tensor(selected_np, device="cuda")
                routes = native_routes(model, ids, layer)
                semantic_part, _ = semantic_reconstruction(routes, basis["semantic"], config["semantic_mode_cosine_threshold"])
                residual = routes - semantic_part
                conditions = {
                    "native": routes,
                    "deleted": torch.where(selected[..., None], semantic_part, routes),
                    "global": torch.where(selected[..., None], semantic_part + project(residual, basis["global"]), routes),
                    "local": torch.where(selected[..., None], semantic_part + project(residual, basis["local"]), routes),
                }
                for index, random in enumerate(basis["random"]):
                    conditions[f"random_{index}"] = torch.where(selected[..., None], semantic_part + project(residual, random), routes)
                names, logits = intervene(model, ids, layer, conditions)
                outputs = {name: logits[index] for index, name in enumerate(names)}
                maximum_error = max(maximum_error, float((outputs["native"] - reference).abs().max()))
                example_mask = torch.as_tensor(selected_examples, device="cuda")
                for name in methods:
                    update_metric(metrics[(layer, stratum)][name], outputs["native"][example_mask], outputs["deleted"][example_mask], outputs[name][example_mask], targets[example_mask])
        print(f"finite stratified destination={destination}", flush=True)

    results = {}
    finite_advantages = []; observational_advantages = []
    local_wins = []; ce_wins = []; random_wins = []
    for target in config["targets"]:
        layer = target["layer"]; stratum = target["stratum"]; key = (layer, stratum)
        rows = {name: finish(value) for name, value in metrics[key].items()}
        logit_advantage = rows["local"]["causal_fidelity"] - rows["global"]["causal_fidelity"]
        ce_advantage = rows["local"]["ce_effect_fidelity"] - rows["global"]["ce_effect_fidelity"]
        random_p95 = float(np.quantile([rows[f"random_{index}"]["causal_fidelity"] for index in range(config["null_replicates"])], .95))
        observed = observational[key]["local_advantage"]
        results[f"L{layer}:{stratum}"] = {"metrics": rows, "finite_logit_advantage": logit_advantage, "finite_ce_advantage": ce_advantage, "random_logit_fidelity_p95": random_p95, "observational_advantage": observed}
        finite_advantages.append(logit_advantage); observational_advantages.append(observed)
        local_wins.append(logit_advantage > 0); ce_wins.append(ce_advantage > 0); random_wins.append(rows["local"]["causal_fidelity"] > random_p95)
    correlation = float(np.corrcoef(observational_advantages, finite_advantages)[0, 1]) if np.std(finite_advantages) and np.std(observational_advantages) else 0.0
    hypotheses = {
        "F0_native_replay_is_exact": {"supported": maximum_error <= .02, "maximum_logit_error": maximum_error},
        "F1_specialized_charts_improve_finite_logit_recovery": {"supported": sum(local_wins) >= 3 and float(np.mean(finite_advantages)) >= .05, "wins": int(sum(local_wins)), "mean_advantage": float(np.mean(finite_advantages)), "by_target": finite_advantages},
        "F2_specialized_charts_improve_finite_ce_recovery": {"supported": sum(ce_wins) >= 3 and float(np.mean([row["finite_ce_advantage"] for row in results.values()])) >= .05, "wins": int(sum(ce_wins)), "mean_advantage": float(np.mean([row["finite_ce_advantage"] for row in results.values()]))},
        "F3_specialized_charts_beat_equal_rank_random_controls": {"supported": all(random_wins), "wins": int(sum(random_wins)), "targets": len(random_wins)},
        "F4_local_geometry_predicts_finite_advantage": {"supported": correlation >= .50, "correlation": correlation},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite causality of natural-stratum route charts", "", "For each frozen target, only residual route components belonging to the selected natural stratum are replaced. All other source edges remain native. Local and global rank-two charts, deletion, and sixteen random charts are propagated through the full suffix.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | Examples | Local/global logit fidelity | Advantage | Local/global CE fidelity | Advantage | Random p95 | Observational advantage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        local = row["metrics"]["local"]; global_row = row["metrics"]["global"]
        lines.append(f"| {name} | {local['examples']} | {local['causal_fidelity']:.3f}/{global_row['causal_fidelity']:.3f} | {row['finite_logit_advantage']:+.3f} | {local['ce_effect_fidelity']:.3f}/{global_row['ce_effect_fidelity']:.3f} | {row['finite_ce_advantage']:+.3f} | {row['random_logit_fidelity_p95']:.3f} | {row['observational_advantage']:+.3f} |")
    lines += ["", "## Guardrails", "", "The four targets were selected after the observational census but before this fresh panel. This validates the selected cells, not the prevalence of specialization across every cell. Deleting a stratum's residual route component defines a conditional route intervention, not a complete causal decomposition of the model. Loss strata use discovery-panel quartiles and are selected using native loss before intervention."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
