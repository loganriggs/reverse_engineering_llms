#!/usr/bin/env python3
"""Task-agnostic census of route geometry across natural FineWeb strata.

Unlike the earlier CE atlas, this script retains every source edge.  It asks
whether one global low-rank residual chart works equally well for local versus
long-range routes, different predicted-token forms, and easy versus surprising
predictions.  Semantic labels are used only for already-frozen route modes.
"""

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
from run_bilin18_ce_adjoint_residual_atlas import (
    forward_with_route_leaves,
    semantic_reconstruction,
    subspace_cosine,
)
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_stratified_natural_route_census.json")
OUTPUT = TENSOR_ROOT / "bilin18_stratified_natural_route_census.json"
ARRAYS = TENSOR_ROOT / "bilin18_stratified_natural_route_census.npz"
REPORT = TENSOR_ROOT / "BILIN18_STRATIFIED_NATURAL_ROUTE_CENSUS.md"


def distance_class(distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance)
    result = np.full(distance.shape, "far65+", dtype="U16")
    result[distance == 0] = "self"
    result[(distance >= 1) & (distance <= 4)] = "local1-4"
    result[(distance >= 5) & (distance <= 16)] = "near5-16"
    result[(distance >= 17) & (distance <= 64)] = "mid17-64"
    return result


def token_class_from_flags(flags: np.ndarray) -> np.ndarray:
    """Assign one mutually exclusive coarse form class with fixed precedence."""
    flags = np.asarray(flags, dtype=bool)
    index = {name: FEATURES.index(name) for name in FEATURES}
    result = np.full(len(flags), "other", dtype="U16")
    function = np.zeros(len(flags), dtype=bool)
    for name in ("determiner", "preposition", "conjunction", "pronoun", "auxiliary", "modal"):
        function |= flags[:, index[name]]
    continuation = flags[:, index["word_continuation"]]
    capitalized = flags[:, index["capitalized"]]
    numeric = flags[:, index["numeric"]] | flags[:, index["alphanumeric"]]
    punctuation = flags[:, index["punctuation"]] | flags[:, index["newline"]] | flags[:, index["tab"]]
    specialized = (
        flags[:, index["programming_keyword"]] | flags[:, index["code_operator"]]
        | flags[:, index["url_web"]] | flags[:, index["currency"]]
    )
    result[continuation] = "continuation"
    result[capitalized] = "capitalized"
    result[function] = "function"
    result[numeric] = "numeric"
    result[specialized] = "specialized"
    result[punctuation] = "punctuation"
    return result


def weighted_basis(values: np.ndarray, effects: np.ndarray, rank: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    effects = np.asarray(effects, dtype=np.float64)
    weights = np.sqrt(np.abs(effects) / max(np.mean(np.abs(effects)), 1e-30))
    _, _, vh = np.linalg.svd(values * weights[:, None], full_matrices=False)
    return vh[:rank]


def project(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (values @ basis.T) @ basis


def effect_fidelity(values: np.ndarray, gradients: np.ndarray, basis: np.ndarray) -> float:
    target = np.einsum("ni,ni->n", values, gradients)
    predicted = np.einsum("ni,ni->n", project(values, basis), gradients)
    return float(1 - np.square(target - predicted).sum() / max(np.square(target).sum(), 1e-30))


def random_basis(dimension: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((dimension, rank)))
    return q.T


def loss_classes(discovery_loss: np.ndarray, values: np.ndarray) -> np.ndarray:
    boundaries = np.quantile(discovery_loss, [.25, .50, .75])
    return np.asarray(["easy", "medium-low", "medium-high", "hard"])[np.digitize(values, boundaries)]


def all_strata(panel: dict[str, np.ndarray], discovery_loss: np.ndarray) -> dict[str, np.ndarray]:
    count = len(panel["distance"])
    result = {"all": np.ones(count, dtype=bool)}
    families = {
        "distance": distance_class(panel["distance"]),
        "next": panel["next_class"],
        "source": panel["source_class"],
        "loss": loss_classes(discovery_loss, panel["loss"]),
    }
    for family, labels in families.items():
        for label in np.unique(labels):
            result[f"{family}:{label}"] = labels == label
    return result


def collect_panel(model, corpus, global_start: int, config: dict, token_flags: np.ndarray):
    layers = model.config.n_layer
    buckets = [{key: [] for key in ("routes", "residuals", "gradients", "semantic_effect", "full_effect", "member", "distance", "source_id", "next_id", "loss", "global_row", "destination", "source_position")} for _ in range(layers)]
    maximum_additive_error = 0.0
    centers = config["_centers"]
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length + 1], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :length], device="cuda")
            targets = torch.as_tensor(raw[:, length], device="cuda")
            logits, leaves = forward_with_route_leaves(model, ids)
            individual_loss = F.cross_entropy(logits, targets, reduction="none")
            individual_loss.sum().backward()
            for layer, pattern in enumerate(leaves):
                routes = pattern[:, :, destination, :destination + 1].detach().float().permute(0, 2, 1)
                gradients = pattern.grad[:, :, destination, :destination + 1].detach().float().permute(0, 2, 1)
                semantic, selected = semantic_reconstruction(routes, centers[layer], config["semantic_mode_cosine_threshold"])
                residual = routes - semantic
                full_effect = (routes * gradients).sum(-1)
                semantic_effect = (semantic * gradients).sum(-1)
                residual_effect = (residual * gradients).sum(-1)
                maximum_additive_error = max(maximum_additive_error, float((full_effect - semantic_effect - residual_effect).abs().max()))
                batch = len(raw); sources = destination + 1
                bucket = buckets[layer]
                bucket["routes"].append(routes.reshape(-1, routes.shape[-1]).cpu().numpy())
                bucket["residuals"].append(residual.reshape(-1, residual.shape[-1]).cpu().numpy())
                bucket["gradients"].append(gradients.reshape(-1, gradients.shape[-1]).cpu().numpy())
                bucket["semantic_effect"].append(semantic_effect.reshape(-1).cpu().numpy())
                bucket["full_effect"].append(full_effect.reshape(-1).cpu().numpy())
                bucket["member"].append(selected.reshape(-1).cpu().numpy())
                bucket["distance"].append(np.tile(destination - np.arange(sources), batch))
                bucket["source_id"].append(raw[:, :sources].reshape(-1))
                bucket["next_id"].append(np.repeat(raw[:, length], sources))
                bucket["loss"].append(np.repeat(individual_loss.detach().cpu().numpy(), sources))
                bucket["global_row"].append(np.repeat(global_start + start + np.arange(batch), sources))
                bucket["destination"].append(np.full(batch * sources, destination, dtype=np.int64))
                bucket["source_position"].append(np.tile(np.arange(sources), batch))
            del logits, leaves
        print(f"stratified census panel={global_start} destination={destination}", flush=True)
    output = []
    for bucket in buckets:
        row = {key: np.concatenate(value) for key, value in bucket.items()}
        row["source_class"] = token_class_from_flags(token_flags[row["source_id"]])
        row["next_class"] = token_class_from_flags(token_flags[row["next_id"]])
        output.append(row)
    return output, maximum_additive_error


def summarize_semantics(panel: dict[str, np.ndarray], mask: np.ndarray) -> dict:
    full = panel["full_effect"][mask]
    semantic = panel["semantic_effect"][mask]
    residual = full - semantic
    return {
        "count": int(mask.sum()),
        "semantic_effect_fidelity": float(1 - np.square(residual).sum() / max(np.square(full).sum(), 1e-30)),
        "semantic_absolute_ratio": float(np.abs(semantic).sum() / max(np.abs(full).sum(), 1e-30)),
        "semantic_member_fraction": float(panel["member"][mask].mean()),
        "full_effect_square": float(np.square(full).sum()),
    }


def main():
    config = json.loads(CONFIG.read_text())
    semantic = json.loads(Path(config["semantic_source"]).read_text())
    validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {
        row["candidate"] for name, row in validation["hypotheses"].items()
        if name.startswith("S") and name[1:2].isdigit()
        and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]
    }
    raw_centers = {layer: [] for layer in range(18)}
    for row in semantic["results"]:
        if row["name"] in passed:
            raw_centers[int(row["layer"])].append(row["center"])
    config["_centers"] = {
        layer: torch.as_tensor(value, device="cuda", dtype=torch.float32) if value else None
        for layer, value in raw_centers.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    token_flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    source = np.load(config["corpus"], mmap_mode="r")
    panels = {
        "discovery": (config["discovery_start"], source[config["discovery_start"]:config["discovery_start"] + config["rows_per_panel"]]),
        "replication": (config["replication_start"], source[config["replication_start"]:config["replication_start"] + config["rows_per_panel"]]),
    }
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    collected = {}; errors = []
    for name, (offset, corpus) in panels.items():
        collected[name], error = collect_panel(model, corpus, offset, config, token_flags)
        errors.append(error)

    rng = np.random.default_rng(config["seed"])
    results = []; null_cosines = {rank: [] for rank in config["ranks"]}
    arrays = {}
    for layer in range(18):
        discovery = collected["discovery"][layer]; replication = collected["replication"][layer]
        d_strata = all_strata(discovery, discovery["loss"]); r_strata = all_strata(replication, discovery["loss"])
        for key in ("residuals", "gradients", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss", "global_row", "destination", "source_position"):
            arrays[f"discovery_{key}_L{layer}"] = discovery[key]
            arrays[f"replication_{key}_L{layer}"] = replication[key]
        for rank in config["ranks"]:
            global_basis = weighted_basis(discovery["residuals"], discovery["full_effect"] - discovery["semantic_effect"], rank)
            replication_global = weighted_basis(replication["residuals"], replication["full_effect"] - replication["semantic_effect"], rank)
            global_recurrence = subspace_cosine(global_basis, replication_global)
            for stratum, dmask in d_strata.items():
                rmask = r_strata.get(stratum)
                if rmask is None or dmask.sum() < config["minimum_stratum_edges"] or rmask.sum() < config["minimum_stratum_edges"]:
                    continue
                local_basis = weighted_basis(discovery["residuals"][dmask], (discovery["full_effect"] - discovery["semantic_effect"])[dmask], rank)
                local_replication = weighted_basis(replication["residuals"][rmask], (replication["full_effect"] - replication["semantic_effect"])[rmask], rank)
                local_fidelity = effect_fidelity(replication["residuals"][rmask], replication["gradients"][rmask], local_basis)
                global_fidelity = effect_fidelity(replication["residuals"][rmask], replication["gradients"][rmask], global_basis)
                random_fidelities = [effect_fidelity(replication["residuals"][rmask], replication["gradients"][rmask], random_basis(9, rank, rng)) for _ in range(config["null_replicates"])]
                row = {
                    "layer": layer, "rank": rank, "stratum": stratum,
                    "discovery_count": int(dmask.sum()), "replication_count": int(rmask.sum()),
                    "local_effect_fidelity": local_fidelity,
                    "global_effect_fidelity": global_fidelity,
                    "local_advantage": local_fidelity - global_fidelity,
                    "local_recurrence": subspace_cosine(local_basis, local_replication),
                    "global_recurrence": global_recurrence,
                    "random_effect_fidelity_mean": float(np.mean(random_fidelities)),
                    "random_effect_fidelity_p95": float(np.quantile(random_fidelities, .95)),
                    **summarize_semantics(replication, rmask),
                }
                results.append(row)
                if stratum != "all":
                    null_cosines[rank].extend(subspace_cosine(local_basis, random_basis(9, rank, rng)) for _ in range(config["null_replicates"]))

    np.savez_compressed(ARRAYS, **arrays)
    eligible = [row for row in results if row["stratum"] != "all"]
    rank2 = [row for row in eligible if row["rank"] == 2]
    rank4 = [row for row in eligible if row["rank"] == 4]
    by_layer_ranges = {}
    for layer in range(18):
        values = [row["semantic_effect_fidelity"] for row in rank2 if row["layer"] == layer]
        by_layer_ranges[str(layer)] = float(max(values) - min(values)) if values else 0.0
    heterogeneous = [int(layer) for layer, value in by_layer_ranges.items() if value >= .20]
    improved_rank2 = [row for row in rank2 if row["local_advantage"] >= .05]
    observed_recurrence = float(np.mean([row["local_recurrence"] for row in rank2]))
    null_p95 = float(np.quantile(null_cosines[2], .95))
    median_advantage_2 = float(np.median([row["local_advantage"] for row in rank2]))
    median_advantage_4 = float(np.median([row["local_advantage"] for row in rank4]))
    complete_layers = []
    for layer in range(18):
        values = [row["semantic_effect_fidelity"] for row in rank2 if row["layer"] == layer]
        if values and min(values) >= .80:
            complete_layers.append(layer)
    hypotheses = {
        "B0_panels_are_fresh_and_label_blind": {"supported": config["discovery_start"] >= 6144 and config["discovery_start"] + config["rows_per_panel"] <= config["replication_start"], "ranges": [[config["discovery_start"], config["discovery_start"] + config["rows_per_panel"] - 1], [config["replication_start"], config["replication_start"] + config["rows_per_panel"] - 1]]},
        "B1_known_semantic_coverage_is_heterogeneous": {"supported": len(heterogeneous) >= 6, "layers": heterogeneous, "ranges": by_layer_ranges},
        "B2_one_global_low_rank_chart_is_not_universal": {"supported": len(improved_rank2) >= .25 * len(rank2), "passing_cells": len(improved_rank2), "eligible_cells": len(rank2), "fraction": len(improved_rank2) / max(len(rank2), 1)},
        "B3_stratum_specific_structure_replicates": {"supported": observed_recurrence > null_p95, "observed": observed_recurrence, "null_p95": null_p95},
        "B4_extra_rank_reduces_regime_dependence": {"supported": median_advantage_4 < median_advantage_2, "rank2_median_advantage": median_advantage_2, "rank4_median_advantage": median_advantage_4},
        "B5_current_named_modes_are_not_close_to_complete": {"supported": not complete_layers, "complete_layers": complete_layers},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    clean_config = {key: value for key, value in config.items() if key != "_centers"}
    payload = {"config": clean_config, "hypotheses": hypotheses, "maximum_additive_error": max(errors), "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = ["# Stratified natural route census", "", "Every source edge at four prediction positions on two fresh FineWeb panels is retained. Previously named modes are frozen; all other bases are fitted without task labels. Equal-rank global and stratum-specific residual charts are compared on the replication panel.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "## Rank-two stratum summary", "", "| Stratum | Cells | Semantic fidelity median [min,max] | Local/global residual fidelity | Local advantage | Recurrence | Local beats random p95 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for stratum in sorted(set(row["stratum"] for row in rank2)):
        cells = [row for row in rank2 if row["stratum"] == stratum]
        semantic_values = [row["semantic_effect_fidelity"] for row in cells]
        lines.append(f"| {stratum} | {len(cells)} | {np.median(semantic_values):.3f} [{min(semantic_values):.3f},{max(semantic_values):.3f}] | {np.mean([row['local_effect_fidelity'] for row in cells]):.3f}/{np.mean([row['global_effect_fidelity'] for row in cells]):.3f} | {np.mean([row['local_advantage'] for row in cells]):+.3f} | {np.mean([row['local_recurrence'] for row in cells]):.3f} | {np.mean([row['local_effect_fidelity'] > row['random_effect_fidelity_p95'] for row in cells]):.2f} |")
    best = sorted(rank2, key=lambda row: row["local_advantage"], reverse=True)[:12]
    worst = sorted(rank2, key=lambda row: row["local_advantage"])[:12]
    lines += ["", "## Largest departures from the global rank-two chart", "", "| L | Stratum | Local/global | Advantage | Semantic fidelity | Recurrence |", "|---:|---|---:|---:|---:|---:|"]
    for row in best + worst:
        lines.append(f"| {row['layer']} | {row['stratum']} | {row['local_effect_fidelity']:.3f}/{row['global_effect_fidelity']:.3f} | {row['local_advantage']:+.3f} | {row['semantic_effect_fidelity']:.3f} | {row['local_recurrence']:.3f} |")
    lines += ["", "## Guardrails", "", "The target is the local frozen-route CE adjoint, not a complete attribution of model loss. Coarse token-form and distance strata are observational partitions, not semantic tasks. A stratum-specific basis can exploit stable distributional covariance without constituting a human-readable feature; finite interventions and held-out semantic predictions are required before naming it. Weighted PCA is a declared compression heuristic rather than the mathematically optimal subspace for the varying per-edge gradient metric."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
