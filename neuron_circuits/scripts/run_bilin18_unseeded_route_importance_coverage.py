#!/usr/bin/env python3
"""Measure task-agnostic route-ontology coverage on held-out FineWeb.

The audit distinguishes reconstruction of routing coefficients from
reconstruction of the contextual OV write.  All selection and semantic labels
are frozen before this panel.  Squared contextual edge-write norm supplies the
importance weighting; no downstream task or answer labels are used.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_unseeded_route_importance_coverage.json")
OUTPUT = TENSOR_ROOT / "bilin18_unseeded_route_importance_coverage.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSEEDED_ROUTE_IMPORTANCE_COVERAGE.md"
METHODS = ("pca4", "centroid4", "atlas_modes", "semantic_modes")


def fit_basis(values, rank):
    _, _, vh = np.linalg.svd(np.asarray(values, dtype=np.float64), full_matrices=False)
    return vh[:rank].astype(np.float32)


def project_subspace(routes, basis):
    """Orthogonally project [..., heads] raw route coefficients."""
    return (routes @ basis.T) @ basis


def quantize_centroids(routes, centers):
    """Preserve route norm while replacing its direction by nearest center."""
    norm = np.linalg.norm(routes, axis=-1, keepdims=True)
    unit = routes / np.maximum(norm, 1e-30)
    centers = np.asarray(centers, dtype=np.float32)
    assignment = np.argmax(unit @ centers.T, axis=-1)
    return norm * centers[assignment]


def gated_modes(routes, centers, threshold):
    """Reconstruct only routes sufficiently aligned to a frozen mode."""
    if len(centers) == 0:
        return np.zeros_like(routes), np.zeros(routes.shape[:-1], dtype=bool)
    norm = np.linalg.norm(routes, axis=-1, keepdims=True)
    unit = routes / np.maximum(norm, 1e-30)
    centers = np.asarray(centers, dtype=np.float32)
    similarities = unit @ centers.T
    assignment = np.argmax(similarities, axis=-1)
    selected = np.max(similarities, axis=-1) >= threshold
    reconstructed = norm * centers[assignment]
    return np.where(selected[..., None], reconstructed, 0), selected


def empty_stats():
    return {
        "route_target": 0.0, "route_error": 0.0, "route_dot": 0.0, "route_approx": 0.0,
        "edge_target": 0.0, "edge_error": 0.0, "edge_dot": 0.0, "edge_approx": 0.0,
        "write_target": 0.0, "write_error": 0.0, "write_dot": 0.0, "write_approx": 0.0,
        "assigned_edge_energy": 0.0, "edges": 0, "assigned_edges": 0,
    }


def update_stats(stats, routes, reconstructed, source_writes, selected=None):
    if torch.is_tensor(routes):
        route_delta = reconstructed - routes
        stats["route_target"] += float(routes.square().sum())
        stats["route_error"] += float(route_delta.square().sum())
        stats["route_dot"] += float((routes * reconstructed).sum())
        stats["route_approx"] += float(reconstructed.square().sum())
        edge = torch.einsum("bsh,bshk->bsk", routes, source_writes)
        edge_hat = torch.einsum("bsh,bshk->bsk", reconstructed, source_writes)
        edge_delta = edge_hat - edge
        edge_energy = edge.square().sum(-1)
        stats["edge_target"] += float(edge_energy.sum())
        stats["edge_error"] += float(edge_delta.square().sum())
        stats["edge_dot"] += float((edge * edge_hat).sum())
        stats["edge_approx"] += float(edge_hat.square().sum())
        write = edge.sum(1); write_hat = edge_hat.sum(1)
        stats["write_target"] += float(write.square().sum())
        stats["write_error"] += float((write_hat - write).square().sum())
        stats["write_dot"] += float((write * write_hat).sum())
        stats["write_approx"] += float(write_hat.square().sum())
        stats["edges"] += edge_energy.numel()
        if selected is not None:
            stats["assigned_edge_energy"] += float(edge_energy[selected].sum())
            stats["assigned_edges"] += int(selected.sum())
        return
    routes = np.asarray(routes, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    values = np.asarray(source_writes, dtype=np.float64)
    route_delta = reconstructed - routes
    stats["route_target"] += float(np.square(routes).sum())
    stats["route_error"] += float(np.square(route_delta).sum())
    stats["route_dot"] += float((routes * reconstructed).sum())
    stats["route_approx"] += float(np.square(reconstructed).sum())

    edge = np.einsum("bsh,bshk->bsk", routes, values)
    edge_hat = np.einsum("bsh,bshk->bsk", reconstructed, values)
    edge_delta = edge_hat - edge
    edge_energy = np.square(edge).sum(axis=-1)
    stats["edge_target"] += float(edge_energy.sum())
    stats["edge_error"] += float(np.square(edge_delta).sum())
    stats["edge_dot"] += float((edge * edge_hat).sum())
    stats["edge_approx"] += float(np.square(edge_hat).sum())

    write = edge.sum(axis=1)
    write_hat = edge_hat.sum(axis=1)
    stats["write_target"] += float(np.square(write).sum())
    stats["write_error"] += float(np.square(write_hat - write).sum())
    stats["write_dot"] += float((write * write_hat).sum())
    stats["write_approx"] += float(np.square(write_hat).sum())
    stats["edges"] += int(edge_energy.size)
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        stats["assigned_edge_energy"] += float(edge_energy[selected].sum())
        stats["assigned_edges"] += int(selected.sum())


def update_quadratic_edge_stats(stats, routes, reconstructed, grams, selected=None):
    """Update route/edge metrics from exact per-edge head-write Gram matrices."""
    routes = np.asarray(routes, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    grams = np.asarray(grams, dtype=np.float64)
    delta = reconstructed - routes
    target = np.einsum("ni,nij,nj->n", routes, grams, routes)
    error = np.einsum("ni,nij,nj->n", delta, grams, delta)
    dot = np.einsum("ni,nij,nj->n", reconstructed, grams, routes)
    approx = np.einsum("ni,nij,nj->n", reconstructed, grams, reconstructed)
    stats["route_target"] += float(np.square(routes).sum())
    stats["route_error"] += float(np.square(delta).sum())
    stats["route_dot"] += float((routes * reconstructed).sum())
    stats["route_approx"] += float(np.square(reconstructed).sum())
    stats["edge_target"] += float(target.sum())
    stats["edge_error"] += float(error.sum())
    stats["edge_dot"] += float(dot.sum())
    stats["edge_approx"] += float(approx.sum())
    stats["edges"] += len(routes)
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        stats["assigned_edge_energy"] += float(target[selected].sum())
        stats["assigned_edges"] += int(selected.sum())


def finish(stats):
    result = {}
    for prefix in ("route", "edge", "write"):
        target = stats[f"{prefix}_target"]
        approx = stats[f"{prefix}_approx"]
        result[f"{prefix}_fidelity"] = 1 - stats[f"{prefix}_error"] / max(target, 1e-30)
        result[f"{prefix}_cosine"] = stats[f"{prefix}_dot"] / max(np.sqrt(target * approx), 1e-30)
        result[f"{prefix}_norm_ratio"] = np.sqrt(approx / max(target, 1e-30))
    result["assigned_edge_energy_fraction"] = stats["assigned_edge_energy"] / max(stats["edge_target"], 1e-30)
    result["assigned_edge_fraction"] = stats["assigned_edges"] / max(stats["edges"], 1)
    result["edges"] = stats["edges"]
    return {key: float(value) if isinstance(value, (float, np.floating)) else value for key, value in result.items()}


def pooled(rows):
    total = empty_stats()
    for row in rows:
        raw = row["_raw"]
        for key in total:
            total[key] += raw[key]
    return finish(total)


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    geometry = json.loads(Path(config["geometry_source"]).read_text())
    arrays = np.load(config["geometry_arrays"])
    semantic = json.loads(Path(config["semantic_source"]).read_text())
    validation = json.loads(Path(config["semantic_validation"]).read_text())

    layers = len(geometry["layer_summary"])
    bases = {layer: fit_basis(arrays[f"discovery_routes_L{layer}"], config["continuous_rank"]) for layer in range(layers)}
    centroids = {layer: np.asarray([row["center"] for row in geometry["clusters"][str(layer)]], dtype=np.float32) for layer in range(layers)}
    atlas = {layer: [] for layer in range(layers)}
    for row in semantic["results"]:
        atlas[int(row["layer"])].append(np.asarray(row["center"], dtype=np.float32))

    passed_names = {
        row["candidate"] for name, row in validation["hypotheses"].items()
        if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]
    }
    semantic_modes = {layer: [] for layer in range(layers)}
    for row in semantic["results"]:
        if row["name"] in passed_names:
            semantic_modes[int(row["layer"])].append(np.asarray(row["center"], dtype=np.float32))

    prior_stops = []
    heldout_semantic = json.loads(Path(validation["config"]["heldout"]).read_text())
    for source_config in (geometry["config"], semantic["config"], heldout_semantic["config"]):
        for start_name in ("discovery_start", "replication_start", "start_row"):
            if start_name in source_config:
                prior_stops.append(source_config[start_name] + source_config.get("rows_per_panel", source_config.get("rows", 0)))

    corpus = np.load(config["corpus"], mmap_mode="r")[config["start_row"]:config["start_row"] + config["rows"], :config["sequence_length"]]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    stats = {layer: {method: empty_stats() for method in METHODS} for layer in range(layers)}
    sweep_stats = {
        family: {
            layer: {str(threshold): empty_stats() for threshold in config["threshold_sweep"]}
            for layer in range(layers) if centers_by_layer[layer]
        }
        for family, centers_by_layer in (("atlas_modes", atlas), ("semantic_modes", semantic_modes))
    }
    null_samples = {layer: {"routes": [], "grams": []} for layer in range(layers) if semantic_modes[layer]}
    torch_bases = {layer: torch.as_tensor(value, device="cuda") for layer, value in bases.items()}
    torch_centroids = {layer: torch.as_tensor(value, device="cuda") for layer, value in centroids.items()}
    torch_atlas = {layer: torch.as_tensor(np.stack(value), device="cuda") if value else None for layer, value in atlas.items()}
    torch_semantic = {layer: torch.as_tensor(np.stack(value), device="cuda") if value else None for layer, value in semantic_modes.items()}
    sampling_rng = np.random.default_rng(config["seed"] + 1)
    exact_error = 0.0
    for start in range(0, len(corpus), config["batch_size"]):
        raw = np.asarray(corpus[start:start + config["batch_size"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
            source_writes = torch.einsum("bshd,hdk->bshk", details["mixed_value"].float(), projection)
            for destination in config["destinations"]:
                routes = details["pattern"][:, :, destination, :destination + 1].float().permute(0, 2, 1)
                values = source_writes[:, :destination + 1]
                direct = torch.einsum("bsh,bshk->bk", routes, values)
                bias = getattr(block.attn.c_proj, "bias", None)
                observed = details["write"][:, destination].float()
                if bias is not None:
                    observed = observed - bias.float()
                exact_error = max(exact_error, float((direct - observed).abs().max()))

                pca = (routes @ torch_bases[layer].T) @ torch_bases[layer]
                route_norm = routes.norm(dim=-1, keepdim=True)
                route_unit = routes / route_norm.clamp_min(1e-30)
                similarities = route_unit @ torch_centroids[layer].T
                center = route_norm * torch_centroids[layer][similarities.argmax(-1)]
                if torch_atlas[layer] is None:
                    atlas_hat = torch.zeros_like(routes); atlas_selected = torch.zeros(routes.shape[:-1], dtype=torch.bool, device=routes.device)
                else:
                    atlas_similarity = route_unit @ torch_atlas[layer].T
                    atlas_assignment = atlas_similarity.argmax(-1); atlas_selected = atlas_similarity.max(-1).values >= config["mode_cosine_threshold"]
                    atlas_hat = torch.where(atlas_selected[..., None], route_norm * torch_atlas[layer][atlas_assignment], 0)
                if torch_semantic[layer] is None:
                    semantic_hat = torch.zeros_like(routes); semantic_selected = torch.zeros(routes.shape[:-1], dtype=torch.bool, device=routes.device)
                else:
                    semantic_similarity = route_unit @ torch_semantic[layer].T
                    semantic_assignment = semantic_similarity.argmax(-1); semantic_selected = semantic_similarity.max(-1).values >= config["mode_cosine_threshold"]
                    semantic_hat = torch.where(semantic_selected[..., None], route_norm * torch_semantic[layer][semantic_assignment], 0)
                update_stats(stats[layer]["pca4"], routes, pca, values)
                update_stats(stats[layer]["centroid4"], routes, center, values)
                update_stats(stats[layer]["atlas_modes"], routes, atlas_hat, values, atlas_selected)
                update_stats(stats[layer]["semantic_modes"], routes, semantic_hat, values, semantic_selected)
                for family, center_tensor, similarities_for_family in (
                    ("atlas_modes", torch_atlas[layer], None if torch_atlas[layer] is None else atlas_similarity),
                    ("semantic_modes", torch_semantic[layer], None if torch_semantic[layer] is None else semantic_similarity),
                ):
                    if center_tensor is None:
                        continue
                    assignment = similarities_for_family.argmax(-1)
                    full_reconstruction = route_norm * center_tensor[assignment]
                    maximum_similarity = similarities_for_family.max(-1).values
                    for threshold in config["threshold_sweep"]:
                        selected_at_threshold = maximum_similarity >= threshold
                        gated = torch.where(selected_at_threshold[..., None], full_reconstruction, 0)
                        update_stats(sweep_stats[family][layer][str(threshold)], routes, gated, values, selected_at_threshold)
                if layer in null_samples:
                    count = min(config["null_edges_per_batch_destination"], destination + 1)
                    chosen = np.sort(sampling_rng.choice(destination + 1, size=count, replace=False))
                    chosen_t = torch.as_tensor(chosen, device=routes.device)
                    sampled_routes = routes[:, chosen_t]
                    sampled_values = values[:, chosen_t]
                    grams = torch.einsum("bihk,bijk->bihj", sampled_values, sampled_values)
                    null_samples[layer]["routes"].append(sampled_routes.reshape(-1, heads).cpu().numpy())
                    null_samples[layer]["grams"].append(grams.reshape(-1, heads, heads).cpu().numpy())
            pre = mixed + details["write"]
            normalized = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        if start % 64 == 0:
            print(f"coverage rows {start}/{len(corpus)}", flush=True)

    rows = []
    for layer in range(layers):
        row = {
            "layer": layer,
            "atlas_mode_count": len(atlas[layer]),
            "semantic_mode_count": len(semantic_modes[layer]),
        }
        for method in METHODS:
            row[method] = finish(stats[layer][method]); row[method]["_raw"] = stats[layer][method]
        rows.append(row)

    # Destroy mode/head binding while keeping the held-out contextual OV geometry,
    # mode count, and each center's coefficient multiset.
    rng = np.random.default_rng(config["seed"]); null_fidelity = []
    semantic_layers = [layer for layer in range(layers) if semantic_modes[layer]]
    sampled = {
        layer: {
            "routes": np.concatenate(null_samples[layer]["routes"]),
            "grams": np.concatenate(null_samples[layer]["grams"]),
        }
        for layer in semantic_layers
    }
    observed_sample_rows = []
    for layer in semantic_layers:
        routes = sampled[layer]["routes"]
        reconstructed, selected = gated_modes(routes, semantic_modes[layer], config["mode_cosine_threshold"])
        item = empty_stats(); update_quadratic_edge_stats(item, routes, reconstructed, sampled[layer]["grams"], selected)
        observed_sample_rows.append({"_raw": item})
    observed_sample = pooled(observed_sample_rows)
    for _ in range(config["null_replicates"]):
        null_rows = []
        for layer in semantic_layers:
            centers = np.stack(semantic_modes[layer])
            permuted = np.stack([center[rng.permutation(heads)] for center in centers])
            raw_routes = sampled[layer]["routes"]
            reconstructed, selected = gated_modes(raw_routes, permuted, config["mode_cosine_threshold"])
            item = empty_stats(); update_quadratic_edge_stats(item, raw_routes, reconstructed, sampled[layer]["grams"], selected)
            null_rows.append({"_raw": item})
        null_fidelity.append(pooled(null_rows)["edge_fidelity"])

    clean_rows = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key not in METHODS}
        for method in METHODS:
            clean[method] = {key: value for key, value in row[method].items() if key != "_raw"}
        clean_rows.append(clean)

    candidate_layers = [row for row in rows if row["atlas_mode_count"]]
    validated_layers = [row for row in rows if row["semantic_mode_count"]]
    atlas_pooled = pooled([{"_raw": row["atlas_modes"]["_raw"]} for row in candidate_layers])
    semantic_pooled = pooled([{"_raw": row["semantic_modes"]["_raw"]} for row in validated_layers])
    all_atlas_pooled = pooled([{"_raw": row["atlas_modes"]["_raw"]} for row in rows])
    all_semantic_pooled = pooled([{"_raw": row["semantic_modes"]["_raw"]} for row in rows])
    global_edge_target = sum(row["pca4"]["_raw"]["edge_target"] for row in rows)
    threshold_sweep = {"atlas_modes": {}, "semantic_modes": {}}
    for family in threshold_sweep:
        for threshold in config["threshold_sweep"]:
            key = str(threshold)
            family_rows = [{"_raw": sweep_stats[family][layer][key]} for layer in sweep_stats[family]]
            conditional = pooled(family_rows)
            raw = empty_stats()
            for row in family_rows:
                for name in raw:
                    raw[name] += row["_raw"][name]
            threshold_sweep[family][key] = {
                "conditional_layers": conditional,
                "model_wide_assigned_edge_energy_fraction": raw["assigned_edge_energy"] / max(global_edge_target, 1e-30),
                "model_wide_edge_fidelity": (raw["edge_target"] - raw["edge_error"]) / max(global_edge_target, 1e-30),
            }
    pca_wins = [row["layer"] for row in rows if row["pca4"]["edge_fidelity"] > row["centroid4"]["edge_fidelity"]]
    route_values, edge_values = [], []
    for row in rows:
        for method in ("pca4", "centroid4"):
            route_values.append(row[method]["route_fidelity"]); edge_values.append(row[method]["edge_fidelity"])
    route_edge_correlation = pearson(route_values, edge_values)
    null_p95 = float(np.quantile(null_fidelity, .95))
    central_sweep = [threshold_sweep["semantic_modes"][str(value)] for value in (.6, .7, .8)]
    hypotheses = {
        "C0_panel_is_prospectively_held_out": {"supported": config["start_row"] >= max(prior_stops), "start_row": config["start_row"], "latest_prior_stop": max(prior_stops)},
        "C1_continuous_code_beats_centroid_code": {"supported": len(pca_wins) >= 12, "layers": pca_wins},
        "C2_geometric_atlas_covers_material_natural_computation": {"supported": atlas_pooled["assigned_edge_energy_fraction"] >= .15, "candidate_layers": [row["layer"] for row in candidate_layers], "coverage": atlas_pooled["assigned_edge_energy_fraction"], "edge_fidelity": atlas_pooled["edge_fidelity"]},
        "C3_validated_semantic_modes_cover_material_natural_computation": {"supported": semantic_pooled["assigned_edge_energy_fraction"] >= .10, "validated_layers": [row["layer"] for row in validated_layers], "coverage": semantic_pooled["assigned_edge_energy_fraction"], "edge_fidelity": semantic_pooled["edge_fidelity"]},
        "C4_semantic_binding_beats_head_permutation": {"supported": observed_sample["edge_fidelity"] > null_p95, "observed_sample": observed_sample["edge_fidelity"], "sample_edges": observed_sample["edges"], "null_mean": float(np.mean(null_fidelity)), "null_p95": null_p95},
        "C5_route_reconstruction_predicts_contextual_write_reconstruction": {"supported": route_edge_correlation >= .70, "correlation": route_edge_correlation},
        "C6_semantic_coverage_is_not_a_single_threshold_artifact": {
            "supported": all(row["model_wide_assigned_edge_energy_fraction"] >= .05 and row["model_wide_edge_fidelity"] > 0 for row in central_sweep),
            "thresholds": {str(value): threshold_sweep["semantic_modes"][str(value)] for value in (.6, .7, .8)},
        },
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {
        "config": config, "hypotheses": hypotheses, "exact_attention_write_error": exact_error,
        "passed_semantic_modes": sorted(passed_names), "layer_results": clean_rows,
        "pooled_candidate_layers": {"atlas_modes": atlas_pooled, "semantic_modes": semantic_pooled},
        "pooled_all_layers": {"atlas_modes": all_atlas_pooled, "semantic_modes": all_semantic_pooled},
        "threshold_sweep": threshold_sweep,
        "semantic_same_panel_gram_sample": observed_sample,
        "semantic_head_permutation_null_edge_fidelity": null_fidelity,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Importance-weighted coverage of the task-agnostic route ontology", "",
        "All route modes and semantic descriptions were frozen before this disjoint FineWeb panel. Route fidelity measures the nine routing coefficients; edge fidelity measures the contextual OV vector for each source edge; write fidelity measures their summed local attention write. Assigned energy is the fraction of native per-edge squared write norm whose route lies within the frozen mode gate.", "", "## Locked hypotheses", "",
    ]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | Atlas/semantic modes | PCA route/edge/write | Centroid route/edge/write | Atlas energy/edge | Semantic energy/edge |", "|---:|---:|---:|---:|---:|---:|"]
    for row in clean_rows:
        p, c, a, s = (row[name] for name in METHODS)
        lines.append(f"| {row['layer']} | {row['atlas_mode_count']}/{row['semantic_mode_count']} | {p['route_fidelity']:.3f}/{p['edge_fidelity']:.3f}/{p['write_fidelity']:.3f} | {c['route_fidelity']:.3f}/{c['edge_fidelity']:.3f}/{c['write_fidelity']:.3f} | {a['assigned_edge_energy_fraction']:.3f}/{a['edge_fidelity']:.3f} | {s['assigned_edge_energy_fraction']:.3f}/{s['edge_fidelity']:.3f} |")
    lines += [
        "", "## Model-wide coverage and threshold sensitivity", "",
        f"At the primary `0.70` gate, geometric atlas modes contain **{all_atlas_pooled['assigned_edge_energy_fraction']:.3f}** of attention-edge energy model-wide and reconstruct **{all_atlas_pooled['edge_fidelity']:.3f}**. Prospectively validated semantic modes contain **{all_semantic_pooled['assigned_edge_energy_fraction']:.3f}** model-wide and reconstruct **{all_semantic_pooled['edge_fidelity']:.3f}**.", "",
        "| Gate | Atlas energy/fidelity model-wide | Semantic energy/fidelity model-wide |", "|---:|---:|---:|",
    ]
    for threshold in config["threshold_sweep"]:
        key = str(threshold); a = threshold_sweep["atlas_modes"][key]; s = threshold_sweep["semantic_modes"][key]
        lines.append(f"| {threshold:.2f} | {a['model_wide_assigned_edge_energy_fraction']:.3f}/{a['model_wide_edge_fidelity']:.3f} | {s['model_wide_assigned_edge_energy_fraction']:.3f}/{s['model_wide_edge_fidelity']:.3f} |")
    lines += [
        "", "## Interpretation guardrails", "",
        "Coverage is conditional on the audited attention-route representation. It does not cover MLP-only computation, QK mechanisms not expressed by the realized route, or causal interactions in later layers. A mode can contain substantial native edge energy while reconstructing it poorly, because route direction and contextual head values interact. Conversely, good local reconstruction is not yet end-to-end next-token sufficiency; that requires finite replacement through the suffix.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
