#!/usr/bin/env python3
"""Discover prediction-relevant route structure outside known semantic modes."""

from __future__ import annotations

import heapq
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unseeded_route_coalition_geometry import best_center_matching
from run_bilin18_unsupervised_weight_circuit_atlas import spherical_kmeans


CONFIG = Path("configs/bilin18_ce_adjoint_residual_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_ce_adjoint_residual_atlas.json"
ARRAYS = TENSOR_ROOT / "bilin18_ce_adjoint_residual_atlas.npz"
REPORT = TENSOR_ROOT / "BILIN18_CE_ADJOINT_RESIDUAL_ATLAS.md"


def semantic_reconstruction(routes, centers, threshold):
    if centers is None:
        return torch.zeros_like(routes), torch.zeros(routes.shape[:-1], dtype=torch.bool, device=routes.device)
    norm = routes.norm(dim=-1, keepdim=True); unit = routes / norm.clamp_min(1e-30)
    similarity = unit @ centers.T; assignment = similarity.argmax(-1)
    selected = similarity.max(-1).values >= threshold
    return torch.where(selected[..., None], norm * centers[assignment], 0), selected


def fit_basis(values, rank):
    _, _, vh = np.linalg.svd(np.asarray(values, dtype=np.float64), full_matrices=False)
    return vh[:rank]


def subspace_cosine(first, second):
    return float(np.linalg.svd(np.asarray(first) @ np.asarray(second).T, compute_uv=False).mean())


def reconstruction_fidelity(values, reconstructed):
    values = np.asarray(values, dtype=np.float64); reconstructed = np.asarray(reconstructed, dtype=np.float64)
    return float(1 - np.square(values - reconstructed).sum() / max(np.square(values).sum(), 1e-30))


def push(heap, score, serial, record, limit):
    item = (float(score), serial, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def differentiable_attention_parts(attention, normalized, value_bus, config, tt):
    """Checkpoint-exact attention pieces without the inference-only wrapper."""
    batch, length, dimension = normalized.shape
    heads, head_dim = config.n_head, dimension // config.n_head

    def projected(module):
        value = module(normalized).view(batch, length, heads, head_dim)
        value = F.rms_norm(value, (head_dim,))
        cosine, sine = attention.rotary(value)
        return tt.apply_rotary_emb(value, cosine, sine)

    q, k = projected(attention.c_q), projected(attention.c_k)
    q2, k2 = projected(attention.c_q2), projected(attention.c_k2)
    current_value = attention.c_v(normalized).view(batch, length, heads, head_dim)
    first_value = current_value if value_bus is None else value_bus.view_as(current_value)
    mixed_value = (1 - attention.lamb) * current_value + attention.lamb * first_value
    score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / head_dim
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head_dim
    causal = torch.tril(torch.ones(length, length, device=normalized.device, dtype=torch.bool))
    pattern = (score1 * score2).masked_fill(~causal, 0)
    return pattern, mixed_value, first_value


def forward_with_route_leaves(model, ids):
    """Treat every realized route tensor as an independent local variable.

    Downstream values and MLP states recompute, but downstream QK routes are frozen
    at their native values. This is a frozen-route CE adjoint, not a total
    derivative through every later QK calculation.
    """
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,)); x0 = x; bus = None; leaves = []
    for block in model.blocks:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        native_pattern, mixed_value, bus = differentiable_attention_parts(
            block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt,
        )
        pattern = native_pattern.detach().requires_grad_(True); leaves.append(pattern)
        attended = torch.einsum("bhqk,bkhd->bhqd", pattern, mixed_value)
        attended = attended.transpose(1, 2).contiguous().view(ids.shape[0], ids.shape[1], dimension)
        write = block.attn.c_proj(attended)
        pre = mixed + write; normalized = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30), leaves


def collect_panel(model, corpus, global_start, config, centers):
    layers = model.config.n_layer
    heaps = [[] for _ in range(layers)]; serial = 0
    summary = [{"full_square": 0.0, "residual_square": 0.0, "absolute_full": 0.0, "absolute_semantic": 0.0, "absolute_residual": 0.0, "members": 0, "edges": 0} for _ in range(layers)]
    maximum_additive_error = 0.0
    for destination in config["destinations"]:
        length = destination + 1
        for start in range(0, len(corpus), config["batch_size"]):
            raw = np.asarray(corpus[start:start + config["batch_size"], :length], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda")
            targets = torch.as_tensor(np.asarray(corpus[start:start + len(raw), length], dtype=np.int64), device="cuda")
            logits, leaves = forward_with_route_leaves(model, ids)
            loss = F.cross_entropy(logits, targets, reduction="sum"); loss.backward()
            for layer, pattern in enumerate(leaves):
                routes = pattern[:, :, destination, :destination + 1].detach().float().permute(0, 2, 1)
                gradient = pattern.grad[:, :, destination, :destination + 1].detach().float().permute(0, 2, 1)
                semantic, selected = semantic_reconstruction(routes, centers[layer], config["semantic_mode_cosine_threshold"])
                residual = routes - semantic
                full_effect = (routes * gradient).sum(-1)
                semantic_effect = (semantic * gradient).sum(-1)
                residual_effect = (residual * gradient).sum(-1)
                maximum_additive_error = max(maximum_additive_error, float((full_effect - semantic_effect - residual_effect).abs().max()))
                row = summary[layer]
                row["full_square"] += float(full_effect.square().sum()); row["residual_square"] += float(residual_effect.square().sum())
                row["absolute_full"] += float(full_effect.abs().sum()); row["absolute_semantic"] += float(semantic_effect.abs().sum()); row["absolute_residual"] += float(residual_effect.abs().sum())
                row["members"] += int(selected.sum()); row["edges"] += selected.numel()
                score = residual_effect.abs()
                amount = min(config["retained_edges_per_layer"], score.numel())
                values, indices = torch.topk(score.reshape(-1), amount)
                for value, flat in zip(values.cpu().tolist(), indices.cpu().tolist()):
                    batch_index, source = divmod(flat, destination + 1)
                    vector = residual[batch_index, source]; vector_norm = vector.norm().clamp_min(1e-30)
                    route = routes[batch_index, source]; route_norm = route.norm().clamp_min(1e-30)
                    record = {
                        "residual_route": (vector / vector_norm).cpu().tolist(),
                        "native_route": (route / route_norm).cpu().tolist(),
                        "gradient": F.normalize(gradient[batch_index, source], dim=0).cpu().tolist(),
                        "global_row": global_start + start + batch_index, "destination": destination, "source": source,
                        "distance": destination - source, "source_token_id": int(raw[batch_index, source]),
                        "destination_token_id": int(raw[batch_index, destination]), "next_token_id": int(corpus[start + batch_index, length]),
                        "residual_ce_effect": float(residual_effect[batch_index, source]), "native_ce_effect": float(full_effect[batch_index, source]),
                        "semantic_ce_effect": float(semantic_effect[batch_index, source]), "semantic_member": bool(selected[batch_index, source]),
                    }
                    push(heaps[layer], value, serial, record, config["retained_edges_per_layer"]); serial += 1
            del logits, leaves, loss
            if start % 32 == 0:
                print(f"CE residual panel {global_start} d={destination} rows {start}/{len(corpus)}", flush=True)
    results = [[item[-1] for item in sorted(heap, reverse=True)] for heap in heaps]
    for row in summary:
        row["semantic_effect_fidelity"] = 1 - row["residual_square"] / max(row["full_square"], 1e-30)
        row["absolute_semantic_ratio"] = row["absolute_semantic"] / max(row["absolute_full"], 1e-30)
        row["absolute_residual_ratio"] = row["absolute_residual"] / max(row["absolute_full"], 1e-30)
        row["member_fraction"] = row["members"] / max(row["edges"], 1)
    return results, summary, maximum_additive_error


def main():
    config = json.loads(CONFIG.read_text()); semantic = json.loads(Path(config["semantic_source"]).read_text()); validation = json.loads(Path(config["semantic_validation"]).read_text())
    passed = {row["candidate"] for name, row in validation["hypotheses"].items() if name.startswith("S") and name[1:2].isdigit() and int(name[1:].split("_", 1)[0]) < 10 and row["supported"]}
    centers_np = {layer: [] for layer in range(18)}
    for row in semantic["results"]:
        if row["name"] in passed: centers_np[row["layer"]].append(row["center"])
    centers = {layer: torch.as_tensor(np.asarray(value, dtype=np.float32), device="cuda") if value else None for layer, value in centers_np.items()}
    source = np.load(config["corpus"], mmap_mode="r")
    discovery_corpus = source[config["discovery_start"]:config["discovery_start"] + config["rows_per_panel"]]
    replication_corpus = source[config["replication_start"]:config["replication_start"] + config["rows_per_panel"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    discovery, discovery_summary, error_a = collect_panel(model, discovery_corpus, config["discovery_start"], config, centers)
    replication, replication_summary, error_b = collect_panel(model, replication_corpus, config["replication_start"], config, centers)

    layer_results = []; discovery_bases = {}; replication_bases = {}
    for layer in range(18):
        first = np.asarray([row["residual_route"] for row in discovery[layer]], dtype=np.float32)
        second = np.asarray([row["residual_route"] for row in replication[layer]], dtype=np.float32)
        first_basis = fit_basis(first, config["residual_rank"]); second_basis = fit_basis(second, config["residual_rank"])
        discovery_bases[layer] = first_basis; replication_bases[layer] = second_basis
        subspace = subspace_cosine(first_basis, second_basis)
        pca_reconstruction = (second @ first_basis.T) @ first_basis
        pca_fidelity = reconstruction_fidelity(second, pca_reconstruction)
        assignment_a, centers_a, _, _ = spherical_kmeans(torch.as_tensor(first), config["clusters"], config["kmeans_restarts"], config["kmeans_iterations"], config["seed"] + layer)
        assignment_b, centers_b, _, _ = spherical_kmeans(torch.as_tensor(second), config["clusters"], config["kmeans_restarts"], config["kmeans_iterations"], config["seed"] + 1000 + layer)
        matching = best_center_matching(centers_a.numpy(), centers_b.numpy())
        assigned = (second @ centers_a.numpy().T).argmax(1); centroid_reconstruction = centers_a.numpy()[assigned]
        centroid_fidelity = reconstruction_fidelity(second, centroid_reconstruction)
        layer_results.append({
            "layer": layer, "subspace_cosine": subspace, "matched_centroid_cosine": matching["mean_cosine"],
            "pca_fidelity": pca_fidelity, "centroid_fidelity": centroid_fidelity,
            "discovery_distinct_sources": len(set(row["source_token_id"] for row in discovery[layer])),
            "replication_distinct_sources": len(set(row["source_token_id"] for row in replication[layer])),
            "discovery_semantic_effect_fidelity": discovery_summary[layer]["semantic_effect_fidelity"],
            "replication_semantic_effect_fidelity": replication_summary[layer]["semantic_effect_fidelity"],
        })
        print(f"CE residual L{layer} subspace={subspace:.3f} pca={pca_fidelity:.3f} centroid={centroid_fidelity:.3f}", flush=True)

    pooled = {}
    for name, rows in (("discovery", discovery_summary), ("replication", replication_summary)):
        full = sum(row["full_square"] for row in rows); residual = sum(row["residual_square"] for row in rows)
        pooled[name] = {"semantic_effect_fidelity": 1 - residual / max(full, 1e-30), "full_square": full, "residual_square": residual}
    rng = np.random.default_rng(config["seed"]); null_means = []
    for _ in range(config["null_replicates"]):
        values = []
        for layer in range(18):
            second = np.asarray([row["residual_route"] for row in replication[layer]])
            shuffled = np.stack([row[rng.permutation(model.config.n_head)] for row in second])
            null_basis = fit_basis(shuffled, config["residual_rank"])
            values.append(subspace_cosine(discovery_bases[layer], null_basis))
        null_means.append(float(np.mean(values)))
    recurrent = [row["layer"] for row in layer_results if row["subspace_cosine"] >= .80]
    pca_wins = [row["layer"] for row in layer_results if row["pca_fidelity"] > row["centroid_fidelity"]]
    diverse = [row["layer"] for row in layer_results if min(row["discovery_distinct_sources"], row["replication_distinct_sources"]) >= 32]
    true_mean = float(np.mean([row["subspace_cosine"] for row in layer_results])); null_p95 = float(np.quantile(null_means, .95))
    hypotheses = {
        "A0_selection_is_task_and_label_blind": {"supported": config["discovery_start"] + config["rows_per_panel"] <= config["replication_start"], "ranges": [[config["discovery_start"], config["discovery_start"] + config["rows_per_panel"] - 1], [config["replication_start"], config["replication_start"] + config["rows_per_panel"] - 1]]},
        "A1_known_semantic_modes_leave_most_ce_sensitivity_unexplained": {"supported": all(row["semantic_effect_fidelity"] < .50 for row in pooled.values()), "panels": pooled},
        "A2_ce_important_residual_subspaces_recur": {"supported": len(recurrent) >= 12, "layers": recurrent},
        "A3_recurrence_beats_destroyed_head_binding": {"supported": true_mean > null_p95, "observed": true_mean, "null_mean": float(np.mean(null_means)), "null_p95": null_p95},
        "A4_residual_geometry_is_better_continuous_than_discrete": {"supported": len(pca_wins) >= 12, "layers": pca_wins},
        "A5_residual_edges_are_not_token_lookup_artifacts": {"supported": len(diverse) == 18, "layers": diverse},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "maximum_additive_error": max(error_a, error_b), "layer_results": layer_results, "discovery_summary": discovery_summary, "replication_summary": replication_summary, "null_mean_subspace_cosine": null_means}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    arrays = {}
    for panel_name, panel in (("discovery", discovery), ("replication", replication)):
        for layer, rows in enumerate(panel):
            arrays[f"{panel_name}_residual_routes_L{layer}"] = np.asarray([row["residual_route"] for row in rows], dtype=np.float32)
            arrays[f"{panel_name}_native_routes_L{layer}"] = np.asarray([row["native_route"] for row in rows], dtype=np.float32)
            arrays[f"{panel_name}_gradients_L{layer}"] = np.asarray([row["gradient"] for row in rows], dtype=np.float32)
            arrays[f"{panel_name}_metadata_L{layer}"] = np.asarray([[row[key] for key in ("global_row", "destination", "source", "distance", "source_token_id", "destination_token_id", "next_token_id", "semantic_member")] for row in rows], dtype=np.int64)
            arrays[f"{panel_name}_effects_L{layer}"] = np.asarray([[row[key] for key in ("native_ce_effect", "semantic_ce_effect", "residual_ce_effect")] for row in rows], dtype=np.float32)
    np.savez_compressed(ARRAYS, **arrays)
    lines = ["# CE-adjoint residual route atlas", "", "Every realized source route is treated as an independent local variable while downstream values and MLP states recompute and downstream QK routes stay frozen. Previously validated semantic modes are subtracted. The highest absolute residual next-token CE effects are retained without token labels on two disjoint FineWeb panels.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| L | Semantic CE fidelity D/R | Residual subspace cosine | k4 center cosine | PCA/k4 reconstruction | Distinct sources D/R |", "|---:|---:|---:|---:|---:|---:|"]
    for row in layer_results:
        lines.append(f"| {row['layer']} | {row['discovery_semantic_effect_fidelity']:.3f}/{row['replication_semantic_effect_fidelity']:.3f} | {row['subspace_cosine']:.3f} | {row['matched_centroid_cosine']:.3f} | {row['pca_fidelity']:.3f}/{row['centroid_fidelity']:.3f} | {row['discovery_distinct_sources']}/{row['replication_distinct_sources']} |")
    lines += ["", "## Guardrails", "", "The adjoint is local to the realized route variables and freezes downstream QK routes; it is not a finite deletion and not an additive attribution of total loss across layers. Top-residual selection conditions the atlas on large first-order effects. Recurrence and null comparisons test structure in that selected residual, not the fraction of the entire model explained."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
