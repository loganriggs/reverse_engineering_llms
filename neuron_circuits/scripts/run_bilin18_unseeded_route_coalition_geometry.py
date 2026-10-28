#!/usr/bin/env python3
"""Discover natural nine-head route coalitions without tasks or token labels."""

from __future__ import annotations

import heapq
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_unsupervised_weight_circuit_atlas import spherical_kmeans


CONFIG = Path("configs/bilin18_unseeded_route_coalition_geometry.json")
OUTPUT = TENSOR_ROOT / "bilin18_unseeded_route_coalition_geometry.json"
ARRAYS = TENSOR_ROOT / "bilin18_unseeded_route_coalition_geometry.npz"
REPORT = TENSOR_ROOT / "BILIN18_UNSEEDED_ROUTE_COALITION_GEOMETRY.md"


def effective_support(vector):
    vector = np.abs(np.asarray(vector, dtype=np.float64))
    return float(vector.sum() ** 2 / max(np.square(vector).sum(), 1e-30))


def best_center_matching(first, second):
    similarity = np.asarray(first) @ np.asarray(second).T
    best = None
    for permutation in itertools.permutations(range(len(second))):
        values = np.asarray([similarity[index, permutation[index]] for index in range(len(first))])
        score = float(values.mean())
        if best is None or score > best[0]:
            best = score, permutation, values
    score, permutation, values = best
    return {"mean_cosine": score, "permutation": list(permutation), "cosines": values.tolist()}


def push(heap, score, serial, record, limit):
    item = (float(score), serial, record)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


@torch.inference_mode()
def collect_panel(model, corpus, global_start, config):
    layers = model.config.n_layer; heads = model.config.n_head; dimension = model.config.n_embd
    head_dim = dimension // heads; destinations = torch.as_tensor(config["destinations"], device="cuda")
    heaps = [[] for _ in range(layers)]; serial = 0
    for start in range(0, len(corpus), config["batch_size"]):
        raw = np.asarray(corpus[start:start + config["batch_size"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            pattern = details["pattern"][:, :, destinations].float().permute(0, 2, 3, 1)
            projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
            source_writes = torch.einsum("bshd,hdk->bshk", details["mixed_value"].float(), projection)
            write_norm = source_writes.norm(dim=-1)
            approximate = (pattern.square() * write_norm[:, None].square()).sum(-1).sqrt()
            causal = torch.arange(raw.shape[1], device="cuda")[None, None, :] <= destinations[None, :, None]
            approximate = approximate.masked_fill(~causal, -torch.inf)
            amount = min(config["batch_candidates_per_layer"], approximate.numel())
            scores, indices = torch.topk(approximate.reshape(-1), amount)
            for score, flat_index in zip(scores.cpu().tolist(), indices.cpu().tolist()):
                batch_index, remainder = divmod(flat_index, len(config["destinations"]) * raw.shape[1])
                destination_index, source = divmod(remainder, raw.shape[1]); destination = config["destinations"][destination_index]
                route = pattern[batch_index, destination_index, source]
                route_norm = route.norm().clamp_min(1e-30)
                record = {
                    "route": (route / route_norm).cpu().tolist(), "raw_route_norm": float(route_norm),
                    "global_row": global_start + start + batch_index,
                    "source": source, "destination": destination, "distance": destination - source,
                    "source_token_id": int(raw[batch_index, source]),
                    "destination_token_id": int(raw[batch_index, destination]),
                    "next_token_id": int(raw[batch_index, destination + 1]),
                    "source_equals_next": bool(raw[batch_index, source] == raw[batch_index, destination + 1]),
                    "source_equals_destination": bool(raw[batch_index, source] == raw[batch_index, destination]),
                    "magnitude_proxy": score,
                }
                push(heaps[layer], score, serial, record, config["retained_edges_per_layer"]); serial += 1
            pre = mixed + details["write"]
            normalized = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        if start % 64 == 0: print(f"unseeded panel {global_start} rows {start}/{len(corpus)}", flush=True)
    return [[item[-1] for item in sorted(heap, reverse=True)] for heap in heaps]


def cluster_panel(rows, clusters, restarts, iterations, seed):
    values = torch.as_tensor([row["route"] for row in rows], dtype=torch.float32)
    assignment, centers, own, margin = spherical_kmeans(values, clusters, restarts, iterations, seed)
    return {
        "values": values.numpy(), "assignment": assignment, "centers": centers.numpy(),
        "sizes": np.bincount(assignment, minlength=clusters),
        "own": own, "margin": margin,
    }


def main():
    config = json.loads(CONFIG.read_text()); source = np.load(config["corpus"], mmap_mode="r")
    discovery_corpus = source[config["discovery_start"]:config["discovery_start"] + config["rows_per_panel"], :config["sequence_length"]]
    replication_corpus = source[config["replication_start"]:config["replication_start"] + config["rows_per_panel"], :config["sequence_length"]]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    discovery = collect_panel(model, discovery_corpus, config["discovery_start"], config)
    replication = collect_panel(model, replication_corpus, config["replication_start"], config)
    layer_rows = []; clusters_by_layer = {}
    for layer in range(model.config.n_layer):
        first = cluster_panel(discovery[layer], config["clusters_per_layer"], config["kmeans_restarts"], config["kmeans_iterations"], config["seed"] + 101 * layer)
        second = cluster_panel(replication[layer], config["clusters_per_layer"], config["kmeans_restarts"], config["kmeans_iterations"], config["seed"] + 10000 + 101 * layer)
        matching = best_center_matching(first["centers"], second["centers"])
        matched_second = np.asarray(matching["permutation"])
        candidates = []
        for cluster in range(config["clusters_per_layer"]):
            other = int(matched_second[cluster])
            first_members = first["assignment"] == cluster; second_members = second["assignment"] == other
            center = first["centers"][cluster]
            candidates.append({
                "layer": layer, "cluster": cluster, "replication_cluster": other,
                "center": center.tolist(), "center_cosine": matching["cosines"][cluster],
                "discovery_size": int(first_members.sum()), "replication_size": int(second_members.sum()),
                "discovery_member_cosine": float(first["own"][first_members].mean()),
                "replication_member_cosine": float(second["own"][second_members].mean()),
                "effective_support": effective_support(center),
                "discovery_margin": float(first["margin"][first_members].mean()),
                "replication_margin": float(second["margin"][second_members].mean()),
            })
        clusters_by_layer[str(layer)] = candidates
        layer_rows.append({"layer": layer, "mean_center_cosine": matching["mean_cosine"], "cluster_cosines": matching["cosines"]})
        print(f"route clusters L{layer} cosine={matching['mean_cosine']:.3f}", flush=True)

    # Null independently permutes head identities for every replication edge,
    # destroying coalition binding while preserving each edge's coefficient multiset.
    rng = np.random.default_rng(config["seed"] + 99); null = []
    for replicate in range(config["null_replicates"]):
        layer_scores = []
        for layer in range(model.config.n_layer):
            values = np.asarray([row["route"] for row in replication[layer]])
            shuffled = np.stack([row[rng.permutation(model.config.n_head)] for row in values])
            assignment, centers, _, _ = spherical_kmeans(
                torch.as_tensor(shuffled, dtype=torch.float32), config["clusters_per_layer"],
                3, config["kmeans_iterations"], config["seed"] + 100000 * replicate + layer,
            )
            first_centers = np.asarray([row["center"] for row in clusters_by_layer[str(layer)]])
            layer_scores.append(best_center_matching(first_centers, centers.numpy())["mean_cosine"])
        null.append(float(np.mean(layer_scores)))

    eligible = []
    for rows in clusters_by_layer.values():
        for row in rows:
            if (row["center_cosine"] >= config["minimum_center_cosine"]
                    and min(row["discovery_size"], row["replication_size"]) >= config["minimum_cluster_size"]
                    and min(row["discovery_member_cosine"], row["replication_member_cosine"]) >= config["minimum_member_cosine"]):
                row = dict(row)
                row["selection_score"] = row["center_cosine"] * min(row["discovery_member_cosine"], row["replication_member_cosine"])
                eligible.append(row)
    eligible.sort(key=lambda row: row["selection_score"], reverse=True)
    selected = eligible[:config["maximum_candidates"]]
    real_mean = float(np.mean([row["mean_center_cosine"] for row in layer_rows])); null_p95 = float(np.quantile(null, .95))
    stable_layers = [row["layer"] for row in layer_rows if row["mean_center_cosine"] >= config["minimum_center_cosine"]]
    hypotheses = {
        "U0_two_disjoint_panels_are_used": {"supported": config["discovery_start"] + config["rows_per_panel"] <= config["replication_start"], "discovery_range": [config["discovery_start"], config["discovery_start"] + config["rows_per_panel"] - 1], "replication_range": [config["replication_start"], config["replication_start"] + config["rows_per_panel"] - 1]},
        "U1_discrete_route_coalitions_replicate": {"supported": len(stable_layers) >= 8, "stable_layers": stable_layers, "mean_cosines": {str(row["layer"]): row["mean_center_cosine"] for row in layer_rows}},
        "U2_recurrence_beats_destroyed_head_binding": {"supported": real_mean > null_p95, "real_mean": real_mean, "null_mean": float(np.mean(null)), "null_p95": null_p95},
        "U3_stable_candidates_exist": {"supported": len(eligible) >= 12, "eligible": len(eligible), "selected": len(selected)},
        "U4_stable_candidates_are_not_only_single_heads": {"supported": sum(row["effective_support"] >= 2 for row in selected) >= len(selected) / 2, "multihead": sum(row["effective_support"] >= 2 for row in selected), "selected": len(selected)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "layer_summary": layer_rows, "null_mean_center_cosine": null, "clusters": clusters_by_layer, "selected_candidates": selected}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    arrays = {}
    for panel_name, panel in (("discovery", discovery), ("replication", replication)):
        for layer, rows in enumerate(panel):
            arrays[f"{panel_name}_routes_L{layer}"] = np.asarray([row["route"] for row in rows], dtype=np.float32)
            arrays[f"{panel_name}_metadata_L{layer}"] = np.asarray([[row[key] for key in ("global_row", "source", "destination", "distance", "source_token_id", "destination_token_id", "next_token_id", "source_equals_next", "source_equals_destination")] for row in rows], dtype=np.int64)
    np.savez_compressed(ARRAYS, **arrays)
    lines = ["# Unseeded natural route-coalition geometry", "", "High-magnitude natural attention edges are represented only by their normalized signed nine-head route vectors. Four-cluster layer-local partitions are fit independently on two disjoint FineWeb panels and matched without decoding token strings. Per-edge head permutations provide a destroyed-binding null.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Layer | Mean matched cosine | Cluster cosines |", "|---:|---:|---|"]
    for row in layer_rows:
        lines.append(f"| {row['layer']} | {row['mean_center_cosine']:.3f} | " + ", ".join(f"{x:.3f}" for x in row["cluster_cosines"]) + " |")
    lines += ["", "## Frozen candidates for semantic discovery", "", "| Rank | Layer/cluster | Center cosine | D/R size | D/R coherence | Effective heads |", "|---:|---|---:|---:|---:|---:|"]
    for index, row in enumerate(selected, 1):
        lines.append(f"| {index} | L{row['layer']}C{row['cluster']} | {row['center_cosine']:.3f} | {row['discovery_size']}/{row['replication_size']} | {row['discovery_member_cosine']:.3f}/{row['replication_member_cosine']:.3f} | {row['effective_support']:.2f} |")
    lines += ["", "## Guardrail", "", "These are input-conditioned route-coordinate clusters, not semantic features. Candidate selection used no decoded tokens, but the high-magnitude edge sampler and fixed k=4 are part of the ontology. If clusters fail recurrence or the destroyed-binding null, a continuous routing-field description should be preferred."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
