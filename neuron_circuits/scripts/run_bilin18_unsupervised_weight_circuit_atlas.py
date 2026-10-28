#!/usr/bin/env python3
"""Discover attention circuit families without using semantic labels.

The discovery object is a complete one-edge attention computation evaluated on
isotropic random states.  It includes both bilinear QK factors, relative rotary
position, the persistent layer-0 value bus, and the native OV write.  These
random-function signatures are invariant to the internal head basis and to the
QK factor rescalings/swaps that leave the computation unchanged.

Cluster count, memberships, medoids, and downstream MLP readers are frozen
before tokenizer strings or natural-text examples are decoded.  A disjoint
FineWeb panel is then used only for semantic interpretation and route evidence.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_unsupervised_weight_circuit_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_unsupervised_weight_circuit_atlas.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSUPERVISED_WEIGHT_CIRCUIT_ATLAS.md"
FIGURE = TENSOR_ROOT / "bilin18_unsupervised_weight_circuit_atlas.png"


def rotary_at_distance(x: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Apply the checkpoint's split-half rotary convention at chosen positions."""
    half = x.shape[-1] // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=x.device).float() / (2 * half)))
    angle = distance.float()[:, None] * inv[None]
    cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cosine + second * sine,
                      first * (-sine) + second * cosine), dim=-1)


@torch.inference_mode()
def random_function_signatures(model, probes: int, seed: int, distances: list[int],
                               value_mode: str = "native"):
    """Return normalized, symmetry-invariant random-function signatures."""
    device = model.embedding.weight.device
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    generator = torch.Generator(device=device).manual_seed(seed)
    states = [F.normalize(torch.randn(probes, dimension, generator=generator, device=device), dim=1)
              * math.sqrt(dimension) for _ in range(4)]
    query_state, key_state, value_state, bus_state = states
    readers = F.normalize(torch.randn(probes, dimension, generator=generator, device=device), dim=1)
    distance = torch.as_tensor(
        np.random.default_rng(seed).choice(distances, probes), device=device,
    )
    block0 = model.blocks[0].attn
    bus_value = block0.c_v(bus_state).float().reshape(probes, heads, head_dim)
    signatures = []
    norms = []
    for layer, block in enumerate(model.blocks):
        attention = block.attn
        def project(module, state):
            projected = module(state).float().reshape(probes, heads, head_dim)
            return F.rms_norm(projected, (head_dim,))
        q1 = rotary_at_distance(project(attention.c_q, query_state), distance)
        q2 = rotary_at_distance(project(attention.c_q2, query_state), distance)
        k1 = project(attention.c_k, key_state)
        k2 = project(attention.c_k2, key_state)
        route = (q1 * k1).sum(-1) * (q2 * k2).sum(-1) / (head_dim * head_dim)
        current = attention.c_v(value_state).float().reshape(probes, heads, head_dim)
        if value_mode == "native":
            mixed = (1 - attention.lamb.float()) * current + attention.lamb.float() * bus_value
        elif value_mode == "current":
            mixed = current
        elif value_mode == "bus":
            mixed = bus_value
        else:
            raise ValueError(f"unknown value_mode {value_mode}")
        projection = attention.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
        output_reader = torch.einsum("pd,hkd->phk", readers, projection)
        value_write = (mixed * output_reader).sum(-1)
        # Do not multiply the two sketches.  That unbiased estimator of the
        # complete edge-operator kernel has extreme variance because it is a
        # product of five random projections.  A head is canonically the pair
        # (address operator, value/write operator), so concatenate their
        # separately normalized invariant evaluations with equal weight.
        # Internal QK rescalings/swaps and value-head basis rotations still
        # leave this representation unchanged.
        route_signature = F.normalize(route.T, dim=1)
        write_signature = F.normalize(value_write.T, dim=1)
        signature = torch.cat((route_signature, write_signature), dim=1) / math.sqrt(2)
        norms.extend((route.T.norm(dim=1) * value_write.T.norm(dim=1)).sqrt().cpu().tolist())
        signatures.append(signature.cpu())
        print(f"weight signature layer {layer}", flush=True)
    return torch.cat(signatures), np.asarray(norms, dtype=np.float64)


def spherical_kmeans(values: torch.Tensor, clusters: int, restarts: int,
                     iterations: int, seed: int):
    values = F.normalize(values.float(), dim=1)
    best = None
    for restart in range(restarts):
        generator = torch.Generator().manual_seed(seed + 1009 * restart)
        first = int(torch.randint(len(values), (), generator=generator))
        chosen = [first]
        nearest = 1 - values @ values[first]
        for _ in range(1, clusters):
            probability = nearest.clamp_min(0).square()
            if float(probability.sum()) <= 0:
                candidate = int(torch.randint(len(values), (), generator=generator))
            else:
                candidate = int(torch.multinomial(probability, 1, generator=generator))
            chosen.append(candidate)
            nearest = torch.minimum(nearest, 1 - values @ values[candidate])
        centers = values[chosen].clone()
        assignment = None
        for _ in range(iterations):
            score = values @ centers.T
            updated_assignment = score.argmax(1)
            if assignment is not None and torch.equal(assignment, updated_assignment):
                break
            assignment = updated_assignment
            sums = torch.zeros_like(centers)
            sums.index_add_(0, assignment, values)
            empty = sums.norm(dim=1) < 1e-12
            if empty.any():
                worst = score.max(1).values.argsort()[:int(empty.sum())]
                sums[empty] = values[worst]
            centers = F.normalize(sums, dim=1)
        objective = float((values * centers[assignment]).sum())
        if best is None or objective > best[0]:
            best = objective, assignment.clone(), centers.clone()
    _, assignment, centers = best
    score = values @ centers.T
    own = score.gather(1, assignment[:, None]).squeeze(1)
    masked = score.clone(); masked[torch.arange(len(values)), assignment] = -torch.inf
    margin = own - masked.max(1).values
    return assignment.numpy(), centers, own.numpy(), margin.numpy()


def comb2(values):
    values = np.asarray(values, dtype=np.int64)
    return values * (values - 1) // 2


def adjusted_rand(first, second):
    first = np.asarray(first); second = np.asarray(second)
    contingency = np.zeros((first.max() + 1, second.max() + 1), dtype=np.int64)
    np.add.at(contingency, (first, second), 1)
    observed = comb2(contingency).sum()
    rows = comb2(contingency.sum(1)).sum(); cols = comb2(contingency.sum(0)).sum()
    total = comb2(np.asarray([len(first)]))[0]
    expected = rows * cols / max(total, 1)
    maximum = .5 * (rows + cols)
    return float((observed - expected) / max(maximum - expected, 1e-30))


def effective_support(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.sum() ** 2 / max(np.square(values).sum(), 1e-30))


def clean_token(tokenizer, index):
    return tokenizer.decode([int(index)]).replace("\n", "\\n").replace("\t", "\\t")


@torch.inference_mode()
def decode_head(model, layer, head, tokenizer, count):
    block = model.blocks[layer].attn
    dimension = model.config.n_embd; head_dim = dimension // model.config.n_head
    sl = slice(head * head_dim, (head + 1) * head_dim)
    value = block.c_v.weight[sl].float()
    output = block.c_proj.weight[:, sl].float()
    qo, ro = torch.linalg.qr(output, mode="reduced")
    qv, rv = torch.linalg.qr(value.T, mode="reduced")
    u, singular, vh = torch.linalg.svd(ro @ rv.T, full_matrices=False)
    output_mode = qo @ u[:, 0]
    input_mode = qv @ vh.T[:, 0]
    embedding = model.embedding.weight[:tokenizer.vocab_size].float()
    unembedding = model.lm_head.weight[:tokenizer.vocab_size].float()
    def tails(matrix, direction):
        score = F.normalize(matrix, dim=1) @ F.normalize(direction, dim=0)
        positive = torch.topk(score, count).indices.cpu().tolist()
        negative = torch.topk(-score, count).indices.cpu().tolist()
        return [clean_token(tokenizer, i) for i in positive], [clean_token(tokenizer, i) for i in negative]
    input_positive, input_negative = tails(embedding, input_mode)
    output_positive, output_negative = tails(unembedding, output_mode)
    return {
        "leading_singular_fraction": float(singular[0].square() / singular.square().sum()),
        "input_positive": input_positive, "input_negative": input_negative,
        "output_positive": output_positive, "output_negative": output_negative,
    }


@torch.inference_mode()
def target_readers(model, layer, head, count):
    block = model.blocks[layer]
    dimension = model.config.n_embd; head_dim = dimension // model.config.n_head
    sl = slice(head * head_dim, (head + 1) * head_dim)
    output = block.attn.c_proj.weight[:, sl].float()
    left = block.mlp.Left.weight.float() @ output
    right = block.mlp.Right.weight.float() @ output
    strength = left.square().sum(1) + right.square().sum(1)
    top = torch.topk(strength, count)
    return [{"unit": int(unit), "strength": float(value)}
            for value, unit in zip(top.values, top.indices)]


def push(heap, item, limit):
    key = item[0]
    packed = (key, item[1], item[2], item[3], item[4], item[5], item[6])
    if len(heap) < limit: heapq.heappush(heap, packed)
    elif key > heap[0][0]: heapq.heapreplace(heap, packed)


@torch.inference_mode()
def collect_route_examples(model, corpus, config):
    positions = config["semantic_positions"]
    heads = model.config.n_head; dimension = model.config.n_embd
    head_dim = dimension // heads
    heaps = {(layer, head): [] for layer in range(model.config.n_layer) for head in range(heads)}
    for start in range(0, len(corpus), config["batch_size"]):
        raw = np.asarray(corpus[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device=model.embedding.weight.device)
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
            source_write_norm = torch.einsum("bshk,hkd->bshd", details["mixed_value"].float(), projection).norm(dim=-1)
            pattern = details["pattern"][:, :, positions].abs().float().permute(0, 2, 1, 3)
            magnitude = pattern * source_write_norm.permute(0, 2, 1)[:, None]
            for head in range(heads):
                flat = magnitude[:, :, head].reshape(-1)
                amount = min(config["examples_per_head"], flat.numel())
                values, indices = torch.topk(flat, amount)
                for value, index in zip(values.cpu().tolist(), indices.cpu().tolist()):
                    batch_index, remainder = divmod(index, len(positions) * config["sequence_length"])
                    position_index, source = divmod(remainder, config["sequence_length"])
                    destination = positions[position_index]
                    if source <= destination:
                        push(heaps[(layer, head)], (float(value), start + batch_index,
                             int(source), int(destination), int(raw[batch_index, source]),
                             int(raw[batch_index, destination]),
                             float(details["pattern"][batch_index, head, destination, source])),
                             config["examples_per_head"])
            pre = mixed + details["write"]
            normalized = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        if start % (config["batch_size"] * 16) == 0:
            print(f"semantic routes {start}/{len(corpus)}", flush=True)
    return heaps


def render_example(item, corpus, tokenizer, global_start):
    magnitude, local_row, source, destination, source_id, destination_id, signed_pattern = item
    lo = max(0, source - 8); hi = min(corpus.shape[1], destination + 5)
    text = tokenizer.decode(np.asarray(corpus[local_row, lo:hi]).tolist()).replace("\n", "\\n").replace("\t", "\\t")
    return {
        "magnitude": magnitude, "row": global_start + local_row,
        "source_position": source, "destination_position": destination,
        "distance": destination - source, "source_token": clean_token(tokenizer, source_id),
        "destination_token": clean_token(tokenizer, destination_id),
        "signed_pattern": signed_pattern, "context": text,
    }


def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.float().eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    signature_a, norms = random_function_signatures(
        model, config["signature_probes"], config["seed_a"], config["relative_distances"])
    signature_b, _ = random_function_signatures(
        model, config["signature_probes"], config["seed_b"], config["relative_distances"])
    candidates = []
    for count in config["cluster_counts"]:
        a, centers_a, own_a, margin_a = spherical_kmeans(
            signature_a, count, config["kmeans_restarts"], config["kmeans_iterations"], config["seed_a"] + count)
        b, centers_b, own_b, margin_b = spherical_kmeans(
            signature_b, count, config["kmeans_restarts"], config["kmeans_iterations"], config["seed_b"] + count)
        sizes = np.bincount(a, minlength=count)
        candidates.append({
            "clusters": count, "ari": adjusted_rand(a, b),
            "mean_own_cosine": float(own_a.mean()), "mean_margin": float(margin_a.mean()),
            "minimum_size": int(sizes.min()), "assignment_a": a, "assignment_b": b,
            "centers_a": centers_a,
        })
    eligible = [row for row in candidates if row["minimum_size"] >= config["minimum_cluster_size"]]
    selected = max(eligible or candidates, key=lambda row: (row["ari"] * max(row["mean_margin"], 0), row["ari"]))

    # Treat a global taxonomy as a falsification target rather than an
    # assumption.  Local motifs are mutual nearest-neighbor edges that recur in
    # both independently sampled invariant sketches.
    gram_a = (signature_a @ signature_a.T).numpy()
    gram_b = (signature_b @ signature_b.T).numpy()
    triangle = np.triu_indices(len(signature_a), 1)
    gram_pearson = float(np.corrcoef(gram_a[triangle], gram_b[triangle])[0, 1])
    np.fill_diagonal(gram_a, -np.inf); np.fill_diagonal(gram_b, -np.inf)
    neighbor_k = config["neighbor_k"]
    neighbors_a = np.argpartition(-gram_a, neighbor_k - 1, axis=1)[:, :neighbor_k]
    neighbors_b = np.argpartition(-gram_b, neighbor_k - 1, axis=1)[:, :neighbor_k]
    sets_a = [set(row.tolist()) for row in neighbors_a]
    sets_b = [set(row.tolist()) for row in neighbors_b]
    neighbor_jaccard = np.asarray([
        len(sets_a[index] & sets_b[index]) / len(sets_a[index] | sets_b[index])
        for index in range(len(signature_a))
    ])
    graph = nx.Graph(); graph.add_nodes_from(range(len(signature_a)))
    for first in range(len(signature_a)):
        for second in sets_a[first] & sets_b[first]:
            second = int(second)
            if first in sets_a[second] and first in sets_b[second]:
                graph.add_edge(first, second,
                               score=float(min(gram_a[first, second], gram_b[first, second])))
    components = [set(component) for component in nx.connected_components(graph) if len(component) > 1]
    components.sort(key=lambda component: (
        len(component),
        np.mean([data["score"] for _, _, data in graph.subgraph(component).edges(data=True)]),
    ), reverse=True)
    components = components[:config["maximum_motifs"]]

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    source = np.load(config["corpus"], mmap_mode="r")
    corpus = source[config["semantic_start"]:config["semantic_start"] + config["semantic_rows"]]
    heaps = collect_route_examples(model, corpus, config)

    motif_rows = []
    heads = model.config.n_head
    evidence_multihead = 0; concentrated = 0
    for motif, component in enumerate(components):
        members = np.asarray(sorted(component), dtype=np.int64)
        stable_gram = np.minimum(gram_a[np.ix_(members, members)], gram_b[np.ix_(members, members)])
        medoid_index = int(members[int(stable_gram.mean(1).argmax())])
        member_pairs = [(int(index // heads), int(index % heads)) for index in members]
        medoid = (medoid_index // heads, medoid_index % heads)
        examples = []
        contributing = set()
        for index in members:
            layer, head = divmod(int(index), heads)
            for item in heaps[(layer, head)]:
                examples.append((item[0], layer, head, item))
        examples.sort(reverse=True, key=lambda row: row[0])
        # Prevent one high-scale head from monopolizing the cluster view.
        balanced, counts = [], {}
        for _, layer, head, item in examples:
            key = (layer, head)
            if counts.get(key, 0) >= 4: continue
            counts[key] = counts.get(key, 0) + 1
            contributing.add(key); balanced.append((layer, head, item))
            if len(balanced) == config["examples_per_cluster"]: break
        reader_rows = []
        for layer, head in member_pairs:
            for row in target_readers(model, layer, head, config["target_neurons_per_head"]):
                reader_rows.append({"layer": layer, "head": head, **row})
        reader_strength = np.asarray([row["strength"] for row in reader_rows])
        reader_support = effective_support(reader_strength)
        if len(contributing) >= 2: evidence_multihead += 1
        if reader_support <= .5 * len(reader_rows): concentrated += 1
        subgraph = graph.subgraph(component)
        motif_rows.append({
            "motif": motif, "size": int(len(members)),
            "layers": sorted(set(layer for layer, _ in member_pairs)),
            "members": [{"layer": layer, "head": head} for layer, head in member_pairs],
            "medoid": {"layer": medoid[0], "head": medoid[1]},
            "stable_edges": int(subgraph.number_of_edges()),
            "mean_stable_edge_cosine": float(np.mean([data["score"] for _, _, data in subgraph.edges(data=True)])),
            "minimum_stable_edge_cosine": float(min(data["score"] for _, _, data in subgraph.edges(data=True))),
            "medoid_decode": decode_head(model, *medoid, tokenizer, config["decode_tokens"]),
            "route_contributing_heads": len(contributing),
            "route_examples": [
                {"layer": layer, "head": head,
                 **render_example(item, corpus, tokenizer, config["semantic_start"])}
                for layer, head, item in balanced
            ],
            "reader_effective_support": reader_support,
            "top_readers": sorted(reader_rows, key=lambda row: row["strength"], reverse=True)[:24],
        })
    candidates_json = [{key: value for key, value in row.items()
                        if key not in ("assignment_a", "assignment_b", "centers_a")}
                       for row in candidates]
    cross_layer = sum(len(row["layers"]) >= 3 for row in motif_rows)
    best_global_ari = max(row["ari"] for row in candidates)
    hypotheses = {
        "U1_pairwise_geometry_replicates": {"supported": gram_pearson >= .75,
            "pairwise_similarity_pearson": gram_pearson},
        "U2_global_clusters_are_stable": {"supported": best_global_ari >= .50,
            "best_adjusted_rand": best_global_ari},
        "U3_local_neighborhoods_replicate": {"supported": float(np.median(neighbor_jaccard)) >= .25,
            "median_neighbor_jaccard": float(np.median(neighbor_jaccard)),
            "mean_neighbor_jaccard": float(neighbor_jaccard.mean()),
            "stable_mutual_edges": int(graph.number_of_edges()),
            "isolated_heads": int(len(list(nx.isolates(graph))))},
        "U4_local_motifs_cross_layers": {"supported": cross_layer >= 1,
            "cross_layer_motifs": cross_layer},
        "U5_route_evidence_is_not_single_head": {"supported": evidence_multihead >= math.ceil(len(motif_rows) / 2),
            "multihead_motifs": evidence_multihead, "motifs": len(motif_rows)},
        "U6_reader_evidence_is_concentrated": {"supported": concentrated >= math.ceil(len(motif_rows) / 2),
            "concentrated_motifs": concentrated, "motifs": len(motif_rows)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {
        "config": config, "method": {
            "discovery": "paired isotropic random-function sketches of each head's QK1*QK2 address and OV write operators",
            "semantic_blinding": "global-partition tests and local mutual-neighbor motifs frozen before tokenizer/natural-text decoding",
            "semantic_panel": f"FineWeb rows {config['semantic_start']}..{config['semantic_start'] + config['semantic_rows'] - 1}",
            "recursive_generalization_status": "finished-model proxy only; no training checkpoints available",
        },
        "candidate_partitions": candidates_json,
        "best_global_partition": {key: value for key, value in selected.items()
                               if key not in ("assignment_a", "assignment_b", "centers_a")},
        "local_motif_discovery": {
            "neighbor_k": neighbor_k, "pairwise_similarity_pearson": gram_pearson,
            "median_neighbor_jaccard": float(np.median(neighbor_jaccard)),
            "mean_neighbor_jaccard": float(neighbor_jaccard.mean()),
            "stable_mutual_edges": int(graph.number_of_edges()),
            "isolated_heads": int(len(list(nx.isolates(graph)))),
            "nontrivial_components_before_cap": int(sum(len(c) > 1 for c in nx.connected_components(graph))),
        },
        "signature_norm_quantiles": {str(q): float(np.quantile(norms, q)) for q in (.0, .1, .5, .9, 1.)},
        "hypotheses": hypotheses, "motifs": motif_rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = ["# Label-blind weight-circuit atlas", "",
             "Global-cluster tests and local motifs were frozen from symmetry-invariant random-function evaluations of each native attention head before any token strings or FineWeb contexts were decoded.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Partition", "",
              "| k | sketch ARI | own-centroid cosine | centroid advantage | minimum size |",
              "|---:|---:|---:|---:|---:|"]
    for row in candidates_json:
        lines.append(f"| {row['clusters']} | {row['ari']:.3f} | {row['mean_own_cosine']:.3f} | {row['mean_margin']:.3f} | {row['minimum_size']} |")
    lines += ["", f"No global partition passed stability (best ARI `{best_global_ari:.3f}`). The frozen objects below are connected components of top-{neighbor_k} mutual-neighbor edges recurring in both sketches.", "",
              "| motif | size | layers | medoid | stable edges | heads in natural evidence | reader support | input-mode token hints |",
              "|---:|---:|---|---|---:|---:|---:|---|"]
    for row in motif_rows:
        hint = ", ".join(f"`{x}`" for x in row["medoid_decode"]["input_positive"][:5])
        layers = ",".join(map(str, row["layers"]))
        lines.append(f"| {row['motif']} | {row['size']} | {layers} | L{row['medoid']['layer']}H{row['medoid']['head']} | {row['stable_edges']} | {row['route_contributing_heads']} | {row['reader_effective_support']:.1f} | {hint} |")
    lines += ["", "## Interpretation guardrails", "",
              "- Token hints come from only the medoid's leading OV mode; they do not name the whole cluster.",
              "- High-energy natural edges show invocation, not task-level necessity. Finite edge removal is still required.",
              "- A recursive-generalization reading must identify a local semantic bridge among apparent branches and predict held-out contexts/readers. A broad label invented after viewing all examples is not a pass.",
              "- Without training checkpoints, branch-growth order cannot be established; positive results are only consistent with recursive generalization."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot([r["clusters"] for r in candidates_json], [r["ari"] for r in candidates_json], "o-", label="sketch ARI")
    axes[0].plot([r["clusters"] for r in candidates_json], [r["mean_margin"] for r in candidates_json], "o-", label="centroid advantage")
    axes[0].legend(); axes[0].set_xlabel("clusters"); axes[0].set_title("Global partitions fail stability")
    size = [r["size"] for r in motif_rows]; depth = [len(r["layers"]) for r in motif_rows]
    axes[1].scatter(size, depth, c=[r["mean_stable_edge_cosine"] for r in motif_rows], cmap="viridis")
    axes[1].set_xlabel("native heads"); axes[1].set_ylabel("distinct layers"); axes[1].set_title("Frozen circuit-family breadth")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"best_global": payload["best_global_partition"], "local": payload["local_motif_discovery"], "hypotheses": hypotheses}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
