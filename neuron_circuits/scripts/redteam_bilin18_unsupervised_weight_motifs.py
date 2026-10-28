#!/usr/bin/env python3
"""Red-team label-blind head motifs for architectural value-bus confounds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from run_bilin18_unsupervised_weight_circuit_atlas import (
    CONFIG, load_full_model, random_function_signatures,
)
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


OUTPUT = TENSOR_ROOT / "bilin18_unsupervised_weight_motif_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSUPERVISED_WEIGHT_MOTIF_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_unsupervised_weight_motif_redteam.png"


def graph_from(first, second, k):
    gram_a = (first @ first.T).numpy(); gram_b = (second @ second.T).numpy()
    triangle = np.triu_indices(len(first), 1)
    correlation = float(np.corrcoef(gram_a[triangle], gram_b[triangle])[0, 1])
    np.fill_diagonal(gram_a, -np.inf); np.fill_diagonal(gram_b, -np.inf)
    near_a = np.argpartition(-gram_a, k - 1, axis=1)[:, :k]
    near_b = np.argpartition(-gram_b, k - 1, axis=1)[:, :k]
    sets_a = [set(row.tolist()) for row in near_a]; sets_b = [set(row.tolist()) for row in near_b]
    graph = nx.Graph(); graph.add_nodes_from(range(len(first)))
    for i in range(len(first)):
        for j in sets_a[i] & sets_b[i]:
            j = int(j)
            if i in sets_a[j] and i in sets_b[j]:
                graph.add_edge(i, j, score=float(min(gram_a[i, j], gram_b[i, j])))
    neighbor_jaccard = [len(sets_a[i] & sets_b[i]) / len(sets_a[i] | sets_b[i]) for i in range(len(first))]
    return graph, correlation, float(np.median(neighbor_jaccard))


def edge_set(graph):
    return {tuple(sorted(edge)) for edge in graph.edges}


def edge_jaccard(first, second):
    first, second = edge_set(first), edge_set(second)
    return float(len(first & second) / max(len(first | second), 1))


def structure(graph, heads):
    edges = list(graph.edges)
    return {
        "edges": len(edges),
        "same_head_index_fraction": float(np.mean([a % heads == b % heads for a, b in edges])) if edges else 0.,
        "same_layer_fraction": float(np.mean([a // heads == b // heads for a, b in edges])) if edges else 0.,
        "adjacent_layer_fraction": float(np.mean([abs(a // heads - b // heads) == 1 for a, b in edges])) if edges else 0.,
        "isolated_heads": len(list(nx.isolates(graph))),
        "component_sizes": sorted([len(c) for c in nx.connected_components(graph) if len(c) > 1], reverse=True),
    }


def main():
    config = json.loads(CONFIG.read_text())
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.float().eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    signatures = {}
    for mode in ("native", "current", "bus"):
        for name, seed in (("a", config["seed_a"]), ("b", config["seed_b"])):
            value, _ = random_function_signatures(
                model, config["signature_probes"], seed,
                config["relative_distances"], value_mode=mode,
            )
            signatures[(mode, name)] = value
    probes = config["signature_probes"]
    representations = {
        "native_joint": (signatures[("native", "a")], signatures[("native", "b")]),
        "address_only": (signatures[("native", "a")][:, :probes], signatures[("native", "b")][:, :probes]),
        "native_write_only": (signatures[("native", "a")][:, probes:], signatures[("native", "b")][:, probes:]),
        "current_joint": (signatures[("current", "a")], signatures[("current", "b")]),
        "current_write_only": (signatures[("current", "a")][:, probes:], signatures[("current", "b")][:, probes:]),
        "bus_write_only": (signatures[("bus", "a")][:, probes:], signatures[("bus", "b")][:, probes:]),
    }
    graphs, results = {}, {}
    for name, (first, second) in representations.items():
        graph, correlation, neighbor_jaccard = graph_from(first, second, config["neighbor_k"])
        graphs[name] = graph
        results[name] = {
            "pairwise_similarity_pearson": correlation,
            "median_neighbor_jaccard": neighbor_jaccard,
            **structure(graph, model.config.n_head),
        }
    results["current_joint"]["edge_jaccard_with_native_joint"] = edge_jaccard(
        graphs["current_joint"], graphs["native_joint"])
    results["address_only"]["edge_jaccard_with_native_joint"] = edge_jaccard(
        graphs["address_only"], graphs["native_joint"])
    results["native_write_only"]["edge_jaccard_with_native_joint"] = edge_jaccard(
        graphs["native_write_only"], graphs["native_joint"])
    robust_edges = edge_set(graphs["native_joint"]) & edge_set(graphs["current_joint"])
    robust_graph = nx.Graph(); robust_graph.add_nodes_from(range(model.config.n_layer * model.config.n_head))
    robust_graph.add_edges_from(sorted(robust_edges))
    robust_components = sorted(
        [sorted(component) for component in nx.connected_components(robust_graph) if len(component) > 1],
        key=lambda component: (len(component), component), reverse=True,
    )
    nodes = model.config.n_layer * model.config.n_head
    possible = [(i, j) for i in range(nodes) for j in range(i + 1, nodes)]
    baseline = {
        "same_head_index_fraction": float(np.mean([i % model.config.n_head == j % model.config.n_head for i, j in possible])),
        "same_layer_fraction": float(np.mean([i // model.config.n_head == j // model.config.n_head for i, j in possible])),
        "adjacent_layer_fraction": float(np.mean([abs(i // model.config.n_head - j // model.config.n_head) == 1 for i, j in possible])),
    }
    native = results["native_joint"]
    hypotheses = {
        "R1_native_edges_are_not_bus_index_artifacts": {
            "supported": native["same_head_index_fraction"] <= 2 * baseline["same_head_index_fraction"],
            "observed": native["same_head_index_fraction"], "two_x_baseline": 2 * baseline["same_head_index_fraction"],
        },
        "R2_native_edges_survive_bus_removal": {
            "supported": results["current_joint"]["edge_jaccard_with_native_joint"] >= .50,
            "edge_jaccard": results["current_joint"]["edge_jaccard_with_native_joint"],
        },
        "R3_joint_motifs_are_not_address_only": {
            "supported": results["address_only"]["edge_jaccard_with_native_joint"] < .80,
            "edge_jaccard": results["address_only"]["edge_jaccard_with_native_joint"],
        },
        "R4_joint_motifs_are_not_write_only": {
            "supported": results["native_write_only"]["edge_jaccard_with_native_joint"] < .80,
            "edge_jaccard": results["native_write_only"]["edge_jaccard_with_native_joint"],
        },
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {
        "config": config, "baseline": baseline, "representations": results,
        "hypotheses": hypotheses,
        "bus_robust": {
            # Keep the actual frozen graph as well as its size.  The generic
            # structure() summary also has an `edges` count, so use an
            # unambiguous key for downstream external-anchor validation.
            "edge_pairs": [[a, b] for a, b in sorted(robust_edges)],
            "components": [
                [{"layer": node // model.config.n_head, "head": node % model.config.n_head}
                 for node in component]
                for component in robust_components
            ],
            **structure(robust_graph, model.config.n_head),
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2))
    lines = ["# Unsupervised weight-motif red team", "",
             "The frozen native motifs were recomputed after removing the persistent layer-0 value bus, and from address/write halves separately.", "", "## Results", "",
             "| representation | edges | pairwise r | neighbor Jaccard | same head index | same layer | adjacent layer | native-edge Jaccard |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in results.items():
        overlap = row.get("edge_jaccard_with_native_joint")
        lines.append(f"| {name} | {row['edges']} | {row['pairwise_similarity_pearson']:.3f} | {row['median_neighbor_jaccard']:.3f} | {row['same_head_index_fraction']:.3f} | {row['same_layer_fraction']:.3f} | {row['adjacent_layer_fraction']:.3f} | {overlap if overlap is not None else '—'} |")
    lines += ["", "Random-pair structural baselines: " + ", ".join(f"{key}={value:.3f}" for key, value in baseline.items()) + ".", "", "## Locked decisions", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'passed' if row['supported'] else 'failed'}.**")
    REPORT.write_text("\n".join(lines) + "\n")
    names = list(results); x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - .2, [results[n]["same_head_index_fraction"] for n in names], .4, label="same head index")
    ax.bar(x + .2, [results[n]["same_layer_fraction"] for n in names], .4, label="same layer")
    ax.axhline(baseline["same_head_index_fraction"], color="black", linestyle="--", alpha=.5)
    ax.set_xticks(x, names, rotation=30, ha="right"); ax.legend(); ax.set_ylabel("stable-edge fraction")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "native": native, "current": results["current_joint"]}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
