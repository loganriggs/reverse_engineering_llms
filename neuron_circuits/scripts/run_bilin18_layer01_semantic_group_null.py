#!/usr/bin/env python3
"""Permutation null for whether declared semantic edit families predict native channel banks."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer01_native_rotation_redteam import normalized_score
from run_bilin18_layer01_native_rotation_tournament import coordinate_scores
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_semantic_group_null.json")
OUTPUT = TENSOR_ROOT / "layer01_semantic_group_null.json"
REPORT = TENSOR_ROOT / "LAYER01_SEMANTIC_GROUP_NULL.md"
FIGURE = TENSOR_ROOT / "layer01_semantic_group_null.png"


def group_members(labels: np.ndarray, index: int) -> np.ndarray:
    return np.flatnonzero((labels == labels[index]) & (np.arange(len(labels)) != index))


def top_projection(discovery: np.ndarray, evaluation: np.ndarray, rank: int) -> float:
    indices = np.argpartition(discovery, -rank)[-rank:]
    return float(evaluation[indices].sum())


def evaluate_grouping(
    labels: np.ndarray,
    discovery: np.ndarray,
    evaluation: np.ndarray,
    ranks: list[int],
) -> dict[int, float]:
    values = {rank: [] for rank in ranks}
    for index in range(len(labels)):
        members = group_members(labels, index)
        score = discovery[members].mean(0)
        for rank in ranks:
            values[rank].append(top_projection(score, evaluation[index], rank))
    return {rank: float(np.median(rows)) for rank, rows in values.items()}


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    red_config = json.loads(Path(config["source_config"]).read_text())
    tournament_config = json.loads(Path(red_config["source_config"]).read_text())
    source_config = json.loads(Path(tournament_config["source_config"]).read_text())
    fixed, parent = recover_configs(source_config)
    saved = json.loads(Path(config["source_results"]).read_text())
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    ranks = config["ranks"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }
    specs = fixed["evaluation"]

    def reconstruct(record, layer):
        corpus = record["corpus"]
        spec = specs[corpus]
        rows = arrays[corpus][spec["start"]:spec["start"] + spec["rows"]]
        raw = np.asarray(
            rows[record["row"], record["source_start"]:record["source_start"] + fixed["sequence_length"]],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[edit_position] = int(record["pair"].split("->")[1])
        base = forward_layers(model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer)
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )
        return base[layer], counterfactual[layer]

    discovery_saved = [
        row for row in saved["semantic_records"]
        if row["corpus"] == config["discovery_corpus"]
        and row["context_index"] == config["discovery_context_index"]
    ]
    pairs = [row["pair"] for row in discovery_saved]
    families = [row["family"] for row in discovery_saved]
    family_ids = {name: index for index, name in enumerate(dict.fromkeys(families))}
    true_labels = np.asarray([family_ids[name] for name in families])
    query_by_corpus = {
        corpus: {
            row["pair"]: row for row in saved["semantic_records"]
            if row["corpus"] == corpus and row["context_index"] == config["evaluation_context_index"]
        }
        for corpus in fixed["evaluation"]
    }
    rng = np.random.default_rng(config["seed"])
    permutation_labels = [rng.permutation(true_labels) for _ in range(config["permutations"])]
    results = {}

    for layer in config["layers"]:
        decoder = model.blocks[layer].mlp.Down.weight.float()
        discovery = []
        for record in discovery_saved:
            base, counterfactual = reconstruct(record, layer)
            state = channel_state(model, layer, base, counterfactual, final_position)
            discovery.append(coordinate_scores(
                state["delta_product"], decoder, state["target_write"], final_position,
            ).abs().cpu().numpy())
        discovery = np.stack(discovery)
        decoder_norm = decoder.norm(dim=0).cpu().numpy().clip(1e-12)
        discovery_normed = discovery / decoder_norm[None]
        normalized_discovery = discovery / np.linalg.norm(discovery, axis=1, keepdims=True).clip(1e-30)
        similarities = normalized_discovery @ normalized_discovery.T
        np.fill_diagonal(similarities, -np.inf)
        layer_rows = {}
        for corpus in fixed["evaluation"]:
            evaluation = []
            for pair in pairs:
                base, counterfactual = reconstruct(query_by_corpus[corpus][pair], layer)
                state = channel_state(model, layer, base, counterfactual, final_position)
                evaluation.append(coordinate_scores(
                    state["delta_product"], decoder, state["target_write"], final_position,
                ).cpu().numpy())
            evaluation = np.stack(evaluation)
            true = evaluate_grouping(true_labels, discovery, evaluation, ranks)
            true_normed = evaluate_grouping(true_labels, discovery_normed, evaluation, ranks)
            null = {rank: [] for rank in ranks}
            null_normed = {rank: [] for rank in ranks}
            for labels in permutation_labels:
                row = evaluate_grouping(labels, discovery, evaluation, ranks)
                row_normed = evaluate_grouping(labels, discovery_normed, evaluation, ranks)
                for rank in ranks:
                    null[rank].append(row[rank])
                    null_normed[rank].append(row_normed[rank])
            global_values = {rank: [] for rank in ranks}
            nearest_values = {rank: [] for rank in ranks}
            for index in range(len(pairs)):
                other = np.arange(len(pairs)) != index
                global_score = discovery[other].mean(0)
                mate_count = len(group_members(true_labels, index))
                nearest = np.argsort(similarities[index])[-mate_count:]
                nearest_score = discovery[nearest].mean(0)
                for rank in ranks:
                    global_values[rank].append(top_projection(global_score, evaluation[index], rank))
                    nearest_values[rank].append(top_projection(nearest_score, evaluation[index], rank))
            layer_rows[corpus] = {
                str(rank): {
                    "true": true[rank],
                    "true_normed": true_normed[rank],
                    "null_median": float(np.median(null[rank])),
                    "null_p95": float(np.quantile(null[rank], .95)),
                    "null_p99": float(np.quantile(null[rank], .99)),
                    "empirical_p": float((1 + np.sum(np.asarray(null[rank]) >= true[rank])) / (1 + len(null[rank]))),
                    "normed_null_median": float(np.median(null_normed[rank])),
                    "normed_null_p95": float(np.quantile(null_normed[rank], .95)),
                    "normed_empirical_p": float((1 + np.sum(np.asarray(null_normed[rank]) >= true_normed[rank])) / (1 + len(null_normed[rank]))),
                    "global": float(np.median(global_values[rank])),
                    "nearest_pair_oracle": float(np.median(nearest_values[rank])),
                }
                for rank in ranks
            }
            print("group null", layer, corpus, flush=True)
        results[str(layer)] = layer_rows

    def row(layer, corpus, rank=256):
        return results[str(layer)][corpus][str(rank)]

    hypotheses = {
        "G1_layer1_human_families_beat_random_groups": {
            "supported": all(row(1, c)["true"] - row(1, c)["null_p99"] >= .02 for c in fixed["evaluation"]),
        },
        "G2_layer0_human_families_beat_random_groups": {
            "supported": all(row(0, c)["true"] - row(0, c)["null_p99"] >= .02 for c in fixed["evaluation"]),
        },
        "G3_semantic_grouping_emerges_by_layer1": {
            "supported": all(
                (row(1, c)["true"] - row(1, c)["null_median"])
                - (row(0, c)["true"] - row(0, c)["null_median"]) >= .02
                for c in fixed["evaluation"]
            ),
        },
        "G4_layer1_human_families_beat_global": {
            "supported": all(row(1, c)["true"] - row(1, c)["global"] >= .02 for c in fixed["evaluation"]),
        },
        "G5_family_advantage_transfers_ood": {
            "supported": all(row(layer, "tinystories")["true"] - row(layer, "tinystories")["null_median"] > 0 for layer in config["layers"]),
        },
        "G6_semantic_grouping_survives_decoder_norm_control": {
            "supported": all(
                row(layer, c)["true_normed"] > row(layer, c)["normed_null_p95"]
                for layer in config["layers"] for c in fixed["evaluation"]
            ),
        },
        "G7_pair_similarity_oracle_beats_human_family": {
            "supported": all(any(
                row(layer, c)["nearest_pair_oracle"] - row(layer, c)["true"] >= .03
                for c in fixed["evaluation"]
            ) for layer in config["layers"]),
        },
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses, "pairs": pairs,
        "families": families, "layers": results,
    }, indent=2))

    lines = [
        "# Do human semantic families predict native Layer-0/1 channel banks?",
        "",
        "For each held-out lexical pair, the native bank is selected from the other pairs in its declared family. Its exact additive projection onto the current MLP-write change is compared with 2,000 size-matched random regroupings of the same 19 edits. No permuted bank sees the held-out pair. Decoder-norm-normalized selection, a global bank, and a target-informed nearest-pair oracle are controls.",
        "",
        "## Locked hypotheses", "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "", "![Semantic grouping null](layer01_semantic_group_null.png)", "",
        "| Layer | Corpus | Rank | Human family | Null median / p95 / p99 | p | Normed family / null p95 | Global | Nearest-pair oracle |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in config["layers"]:
        for corpus in fixed["evaluation"]:
            for rank in ranks:
                item = row(layer, corpus, rank)
                lines.append(
                    f"| {layer} | {corpus} | {rank} | {item['true']:.3f} | "
                    f"{item['null_median']:.3f} / {item['null_p95']:.3f} / {item['null_p99']:.3f} | "
                    f"{item['empirical_p']:.4f} | {item['true_normed']:.3f} / {item['normed_null_p95']:.3f} | "
                    f"{item['global']:.3f} | {item['nearest_pair_oracle']:.3f} |"
                )
    lines += [
        "",
        "The projection is the exact sum of selected channels' signed inner products with the full MLP-write change, divided by full-write squared norm. It is additive and inexpensive enough for permutation testing, but unlike finite downstream fidelity it does not penalize orthogonal write energy. Conclusions should therefore require agreement with the preceding finite LOPO experiment.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    x = np.arange(4)
    labels = ["L0 Fine", "L0 Tiny", "L1 Fine", "L1 Tiny"]
    rows = [row(0, "fineweb"), row(0, "tinystories"), row(1, "fineweb"), row(1, "tinystories")]
    axes[0].bar(x - .2, [item["true"] for item in rows], .4, label="human family")
    axes[0].bar(x + .2, [item["null_median"] for item in rows], .4, label="random-group median")
    axes[0].scatter(x, [item["null_p99"] for item in rows], marker="_", s=160, color="black", label="null p99")
    axes[0].set_xticks(x, labels, rotation=20)
    axes[0].set_ylabel("rank-256 target projection")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=.25)
    axes[1].bar(x - .2, [item["true"] for item in rows], .4, label="human family")
    axes[1].bar(x + .2, [item["global"] for item in rows], .4, label="global")
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_ylabel("rank-256 target projection")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=.25)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps(hypotheses, indent=2), flush=True)
    print(REPORT, flush=True)


if __name__ == "__main__":
    main()
