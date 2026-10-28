#!/usr/bin/env python3
"""Test lexical and semantic recurrence of the value code carried by Layer 4."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import edge_mask, median
from run_bilin18_layer8_semantic_attention_value_bus_causality import (
    factor_shapley,
    four_factor_edge_write,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer4_current_value_code_recurrence.json")
OUTPUT = TENSOR_ROOT / "layer4_current_value_code_recurrence.json"
REPORT = TENSOR_ROOT / "LAYER4_CURRENT_VALUE_CODE_RECURRENCE.md"
FIGURE = TENSOR_ROOT / "layer4_current_value_code_recurrence.png"
REPRESENTATIONS = ("source_current_value", "routed_current_value")


def normalize(vector):
    vector = vector.float().flatten()
    return vector / vector.norm().clamp_min(1e-30)


@torch.inference_mode()
def representations(model, layer, base, counterfactual, edges, edit_position, final_position):
    source_current_value = normalize(
        counterfactual["details"]["current_value"][0, edit_position]
        - base["details"]["current_value"][0, edit_position]
    )
    mask = edge_mask(
        model.config.n_head, base["output"].shape[1], edges, base["output"].device,
    )
    worlds = [
        four_factor_edge_write(model.blocks[layer], base, counterfactual, mask, factor_mask)
        for factor_mask in range(16)
    ]
    routed_current_value = factor_shapley(worlds, "insertion")["current_value"]
    routed_current_value = normalize(
        routed_current_value[0, edit_position:final_position + 1]
    )
    return {
        "source_current_value": source_current_value.cpu().numpy(),
        "routed_current_value": routed_current_value.cpu().numpy(),
    }


def compare(query_records, references, representation):
    reference_matrix = np.stack([references[pair][representation] for pair in references])
    reference_pairs = list(references)
    similarities = []
    correct = 0
    same_pair, same_family, different_family = [], [], []
    family_by_pair = {pair: references[pair]["family"] for pair in references}
    for record in query_records:
        scores = reference_matrix @ record[representation]
        predicted = reference_pairs[int(np.argmax(scores))]
        correct += int(predicted == record["pair"])
        for pair, score in zip(reference_pairs, scores):
            score = float(score)
            if pair == record["pair"]:
                same_pair.append(score)
            elif family_by_pair[pair] == record["family"]:
                same_family.append(score)
            else:
                different_family.append(score)
            similarities.append({
                "query_pair": record["pair"],
                "reference_pair": pair,
                "score": score,
                "relation": (
                    "same_pair" if pair == record["pair"]
                    else "same_family" if family_by_pair[pair] == record["family"]
                    else "different_family"
                ),
            })
    family_gap = median(same_family) - median(different_family)
    return {
        "same_pair_cosine": median(same_pair),
        "same_family_cosine": median(same_family),
        "different_family_cosine": median(different_family),
        "family_gap": family_gap,
        "pair_accuracy": correct / len(query_records),
        "similarities": similarities,
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    value_config = json.loads(Path(config["source_config"]).read_text())
    factor_config = json.loads(Path(value_config["source_config"]).read_text())
    frozen_config = json.loads(Path(factor_config["source_config"]).read_text())
    rank_config = json.loads(Path(frozen_config["source_config"]).read_text())
    causal_config = json.loads(Path(rank_config["source_config"]).read_text())
    atlas_config = json.loads(Path(causal_config["source_config"]).read_text())
    path_config = json.loads(Path(atlas_config["source_config"]).read_text())
    fixed = json.loads(Path(path_config["source_config"]).read_text())
    parent = json.loads(Path(fixed["source_config"]).read_text())
    frozen = json.loads(Path(config["frozen_results"]).read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    layer = config["layer"]
    edit_position, final_position = fixed["edit_position"], fixed["causal_position"]
    edges = [
        tuple(row["edge"])
        for row in frozen["discovered_routes"][str(layer)]["magnitude"][:config["route_rank"]]
    ]
    records = []
    for corpus_index, (corpus, corpus_config) in enumerate(fixed["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(path_config["seed"] + corpus_index)
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    atlas_config["contexts_per_pair"], rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    base_raw = np.asarray(
                        rows[row_index, start:start + fixed["sequence_length"]], dtype=np.int64,
                    ).copy()
                    counterfactual_raw = base_raw.copy()
                    counterfactual_raw[edit_position] = target_token
                    base = forward_layers(
                        model, torch.as_tensor(base_raw, device="cuda")[None], last_layer=layer,
                    )[layer]
                    counterfactual = forward_layers(
                        model, torch.as_tensor(counterfactual_raw, device="cuda")[None], last_layer=layer,
                    )[layer]
                    values = representations(
                        model, layer, base, counterfactual, edges,
                        edit_position, final_position,
                    )
                    records.append({
                        "corpus": corpus,
                        "context_index": context_index,
                        "family": family,
                        "pair": f"{source_token}->{target_token}",
                        "row": int(row_index),
                        **values,
                    })
                print(corpus, family, source_token, "->", target_token, flush=True)

    references = {
        record["pair"]: record
        for record in records
        if record["corpus"] == config["reference_corpus"]
        and record["context_index"] == config["reference_context_index"]
    }
    results = {}
    raw_similarities = {}
    for corpus in fixed["evaluation"]:
        query = [
            record for record in records
            if record["corpus"] == corpus
            and record["context_index"] == config["query_context_index"]
        ]
        results[corpus] = {}
        raw_similarities[corpus] = {}
        for representation in REPRESENTATIONS:
            result = compare(query, references, representation)
            raw_similarities[corpus][representation] = result.pop("similarities")
            results[corpus][representation] = result

    hypotheses = {
        "V1_source_value_code_is_lexically_recurrent": {
            "supported": all(
                results[corpus]["source_current_value"]["same_pair_cosine"] >= 0.20
                for corpus in results
            ),
        },
        "V2_routed_value_code_is_lexically_recurrent": {
            "supported": all(
                results[corpus]["routed_current_value"]["same_pair_cosine"] >= 0.20
                for corpus in results
            ),
        },
        "V3_routed_code_identifies_lexical_pair": {
            "supported": all(
                results[corpus]["routed_current_value"]["pair_accuracy"] >= 0.50
                for corpus in results
            ),
        },
        "V4_source_value_abstracts_semantic_family": {
            "supported": all(
                results[corpus]["source_current_value"]["family_gap"] >= 0.05
                for corpus in results
            ),
        },
        "V5_routed_value_abstracts_semantic_family": {
            "supported": all(
                results[corpus]["routed_current_value"]["family_gap"] >= 0.05
                for corpus in results
            ),
        },
        "V6_routing_improves_semantic_abstraction": {
            "supported": all(
                results[corpus]["routed_current_value"]["family_gap"]
                - results[corpus]["source_current_value"]["family_gap"] >= 0.05
                for corpus in results
            ),
        },
    }
    serializable_records = [
        {key: value for key, value in record.items() if key not in REPRESENTATIONS}
        for record in records
    ]
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
        "records": serializable_records,
        "similarities": raw_similarities,
    }, indent=2))

    lines = [
        "# Recurrence of the Layer-4 current-value code",
        "",
        "The dominant current-V path is inspected before and after the frozen 64-edge attention skeleton. The raw code is the changed current-value vector at the edited source. The routed code is the exact current-V Shapley write over the seventeen affected positions. One FineWeb context per lexical pair is the frozen reference; disjoint FineWeb and TinyStories contexts are queried against those references.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Current-value recurrence](layer4_current_value_code_recurrence.png)",
        "",
        "| Query corpus | Representation | Same-pair cosine | Same-family / different-family | Family gap | Pair retrieval |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, corpus_results in results.items():
        for representation in REPRESENTATIONS:
            row = corpus_results[representation]
            lines.append(
                f"| {corpus} | {representation} | {row['same_pair_cosine']:.3f} | "
                f"{row['same_family_cosine']:.3f} / {row['different_family_cosine']:.3f} | "
                f"{row['family_gap']:.3f} | {row['pair_accuracy']:.3f} |"
            )
    lines += [
        "",
        "Pair recurrence means that a particular token replacement produces a repeatable transported direction. Family recurrence is the stronger claim that different word pairs in the same hand-labelled category share a direction. Neither establishes that the code is sufficient for the full behavioral change.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    width = 0.35
    corpora = list(results)
    x = np.arange(len(corpora))
    for offset, representation in ((-width / 2, REPRESENTATIONS[0]), (width / 2, REPRESENTATIONS[1])):
        axes[0].bar(
            x + offset,
            [results[corpus][representation]["same_pair_cosine"] for corpus in corpora],
            width,
            label=representation,
        )
        axes[1].bar(
            x + offset,
            [results[corpus][representation]["family_gap"] for corpus in corpora],
            width,
            label=representation,
        )
    axes[0].set_title("Same lexical-pair recurrence")
    axes[1].set_title("Same-family cosine advantage")
    for axis in axes:
        axis.set_xticks(x, corpora)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
