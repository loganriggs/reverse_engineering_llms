#!/usr/bin/env python3
"""Prospectively test concrete read relations with position/frequency matched controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_contextual_read_function_discovery import (
    collect_targets, edge_predicates, edge_template, frequency_quantiles,
)
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import token_categories


CONFIG = Path("configs/bilin18_prospective_read_relation_test.json")
OUTPUT = TENSOR_ROOT / "bilin18_prospective_read_relation_test.json"
REPORT = TENSOR_ROOT / "BILIN18_PROSPECTIVE_READ_RELATION_TEST.md"
FIGURE = TENSOR_ROOT / "bilin18_prospective_read_relation_test.png"


def candidate_masks(predicates, names):
    p = {name: predicates[:, index] for index, name in enumerate(names)}
    distance_one = p["distance:1-1"]
    function_word = np.logical_or.reduce([p[f"source:{name}"] for name in ("determiner", "preposition", "conjunction", "pronoun", "auxiliary", "negation", "modal")])
    return {
        "adjacent_name_piece": distance_one & p["source:capitalized"] & (p["destination:capitalized"] | p["destination:word_continuation"]),
        "adjacent_function_word": distance_one & function_word & p["destination:alphabetic"],
        "boundary_source": (p["source:newline"] | p["source:period"]),
        "long_range_capitalized": (p["distance:16-31"] | p["distance:32-63"] | p["distance:64-128"]) & p["source:capitalized"],
        "repeated_same_token": p["relation:same_token"] & ~p["distance:0-0"],
        "induction_like_previous_match": p["relation:matching_previous_token"] & ~(p["distance:0-0"] | distance_one),
    }


def strata(raw, frequency_bins, positions):
    source, destination = edge_template(positions)
    source_ids = raw[:, source].reshape(-1); destination_ids = raw[:, destination].reshape(-1)
    distance = np.broadcast_to(destination - source, (len(raw), len(source))).reshape(-1)
    distance_bin = np.minimum(np.floor(np.log2(np.maximum(distance, 1))).astype(np.int64) + (distance > 0), 7)
    destination_exact = np.broadcast_to(destination, (len(raw), len(source))).reshape(-1)
    # Mixed-radix ID: exact destination, log-distance, source frequency, destination frequency.
    return (((destination_exact * 8 + distance_bin) * 8 + frequency_bins[source_ids]) * 8 + frequency_bins[destination_ids])


def matched_samples(candidate, stratum, seed, repetitions, minimum):
    rng = np.random.default_rng(seed)
    candidate_indices = np.flatnonzero(candidate)
    if len(candidate_indices) < minimum: return None
    candidates_by = {}
    controls_by = {}
    order = np.argsort(stratum, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(stratum[order])) + 1, len(order)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        local = order[start:end]
        group = int(stratum[local[0]])
        positive = local[candidate[local]]; negative = local[~candidate[local]]
        if len(positive) and len(negative):
            candidates_by[group] = positive; controls_by[group] = negative
    eligible = np.concatenate(list(candidates_by.values())) if candidates_by else np.empty(0, dtype=int)
    if len(eligible) < minimum: return None
    samples = []
    for _ in range(repetitions):
        positives, negatives = [], []
        for group, positive in candidates_by.items():
            negative = controls_by[group]
            count = min(len(positive), len(negative))
            positives.append(rng.choice(positive, count, replace=False)); negatives.append(rng.choice(negative, count, replace=False))
        positives = np.concatenate(positives); negatives = np.concatenate(negatives)
        samples.append((positives, negatives))
    return {"candidate_edges": int(len(candidate_indices)), "matched_edges_per_class": int(len(positives)), "samples": samples}


def matched_auc(candidate, values, stratum, seed, repetitions, minimum, prepared=None):
    prepared = prepared or matched_samples(candidate, stratum, seed, repetitions, minimum)
    if prepared is None: return None
    results = []
    for positives, negatives in prepared["samples"]:
        labels = np.r_[np.ones(len(positives), dtype=bool), np.zeros(len(negatives), dtype=bool)]
        scores = np.r_[values[positives], values[negatives]]
        results.append(float(auc(labels, scores)))
    return {"candidate_edges": prepared["candidate_edges"], "matched_edges_per_class": prepared["matched_edges_per_class"],
            "auc_mean": float(np.mean(results)), "auc_q05": float(np.quantile(results, .05)), "auc_q95": float(np.quantile(results, .95))}


def pair_pass(results, candidate, expected, heads):
    values = []
    for split in ("first", "second"):
        expected_auc = results[split][candidate][expected]["auc_mean"]
        controls = [results[split][candidate][head]["auc_mean"] for head in heads if head != expected and results[split][candidate][head] is not None]
        values.append(expected_auc >= .60 and expected_auc >= float(np.median(controls)) + .05)
    return all(values)


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(config["corpus"], mmap_mode="r")
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    human = token_categories(flags); frequency_bins, _ = frequency_quantiles(corpus, tokenizer.vocab_size)
    raw_splits = {
        "first": np.asarray(corpus[config["first_start"]:config["first_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64),
        "second": np.asarray(corpus[config["second_start"]:config["second_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64),
    }
    prior = json.loads(Path(config["ranked_heads_artifact"]).read_text())
    heads = [tuple(pair) for pair in prior["selected"]["top8"]]
    heads_by_layer = {}
    for key in heads: heads_by_layer.setdefault(key[0], []).append(key)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    probes = {layer: operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)[0] for layer in heads_by_layer}
    results = {}
    for split_index, (split, raw) in enumerate(raw_splits.items()):
        print(split, flush=True)
        predicates, names, _, _ = edge_predicates(raw, flags, human, frequency_bins, config["positions"])
        candidates = candidate_masks(predicates, names); group = strata(raw, frequency_bins, config["positions"])
        targets = collect_targets(model, raw, heads_by_layer, probes, config)
        results[split] = {}
        for candidate_name, candidate in candidates.items():
            results[split][candidate_name] = {}
            prepared = matched_samples(
                candidate, group, config["seed"] + split_index * 1000,
                config["control_repetitions"], config["minimum_candidate_edges"],
            )
            for head_index, head in enumerate(heads):
                name = f"L{head[0]}H{head[1]}"
                results[split][candidate_name][name] = matched_auc(
                    candidate, targets[head], group, config["seed"] + split_index * 1000 + head_index,
                    config["control_repetitions"], config["minimum_candidate_edges"], prepared=prepared,
                )

    expectations = {name: f"L{pair[0]}H{pair[1]}" for name, pair in config["candidate_expectations"].items()}
    head_names = [f"L{layer}H{head}" for layer, head in heads]
    passes = {candidate: pair_pass(results, candidate, expected, head_names) for candidate, expected in expectations.items()}
    unsupervised = []
    for candidate in ("repeated_same_token", "induction_like_previous_match"):
        for head in head_names:
            rows = [results[split][candidate][head] for split in results]
            if all(row is not None and row["auc_mean"] >= .65 for row in rows): unsupervised.append({"candidate": candidate, "head": head})
    hypotheses = {
        "P1_adjacent_name_is_specific": {"supported": passes["adjacent_name_piece"]},
        "P2_boundary_read_is_specific": {"supported": passes["boundary_source"]},
        "P3_function_word_read_is_specific": {"supported": passes["adjacent_function_word"]},
        "P4_long_range_capitalized_is_specific": {"supported": passes["long_range_capitalized"]},
        "P5_some_new_relation_is_validated": {"supported": sum(passes.values()) >= 2, "count": sum(passes.values())},
        "P6_unsupervised_relation_candidates_exist": {"supported": bool(unsupervised), "candidates": unsupervised},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "pair_passes": passes, "results": results}, indent=2, allow_nan=False))
    lines = ["# Prospective matched tests of candidate read relations", "", "Candidate relations were frozen from earlier descriptive examples, then tested on two later FineWeb blocks. Every candidate edge is matched to a non-candidate edge at the same destination position, log-distance bin, source-frequency bin, and destination-frequency bin. AUROC asks whether exact local operator-effect magnitude ranks candidate edges above controls.", "", "## Locked results", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "## Expected pairs", "", "| Candidate | Expected head | split AUCs | median-other AUCs | pass |", "|---|---|---:|---:|---|"]
    for candidate, expected in expectations.items():
        expected_values, other_values = [], []
        for split in ("first", "second"):
            expected_values.append(results[split][candidate][expected]["auc_mean"])
            other_values.append(float(np.median([results[split][candidate][head]["auc_mean"] for head in head_names if head != expected and results[split][candidate][head] is not None])))
        lines.append(f"| {candidate} | {expected} | {expected_values[0]:.3f} / {expected_values[1]:.3f} | {other_values[0]:.3f} / {other_values[1]:.3f} | {'yes' if passes[candidate] else 'no'} |")
    lines += ["", "## Full head matrix (mean across splits)", "", "| Candidate | " + " | ".join(head_names) + " |", "|---|" + "---:|" * len(head_names)]
    for candidate in results["first"]:
        values = []
        for head in head_names:
            pair = [results[split][candidate][head] for split in ("first", "second")]
            values.append("n/a" if any(row is None for row in pair) else f"{np.mean([row['auc_mean'] for row in pair]):.3f}")
        lines.append(f"| {candidate} | " + " | ".join(values) + " |")
    lines += ["", "## Guardrails", "", "This is a prospective validation of local read-effect magnitude, not a final-loss intervention. Matching is exact on coarse declared strata but not on token identity, syntax, or all correlated lexical properties. The candidate hypotheses came from selected high-effect examples, so only these later corpus blocks are confirmatory. Failure means the proposed readable relation is not a sufficiently specific description under this metric; it does not prove the head lacks other semantics."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    candidates = list(expectations); x = np.arange(len(candidates)); width = .09
    for index, head in enumerate(head_names):
        values = [np.mean([results[split][candidate][head]["auc_mean"] for split in ("first", "second")]) for candidate in candidates]
        axis.bar(x + (index - 3.5) * width, values, width=width, label=head)
    axis.axhline(.5, color="black", linewidth=.8); axis.set(xticks=x, xticklabels=candidates, ylabel="Matched effect AUROC", title="Prospective candidate read relations"); axis.tick_params(axis="x", rotation=20); axis.legend(ncol=4, fontsize=8); axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "passes": passes}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
