#!/usr/bin/env python3
"""Discover held-out contextual predicates for exact late-head edge effects."""

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
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, rmsnorm_pullback_readers, token_categories
from run_bilin18_variable_function_atlas import factored_frobenius_cosines


CONFIG = Path("configs/bilin18_contextual_read_function_discovery.json")
OUTPUT = TENSOR_ROOT / "bilin18_contextual_read_function_discovery.json"
REPORT = TENSOR_ROOT / "BILIN18_CONTEXTUAL_READ_FUNCTION_DISCOVERY.md"
FIGURE = TENSOR_ROOT / "bilin18_contextual_read_function_discovery.png"


def edge_template(positions):
    source = np.concatenate([np.arange(position + 1) for position in positions])
    destination = np.concatenate([np.full(position + 1, position) for position in positions])
    return source.astype(np.int64), destination.astype(np.int64)


def frequency_quantiles(corpus, vocab_size, bins=8):
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=vocab_size)
    order = np.argsort(counts, kind="mergesort")
    ranks = np.empty(vocab_size, dtype=np.int64); ranks[order] = np.arange(vocab_size)
    return np.minimum(ranks * bins // vocab_size, bins - 1), counts


def _latest_occurrence(raw, source, destination):
    output = np.zeros((len(raw), len(source)), dtype=bool)
    offsets = np.r_[0, np.cumsum(np.bincount(destination, minlength=raw.shape[1]))]
    for row_index, tokens in enumerate(raw):
        for destination_position in np.unique(destination):
            selected = np.flatnonzero(destination == destination_position)
            seen = set()
            for local in selected[::-1]:
                token = int(tokens[source[local]])
                if token not in seen:
                    output[row_index, local] = True; seen.add(token)
    return output


def edge_predicates(raw, flags, human_categories, frequency_bins, positions):
    source, destination = edge_template(positions)
    batch, edge_count = len(raw), len(source)
    source_ids = raw[:, source]; destination_ids = raw[:, destination]
    distance = destination - source
    columns, names = [], []

    def add(name, values):
        names.append(name); columns.append(np.asarray(values, dtype=bool).reshape(batch, edge_count))

    # Baseline address and prevalence primitives.
    for low, high in zip((0, 1, 2, 4, 8, 16, 32, 64), (1, 2, 4, 8, 16, 32, 64, 129)):
        add(f"distance:{low}-{high - 1}", np.broadcast_to((distance >= low) & (distance < high), (batch, edge_count)))
    for block in range(8):
        add(f"source_position:{16 * block}-{16 * block + 15}", np.broadcast_to(source // 16 == block, (batch, edge_count)))
    for value in range(8):
        add(f"source_frequency_bin:{value}", frequency_bins[source_ids] == value)
        add(f"destination_frequency_bin:{value}", frequency_bins[destination_ids] == value)
    for threshold in (2, 4, 8):
        lengths = np.asarray([len(str(index)) for index in range(flags.shape[0])])  # overwritten below by caller-compatible proxy
        del lengths
        # Token-piece length is intentionally omitted here; exact string predicates are added in main.
    baseline_end = len(names)

    for prefix, ids in (("source", source_ids), ("destination", destination_ids)):
        for feature_index, feature in enumerate(FEATURES):
            add(f"{prefix}:{feature}", flags[ids, feature_index])
    lexical_end = len(names)

    src = flags[source_ids]
    dst = flags[destination_ids]
    feature_index = {name: FEATURES.index(name) for name in FEATURES}
    source_category = human_categories[source_ids]; destination_category = human_categories[destination_ids]
    add("relation:same_token", source_ids == destination_ids)
    add("relation:same_coarse_category", source_category == destination_category)
    add("relation:both_alphabetic", src[:, :, feature_index["alphabetic"]] & dst[:, :, feature_index["alphabetic"]])
    add("relation:both_capitalized", src[:, :, feature_index["capitalized"]] & dst[:, :, feature_index["capitalized"]])
    add("relation:capitalized_to_continuation", src[:, :, feature_index["capitalized"]] & dst[:, :, feature_index["word_continuation"]])
    add("relation:space_word_to_continuation", src[:, :, feature_index["leading_space"]] & dst[:, :, feature_index["word_continuation"]])
    add("relation:punctuation_to_capitalized", src[:, :, feature_index["punctuation"]] & dst[:, :, feature_index["capitalized"]])
    add("relation:newline_to_capitalized", src[:, :, feature_index["newline"]] & dst[:, :, feature_index["capitalized"]])
    function_word = np.logical_or.reduce([src[:, :, feature_index[name]] for name in ("determiner", "preposition", "conjunction", "pronoun", "auxiliary", "negation", "modal")])
    suffix = np.logical_or.reduce([src[:, :, feature_index[name]] for name in FEATURES if name.startswith("suffix_")])
    destination_function = np.logical_or.reduce([dst[:, :, feature_index[name]] for name in ("determiner", "preposition", "conjunction", "pronoun", "auxiliary", "negation", "modal")])
    add("relation:function_word_to_alphabetic", function_word & dst[:, :, feature_index["alphabetic"]])
    add("relation:suffix_to_function_word", suffix & destination_function)
    add("relation:source_equals_previous_destination", source_ids == raw[:, np.maximum(destination - 1, 0)])
    source_previous = np.maximum(source - 1, 0)
    add("relation:matching_previous_token", raw[:, source_previous] == raw[:, np.maximum(destination - 1, 0)])
    source_next = np.minimum(source + 1, raw.shape[1] - 1)
    add("relation:source_next_equals_destination", raw[:, source_next] == destination_ids)
    add("relation:source_previous_equals_destination", raw[:, source_previous] == destination_ids)
    add("context:latest_occurrence_of_source_token", _latest_occurrence(raw, source, destination))
    newline = flags[raw, feature_index["newline"]].astype(np.int64)
    punctuation = flags[raw, feature_index["punctuation"]].astype(np.int64)
    for name, indicator in (("newline_between", newline), ("punctuation_between", punctuation)):
        prefix = np.concatenate([np.zeros((batch, 1), dtype=np.int64), np.cumsum(indicator, axis=1)], axis=1)
        add(f"context:{name}", prefix[:, destination] - prefix[:, source] > 0)
    return np.stack(columns, axis=2).reshape(batch * edge_count, -1), names, baseline_end, lexical_end


def balanced_indices(labels, seed):
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    count = min(len(positive), len(negative))
    return np.concatenate([rng.choice(positive, count, replace=False), rng.choice(negative, count, replace=False)])


def fit_tree(predicates, labels, names, leaves, maximum_depth, minimum_leaf, allowed=None):
    if allowed is None: allowed = np.arange(predicates.shape[1])
    nodes = [{"indices": np.arange(len(labels)), "path": [], "depth": 0}]
    selected = []
    while len(nodes) < leaves:
        best = None
        for node_index, node in enumerate(nodes):
            indices = node["indices"]
            if node["depth"] >= maximum_depth or len(indices) < 2 * minimum_leaf: continue
            local_y = labels[indices].astype(np.float64)
            parent = local_y.sum() ** 2 / len(local_y)
            local_x = predicates[indices][:, allowed]
            left_n = local_x.sum(0).astype(np.float64); right_n = len(indices) - left_n
            left_y = local_x.T @ local_y; right_y = local_y.sum() - left_y
            gain = left_y ** 2 / np.maximum(left_n, 1) + right_y ** 2 / np.maximum(right_n, 1) - parent
            gain[(left_n < minimum_leaf) | (right_n < minimum_leaf)] = -np.inf
            local_feature = int(np.argmax(gain))
            if np.isfinite(gain[local_feature]) and (best is None or gain[local_feature] > best[0]):
                best = (float(gain[local_feature]), node_index, int(allowed[local_feature]))
        if best is None: break
        _, node_index, feature = best
        node = nodes.pop(node_index); truth = predicates[node["indices"], feature]
        nodes.append({"indices": node["indices"][~truth], "path": node["path"] + [(feature, False)], "depth": node["depth"] + 1})
        nodes.append({"indices": node["indices"][truth], "path": node["path"] + [(feature, True)], "depth": node["depth"] + 1})
        selected.append(feature)
    rules = []
    for node in nodes:
        value = float(labels[node["indices"]].mean())
        rules.append({"path": node["path"], "value": value, "examples": len(node["indices"])})
    return {"rules": rules, "selected": selected, "selected_names": [names[index] for index in selected]}


def predict_tree(tree, predicates):
    prediction = np.zeros(len(predicates), dtype=np.float64)
    assigned = np.zeros(len(predicates), dtype=bool)
    for rule in tree["rules"]:
        mask = np.ones(len(predicates), dtype=bool)
        for feature, value in rule["path"]: mask &= predicates[:, feature] == value
        prediction[mask] = rule["value"]; assigned |= mask
    assert assigned.all()
    return prediction


def predicate_jaccard(first, second):
    first, second = set(first), set(second)
    return float(len(first & second) / max(len(first | second), 1))


@torch.inference_mode()
def collect_targets(model, raw, heads_by_layer, probes, config):
    dimension = model.config.n_embd; native_heads = model.config.n_head; head_dim = dimension // native_heads
    source, destination = edge_template(config["positions"]); edge_count = len(source)
    output = {key: np.empty(len(raw) * edge_count, dtype=np.float32) for keys in heads_by_layer.values() for key in keys}
    offset = 0
    for start in range(0, len(raw), config["batch_size"]):
        ids = torch.as_tensor(raw[start:start + config["batch_size"]], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks[:max(heads_by_layer) + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
            if layer in heads_by_layer:
                pre = mixed[:, config["positions"]].float() + details["write"][:, config["positions"]].float()
                readers = rmsnorm_pullback_readers(pre, probes[layer])
                projection = block.attn.c_proj.weight.float().T.reshape(native_heads, head_dim, dimension)
                head_readers = torch.einsum("bpqo,hdo->bphqd", readers, projection)
                value_effect = torch.einsum("bkhd,bphqd->bphqk", details["mixed_value"].float(), head_readers)
                pattern = details["pattern"][:, :, config["positions"]].permute(0, 2, 1, 3)
                energy = (value_effect * pattern[:, :, :, None]).square().sum(3)
                indices = torch.as_tensor(source, device="cuda")
                position_indices = torch.repeat_interleave(
                    torch.arange(len(config["positions"]), device="cuda"),
                    torch.as_tensor(np.asarray(config["positions"]) + 1, device="cuda"),
                )
                for key in heads_by_layer[layer]:
                    head = key[1]
                    selected = energy[:, position_indices, head, indices]
                    values = selected.reshape(-1).cpu().numpy()
                    output[key][offset:offset + len(values)] = values
            pre_full = mixed + details["write"]; state = F.rms_norm(pre_full, (dimension,))
            x = pre_full + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        offset += len(ids) * edge_count
        if start % (config["batch_size"] * 16) == 0: print("  target rows", start, flush=True)
    return output


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(config["corpus"], mmap_mode="r")
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    human = token_categories(flags); frequency_bins, _ = frequency_quantiles(corpus, tokenizer.vocab_size)
    discovery = np.asarray(corpus[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    predicates_a, names, baseline_end, lexical_end = edge_predicates(discovery, flags, human, frequency_bins, config["positions"])
    predicates_b, names_b, _, _ = edge_predicates(evaluation, flags, human, frequency_bins, config["positions"])
    assert names == names_b

    prior = json.loads(Path(config["ranked_heads_artifact"]).read_text())
    ranked = [tuple(pair) for pair in prior["selected"]["top8"]]
    importance = {(row["layer"], row["head"]): row["discovery_importance"] for row in prior["importance"]}
    bottom = sorted(importance, key=importance.get)[:8]
    keys = ranked + bottom
    heads_by_layer = {}
    for key in keys: heads_by_layer.setdefault(key[0], []).append(key)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    probes = {layer: operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)[0] for layer in heads_by_layer}
    print("discovery targets", flush=True); target_a = collect_targets(model, discovery, heads_by_layer, probes, config)
    print("evaluation targets", flush=True); target_b = collect_targets(model, evaluation, heads_by_layer, probes, config)

    rows = []
    allowed_sets = {"baseline": np.arange(baseline_end), "lexical": np.arange(lexical_end), "contextual": np.arange(len(names))}
    split_point = len(discovery) // 2 * len(edge_template(config["positions"])[0])
    for key in keys:
        threshold_a = float(np.quantile(target_a[key], 1 - config["tail_fraction"])); threshold_b = float(np.quantile(target_b[key], 1 - config["tail_fraction"]))
        labels_a = target_a[key] >= threshold_a; labels_b = target_b[key] >= threshold_b
        train = balanced_indices(labels_a, config["seed"] + key[0] * 100 + key[1])
        methods, trees = {}, {}
        for name, allowed in allowed_sets.items():
            tree = fit_tree(predicates_a[train], labels_a[train], names, config["tree_leaves"], config["maximum_depth"], config["minimum_leaf_examples"], allowed)
            score = predict_tree(tree, predicates_b)
            methods[name] = float(auc(labels_b, score)); trees[name] = tree
        half_trees = []
        for half, region in enumerate((np.arange(split_point), np.arange(split_point, len(labels_a)))):
            local = region[balanced_indices(labels_a[region], config["seed"] + 10000 + half + key[0] * 100 + key[1])]
            half_trees.append(fit_tree(predicates_a[local], labels_a[local], names, config["tree_leaves"], config["maximum_depth"], config["minimum_leaf_examples"], allowed_sets["contextual"]))
        stability = predicate_jaccard(half_trees[0]["selected_names"], half_trees[1]["selected_names"])
        rows.append({"layer": key[0], "head": key[1], "ranked": key in ranked, "importance": importance[key],
                     "thresholds": {"discovery": threshold_a, "evaluation": threshold_b}, "aucs": methods,
                     "predicate_stability": stability, "tree": trees["contextual"]})

    ranked_rows = [row for row in rows if row["ranked"]]; bottom_rows = [row for row in rows if not row["ranked"]]
    predictable = sum(row["aucs"]["contextual"] >= .70 for row in ranked_rows)
    context_gain = sum(row["aucs"]["contextual"] >= row["aucs"]["baseline"] + .05 for row in ranked_rows)
    relation_gain = sum(row["aucs"]["contextual"] >= row["aucs"]["lexical"] + .03 for row in ranked_rows)
    stable = sum(row["predicate_stability"] >= .30 for row in ranked_rows)
    ranked_median = float(np.median([row["aucs"]["contextual"] for row in ranked_rows])); bottom_median = float(np.median([row["aucs"]["contextual"] for row in bottom_rows]))
    threshold_ratios = [max(row["thresholds"].values()) / max(min(row["thresholds"].values()), 1e-30) for row in rows]

    # Test the user's variable/function distinction: shared readable predicates, distinct exact OV maps.
    semantic_usage = {}
    factors = []
    for row in ranked_rows:
        key = (row["layer"], row["head"])
        for predicate in set(row["tree"]["selected_names"]):
            if not predicate.startswith(("distance:", "source_position:", "source_frequency_bin:", "destination_frequency_bin:")):
                semantic_usage.setdefault(predicate, []).append(key)
        block = model.blocks[key[0]].attn; native_heads = model.config.n_head; head_dim = model.config.n_embd // native_heads
        sl = slice(key[1] * head_dim, (key[1] + 1) * head_dim)
        factors.append((block.c_proj.weight[:, sl].detach().float().cpu(), block.c_v.weight[sl].detach().float().cpu()))
    ov_cosine, _ = factored_frobenius_cosines(factors); ranked_index = {key: index for index, key in enumerate(ranked)}
    reused = []
    for predicate, predicate_heads in semantic_usage.items():
        if len(predicate_heads) < 2: continue
        pair_values = [abs(ov_cosine[ranked_index[a], ranked_index[b]]) for i, a in enumerate(predicate_heads) for b in predicate_heads[i + 1:]]
        reused.append({"predicate": predicate, "heads": [f"L{a}H{b}" for a, b in predicate_heads], "median_abs_ov_cosine": float(np.median(pair_values))})
    distinct_reused = [row for row in reused if row["median_abs_ov_cosine"] < .20]

    hypotheses = {
        "C0_targets_are_replicable": {"supported": float(np.median(threshold_ratios)) < 2, "median_threshold_ratio": float(np.median(threshold_ratios))},
        "C1_contextual_rules_predict_effects": {"supported": predictable >= 4, "count": predictable},
        "C2_context_adds_beyond_position_frequency": {"supported": context_gain >= 4, "count": context_gain},
        "C3_relations_add_beyond_static_lexicon": {"supported": relation_gain >= 3, "count": relation_gain},
        "C4_rules_are_split_stable": {"supported": stable >= 4, "count": stable},
        "C5_ranked_heads_are_more_readable": {"supported": ranked_median >= bottom_median + .05, "ranked_median": ranked_median, "bottom_median": bottom_median},
        "C6_variables_are_reused_by_distinct_functions": {"supported": len(distinct_reused) >= 3, "count": len(distinct_reused)},
    }
    for result in hypotheses.values(): result["supported"] = bool(result["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "predicate_names": names, "rows": rows, "reused_variables": reused}, indent=2, allow_nan=False))

    lines = ["# Contextual read-function discovery", "", "A bounded 16-leaf readable tree predicts the top 2% of exact local attention-edge effects. Rules are selected on one 61,440-token FineWeb split and scored on another. Baseline rules see only distance, absolute source position, and source/destination frequency. Lexical rules add 50 preregistered static properties for each endpoint. Contextual rules add endpoint relations, repetition, induction-like matches, and intervening boundaries.", "", "## Locked results", ""]
    for name, result in hypotheses.items(): lines.append(f"- **{name}: {'supported' if result['supported'] else 'not supported'}.**")
    lines += ["", "## Head results", "", "| Head | group | baseline | lexical | contextual | split-rule Jaccard |", "|---|---|---:|---:|---:|---:|"]
    for row in rows: lines.append(f"| L{row['layer']}H{row['head']} | {'ranked' if row['ranked'] else 'bottom'} | {row['aucs']['baseline']:.3f} | {row['aucs']['lexical']:.3f} | {row['aucs']['contextual']:.3f} | {row['predicate_stability']:.3f} |")
    lines += ["", "## Reused candidate variables", ""]
    for row in sorted(reused, key=lambda item: (-len(item["heads"]), item["predicate"])):
        lines.append(f"- `{row['predicate']}`: {', '.join(row['heads'])}; median |OV cosine| `{row['median_abs_ov_cosine']:.3f}`.")
    lines += ["", "## Selected rules for ranked heads", ""]
    for row in ranked_rows:
        lines.append(f"- **L{row['layer']}H{row['head']}**: `{', '.join(row['tree']['selected_names'])}`")
    lines += ["", "## Guardrails", "", "These trees explain the magnitude of a frozen local operator effect, not the sign, output semantics, or final loss. The positive class is defined separately as the top 2% in each split, so AUROC measures rank transfer rather than calibration. Predicates encode substantial human prior knowledge. Greedy trees can select correlated proxies and predicate identity can be unstable even when predictions transfer. A reused predicate with low OV cosine is evidence that one readable variable participates in distinct native transforms; it does not prove that the predicate is the model's unique internal ontology."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ordered = ranked_rows; x = np.arange(len(ordered)); labels = [f"L{row['layer']}H{row['head']}" for row in ordered]
    for offset, method in enumerate(("baseline", "lexical", "contextual")): axes[0].bar(x + (offset - 1) * .25, [row["aucs"][method] for row in ordered], width=.24, label=method)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Held-out tail AUROC", title="Readable read-function programs"); axes[0].tick_params(axis="x", rotation=45); axes[0].legend()
    axes[1].scatter([row["importance"] for row in rows], [row["aucs"]["contextual"] for row in rows], c=["tab:red" if row["ranked"] else "tab:blue" for row in rows]); axes[1].set(xlabel="Derivative importance", ylabel="Contextual AUROC", title="Ranked vs bottom controls")
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "ranked_aucs": {f"L{row['layer']}H{row['head']}": row["aucs"] for row in ranked_rows}}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
