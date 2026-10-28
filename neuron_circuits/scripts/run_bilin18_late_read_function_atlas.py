#!/usr/bin/env python3
"""Large-corpus atlas of late attention read functions and MDL-style controls."""

from __future__ import annotations

import heapq
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import (
    CATEGORIES, distance_ids, edge_type_ids, rmsnorm_pullback_readers, token_categories,
)


CONFIG = Path("configs/bilin18_late_read_function_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_late_read_function_atlas.json"
REPORT = TENSOR_ROOT / "BILIN18_LATE_READ_FUNCTION_ATLAS.md"
FIGURE = TENSOR_ROOT / "bilin18_late_read_function_atlas.png"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def top_jaccard(a, b, count):
    left = set(np.argsort(np.asarray(a).reshape(-1))[-count:].tolist())
    right = set(np.argsort(np.asarray(b).reshape(-1))[-count:].tolist())
    return float(len(left & right) / len(left | right))


def program_predictions(matrix):
    """Increasingly expressive, readable predictions of a distance x category table."""
    matrix = np.asarray(matrix, dtype=np.float64)
    row = matrix.mean(1, keepdims=True)
    col = matrix.mean(0, keepdims=True)
    mean = matrix.mean()
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    return {
        "constant": np.full_like(matrix, mean),
        "distance": np.broadcast_to(row, matrix.shape),
        "category": np.broadcast_to(col, matrix.shape),
        "additive": np.maximum(row + col - mean, 0),
        "rank1": np.maximum(s[0] * np.outer(u[:, 0], vh[0]), 0),
    }


def scaled_fidelity(prediction, target):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    scale = float(prediction @ target / max(prediction @ prediction, 1e-30))
    error = np.square(target - scale * prediction).sum()
    return float(1 - error / max(np.square(target).sum(), 1e-30))


def frequency_categories(corpus, vocab_size, count):
    frequencies = np.bincount(np.asarray(corpus).reshape(-1), minlength=vocab_size)
    order = np.argsort(frequencies, kind="mergesort")
    ranks = np.empty(vocab_size, dtype=np.int64)
    ranks[order] = np.arange(vocab_size)
    categories = np.minimum(ranks * count // vocab_size, count - 1)
    return categories, frequencies


def shuffled_categories(human, frequencies, seed):
    """Shuffle human labels only within coarse frequency strata."""
    rng = np.random.default_rng(seed)
    result = np.asarray(human).copy()
    log_frequency = np.floor(np.log2(np.asarray(frequencies) + 1)).astype(np.int64)
    for level in np.unique(log_frequency):
        members = np.flatnonzero(log_frequency == level)
        result[members] = result[rng.permutation(members)]
    return result


def decode_route(index, distance_bins, category_names):
    category_count = len(category_names)
    distance, category = divmod(int(index), category_count)
    low, high = distance_bins[distance], distance_bins[distance + 1]
    return {"distance": f"{low}" if high == low + 1 else f"{low}-{high - 1}", "category": category_names[category]}


def push_examples(heaps, layer, edge_energy, pattern, ids, row_offset, positions, limit):
    # Preserve a small set of the largest exact local edge effects per native head.
    batch, pos_count, heads, length = edge_energy.shape
    for head in range(heads):
        flat = edge_energy[:, :, head].reshape(-1)
        count = min(limit, flat.numel())
        values, indices = torch.topk(flat, count)
        for value, index in zip(values.tolist(), indices.tolist()):
            batch_index, remainder = divmod(index, pos_count * length)
            position_index, source = divmod(remainder, length)
            destination = positions[position_index]
            if source > destination:
                continue
            item = (
                float(value), row_offset + batch_index, int(source), int(destination),
                int(ids[batch_index, source]), int(ids[batch_index, destination]),
                float(pattern[batch_index, position_index, head, source]),
            )
            heap = heaps[(layer, head)]
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)


@torch.inference_mode()
def collect(model, raw, row_start, category_maps, config, probes, keep_examples):
    layers = set(config["layers"])
    heads = model.config.n_head
    dimension = model.config.n_embd
    head_dim = dimension // heads
    distances = len(config["distance_bins"]) - 1
    distance = distance_ids(config["positions"], config["sequence_length"], config["distance_bins"], "cuda")
    scores = {
        name: {layer: torch.zeros((heads, distances, len(names)), device="cuda", dtype=torch.float64)
               for layer in layers}
        for name, (_, names) in category_maps.items()
    }
    maximum_error = 0.0
    heaps = {(layer, head): [] for layer in layers for head in range(heads)}
    for start in range(0, len(raw), config["batch_size"]):
        ids = torch.as_tensor(raw[start:start + config["batch_size"]], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,))
        x, bus = x0, None
        for layer, block in enumerate(model.blocks[:max(layers) + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            if layer in layers:
                pre = mixed[:, config["positions"]].float() + details["write"][:, config["positions"]].float()
                readers = rmsnorm_pullback_readers(pre, probes[layer])
                projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
                head_readers = torch.einsum("bpqo,hdo->bphqd", readers, projection)
                value_effect = torch.einsum("bkhd,bphqd->bphqk", details["mixed_value"].float(), head_readers)
                pattern = details["pattern"][:, :, config["positions"]].permute(0, 2, 1, 3)
                edge_effect = value_effect * pattern[:, :, :, None]
                full_effect = edge_effect.sum((2, 4))
                if keep_examples:
                    push_examples(
                        heaps, layer, edge_effect.square().sum(3), pattern, ids,
                        row_start + start, config["positions"], config["example_count"],
                    )
                for name, (lookup, names) in category_maps.items():
                    category = torch.as_tensor(lookup[np.asarray(raw[start:start + len(ids)])], device="cuda")
                    type_ids = edge_type_ids(category, distance, heads, len(names))
                    n = len(ids) * len(config["positions"])
                    grouped = torch.zeros(n, heads * distances * len(names), len(probes[layer]), device="cuda")
                    flat_ids = type_ids.flatten(0, 1).reshape(n, -1)
                    flat_effect = edge_effect.flatten(0, 1).permute(0, 1, 3, 2).reshape(n, -1, len(probes[layer]))
                    valid = flat_ids >= 0
                    grouped.scatter_add_(1, flat_ids.clamp_min(0)[:, :, None].expand_as(flat_effect), flat_effect * valid[:, :, None])
                    local_scores = grouped.square().sum((0, 2)).reshape(heads, distances, len(names)).double()
                    scores[name][layer] += local_scores
                    recomposed = grouped.sum(1).reshape_as(full_effect)
                    error = (full_effect - recomposed).norm() / full_effect.norm().clamp_min(1e-30)
                    maximum_error = max(maximum_error, float(error))
            pre_full = mixed + details["write"]
            state = F.rms_norm(pre_full, (dimension,))
            x = pre_full + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        if start % (config["batch_size"] * 16) == 0:
            print(f"  rows {start}-{min(start + config['batch_size'], len(raw))}", flush=True)
    cpu_scores = {name: {layer: value.cpu().numpy() for layer, value in by_layer.items()} for name, by_layer in scores.items()}
    return cpu_scores, heaps, maximum_error


def render_example(item, corpus, tokenizer, human):
    score, row, source, destination, source_id, destination_id, attention = item
    local = np.asarray(corpus[row, max(0, source - 4):min(corpus.shape[1], destination + 3)])
    text = tokenizer.decode(local.tolist()).replace("\n", "\\n").replace("\t", "\\t")
    return {
        "score": score, "row": row, "source_position": source, "destination_position": destination,
        "distance": destination - source, "source_token": tokenizer.decode([source_id]).replace("\n", "\\n"),
        "destination_token": tokenizer.decode([destination_id]).replace("\n", "\\n"),
        "source_category": CATEGORIES[int(human[source_id])], "attention": attention, "context": text,
    }


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(config["corpus"], mmap_mode="r")
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    human = token_categories(flags)
    frequency, counts = frequency_categories(corpus, tokenizer.vocab_size, config["category_count"])
    shuffled = shuffled_categories(human, counts, config["seed"])
    names = [f"frequency_quantile_{index}" for index in range(config["category_count"])]
    category_maps = {
        "human": (human, list(CATEGORIES)),
        "frequency": (frequency, names),
        "shuffled_human": (shuffled, list(CATEGORIES)),
    }
    discovery = np.asarray(corpus[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    probes = {layer: operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)[0] for layer in config["layers"]}
    print("discovery split", flush=True)
    score_a, _, error_a = collect(model, discovery, config["discovery_start"], category_maps, config, probes, False)
    print("evaluation split", flush=True)
    score_b, heaps, error_b = collect(model, evaluation, config["evaluation_start"], category_maps, config, probes, True)

    prior = json.loads(Path(config["ranked_heads_artifact"]).read_text())
    top8 = {tuple(pair) for pair in prior["selected"]["top8"]}
    importance = {(row["layer"], row["head"]): row["discovery_importance"] for row in prior["importance"]}
    rows = []
    for layer in config["layers"]:
        for head in range(model.config.n_head):
            schemes = {}
            for scheme, (_, category_names) in category_maps.items():
                first = np.sqrt(np.maximum(score_a[scheme][layer][head], 0))
                second = np.sqrt(np.maximum(score_b[scheme][layer][head], 0))
                literal = scaled_fidelity(first, second)
                fits = {name: scaled_fidelity(prediction, second) for name, prediction in program_predictions(first).items()}
                best_name = max(fits, key=fits.get)
                schemes[scheme] = {
                    "joint_cosine": cosine(first, second), "literal_fidelity": literal,
                    "simple_fidelities": fits, "best_simple": best_name,
                    "best_simple_fidelity": fits[best_name],
                    "top_route_jaccard": top_jaccard(first, second, config["top_route_budget"]),
                    "top_routes": [decode_route(index, config["distance_bins"], category_names)
                                   for index in np.argsort(first.reshape(-1))[::-1][:8]],
                }
            key = (layer, head)
            examples = [render_example(item, corpus, tokenizer, human) for item in sorted(heaps[key], reverse=True)]
            rows.append({"layer": layer, "head": head, "derivative_ranked_top8": key in top8,
                         "derivative_importance": importance[key], "schemes": schemes, "examples": examples})

    ranked = [row for row in rows if row["derivative_ranked_top8"]]
    others = [row for row in rows if not row["derivative_ranked_top8"]]
    value = lambda row, scheme, metric: row["schemes"][scheme][metric]
    median_ranked_joint = float(np.median([value(row, "human", "joint_cosine") for row in ranked]))
    median_other_joint = float(np.median([value(row, "human", "joint_cosine") for row in others]))
    medians_simple = {scheme: float(np.median([value(row, scheme, "best_simple_fidelity") for row in ranked])) for scheme in category_maps}
    importance_readability = spearman(
        [row["derivative_importance"] for row in rows],
        [value(row, "human", "best_simple_fidelity") for row in rows],
    )
    compact_count = sum(value(row, "human", "best_simple_fidelity") >= value(row, "human", "literal_fidelity") - .10 for row in ranked)
    replicated_count = sum(value(row, "human", "top_route_jaccard") >= .50 for row in ranked)
    maximum_error = max(error_a, error_b)
    hypotheses = {
        "H0_exact_partitions": {"supported": maximum_error <= 1e-5, "maximum_error": maximum_error},
        "H1_ranked_profiles_are_stable": {"supported": median_ranked_joint >= .90, "median_joint_cosine": median_ranked_joint},
        "H2_human_grammar_beats_controls": {"supported": all(medians_simple["human"] >= medians_simple[name] + .05 for name in ("frequency", "shuffled_human")), "median_simple_fidelity": medians_simple},
        "H3_derivative_importance_predicts_readability": {"supported": importance_readability >= .30, "spearman": importance_readability},
        "H4_ranked_heads_have_compact_programs": {"supported": compact_count >= 4, "count": compact_count},
        "H5_route_types_replicate": {"supported": replicated_count >= 4, "count": replicated_count},
        "H6_ranked_heads_are_more_stable_than_rest": {"supported": median_ranked_joint >= median_other_joint + .05, "ranked_median": median_ranked_joint, "other_median": median_other_joint},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "rows": rows}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = [
        "# Large-corpus atlas of late read functions", "",
        "This test asks whether late heads selected by an exact local derivative program admit shorter, reusable descriptions on two disjoint 98,304-token FineWeb panels. A function invocation is described by relative-distance address, source-variable class, and its native head transform. Human token classes are compared with equally sized frequency bins and human labels shuffled within frequency strata.", "",
        "## Locked results", "",
    ]
    for name, result in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if result['supported'] else 'not supported'}.**")
    lines += ["", "## Ranked heads", "", "| Head | joint cos | literal fit | best simple program | simple fit | route Jaccard | leading route |", "|---|---:|---:|---|---:|---:|---|"]
    for row in sorted(ranked, key=lambda item: item["derivative_importance"], reverse=True):
        human_row = row["schemes"]["human"]
        route = human_row["top_routes"][0]
        lines.append(f"| L{row['layer']}H{row['head']} | {human_row['joint_cosine']:.3f} | {human_row['literal_fidelity']:.3f} | {human_row['best_simple']} | {human_row['best_simple_fidelity']:.3f} | {human_row['top_route_jaccard']:.3f} | {route['distance']} / {route['category']} |")
    lines += ["", "## Concrete high-effect edges", ""]
    for row in sorted(ranked, key=lambda item: item["derivative_importance"], reverse=True):
        lines.append(f"### L{row['layer']}H{row['head']}")
        for example in row["examples"][:4]:
            lines.append(f"- `{example['source_token']}` ({example['source_category']}, distance {example['distance']}) -> `{example['destination_token']}`; attention `{example['attention']:.3f}`; context: `{example['context']}`")
        lines.append("")
    lines += [
        "## Interpretation guardrails", "",
        "The route score is the squared norm of a signed attention-edge effect on a frozen weight-derived local operator sketch. It is not attention weight alone and not final-token loss. Squared grouped effects contain within-type cross terms, so scores are useful route profiles rather than an additive conservation law. A stable literal table is a reproducible lookup, not automatically a readable feature. A simple distance/category model is evidence for a shorter program only when it survives the frequency and shuffled-label controls. High-effect examples are selected after measuring magnitude and therefore illustrate, rather than independently validate, a label.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    labels = [f"L{row['layer']}H{row['head']}" for row in sorted(ranked, key=lambda item: item["derivative_importance"], reverse=True)]
    ordered = sorted(ranked, key=lambda item: item["derivative_importance"], reverse=True)
    axes[0].bar(np.arange(len(ordered)), [value(row, "human", "joint_cosine") for row in ordered]); axes[0].set(title="Route-profile replication", ylabel="split cosine", xticks=np.arange(len(ordered)), xticklabels=labels); axes[0].tick_params(axis="x", rotation=55)
    x = np.arange(len(category_maps)); axes[1].bar(x, [medians_simple[name] for name in category_maps]); axes[1].set(title="Readable-program controls", ylabel="median held-out fidelity", xticks=x, xticklabels=list(category_maps)); axes[1].tick_params(axis="x", rotation=30)
    axes[2].scatter([row["derivative_importance"] for row in rows], [value(row, "human", "best_simple_fidelity") for row in rows], c=["tab:red" if row["derivative_ranked_top8"] else "tab:blue" for row in rows]); axes[2].set(title=f"Importance vs readability (rho={importance_readability:.2f})", xlabel="derivative importance", ylabel="simple-program fidelity")
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "ranked_heads": labels}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
