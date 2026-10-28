#!/usr/bin/env python3
"""Test whether local Layer-8 causal axes recur within exact prediction families."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_behavioral_bottleneck import (
    normalized_rows,
    orthonormal_row_space,
    subspace_overlap,
)
from run_bilin18_layer8_direct_tensor_jacobian import (
    analytic_suffix,
    bilinear_mlp_jvp,
    output_fisher_response,
)
from run_bilin18_layer8_full_tensor_gram_rank import gram_eigensystem
from run_bilin18_layer8_router_causal_validation import prefix_to_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_exact_prediction_recurrence.json")
OUTPUT = TENSOR_ROOT / "layer8_exact_prediction_recurrence.json"
REPORT = TENSOR_ROOT / "LAYER8_EXACT_PREDICTION_RECURRENCE.md"
FIGURE = TENSOR_ROOT / "layer8_exact_prediction_recurrence.png"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def best_permutation(first, second):
    similarity = np.abs(first @ second.T)
    return max(
        itertools.permutations(range(len(first))),
        key=lambda permutation: np.mean([
            similarity[index, permutation[index]] for index in range(len(first))
        ]),
    )


def pair_metrics(first, second):
    permutation = best_permutation(first["_fisher"], second["_fisher"])
    jaccards = []
    for axis, target in enumerate(permutation):
        left, right = set(first["_tokens"][axis]), set(second["_tokens"][target])
        jaccards.append(len(left & right) / max(len(left | right), 1))
    return {
        "fisher": subspace_overlap(first["_fisher"], second["_fisher"]),
        "logit": subspace_overlap(first["_logit"], second["_logit"]),
        "input": subspace_overlap(first["_input"], second["_input"]),
        "token_jaccard": float(np.mean(jaccards)),
    }


def summarize_pairs(pairs):
    return {key: median([row[key] for row in pairs]) for key in pairs[0]}


@torch.inference_mode()
def scan_predictions(model, rows, batch_size):
    predictions = []
    dimension = model.config.n_embd
    for start in range(0, len(rows), batch_size):
        ids = torch.as_tensor(
            np.asarray(rows[start:start + batch_size, :128], dtype=np.int64), device="cuda",
        )
        x0 = F.rms_norm(model.embedding(ids), (dimension,))
        x, bus = x0, None
        for block in model.blocks:
            x, bus = block(x, bus, x0)
        logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (dimension,)))[:, -1].float() / 30)
        predictions.extend(logits.argmax(-1).cpu().tolist())
    return np.asarray(predictions, dtype=np.int64)


@torch.inference_mode()
def axis_record(model, block, basis, ids, position, axes, top_tokens):
    x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
    native_write = block.mlp.Down(
        block.mlp.Left(normalized) * block.mlp.Right(normalized)
    ) + block.mlp.Down_bias
    x8 = pre + native_write
    x = normalized[0, position].float()
    write_tangents = bilinear_mlp_jvp(block, x, basis.T)
    logits, logit_tangents, _ = analytic_suffix(
        model, x8, x0, value_bus, position, write_tangents, include_attention=True,
    )
    response = output_fisher_response(logits, logit_tangents)
    _, vectors = gram_eigensystem(response)
    probability = F.softmax(logits.float(), -1)
    centered = logit_tangents - (logit_tangents * probability[None]).sum(-1, keepdim=True)
    logit_axes = vectors[:, :axes].T @ centered
    token_sets = []
    for axis in range(axes):
        probability_delta = probability * logit_axes[axis]
        positive = torch.topk(probability_delta, top_tokens).indices.tolist()
        negative = torch.topk(-probability_delta, top_tokens).indices.tolist()
        token_sets.append(positive + negative)
    return {
        "_fisher": normalized_rows((response @ vectors[:, :axes]).T).cpu().numpy(),
        "_logit": orthonormal_row_space(logit_axes).cpu().numpy(),
        "_input": vectors[:, :axes].T.cpu().numpy(),
        "_tokens": token_sets,
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    tensor_config = {**source_config, "positions": source_config["discovery_positions"]}
    discovery_x, _ = collect(model, 8, source_config["tensor_discovery"], tensor_config)
    _, _, tensor_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = tensor_vh.T[:, :source_config["dimension"]]
    block = model.blocks[8]
    position = source_config["causal_position"]
    private, results = {}, {}

    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        predictions = scan_predictions(model, rows, config["scan_batch_size"])
        rng = np.random.default_rng(config["seed"] + corpus_index)
        groups = {}
        for token_id in config["prediction_token_ids"]:
            candidates = np.flatnonzero(predictions == token_id)
            if len(candidates) < config["sites_per_token"]:
                raise RuntimeError(f"{corpus} token {token_id} has only {len(candidates)} sites")
            selected = rng.choice(candidates, config["sites_per_token"], replace=False)
            records = []
            for site_number, row_index in enumerate(selected):
                ids = torch.as_tensor(
                    np.asarray(rows[row_index, :position + 1], dtype=np.int64), device="cuda",
                )[None]
                record = axis_record(
                    model, block, basis, ids, position, config["axes"], config["top_tokens"],
                )
                record.update({"row": int(row_index), "prediction_token_id": token_id})
                records.append(record)
                print(corpus, tokenizer.decode([token_id]).replace("\n", "\\n"), site_number + 1, flush=True)
            groups[str(token_id)] = records
        private[corpus] = groups
        within, between = [], []
        for token, records in groups.items():
            within.extend(pair_metrics(records[i], records[j]) for i in range(len(records)) for j in range(i + 1, len(records)))
        tokens = list(groups)
        for first_index in range(len(tokens)):
            for second_index in range(first_index + 1, len(tokens)):
                between.extend(pair_metrics(first, second) for first in groups[tokens[first_index]] for second in groups[tokens[second_index]])
        within_summary, between_summary = summarize_pairs(within), summarize_pairs(between)
        results[corpus] = {
            "counts": {str(token): int((predictions == token).sum()) for token in config["prediction_token_ids"]},
            "within": within_summary,
            "between": between_summary,
            "advantage": {key: within_summary[key] - between_summary[key] for key in within_summary},
        }

    cross_same, cross_different = [], []
    for token in map(str, config["prediction_token_ids"]):
        cross_same.extend(pair_metrics(first, second) for first in private["fineweb"][token] for second in private["wikitext"][token])
    token_strings = list(map(str, config["prediction_token_ids"]))
    for first_token in token_strings:
        for second_token in token_strings:
            if first_token != second_token:
                cross_different.extend(pair_metrics(first, second) for first in private["fineweb"][first_token] for second in private["wikitext"][second_token])
    cross = {
        "same_prediction": summarize_pairs(cross_same),
        "different_prediction": summarize_pairs(cross_different),
    }
    cross["advantage"] = {
        key: cross["same_prediction"][key] - cross["different_prediction"][key]
        for key in cross["same_prediction"]
    }
    random_expectation = config["axes"] / source_config["dimension"]
    hypotheses = {
        "E1_exact_prediction_improves_fisher_recurrence": {"supported": all(results[c]["advantage"]["fisher"] >= 0.05 for c in results)},
        "E2_exact_prediction_improves_logit_recurrence": {"supported": all(results[c]["advantage"]["logit"] >= 0.05 for c in results)},
        "E3_exact_prediction_improves_input_recurrence": {"supported": all(results[c]["advantage"]["input"] >= 0.05 for c in results)},
        "E4_exact_prediction_stabilizes_token_labels": {"supported": all(results[c]["within"]["token_jaccard"] >= 0.10 for c in results)},
        "E5_input_verbs_transfer_cross_corpus": {"supported": cross["same_prediction"]["input"] >= 0.10},
        "E6_input_recurrence_exceeds_random_subspaces": {"supported": all(results[c]["within"]["input"] >= 3 * random_expectation for c in results)},
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "random_input_overlap_expectation": random_expectation,
        "results": results,
        "cross_corpus": cross,
    }, indent=2))

    labels = {str(token): tokenizer.decode([token]).replace("\n", "\\n") for token in config["prediction_token_ids"]}
    lines = [
        "# Exact-prediction conditioning of local Layer-8 axes",
        "",
        "Four prediction tokens were frozen after a count-only scan: ` the`, `.`, `,`, and newline. Eight sites per token and corpus were then selected without inspecting axes. The test asks whether the locally strongest causal subspaces recur after conditioning on the exact native top prediction.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Exact prediction recurrence](layer8_exact_prediction_recurrence.png)",
        "",
        "| Corpus | Comparison | Fisher overlap | Logit overlap | Input overlap | Token Jaccard |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for comparison in ("within", "between", "advantage"):
            row = result[comparison]
            lines.append(f"| {corpus} | {comparison} | {row['fisher']:.3f} | {row['logit']:.3f} | {row['input']:.3f} | {row['token_jaccard']:.3f} |")
    for comparison in ("same_prediction", "different_prediction", "advantage"):
        row = cross[comparison]
        lines.append(f"| cross-corpus | {comparison} | {row['fisher']:.3f} | {row['logit']:.3f} | {row['input']:.3f} | {row['token_jaccard']:.3f} |")
    lines += [
        "",
        f"The random rank-{config['axes']} input-subspace overlap expectation is `{random_expectation:.4f}`. Prediction counts were " + "; ".join(
            f"{corpus}: " + ", ".join(f"`{labels[token]}`={count}" for token, count in result["counts"].items())
            for corpus, result in results.items()
        ) + ".",
        "",
        "Exact output identity is a stronger conditioning variable than coarse token class, but any gain confined to Fisher/output coordinates can arise from the shared softmax readout. Recurrent internal verbs require input-subspace recurrence above the random baseline and cross-corpus transfer.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes_plot = plt.subplots(1, 2, figsize=(10.5, 4.2))
    metrics = ("fisher", "logit", "input")
    x = np.arange(len(metrics))
    width = 0.35
    for offset, corpus in ((-width / 2, "fineweb"), (width / 2, "wikitext")):
        axes_plot[0].bar(x + offset, [results[corpus]["within"][metric] for metric in metrics], width, label=corpus)
        axes_plot[1].bar(x + offset, [results[corpus]["advantage"][metric] for metric in metrics], width, label=corpus)
    axes_plot[0].set_title("Within exact prediction")
    axes_plot[1].set_title("Within minus between")
    for axis in axes_plot:
        axis.set_xticks(x, metrics)
        axis.set_ylabel("subspace overlap")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
