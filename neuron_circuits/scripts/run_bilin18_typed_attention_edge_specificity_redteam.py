#!/usr/bin/env python3
"""Red-team which coordinates of the typed attention-edge vocabulary matter."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import (
    CATEGORIES,
    discovery_scores,
    evaluate_supports,
    token_categories,
)


CONFIG = Path("configs/bilin18_typed_attention_edge_specificity_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_typed_attention_edge_specificity_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_TYPED_ATTENTION_EDGE_SPECIFICITY_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_typed_attention_edge_specificity_redteam.png"


def encode(head, distance, category, distances, categories):
    return head * distances * categories + distance * categories + category


def decode(indices, distances, categories):
    indices = np.asarray(indices)
    head = indices // (distances * categories)
    remainder = indices % (distances * categories)
    return head, remainder // categories, remainder % categories


def randomized_support(selected, axis, heads, distances, categories, generator):
    h, d, c = decode(selected, distances, categories)
    output = []
    if axis == "category":
        for head, distance in sorted(set(zip(h, d))):
            count = int(np.sum((h == head) & (d == distance)))
            for category in generator.choice(categories, count, replace=False): output.append(encode(head, distance, category, distances, categories))
    elif axis == "distance":
        for head, category in sorted(set(zip(h, c))):
            count = int(np.sum((h == head) & (c == category)))
            for distance in generator.choice(distances, count, replace=False): output.append(encode(head, distance, category, distances, categories))
    elif axis == "head":
        for distance, category in sorted(set(zip(d, c))):
            count = int(np.sum((d == distance) & (c == category)))
            for head in generator.choice(heads, count, replace=False): output.append(encode(head, distance, category, distances, categories))
    else: raise ValueError(axis)
    return np.unique(output)


def closure(selected, dropped, heads, distances, categories):
    h, d, c = decode(selected, distances, categories); output = []
    if dropped == "category":
        for head, distance in set(zip(h, d)):
            output.extend(encode(head, distance, category, distances, categories) for category in range(categories))
    elif dropped == "distance":
        for head, category in set(zip(h, c)):
            output.extend(encode(head, distance, category, distances, categories) for distance in range(distances))
    elif dropped == "head":
        for distance, category in set(zip(d, c)):
            output.extend(encode(head, distance, category, distances, categories) for head in range(heads))
    return np.unique(output)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    categories_by_token = token_categories(flags)
    source = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(source[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(source[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; heads = model.config.n_head; distances = len(config["distance_bins"]) - 1; categories = len(CATEGORIES)
    for layer in config["layers"]:
        print("typed specificity layer", layer, flush=True)
        probes, _, _ = operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)
        scores, _ = discovery_scores(model, layer, discovery, categories_by_token, config, probes)
        order = scores.argsort(descending=True).cpu().numpy(); supports = []; names = []
        generator = np.random.default_rng(config["seed"] + 100 + layer)
        for budget in config["budgets"]:
            selected = order[:budget]; supports.append(selected); names.append(f"selected_{budget}")
            for axis in ("category", "distance", "head"):
                for index in range(config["random_supports"]):
                    supports.append(randomized_support(selected, axis, heads, distances, categories, generator)); names.append(f"{axis}_{budget}_{index}")
                supports.append(closure(selected, axis, heads, distances, categories)); names.append(f"closure_{axis}_{budget}")
        recoveries, error = evaluate_supports(model, layer, evaluation, categories_by_token, supports, config, probes)
        by_name = dict(zip(names, recoveries)); layer_result = {"partition_error": error, "budgets": {}}
        for budget in config["budgets"]:
            row = {"selected": by_name[f"selected_{budget}"]}
            for axis in ("category", "distance", "head"):
                random = [by_name[f"{axis}_{budget}_{index}"] for index in range(config["random_supports"])]
                row[f"{axis}_random_p50"] = float(np.median(random)); row[f"{axis}_random_p95"] = float(np.quantile(random, .95))
                row[f"{axis}_closure"] = by_name[f"closure_{axis}_{budget}"]
                row[f"{axis}_closure_size"] = int(len(supports[names.index(f"closure_{axis}_{budget}")]))
            layer_result["budgets"][str(budget)] = row
        results[str(layer)] = layer_result
    row32 = {str(layer): results[str(layer)]["budgets"]["32"] for layer in config["layers"]}
    hypotheses = {
        "S0_typed_vocabulary_replicates": {"supported": all(results[str(layer)]["budgets"]["32"]["selected"] >= .50 and results[str(layer)]["budgets"]["128"]["selected"] >= .80 for layer in config["layers"])},
        "S1_source_category_is_specific": {"supported": sum(row32[str(layer)]["selected"] >= row32[str(layer)]["category_random_p95"] + .05 for layer in config["layers"]) >= 4},
        "S2_distance_is_specific": {"supported": sum(row32[str(layer)]["selected"] >= row32[str(layer)]["distance_random_p95"] + .05 for layer in config["layers"]) >= 4},
        "S3_head_is_specific": {"supported": sum(row32[str(layer)]["selected"] >= row32[str(layer)]["head_random_p95"] + .05 for layer in config["layers"]) >= 4},
        "S4_category_condition_excludes_harmful_edges": {"supported": sum(row32[str(layer)]["selected"] >= row32[str(layer)]["category_closure"] + .10 for layer in config["layers"]) >= 4},
        "S5_distance_condition_excludes_harmful_edges": {"supported": sum(row32[str(layer)]["selected"] >= row32[str(layer)]["distance_closure"] + .10 for layer in config["layers"]) >= 4},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Specificity red team for the typed attention-edge vocabulary", "", "The typed vocabulary is reselected and evaluated on later disjoint FineWeb. Three controls preserve two coordinates of every selected type while randomizing the third: source category within head-distance, distance within head-category, or head within distance-category. Closure controls drop one condition and keep every matching edge of that broader type.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Typed specificity](bilin18_typed_attention_edge_specificity_redteam.png)", "", "| Layer | Selected32 | Category random p95 / closure(size) | Distance random p95 / closure(size) | Head random p95 / closure(size) |", "|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = row32[str(layer)]
        lines.append(f"| {layer} | {row['selected']:.3f} | {row['category_random_p95']:.3f} / {row['category_closure']:.3f} ({row['category_closure_size']}) | {row['distance_random_p95']:.3f} / {row['distance_closure']:.3f} ({row['distance_closure_size']}) | {row['head_random_p95']:.3f} / {row['head_closure']:.3f} ({row['head_closure_size']}) |")
    lines += ["", "## Guardrails", "", "Randomized supports preserve group counts but can contain fewer unique types after collisions; categories also differ greatly in frequency. Closures use more executed edges and therefore test descriptive necessity, not equal-compute efficiency. If a closure improves recovery, the dropped coordinate was useful for ranking individual types but unnecessary for the broader executable program. All rankings remain linearized at native state; evaluation recomputes nonlinear RMSNorm exactly."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    layers = config["layers"]
    axes[0].plot(layers, [row32[str(layer)]["selected"] for layer in layers], marker="o", label="selected")
    for axis_name in ("category", "distance", "head"): axes[0].plot(layers, [row32[str(layer)][f"{axis_name}_random_p95"] for layer in layers], marker="o", label=f"{axis_name} p95")
    axes[1].plot(layers, [row32[str(layer)]["selected"] for layer in layers], marker="o", label="selected")
    for axis_name in ("category", "distance", "head"): axes[1].plot(layers, [row32[str(layer)][f"{axis_name}_closure"] for layer in layers], marker="o", label=f"drop {axis_name}")
    axes[0].set(xlabel="Layer", ylabel="Recovery", title="Matched coordinate randomization"); axes[1].set(xlabel="Layer", ylabel="Recovery", title="Broader closure programs")
    for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=8)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "layer32": row32}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
