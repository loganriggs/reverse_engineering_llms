#!/usr/bin/env python3
"""Large-corpus weight-derived route atlas for newly selected late heads."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_late_read_function_atlas import (
    collect, cosine, decode_route, frequency_categories, program_predictions,
    render_example, scaled_fidelity, shuffled_categories, top_jaccard,
)
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, token_categories


CONFIG = Path("configs/bilin18_selected_late_head_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_selected_late_head_atlas.json"
REPORT = TENSOR_ROOT / "BILIN18_SELECTED_LATE_HEAD_ATLAS.md"
FIGURE = TENSOR_ROOT / "bilin18_selected_late_head_atlas.png"


def main():
    config = json.loads(CONFIG.read_text()); selected = [tuple(item) for item in config["selected_heads"]]; layers = sorted({layer for layer, _ in selected}); config["layers"] = layers
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); corpus = np.load(config["corpus"], mmap_mode="r")
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); human = token_categories(flags)
    frequency, counts = frequency_categories(corpus, tokenizer.vocab_size, config["category_count"]); shuffled = shuffled_categories(human, counts, config["seed"]); frequency_names = [f"frequency_quantile_{index}" for index in range(config["category_count"])]
    category_maps = {"human": (human, list(CATEGORIES)), "frequency": (frequency, frequency_names), "shuffled_human": (shuffled, list(CATEGORIES))}
    discovery = np.asarray(corpus[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64); evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    probes = {layer: operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)[0] for layer in layers}
    print("selected late atlas discovery", flush=True); score_a, _, error_a = collect(model, discovery, config["discovery_start"], category_maps, config, probes, False)
    print("selected late atlas evaluation", flush=True); score_b, heaps, error_b = collect(model, evaluation, config["evaluation_start"], category_maps, config, probes, True)
    rows = []
    for layer, head in selected:
        schemes = {}
        for scheme, (_, category_names) in category_maps.items():
            first = np.sqrt(np.maximum(score_a[scheme][layer][head], 0)); second = np.sqrt(np.maximum(score_b[scheme][layer][head], 0)); predictions = program_predictions(first); fits = {name: scaled_fidelity(prediction, second) for name, prediction in predictions.items()}; best = max(fits, key=fits.get)
            schemes[scheme] = {"joint_cosine": cosine(first, second), "literal_fidelity": scaled_fidelity(first, second), "best_simple": best, "best_simple_fidelity": fits[best], "simple_fidelities": fits, "top_route_jaccard": top_jaccard(first, second, config["top_route_budget"]), "top_routes": [decode_route(index, config["distance_bins"], category_names) for index in np.argsort(first.reshape(-1))[::-1][:8]]}
        examples = [render_example(item, corpus, tokenizer, human) for item in sorted(heaps[(layer, head)], reverse=True)]
        rows.append({"layer": layer, "head": head, "schemes": schemes, "examples": examples})
    maximum_error = max(error_a, error_b)
    human_advantage = {f"L{row['layer']}H{row['head']}": row["schemes"]["human"]["best_simple_fidelity"] - max(row["schemes"][name]["best_simple_fidelity"] for name in ("frequency", "shuffled_human")) for row in rows}
    hypotheses = {
        "W0_exact_edge_partitions": {"supported": maximum_error <= 1e-5, "observed": maximum_error},
        "W1_selected_route_profiles_replicate": {"supported": all(row["schemes"]["human"]["joint_cosine"] >= .90 for row in rows), "observed": {f"L{row['layer']}H{row['head']}": row["schemes"]["human"]["joint_cosine"] for row in rows}},
        "W2_selected_heads_have_compact_read_programs": {"supported": all(row["schemes"]["human"]["best_simple_fidelity"] >= row["schemes"]["human"]["literal_fidelity"] - .10 for row in rows), "observed": {f"L{row['layer']}H{row['head']}": {"simple": row["schemes"]["human"]["best_simple_fidelity"], "literal": row["schemes"]["human"]["literal_fidelity"]} for row in rows}},
        "W3_human_labels_add_information": {"supported": any(value >= .05 for value in human_advantage.values()), "observed": human_advantage},
        "W4_route_support_is_not_one_literal_lookup": {"supported": all(row["schemes"]["human"]["joint_cosine"] >= .90 and row["schemes"]["human"]["top_route_jaccard"] < .95 for row in rows), "observed": {f"L{row['layer']}H{row['head']}": row["schemes"]["human"]["top_route_jaccard"] for row in rows}},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "rows": rows}, indent=2, allow_nan=False))
    lines = ["# Large-corpus route atlas for newly implicated late heads", "", "L12H4 and L15H5 were selected only after the semantic construction experiment. Their native QK×QK×OV edge effects are now summarized on two disjoint 98,304-token FineWeb panels using frozen weight-derived operator probes. Human source-token categories are compared with frequency bins and frequency-stratified shuffled labels.", "", "## Locked decisions", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{row['observed']}`" for name, row in hypotheses.items())
    lines += ["", "![Selected late head atlas](bilin18_selected_late_head_atlas.png)", "", "| Head | Joint cosine | Literal fidelity | Best human program | Simple fidelity | Route Jaccard | Leading routes |", "|---|---:|---:|---|---:|---:|---|"]
    for row in rows:
        h = row["schemes"]["human"]; leading = ", ".join(f"{item['distance']}/{item['category']}" for item in h["top_routes"][:3]); lines.append(f"| L{row['layer']}H{row['head']} | {h['joint_cosine']:.3f} | {h['literal_fidelity']:.3f} | {h['best_simple']} | {h['best_simple_fidelity']:.3f} | {h['top_route_jaccard']:.3f} | {leading} |")
    lines += ["", "## High-effect held-out edges", ""]
    for row in rows:
        lines.append(f"### L{row['layer']}H{row['head']}")
        for example in row["examples"][:6]: lines.append(f"- `{example['source_token']}` ({example['source_category']}, distance {example['distance']}) → `{example['destination_token']}`; coefficient `{example['attention']:.3f}`; `{example['context']}`")
        lines.append("")
    lines += ["## Interpretation boundary", "", "This is a weight-conditioned routing atlas, not a semantic classifier. Stable distance/category structure supports a reusable read verb; failure of human categories to beat controls means the verb may be easier to describe geometrically than linguistically. High-effect examples illustrate the frozen operator profile and are not independent validation of a semantic name."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    labels = [f"L{row['layer']}H{row['head']}" for row in rows]; x = np.arange(len(rows)); axes[0].bar(x-.15, [row["schemes"]["human"]["literal_fidelity"] for row in rows], .3, label="literal"); axes[0].bar(x+.15, [row["schemes"]["human"]["best_simple_fidelity"] for row in rows], .3, label="simple"); axes[0].set(xticks=x, xticklabels=labels, ylim=(0,1.05), ylabel="Held-out fidelity", title="Compact route descriptions"); axes[0].legend();
    names = ["human", "frequency", "shuffled_human"]; width=.22
    for i,name in enumerate(names): axes[1].bar(x+(i-1)*width, [row["schemes"][name]["best_simple_fidelity"] for row in rows], width, label=name)
    axes[1].set(xticks=x, xticklabels=labels, ylim=(0,1.05), ylabel="Best simple fidelity", title="Semantic-category controls"); axes[1].legend(fontsize=8)
    for axis in axes: axis.grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "heads": labels}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
