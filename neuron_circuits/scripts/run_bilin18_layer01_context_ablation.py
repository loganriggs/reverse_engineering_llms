#!/usr/bin/env python3
"""Red-team the supervised census with an isolated-current-token control."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER, exact_attention_forward, load_front
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_layer01_supervised_feature_census import (
    FEATURES, coalition_metrics, unigram_scores, vocabulary_features,
)
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_context_ablation.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer01_context_ablation.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER01_CONTEXT_ABLATION.md"
FIGURE = TENSOR_ROOT / "bilin18_layer01_context_ablation.png"


@torch.inference_mode()
def prefix_products(embedding, blocks, ids, positions, config, tt):
    dimension = config.n_embd
    x0 = F.rms_norm(embedding(ids), (dimension,))
    x, value_bus = x0, None
    products = {}
    for layer, block in enumerate(blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = exact_attention_forward(block.attn, F.rms_norm(mixed, (dimension,)), value_bus, config, tt)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        products[layer] = product[:, positions].flatten(0, 1).float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    return products


def collect_context(embedding, blocks, rows, positions, flags, batch_size, model_config, tt):
    total = len(rows) * len(positions)
    width = blocks[0].mlp.Left.out_features
    arrays = {layer: np.empty((total, width), dtype=np.float16) for layer in range(len(blocks))}
    labels = np.empty((total, len(FEATURES)), dtype=np.uint8)
    current_ids = np.empty(total, dtype=np.int32)
    positions_np = np.asarray(positions)
    offset = 0
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        products = prefix_products(embedding, blocks, ids, positions, model_config, tt)
        count = len(raw) * len(positions)
        for layer, values in products.items(): arrays[layer][offset:offset + count] = values.cpu().to(torch.float16).numpy()
        labels[offset:offset + count] = flags[raw[:, positions_np + 1].reshape(-1)]
        current_ids[offset:offset + count] = raw[:, positions_np].reshape(-1)
        offset += count
    return arrays, labels, current_ids


@torch.inference_mode()
def isolated_product_table(embedding, blocks, token_ids, batch_size, model_config, tt):
    width = blocks[0].mlp.Left.out_features
    tables = {layer: np.empty((len(token_ids), width), dtype=np.float16) for layer in range(len(blocks))}
    for start in range(0, len(token_ids), batch_size):
        selected = token_ids[start:start + batch_size]
        ids = torch.as_tensor(selected, device="cuda")[:, None]
        products = prefix_products(embedding, blocks, ids, [0], model_config, tt)
        for layer, values in products.items(): tables[layer][start:start + len(selected)] = values.cpu().to(torch.float16).numpy()
    return tables


def main():
    red = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(red["source_config"]).read_text())
    source_results = json.loads(Path(red["source_results"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    corpus = np.load(source_config["corpus"], mmap_mode="r")
    start = source_config["start_row"]
    discovery_rows = corpus[start:start + source_config["discovery_rows"]]
    heldout_start = start + source_config["discovery_rows"]
    heldout_rows = corpus[heldout_start:heldout_start + source_config["heldout_rows"]]
    embedding, blocks, model_config, tt = load_front(1, "cuda")
    discovery, discovery_labels, discovery_ids = collect_context(embedding, blocks, discovery_rows, source_config["positions"], flags, source_config["batch_size"], model_config, tt)
    heldout, heldout_labels, heldout_ids = collect_context(embedding, blocks, heldout_rows, source_config["positions"], flags, source_config["batch_size"], model_config, tt)
    unique_ids, inverse = np.unique(np.concatenate((discovery_ids, heldout_ids)), return_inverse=True)
    isolated_tables = isolated_product_table(embedding, blocks, unique_ids, 256, model_config, tt)
    split = len(discovery_ids)
    discovery_inverse, heldout_inverse = inverse[:split], inverse[split:]
    unigram = unigram_scores(discovery_ids, discovery_labels, heldout_ids, tokenizer.vocab_size)

    contextual, isolated = {}, {}
    for layer in source_config["layers"]:
        contextual[str(layer)] = coalition_metrics(discovery[layer], heldout[layer], discovery_labels, heldout_labels, source_config["coalition_sizes"])
        isolated[str(layer)] = coalition_metrics(isolated_tables[layer][discovery_inverse], isolated_tables[layer][heldout_inverse], discovery_labels, heldout_labels, source_config["coalition_sizes"])
    competent_names = {row["feature"] for row in source_results["features"] if row["competent"]}
    rows = []
    for index, name in enumerate(FEATURES):
        if name not in competent_names: continue
        row = {"feature": name, "unigram_auc": float(auc(heldout_labels[:, index], unigram[:, index])), "layers": {}}
        for layer in source_config["layers"]:
            key = str(layer); c = contextual[key][index]; i = isolated[key][index]
            row["layers"][key] = {"contextual": c["aucs"], "isolated": i["aucs"], "gain64": c["aucs"]["64"] - i["aucs"]["64"]}
        rows.append(row)
    med = lambda values: float(np.median(values))
    summary = {"features": len(rows), "layers": {}}
    for layer in source_config["layers"]:
        key = str(layer)
        summary["layers"][key] = {
            "contextual64_median": med([row["layers"][key]["contextual"]["64"] for row in rows]),
            "isolated64_median": med([row["layers"][key]["isolated"]["64"] for row in rows]),
            "gain64_median": med([row["layers"][key]["gain64"] for row in rows]),
            "fraction_gain_ge_005": float(np.mean([row["layers"][key]["gain64"] >= .05 for row in rows])),
        }
    best_layer = max(source_config["layers"], key=lambda layer: summary["layers"][str(layer)]["gain64_median"])
    median_unigram = med([row["unigram_auc"] for row in rows])
    hypotheses = {
        "C0_context_beats_isolated_token_code": {"supported": summary["layers"][str(best_layer)]["gain64_median"] >= .03},
        "C1_context_matters_for_many_features": {"supported": summary["layers"][str(best_layer)]["fraction_gain_ge_005"] >= 1/3},
        "C2_layer1_adds_more_context_than_layer0": {"supported": summary["layers"]["1"]["gain64_median"] - summary["layers"]["0"]["gain64_median"] >= .02},
        "C3_contextual_result_is_not_unigram_memorization": {"supported": max(summary["layers"][str(layer)]["contextual64_median"] for layer in source_config["layers"]) - median_unigram >= .03},
        "C4_same_examples_and_labels": {"supported": True},
    }
    summary.update({"best_context_gain_layer": best_layer, "median_unigram_auc": median_unigram, "unique_current_tokens": len(unique_ids)})
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summary": summary, "features": rows}, indent=2, allow_nan=False))
    lines = ["# Context ablation for the supervised layer-0/1 census", "", "The supervised census is rerun on the identical FineWeb sites. The red-team baseline sees only the current token: every distinct token is forwarded as a one-token sequence, and a separate fixed native-channel coalition is fitted to predict the same next-token labels. This is stronger than a finite-data unigram lookup because morphologically related tokens can share the model's isolated-token code.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Context ablation](bilin18_layer01_context_ablation.png)", "", f"Competent labels: **{len(rows)}**; unique current tokens: **{len(unique_ids):,}**; median unigram AUROC: **{median_unigram:.3f}**.", "", "| Layer | Contextual 64 | Isolated-token 64 | Median context gain | Fraction gain ≥ .05 |", "|---:|---:|---:|---:|---:|"]
    for layer in source_config["layers"]:
        row = summary["layers"][str(layer)]; lines.append(f"| {layer} | {row['contextual64_median']:.3f} | {row['isolated64_median']:.3f} | {row['gain64_median']:.3f} | {row['fraction_gain_ge_005']:.1%} |")
    lines += ["", "| Feature | Unigram | L0 context / isolated / gain | L1 context / isolated / gain |", "|---|---:|---:|---:|"]
    for row in sorted(rows, key=lambda item: item["layers"][str(best_layer)]["gain64"], reverse=True):
        l0, l1 = row["layers"]["0"], row["layers"]["1"]
        lines.append(f"| {row['feature']} | {row['unigram_auc']:.3f} | {l0['contextual']['64']:.3f}/{l0['isolated']['64']:.3f}/{l0['gain64']:+.3f} | {l1['contextual']['64']:.3f}/{l1['isolated']['64']:.3f}/{l1['gain64']:+.3f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for layer in source_config["layers"]:
        key = str(layer); axes[0].scatter([row["layers"][key]["isolated"]["64"] for row in rows], [row["layers"][key]["contextual"]["64"] for row in rows], alpha=.75, label=f"layer {layer}")
    axes[0].plot([.5, 1], [.5, 1], color="black", linestyle="--"); axes[0].set(xlabel="Isolated-token AUROC", ylabel="Contextual AUROC", title="Same labels, with and without prefix context"); axes[0].legend(); axes[0].grid(alpha=.2)
    ordered = sorted(rows, key=lambda row: row["layers"][str(best_layer)]["gain64"], reverse=True)
    axes[1].barh(range(len(ordered)), [row["layers"][str(best_layer)]["gain64"] for row in ordered]); axes[1].set_yticks(range(len(ordered)), [row["feature"] for row in ordered], fontsize=7); axes[1].invert_yaxis(); axes[1].axvline(.05, color="black", linestyle="--"); axes[1].set(xlabel="Contextual − isolated AUROC", title=f"Layer {best_layer} context advantage"); axes[1].grid(axis="x", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
