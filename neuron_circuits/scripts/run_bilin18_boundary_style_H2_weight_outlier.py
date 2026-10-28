#!/usr/bin/env python3
"""Test whether a weight-only c_v outlier identifies L17H2's edit compressor."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_H2_value_echo import evaluate
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_H2_weight_outlier.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_H2_weight_outlier.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_H2_WEIGHT_OUTLIER.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_H2_weight_outlier.png"


def spectral_atlas(model):
    heads = model.config.n_head
    head_dim = model.config.n_embd // heads
    rows = []
    bases = {}
    for layer, block in enumerate(model.blocks):
        weight = block.attn.c_v.weight.detach().float().cpu()
        for head in range(heads):
            matrix = weight[head * head_dim : (head + 1) * head_dim].double()
            left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
            score = float(singular[0] / singular.square().mean().sqrt())
            rows.append({"layer": layer, "head": head, "score": score, "top": float(singular[0]), "second": float(singular[1])})
            bases[(layer, head)] = left.float().T.contiguous()
    return rows, bases


def random_bases(dimension, maximum_rank, count, seed):
    generator = torch.Generator().manual_seed(seed)
    result = []
    for _ in range(count):
        matrix = torch.randn(dimension, maximum_rank, generator=generator)
        q, _ = torch.linalg.qr(matrix)
        result.append(q.T.contiguous())
    return result


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    lexical = json.loads(Path(config["lexical_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack(
        [vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]
    )
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[
        :, FEATURES.index("period")
    ].astype(bool)
    old_pairs = token_pairs(tokenizer, base["spelling_pairs"])
    new_pairs = token_pairs(tokenizer, lexical["new_spelling_pairs"])

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    atlas, bases = spectral_atlas(model)
    weight_basis = bases[(17, config["head"])]
    random = random_bases(
        weight_basis.shape[1], max(config["ranks"]), config["random_subspaces"], config["seed"]
    )
    ranked = sorted(atlas, key=lambda row: row["score"], reverse=True)
    h2 = next(row for row in atlas if row["layer"] == 17 and row["head"] == config["head"])
    h2_rank = next(index + 1 for index, row in enumerate(ranked) if row is h2)
    gaussian_edge = 1 + np.sqrt((model.config.n_embd // model.config.n_head) / model.config.n_embd)

    results = []
    for fold in range(config["folds"]):
        target_indices = set(range(fold * config["target_pairs_per_fold"], (fold + 1) * config["target_pairs_per_fold"]))
        targets = [pair for index, pair in enumerate(old_pairs) if index in target_indices]
        cues = [pair for index, pair in enumerate(old_pairs) if index not in target_indices]
        contrasts = make_contrasts(model.lm_head.weight.shape[0], targets, 0, config["seed"] + fold, "cuda")
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + fold * 10 + split)
            print("weight outlier original", fold, split, len(records), flush=True)
            results.append({"case": "original", "fold": fold, "split": split, **evaluate(model, raw, records, contrasts, weight_basis, random, config)})

    new_results = []
    contrasts = make_contrasts(model.lm_head.weight.shape[0], old_pairs, 0, config["seed"] + 100, "cuda")
    for split, (begin, end) in enumerate(base["splits"]):
        raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
        records = select_records(raw, boundary_flags, new_pairs, counts, 127, config["examples_per_label_per_split"], config["seed"] + 100 + split)
        print("weight outlier new", split, len(records), flush=True)
        new_results.append({"case": "new_cues", "split": split, **evaluate(model, raw, records, contrasts, weight_basis, random, config)})

    heldout = [row for row in results if not (row["fold"] == 0 and row["split"] == 0)]
    all_results = results + new_results
    w0 = h2["score"] >= gaussian_edge + 0.10
    w1 = h2_rank <= 10
    w2 = all(row["value_fidelity"]["1"] >= 0.50 for row in heldout)
    w3 = all(row["causal_value_fidelity"]["1"] >= 0.95 for row in heldout)
    w4 = all(row["value_fidelity"]["1"] >= 0.50 and row["causal_value_fidelity"]["1"] >= 0.95 for row in new_results)
    w5 = all(max(row["random_causal_value_fidelity"]["1"]) < 0.25 for row in all_results)
    hypotheses = {
        "W0_H2_has_a_spectral_outlier": {"supported": w0, "score": h2["score"], "gaussian_edge": float(gaussian_edge)},
        "W1_H2_is_prospectively_ranked": {"supported": w1, "rank": h2_rank},
        "W2_weight_rank1_reconstructs_value_changes": {"supported": w2},
        "W3_weight_rank1_is_causally_sufficient": {"supported": w3},
        "W4_weight_rank1_generalizes_new_lexemes": {"supported": w4},
        "W5_random_rank1_fails": {"supported": w5},
        "W6_weight_only_outlier_identifies_the_read_function": {"supported": all((w0, w1, w2, w3, w4, w5))},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "h2": h2, "h2_rank": h2_rank, "gaussian_edge": float(gaussian_edge), "spectral_atlas": ranked, "results": all_results}, indent=2))

    lines = [
        "# Weight-only outlier test for the L17H2 edit compressor",
        "",
        "Every c_v head slice in all 18 layers is scored by top singular value divided by RMS singular value. The L17H2 left singular basis is then used without activation fitting to reconstruct post-boundary value changes and finite spelling effects, including new lexemes.",
        "",
        "## Locked results",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", f"L17H2 score: `{h2['score']:.3f}`; Gaussian rectangular edge: `{gaussian_edge:.3f}`; global head rank: `{h2_rank}/162`.", "", "| Case | Fold/split | raw weight-R1 | causal weight-R1 | best random causal R1 |", "|---|---|---:|---:|---:|"]
    for row in all_results:
        cell = f"{row.get('fold', '-')}/{row['split']}"
        lines.append(f"| {row['case']} | {cell} | {row['value_fidelity']['1']:.3f} | {row['causal_value_fidelity']['1']:.3f} | {max(row['random_causal_value_fidelity']['1']):.3f} |")
    lines += ["", "## Interpretation boundary", "", "Passing the behavioral gates would show that the circuit-relevant value coordinate is directly readable from weights, without activation-derived PCA. Spectral rank determines whether the same outlier would identify this head in a blind whole-model search; behavioral success alone does not make the outlier globally unique or semantic."]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    scores = [row["score"] for row in ranked]
    axes[0].plot(np.arange(1, len(scores) + 1), scores)
    axes[0].scatter([h2_rank], [h2["score"]], color="red", label="L17H2")
    axes[0].axhline(gaussian_edge, color="black", linestyle="--", label="Gaussian edge")
    axes[0].set(xlabel="Global head rank", ylabel="Top / RMS singular", title="All c_v head slices")
    axes[0].legend()
    labels = [f"{row['case'][:3]}-{row.get('fold','n')}-{row['split']}" for row in all_results]
    axes[1].bar(np.arange(len(all_results)), [row["causal_value_fidelity"]["1"] for row in all_results])
    axes[1].set(xticks=np.arange(len(labels)), xticklabels=labels, ylabel="Finite causal fidelity", title="Weight-only rank-1 component")
    axes[1].tick_params(axis="x", rotation=45)
    for axis in axes: axis.grid(alpha=0.2)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "h2": h2, "h2_rank": h2_rank, "results": all_results}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
