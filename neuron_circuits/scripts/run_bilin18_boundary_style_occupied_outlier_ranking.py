#!/usr/bin/env python3
"""Test which ingredients prospectively select a causally relevant weight outlier."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_H2_value_echo import layer17_details, project
from run_bilin18_boundary_style_H2_weight_outlier import spectral_atlas
from run_bilin18_boundary_style_final_attention_atlas import score_and_gradient
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_occupied_outlier_ranking.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_occupied_outlier_ranking.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_OCCUPIED_OUTLIER_RANKING.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_occupied_outlier_ranking.png"


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


@torch.inference_mode()
def collect(model, raw, records, contrasts, bases, spectral_scores, config):
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    destination = 127
    declared = contrasts[:, 0]
    token_index = torch.nonzero(declared).flatten()
    token_weights = declared[token_index].double()
    output_weights = model.lm_head.weight[token_index].double()
    bases_gpu = [bases[(17, head)][0].to("cuda") for head in range(heads)]
    projection = model.blocks[17].attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)

    sums = {
        head: {
            "value_total": 0.0,
            "value_rank1": 0.0,
            "full_write_sq": 0.0,
            "rank1_write_sq": 0.0,
            "direct": 0.0,
            "finite_full": 0.0,
            "finite_rank1": 0.0,
        }
        for head in range(heads)
    }
    count = 0
    for start in range(0, len(records), config["batch_size"]):
        batch = records[start : start + config["batch_size"]]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch):
            swapped[index, row[3]] = row[5]
        cue = torch.as_tensor([row[3] for row in batch], device="cuda")
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        direction = torch.as_tensor([1 if not row[4] else -1 for row in batch], device="cuda", dtype=torch.float32)
        block, mixed, details = layer17_details(model, original, swapped, cue, boundary)
        pattern = (details["score1"] * details["score2"])[:, :, destination].float()
        values = details["mixed_value"].float().permute(0, 2, 1, 3)
        position = torch.arange(original.shape[1], device="cuda")[None]
        mask = ((position > boundary[:, None]) & (position < destination))[:, :, None].float()

        full_changes, rank1_changes = [], []
        for head in range(heads):
            p0, p1 = pattern[:size, head], pattern[size:, head]
            v0, v1 = values[:size, head], values[size:, head]
            delta = v1 - v0
            rank1_delta = project(delta, bases_gpu[head][None])
            original_edge = p0[:, :, None] * v0
            full_edge = p1[:, :, None] * v1
            rank1_edge = p1[:, :, None] * (v0 + rank1_delta)
            full_change = (((full_edge - original_edge) * mask).sum(1) @ projection[head])
            rank1_change = (((rank1_edge - original_edge) * mask).sum(1) @ projection[head])
            full_changes.append(full_change)
            rank1_changes.append(rank1_change)
            selected = mask.bool().expand_as(delta)
            sums[head]["value_total"] += float(delta[selected].double().square().sum())
            sums[head]["value_rank1"] += float(rank1_delta[selected].double().square().sum())
            sums[head]["full_write_sq"] += float(full_change.double().square().sum())
            sums[head]["rank1_write_sq"] += float(rank1_change.double().square().sum())

        base_attention = details["write"][:size].float()
        variants = [base_attention]
        for changes in (full_changes, rank1_changes):
            for change in changes:
                modified = base_attention.clone()
                modified[:, destination] += change
                variants.append(modified)
        repeated = len(variants)
        attention = torch.cat(variants)
        mixed_original = mixed[:size].float().repeat(repeated, 1, 1)
        normalized = F.rms_norm(mixed_original + attention, (dimension,))
        final = mixed_original + attention + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        destination_states = F.rms_norm(final, (dimension,))[:, destination]
        logits = 30 * torch.tanh(model.lm_head(destination_states).float() / 30)
        scores = (logits @ contrasts[:, :1]).reshape(repeated, size)
        for head in range(heads):
            sums[head]["finite_full"] += float((direction * (scores[1 + head] - scores[0])).sum())
            sums[head]["finite_rank1"] += float((direction * (scores[1 + heads + head] - scores[0])).sum())

        # Output-conditioned local score: exact output gradient, but no final-MLP Jacobian.
        base_state = final[:size, destination].double()
        _, gradient = score_and_gradient(base_state, output_weights, token_weights)
        for head in range(heads):
            sums[head]["direct"] += float((direction.double() * (gradient * rank1_changes[head].double()).sum(-1)).sum())
        count += size

    rows = []
    for head in range(heads):
        row = sums[head]
        rows.append({
            "head": head,
            "spectral_score": spectral_scores[head],
            "value_rank1_fraction": row["value_rank1"] / max(row["value_total"], 1e-30),
            "full_write_rms": np.sqrt(row["full_write_sq"] / count),
            "rank1_write_rms": np.sqrt(row["rank1_write_sq"] / count),
            "direct_quotient_mean": row["direct"] / count,
            "finite_full_mean": row["finite_full"] / count,
            "finite_rank1_mean": row["finite_rank1"] / count,
        })
    direct_correlation = pearson([row["direct_quotient_mean"] for row in rows], [row["finite_rank1_mean"] for row in rows])
    rows_without_h2 = [row for row in rows if row["head"] != 2]
    leave_h2_out_correlation = pearson(
        [row["direct_quotient_mean"] for row in rows_without_h2],
        [row["finite_rank1_mean"] for row in rows_without_h2],
    )
    return {
        "examples": count,
        "heads": rows,
        "direct_finite_correlation": direct_correlation,
        "leave_H2_out_correlation": leave_h2_out_correlation,
    }


def rank_of(rows, field, head, reverse=True):
    ordered = sorted(rows, key=lambda row: row[field], reverse=reverse)
    return next(index + 1 for index, row in enumerate(ordered) if row["head"] == head)


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    pairs = token_pairs(tokenizer, base["spelling_pairs"])

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    atlas, bases = spectral_atlas(model)
    spectral_scores = {row["head"]: row["score"] for row in atlas if row["layer"] == 17}

    results = []
    for fold in range(config["folds"]):
        target_indices = set(range(fold * config["target_pairs_per_fold"], (fold + 1) * config["target_pairs_per_fold"]))
        targets = [pair for index, pair in enumerate(pairs) if index in target_indices]
        cues = [pair for index, pair in enumerate(pairs) if index not in target_indices]
        contrasts = make_contrasts(model.lm_head.weight.shape[0], targets, 0, config["seed"] + fold, "cuda")
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(raw, boundary_flags, cues, counts, 127, config["examples_per_label_per_split"], config["seed"] + fold * 10 + split)
            print("occupied outlier", fold, split, len(records), flush=True)
            results.append({"fold": fold, "split": split, **collect(model, raw, records, contrasts, bases, spectral_scores, config)})

    discovery = results[0]
    context_head = max(discovery["heads"], key=lambda row: row["rank1_write_rms"])["head"]
    quotient_head = max(discovery["heads"], key=lambda row: row["direct_quotient_mean"])["head"]
    heldout = results[1:]
    context_ranks = [rank_of(row["heads"], "finite_rank1_mean", context_head) for row in heldout]
    quotient_ranks = [rank_of(row["heads"], "finite_rank1_mean", quotient_head) for row in heldout]
    h2_ratios = []
    for result in results:
        h2 = next(row for row in result["heads"] if row["head"] == 2)
        h2_ratios.append(h2["finite_rank1_mean"] / max(abs(h2["finite_full_mean"]), 1e-30))

    d0 = True
    d1 = context_head == 2 and all(rank <= 3 for rank in context_ranks)
    d2 = quotient_head == 2 and all(rank == 1 for rank in quotient_ranks)
    d3 = all(
        result["direct_finite_correlation"] >= 0.8
        and result["leave_H2_out_correlation"] >= 0.8
        for result in results
    )
    d4 = all(ratio >= 0.9 for ratio in h2_ratios)
    hypotheses = {
        "D0_selection_is_discovery_frozen": {"supported": d0},
        "D1_contextual_occupied_outlier_selects_H2": {"supported": d1, "selected_head": context_head, "heldout_ranks": context_ranks},
        "D2_output_quotient_selects_H2": {"supported": d2, "selected_head": quotient_head, "heldout_ranks": quotient_ranks},
        "D3_quotient_score_predicts_finite_effects": {"supported": d3},
        "D4_rank1_preserves_H2_finite_effect": {"supported": d4, "ratios": h2_ratios},
        "D5_context_is_enough_without_semantic_output": {"supported": d1},
        "D6_occupied_outlier_discovery_rule": {"supported": all((d0, d1, d2, d3, d4))},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = ["# Occupied-outlier ranking for Layer-17 spelling consumers", "", "Four candidate rankings are compared across all Layer-17 heads: spectral strength, value-axis occupancy, routed rank-1 write norm, and direct output-conditioned tensor score. Rankings are selected on fold 0/split 0; full finite rank-1 head effects on the other five cells are untouched evaluation.", "", "## Locked results", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Fold/split | H2 spectral/occupancy/routed/quotient/finite ranks | direct→finite r | H2 rank1/full |", "|---|---:|---:|---:|"]
    for result, ratio in zip(results, h2_ratios):
        ranks = [rank_of(result["heads"], field, 2) for field in ("spectral_score", "value_rank1_fraction", "rank1_write_rms", "direct_quotient_mean", "finite_rank1_mean")]
        lines.append(f"| {result['fold']}/{result['split']} | {'/'.join(map(str, ranks))} | {result['direct_finite_correlation']:.3f} ({result['leave_H2_out_correlation']:.3f} without H2) | {ratio:.3f} |")
    lines += ["", "## Interpretation boundary", "", "Routed write norm uses weights, the counterfactual state change, and the source relation but no output-token semantics. The quotient score additionally uses the declared output contrast while omitting the final-MLP Jacobian. Their comparison states whether contextual tensor occupancy is enough to shortlist the function or whether semantic destination information is essential."]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = [f"f{row['fold']}s{row['split']}" for row in results]
    x = np.arange(len(results)); figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    fields = (("spectral_score", "spectral"), ("value_rank1_fraction", "occupancy"), ("rank1_write_rms", "routed"), ("direct_quotient_mean", "quotient"), ("finite_rank1_mean", "finite"))
    for field, label in fields: axes[0].plot(x, [rank_of(row["heads"], field, 2) for row in results], marker="o", label=label)
    axes[0].invert_yaxis(); axes[0].set(xticks=x, xticklabels=labels, ylabel="H2 rank (1 is best)", title="Which ingredients select H2?"); axes[0].legend(fontsize=8)
    axes[1].plot(x, [row["direct_finite_correlation"] for row in results], marker="o"); axes[1].set(xticks=x, xticklabels=labels, ylim=(-1, 1.05), ylabel="Pearson r", title="Quotient score versus finite head effects")
    for axis in axes: axis.grid(alpha=0.2)
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
