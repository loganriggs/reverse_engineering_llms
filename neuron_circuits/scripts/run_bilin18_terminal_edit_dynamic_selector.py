#!/usr/bin/env python3
"""Resolve static family heads versus per-example conditional head coalitions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import select_records
from run_bilin18_boundary_style_H2_value_echo import project
from run_bilin18_boundary_style_H2_weight_outlier import spectral_atlas
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_terminal_edit_discovery_generalization import (
    attention_input_covector,
    final_logits,
    full_layer17_details,
    output_covector,
)


CONFIG = Path("configs/bilin18_terminal_edit_dynamic_selector.json")
OUTPUT = TENSOR_ROOT / "bilin18_terminal_edit_dynamic_selector.json"
REPORT = TENSOR_ROOT / "BILIN18_TERMINAL_EDIT_DYNAMIC_SELECTOR.md"
FIGURE = TENSOR_ROOT / "bilin18_terminal_edit_dynamic_selector.png"


def row_correlation(first, second):
    first = first - first.mean(1, keepdim=True)
    second = second - second.mean(1, keepdim=True)
    numerator = (first * second).sum(1)
    denominator = first.square().sum(1).sqrt() * second.square().sum(1).sqrt()
    return numerator / denominator.clamp_min(1e-30)


@torch.inference_mode()
def collect_examples(model, raw, records, bases, config):
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    destination = 127
    projection = model.blocks[17].attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
    bases_gpu = [bases[(17, head)][0].to("cuda") for head in range(heads)]
    accumulated = {name: [] for name in ("write_norm", "direct", "finite_rank1", "finite_full", "endpoint_energy")}

    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch): swapped[index, row[3]] = row[5]
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        block, mixed, details = full_layer17_details(model, original, swapped)
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
            full_changes.append((((full_edge - original_edge) * mask).sum(1) @ projection[head]))
            rank1_changes.append((((rank1_edge - original_edge) * mask).sum(1) @ projection[head]))

        base_attention = details["write"][:size].float()
        original_logits, original_pre, original_final = final_logits(model, block, mixed[:size].float(), base_attention)
        swapped_logits, _, _ = final_logits(model, block, mixed[size:].float(), details["write"][size:].float())
        endpoint_delta = swapped_logits - original_logits
        final_covector, centered_endpoint, probabilities = output_covector(model, original_final, endpoint_delta, original_logits)
        pre_covector = attention_input_covector(block, original_pre, final_covector)
        direct = torch.stack([(pre_covector * change.double()).sum(-1) for change in rank1_changes], 1)
        write_norm = torch.stack([change.double().norm(dim=-1) for change in rank1_changes], 1)

        changes = full_changes + rank1_changes
        variants = []
        for change in changes:
            modified = base_attention.clone(); modified[:, destination] += change; variants.append(modified)
        variant_logits, _, _ = final_logits(
            model, block, mixed[:size].float().repeat(len(variants), 1, 1), torch.cat(variants)
        )
        variant_logits = variant_logits.reshape(len(variants), size, -1)
        finite = []
        for index in range(len(variants)):
            delta_logits = variant_logits[index].double() - original_logits.double()
            centered_delta = delta_logits - (probabilities * delta_logits).sum(-1, keepdim=True)
            finite.append((probabilities * centered_endpoint * centered_delta).sum(-1))
        finite = torch.stack(finite, 1)
        accumulated["write_norm"].append(write_norm.cpu())
        accumulated["direct"].append(direct.cpu())
        accumulated["finite_full"].append(finite[:, :heads].cpu())
        accumulated["finite_rank1"].append(finite[:, heads:].cpu())
        accumulated["endpoint_energy"].append((probabilities * centered_endpoint.square()).sum(-1).cpu())
    return {name: torch.cat(chunks).numpy() for name, chunks in accumulated.items()}


def summarize(arrays, static_head):
    finite_rank1 = arrays["finite_rank1"]
    finite_full = arrays["finite_full"]
    direct = arrays["direct"]
    write = arrays["write_norm"]
    oracle_rank1 = finite_rank1.argmax(1)
    oracle_full = finite_full.argmax(1)
    direct_choice = direct.argmax(1)
    context_choice = write.argmax(1)
    row = np.arange(len(finite_rank1))
    oracle_effect = np.maximum(finite_rank1[row, oracle_rank1], 0)
    denominator = max(float(oracle_effect.sum()), 1e-30)
    positive = np.maximum(finite_full, 0)
    positive_sum = positive.sum(1)
    support = positive_sum ** 2 / np.maximum((positive ** 2).sum(1), 1e-30)
    top_share = positive.max(1) / np.maximum(positive_sum, 1e-30)
    correlations = row_correlation(torch.from_numpy(finite_rank1), torch.from_numpy(finite_full)).numpy()
    return {
        "examples": len(row),
        "winner_counts_rank1": np.bincount(oracle_rank1, minlength=finite_rank1.shape[1]).tolist(),
        "winner_counts_full": np.bincount(oracle_full, minlength=finite_full.shape[1]).tolist(),
        "direct_winner_accuracy": float(np.mean(direct_choice == oracle_rank1)),
        "context_winner_accuracy": float(np.mean(context_choice == oracle_rank1)),
        "static_winner_accuracy": float(np.mean(oracle_rank1 == static_head)),
        "direct_oracle_effect_fraction": float(np.maximum(finite_rank1[row, direct_choice], 0).sum() / denominator),
        "context_oracle_effect_fraction": float(np.maximum(finite_rank1[row, context_choice], 0).sum() / denominator),
        "static_oracle_effect_fraction": float(np.maximum(finite_rank1[:, static_head], 0).sum() / denominator),
        "full_positive_effective_support_median": float(np.median(support)),
        "full_top_positive_share_median": float(np.median(top_share)),
        "rank1_full_winner_agreement": float(np.mean(oracle_rank1 == oracle_full)),
        "rank1_full_correlation_median": float(np.median(correlations)),
        "endpoint_energy_median": float(np.median(arrays["endpoint_energy"])),
    }


def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["parent_config"]).read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    families = {
        "spelling": token_pairs(tokenizer, base["spelling_pairs"]),
        "unrelated": token_pairs(tokenizer, parent["control_pairs"]),
    }
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    _, bases = spectral_atlas(model)

    raw_results = {}
    aggregate_heads = {}
    for family_index, (family, pairs) in enumerate(families.items()):
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(raw, boundary_flags, pairs, counts, 127, config["examples_per_label_per_split"], config["seed"] + 100 * family_index + split)
            print("dynamic selector", family, split, len(records), flush=True)
            arrays = collect_examples(model, raw, records, bases, config)
            raw_results[(family, split)] = arrays
            if split == 0: aggregate_heads[family] = int(arrays["finite_rank1"].mean(0).argmax())

    results = []
    for family in families:
        for split in range(2):
            results.append({"family": family, "split": split, "static_head": aggregate_heads[family], **summarize(raw_results[(family, split)], aggregate_heads[family])})
    d0 = all(row["direct_winner_accuracy"] >= .95 for row in results)
    d1 = all(row["full_positive_effective_support_median"] <= 3 and row["full_top_positive_share_median"] >= .5 for row in results)
    heldout = [row for row in results if row["split"] == 1]
    d2 = all(row["static_winner_accuracy"] >= .5 for row in heldout)
    d3 = all(row["context_winner_accuracy"] >= .5 and row["context_oracle_effect_fraction"] >= .7 for row in results)
    d4 = all(row["direct_oracle_effect_fraction"] >= .95 for row in results)
    d5 = all(row["rank1_full_correlation_median"] >= .7 and row["rank1_full_winner_agreement"] >= .5 for row in results)
    hypotheses = {
        "D0_analytic_selector_matches_finite_rank1": {"supported": d0},
        "D1_terminal_positive_support_is_sparse": {"supported": d1},
        "D2_one_family_head_explains_most_examples": {"supported": d2},
        "D3_context_only_dynamic_selection_is_useful": {"supported": d3},
        "D4_output_conditioned_dynamic_selection_is_near_oracle": {"supported": d4},
        "D5_weight_rank1_preserves_complete_head_order": {"supported": d5},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "aggregate_heads": aggregate_heads, "results": results}, indent=2))

    lines = ["# Dynamic terminal-head selector", "", "Natural edits are analyzed example by example. A static family head, the largest routed rank-1 write, and the output-conditioned tensor score are compared with the actual finite per-example head winner. Complete-head effects measure whether the terminal coalition itself is sparse; rank-1 versus complete-head agreement tests whether weight outliers preserve the native head ordering.", "", "## Locked results", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Family/split | static H | winner accuracy static/context/quotient | effect fraction static/context/quotient | full support/top share | rank1↔full r/winner |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['family']}/{row['split']} | H{row['static_head']} | {row['static_winner_accuracy']:.3f}/{row['context_winner_accuracy']:.3f}/{row['direct_winner_accuracy']:.3f} | {row['static_oracle_effect_fraction']:.3f}/{row['context_oracle_effect_fraction']:.3f}/{row['direct_oracle_effect_fraction']:.3f} | {row['full_positive_effective_support_median']:.2f}/{row['full_top_positive_share_median']:.3f} | {row['rank1_full_correlation_median']:.3f}/{row['rank1_full_winner_agreement']:.3f} |")
    lines += ["", "## Interpretation boundary", "", "A quotient-conditioned selector is an executable explanation for a declared output change, not an autonomous semantic feature detector. Context-only success would indicate that occupancy and routing identify the invoked operation before looking downstream. Sparse complete-head support would favor a small conditional coalition; diffuse support would mean that the failure of static heads reflects genuine multi-head computation rather than merely changing winners."]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = [f"{row['family'][:3]}s{row['split']}" for row in results]; x = np.arange(len(results))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for offset, (field, label) in enumerate((("static_winner_accuracy", "static"), ("context_winner_accuracy", "context"), ("direct_winner_accuracy", "quotient"))): axes[0].bar(x + (offset - 1) * .25, [row[field] for row in results], .24, label=label)
    axes[0].set(xticks=x, xticklabels=labels, ylim=(0, 1.05), ylabel="Winner accuracy", title="Per-example operation selection"); axes[0].legend()
    axes[1].bar(x - .18, [row["full_positive_effective_support_median"] for row in results], .36, label="effective support")
    axes[1].bar(x + .18, [row["full_top_positive_share_median"] for row in results], .36, label="top share")
    axes[1].set(xticks=x, xticklabels=labels, title="Complete-head causal concentration"); axes[1].legend()
    for axis in axes: axis.grid(alpha=.2)
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "aggregate_heads": aggregate_heads, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
