#!/usr/bin/env python3
"""Prospectively test occupied-outlier discovery on natural, unrelated edits.

The earlier spelling experiment begins with a known L8H2 circuit edit and a
human-declared spelling-logit contrast.  Here the original and swapped prefixes
are run naturally through the whole network.  Importance is defined by the
Fisher inner product with the model's own changed next-token distribution, so
the test does not assume a human semantic output axis.
"""

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
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_terminal_edit_discovery_generalization.json")
OUTPUT = TENSOR_ROOT / "bilin18_terminal_edit_discovery_generalization.json"
REPORT = TENSOR_ROOT / "BILIN18_TERMINAL_EDIT_DISCOVERY_GENERALIZATION.md"
FIGURE = TENSOR_ROOT / "bilin18_terminal_edit_discovery_generalization.png"


def rms_vjp(state: torch.Tensor, cotangent: torch.Tensor) -> torch.Tensor:
    """Vector-Jacobian product through PyTorch's default float RMSNorm."""
    eps = torch.finfo(state.dtype).eps
    radius = (state.square().mean(-1, keepdim=True) + eps).sqrt()
    return cotangent / radius - state * (cotangent * state).mean(-1, keepdim=True) / radius.pow(3)


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def rank_of(rows, field, head):
    ordered = sorted(rows, key=lambda row: row[field], reverse=True)
    return next(index + 1 for index, row in enumerate(ordered) if row["head"] == head)


@torch.inference_mode()
def full_layer17_details(model, original, swapped):
    dimension = model.config.n_embd
    ids = torch.as_tensor(np.concatenate((original, swapped)), device="cuda")
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, bus = x0, None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(
            block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt
        )
        if layer == 17:
            return block, mixed, details
        pre = mixed + details["write"]
        normalized = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        bus = details["value_bus"]
    raise AssertionError("Layer 17 was not reached")


def final_logits(model, block, mixed, attention, destination=127):
    dimension = mixed.shape[-1]
    pre = mixed + attention
    normalized = F.rms_norm(pre, (dimension,))
    final = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    final_state = final[:, destination]
    raw = model.lm_head(F.rms_norm(final_state, (dimension,))).float()
    return 30 * torch.tanh(raw / 30), pre[:, destination], final_state


def output_covector(model, final_state, endpoint_delta, original_logits):
    """Gradient of the endpoint Fisher inner product through final RMSNorm."""
    probabilities = original_logits.softmax(-1).double()
    centered = endpoint_delta.double() - (probabilities * endpoint_delta.double()).sum(-1, keepdim=True)
    token_weights = probabilities * centered
    state = final_state.double()
    normalized = F.rms_norm(state, (state.shape[-1],))
    weights = model.lm_head.weight.double()
    raw = normalized @ weights.T
    derivative = token_weights / torch.cosh(raw / 30).square()
    return rms_vjp(state, derivative @ weights), centered, probabilities


def attention_input_covector(block, pre, final_covector):
    """Exact local VJP through residual + bilinear MLP at Layer 17."""
    state = pre.double()
    normalized = F.rms_norm(state, (state.shape[-1],))
    left_weight = block.mlp.Left.weight.double()
    right_weight = block.mlp.Right.weight.double()
    down_weight = block.mlp.Down.weight.double()
    left = normalized @ left_weight.T
    right = normalized @ right_weight.T
    channel = final_covector @ down_weight
    normalized_covector = (channel * right) @ left_weight + (channel * left) @ right_weight
    return final_covector + rms_vjp(state, normalized_covector)


@torch.inference_mode()
def collect(model, raw, records, bases, spectral_scores, random_axes, config):
    dimension = model.config.n_embd
    heads = model.config.n_head
    head_dim = dimension // heads
    destination = 127
    projection = model.blocks[17].attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
    bases_gpu = [bases[(17, head)][0].to("cuda") for head in range(heads)]
    random_gpu = [axis.to("cuda") for axis in random_axes]
    sums = {
        head: dict(value_total=0.0, value_rank1=0.0, rank1_write_sq=0.0,
                   direct=0.0, finite_full=0.0, finite_rank1=0.0)
        for head in range(heads)
    }
    random_finite = np.zeros(len(random_axes), dtype=np.float64)
    h2_prefix_finite = 0.0
    endpoint_energy = 0.0
    endpoint_kl = 0.0
    label_counts = {0: 0, 1: 0}
    count = 0

    for start in range(0, len(records), config["batch_size"]):
        batch = records[start:start + config["batch_size"]]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch):
            swapped[index, row[3]] = row[5]
            label_counts[int(row[4])] += 1
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        block, mixed, details = full_layer17_details(model, original, swapped)
        pattern = (details["score1"] * details["score2"])[:, :, destination].float()
        values = details["mixed_value"].float().permute(0, 2, 1, 3)
        position = torch.arange(original.shape[1], device="cuda")[None]
        suffix_mask = ((position > boundary[:, None]) & (position < destination))[:, :, None].float()
        prefix_mask = (position <= boundary[:, None])[:, :, None].float()

        full_changes, rank1_changes = [], []
        for head in range(heads):
            p0, p1 = pattern[:size, head], pattern[size:, head]
            v0, v1 = values[:size, head], values[size:, head]
            delta = v1 - v0
            rank1_delta = project(delta, bases_gpu[head][None])
            original_edge = p0[:, :, None] * v0
            full_edge = p1[:, :, None] * v1
            rank1_edge = p1[:, :, None] * (v0 + rank1_delta)
            full_change = (((full_edge - original_edge) * suffix_mask).sum(1) @ projection[head])
            rank1_change = (((rank1_edge - original_edge) * suffix_mask).sum(1) @ projection[head])
            full_changes.append(full_change)
            rank1_changes.append(rank1_change)
            selected = suffix_mask.bool().expand_as(delta)
            sums[head]["value_total"] += float(delta[selected].double().square().sum())
            sums[head]["value_rank1"] += float(rank1_delta[selected].double().square().sum())
            sums[head]["rank1_write_sq"] += float(rank1_change.double().square().sum())

        # Relation control for the already implicated head.
        head = 2
        p0, p1 = pattern[:size, head], pattern[size:, head]
        v0, v1 = values[:size, head], values[size:, head]
        h2_prefix_change = ((((p1[:, :, None] * v1) - (p0[:, :, None] * v0)) * prefix_mask).sum(1) @ projection[head])

        # Frozen random value directions for H2, retaining its native routing.
        h2_delta = v1 - v0
        h2_original_edge = p0[:, :, None] * v0
        random_changes = []
        for axis in random_gpu:
            random_delta = project(h2_delta, axis[None])
            changed = p1[:, :, None] * (v0 + random_delta)
            random_changes.append(((changed - h2_original_edge) * suffix_mask).sum(1) @ projection[head])

        base_attention = details["write"][:size].float()
        swapped_attention = details["write"][size:].float()
        original_logits, original_pre, original_final = final_logits(model, block, mixed[:size].float(), base_attention)
        swapped_logits, _, _ = final_logits(model, block, mixed[size:].float(), swapped_attention)
        endpoint_delta = swapped_logits - original_logits
        final_covector, centered_endpoint, probabilities = output_covector(
            model, original_final, endpoint_delta, original_logits
        )
        pre_covector = attention_input_covector(block, original_pre, final_covector)
        energy = (probabilities * centered_endpoint.square()).sum(-1)
        endpoint_energy += float(energy.sum())
        endpoint_kl += float((probabilities * (original_logits.double() - swapped_logits.double())).sum(-1).sum())
        for head in range(heads):
            sums[head]["direct"] += float((pre_covector * rank1_changes[head].double()).sum())

        changes = full_changes + rank1_changes + [h2_prefix_change] + random_changes
        variants = []
        for change in changes:
            modified = base_attention.clone()
            modified[:, destination] += change
            variants.append(modified)
        repeated = len(variants)
        variant_logits, _, _ = final_logits(
            model,
            block,
            mixed[:size].float().repeat(repeated, 1, 1),
            torch.cat(variants),
        )
        variant_logits = variant_logits.reshape(repeated, size, -1)
        finite = []
        for index in range(repeated):
            delta_logits = variant_logits[index].double() - original_logits.double()
            centered_delta = delta_logits - (probabilities * delta_logits).sum(-1, keepdim=True)
            finite.append((probabilities * centered_endpoint * centered_delta).sum(-1))
        for head in range(heads):
            sums[head]["finite_full"] += float(finite[head].sum())
            sums[head]["finite_rank1"] += float(finite[heads + head].sum())
        h2_prefix_finite += float(finite[2 * heads].sum())
        for index in range(len(random_axes)):
            random_finite[index] += float(finite[2 * heads + 1 + index].sum())
        count += size

    rows = []
    for head in range(heads):
        row = sums[head]
        rows.append({
            "head": head,
            "spectral_score": spectral_scores[head],
            "value_rank1_fraction": row["value_rank1"] / max(row["value_total"], 1e-30),
            "rank1_write_rms": np.sqrt(row["rank1_write_sq"] / count),
            "direct_quotient_mean": row["direct"] / count,
            "finite_full_mean": row["finite_full"] / count,
            "finite_rank1_mean": row["finite_rank1"] / count,
        })
    strongest = max(rows, key=lambda row: row["finite_rank1_mean"])["head"]
    without = [row for row in rows if row["head"] != strongest]
    return {
        "examples": count,
        "label_counts": label_counts,
        "endpoint_fisher_energy_mean": endpoint_energy / count,
        "endpoint_logit_kl_proxy_mean": endpoint_kl / count,
        "heads": rows,
        "direct_finite_correlation": pearson(
            [row["direct_quotient_mean"] for row in rows],
            [row["finite_rank1_mean"] for row in rows],
        ),
        "leave_winner_out_correlation": pearson(
            [row["direct_quotient_mean"] for row in without],
            [row["finite_rank1_mean"] for row in without],
        ),
        "strongest_finite_head": strongest,
        "h2_prefix_finite_mean": h2_prefix_finite / count,
        "h2_random_rank1_finite_means": (random_finite / count).tolist(),
    }


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    families = {
        "spelling": token_pairs(tokenizer, base["spelling_pairs"]),
        "unrelated": token_pairs(tokenizer, config["control_pairs"]),
    }

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    atlas, bases = spectral_atlas(model)
    spectral_scores = {row["head"]: row["score"] for row in atlas if row["layer"] == 17}
    generator = torch.Generator().manual_seed(config["seed"])
    random_axes = []
    for _ in range(config["random_rank1_controls"]):
        axis = torch.randn(model.config.n_embd // model.config.n_head, generator=generator)
        random_axes.append(axis / axis.norm())

    results = []
    for family_index, (family, pairs) in enumerate(families.items()):
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(
                raw, boundary_flags, pairs, counts, 127,
                config["examples_per_label_per_split"],
                config["seed"] + 100 * family_index + split,
            )
            print("terminal edit", family, split, len(records), flush=True)
            results.append({
                "family": family,
                "split": split,
                **collect(model, raw, records, bases, spectral_scores, random_axes, config),
            })

    lookup = {(row["family"], row["split"]): row for row in results}
    selections = {}
    for family in families:
        discovery = lookup[(family, 0)]
        evaluation = lookup[(family, 1)]
        context_head = max(discovery["heads"], key=lambda row: row["rank1_write_rms"])["head"]
        quotient_head = max(discovery["heads"], key=lambda row: row["direct_quotient_mean"])["head"]
        selections[family] = {
            "context_head": context_head,
            "context_heldout_finite_rank": rank_of(evaluation["heads"], "finite_rank1_mean", context_head),
            "quotient_head": quotient_head,
            "quotient_heldout_finite_rank": rank_of(evaluation["heads"], "finite_rank1_mean", quotient_head),
            "h2_heldout_finite_rank": rank_of(evaluation["heads"], "finite_rank1_mean", 2),
        }

    heldout = [lookup[(family, 1)] for family in families]
    t0 = all(row["label_counts"][0] == row["label_counts"][1] and row["examples"] >= 40 for row in results)
    t1 = all(row["direct_finite_correlation"] >= .8 and row["leave_winner_out_correlation"] >= .7 for row in results)
    t2 = all(row["context_heldout_finite_rank"] <= 3 for row in selections.values())
    t3 = all(row["quotient_heldout_finite_rank"] == 1 for row in selections.values())
    t4 = all(row["context_head"] == 2 and row["h2_heldout_finite_rank"] <= 3 for row in selections.values())
    t5 = True
    t6 = True
    for row in heldout:
        h2 = next(head for head in row["heads"] if head["head"] == 2)
        t5 &= h2["finite_rank1_mean"] > max(row["h2_random_rank1_finite_means"])
        t6 &= h2["finite_full_mean"] > 0 and h2["finite_full_mean"] >= 2 * max(row["h2_prefix_finite_mean"], 0)
    hypotheses = {
        "T0_panels_are_balanced": {"supported": t0},
        "T1_tensor_quotient_predicts_finite_head_effects": {"supported": t1},
        "T2_context_only_selection_transfers_within_family": {"supported": t2},
        "T3_output_conditioned_selection_transfers_within_family": {"supported": t3},
        "T4_H2_is_a_universal_terminal_edit_compressor": {"supported": t4},
        "T5_H2_weight_mode_beats_random_rank1_controls": {"supported": t5},
        "T6_H2_postboundary_relation_is_privileged": {"supported": t6},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "selections": selections, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Prospective terminal edit discovery beyond spelling",
        "",
        "Natural one-token swaps are propagated through all 18 layers. Layer-17 post-boundary head edits are ranked either by routed weight-mode occupancy alone or by their analytic alignment with the model's own changed next-token distribution. The latter uses the original-output Fisher metric and is validated with finite interventions through the final bilinear MLP and normalization.",
        "",
        "## Locked results",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Family/split | N | endpoint Fisher energy | winner | H2 ranks spectral/occupancy/routed/quotient/finite | direct→finite r (without winner) |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        h2r = [rank_of(row["heads"], field, 2) for field in ("spectral_score", "value_rank1_fraction", "rank1_write_rms", "direct_quotient_mean", "finite_rank1_mean")]
        lines.append(
            f"| {row['family']}/{row['split']} | {row['examples']} | {row['endpoint_fisher_energy_mean']:.4g} | H{row['strongest_finite_head']} | {'/'.join(map(str, h2r))} | {row['direct_finite_correlation']:.3f} ({row['leave_winner_out_correlation']:.3f}) |"
        )
    lines += ["", "## Frozen selections", "", "| Family | context selection/rank | quotient selection/rank | H2 held-out rank |", "|---|---:|---:|---:|"]
    for family, row in selections.items():
        lines.append(f"| {family} | H{row['context_head']} / {row['context_heldout_finite_rank']} | H{row['quotient_head']} / {row['quotient_heldout_finite_rank']} | {row['h2_heldout_finite_rank']} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "The endpoint is the model's own output change, not a human-labeled concept. Success therefore establishes behavioral mechanism discovery, not semantic interpretation. Routed write magnitude can identify an occupied transport operation without saying whether it helps the relevant output change; the Fisher quotient supplies that missing destination type. Failure of universal H2 reuse would not contradict the earlier generic-axis result: many edits can share one H2 value direction without H2 being their dominant terminal consumer.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    labels = [f"{row['family'][:3]}s{row['split']}" for row in results]
    x = np.arange(len(results))
    for field, label in (("rank1_write_rms", "routed"), ("direct_quotient_mean", "quotient"), ("finite_rank1_mean", "finite")):
        axes[0].plot(x, [rank_of(row["heads"], field, 2) for row in results], marker="o", label=label)
    axes[0].invert_yaxis(); axes[0].set(xticks=x, xticklabels=labels, ylabel="H2 rank", title="Does H2 generalize beyond spelling?"); axes[0].legend()
    axes[1].plot(x, [row["direct_finite_correlation"] for row in results], marker="o", label="all heads")
    axes[1].plot(x, [row["leave_winner_out_correlation"] for row in results], marker="o", label="without winner")
    axes[1].set(xticks=x, xticklabels=labels, ylim=(-1, 1.05), ylabel="Pearson r", title="Tensor prediction versus finite effects"); axes[1].legend()
    for axis in axes: axis.grid(alpha=.2)
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "selections": selections}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
