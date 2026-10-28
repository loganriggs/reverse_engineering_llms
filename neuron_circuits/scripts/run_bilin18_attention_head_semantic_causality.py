#!/usr/bin/env python3
"""Prospectively test frozen semantic head roles by exact finite ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_attention_head_semantic_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_attention_head_semantic_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_ATTENTION_HEAD_SEMANTIC_CAUSALITY.md"


def selected_sites(raw, positions, token_flags, sites_per_class, seed):
    labels = token_flags[raw[:, np.asarray(positions) + 1]].reshape(-1).astype(bool)
    rows = np.repeat(np.arange(len(raw)), len(positions))
    position_index = np.tile(np.arange(len(positions)), len(raw))
    generator = np.random.default_rng(seed)
    chosen = []
    for value in (False, True):
        candidates = np.flatnonzero(labels == value)
        generator.shuffle(candidates)
        chosen.extend(candidates[:sites_per_class])
    chosen = np.asarray(chosen, dtype=np.int64)
    return rows[chosen], position_index[chosen], labels[chosen]


@torch.inference_mode()
def reference_logits(model, ids, position):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    for block in model.blocks:
        x, bus = block(x, bus, x0)
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
    return 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def variant_logits(model, ids, position, target_layer, removal_scale=1.0):
    """Return native, each single-head-removed, and all-attention-removed logits."""
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    for layer, block in enumerate(model.blocks):
        if layer < target_layer:
            x, bus = block(x, bus, x0)
            continue
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        attended = torch.einsum("bhqk,bkhd->bhqd", details["pattern"], details["mixed_value"])
        at_position = attended[:, :, position].permute(0, 1, 2).contiguous()
        projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
        head_writes = torch.einsum("bhd,hdo->bho", at_position.float(), projection)
        variants = heads + 2
        attention = details["write"].float().unsqueeze(0).repeat(variants, 1, 1, 1)
        for head in range(heads):
            attention[1 + head, :, position] -= removal_scale * head_writes[:, head]
        attention[-1, :, position] -= removal_scale * details["write"][:, position].float()
        x0 = x0.float().unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1)
        mixed = mixed.float().unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1)
        bus = details["value_bus"].float().unsqueeze(0).repeat(variants, 1, 1, 1, 1).flatten(0, 1)
        pre = mixed + attention.flatten(0, 1)
        state = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        for suffix in model.blocks[layer + 1:]:
            x, bus = suffix(x, bus, x0)
        raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
        logits = 30 * torch.tanh(raw / 30)
        return logits.reshape(variants, len(ids), -1)
    raise AssertionError("target layer not reached")


@torch.inference_mode()
def evaluate_candidate(model, raw, positions, flags, candidate, config, seed):
    feature = FEATURES.index(candidate["feature"])
    row_ids, position_ids, labels = selected_sites(
        raw, positions, flags[:, feature], config["sites_per_class"], seed,
    )
    positive = flags[:, feature].sum(); negative = len(flags) - positive
    contrast = torch.zeros(model.config.vocab_size, device="cuda", dtype=torch.float32)
    contrast[:len(flags)] = torch.as_tensor(
        flags[:, feature] / max(positive, 1) - (1 - flags[:, feature]) / max(negative, 1),
        device="cuda", dtype=torch.float32,
    )
    contributions = [[] for _ in range(model.config.n_head + 1)]
    loss_deltas = [[] for _ in range(model.config.n_head + 1)]
    gathered_labels = []
    maximum_error = 0.0
    for position_index, position in enumerate(positions):
        selected = np.flatnonzero(position_ids == position_index)
        for start in range(0, len(selected), config["batch_size"]):
            indices = selected[start:start + config["batch_size"]]
            if not len(indices): continue
            ids = torch.as_tensor(raw[row_ids[indices], :config["sequence_length"]], device="cuda")
            variants = variant_logits(model, ids, position, candidate["layer"], config.get("removal_scale", 1.0))
            reference = reference_logits(model, ids, position)
            maximum_error = max(maximum_error, float((variants[0] - reference).abs().max()))
            native_score = variants[0] @ contrast
            targets = torch.as_tensor(raw[row_ids[indices], position + 1], device="cuda")
            native_loss = F.cross_entropy(variants[0], targets, reduction="none")
            for variant in range(1, model.config.n_head + 2):
                contributions[variant - 1].extend((native_score - variants[variant] @ contrast).cpu().tolist())
                loss_deltas[variant - 1].extend((F.cross_entropy(variants[variant], targets, reduction="none") - native_loss).cpu().tolist())
            gathered_labels.extend(labels[indices].tolist())
    return {
        "labels": np.asarray(gathered_labels, dtype=bool),
        "contributions": np.asarray(contributions, dtype=np.float64),
        "loss_deltas": np.asarray(loss_deltas, dtype=np.float64),
        "maximum_native_error": maximum_error,
        "positive_sites": int(labels.sum()), "negative_sites": int((~labels).sum()),
    }


def effect_size(values, labels, sign):
    signed = sign * values
    pooled = np.sqrt(.5 * (signed[labels].var() + signed[~labels].var()) + 1e-30)
    return float((signed[labels].mean() - signed[~labels].mean()) / pooled)


def summarize(calibration, evaluation, selected_head):
    signs = []
    for row in calibration["contributions"]:
        difference = row[calibration["labels"]].mean() - row[~calibration["labels"]].mean()
        signs.append(1 if difference >= 0 else -1)
    rows = []
    for index, row in enumerate(evaluation["contributions"]):
        score = signs[index] * row
        rows.append({
            "head": index if index < len(rows) or index < 9 else "all",
            "sign": signs[index], "causal_auc": float(auc(evaluation["labels"], score)),
            "effect_size": effect_size(row, evaluation["labels"], signs[index]),
            "mean_absolute_class_effect": float(np.mean(np.abs(row))),
            "mean_loss_delta": float(evaluation["loss_deltas"][index].mean()),
        })
    head_rows = rows[:-1]; selected = head_rows[selected_head]
    selected["rank"] = 1 + sum(row["causal_auc"] > selected["causal_auc"] for row in head_rows)
    selected["control_median_auc"] = float(np.median([row["causal_auc"] for row in head_rows if row is not selected]))
    return {"selected": selected, "all_attention": rows[-1], "heads": head_rows}


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    source = np.load(config["corpus"], mmap_mode="r")
    calibration_raw = np.asarray(source[config["calibration_start"]:config["calibration_start"] + config["calibration_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    evaluation_raw = np.asarray(source[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    records = []; maximum_error = 0.0
    for index, candidate in enumerate(config["candidates"]):
        print("causal head role", index + 1, "/", len(config["candidates"]), candidate, flush=True)
        calibration = evaluate_candidate(model, calibration_raw, config["positions"], flags, candidate, config, config["seed"] + index)
        evaluation = evaluate_candidate(model, evaluation_raw, config["positions"], flags, candidate, config, config["seed"] + 100 + index)
        maximum_error = max(maximum_error, calibration["maximum_native_error"], evaluation["maximum_native_error"])
        records.append({**candidate, "calibration_sites": {"positive": calibration["positive_sites"], "negative": calibration["negative_sites"]}, "evaluation_sites": {"positive": evaluation["positive_sites"], "negative": evaluation["negative_sites"]}, **summarize(calibration, evaluation, candidate["head"])})
    hypotheses = {
        "C0_manual_native_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
        "C1_probe_roles_have_causal_discrimination": {"supported": sum(row["selected"]["causal_auc"] >= .60 for row in records) >= 4},
        "C2_selected_heads_beat_other_heads": {"supported": sum(row["selected"]["causal_auc"] >= row["selected"]["control_median_auc"] + .05 for row in records) >= 4},
        "C3_selected_heads_are_top_three_causal": {"supported": sum(row["selected"]["rank"] <= 3 for row in records) >= 4},
        "C4_attention_role_is_not_only_decodable": {"supported": sum(row["selected"]["effect_size"] >= .30 for row in records) >= 4},
        "C5_full_attention_contains_the_role": {"supported": sum(row["all_attention"]["causal_auc"] >= .60 for row in records) >= 4},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "records": records}, indent=2, allow_nan=False))
    lines = ["# Prospective finite-causal test of semantic attention-head roles", "", "Six head/feature candidates were frozen after the residual-conditioned probe audit, then evaluated on a third untouched FineWeb split. Each intervention removes one complete native head write at the queried position, recomputes that layer's MLP and the exact remaining transformer, and measures the feature-class logit contrast. Causal sign is fixed on a separate calibration split.", "", f"Maximum native-branch versus checkpoint logit error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Candidate | Signed causal AUROC | Effect size | Rank / 9 | Control median | All-attention AUROC | Mean |class effect| |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        selected = row["selected"]
        lines.append(f"| L{row['layer']} H{row['head']} `{row['feature']}` | {selected['causal_auc']:.3f} | {selected['effect_size']:.3f} | {selected['rank']} | {selected['control_median_auc']:.3f} | {row['all_attention']['causal_auc']:.3f} | {selected['mean_absolute_class_effect']:.5f} |")
    lines += ["", "## Guardrails", "", "The candidates were exploratory on an earlier panel but are prospective on this evaluation split. Signed AUROC tests whether a head's finite causal contribution distinguishes positive from negative contexts; it does not require that the head always promotes the positive class. Removing a head is a large intervention and may expose compensation or nonlinear interactions. Control heads share the same layer and examples but are not matched for write norm. The class contrast averages over the declared vocabulary feature, so rare or heterogeneous labels can still obscure a narrower causal role."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "records": [{"layer":r["layer"],"head":r["head"],"feature":r["feature"],"selected":r["selected"],"all_attention":r["all_attention"]} for r in records]}, indent=2))
    print(REPORT)


if __name__ == "__main__": main()
