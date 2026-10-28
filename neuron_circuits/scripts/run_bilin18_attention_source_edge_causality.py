#!/usr/bin/env python3
"""Test source-token head roles against equal-count, distance-matched edge controls."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_attention_head_semantic_causality import reference_logits
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_attention_source_edge_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_attention_source_edge_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_ATTENTION_SOURCE_EDGE_CAUSALITY.md"


def distance_bin(distance, boundaries):
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if lower <= distance < upper: return index
    raise ValueError(distance)


def masks_for_site(tokens, position, token_flags, boundaries, controls, generator):
    feature = token_flags[tokens[:position + 1]].astype(bool)
    selected = np.flatnonzero(feature)
    if not len(selected): return None
    bins = np.asarray([distance_bin(position - source, boundaries) for source in range(position + 1)])
    informative = any(np.sum(bins == bins[source]) > 1 for source in selected)
    if not informative: return None
    random = np.zeros((controls, len(tokens)), dtype=bool)
    for control in range(controls):
        for bin_index in np.unique(bins[selected]):
            count = int(np.sum(bins[selected] == bin_index))
            available = np.flatnonzero((bins == bin_index) & ~feature)
            if len(available) < count: available = np.flatnonzero(bins == bin_index)
            chosen = generator.choice(available, count, replace=len(available) < count)
            random[control, chosen] = True
    selected_mask = np.zeros(len(tokens), dtype=bool); selected_mask[selected] = True
    return selected_mask, random


def select_sites(raw, positions, token_flags, config, seed):
    generator = np.random.default_rng(seed)
    candidates = []
    for row in range(len(raw)):
        for position in positions:
            masks = masks_for_site(raw[row, :config["sequence_length"]], position, token_flags, config["distance_bins"], config["random_controls"], generator)
            if masks is not None: candidates.append((row, position, *masks))
    generator.shuffle(candidates)
    return candidates[:config["sites"]]


def kl(reference, prediction):
    logp = F.log_softmax(reference.float(), -1); logq = F.log_softmax(prediction.float(), -1)
    # Floating reduction can produce tiny negative values around identical logits.
    return (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)


def value_norm_matched_masks(value_norm, selected_mask, position, boundaries, controls):
    """Match selected source count, distance bin, and aggregate value norm."""
    batch, length = selected_mask.shape
    output = torch.zeros(batch, controls, length, dtype=torch.bool, device=selected_mask.device)
    bins = torch.as_tensor([distance_bin(position - source, boundaries) if source <= position else -1 for source in range(length)], device=selected_mask.device)
    for batch_index in range(batch):
        for bin_index in torch.unique(bins[selected_mask[batch_index]]).tolist():
            selected = torch.where(selected_mask[batch_index] & (bins == bin_index))[0]
            count = len(selected)
            available = torch.where((~selected_mask[batch_index]) & (bins == bin_index))[0]
            if len(available) < count: available = torch.where(bins == bin_index)[0]
            target = value_norm[batch_index, selected].mean()
            order = available[(value_norm[batch_index, available] - target).abs().argsort()]
            for control in range(controls):
                start = control % len(order)
                chosen = torch.roll(order, -start)[:count]
                output[batch_index, control, chosen] = True
    return output


@torch.inference_mode()
def edge_variants(model, ids, position, layer, head, selected_mask, random_masks, probes, config):
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    for block_index, block in enumerate(model.blocks):
        if block_index < layer:
            x, bus = block(x, bus, x0); continue
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        source = details["pattern"][:, head, position, :, None].float() * details["mixed_value"][:, :, head].float()
        if config.get("value_norm_match"):
            value_norm = details["mixed_value"][:, :, head].float().norm(dim=-1)
            random_masks = value_norm_matched_masks(
                value_norm, selected_mask, position, config["distance_bins"], config["random_controls"],
            )
        projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)[head]
        selected_write = torch.einsum("bkd,bk,do->bo", source, selected_mask.float(), projection)
        random_write = torch.einsum("bkd,brk,do->bro", source, random_masks.float(), projection)
        full_write = source.sum(1) @ projection
        controls = random_masks.shape[1]; variants = controls + 3
        attention = details["write"].float().unsqueeze(0).repeat(variants, 1, 1, 1)
        attention[1, :, position] -= selected_write
        for control in range(controls): attention[2 + control, :, position] -= random_write[:, control]
        attention[-1, :, position] -= full_write
        x0 = x0.float().unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1)
        mixed = mixed.float().unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1)
        bus = details["value_bus"].float().unsqueeze(0).repeat(variants, 1, 1, 1, 1).flatten(0, 1)
        pre = mixed + attention.flatten(0, 1)
        state = F.rms_norm(pre, (dimension,))
        local = (state[:, position] @ probes.T).reshape(variants, len(ids), -1)
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        for suffix in model.blocks[block_index + 1:]: x, bus = suffix(x, bus, x0)
        raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
        logits = (30 * torch.tanh(raw / 30)).reshape(variants, len(ids), -1)
        return logits, local
    raise AssertionError("layer not reached")


@torch.inference_mode()
def evaluate(model, raw, sites, candidate, config):
    block = model.blocks[candidate["layer"]]
    probes, _, _ = operator_probe_vectors(block, config["operator_probes"], config["seed"] + candidate["layer"] * 17 + candidate["head"])
    selected_local, random_local, full_local = [], [], []
    selected_kl, random_kl, full_kl = [], [], []
    maximum_error = 0.0
    for position in config["positions"]:
        group = [site for site in sites if site[1] == position]
        for start in range(0, len(group), config["batch_size"]):
            batch = group[start:start + config["batch_size"]]
            if not batch: continue
            ids = torch.as_tensor(np.stack([raw[site[0], :config["sequence_length"]] for site in batch]), device="cuda")
            selected_mask = torch.as_tensor(np.stack([site[2] for site in batch]), device="cuda")
            random_masks = torch.as_tensor(np.stack([site[3] for site in batch]), device="cuda")
            logits, local = edge_variants(model, ids, position, candidate["layer"], candidate["head"], selected_mask, random_masks, probes, config)
            reference = reference_logits(model, ids, position)
            maximum_error = max(maximum_error, float((logits[0] - reference).abs().max()))
            selected_local.extend((local[0] - local[1]).square().sum(-1).cpu().tolist())
            controls = config["random_controls"]
            random_local.extend((local[0, :, None] - local[2:2 + controls].permute(1, 0, 2)).square().sum(-1).cpu().tolist())
            full_local.extend((local[0] - local[-1]).square().sum(-1).cpu().tolist())
            selected_kl.extend(kl(logits[0], logits[1]).cpu().tolist())
            random_kl.extend(torch.stack([kl(logits[0], logits[2 + control]) for control in range(controls)], 1).cpu().tolist())
            full_kl.extend(kl(logits[0], logits[-1]).cpu().tolist())
    selected_local = np.asarray(selected_local); random_local = np.asarray(random_local); full_local = np.asarray(full_local)
    selected_kl = np.asarray(selected_kl); random_kl = np.asarray(random_kl); full_kl = np.asarray(full_kl)
    epsilon = 1e-30
    paired_local = random_local.mean(1); paired_kl = random_kl.mean(1)
    local_scale = max(float(np.median(paired_local[paired_local > 0])) if np.any(paired_local > 0) else 1e-30, 1e-30)
    kl_scale = max(float(np.median(paired_kl[paired_kl > 0])) if np.any(paired_kl > 0) else 1e-30, 1e-30)
    local_keep = np.maximum(selected_local / local_scale, paired_local / local_scale) <= np.quantile(np.maximum(selected_local / local_scale, paired_local / local_scale), .95)
    kl_keep = np.maximum(selected_kl / kl_scale, paired_kl / kl_scale) <= np.quantile(np.maximum(selected_kl / kl_scale, paired_kl / kl_scale), .95)
    generator = np.random.default_rng(config["seed"] + candidate["layer"] * 31 + candidate["head"])
    local_bootstrap, kl_bootstrap = [], []
    for _ in range(1000):
        sample = generator.integers(0, len(selected_local), len(selected_local))
        local_bootstrap.append(selected_local[sample].sum() / (paired_local[sample].sum() + epsilon))
        kl_bootstrap.append(selected_kl[sample].sum() / (paired_kl[sample].sum() + epsilon))
    return {
        "sites": len(selected_local), "maximum_native_error": maximum_error,
        "local_effect_ratio": float(selected_local.sum() / (random_local.mean(1).sum() + epsilon)),
        "local_site_win_rate": float(np.mean(selected_local > random_local.mean(1))),
        "downstream_kl_ratio": float(selected_kl.sum() / (random_kl.mean(1).sum() + epsilon)),
        "downstream_site_win_rate": float(np.mean(selected_kl > random_kl.mean(1))),
        "local_fraction_of_full_head": float(selected_local.sum() / (full_local.sum() + epsilon)),
        "downstream_fraction_of_full_head": float(selected_kl.sum() / (full_kl.sum() + epsilon)),
        "local_trimmed_ratio": float(selected_local[local_keep].sum() / (paired_local[local_keep].sum() + epsilon)),
        "downstream_trimmed_ratio": float(selected_kl[kl_keep].sum() / (paired_kl[kl_keep].sum() + epsilon)),
        "local_bootstrap_interval": [float(value) for value in np.quantile(local_bootstrap, [.025, .975])],
        "downstream_bootstrap_interval": [float(value) for value in np.quantile(kl_bootstrap, [.025, .975])],
        "median_local_log2_ratio": float(np.median(np.log2((selected_local + local_scale * 1e-4) / (paired_local + local_scale * 1e-4)))),
        "median_downstream_log2_ratio": float(np.median(np.log2((selected_kl + kl_scale * 1e-4) / (paired_kl + kl_scale * 1e-4)))),
        "mean_selected_local_effect": float(selected_local.mean()),
        "mean_random_local_effect": float(random_local.mean()),
        "mean_selected_kl": float(selected_kl.mean()),
        "mean_random_kl": float(random_kl.mean()),
    }


def finalize(config, records, maximum_error):
    both = [row["local_effect_ratio"] >= 1.5 and row["downstream_kl_ratio"] >= 1.5 for row in records]
    hypotheses = {
        "E0_edge_intervention_reproduces_native": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
        "E1_source_features_have_privileged_local_effect": {"supported": sum(row["local_effect_ratio"] >= 1.5 for row in records) >= 5},
        "E2_source_features_win_sitewise": {"supported": sum(row["local_site_win_rate"] >= .60 for row in records) >= 5},
        "E3_source_features_have_privileged_downstream_effect": {"supported": sum(row["downstream_kl_ratio"] >= 1.5 for row in records) >= 4},
        "E4_source_features_explain_material_head_effect": {"supported": sum(row["local_fraction_of_full_head"] >= .25 for row in records) >= 4},
        "E5_semantic_routing_survives_distance_control": {"supported": sum(both) >= 4, "passing_candidates": int(sum(both))},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "records": records}, indent=2, allow_nan=False))
    lines = ["# Distance-matched causal test of attention source roles", "", "Eight stable head/source-feature enrichments were frozen from earlier FineWeb panels and tested on a later untouched split. At each eligible destination, all edges from matching source tokens are removed from one head. Eight controls remove equal numbers of sources from the same relative-distance bins in the same sequence. Both the immediate bilinear-operator sketch and the exact final-logit KL are recomputed.", "", f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Candidate source role | Sites | Local effect ratio / win | Downstream KL ratio / win | Local/downstream fraction of full head |", "|---|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| L{row['layer']} H{row['head']} reads `{row['source_feature']}` | {row['sites']} | {row['local_effect_ratio']:.2f} / {row['local_site_win_rate']:.2f} | {row['downstream_kl_ratio']:.2f} / {row['downstream_site_win_rate']:.2f} | {row['local_fraction_of_full_head']:.2f}/{row['downstream_fraction_of_full_head']:.2f} |")
    lines += ["", "## Guardrails", "", "Controls match source count and coarse relative-distance bin, not exact token frequency, overlapping lexical categories, or value direction. Squared local sketch change and KL are effect magnitudes, so they establish privileged routing but not whether the source feature is promoted or suppressed. Candidates were chosen from stable enrichment on earlier panels; this split is prospective for causal effect. A failure can mean the enrichment is descriptive routing with small behavioral consequence, not that the measured QK preference was absent."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "records": records}, indent=2))
    print(REPORT)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    partial_index = os.environ.get("BILIN_SOURCE_EDGE_CANDIDATE")
    if os.environ.get("BILIN_SOURCE_EDGE_AGGREGATE"):
        partials = [json.loads((TENSOR_ROOT / f"bilin18_attention_source_edge_causality_partial_{index}.json").read_text()) for index in range(len(config["candidates"]))]
        records = [row["record"] for row in partials]
        finalize(config, records, max(row["maximum_error"] for row in partials))
        return
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    source = np.load(config["corpus"], mmap_mode="r")
    raw = np.asarray(source[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    records = []; maximum_error = 0.0
    indexed_candidates = list(enumerate(config["candidates"]))
    if partial_index is not None:
        indexed_candidates = [indexed_candidates[int(partial_index)]]
    for index, candidate in indexed_candidates:
        feature = FEATURES.index(candidate["source_feature"])
        sites = select_sites(raw, config["positions"], flags[:, feature], config, config["seed"] + index)
        print("source edge causal", index + 1, "/", len(config["candidates"]), candidate, "sites", len(sites), flush=True)
        result = evaluate(model, raw, sites, candidate, config)
        maximum_error = max(maximum_error, result["maximum_native_error"])
        records.append({**candidate, **result})
    if partial_index is not None:
        destination = TENSOR_ROOT / f"bilin18_attention_source_edge_causality_partial_{partial_index}.json"
        destination.write_text(json.dumps({"maximum_error": maximum_error, "record": records[0]}, indent=2, allow_nan=False))
        print(destination)
        return
    finalize(config, records, maximum_error)


if __name__ == "__main__": main()
