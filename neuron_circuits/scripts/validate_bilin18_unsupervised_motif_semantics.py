#!/usr/bin/env python3
"""Held-out routing and finite causal tests for frozen unsupervised motifs."""

from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_qkov_causal_edges import attention_details, forward_prefix, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_unsupervised_motif_semantic_validation.json")
OUTPUT = TENSOR_ROOT / "bilin18_unsupervised_motif_semantic_validation.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSUPERVISED_MOTIF_SEMANTIC_VALIDATION.md"


def push(heap, item, limit):
    packed = (item[0], item[1], item[2], item[3], item[4], item[5])
    if len(heap) < limit: heapq.heappush(heap, packed)
    elif item[0] > heap[0][0]: heapq.heapreplace(heap, packed)


def masks(kind, ids, positions, boundary_ids):
    source = ids[:, None, :]
    destination = ids[:, positions, None]
    length = ids.shape[1]
    source_position = torch.arange(length, device=ids.device)[None, None]
    destination_position = torch.as_tensor(positions, device=ids.device)[None, :, None]
    causal_prior = source_position < destination_position
    if kind == "prior_exact_token_match":
        return (source == destination) & causal_prior
    if kind == "distance_one":
        return (source_position == destination_position - 1).expand(ids.shape[0], -1, -1)
    if kind == "boundary_source":
        result = torch.zeros_like(source, dtype=torch.bool)
        for token in boundary_ids: result |= source == token
        return result & causal_prior
    raise ValueError(kind)


def safe_enrichment(energy, total, count, available):
    share = energy / max(total, 1e-30)
    prevalence = count / max(available, 1)
    return share / max(prevalence, 1e-30), share, prevalence


@torch.inference_mode()
def collect(model, panel, config, boundary_ids):
    positions = config["positions"]
    families = config["families"]
    needed_layers = sorted({layer for family in families.values() for layer, _ in family["heads"]})
    masks_by_layer = {
        layer: sorted({family["mask"] for family in families.values()
                       if any(target_layer == layer for target_layer, _ in family["heads"])})
        for layer in needed_layers
    }
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    totals = {(layer, head): [0., 0, 0.] for layer in needed_layers for head in range(heads)}
    semantic = {(kind, layer, head): [0., 0] for layer in needed_layers
                for kind in masks_by_layer[layer] for head in range(heads)}
    heaps = {(name, layer, head): [] for name, family in families.items() for layer, head in family["heads"]}
    for start in range(0, len(panel), config["batch_size"]):
        raw = np.asarray(panel[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device=model.embedding.weight.device)
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for layer, block in enumerate(model.blocks[:max(needed_layers) + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            if layer in needed_layers:
                projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
                source_norm = torch.einsum("bshk,hkd->bshd", details["mixed_value"].float(), projection).norm(dim=-1)
                pattern = details["pattern"][:, :, positions].abs().float().permute(0, 2, 1, 3)
                magnitude = pattern * source_norm.permute(0, 2, 1)[:, None]
                causal = torch.arange(config["sequence_length"], device=ids.device)[None, :] <= torch.as_tensor(positions, device=ids.device)[:, None]
                for head in range(heads):
                    valid = magnitude[:, :, head] * causal[None]
                    totals[(layer, head)][0] += float(valid.sum())
                    totals[(layer, head)][1] += int((valid > 0).sum())
                    totals[(layer, head)][2] += int(causal.sum()) * len(ids)
                cached_masks = {}
                for kind in masks_by_layer[layer]:
                    semantic_mask = masks(kind, ids, positions, boundary_ids)
                    cached_masks[kind] = semantic_mask
                    for head in range(heads):
                        values = magnitude[:, :, head]
                        semantic[(kind, layer, head)][0] += float(values[semantic_mask].sum())
                        semantic[(kind, layer, head)][1] += int(semantic_mask.sum())
                for name, family in families.items():
                    for target_layer, head in family["heads"]:
                        if target_layer != layer: continue
                        semantic_mask = cached_masks[family["mask"]]
                        values = magnitude[:, :, head]
                        flat_values = values.masked_fill(~semantic_mask, -torch.inf).reshape(-1)
                        amount = min(config["causal_sites_per_head"], int(torch.isfinite(flat_values).sum()))
                        if amount:
                            top_values, top_indices = torch.topk(flat_values, amount)
                            for value, index in zip(top_values.cpu().tolist(), top_indices.cpu().tolist()):
                                batch_index, remainder = divmod(index, len(positions) * config["sequence_length"])
                                position_index, source_position = divmod(remainder, config["sequence_length"])
                                destination = positions[position_index]
                                row_values = values[batch_index, position_index]
                                nonsemantic = (~semantic_mask[batch_index, position_index]) & causal[position_index]
                                candidate = torch.nonzero(nonsemantic, as_tuple=False).flatten()
                                if not len(candidate): continue
                                control = int(candidate[(row_values[candidate] - value).abs().argmin()])
                                push(heaps[(name, layer, head)], (
                                    float(value), start + batch_index, int(source_position), int(destination),
                                    control, float(row_values[control]),
                                ), config["causal_sites_per_head"])
            pre = mixed + details["write"]
            normalized = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
        if start % (config["batch_size"] * 32) == 0:
            print(f"held-out routing {start}/{len(panel)}", flush=True)
    return totals, semantic, heaps


def kl(reference, changed):
    log_p = F.log_softmax(reference, -1); p = log_p.exp()
    return float((p * (log_p - F.log_softmax(changed, -1))).sum())


@torch.inference_mode()
def causal(model, panel, heaps, tokenizer, global_start):
    results = {}
    for key, heap in heaps.items():
        name, layer, head = key
        rows = []
        for magnitude, row, source, destination, control_source, control_magnitude in sorted(heap, reverse=True):
            raw = np.asarray(panel[row, :destination + 2], dtype=np.int64)
            ids = torch.as_tensor(raw[:destination + 1], device=model.embedding.weight.device)[None]
            target = int(raw[destination + 1])
            base, _ = forward_prefix(model, ids, layer, 0)
            changed, _ = forward_prefix(model, ids, layer, 0, edge=("edge", head, destination, source))
            control, _ = forward_prefix(model, ids, layer, 0, edge=("edge", head, destination, control_source))
            base_logp = F.log_softmax(base, -1)
            changed_logp = F.log_softmax(changed, -1)
            control_logp = F.log_softmax(control, -1)
            lo = max(0, source - 6); hi = min(len(raw), destination + 2)
            rows.append({
                "row": global_start + row, "layer": layer, "head": head,
                "source": source, "destination": destination,
                "source_token": tokenizer.decode([int(raw[source])]).replace("\n", "\\n"),
                "destination_token": tokenizer.decode([int(raw[destination])]).replace("\n", "\\n"),
                "target_token": tokenizer.decode([target]).replace("\n", "\\n"),
                "context": tokenizer.decode(raw[lo:hi].tolist()).replace("\n", "\\n"),
                "magnitude": magnitude, "control_source": control_source,
                "control_magnitude": control_magnitude,
                "semantic_kl": kl(base, changed), "control_kl": kl(base, control),
                "semantic_target_logprob_change": float(changed_logp[target] - base_logp[target]),
                "control_target_logprob_change": float(control_logp[target] - base_logp[target]),
            })
        results[f"{name}/L{layer}H{head}"] = rows
    return results


def median(values):
    return float(np.median(values)) if values else 0.


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    boundary_strings = ["\n", ".", "!", "?", ";", ":"]
    boundary_ids = {tokenizer.eos_token_id}
    for text in boundary_strings:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1: boundary_ids.add(ids[0])
    source = np.load(config["corpus"], mmap_mode="r")
    panel = source[config["start"]:config["start"] + config["rows"]]
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.float().eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    totals, semantic, heaps = collect(model, panel, config, boundary_ids)
    causal_rows = causal(model, panel, heaps, tokenizer, config["start"])
    families = {}
    control_enrichments = {}
    for name, family in config["families"].items():
        per_head = []
        layers = sorted(set(layer for layer, _ in family["heads"]))
        # Semantic-mask counts do not depend on head within a layer.
        for layer, head in family["heads"]:
            sem_energy, sem_count = semantic[(family["mask"], layer, head)]
            total_energy, _, available = totals[(layer, head)]
            enrichment, share, prevalence = safe_enrichment(sem_energy, total_energy, sem_count, available)
            per_head.append({"layer": layer, "head": head, "enrichment": enrichment,
                             "energy_share": share, "edge_prevalence": prevalence})
        # All native heads in the same layers are the comparison population.
        layer_controls = []
        for layer in layers:
            for head in range(model.config.n_head):
                sem_energy, sem_count = semantic[(family["mask"], layer, head)]
                total_energy, _, available = totals[(layer, head)]
                layer_controls.append(safe_enrichment(sem_energy, total_energy, sem_count, available)[0])
        semantic_causal = [row for key, rows in causal_rows.items() if key.startswith(name + "/") for row in rows]
        semantic_kl = [row["semantic_kl"] for row in semantic_causal]
        control_kl = [row["control_kl"] for row in semantic_causal]
        families[name] = {
            "interpretation": family["interpretation"], "mask": family["mask"],
            "per_head": per_head, "median_enrichment": median([row["enrichment"] for row in per_head]),
            "same_mask_control_median_enrichment": median(layer_controls),
            "causal_sites": len(semantic_causal), "median_semantic_kl": median(semantic_kl),
            "median_control_kl": median(control_kl),
            "semantic_control_kl_ratio": median(semantic_kl) / max(median(control_kl), 1e-30),
            "median_abs_target_logprob_change": median([abs(row["semantic_target_logprob_change"]) for row in semantic_causal]),
            "examples": sorted(semantic_causal, key=lambda row: row["semantic_kl"], reverse=True)[:12],
        }
    hypotheses = {
        "S1_semantic_routes_enrich": {"supported": all(row["median_enrichment"] >= 2 for row in families.values()),
            "family_medians": {name: row["median_enrichment"] for name, row in families.items()}},
        "S2_frozen_heads_beat_layer_controls": {"supported": all(row["median_enrichment"] > row["same_mask_control_median_enrichment"] for row in families.values()),
            "differences": {name: row["median_enrichment"] - row["same_mask_control_median_enrichment"] for name, row in families.items()}},
        "S3_semantic_edges_are_causal": {"supported": all(row["median_semantic_kl"] > 0 and row["median_abs_target_logprob_change"] > 0 for row in families.values())},
        "S4_semantic_edges_beat_matched_controls": {"supported": sum(row["semantic_control_kl_ratio"] >= 1.5 for row in families.values()) >= 2,
            "ratios": {name: row["semantic_control_kl_ratio"] for name, row in families.items()}},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    payload = {"config": config, "boundary_ids": sorted(boundary_ids), "hypotheses": hypotheses,
               "families": families, "causal_records": causal_rows}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Held-out semantic validation of unsupervised motifs", "",
             "Semantic hypotheses were frozen after interpreting the weight-discovered motifs, then evaluated on untouched FineWeb rows 19,200–19,999.", "", "## Locked outcomes", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| family | route enrichment | comparison enrichment | causal sites | semantic KL | matched-control KL | ratio | |Δ target log p| |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in families.items():
        lines.append(f"| {name} | {row['median_enrichment']:.2f} | {row['same_mask_control_median_enrichment']:.2f} | {row['causal_sites']} | {row['median_semantic_kl']:.3g} | {row['median_control_kl']:.3g} | {row['semantic_control_kl_ratio']:.2f} | {row['median_abs_target_logprob_change']:.3g} |")
    lines += ["", "Routing enrichment is semantic-edge energy share divided by that edge type's prevalence. The comparison column is the median across all native heads in the same layers. Finite controls remove a nonsemantic source edge at the same head and destination with the closest available local-write magnitude."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "families": {name: {key: value for key, value in row.items() if key not in ("per_head", "examples")} for name, row in families.items()}}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
