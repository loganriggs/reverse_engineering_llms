#!/usr/bin/env python3
"""Propagate fixed typed-edge attention programs through the exact suffix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_attention_head_semantic_causality import reference_logits
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_specificity_redteam import closure
from run_bilin18_typed_attention_edge_vocabulary import (
    CATEGORIES,
    discovery_scores,
    distance_ids,
    edge_type_ids,
    token_categories,
)


CONFIG = Path("configs/bilin18_typed_attention_edge_suffix_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_typed_attention_edge_suffix_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_TYPED_ATTENTION_EDGE_SUFFIX_CAUSALITY.md"
PARTIAL = TENSOR_ROOT / "bilin18_typed_attention_edge_suffix_causality.partial.json"


def kl(reference, prediction):
    logp = F.log_softmax(reference.float(), -1); logq = F.log_softmax(prediction.float(), -1)
    return (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)


def spearman(first, second):
    def ranks(values):
        order = np.argsort(values); result = np.empty(len(values), dtype=np.float64); result[order] = np.arange(len(values)); return result
    return float(np.corrcoef(ranks(np.asarray(first)), ranks(np.asarray(second)))[0, 1])


@torch.inference_mode()
def variants(model, ids, position, layer, supports, categories_by_token, config, probes):
    dimension = model.config.n_embd; heads = model.config.n_head
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    distance = distance_ids([position], config["sequence_length"], config["distance_bins"], "cuda")
    category = torch.as_tensor(categories_by_token[ids.cpu().numpy()], device="cuda")
    for block_index, block in enumerate(model.blocks):
        if block_index < layer:
            x, bus = block(x, bus, x0); continue
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
        type_ids = edge_type_ids(category, distance, heads, len(CATEGORIES))[:, 0]
        pattern = details["pattern"][:, :, position]
        writes = [torch.zeros(len(ids), dimension, device="cuda")]
        for support in supports:
            keep = torch.isin(type_ids, support)
            attended = torch.einsum("bhk,bkhd->bhd", pattern * keep, details["mixed_value"].float())
            writes.append(block.attn.c_proj(attended.reshape(len(ids), dimension)).float())
        native_write = details["write"][:, position].float()
        writes = [native_write] + writes
        variant_count = len(writes)
        attention = details["write"].float().unsqueeze(0).repeat(variant_count, 1, 1, 1)
        for index, write in enumerate(writes): attention[index, :, position] = write
        x0 = x0.float().unsqueeze(0).repeat(variant_count, 1, 1, 1).flatten(0, 1)
        mixed = mixed.float().unsqueeze(0).repeat(variant_count, 1, 1, 1).flatten(0, 1)
        bus = details["value_bus"].float().unsqueeze(0).repeat(variant_count, 1, 1, 1, 1).flatten(0, 1)
        pre = mixed + attention.flatten(0, 1); state = F.rms_norm(pre, (dimension,))
        local = (state[:, position] @ probes.T).reshape(variant_count, len(ids), -1)
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        for suffix in model.blocks[block_index + 1:]: x, bus = suffix(x, bus, x0)
        raw = model.lm_head(F.rms_norm(x, (dimension,))[:, position]).float()
        return (30 * torch.tanh(raw / 30)).reshape(variant_count, len(ids), -1), local
    raise AssertionError("layer not reached")


@torch.inference_mode()
def evaluate(model, raw, layer, supports, names, categories, config, probes):
    accum = {name: {"kl": 0.0, "local": 0.0, "ce": 0.0, "top1": 0} for name in names}
    denominator_kl = 0.0; denominator_local = 0.0; count = 0; maximum_error = 0.0
    for position in config["positions"]:
        for start in range(0, len(raw), config["evaluation_batch_size"]):
            batch = raw[start:start + config["evaluation_batch_size"]]
            ids = torch.as_tensor(batch[:, :config["sequence_length"]], device="cuda")
            logits, local = variants(model, ids, position, layer, supports, categories, config, probes)
            reference = reference_logits(model, ids, position)
            maximum_error = max(maximum_error, float((logits[0] - reference).abs().max()))
            targets = torch.as_tensor(batch[:, position + 1], device="cuda")
            native_ce = F.cross_entropy(logits[0], targets, reduction="none")
            base_kl = kl(logits[0], logits[1]); base_local = (local[0] - local[1]).square().sum(-1)
            denominator_kl += float(base_kl.sum()); denominator_local += float(base_local.sum()); count += len(ids)
            for index, name in enumerate(names):
                variant = index + 2
                accum[name]["kl"] += float(kl(logits[0], logits[variant]).sum())
                accum[name]["local"] += float((local[0] - local[variant]).square().sum())
                accum[name]["ce"] += float((F.cross_entropy(logits[variant], targets, reduction="none") - native_ce).sum())
                accum[name]["top1"] += int((logits[variant].argmax(-1) == logits[0].argmax(-1)).sum())
    results = {}
    for name, row in accum.items():
        results[name] = {
            "kl_recovery": 1 - row["kl"] / max(denominator_kl, 1e-30),
            "local_operator_recovery": 1 - row["local"] / max(denominator_local, 1e-30),
            "mean_ce_delta": row["ce"] / count, "top1_agreement": row["top1"] / count,
        }
    return results, maximum_error, count


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    categories = token_categories(flags)
    source = np.load(config["selection_corpus"], mmap_mode="r")
    discovery = np.asarray(source[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(source[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    selection_config = {**config, "batch_size": config["selection_batch_size"]}
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; maximum_error = 0.0; local_values = []; kl_values = []
    if PARTIAL.exists():
        saved = json.loads(PARTIAL.read_text())
        if saved.get("config") == config:
            results = saved.get("results", {})
            maximum_error = float(saved.get("maximum_error", 0.0))
    heads = model.config.n_head; distances = len(config["distance_bins"]) - 1; category_count = len(CATEGORIES)
    for layer in config["layers"]:
        if str(layer) in results:
            print("typed suffix causal layer", layer, "(resumed)", flush=True)
            continue
        print("typed suffix causal layer", layer, flush=True)
        probes, _, _ = operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)
        scores, _ = discovery_scores(model, layer, discovery, categories, selection_config, probes)
        order = scores.argsort(descending=True).cpu().numpy(); fine32 = order[:32]; fine128 = order[:128]
        supports = [torch.as_tensor(value, device="cuda") for value in (
            fine32, fine128,
            closure(fine32, "category", heads, distances, category_count),
            closure(fine32, "distance", heads, distances, category_count),
            closure(fine32, "head", heads, distances, category_count),
        )]
        names = ["fine32", "fine128", "drop_category", "drop_distance", "drop_head"]
        row, error, sites = evaluate(model, evaluation, layer, supports, names, categories, config, probes)
        maximum_error = max(maximum_error, error); results[str(layer)] = {"sites": sites, "methods": row}
        PARTIAL.write_text(json.dumps({"config": config, "maximum_error": maximum_error, "results": results}, indent=2, allow_nan=False))
    for layer in config["layers"]:
        for name in names:
            row = results[str(layer)]["methods"][name]
            local_values.append(row["local_operator_recovery"]); kl_values.append(row["kl_recovery"])
    correlation = spearman(local_values, kl_values)
    outside = np.mean([results[str(layer)]["methods"]["fine32"]["kl_recovery"] for layer in (0,1,5)])
    middle = np.mean([results[str(layer)]["methods"]["fine32"]["kl_recovery"] for layer in (2,3,4)])
    hypotheses = {
        "C0_manual_program_matches_native": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
        "C1_typed128_recovers_suffix_behavior": {"supported": all(results[str(layer)]["methods"]["fine128"]["kl_recovery"] >= .80 for layer in config["layers"])},
        "C2_typed32_is_behaviorally_useful": {"supported": sum(results[str(layer)]["methods"]["fine32"]["kl_recovery"] >= .50 for layer in config["layers"]) >= 4},
        "C3_local_operator_recovery_predicts_suffix_recovery": {"supported": correlation >= .70, "spearman": correlation},
        "C4_broader_closures_preserve_behavior": {"supported": all(max(results[str(layer)]["methods"][name]["kl_recovery"] for name in ("drop_category","drop_distance","drop_head")) >= results[str(layer)]["methods"]["fine32"]["kl_recovery"] for layer in config["layers"])},
        "C5_middle_layers_remain_broader_behaviorally": {"supported": outside - middle >= .15, "outside_mean": float(outside), "middle_mean": float(middle)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Full-suffix causality of the typed attention-edge vocabulary", "", "Fine typed-edge supports and their broader closures are selected on discovery FineWeb. At 2,048 later held-out token positions per layer, the native local attention write is replaced only at the queried position, that layer's bilinear MLP is recomputed, and the exact remaining transformer produces logits. Recovery is normalized against deleting the complete local attention write.", "", f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Fine32 local/KL | Fine128 local/KL | Drop category KL | Drop distance KL | Drop head KL | Fine32 CE / top1 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]["methods"]; pair=lambda name:f"{row[name]['local_operator_recovery']:.3f}/{row[name]['kl_recovery']:.3f}"
        lines.append(f"| {layer} | {pair('fine32')} | {pair('fine128')} | {row['drop_category']['kl_recovery']:.3f} | {row['drop_distance']['kl_recovery']:.3f} | {row['drop_head']['kl_recovery']:.3f} | {row['fine32']['mean_ce_delta']:+.4f} / {row['fine32']['top1_agreement']:.3f} |")
    lines += ["", "Local-operator versus suffix-KL Spearman correlation across the 30 method/layer cells: `" + f"{correlation:.3f}`.", "", "## Guardrails", "", "Only one layer's attention write at one destination is replaced; all other positions and the full suffix are native. KL recovery can be high even when individual top-1 predictions change, and CE delta can be negative on finite samples. Closure methods execute more fine edges than fine32 while using fewer broad predicates, so they optimize description length rather than equal execution. The selected vocabulary is human-declared and does not test whether a learned latent edge ontology would be shorter."]
    REPORT.write_text("\n".join(lines) + "\n")
    PARTIAL.unlink(missing_ok=True)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
