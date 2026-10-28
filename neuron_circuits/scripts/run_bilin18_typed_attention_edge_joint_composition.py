#!/usr/bin/env python3
"""Execute frozen typed-edge attention programs jointly through Layers 0--5."""

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
    resolve_categories,
    token_categories,
)


CONFIG = Path("configs/bilin18_typed_attention_edge_joint_composition.json")
OUTPUT = TENSOR_ROOT / "bilin18_typed_attention_edge_joint_composition.json"
REPORT = TENSOR_ROOT / "BILIN18_TYPED_ATTENTION_EDGE_JOINT_COMPOSITION.md"


def kl(reference, prediction):
    logp = F.log_softmax(reference.float(), -1)
    logq = F.log_softmax(prediction.float(), -1)
    return (logp.exp() * (logp - logq)).sum(-1).clamp_min(0)


def build_variant_specs(prefix_ends, methods):
    specs = [{"name": "native", "prefix_end": -1, "method": "native"}]
    for prefix_end in prefix_ends:
        specs.append({"name": f"none_0_{prefix_end}", "prefix_end": prefix_end, "method": "none"})
        for method in methods:
            specs.append({"name": f"{method}_0_{prefix_end}", "prefix_end": prefix_end, "method": method})
    return specs


@torch.inference_mode()
def select_supports(model, discovery, categories, config):
    heads = model.config.n_head
    distances = len(config["distance_bins"]) - 1
    category_count = int(config.get("category_count", int(np.max(categories)) + 1 if not callable(categories) else 0))
    if category_count <= 0:
        raise ValueError("contextual category functions require config['category_count']")
    selection_config = {
        **config,
        "batch_size": config["selection_batch_size"],
    }
    supports = {}
    for layer in config["layers"]:
        print("select joint typed supports layer", layer, flush=True)
        probes, _, _ = operator_probe_vectors(
            model.blocks[layer], config["operator_probes"], config["seed"] + layer,
        )
        scores, _ = discovery_scores(
            model, layer, discovery, categories, selection_config, probes,
        )
        order = scores.argsort(descending=True).cpu().numpy()
        active = torch.where(scores > 0)[0].cpu().numpy()
        generator = np.random.default_rng(config["seed"] + 1000 + layer)
        fine32, fine128 = order[:32], order[:128]
        supports[layer] = {
            "fine32": fine32,
            "fine128": fine128,
            "random128": generator.choice(active, min(128, len(active)), replace=False),
            "all": np.arange(len(scores)),
            "drop_category": closure(fine32, "category", heads, distances, category_count),
            "drop_distance": closure(fine32, "distance", heads, distances, category_count),
            "drop_head": closure(fine32, "head", heads, distances, category_count),
        }
        for budget in config.get("support_budgets", []):
            supports[layer][f"fine{budget}"] = order[:budget]
    return supports


@torch.inference_mode()
def joint_logits(model, ids, categories_by_token, config, supports, specs):
    dimension = model.config.n_embd
    heads = model.config.n_head
    length = config["sequence_length"]
    variants = len(specs)
    batch = len(ids)
    base_x0 = F.rms_norm(model.embedding(ids), (dimension,)).float()
    x0 = base_x0.unsqueeze(0).repeat(variants, 1, 1, 1).flatten(0, 1)
    x, bus = x0, None
    base_category = resolve_categories(categories_by_token, ids.cpu().numpy())
    distance = distance_ids(range(length), length, config["distance_bins"], "cuda")
    category_count = int(config.get("category_count", int(np.max(categories_by_token)) + 1 if not callable(categories_by_token) else 0))
    base_type_ids = edge_type_ids(base_category, distance, heads, category_count)
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        details = attention_details(
            block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt,
        )
        bus = details["value_bus"]
        if layer in config["layers"]:
            writes = []
            pattern = details["pattern"].reshape(variants, batch, heads, length, length)
            values = details["mixed_value"].reshape(variants, batch, length, heads, -1)
            native_writes = details["write"].reshape(variants, batch, length, dimension)
            for variant, spec in enumerate(specs):
                if layer > spec["prefix_end"] or spec["method"] == "native":
                    writes.append(native_writes[variant])
                    continue
                if spec["method"] == "none":
                    writes.append(torch.zeros_like(native_writes[variant]))
                    continue
                support = torch.as_tensor(supports[layer][spec["method"]], device="cuda")
                keep = torch.isin(base_type_ids, support)
                attended = torch.einsum(
                    "bqhk,bkhd->bqhd",
                    pattern[variant].permute(0, 2, 1, 3) * keep,
                    values[variant],
                )
                writes.append(block.attn.c_proj(attended.reshape(batch, length, dimension)).float())
            write = torch.stack(writes).flatten(0, 1)
        else:
            write = details["write"].float()
        pre = mixed + write
        state = F.rms_norm(pre, (dimension,))
        x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))).float()
    return (30 * torch.tanh(raw / 30)).reshape(variants, batch, length, -1)


@torch.inference_mode()
def evaluate(model, raw, categories, config, supports):
    specs = build_variant_specs(config["prefix_ends"], config["methods"])
    spec_index = {spec["name"]: index for index, spec in enumerate(specs)}
    totals = {
        prefix: {
            method: {"error_kl": 0.0, "ce_delta": 0.0, "top1": 0}
            for method in config["methods"]
        }
        for prefix in config["prefix_ends"]
    }
    denominators = {prefix: 0.0 for prefix in config["prefix_ends"]}
    maximum_error = 0.0
    count = 0
    positions = torch.as_tensor(config["positions"], device="cuda")
    for start in range(0, len(raw), config["evaluation_batch_size"]):
        batch_raw = raw[start:start + config["evaluation_batch_size"]]
        ids = torch.as_tensor(batch_raw[:, :config["sequence_length"]], device="cuda")
        logits = joint_logits(model, ids, categories, config, supports, specs)
        native = logits[0, :, positions]
        target = torch.as_tensor(batch_raw[:, np.asarray(config["positions"]) + 1], device="cuda")
        native_ce = F.cross_entropy(native.flatten(0, 1), target.flatten(), reduction="none").reshape_as(target)
        reference = torch.stack([reference_logits(model, ids, int(position)) for position in config["positions"]], 1)
        maximum_error = max(maximum_error, float((native - reference).abs().max()))
        for prefix in config["prefix_ends"]:
            baseline = logits[spec_index[f"none_0_{prefix}"], :, positions]
            denominators[prefix] += float(kl(native, baseline).sum())
            for method in config["methods"]:
                prediction = logits[spec_index[f"{method}_0_{prefix}"], :, positions]
                row = totals[prefix][method]
                row["error_kl"] += float(kl(native, prediction).sum())
                predicted_ce = F.cross_entropy(
                    prediction.flatten(0, 1), target.flatten(), reduction="none",
                ).reshape_as(target)
                row["ce_delta"] += float((predicted_ce - native_ce).sum())
                row["top1"] += int((prediction.argmax(-1) == native.argmax(-1)).sum())
        count += target.numel()
    results = {}
    for prefix in config["prefix_ends"]:
        results[str(prefix)] = {"sites": count, "baseline_kl": denominators[prefix] / count, "methods": {}}
        for method, row in totals[prefix].items():
            results[str(prefix)]["methods"][method] = {
                "kl_recovery": 1 - row["error_kl"] / max(denominators[prefix], 1e-30),
                "mean_ce_delta": row["ce_delta"] / count,
                "top1_agreement": row["top1"] / count,
            }
    return results, maximum_error


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    categories = token_categories(flags)
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(
        corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]],
        dtype=np.int64,
    )
    evaluation = np.asarray(
        corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1],
        dtype=np.int64,
    )
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    supports = select_supports(model, discovery, categories, config)
    results, maximum_error = evaluate(model, evaluation, categories, config, supports)
    last = str(config["prefix_ends"][-1])
    hypotheses = {
        "J0_native_branch_matches_checkpoint": {
            "supported": maximum_error <= 1e-4, "maximum_error": maximum_error,
        },
        "J1_fine128_composes_through_six_layers": {
            "supported": results[last]["methods"]["fine128"]["kl_recovery"] >= .40,
        },
        "J2_fine32_retains_nontrivial_six_layer_behavior": {
            "supported": results[last]["methods"]["fine32"]["kl_recovery"] >= .10,
        },
        "J3_larger_fine_program_dominates": {
            "supported": all(
                results[str(prefix)]["methods"]["fine128"]["kl_recovery"]
                > results[str(prefix)]["methods"]["fine32"]["kl_recovery"]
                for prefix in config["prefix_ends"]
            ),
        },
        "J4_broad_closure_improves_joint_program": {
            "supported": max(
                results[last]["methods"][method]["kl_recovery"]
                for method in ("drop_category", "drop_distance", "drop_head")
            ) > results[last]["methods"]["fine32"]["kl_recovery"],
        },
        "J5_programs_do_not_depend_on_native_intermediate_states": {
            "supported": all(
                results[str(prefix)]["methods"]["fine128"]["kl_recovery"] > 0
                for prefix in config["prefix_ends"]
            ),
        },
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    payload = {"config": config, "hypotheses": hypotheses, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = [
        "# Joint composition of typed attention-edge programs", "",
        "Discovery-frozen typed-edge programs are executed at every token position in progressively deeper prefixes of Layers 0--5. Each later layer sees the states produced by the earlier approximate programs. The remaining suffix is native. KL recovery is normalized against deleting all attention writes in the same prefix.", "",
        f"Maximum native-branch logit error: `{maximum_error:.2e}`.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Approximate layers | Fine32 | Fine128 | Drop category | Drop distance | Drop head | Fine128 CE / top1 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for prefix in config["prefix_ends"]:
        row = results[str(prefix)]["methods"]
        lines.append(
            f"| 0--{prefix} | {row['fine32']['kl_recovery']:.3f} | {row['fine128']['kl_recovery']:.3f} | "
            f"{row['drop_category']['kl_recovery']:.3f} | {row['drop_distance']['kl_recovery']:.3f} | "
            f"{row['drop_head']['kl_recovery']:.3f} | {row['fine128']['mean_ce_delta']:+.4f} / {row['fine128']['top1_agreement']:.3f} |"
        )
    lines += [
        "", "## Guardrails", "",
        "This is closed-loop composition: QK scores, values, RMSNorm, and bilinear MLPs are recomputed from each approximation's own states. It is not a polynomial expansion. The no-attention denominator changes with prefix depth, so recovery values compare approximation quality within a prefix, not absolute causal importance across prefixes. The token-category vocabulary remains human-declared, and every approximate layer alters all destinations, making this stricter than the prior one-destination suffix test.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
