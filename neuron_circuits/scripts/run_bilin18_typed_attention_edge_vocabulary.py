#!/usr/bin/env python3
"""Test a fixed vocabulary of (head, distance, source-token-type) attention edges."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_typed_attention_edge_vocabulary.json")
OUTPUT = TENSOR_ROOT / "bilin18_typed_attention_edge_vocabulary.json"
REPORT = TENSOR_ROOT / "BILIN18_TYPED_ATTENTION_EDGE_VOCABULARY.md"
FIGURE = TENSOR_ROOT / "bilin18_typed_attention_edge_vocabulary.png"

CATEGORIES = (
    "newline", "tab", "non_ascii", "numeric", "period", "comma",
    "colon_semicolon", "question_exclamation", "opening_delimiter",
    "closing_delimiter", "quote", "dash", "slash", "code_operator",
    "programming_keyword", "function_word", "suffix", "capitalized",
    "lowercase", "other_punctuation", "other",
)


def token_categories(flags):
    index = {name: FEATURES.index(name) for name in FEATURES}
    output = np.full(len(flags), len(CATEGORIES) - 1, dtype=np.int64)
    assigned = np.zeros(len(flags), dtype=bool)
    rules = [
        ("newline", ["newline"]), ("tab", ["tab"]), ("non_ascii", ["non_ascii"]),
        ("numeric", ["numeric", "alphanumeric"]), ("period", ["period"]),
        ("comma", ["comma"]), ("colon_semicolon", ["colon_semicolon"]),
        ("question_exclamation", ["question", "exclamation"]),
        ("opening_delimiter", ["opening_delimiter"]), ("closing_delimiter", ["closing_delimiter"]),
        ("quote", ["quote"]), ("dash", ["dash"]), ("slash", ["slash"]),
        ("code_operator", ["code_operator"]), ("programming_keyword", ["programming_keyword"]),
        ("function_word", ["determiner", "preposition", "conjunction", "pronoun", "auxiliary", "negation", "modal"]),
        ("suffix", ["suffix_ing", "suffix_ed", "suffix_ly", "suffix_s", "suffix_ion", "suffix_ment", "suffix_ness", "suffix_ity", "suffix_ive", "suffix_ous"]),
        ("capitalized", ["capitalized", "all_caps"]), ("lowercase", ["lowercase"]),
        ("other_punctuation", ["punctuation"]),
    ]
    for category, names in rules:
        match = np.any(flags[:, [index[name] for name in names]].astype(bool), axis=1) & ~assigned
        output[match] = CATEGORIES.index(category); assigned |= match
    return output


def distance_ids(positions, length, boundaries, device):
    destination = torch.as_tensor(positions, device=device)[:, None]
    source = torch.arange(length, device=device)[None]
    distance = destination - source
    output = torch.full_like(distance, -1)
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        output[(distance >= lower) & (distance < upper)] = index
    return output


def edge_type_ids(category, distance, heads, category_count):
    # [B,P,H,K], with negative IDs for non-causal positions.
    batch, length = category.shape; positions = distance.shape[0]
    head = torch.arange(heads, device=category.device)[None, None, :, None]
    result = head * (distance.max().item() + 1) * category_count
    result = result + distance[None, :, None, :] * category_count + category[:, None, None, :]
    result = result.expand(batch, positions, heads, length).clone()
    return result.masked_fill(distance[None, :, None, :] < 0, -1)


def resolve_categories(categories, token_ids, device="cuda"):
    """Resolve either a static token lookup or a contextual category function."""
    values = categories(token_ids) if callable(categories) else categories[token_ids]
    return torch.as_tensor(values, device=device, dtype=torch.long)


def rmsnorm_pullback_readers(pre, probes):
    dimension = pre.shape[-1]
    rms = pre.square().mean(-1).sqrt()
    dot = torch.einsum("bpd,qd->bpq", pre, probes)
    return probes[None, None] / rms[:, :, None, None] - pre[:, :, None] * dot[:, :, :, None] / (dimension * rms[:, :, None, None].pow(3))


@torch.inference_mode()
def discovery_scores(model, layer, raw, categories, config, probes):
    heads = model.config.n_head; dimension = model.config.n_embd; head_dim = dimension // heads
    category_count = int(config.get("category_count", int(np.max(categories)) + 1 if not callable(categories) else 0))
    if category_count <= 0:
        raise ValueError("contextual category functions require config['category_count']")
    distance = distance_ids(config["positions"], config["sequence_length"], config["distance_bins"], "cuda")
    type_count = heads * (len(config["distance_bins"]) - 1) * category_count
    scores = torch.zeros(type_count, device="cuda", dtype=torch.float64)
    recomposition_error = 0.0
    for start in range(0, len(raw), config["batch_size"]):
        ids = torch.as_tensor(raw[start:start + config["batch_size"], :config["sequence_length"]], device="cuda")
        raw_ids = raw[start:start + config["batch_size"], :config["sequence_length"]]
        category = resolve_categories(categories, raw_ids)
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks[:layer + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
            if block_index == layer:
                pre = mixed[:, config["positions"]].float() + details["write"][:, config["positions"]].float()
                readers = rmsnorm_pullback_readers(pre, probes)
                projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
                head_readers = torch.einsum("bpqo,hdo->bphqd", readers, projection)
                value_effect = torch.einsum("bkhd,bphqd->bphqk", details["mixed_value"].float(), head_readers)
                edge_effect = value_effect * details["pattern"][:, :, config["positions"]].permute(0, 2, 1, 3)[:, :, :, None]
                type_ids = edge_type_ids(category, distance, heads, category_count)
                n = len(ids) * len(config["positions"])
                grouped = torch.zeros(n, type_count, len(probes), device="cuda")
                flat_ids = type_ids.flatten(0, 1).reshape(n, -1)
                flat_effect = edge_effect.flatten(0, 1).permute(0, 1, 3, 2).reshape(n, -1, len(probes))
                valid = flat_ids >= 0
                safe_ids = flat_ids.clamp_min(0)[:, :, None].expand_as(flat_effect)
                grouped.scatter_add_(1, safe_ids, flat_effect * valid[:, :, None])
                scores += grouped.square().sum((0, 2)).double()
                full_effect = edge_effect.sum((2, 4))
                recomposed = grouped.sum(1).reshape_as(full_effect)
                error = (full_effect - recomposed).norm() / full_effect.norm().clamp_min(1e-30)
                recomposition_error = max(recomposition_error, float(error))
            pre_full = mixed + details["write"]; state = F.rms_norm(pre_full, (dimension,))
            x = pre_full + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    return scores, recomposition_error


def decode_type(index, config):
    categories = len(CATEGORIES); distances = len(config["distance_bins"]) - 1
    head = index // (distances * categories); remainder = index % (distances * categories)
    distance = remainder // categories; category = remainder % categories
    return {"head": int(head), "distance_bin": int(distance), "source_category": CATEGORIES[category]}


@torch.inference_mode()
def evaluate_supports(model, layer, raw, categories, supports, config, probes):
    heads = model.config.n_head; dimension = model.config.n_embd
    category_count = int(np.max(categories)) + 1
    distance = distance_ids(config["positions"], config["sequence_length"], config["distance_bins"], "cuda")
    errors = torch.zeros(len(supports), device="cuda", dtype=torch.float64)
    denominator = torch.tensor(0.0, device="cuda", dtype=torch.float64)
    full_error = 0.0
    support_tensors = [torch.as_tensor(support, device="cuda") for support in supports]
    for start in range(0, len(raw), config["batch_size"]):
        ids = torch.as_tensor(raw[start:start + config["batch_size"], :config["sequence_length"]], device="cuda")
        category = torch.as_tensor(categories[raw[start:start + config["batch_size"], :config["sequence_length"]]], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks[:layer + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt); bus = details["value_bus"]
            if block_index == layer:
                selected_mixed = mixed[:, config["positions"]].float()
                native_write = details["write"][:, config["positions"]].float()
                exact = F.rms_norm(selected_mixed + native_write, (dimension,)) @ probes.T
                baseline = F.rms_norm(selected_mixed, (dimension,)) @ probes.T
                denominator += (exact - baseline).square().sum().double()
                type_ids = edge_type_ids(category, distance, heads, category_count)
                pattern = details["pattern"][:, :, config["positions"]].permute(0, 2, 1, 3)
                for support_index, support in enumerate(support_tensors):
                    keep = torch.isin(type_ids, support)
                    attended = torch.einsum("bphk,bkhd->bphd", pattern * keep, details["mixed_value"].float())
                    write = block.attn.c_proj(attended.reshape(len(ids), len(config["positions"]), dimension)).float()
                    prediction = F.rms_norm(selected_mixed + write, (dimension,)) @ probes.T
                    errors[support_index] += (exact - prediction).square().sum().double()
                attended = torch.einsum("bphk,bkhd->bphd", pattern, details["mixed_value"].float())
                full_write = block.attn.c_proj(attended.reshape(len(ids), len(config["positions"]), dimension)).float()
                full_prediction = F.rms_norm(selected_mixed + full_write, (dimension,)) @ probes.T
                full_error = max(full_error, float((exact - full_prediction).norm() / exact.norm().clamp_min(1e-30)))
            pre = mixed + details["write"]; state = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    return (1 - errors / denominator.clamp_min(1e-30)).cpu().tolist(), full_error


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    categories = token_categories(flags)
    source = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(source[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(source[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"], :config["sequence_length"]], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; maximum_error = 0.0
    for layer in config["layers"]:
        print("typed edge vocabulary layer", layer, flush=True)
        probes, _, _ = operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + layer)
        scores, discovery_error = discovery_scores(model, layer, discovery, categories, config, probes)
        active = torch.where(scores > 0)[0]; order = scores.argsort(descending=True)
        supports, names = [], []
        selected_supports = {}
        generator = np.random.default_rng(config["seed"] + 100 + layer)
        for budget in config["budgets"]:
            chosen = order[:budget].cpu().numpy(); selected_supports[str(budget)] = chosen.tolist()
            supports.append(chosen); names.append(f"selected_{budget}")
            for random_index in range(config["random_supports"]):
                supports.append(generator.choice(active.cpu().numpy(), min(budget, len(active)), replace=False)); names.append(f"random_{budget}_{random_index}")
        all_types = np.arange(len(scores)); supports.append(all_types); names.append("all")
        recoveries, evaluation_error = evaluate_supports(model, layer, evaluation, categories, supports, config, probes)
        by_name = dict(zip(names, recoveries)); rows = {}
        for budget in config["budgets"]:
            random = [by_name[f"random_{budget}_{index}"] for index in range(config["random_supports"])]
            rows[str(budget)] = {"recovery": by_name[f"selected_{budget}"], "random_p50": float(np.median(random)), "random_p95": float(np.quantile(random, .95)), "selected_types": [decode_type(index, config) for index in selected_supports[str(budget)][:min(16, budget)]]}
        maximum_error = max(maximum_error, discovery_error, evaluation_error)
        results[str(layer)] = {"active_types": int(len(active)), "budgets": rows, "all_recovery": by_name["all"], "linear_recomposition_error": discovery_error, "exact_partition_error": evaluation_error}
    middle = np.mean([results[str(layer)]["budgets"]["32"]["recovery"] for layer in (2,3,4)])
    outside = np.mean([results[str(layer)]["budgets"]["32"]["recovery"] for layer in (0,1,5)])
    hypotheses = {
        "T0_typed_edges_partition_attention": {"supported": maximum_error <= 1e-5, "maximum_error": maximum_error},
        "T1_top128_types_are_sufficient": {"supported": all(results[str(layer)]["budgets"]["128"]["recovery"] >= .80 for layer in config["layers"])},
        "T2_top128_types_beat_random": {"supported": sum(results[str(layer)]["budgets"]["128"]["recovery"] >= results[str(layer)]["budgets"]["128"]["random_p95"] + .10 for layer in config["layers"]) >= 4},
        "T3_top32_types_are_useful": {"supported": sum(results[str(layer)]["budgets"]["32"]["recovery"] >= .50 for layer in config["layers"]) >= 4},
        "T4_middle_layers_require_broader_vocabularies": {"supported": outside - middle >= .15, "outside_mean": float(outside), "middle_mean": float(middle)},
        "T5_fixed_typed_vocabulary_generalizes": {"supported": all(results[str(layer)]["budgets"][str(budget)]["recovery"] > 0 for layer in config["layers"] for budget in config["budgets"])},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "categories": CATEGORIES, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Fixed vocabulary of typed attention edges", "", "Every causal attention edge is assigned exactly one source-token category and one relative-distance bin, then paired with its native head. Discovery ranks these typed edges by their linearized effect on a weight-derived local operator sketch. Held-out evaluation keeps only the fixed selected types, recomputes RMSNorm exactly, and compares with equal-size random vocabularies.", "", f"Maximum partition/recomposition error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Typed edge vocabulary](bilin18_typed_attention_edge_vocabulary.png)", "", "| Layer | Active types | Recovery at 8/32/128/512 | Random p95 at 32/128 |", "|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; rec = "/".join(f"{row['budgets'][str(b)]['recovery']:.3f}" for b in config["budgets"]); rand = "/".join(f"{row['budgets'][str(b)]['random_p95']:.3f}" for b in (32,128))
        lines.append(f"| {layer} | {row['active_types']} | {rec} | {rand} |")
    lines += ["", "## Guardrails", "", "The categories are a preregistered mutually exclusive human vocabulary, not learned latent features. Ranking uses a native-state linearization, while evaluation recomputes the nonlinear normalized state. Recovery concerns the attention-dependent local MLP operator, not final-token KL. Random supports are sampled from active types but are not matched for head, distance, or category counts. A positive result would establish a useful fixed routing vocabulary; it would not imply that each type has one output meaning."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for budget in config["budgets"]: axis.plot(config["layers"], [results[str(layer)]["budgets"][str(budget)]["recovery"] for layer in config["layers"]], marker="o", label=str(budget))
    axis.set(xlabel="Layer", ylabel="Attention-dependent operator recovery", title="Fixed typed-edge vocabulary"); axis.grid(alpha=.2); axis.legend(title="Types")
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": {layer:{budget:row['budgets'][budget]['recovery'] for budget in row['budgets']} for layer,row in results.items()}}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
