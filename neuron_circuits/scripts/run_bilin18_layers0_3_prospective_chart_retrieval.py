#!/usr/bin/env python3
"""Prospectively retrieve early semantic suffix-gradient charts from forward state."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_attention_layer_path_program import rotate
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_stratified_gradient_subspace import batch_gradients
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layers0_3_prospective_chart_retrieval.json")
OUTPUT = TENSOR_ROOT / "bilin18_layers0_3_prospective_chart_retrieval.json"
ARRAYS = TENSOR_ROOT / "bilin18_layers0_3_prospective_chart_retrieval.npz"
REPORT = TENSOR_ROOT / "BILIN18_LAYERS0_3_PROSPECTIVE_CHART_RETRIEVAL.md"
FIGURE = TENSOR_ROOT / "bilin18_layers0_3_prospective_chart_retrieval.png"


def unit_rows(array):
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-30)


def current_state_signatures(model, trace, ids, source_layer, position):
    dimension = model.config.n_embd
    token = model.embedding(ids)[:, position].float()
    source_pre = F.rms_norm(trace[source_layer]["pre"].float(), (dimension,))[:, position]
    source_output = F.rms_norm(trace[source_layer]["output"].float(), (dimension,))[:, position]
    final_output = F.rms_norm(trace[-1]["output"].float(), (dimension,))[:, position]
    path = torch.cat([
        F.rms_norm(trace[layer]["output"].float(), (dimension,))[:, position]
        for layer in range(source_layer, len(model.blocks))
    ], dim=-1)
    return {
        "token": token,
        "source_pre": source_pre,
        "source_output": source_output,
        "final_output": final_output,
        "path": path,
    }


def attention_pattern_signature(model, trace, source_layer, position, width):
    """Native current-query attention patterns from the first suffix window."""
    rows = []
    for layer in range(source_layer + 1, min(source_layer + 1 + width, len(model.blocks))):
        block = model.blocks[layer]
        previous = trace[layer - 1]["output"].float()
        x0 = trace[layer]["x0"].float()
        mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
        state = F.rms_norm(mixed, (mixed.shape[-1],))
        batch, length, _ = state.shape
        heads, head = block.attn.n_head, block.attn.head_dim

        def projected(module):
            raw = F.linear(state, module.weight.float()).view(batch, length, heads, head)
            cosine, sine = block.attn.rotary(raw)
            return rotate(F.rms_norm(raw, (head,)), cosine.float(), sine.float())

        q1, k1 = projected(block.attn.c_q), projected(block.attn.c_k)
        q2, k2 = projected(block.attn.c_q2), projected(block.attn.c_k2)
        first = torch.einsum("bhd,bkhd->bhk", q1[:, position], k1) / head
        second = torch.einsum("bhd,bkhd->bhk", q2[:, position], k2) / head
        pattern = first * second
        pattern[..., position + 1:] = 0
        rows.append(pattern.flatten(1))
    return torch.cat(rows, dim=1)


def nearest(bank_signature, query_signature, count, allowed=None):
    bank = unit_rows(bank_signature)
    query = np.asarray(query_signature, dtype=np.float32)
    query = query / max(float(np.linalg.norm(query)), 1e-30)
    score = bank @ query
    if allowed is not None:
        score = np.where(allowed, score, -np.inf)
    order = np.argpartition(score, -count)[-count:]
    return order[np.argsort(score[order])[::-1]], score


def average_gradient(gradients, indices, unit=False):
    selected = gradients[indices].astype(np.float32)
    if unit:
        selected = unit_rows(selected)
    return selected.mean(0)


def oracle_projection(exact, gradients):
    _, singular, vh = np.linalg.svd(gradients.astype(np.float32), full_matrices=False)
    keep = singular > singular[0] * 1e-6
    basis = vh[keep]
    return (exact @ basis.T) @ basis


def select_discovery(source, causal, old_flags, rng, contexts):
    corpus = np.load(source["corpus"], mmap_mode="r")
    rows = corpus[source["start_row"]:source["start_row"] + source["discovery_rows"]]
    feature_indices = [FEATURES.index(name) for name in causal["features"]]
    selections = {}
    for position in source["positions"]:
        chosen_rows, chosen_features = [], []
        for feature in feature_indices:
            candidates = np.flatnonzero(~old_flags[np.asarray(rows[:, position + 1], dtype=np.int64), feature].astype(bool))
            picked = rng.choice(candidates, size=contexts, replace=False)
            chosen_rows.extend(int(value) for value in picked)
            chosen_features.extend([feature] * contexts)
        selections[position] = (chosen_rows, chosen_features)
    return rows, selections


def collect_bank(model, config, causal, source, flags, contrasts):
    rng = np.random.default_rng(config["seed"])
    rows, selections = select_discovery(
        source, causal, flags, rng, config["discovery_contexts_per_feature_position"],
    )
    storage = {str(layer): {"gradients": [], "feature": [], "position": [], "row": [], "token": [], "signatures": {name: [] for name in ("token", "source_pre", "source_output", "final_output", "path", "attention")}} for layer in config["source_layers"]}
    for position, (row_indices, feature_indices) in selections.items():
        ids = torch.as_tensor(np.asarray(rows[row_indices, :source["sequence_length"]], dtype=np.int64), device="cuda")
        contrast_batch = contrasts[:, feature_indices].T
        with torch.no_grad(): trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        for layer in config["source_layers"]:
            gradients = batch_gradients(model, trace, layer, position, contrast_batch)
            with torch.no_grad():
                signatures = current_state_signatures(model, trace, ids, layer, position)
                signatures["attention"] = attention_pattern_signature(model, trace, layer, position, config["attention_signature_width"])
            target = storage[str(layer)]
            target["gradients"].extend(gradients.cpu().numpy().astype(np.float16))
            target["feature"].extend(feature_indices)
            target["position"].extend([position] * len(row_indices))
            target["row"].extend(row_indices)
            target["token"].extend(ids[:, position].cpu().tolist())
            for name, values in signatures.items():
                target["signatures"][name].extend(values.cpu().numpy().astype(np.float16))
        print("chart bank position", position, flush=True)
    arrays = {}
    for layer in config["source_layers"]:
        target = storage[str(layer)]
        target["gradients"] = np.asarray(target["gradients"], dtype=np.float16)
        for key in ("feature", "position", "row", "token"):
            target[key] = np.asarray(target[key], dtype=np.int64)
        for name in target["signatures"]:
            target["signatures"][name] = np.asarray(target["signatures"][name], dtype=np.float16)
        arrays[f"layer{layer}_gradients"] = target["gradients"]
        arrays[f"layer{layer}_feature"] = target["feature"]
        arrays[f"layer{layer}_position"] = target["position"]
        arrays[f"layer{layer}_row"] = target["row"]
        arrays[f"layer{layer}_token"] = target["token"]
        for name, values in target["signatures"].items():
            arrays[f"layer{layer}_{name}"] = values
    np.savez_compressed(ARRAYS, **arrays)
    return storage


def prediction_bank(bank, feature):
    mask = bank["feature"] == feature
    return {
        "gradients": bank["gradients"][mask].astype(np.float32),
        "positions": bank["position"][mask],
        "tokens": bank["token"][mask],
        "signatures": {name: values[mask].astype(np.float32) for name, values in bank["signatures"].items()},
    }


def summarize(records, layer, method):
    rows = [row for row in records if row["layer"] == layer]
    values = [row["metrics"][method] for row in rows]
    shifts = np.asarray([row["finite_shifts"].get(method, np.nan) for row in rows], dtype=np.float64)
    result = {
        key + "_median": float(np.median([value[key] for value in values]))
        for key in ("gradient_cosine", "contribution_cosine", "true_leverage_ratio")
    }
    result["gradient_cosine_q10"] = float(np.quantile([value["gradient_cosine"] for value in values], .1))
    if np.isfinite(shifts).any():
        valid = shifts[np.isfinite(shifts)]
        result.update({"finite_directional_fraction": float(np.mean(valid > 0)), "finite_mean_shift": float(valid.mean())})
    return result


def main():
    config = json.loads(CONFIG.read_text())
    causal = json.loads(Path(config["source_config"]).read_text())
    old = json.loads(Path(config["source_results"]).read_text())
    source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrasts = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    bank = collect_bank(model, config, causal, source, flags, contrasts)
    corpus = np.load(source["corpus"], mmap_mode="r")
    heldout_start = source["start_row"] + source["discovery_rows"]
    rng = np.random.default_rng(config["seed"] + 1)
    permutations = {}
    for layer in config["source_layers"]:
        permutations[str(layer)] = {}
        for feature_name in causal["features"]:
            feature = FEATURES.index(feature_name); size = int(np.sum(bank[str(layer)]["feature"] == feature))
            permutations[str(layer)][str(feature)] = rng.permutation(size)
    records = []
    finite_methods = ["feature_mean", "position_mean", "token_k4", "source_output_k4", "final_output_k4", "path_k4", "path_exclude_position_k4", "attention_k4", "path_shuffled_k4", "oracle_span", "exact"]
    for site_index, row in enumerate(old["records"]):
        feature = FEATURES.index(row["feature"]); contrast = contrasts[:, feature]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64)
        recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad():
            donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"]))
            recipient_trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        position = row["recipient_position"]
        for layer in config["source_layers"]:
            feature_bank = prediction_bank(bank[str(layer)], feature)
            delta = product_at(model, donor_trace, layer, row["donor_position"]) - product_at(model, recipient_trace, layer, position)
            exact = semantic_gradient(model, recipient_trace, layer, position, contrast)
            with torch.no_grad():
                query_signatures = current_state_signatures(model, recipient_trace, recipient_ids, layer, position)
                query_signatures["attention"] = attention_pattern_signature(model, recipient_trace, layer, position, config["attention_signature_width"])
            query_signatures = {name: value[0].cpu().numpy().astype(np.float32) for name, value in query_signatures.items()}
            predictions = {
                "feature_mean": feature_bank["gradients"].mean(0),
                "position_mean": feature_bank["gradients"][feature_bank["positions"] == position].mean(0),
            }
            neighbor_metadata = {}
            primary = config["primary_neighbors"]
            for signature_name in ("token", "source_pre", "source_output", "final_output", "attention"):
                indices, scores = nearest(feature_bank["signatures"][signature_name], query_signatures[signature_name], primary)
                predictions[f"{signature_name}_k{primary}"] = average_gradient(feature_bank["gradients"], indices)
                neighbor_metadata[signature_name] = {"indices": indices.tolist(), "similarity": float(np.median(scores[indices])), "same_position_fraction": float(np.mean(feature_bank["positions"][indices] == position)), "same_token_fraction": float(np.mean(feature_bank["tokens"][indices] == int(recipient_raw[position])))}
            for count in config["neighbors"]:
                indices, scores = nearest(feature_bank["signatures"]["path"], query_signatures["path"], count)
                predictions[f"path_k{count}"] = average_gradient(feature_bank["gradients"], indices)
                if count == primary:
                    neighbor_metadata["path"] = {"indices": indices.tolist(), "similarity": float(np.median(scores[indices])), "same_position_fraction": float(np.mean(feature_bank["positions"][indices] == position)), "same_token_fraction": float(np.mean(feature_bank["tokens"][indices] == int(recipient_raw[position])))}
            allowed = feature_bank["positions"] != position
            indices, scores = nearest(feature_bank["signatures"]["path"], query_signatures["path"], primary, allowed)
            predictions[f"path_exclude_position_k{primary}"] = average_gradient(feature_bank["gradients"], indices)
            path_indices = np.asarray(neighbor_metadata["path"]["indices"])
            permutation = permutations[str(layer)][str(feature)]
            predictions[f"path_shuffled_k{primary}"] = average_gradient(feature_bank["gradients"], permutation[path_indices])
            exact_np = exact.cpu().numpy().astype(np.float32)
            predictions["oracle_span"] = oracle_projection(exact_np, feature_bank["gradients"])
            predictions["exact"] = exact_np
            prediction_tensors = {name: torch.as_tensor(value, device="cuda") for name, value in predictions.items()}
            patches = {"base": torch.zeros_like(delta)}
            for name in finite_methods:
                support = (prediction_tensors[name] * delta).topk(config["support_rank"]).indices
                patches[name] = selected_delta(delta, support)
            with torch.no_grad(): logits = logits_for_deltas(model, layer, recipient_trace[layer], patches, position)
            base_score = class_score(logits["base"], contrast)
            finite = {name: class_score(value, contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({
                "layer": layer, "feature": row["feature"], "selection": row["selection"], "pair": row["pair"], "position": position,
                "metrics": {name: metrics(value, exact, delta, config["support_rank"]) for name, value in prediction_tensors.items()},
                "finite_shifts": finite, "neighbors": neighbor_metadata,
            })
        print(f"evaluation {site_index + 1}/{len(old['records'])} {row['feature']} {row['selection']}", flush=True)
    methods = list(records[0]["metrics"])
    summaries = {str(layer): {method: summarize(records, layer, method) for method in methods} for layer in config["source_layers"]}
    primary_name = f"path_k{config['primary_neighbors']}"; excluded_name = f"path_exclude_position_k{config['primary_neighbors']}"; shuffled_name = f"path_shuffled_k{config['primary_neighbors']}"
    cells = [summaries[str(layer)] for layer in config["source_layers"]]
    hypotheses = {
        "R0_feature_chart_span_is_expressive": {"supported": all(cell["oracle_span"]["gradient_cosine_median"] >= .75 for cell in cells)},
        "R1_path_retrieval_beats_feature_mean": {"supported": all(cell[primary_name]["gradient_cosine_median"] - cell["feature_mean"]["gradient_cosine_median"] >= .10 for cell in cells)},
        "R2_path_retrieval_beats_position_lookup": {"supported": all(cell[primary_name]["gradient_cosine_median"] - cell["position_mean"]["gradient_cosine_median"] >= .05 for cell in cells)},
        "R3_weight_attention_signature_beats_source_state": {"supported": all(cell[f"attention_k{config['primary_neighbors']}"]["gradient_cosine_median"] - cell[f"source_output_k{config['primary_neighbors']}"]["gradient_cosine_median"] >= .05 for cell in cells)},
        "R4_path_retrieval_is_not_only_position": {"supported": all(cell[primary_name]["gradient_cosine_median"] - cell[excluded_name]["gradient_cosine_median"] <= .10 for cell in cells)},
        "R5_path_retrieval_beats_shuffled_chart_map": {"supported": all(cell[primary_name]["gradient_cosine_median"] - cell[shuffled_name]["gradient_cosine_median"] >= .10 for cell in cells)},
        "R6_path_retrieval_preserves_leverage": {"supported": all(cell[primary_name]["true_leverage_ratio_median"] >= .50 for cell in cells)},
        "R7_path_retrieval_is_finitely_directional": {"supported": all(cell[primary_name]["finite_directional_fraction"] >= .75 for cell in cells)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    display = ["feature_mean", "position_mean", f"token_k{primary}", f"source_output_k{primary}", f"final_output_k{primary}", primary_name, excluded_name, f"attention_k{primary}", shuffled_name, "oracle_span", "exact"]
    lines = ["# Prospective Layers 0–3 chart retrieval", "", "A bank of 512 disjoint FineWeb charts contains two examples for each of eight semantic objectives at each of 32 positions. Held-out gradients are never used for chart selection. Retrieval uses only forward-pass signatures; the oracle span sees the held-out gradient only as an expressivity upper bound.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Prospective chart retrieval](bilin18_layers0_3_prospective_chart_retrieval.png)", "", "| Layer | Mean | Position | Token K4 | Source K4 | Final K4 | Path K4 | Path excluding position | Attention K4 | Shuffled path | Oracle span | Exact |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        cell = summaries[str(layer)]; lines.append("| " + str(layer) + " | " + " | ".join(f"{cell[name]['gradient_cosine_median']:.3f}" for name in display) + " |")
    lines += ["", "| Layer | Path contribution cosine | Path leverage | Path finite direction | Oracle-span cosine |", "|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        cell = summaries[str(layer)]; path = cell[primary_name]; lines.append(f"| {layer} | {path['contribution_cosine_median']:.3f} | {path['true_leverage_ratio_median']:.3f} | {path['finite_directional_fraction']:.3f} | {cell['oracle_span']['gradient_cosine_median']:.3f} |")
    lines += ["", "## Guardrails", "", "This is an atlas-retrieval test, not yet a parametric interpreter. The full-path signature requires a complete forward pass and may encode nearly the same state distinctions that determine the output. The attention signature is more architecture-specific but still uses native attention patterns from eight blocks. The oracle-span row establishes what the finite chart bank could represent if its coefficients were known; it is not a prospective result."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    x = np.arange(len(config["source_layers"]))
    for method in ("feature_mean", "position_mean", f"source_output_k{primary}", f"final_output_k{primary}", primary_name, f"attention_k{primary}", "oracle_span"):
        axes[0].plot(x, [summaries[str(layer)][method]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=method)
    axes[0].set_xticks(x, config["source_layers"]); axes[0].set(xlabel="Source layer", ylabel="Median gradient cosine", title="Can forward state select the suffix chart?"); axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    for layer in config["source_layers"]:
        axes[1].plot(config["neighbors"], [summaries[str(layer)][f"path_k{k}"]["gradient_cosine_median"] for k in config["neighbors"]], marker="o", label=f"Layer {layer}")
    axes[1].set(xlabel="Path-signature neighbors", ylabel="Median gradient cosine", title="Atlas smoothing frontier"); axes[1].set_xscale("log", base=2); axes[1].grid(alpha=.2); axes[1].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
