#!/usr/bin/env python3
"""Full semantic-Jacobian atlas and continuous forward-state routers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_stratified_gradient_subspace import batch_gradients
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_layers0_3_prospective_chart_retrieval import (
    attention_pattern_signature, current_state_signatures, nearest, oracle_projection,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layers0_3_semantic_jacobian_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_layers0_3_semantic_jacobian_atlas.json"
ARRAYS = TENSOR_ROOT / "bilin18_layers0_3_semantic_jacobian_atlas.npz"
REPORT = TENSOR_ROOT / "BILIN18_LAYERS0_3_SEMANTIC_JACOBIAN_ATLAS.md"
FIGURE = TENSOR_ROOT / "bilin18_layers0_3_semantic_jacobian_atlas.png"
PARTIAL = TENSOR_ROOT / "bilin18_layers0_3_semantic_jacobian_atlas.partial.json"


def normalized_centered_rows(values, mean=None):
    values = np.asarray(values, dtype=np.float32)
    if mean is None:
        mean = values.mean(0, keepdims=True)
    centered = values - mean
    return centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-30), mean


def make_kernel(signature):
    coordinates, mean = normalized_centered_rows(signature)
    kernel = coordinates @ coordinates.T
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    return {"coordinates": coordinates, "mean": mean, "eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


def kernel_weights(model, query, alpha, allowed=None):
    centered = np.asarray(query, dtype=np.float32)[None] - model["mean"]
    centered /= max(float(np.linalg.norm(centered)), 1e-30)
    cross = model["coordinates"] @ centered[0]
    vectors = model["eigenvectors"]
    weights = vectors @ ((vectors.T @ cross) / (model["eigenvalues"] + alpha))
    if allowed is not None:
        # Refit the same linear kernel on the permitted atlas subset. This is a
        # strict no-same-position control, not zeroing weights after fitting.
        subset = model["coordinates"][allowed]
        kernel = subset @ subset.T
        weights = np.linalg.solve(kernel + alpha * np.eye(len(kernel), dtype=np.float32), subset @ centered[0])
        full = np.zeros(len(model["coordinates"]), dtype=np.float32)
        full[allowed] = weights
        weights = full
    return weights.astype(np.float32)


def kernel_prediction(model, query, targets, alpha, allowed=None):
    weights = kernel_weights(model, query, alpha, allowed)
    mean = targets.astype(np.float32).mean(0)
    return mean + weights @ (targets.astype(np.float32) - mean)


def collect_atlas(model, config, causal, source, contrasts):
    corpus = np.load(source["corpus"], mmap_mode="r")
    rows = corpus[source["start_row"]:source["start_row"] + source["discovery_rows"]]
    feature_indices = [FEATURES.index(name) for name in causal["features"]]
    rng = np.random.default_rng(config["seed"])
    storage = {str(layer): {"gradients": [], "position": [], "row": [], "token": [], "signatures": {name: [] for name in ("path", "attention", "source_output", "final_output")}} for layer in config["source_layers"]}
    for position in source["positions"]:
        selected = rng.choice(len(rows), size=config["contexts_per_position"], replace=False)
        for start in range(0, len(selected), config["collection_context_batch"]):
            chosen = selected[start:start + config["collection_context_batch"]]
            base_ids = torch.as_tensor(np.asarray(rows[chosen, :source["sequence_length"]], dtype=np.int64), device="cuda")
            repeated_ids = base_ids.repeat_interleave(len(feature_indices), dim=0)
            repeated_features = feature_indices * len(chosen)
            with torch.no_grad(): repeated_trace = forward_layers_native(model, repeated_ids, len(model.blocks) - 1)
            representatives = torch.arange(len(chosen), device="cuda") * len(feature_indices)
            for layer in config["source_layers"]:
                gradients = batch_gradients(model, repeated_trace, layer, position, contrasts[:, repeated_features].T)
                gradient_cube = gradients.view(len(chosen), len(feature_indices), -1)
                with torch.no_grad():
                    signatures = current_state_signatures(model, repeated_trace, repeated_ids, layer, position)
                    attention = attention_pattern_signature(model, repeated_trace, layer, position, config["attention_signature_width"])
                target = storage[str(layer)]
                target["gradients"].extend(gradient_cube.cpu().numpy().astype(np.float16))
                target["position"].extend([position] * len(chosen)); target["row"].extend(int(value) for value in chosen)
                target["token"].extend(base_ids[:, position].cpu().tolist())
                for name in ("path", "source_output", "final_output"):
                    target["signatures"][name].extend(signatures[name][representatives].cpu().numpy().astype(np.float16))
                target["signatures"]["attention"].extend(attention[representatives].cpu().numpy().astype(np.float16))
        print("semantic Jacobian position", position, flush=True)
    arrays = {}
    for layer in config["source_layers"]:
        target = storage[str(layer)]
        target["gradients"] = np.asarray(target["gradients"], dtype=np.float16)
        for key in ("position", "row", "token"): target[key] = np.asarray(target[key], dtype=np.int64)
        for name in target["signatures"]: target["signatures"][name] = np.asarray(target["signatures"][name], dtype=np.float16)
        arrays[f"layer{layer}_gradients"] = target["gradients"]
        arrays[f"layer{layer}_position"] = target["position"]
        arrays[f"layer{layer}_row"] = target["row"]
        arrays[f"layer{layer}_token"] = target["token"]
        for name, value in target["signatures"].items(): arrays[f"layer{layer}_{name}"] = value
    np.savez_compressed(ARRAYS, **arrays)
    return storage, feature_indices


def load_atlas(config, causal):
    saved = np.load(ARRAYS)
    storage = {}
    for layer in config["source_layers"]:
        storage[str(layer)] = {
            "gradients": saved[f"layer{layer}_gradients"],
            "position": saved[f"layer{layer}_position"],
            "row": saved[f"layer{layer}_row"],
            "token": saved[f"layer{layer}_token"],
            "signatures": {
                name: saved[f"layer{layer}_{name}"]
                for name in ("path", "attention", "source_output", "final_output")
            },
        }
    return storage, [FEATURES.index(name) for name in causal["features"]]


def make_oracle_bases(atlas, config, feature_count):
    """Cache chart-span bases once; repeated per-example SVD is both slow and fragile."""
    bases = {str(layer): {} for layer in config["source_layers"]}
    with torch.no_grad():
        for layer in config["source_layers"]:
            gradients = torch.as_tensor(atlas[str(layer)]["gradients"].astype(np.float32), device="cuda")
            for slot in range(feature_count):
                basis, _ = torch.linalg.qr(gradients[:, slot].T, mode="reduced")
                bases[str(layer)][slot] = basis
    return bases


def summarize(records, layer, method):
    rows = [row for row in records if row["layer"] == layer]
    values = [row["metrics"][method] for row in rows]
    shifts = np.asarray([row["finite_shifts"].get(method, np.nan) for row in rows])
    result = {key + "_median": float(np.median([value[key] for value in values])) for key in ("gradient_cosine", "contribution_cosine", "true_leverage_ratio")}
    result["gradient_cosine_q10"] = float(np.quantile([value["gradient_cosine"] for value in values], .1))
    if np.isfinite(shifts).any():
        shifts = shifts[np.isfinite(shifts)]; result["finite_directional_fraction"] = float(np.mean(shifts > 0)); result["finite_mean_shift"] = float(shifts.mean())
    return result


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrasts = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    if ARRAYS.exists():
        atlas, feature_indices = load_atlas(config, causal)
        print("loaded existing semantic Jacobian atlas", ARRAYS, flush=True)
    else:
        atlas, feature_indices = collect_atlas(model, config, causal, source, contrasts)
    feature_slot = {feature: index for index, feature in enumerate(feature_indices)}
    kernel_models = {str(layer): {name: make_kernel(atlas[str(layer)]["signatures"][name]) for name in ("path", "attention")} for layer in config["source_layers"]}
    oracle_bases = make_oracle_bases(atlas, config, len(feature_indices))
    rng = np.random.default_rng(config["seed"] + 1); permutations = {str(layer): rng.permutation(len(atlas[str(layer)]["position"])) for layer in config["source_layers"]}
    sparse = json.loads(Path("artifacts/bilin18_tensor_decomposition/bilin18_layers0_3_prospective_chart_retrieval.json").read_text())
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]
    records = json.loads(PARTIAL.read_text())["records"] if PARTIAL.exists() else []
    completed_sites = len(records) // len(config["source_layers"])
    if completed_sites:
        print("resuming after", completed_sites, "evaluation sites", flush=True)
    finite_methods = ["feature_mean", "position_mean", "path_knn8", "path_kernel_a0.1", "path_kernel_exclude_position_a0.1", "attention_kernel_a0.1", "path_kernel_shuffled_a0.1", "oracle_span", "exact"]
    for site_index, row in enumerate(old["records"]):
        if site_index < completed_sites:
            continue
        feature = FEATURES.index(row["feature"]); slot = feature_slot[feature]; contrast = contrasts[:, feature]
        donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64); recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64)
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, max(config["source_layers"])); recipient_trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        position = row["recipient_position"]
        for layer in config["source_layers"]:
            bank = atlas[str(layer)]; target_gradients = bank["gradients"][:, slot].astype(np.float32)
            delta = product_at(model, donor_trace, layer, row["donor_position"]) - product_at(model, recipient_trace, layer, position)
            exact = semantic_gradient(model, recipient_trace, layer, position, contrast); exact_np = exact.cpu().numpy().astype(np.float32)
            with torch.no_grad():
                query = current_state_signatures(model, recipient_trace, recipient_ids, layer, position)
                query["attention"] = attention_pattern_signature(model, recipient_trace, layer, position, config["attention_signature_width"])
            query = {name: value[0].cpu().numpy().astype(np.float32) for name, value in query.items()}
            predictions = {"feature_mean": target_gradients.mean(0), "position_mean": target_gradients[bank["position"] == position].mean(0)}
            for count in config["neighbors"]:
                indices, _ = nearest(bank["signatures"]["path"], query["path"], count)
                predictions[f"path_knn{count}"] = target_gradients[indices].mean(0)
            for signature_name in ("path", "attention"):
                for alpha in config["ridge_alphas"]:
                    predictions[f"{signature_name}_kernel_a{alpha:g}"] = kernel_prediction(kernel_models[str(layer)][signature_name], query[signature_name], target_gradients, alpha)
            allowed = bank["position"] != position; alpha = config["primary_alpha"]
            predictions[f"path_kernel_exclude_position_a{alpha:g}"] = kernel_prediction(kernel_models[str(layer)]["path"], query["path"], target_gradients, alpha, allowed)
            shuffled_targets = target_gradients[permutations[str(layer)]]
            predictions[f"path_kernel_shuffled_a{alpha:g}"] = kernel_prediction(kernel_models[str(layer)]["path"], query["path"], shuffled_targets, alpha)
            basis = oracle_bases[str(layer)][slot]
            predictions["oracle_span"] = (basis @ (basis.T @ exact)).cpu().numpy().astype(np.float32)
            predictions["exact"] = exact_np
            tensors = {name: torch.as_tensor(value, device="cuda") for name, value in predictions.items()}
            patches = {"base": torch.zeros_like(delta)}
            for name in finite_methods:
                support = (tensors[name] * delta).topk(config["support_rank"]).indices; patches[name] = selected_delta(delta, support)
            with torch.no_grad(): logits = logits_for_deltas(model, layer, recipient_trace[layer], patches, position)
            base_score = class_score(logits["base"], contrast); finite = {name: class_score(value, contrast) - base_score for name, value in logits.items() if name != "base"}
            records.append({"layer": layer, "feature": row["feature"], "selection": row["selection"], "position": position, "metrics": {name: metrics(value, exact, delta, config["support_rank"]) for name, value in tensors.items()}, "finite_shifts": finite})
        PARTIAL.write_text(json.dumps({"completed_sites": site_index + 1, "records": records}, allow_nan=False))
        print(f"Jacobian atlas evaluation {site_index + 1}/{len(old['records'])}", flush=True)
    methods = list(records[0]["metrics"]); summaries = {str(layer): {method: summarize(records, layer, method) for method in methods} for layer in config["source_layers"]}
    primary_knn = f"path_knn{config['primary_neighbors']}"; primary_path = f"path_kernel_a{config['primary_alpha']:g}"; primary_attention = f"attention_kernel_a{config['primary_alpha']:g}"; excluded = f"path_kernel_exclude_position_a{config['primary_alpha']:g}"; shuffled = f"path_kernel_shuffled_a{config['primary_alpha']:g}"
    cells = [summaries[str(layer)] for layer in config["source_layers"]]
    sparse_values = [sparse["summaries"][str(layer)][f"path_k{config['primary_neighbors']}"]["gradient_cosine_median"] for layer in config["source_layers"]]
    hypotheses = {
        "J0_full_semantic_chart_span_is_expressive": {"supported": all(cell["oracle_span"]["gradient_cosine_median"] >= .65 for cell in cells)},
        "J1_path_knn_is_useful": {"supported": all(cell[primary_knn]["gradient_cosine_median"] >= .20 for cell in cells)},
        "J2_path_kernel_router_is_useful": {"supported": all(cell[primary_path]["gradient_cosine_median"] >= .35 for cell in cells)},
        "J3_path_kernel_beats_feature_mean": {"supported": all(cell[primary_path]["gradient_cosine_median"] - cell["feature_mean"]["gradient_cosine_median"] >= .20 for cell in cells)},
        "J4_path_kernel_beats_knn": {"supported": all(cell[primary_path]["gradient_cosine_median"] - cell[primary_knn]["gradient_cosine_median"] >= .10 for cell in cells)},
        "J5_path_kernel_is_not_only_position": {"supported": all(cell[primary_path]["gradient_cosine_median"] - cell[excluded]["gradient_cosine_median"] <= .10 for cell in cells)},
        "J6_path_kernel_is_finitely_directional": {"supported": all(cell[primary_path]["finite_directional_fraction"] >= .85 for cell in cells)},
        "J7_path_kernel_preserves_leverage": {"supported": all(cell[primary_path]["true_leverage_ratio_median"] >= .30 for cell in cells)},
        "J8_weight_signature_is_competitive": {"supported": all(cell[primary_path]["gradient_cosine_median"] - cell[primary_attention]["gradient_cosine_median"] <= .10 for cell in cells)},
        "J9_complete_jacobians_improve_sparse_atlas": {"supported": all(cell[primary_knn]["gradient_cosine_median"] - old_value >= .10 for cell, old_value in zip(cells, sparse_values))},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "sparse_atlas_path_knn_baseline": sparse_values, "records": records}, indent=2, allow_nan=False))
    if PARTIAL.exists(): PARTIAL.unlink()
    display = ["feature_mean", "position_mean", primary_knn, primary_path, excluded, primary_attention, shuffled, "oracle_span", "exact"]
    lines = ["# Full semantic-Jacobian atlas for Layers 0–3", "", "At each of 256 disjoint FineWeb states, the atlas stores gradients for all eight semantic output objectives. This removes the sparse atlas's confound between state sampling and output objective. Prospective routers use only forward-pass signatures; oracle projection remains an expressivity ceiling.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Semantic Jacobian atlas](bilin18_layers0_3_semantic_jacobian_atlas.png)", "", "| Layer | Mean | Position | Path K8 | Path kernel | Exclude position | Attention kernel | Shuffled | Oracle span | Exact |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        cell = summaries[str(layer)]; lines.append("| " + str(layer) + " | " + " | ".join(f"{cell[name]['gradient_cosine_median']:.3f}" for name in display) + " |")
    lines += ["", "| Layer | Path contribution cosine | Path leverage | Path finite direction | Sparse/full K8 |", "|---:|---:|---:|---:|---:|"]
    for index, layer in enumerate(config["source_layers"]):
        cell = summaries[str(layer)]; path = cell[primary_path]; lines.append(f"| {layer} | {path['contribution_cosine_median']:.3f} | {path['true_leverage_ratio_median']:.3f} | {path['finite_directional_fraction']:.3f} | {sparse_values[index]:.3f}/{cell[primary_knn]['gradient_cosine_median']:.3f} |")
    lines += ["", "## Guardrails", "", "The kernel router is a continuous atlas interpolator, not yet a short tensor program. It requires a full forward trajectory and 256 stored states. The attention signature uses native weight-derived patterns from eight blocks but omits value contents and later blocks. A broad oracle span with a weak router means the operator family is representable but its chart coordinates are not captured by these forward similarities; a weak oracle span means the bank itself is still too small or the field is not globally low rank."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained"); x = np.arange(len(config["source_layers"]))
    for method in ("feature_mean", primary_knn, primary_path, primary_attention, shuffled, "oracle_span"):
        axes[0].plot(x, [summaries[str(layer)][method]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=method)
    axes[0].set_xticks(x, config["source_layers"]); axes[0].set(xlabel="Source layer", ylabel="Median gradient cosine", title="Continuous semantic-Jacobian atlas"); axes[0].grid(alpha=.2); axes[0].legend(fontsize=8)
    for layer in config["source_layers"]:
        axes[1].plot(config["ridge_alphas"], [summaries[str(layer)][f"path_kernel_a{alpha:g}"]["gradient_cosine_median"] for alpha in config["ridge_alphas"]], marker="o", label=f"Layer {layer}")
    axes[1].set_xscale("log"); axes[1].set(xlabel="Kernel ridge alpha", ylabel="Median gradient cosine", title="Regularization frontier"); axes[1].grid(alpha=.2); axes[1].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
