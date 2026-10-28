#!/usr/bin/env python3
"""Causal transfer of discovery-frozen forward feature coalitions through Layers 0--3."""

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
from run_bilin18_competent_feature_direction_dictionary import collect_candidate_scores
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_competent_feature_causal_transfer.json")
OUTPUT = TENSOR_ROOT / "bilin18_competent_feature_causal_transfer.json"
REPORT = TENSOR_ROOT / "BILIN18_COMPETENT_FEATURE_CAUSAL_TRANSFER.md"
FIGURE = TENSOR_ROOT / "bilin18_competent_feature_causal_transfer.png"


@torch.inference_mode()
def forward_products(model, ids, positions, layers):
    dimension = model.config.n_embd; x0 = F.rms_norm(model.embedding(ids), (dimension,)); x = x0; value_bus = None; result = {}
    for layer, block in enumerate(model.blocks[:max(layers) + 1]):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0; attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus); pre = mixed + attention; state = F.rms_norm(pre, (dimension,)); product = block.mlp.Left(state) * block.mlp.Right(state)
        if layer in layers: result[layer] = product[:, positions].flatten(0, 1).float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    return result


def discovery_rankings(model, rows, positions, flags, feature_indices, layers, batch_size):
    channels = model.blocks[0].mlp.Left.out_features; count = 0; positives = np.zeros(len(feature_indices), dtype=np.float64)
    sums = {layer: np.zeros(channels, dtype=np.float64) for layer in layers}; squares = {layer: np.zeros(channels, dtype=np.float64) for layer in layers}; positive_sums = {layer: np.zeros((len(feature_indices), channels), dtype=np.float64) for layer in layers}
    position_array = np.asarray(positions)
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :129], dtype=np.int64); ids = torch.as_tensor(raw[:, :128], device="cuda"); labels_np = flags[raw[:, position_array + 1].reshape(-1)][:, feature_indices].astype(np.float32); labels = torch.as_tensor(labels_np, device="cuda")
        products = forward_products(model, ids, positions, layers); count += len(labels_np); positives += labels_np.sum(0)
        for layer, product in products.items():
            sums[layer] += product.sum(0).cpu().numpy(); squares[layer] += product.square().sum(0).cpu().numpy(); positive_sums[layer] += (labels.T @ product).cpu().numpy()
        if start % 64 == 0: print(f"causal discovery stats {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    rankings = {}; gaps = {}; raw_gaps = {}
    for layer in layers:
        mean = sums[layer] / count; std = np.sqrt(np.maximum(squares[layer] / count - mean * mean, 1e-8)); positive_mean = positive_sums[layer] / positives[:, None]; negative_mean = (sums[layer][None] - positive_sums[layer]) / (count - positives)[:, None]; raw_gap = positive_mean - negative_mean; gap = raw_gap / std[None]; gaps[str(layer)] = gap.astype(np.float32); raw_gaps[str(layer)] = raw_gap.astype(np.float32); rankings[str(layer)] = np.argsort(np.abs(gap), axis=1)[:, ::-1].copy()
    return rankings, gaps, raw_gaps


def product_gradients_for_layers(model, trace, position, contrast, layers):
    current = final_adjoint(model, trace[-1]["output"], position, contrast).unsqueeze(0); result = {}
    for suffix in range(len(model.blocks) - 1, 0, -1):
        if suffix in layers: result[suffix] = current[0, 0, position] @ model.blocks[suffix].mlp.Down.weight.float()
        block = model.blocks[suffix]; updated = current + mlp_vjp(block, trace[suffix]["pre"].float(), current); previous = trace[suffix - 1]["output"].float(); x0 = trace[suffix]["x0"].float(); mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0; current = block.lambdas[0].float() * (updated + attention_vjp(block, mixed, trace[suffix - 1]["value_bus"], updated))
    if 0 in layers: result[0] = current[0, 0, position] @ model.blocks[0].mlp.Down.weight.float()
    return result


def mean_product_gradients(model, rows, positions, flags, feature_indices, contrasts, layers, count, seed):
    rng = np.random.default_rng(seed); position_array = np.asarray(positions); raw = np.asarray(rows[:, :129], dtype=np.int64); next_labels = flags[raw[:, position_array + 1].reshape(-1)]
    sums = {str(layer): [] for layer in layers}; contribution_sums = {str(layer): [] for layer in layers}
    for local_label, feature_index in enumerate(feature_indices):
        eligible_negative = []
        for position_index in range(len(positions)):
            same_position = np.arange(position_index, len(next_labels), len(positions))
            if next_labels[same_position, feature_index].any(): eligible_negative.extend(same_position[~next_labels[same_position, feature_index].astype(bool)].tolist())
        chosen = rng.choice(np.asarray(eligible_negative), size=count, replace=False); per_layer = {str(layer): [] for layer in layers}; per_contribution = {str(layer): [] for layer in layers}
        for flat in chosen:
            row = int(flat // len(positions)); position = int(positions[flat % len(positions)]); ids = torch.as_tensor(raw[row, :128], device="cuda")[None]
            with torch.no_grad(): trace = forward_layers_native(model, ids, len(model.blocks) - 1)
            position_index = int(flat % len(positions)); same_position = np.arange(position_index, len(next_labels), len(positions)); positive_same_position = same_position[next_labels[same_position, feature_index].astype(bool)]; donor_flat = int(rng.choice(positive_same_position)); donor_row = int(donor_flat // len(positions)); donor_ids = torch.as_tensor(raw[donor_row, :128], device="cuda")[None]
            with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, max(layers))
            product_gradients = product_gradients_for_layers(model, trace, position, contrasts[:, feature_index], layers)
            for layer in layers:
                product_gradient = product_gradients[layer]; per_layer[str(layer)].append(product_gradient.detach().cpu().numpy()); delta = product_at(model, donor_trace, layer, position) - product_at(model, trace, layer, position); per_contribution[str(layer)].append((product_gradient * delta).detach().cpu().numpy())
        for layer in layers:
            sums[str(layer)].append(np.mean(per_layer[str(layer)], axis=0)); contribution_sums[str(layer)].append(np.mean(per_contribution[str(layer)], axis=0))
        print(f"weight-aware discovery gradient {local_label + 1}/{len(feature_indices)}", flush=True)
    return ({layer: np.stack(values).astype(np.float32) for layer, values in sums.items()}, {layer: np.stack(values).astype(np.float32) for layer, values in contribution_sums.items()})


def choose_pairs(scores, labels, positions, feature_indices, count, rng):
    pairs = []
    for local_label, feature_index in enumerate(feature_indices):
        candidates = []
        for position_index, position in enumerate(positions):
            members = np.arange(position_index, len(labels), len(positions)); positive = members[labels[members, feature_index].astype(bool)]; negative = members[~labels[members, feature_index].astype(bool)]
            if len(positive) and len(negative):
                donor = int(positive[np.argmax(scores[positive, feature_index])]); recipient = int(negative[np.argmin(scores[negative, feature_index])]); candidates.append((float(scores[donor, feature_index] - scores[recipient, feature_index]), donor, recipient, int(position)))
        candidates.sort(reverse=True)
        if not candidates: raise RuntimeError(f"no same-position positive/negative pair for feature index {feature_index}")
        selected = [candidates[0]]
        pool = [item for item in candidates[1:] if item[0] > 0]
        if len(pool) < count - 1: raise RuntimeError(f"insufficient positive-margin positions for feature index {feature_index}")
        chosen = rng.choice(len(pool), size=count - 1, replace=False); selected.extend(pool[int(index)] for index in np.atleast_1d(chosen))
        for selection, (_, donor, recipient, position) in zip(("extreme",) + ("uniform",) * (count - 1), selected): pairs.append({"label": local_label, "feature_index": int(feature_index), "selection": selection, "position": position, "donor_row": int(donor // len(positions)), "recipient_row": int(recipient // len(positions)), "model_margin": float(scores[donor, feature_index] - scores[recipient, feature_index])})
    return pairs


def summarize(records, layer, method):
    rows = [row for row in records if row["layer"] == layer]; semantic = np.asarray([row["semantic_shifts"][method] for row in rows]); null = np.asarray([row["null_shifts"][method] for row in rows])
    return {"pairs": len(rows), "directional_fraction": float(np.mean(semantic > 0)), "semantic_mean_shift": float(semantic.mean()), "semantic_median_shift": float(np.median(semantic)), "null_mean_shift": float(null.mean()), "null_mean_absolute_shift": float(np.abs(null).mean()), "semantic_to_null_absolute_ratio": float(semantic.mean() / max(np.abs(null).mean(), 1e-30))}


def jaccard(first, second):
    first = set(first); second = set(second)
    return len(first & second) / len(first | second)


def main():
    config = json.loads(CONFIG.read_text()); census_config = json.loads(Path(config["census_config"]).read_text()); census = json.loads(Path(config["census_results"]).read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive; contrasts_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]
    competent = [row["feature"] for row in census["features"] if row["competent"]]; feature_indices = [FEATURES.index(name) for name in competent]; semantic_contrasts = torch.as_tensor(contrasts_np, device="cuda", dtype=torch.float32)
    shared_rng = np.random.default_rng(config["seed"] + 2); shared_np = contrasts_np[:, feature_indices][shared_rng.permutation(len(flags))]; shared_contrasts = torch.as_tensor(shared_np, device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(census_config["corpus"], mmap_mode="r"); discovery = corpus[census_config["start_row"]:census_config["start_row"] + census_config["discovery_rows"]]; heldout_start = census_config["start_row"] + census_config["discovery_rows"]; heldout = corpus[heldout_start:heldout_start + census_config["heldout_rows"]]
    rankings, gaps, raw_gaps = discovery_rankings(model, discovery, census_config["positions"], flags, feature_indices, config["layers"], config["batch_size"])
    mean_gradients, mean_pair_contributions = mean_product_gradients(model, discovery, census_config["positions"], flags, feature_indices, semantic_contrasts, config["layers"], config["posthoc_gradient_contexts_per_feature"], config["seed"] + 3)
    weight_aware_rankings = {str(layer): np.argsort(raw_gaps[str(layer)] * mean_gradients[str(layer)], axis=1)[:, ::-1].copy() for layer in config["layers"]}
    paired_contribution_rankings = {str(layer): np.argsort(mean_pair_contributions[str(layer)], axis=1)[:, ::-1].copy() for layer in config["layers"]}
    candidate_scores, candidate_labels, _, _ = collect_candidate_scores(model, heldout, census_config["positions"], flags, semantic_contrasts, config["batch_size"]); pairs = choose_pairs(candidate_scores, candidate_labels, census_config["positions"], feature_indices, config["pairs_per_feature"], np.random.default_rng(config["seed"]))
    channel_rng = np.random.default_rng(config["seed"] + 1); channel_count = model.blocks[0].mlp.Left.out_features
    random_channels = {str(layer): np.stack([channel_rng.choice(channel_count, size=64, replace=False) for _ in competent]) for layer in config["layers"]}; records = []
    for pair_index, pair in enumerate(pairs):
        donor_ids = torch.as_tensor(np.asarray(heldout[pair["donor_row"], :census_config["sequence_length"]], dtype=np.int64), device="cuda")[None]; recipient_ids = torch.as_tensor(np.asarray(heldout[pair["recipient_row"], :census_config["sequence_length"]], dtype=np.int64), device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, max(config["layers"])); recipient_trace = forward_layers_native(model, recipient_ids, len(model.blocks) - 1)
        semantic_contrast = semantic_contrasts[:, pair["feature_index"]]; null_contrast = shared_contrasts[:, pair["label"]]
        exact_product_gradients = product_gradients_for_layers(model, recipient_trace, pair["position"], semantic_contrast, config["layers"])
        for layer in config["layers"]:
            delta = product_at(model, donor_trace, layer, pair["position"]) - product_at(model, recipient_trace, layer, pair["position"]); methods = {"base": torch.zeros_like(delta), "full": delta}
            order = rankings[str(layer)][pair["label"]]
            for size in config["coalition_sizes"]: methods[f"coalition_{size}"] = selected_delta(delta, torch.as_tensor(order[:size], device="cuda"))
            for size in (64, 256): methods[f"weight_aware_{size}"] = selected_delta(delta, torch.as_tensor(weight_aware_rankings[str(layer)][pair["label"], :size], device="cuda"))
            for size in (64, 256): methods[f"paired_contribution_{size}"] = selected_delta(delta, torch.as_tensor(paired_contribution_rankings[str(layer)][pair["label"], :size], device="cuda"))
            oracle_order = (exact_product_gradients[layer] * delta).argsort(descending=True)
            for size in (64, 256): methods[f"endpoint_oracle_{size}"] = selected_delta(delta, oracle_order[:size])
            for scale in (.1, .25, .5):
                methods[f"coalition_64_scale_{scale}"] = scale * methods["coalition_64"]
                methods[f"weight_aware_64_scale_{scale}"] = scale * methods["weight_aware_64"]
            methods["random_64"] = selected_delta(delta, torch.as_tensor(random_channels[str(layer)][pair["label"]], device="cuda")); wrong = (pair["label"] + 1) % len(competent); methods["wrong_feature_64"] = selected_delta(delta, torch.as_tensor(rankings[str(layer)][wrong, :64], device="cuda"))
            with torch.no_grad(): logits = logits_for_deltas(model, layer, recipient_trace[layer], methods, pair["position"])
            semantic_base = class_score(logits["base"], semantic_contrast); null_base = class_score(logits["base"], null_contrast); semantic_shifts = {name: class_score(value, semantic_contrast) - semantic_base for name, value in logits.items() if name != "base"}; null_shifts = {name: class_score(value, null_contrast) - null_base for name, value in logits.items() if name != "base"}
            records.append({**pair, "feature": competent[pair["label"]], "layer": layer, "semantic_shifts": semantic_shifts, "null_shifts": null_shifts, "endpoint_oracle_top64": oracle_order[:64].cpu().tolist(), "frozen_top64": order[:64].tolist(), "paired_contribution_top64": paired_contribution_rankings[str(layer)][pair["label"], :64].tolist()})
        print(f"causal pair {pair_index + 1}/{len(pairs)} {competent[pair['label']]} {pair['selection']}", flush=True)
    methods = list(records[0]["semantic_shifts"]); summaries = {str(layer): {method: summarize(records, layer, method) for method in methods} for layer in config["layers"]}
    overlap_summaries = {}
    for layer in config["layers"]:
        rows = [row for row in records if row["layer"] == layer]; same = []; different = []; frozen = []; paired = []
        for first in range(len(rows)):
            frozen.append(jaccard(rows[first]["endpoint_oracle_top64"], rows[first]["frozen_top64"])); paired.append(jaccard(rows[first]["endpoint_oracle_top64"], rows[first]["paired_contribution_top64"]))
            for second in range(first + 1, len(rows)):
                value = jaccard(rows[first]["endpoint_oracle_top64"], rows[second]["endpoint_oracle_top64"])
                (same if rows[first]["label"] == rows[second]["label"] else different).append(value)
        overlap_summaries[str(layer)] = {"same_feature_oracle_jaccard_mean": float(np.mean(same)), "different_feature_oracle_jaccard_mean": float(np.mean(different)), "same_to_different_ratio": float(np.mean(same) / max(np.mean(different), 1e-30)), "oracle_to_frozen_jaccard_mean": float(np.mean(frozen)), "oracle_to_paired_contribution_jaccard_mean": float(np.mean(paired)), "random_expected_jaccard": 64 / (2 * model.blocks[0].mlp.Left.out_features - 64)}
    hypotheses = {"C0_feature_coverage": {"supported": len(competent) >= 25}, "C1_frozen_coalitions_transfer": {"supported": all(summaries[str(layer)]["coalition_64"]["directional_fraction"] >= .65 for layer in config["layers"])}, "C2_frozen_coalitions_beat_random_channels": {"supported": all(summaries[str(layer)]["coalition_64"]["semantic_mean_shift"] > summaries[str(layer)]["random_64"]["semantic_mean_shift"] for layer in config["layers"])}, "C3_feature_specific_channels_matter": {"supported": all(summaries[str(layer)]["coalition_64"]["semantic_mean_shift"] > summaries[str(layer)]["wrong_feature_64"]["semantic_mean_shift"] for layer in config["layers"])}, "C4_effect_is_semantically_specific": {"supported": all(summaries[str(layer)]["coalition_64"]["semantic_mean_shift"] > summaries[str(layer)]["coalition_64"]["null_mean_absolute_shift"] for layer in config["layers"])}, "C5_layer3_is_not_less_causal": {"supported": summaries["3"]["coalition_64"]["directional_fraction"] >= summaries["1"]["coalition_64"]["directional_fraction"]}}
    posthoc = {str(layer): {"weight_aware_64_beats_activation_64_direction": summaries[str(layer)]["weight_aware_64"]["directional_fraction"] > summaries[str(layer)]["coalition_64"]["directional_fraction"], "weight_aware_64_beats_random_mean": summaries[str(layer)]["weight_aware_64"]["semantic_mean_shift"] > summaries[str(layer)]["random_64"]["semantic_mean_shift"], "paired_contribution_64_beats_random_mean": summaries[str(layer)]["paired_contribution_64"]["semantic_mean_shift"] > summaries[str(layer)]["random_64"]["semantic_mean_shift"], "scaled_weight_aware": {str(scale): summaries[str(layer)][f"weight_aware_64_scale_{scale}"] for scale in (.1, .25, .5)}} for layer in config["layers"]}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "posthoc_weight_aware": posthoc, "oracle_overlap_summaries": overlap_summaries, "competent_features": competent, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Competent-feature causal transfer", "", f"Discovery-frozen activation-gap coalitions are estimated on {census_config['discovery_rows']} FineWeb rows, then tested on {len(pairs)} disjoint same-position positive-donor/negative-recipient pairs spanning {len(competent)} competent token properties. Product channels are patched and the complete nonlinear suffix is recomputed.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Post-hoc weight-aware red team", "", "A discovery-only selector ranks channels by the product of the positive-minus-negative mean activation change and the mean exact semantic suffix gradient on four disjoint negative contexts per feature. A stricter covariance-aware variant instead ranks the mean per-pair product `E[delta * gradient]`. An endpoint-informed oracle ranks the exact held-out `delta * gradient` and is a positive control for the intervention pipeline.", "", "![Causal transfer](bilin18_competent_feature_causal_transfer.png)", "", "| Layer | Coalition-64 direction/mean | Pair-contribution-64 direction/mean | Pair-contribution-256 direction/mean | Endpoint-oracle-64 direction/mean | Endpoint-oracle-256 direction/mean | Full mean |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = summaries[str(layer)]; c = row["coalition_64"]; p64 = row["paired_contribution_64"]; p256 = row["paired_contribution_256"]; o64 = row["endpoint_oracle_64"]; o256 = row["endpoint_oracle_256"]; lines.append(f"| {layer} | {c['directional_fraction']:.3f}/{c['semantic_mean_shift']:.4f} | {p64['directional_fraction']:.3f}/{p64['semantic_mean_shift']:.4f} | {p256['directional_fraction']:.3f}/{p256['semantic_mean_shift']:.4f} | {o64['directional_fraction']:.3f}/{o64['semantic_mean_shift']:.4f} | {o256['directional_fraction']:.3f}/{o256['semantic_mean_shift']:.4f} | {row['full']['semantic_mean_shift']:.4f} |")
    lines += ["", "### Coalition identity overlap", "", "| Layer | Same-feature oracle Jaccard | Different-feature oracle Jaccard | Ratio | Oracle/frozen | Oracle/discovery-pair |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = overlap_summaries[str(layer)]; lines.append(f"| {layer} | {row['same_feature_oracle_jaccard_mean']:.4f} | {row['different_feature_oracle_jaccard_mean']:.4f} | {row['same_to_different_ratio']:.2f} | {row['oracle_to_frozen_jaccard_mean']:.4f} | {row['oracle_to_paired_contribution_jaccard_mean']:.4f} |")
    lines += ["", "### Finite-scale red team", "", "| Layer | Weight-aware 0.1 direction/mean | 0.25 | 0.5 | 1.0 |", "|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        row = summaries[str(layer)]; cells = []
        for scale in (.1, .25, .5):
            value = row[f"weight_aware_64_scale_{scale}"]; cells.append(f"{value['directional_fraction']:.3f}/{value['semantic_mean_shift']:.4f}")
        value = row["weight_aware_64"]; cells.append(f"{value['directional_fraction']:.3f}/{value['semantic_mean_shift']:.4f}"); lines.append(f"| {layer} | " + " | ".join(cells) + " |")
    lines += ["", "## Guardrails", "", "The two pairs per feature are an initial causal screen, not a final per-feature estimate. Pair selection requires the model's feature score to be higher for the positive donor than the negative recipient and includes one extreme and one seeded non-extreme position. Coalitions are frozen from a disjoint discovery split and never selected using intervention effects. Because features overlap, the next-feature coalition is an imperfect wrong-feature control; random channels and the Gram-matched output contrast provide complementary baselines. Scaled patches distinguish failure of a local direction from nonlinear failure at the full donor magnitude."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["layers"])); fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for method in ("coalition_1", "coalition_16", "coalition_64", "coalition_256"): axes[0].plot(x, [summaries[str(layer)][method]["directional_fraction"] for layer in config["layers"]], marker="o", label=method.replace("coalition_", "K="))
    axes[0].set_xticks(x, config["layers"]); axes[0].set(xlabel="Source layer", ylabel="Positive semantic shift fraction", title="Frozen coalition causal transfer"); axes[0].legend(); axes[0].grid(alpha=.2)
    width = .25
    for offset, method in enumerate(("coalition_64", "random_64", "wrong_feature_64")): axes[1].bar(x + (offset - 1) * width, [summaries[str(layer)][method]["semantic_mean_shift"] for layer in config["layers"]], width=width, label=method)
    axes[1].set_xticks(x, config["layers"]); axes[1].set(xlabel="Source layer", ylabel="Mean semantic score shift", title="Specificity controls"); axes[1].legend(); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
