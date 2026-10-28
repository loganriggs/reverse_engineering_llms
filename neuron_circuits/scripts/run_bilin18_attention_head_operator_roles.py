#!/usr/bin/env python3
"""Decompose early attention writes into heads and test operator/semantic roles."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_attention_head_operator_roles.json")
OUTPUT = TENSOR_ROOT / "bilin18_attention_head_operator_roles.json"
REPORT = TENSOR_ROOT / "BILIN18_ATTENTION_HEAD_OPERATOR_ROLES.md"
FIGURE = TENSOR_ROOT / "bilin18_attention_head_operator_roles.png"


def operator_map(basis, rank, device):
    mean = basis["mean"].to(device)
    vectors = basis["state_vectors"].to(device)
    rms = basis["state_rms"].to(device)
    modes = basis["operator_modes_first"].to(device)[:, :rank]
    mapping = (vectors / rms[None]) @ modes
    return mean, mapping, basis["operator_singular_first"].to(device)[:rank]


def coordinates(state, mean, mapping):
    return (state - mean) @ mapping


def normalized_state(mixed, head_writes, subset):
    if subset:
        pre = mixed + head_writes[:, subset].sum(1)
    else:
        pre = mixed
    return F.rms_norm(pre, (pre.shape[-1],))


def distance_masks(positions, length, boundaries, device):
    destination = torch.as_tensor(positions, device=device)[:, None]
    source = torch.arange(length, device=device)[None]
    distance = destination - source
    masks = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        masks.append((distance >= lower) & (distance < upper))
    return torch.stack(masks)


@torch.inference_mode()
def collect_layer(model, layer, corpus, start, rows, config, flags, eligible):
    source = np.load(corpus, mmap_mode="r")
    panel = source[start:start + rows]
    positions = np.asarray(config["positions"])
    dimension = model.config.n_embd; heads = model.config.n_head; head_dim = dimension // heads
    mixed_rows, head_rows, labels, current = [], [], [], []
    distance_total = torch.zeros(heads, len(config["distance_bins"]) - 1, device="cuda", dtype=torch.float64)
    feature_total = torch.zeros(heads, len(eligible), device="cuda", dtype=torch.float64)
    total_transport = torch.zeros(heads, device="cuda", dtype=torch.float64)
    distance_baseline = torch.zeros_like(distance_total)
    feature_baseline = torch.zeros_like(feature_total)
    total_baseline = torch.zeros_like(total_transport)
    masks = distance_masks(positions, config["sequence_length"], config["distance_bins"], "cuda")
    flag_tensor = torch.as_tensor(flags[:, eligible], device="cuda", dtype=torch.float32)
    maximum_error = 0.0
    for row_start in range(0, len(panel), config["batch_size"]):
        raw = np.asarray(panel[row_start:row_start + config["batch_size"], :config["sequence_length"] + 1], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :config["sequence_length"]], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks[:layer + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            if block_index == layer:
                attended = torch.einsum("bhqk,bkhd->bhqd", details["pattern"], details["mixed_value"])
                selected = attended[:, :, positions].permute(0, 2, 1, 3).contiguous()
                projection = block.attn.c_proj.weight.float().T.reshape(heads, head_dim, dimension)
                head_write = torch.einsum("bphd,hdo->bpho", selected.float(), projection)
                native = details["write"][:, positions].float()
                error = (head_write.sum(2) - native).norm() / native.norm().clamp_min(1e-30)
                maximum_error = max(maximum_error, float(error))
                mixed_rows.append(mixed[:, positions].flatten(0, 1).float().cpu())
                head_rows.append(head_write.flatten(0, 1).float().cpu())
                labels.append(torch.as_tensor(flags[raw[:, positions + 1].reshape(-1)][:, eligible], dtype=torch.uint8))
                current.append(torch.as_tensor(raw[:, positions].reshape(-1), dtype=torch.int32))
                value_norm = details["mixed_value"].float().norm(dim=-1).permute(0, 2, 1)
                transport = details["pattern"][:, :, positions].abs().float() * value_norm[:, :, None, :]
                available = masks.any(0)[None, None]
                baseline_transport = value_norm[:, :, None, :] * available
                total_transport += transport.sum((0, 2, 3)).double()
                total_baseline += baseline_transport.sum((0, 2, 3)).double()
                for bin_index, mask in enumerate(masks):
                    distance_total[:, bin_index] += (transport * mask[None, None]).sum((0, 2, 3)).double()
                    distance_baseline[:, bin_index] += (baseline_transport * mask[None, None]).sum((0, 2, 3)).double()
                source_features = flag_tensor[ids]
                feature_total += torch.einsum("bhpk,bkf->hf", transport, source_features).double()
                feature_baseline += torch.einsum("bhpk,bkf->hf", baseline_transport, source_features).double()
            pre = mixed + details["write"]
            state = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    total = total_transport.clamp_min(1e-30)
    distance_profile = distance_total / total[:, None]
    feature_profile = feature_total / total[:, None]
    base_total = total_baseline.clamp_min(1e-30)
    distance_expected = distance_baseline / base_total[:, None]
    feature_expected = feature_baseline / base_total[:, None]
    epsilon = 1e-8
    return {
        "mixed": torch.cat(mixed_rows), "heads": torch.cat(head_rows),
        "labels": torch.cat(labels), "current": torch.cat(current),
        "distance_profile": distance_profile.float().cpu(),
        "source_feature_profile": feature_profile.float().cpu(),
        "distance_enrichment": torch.log2((distance_profile + epsilon) / (distance_expected + epsilon)).float().cpu(),
        "source_feature_enrichment": torch.log2((feature_profile + epsilon) / (feature_expected + epsilon)).float().cpu(),
        "recomposition_error": maximum_error,
    }


def weighted_errors(target, prediction, weight):
    return ((target - prediction) * weight).square().sum(-1)


@torch.inference_mode()
def subset_search(train, test, basis, block, config):
    mean, mapping, singular = operator_map(basis, config["rank"], "cuda")
    train_mixed = train["mixed"].cuda(); train_heads = train["heads"].cuda()
    test_mixed = test["mixed"].cuda(); test_heads = test["heads"].cuda()
    train_target_state = normalized_state(train_mixed, train_heads, tuple(range(train_heads.shape[1])))
    test_target_state = normalized_state(test_mixed, test_heads, tuple(range(test_heads.shape[1])))
    train_target = coordinates(train_target_state, mean, mapping)
    test_target = coordinates(test_target_state, mean, mapping)
    train_base = coordinates(normalized_state(train_mixed, train_heads, ()), mean, mapping)
    test_base_state = normalized_state(test_mixed, test_heads, ())
    test_base = coordinates(test_base_state, mean, mapping)
    train_denominator = float(weighted_errors(train_target, train_base, singular).sum())
    test_denominator = float(weighted_errors(test_target, test_base, singular).sum())
    results = {}
    heads = train_heads.shape[1]
    for budget in config["budgets"]:
        combinations = list(itertools.combinations(range(heads), budget))
        train_totals = []
        for start in range(0, len(combinations), config["combination_chunk"]):
            group = combinations[start:start + config["combination_chunk"]]
            mask = torch.zeros(len(group), heads, device="cuda")
            for index, subset in enumerate(group): mask[index, list(subset)] = 1
            write = torch.einsum("ch,nhd->ncd", mask, train_heads)
            state = F.rms_norm(train_mixed[:, None] + write, (train_mixed.shape[1],))
            prediction = coordinates(state, mean, mapping)
            train_totals.extend(weighted_errors(train_target[:, None], prediction, singular).sum(0).cpu().tolist())
        selected_index = int(np.argmin(train_totals)); selected = combinations[selected_index]
        evaluation_totals = []; per_example = torch.full((len(test_mixed),), torch.inf, device="cuda")
        selected_error = None
        for start in range(0, len(combinations), config["combination_chunk"]):
            group = combinations[start:start + config["combination_chunk"]]
            mask = torch.zeros(len(group), heads, device="cuda")
            for index, subset in enumerate(group): mask[index, list(subset)] = 1
            write = torch.einsum("ch,nhd->ncd", mask, test_heads)
            state = F.rms_norm(test_mixed[:, None] + write, (test_mixed.shape[1],))
            prediction = coordinates(state, mean, mapping)
            errors = weighted_errors(test_target[:, None], prediction, singular)
            totals = errors.sum(0)
            evaluation_totals.extend(totals.cpu().tolist())
            per_example = torch.minimum(per_example, errors.min(1).values)
            if selected_index >= start and selected_index < start + len(group):
                selected_error = float(totals[selected_index - start])
        recoveries = 1 - np.asarray(evaluation_totals) / test_denominator
        coordinate_recovery = 1 - selected_error / test_denominator
        oracle_recovery = 1 - float(per_example.sum()) / test_denominator
        selected_state = normalized_state(test_mixed, test_heads, selected)
        probe, _, _ = operator_probe_vectors(block, config["operator_eval_probes"], config["seed"] + budget)
        target_sketch = test_target_state @ probe.T
        base_sketch = test_base_state @ probe.T
        selected_sketch = selected_state @ probe.T
        sketch_denominator = (target_sketch - base_sketch).square().sum().clamp_min(1e-30)
        sketch_recovery = float(1 - (target_sketch - selected_sketch).square().sum() / sketch_denominator)
        results[str(budget)] = {
            "selected_heads": list(selected),
            "discovery_coordinate_recovery": 1 - train_totals[selected_index] / train_denominator,
            "evaluation_coordinate_recovery": coordinate_recovery,
            "evaluation_operator_sketch_recovery": sketch_recovery,
            "all_subsets_p50": float(np.quantile(recoveries, .50)),
            "all_subsets_p95": float(np.quantile(recoveries, .95)),
            "per_example_oracle_coordinate_recovery": oracle_recovery,
        }
    return results, {
        "train_target": train_target.cpu(), "test_target": test_target.cpu(),
        "train_base": train_base.cpu(), "test_base": test_base.cpu(),
        "train_one_head": one_head_coordinates(train_mixed, train_heads, mean, mapping).cpu(),
        "test_one_head": one_head_coordinates(test_mixed, test_heads, mean, mapping).cpu(),
    }


def one_head_coordinates(mixed, heads, mean, mapping):
    state = F.rms_norm(mixed[:, None] + heads, (mixed.shape[1],))
    return coordinates(state, mean, mapping)


def decoder(train, labels, validation, validation_labels):
    mean = train.mean(0); std = train.std(0).clamp_min(1e-4)
    normalized = (train - mean) / std
    positive = labels.bool()
    gap = normalized[positive].mean(0) - normalized[~positive].mean(0)
    return ((validation - mean) / std) @ gap


def supervised_head_roles(representations, train_labels, test_labels, eligible, rows, positions):
    train_one = representations["train_one_head"]
    test_one = representations["test_one_head"]
    train_target = representations["train_target"]
    test_target = representations["test_target"]
    row_index = torch.arange(rows).repeat_interleave(positions)
    fitting = row_index < rows // 2; validation = ~fitting
    records = []
    for local_feature, feature in enumerate(eligible):
        head_validation = []
        for head in range(train_one.shape[1]):
            score = decoder(
                train_one[fitting, head], train_labels[fitting, local_feature],
                train_one[validation, head], train_labels[validation, local_feature],
            )
            head_validation.append(auc(train_labels[validation, local_feature].numpy(), score.numpy()))
        selected = int(np.nanargmax(head_validation))
        score = decoder(train_one[:, selected], train_labels[:, local_feature], test_one[:, selected], test_labels[:, local_feature])
        full_score = decoder(train_target, train_labels[:, local_feature], test_target, test_labels[:, local_feature])
        records.append({
            "feature": FEATURES[feature], "selected_head": selected,
            "validation_auc": float(head_validation[selected]),
            "evaluation_head_auc": float(auc(test_labels[:, local_feature].numpy(), score.numpy())),
            "evaluation_full_auc": float(auc(test_labels[:, local_feature].numpy(), full_score.numpy())),
        })
    return records


def cosine_rows(first, second):
    first = F.normalize(first - first.mean(1, keepdim=True), dim=1)
    second = F.normalize(second - second.mean(1, keepdim=True), dim=1)
    return first @ second.T


def split_role_stability(train, test, permutations, seed):
    first = torch.cat((train["distance_enrichment"], train["source_feature_enrichment"]), 1)
    second = torch.cat((test["distance_enrichment"], test["source_feature_enrichment"]), 1)
    score = cosine_rows(first, second)
    diagonal = float(score.diag().mean())
    generator = np.random.default_rng(seed)
    null = [float(score[torch.arange(len(score)), torch.as_tensor(generator.permutation(len(score)))].mean()) for _ in range(permutations)]
    return {"same_head_cosine": diagonal, "shuffled_p95": float(np.quantile(null, .95)), "matrix": score.tolist()}


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    bases = torch.load(config["bases"], map_location="cpu", weights_only=True)
    competence = json.loads(Path(config["competence_results"]).read_text())
    eligible = [FEATURES.index(row["feature"]) for row in competence["features"] if row["competent"]]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval()
    for module in (model.embedding, model.blocks):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; maximum_error = 0.0
    for layer in config["layers"]:
        print("collect head roles layer", layer, "discovery", flush=True)
        train = collect_layer(model, layer, config["corpus"], config["discovery_start"], config["rows_per_split"], config, flags, eligible)
        print("collect head roles layer", layer, "evaluation", flush=True)
        test = collect_layer(model, layer, config["corpus"], config["evaluation_start"], config["rows_per_split"], config, flags, eligible)
        maximum_error = max(maximum_error, train["recomposition_error"], test["recomposition_error"])
        subsets, representations = subset_search(train, test, bases[str(layer)], model.blocks[layer], config)
        semantic = supervised_head_roles(
            representations, train["labels"], test["labels"], eligible,
            config["rows_per_split"], len(config["positions"]),
        )
        stability = split_role_stability(train, test, 1000, config["seed"] + layer)
        results[str(layer)] = {
            "subsets": subsets, "semantic": semantic, "source_role_stability": stability,
            "discovery_distance_profiles": train["distance_profile"].tolist(),
            "evaluation_distance_profiles": test["distance_profile"].tolist(),
            "evaluation_source_feature_profiles": test["source_feature_profile"].tolist(),
            "discovery_distance_enrichment": train["distance_enrichment"].tolist(),
            "evaluation_distance_enrichment": test["distance_enrichment"].tolist(),
            "discovery_source_feature_enrichment": train["source_feature_enrichment"].tolist(),
            "evaluation_source_feature_enrichment": test["source_feature_enrichment"].tolist(),
        }
        del train, test, representations
    close_counts = {}
    for feature in [FEATURES[index] for index in eligible]:
        close_counts[feature] = sum(
            next(row for row in results[str(layer)]["semantic"] if row["feature"] == feature)["evaluation_head_auc"]
            >= next(row for row in results[str(layer)]["semantic"] if row["feature"] == feature)["evaluation_full_auc"] - .05
            for layer in config["layers"]
        )
    hypotheses = {
        "A0_heads_recompose_attention": {"supported": maximum_error <= 1e-5},
        "A1_four_fixed_heads_recover_attention_operator": {"supported": all(results[str(layer)]["subsets"]["4"]["evaluation_operator_sketch_recovery"] >= .80 for layer in config["layers"])},
        "A2_four_heads_are_nontrivially_selected": {"supported": sum(results[str(layer)]["subsets"]["4"]["evaluation_coordinate_recovery"] >= results[str(layer)]["subsets"]["4"]["all_subsets_p95"] + .03 for layer in config["layers"]) >= 4},
        "A3_fixed_four_approaches_example_oracle": {"supported": all(results[str(layer)]["subsets"]["4"]["evaluation_coordinate_recovery"] >= results[str(layer)]["subsets"]["4"]["per_example_oracle_coordinate_recovery"] - .10 for layer in config["layers"])},
        "A4_two_heads_are_insufficient": {"supported": sum(results[str(layer)]["subsets"]["2"]["evaluation_operator_sketch_recovery"] < .80 for layer in config["layers"]) >= 4},
        "A5_single_heads_have_supervised_roles": {"supported": sum(count > 0 for count in close_counts.values()) >= len(close_counts) / 2},
        "A6_source_roles_are_split_stable": {"supported": all(results[str(layer)]["source_role_stability"]["same_head_cosine"] > results[str(layer)]["source_role_stability"]["shuffled_p95"] for layer in config["layers"])},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "maximum_recomposition_error": maximum_error, "hypotheses": hypotheses, "close_feature_layer_counts": close_counts, "results": results}, indent=2, allow_nan=False))
    lines = ["# Attention-head roles in the early operator interface", "", "Each Layer-0--5 attention write is exactly split into its nine projected native-head writes. Fixed head subsets are selected only on discovery FineWeb to reconstruct the layer's rank-64 operator-state coordinates. Evaluation reports recovery relative to deleting all attention: 0 is the mixed-residual-only baseline and 1 is the complete native attention write.", "", f"Maximum summed-head recomposition error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Head recovery](bilin18_attention_head_operator_roles.png)", "", "| Layer | Fixed 1/2/4/6 head operator recovery | Four heads | Four-head coordinate / all-subset p95 / example oracle | Source-role same/shuffled-p95 |", "|---:|---:|---|---:|---:|"]
    for layer in config["layers"]:
        row = results[str(layer)]; subsets = row["subsets"]
        recovery = "/".join(f"{subsets[str(k)]['evaluation_operator_sketch_recovery']:.3f}" for k in (1,2,4,6))
        four = subsets["4"]
        lines.append(f"| {layer} | {recovery} | {four['selected_heads']} | {four['evaluation_coordinate_recovery']:.3f} / {four['all_subsets_p95']:.3f} / {four['per_example_oracle_coordinate_recovery']:.3f} | {row['source_role_stability']['same_head_cosine']:.3f}/{row['source_role_stability']['shuffled_p95']:.3f} |")
    examples = []
    for layer in config["layers"]:
        ranked = sorted(results[str(layer)]["semantic"], key=lambda row: row["evaluation_head_auc"] - .5, reverse=True)[:3]
        examples.append(f"Layer {layer}: " + ", ".join(f"{row['feature']}→H{row['selected_head']} ({row['evaluation_head_auc']:.3f} vs full {row['evaluation_full_auc']:.3f})" for row in ranked))
    lines += ["", "## Strongest validation-selected single-head semantic decoders", ""] + [f"- {row}" for row in examples]
    lines += ["", "## Guardrails", "", "A head subset is a native architectural decomposition, but high recovery does not make a head monosemantic. The per-example oracle uses evaluation outcomes and is only an upper control. The all-subset p95 is not an independent-sample significance test because subsets share heads. Semantic labels describe the next token and use a validation split for head selection. Source-role stability uses log enrichment relative to value-norm-weighted causally available sources, rather than raw corpus feature frequencies; it still establishes a routing signature, not the head's complete causal purpose."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    layers = config["layers"]
    for budget in (1, 2, 4, 6, 9):
        axes[0].plot(layers, [results[str(layer)]["subsets"][str(budget)]["evaluation_operator_sketch_recovery"] for layer in layers], marker="o", label=f"fixed {budget}")
    axes[0].set(xlabel="Layer", ylabel="Attention-dependent operator recovery", title="Fixed native-head sufficiency")
    axes[1].plot(layers, [results[str(layer)]["subsets"]["4"]["evaluation_coordinate_recovery"] for layer in layers], marker="o", label="fixed four")
    axes[1].plot(layers, [results[str(layer)]["subsets"]["4"]["all_subsets_p95"] for layer in layers], marker="o", label="all-subset p95")
    axes[1].plot(layers, [results[str(layer)]["subsets"]["4"]["per_example_oracle_coordinate_recovery"] for layer in layers], marker="o", label="example oracle")
    axes[1].set(xlabel="Layer", ylabel="Coordinate recovery", title="Selection and oracle controls")
    for axis in axes: axis.grid(alpha=.2); axis.legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "maximum_recomposition_error": maximum_error, "close_feature_layer_counts": close_counts}, indent=2))
    print(REPORT)


if __name__ == "__main__": main()
