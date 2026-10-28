#!/usr/bin/env python3
"""Test whether sparse layer-1 activity limits interference on paths into MLP2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


ROOT = Path("artifacts/bilin18_activation_interference")
OOD = Path("artifacts/bilin18_ood_corpora")
KS = (1, 2, 4, 8, 16, 32)
EXTRA_ZS = (1.0, 2.0, 3.0)
ROUTES = ("direct", "attention", "both")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--float32", action="store_true",
                        help="Upcast the checkpoint to separate algebraic interactions from bf16 noise.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


@torch.inference_mode()
def first_three_products(model, ids):
    d = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (d,))
    x0, value_bus = x, None
    products = {}
    for layer in range(3):
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (d,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (d,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer in (1, 2):
            products[layer] = product.float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    return products


@torch.inference_mode()
def calibrate(model, rows, batch_size=4, length=128):
    width = model.blocks[1].mlp.Left.weight.shape[0]
    sums = {layer: torch.zeros(length, width, device="cuda") for layer in (1, 2)}
    squares = {layer: torch.zeros(length, width, device="cuda") for layer in (1, 2)}
    count = 0
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :length], dtype=np.int64)
        products = first_three_products(model, torch.as_tensor(raw, device="cuda"))
        for layer in (1, 2):
            sums[layer] += products[layer].sum(0)
            squares[layer] += products[layer].square().sum(0)
        count += len(raw)
    means = {layer: sums[layer] / count for layer in (1, 2)}
    stds = {layer: (squares[layer] / count - means[layer].square()).clamp_min(1e-6).sqrt()
            for layer in (1, 2)}
    return means, stds


@torch.inference_mode()
def sparsity_panel(model, rows, mean, std, down_norm, batch_size=4, length=128):
    threshold_sums = {str(z): 0.0 for z in (1, 2, 3)}
    top_sums = {str(k): 0.0 for k in (8, 32, 128, 512, 2048)}
    standardized_top_sums = {str(k): 0.0 for k in (8, 32, 128, 512, 2048)}
    participation_sum = 0.0
    standardized_participation_sum = 0.0
    usage_counts = {32: torch.zeros(mean.shape[1], device="cuda"),
                    128: torch.zeros(mean.shape[1], device="cuda")}
    sampled_top128 = []
    tokens = 0
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :length], dtype=np.int64)
        product = first_three_products(model, torch.as_tensor(raw, device="cuda"))[1]
        z = (product - mean[None]) / std[None]
        for threshold in (1, 2, 3):
            threshold_sums[str(threshold)] += float((z.abs() > threshold).sum())
        write_energy = ((product - mean[None]) * down_norm[None, None]).square()
        z_energy = z.square()
        total = write_energy.sum(-1).clamp_min(1e-20)
        z_total = z_energy.sum(-1).clamp_min(1e-20)
        participation_sum += float((total.square() / write_energy.square().sum(-1).clamp_min(1e-20)).sum())
        standardized_participation_sum += float(
            (z_total.square() / z_energy.square().sum(-1).clamp_min(1e-20)).sum())
        top_values, top_indices = torch.topk(write_energy, 2048, dim=-1)
        for k in (8, 32, 128, 512, 2048):
            top_sums[str(k)] += float(top_values[..., :k].sum(-1).div(total).sum())
            standardized_top_sums[str(k)] += float(
                torch.topk(z_energy, k, dim=-1).values.sum(-1).div(z_total).sum())
        for k in (32, 128):
            usage_counts[k] += torch.bincount(top_indices[..., :k].reshape(-1),
                                               minlength=mean.shape[1])
        if len(sampled_top128) < 2048:
            sampled_top128.extend(top_indices[..., :128].reshape(-1, 128).cpu().numpy())
        tokens += product.shape[0] * product.shape[1]
    width = mean.shape[1]
    usage = {}
    sample = np.asarray(sampled_top128[:2048])
    rng = np.random.default_rng(20261019)
    pairs = rng.integers(0, len(sample), size=(4096, 2))
    for k in (32, 128):
        counts = usage_counts[k].cpu().numpy()
        probability = counts / counts.sum()
        entropy_effective = float(np.exp(-(probability[probability > 0] *
                                           np.log(probability[probability > 0])).sum()))
        overlaps = []
        for first, second in pairs:
            intersection = len(np.intersect1d(sample[first, :k], sample[second, :k],
                                              assume_unique=True))
            overlaps.append(intersection / (2 * k - intersection))
        usage[str(k)] = {"effective_neurons_used": entropy_effective,
                         "random_pair_jaccard_median": float(np.median(overlaps)),
                         "random_pair_jaccard_mean": float(np.mean(overlaps))}
    return {
        "tokens": tokens,
        "fraction_abs_z_above": {key: value / (tokens * width)
                                 for key, value in threshold_sums.items()},
        "write_energy_topk_fraction": {key: value / tokens for key, value in top_sums.items()},
        "standardized_energy_topk_fraction": {
            key: value / tokens for key, value in standardized_top_sums.items()},
        "write_effective_participation": participation_sum / tokens,
        "standardized_effective_participation": standardized_participation_sum / tokens,
        "top_set_diversity": usage,
    }


@torch.inference_mode()
def layer1_state(model, ids):
    d = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (d,))
    x0, value_bus = x, None
    result = None
    for layer in range(2):
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (d,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (d,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1:
            result = {"pre1": pre, "product1": product, "x0": x0, "value_bus": value_bus}
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    result["x1"] = x
    return result


@torch.inference_mode()
def route_products(model, state, modified_product, position):
    block1, block2 = model.blocks[1], model.blocks[2]
    x_modified = state["pre1"] + block1.mlp.Down(modified_product) + block1.mlp.Down_bias
    mixed_base = block2.lambdas[0] * state["x1"] + block2.lambdas[1] * state["x0"]
    mixed_modified = block2.lambdas[0] * x_modified + block2.lambdas[1] * state["x0"]
    attention_base, _ = block2.attn(
        F.rms_norm(mixed_base, (model.config.n_embd,)), state["value_bus"])
    attention_modified, _ = block2.attn(
        F.rms_norm(mixed_modified, (model.config.n_embd,)), state["value_bus"])

    def product(mixed, attention):
        normalized = F.rms_norm(mixed + attention, (model.config.n_embd,))
        return (block2.mlp.Left(normalized) * block2.mlp.Right(normalized))[0, position].float()

    return {
        "base": product(mixed_base, attention_base),
        "direct": product(mixed_modified, attention_base),
        "attention": product(mixed_base, attention_modified),
        "both": product(mixed_modified, attention_modified),
    }


def interaction_metrics(joint, singles, base, scale):
    joint_delta = (joint - base) / scale
    linear = sum((single - base) / scale for single in singles)
    noise = joint_delta - linear
    joint_norm = float(joint_delta.norm())
    linear_norm = float(linear.norm())
    noise_norm = float(noise.norm())
    return {
        "joint_norm": joint_norm,
        "linear_norm": linear_norm,
        "noise_norm": noise_norm,
        "noise_to_joint": noise_norm / max(joint_norm, 1e-12),
        "signal_to_noise": linear_norm / max(noise_norm, 1e-12),
        "linear_joint_cosine": float(F.cosine_similarity(linear, joint_delta, dim=0)),
    }


def main():
    args = parse_args()
    started = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    model = load_full_model("cuda")
    if args.float32:
        model.embedding.float()
        model.blocks.float()
        model.lm_head.float()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    fineweb = np.load(DEFAULT_CORPUS, mmap_mode="r")
    wiki = np.load(OOD / "wikitext103_1200x129.npy", mmap_mode="r")
    tiny = np.load(OOD / "tinystories_1200x129.npy", mmap_mode="r")
    means, stds = calibrate(model, fineweb[10000:10400])
    down_norm = model.blocks[1].mlp.Down.weight.detach().float().norm(dim=0)
    panels = {
        "fineweb": fineweb[18000:18200],
        "wikitext103": wiki[600:800],
        "tinystories": tiny[600:800],
    }
    result = {"sparsity": {}, "interference_records": [], "settings": {
        "ks": list(KS), "extra_activation_z": list(EXTRA_ZS),
        "quiet_definition": "abs(z)<0.25",
        "noise_definition": "joint layer2 effect minus sum of singleton effects",
        "compute_dtype": "float32" if args.float32 else "bfloat16",
    }}
    for name, rows in panels.items():
        result["sparsity"][name] = sparsity_panel(
            model, rows, means[1], stds[1], down_norm)
        print("sparsity", name, result["sparsity"][name]["fraction_abs_z_above"], flush=True)

    site_panels = {
        "fineweb": fineweb[18500:18516],
        "wikitext103": wiki[800:816],
        "tinystories": tiny[800:816],
    }
    positions = (31, 63, 95, 127)
    for corpus_index, (corpus_name, rows) in enumerate(site_panels.items()):
        for row_index in range(16):
            position = positions[row_index % len(positions)]
            raw = np.asarray(rows[row_index, :position + 1], dtype=np.int64)
            ids = torch.as_tensor(raw, device="cuda")[None]
            state = layer1_state(model, ids)
            native = state["product1"][0, position].float()
            z = (native - means[1][position]) / stds[1][position]
            write_score = z.abs() * down_norm
            natural_order = torch.argsort(write_score, descending=True)[:max(KS)]
            quiet = torch.where(z.abs() < .25)[0].cpu().numpy()
            rng = np.random.default_rng(20261019 + corpus_index * 1000 + row_index)
            injected_order = torch.as_tensor(rng.choice(quiet, max(KS), replace=False), device="cuda")
            signs = torch.as_tensor(rng.choice([-1.0, 1.0], max(KS)), device="cuda")
            base_routes = route_products(model, state, state["product1"], position)

            natural_singletons = []
            for unit in natural_order:
                modified = state["product1"].clone()
                modified[0, position, unit] = means[1][position, unit].to(modified.dtype)
                natural_singletons.append(route_products(model, state, modified, position))
            extra_singletons = {}
            for amplitude in EXTRA_ZS:
                extra_singletons[amplitude] = []
                for column, unit in enumerate(injected_order):
                    modified = state["product1"].clone()
                    modified[0, position, unit] = (means[1][position, unit] +
                        signs[column] * amplitude * stds[1][position, unit]).to(modified.dtype)
                    extra_singletons[amplitude].append(
                        route_products(model, state, modified, position))

            experiments = [("natural_remove", None, natural_order, natural_singletons)]
            experiments.extend(("extra_on", amplitude, injected_order, extra_singletons[amplitude])
                               for amplitude in EXTRA_ZS)
            for mode, amplitude, order, singletons in experiments:
                for k in KS:
                    modified = state["product1"].clone()
                    selected = order[:k]
                    if mode == "natural_remove":
                        modified[0, position, selected] = means[1][position, selected].to(modified.dtype)
                    else:
                        modified[0, position, selected] = (means[1][position, selected] +
                            signs[:k] * amplitude * stds[1][position, selected]).to(modified.dtype)
                    joint = route_products(model, state, modified, position)
                    record = {"corpus": corpus_name, "row": row_index, "position": position,
                              "mode": mode, "amplitude_z": amplitude, "k": k, "routes": {}}
                    for route in ROUTES:
                        record["routes"][route] = interaction_metrics(
                            joint[route], [entry[route] for entry in singletons[:k]],
                            base_routes["base"], stds[2][position])
                    both_delta = (joint["both"] - base_routes["base"]) / stds[2][position]
                    direct_delta = (joint["direct"] - base_routes["base"]) / stds[2][position]
                    attention_delta = (joint["attention"] - base_routes["base"]) / stds[2][position]
                    route_cross = both_delta - direct_delta - attention_delta
                    record["direct_attention_cross_noise_to_both"] = float(
                        route_cross.norm() / both_delta.norm().clamp_min(1e-12))
                    record["direct_fraction_of_both_norm"] = float(
                        direct_delta.norm() / both_delta.norm().clamp_min(1e-12))
                    record["attention_fraction_of_both_norm"] = float(
                        attention_delta.norm() / both_delta.norm().clamp_min(1e-12))
                    result["interference_records"].append(record)
            print(f"interference {corpus_name} {row_index + 1}/16", flush=True)

    result["elapsed_seconds"] = time.time() - started
    output = args.output or ROOT / ("results_float32.json" if args.float32 else "results.json")
    output.write_text(json.dumps(result, indent=2))
    print(f"wrote {output} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
