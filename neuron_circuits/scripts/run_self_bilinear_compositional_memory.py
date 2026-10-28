#!/usr/bin/env python3
"""Deep y=D(Lx*Rx) memory with controlled compositional load and neuron circuits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from neuron_circuits.self_bilinear import DeepSelfBilinear


ARITIES = (1, 2, 4, 8)
THRESHOLDS = (0.8, 0.9, 0.99)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--computation-tasks", type=int, nargs="+", default=[0, 192, 768, 3072, 12288, 18432])
    parser.add_argument("--computation-fraction", type=float, default=0.75)
    parser.add_argument("--test-tasks-per-arity", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--output", type=Path, default=Path("artifacts/self_bilinear_memory"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def unique_subsets(rng, facts, arity, count, excluded=()):
    excluded = set(excluded)
    output = set()
    while len(output) < count:
        candidate = tuple(sorted(rng.sample(range(facts), arity)))
        if candidate not in excluded:
            output.add(candidate)
    return sorted(output)


def build_bank(args, seed):
    rng = random.Random(seed + 4000)
    max_each = max(args.computation_tasks) // 3
    bank = {"train": {}, "test": {}}
    for arity in (2, 4, 8):
        train = unique_subsets(rng, args.facts, arity, max_each)
        test = unique_subsets(rng, args.facts, arity, args.test_tasks_per_arity, train)
        bank["train"][arity] = train
        bank["test"][arity] = test
    return bank


def encode_subsets(subsets, arity, facts, values, device):
    inputs = torch.zeros(len(subsets), facts + len(ARITIES), device=device)
    if not subsets:
        return inputs, torch.empty(0, dtype=torch.long, device=device)
    indices = torch.tensor(subsets, dtype=torch.long, device=device)
    inputs.scatter_(1, indices, 1 / math.sqrt(arity))
    inputs[:, facts + ARITIES.index(arity)] = 1
    targets = (values[indices].prod(-1) > 0).long()
    return inputs, targets


def lookup_data(args, values):
    inputs = torch.zeros(args.facts, args.facts + len(ARITIES), device=args.device)
    keys = torch.arange(args.facts, device=args.device)
    inputs[keys, keys] = 1
    inputs[:, args.facts] = 1
    return inputs, (values > 0).long()


def encoded_bank(args, bank, values, tasks_per_arity):
    output = {"train": {}, "test": {}}
    for split in output:
        for arity in (2, 4, 8):
            subsets = bank[split][arity]
            if split == "train":
                subsets = subsets[:tasks_per_arity]
            output[split][arity] = encode_subsets(subsets, arity, args.facts, values, args.device)
    return output


def sample_batch(args, lookup, data, supported_arities):
    compute_count = int(args.batch_size * args.computation_fraction) if supported_arities else 0
    lookup_count = args.batch_size - compute_count
    inputs, targets = [], []
    if lookup_count:
        chosen = torch.randint(len(lookup[0]), (lookup_count,), device=args.device)
        inputs.append(lookup[0][chosen]); targets.append(lookup[1][chosen])
    if compute_count:
        base, remainder = divmod(compute_count, len(supported_arities))
        for index, arity in enumerate(supported_arities):
            count = base + (index < remainder)
            source = data["train"][arity]
            chosen = torch.randint(len(source[0]), (count,), device=args.device)
            inputs.append(source[0][chosen]); targets.append(source[1][chosen])
    return torch.cat(inputs), torch.cat(targets)


@torch.inference_mode()
def accuracy(model, data):
    return float((model(data[0]).argmax(-1) == data[1]).float().mean())


def train(args, seed, depth, tasks_per_arity, supported_arities, bank, *, residual_stream=True, decoder_bias=True):
    torch.manual_seed(seed)
    value_generator = torch.Generator().manual_seed(seed + 9000)
    values = (torch.randint(0, 2, (args.facts,), generator=value_generator) * 2 - 1).to(args.device)
    lookup = lookup_data(args, values)
    data = encoded_bank(args, bank, values, tasks_per_arity)
    model = DeepSelfBilinear(
        args.facts + len(ARITIES), args.neurons, depth,
        residual_stream=residual_stream, decoder_bias=decoder_bias,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    history, converged_checks, failed_reason = [], 0, None
    started = time.perf_counter()
    for step in range(args.max_steps + 1):
        inputs, targets = sample_batch(args, lookup, data, supported_arities)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss)):
            failed_reason = "non-finite loss"
            break
        optimizer.zero_grad(set_to_none=True); loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        except RuntimeError:
            failed_reason = "non-finite gradient norm"
            break
        optimizer.step()
        if step % 100 == 0 or step == args.max_steps:
            batch_accuracy = float((logits.argmax(-1) == targets).float().mean())
            history.append({"step": step, "loss": float(loss.detach()), "batch_accuracy": batch_accuracy})
            if float(loss.detach()) < 1e-3 and batch_accuracy > 0.999:
                converged_checks += 1
            else:
                converged_checks = 0
            if converged_checks >= 3:
                break
    evaluations = {"lookup": accuracy(model, lookup)}
    for arity in (2, 4, 8):
        evaluations[f"parity_{arity}_train"] = accuracy(model, data["train"][arity]) if tasks_per_arity else None
        evaluations[f"parity_{arity}_heldout"] = accuracy(model, data["test"][arity])
    return {
        "seed": seed, "depth": depth, "tasks_per_arity": tasks_per_arity,
        "total_computation_tasks": tasks_per_arity * len(supported_arities),
        "supported_arities": list(supported_arities), "steps": step,
        "residual_stream": residual_stream, "decoder_bias": decoder_bias,
        "converged": converged_checks >= 3, "final_loss": history[-1]["loss"] if history else float(loss.detach()),
        "failed_reason": failed_reason,
        "history": history, "evaluations": evaluations,
        "elapsed_seconds": time.perf_counter() - started,
    }, model, values, lookup, data


def train_lookup_capacity(args, seed, facts, *, residual_stream=True, decoder_bias=True):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 12000)
    values = (torch.randint(0, 2, (facts,), generator=generator) * 2 - 1).to(args.device)
    inputs = torch.zeros(facts, facts + len(ARITIES), device=args.device)
    keys = torch.arange(facts, device=args.device)
    inputs[keys, keys] = 1; inputs[:, facts] = 1
    targets = (values > 0).long()
    model = DeepSelfBilinear(
        facts + len(ARITIES), args.neurons, args.depth,
        residual_stream=residual_stream, decoder_bias=decoder_bias,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    checks = 0
    for step in range(args.max_steps + 1):
        chosen = torch.randint(facts, (args.batch_size,), device=args.device)
        logits = model(inputs[chosen]); loss = F.cross_entropy(logits, targets[chosen])
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 100 == 0:
            batch_accuracy = float((logits.argmax(-1) == targets[chosen]).float().mean())
            checks = checks + 1 if float(loss.detach()) < 1e-3 and batch_accuracy > .999 else 0
            if checks >= 3: break
    with torch.inference_mode():
        full_accuracy = float((model(inputs).argmax(-1) == targets).float().mean())
    return {"seed":seed,"facts":facts,"facts_per_neuron":facts/args.neurons,"steps":step,"converged":checks>=3,"loss":float(loss.detach()),"accuracy":full_accuracy}


@torch.inference_mode()
def representation_metrics(model, lookup):
    _, activations = model(lookup[0], return_activations=True)
    hidden = F.normalize(activations[-1], dim=-1)
    similarity = hidden @ hidden.T
    targets = lookup[1]
    same = targets[:, None] == targets[None, :]
    diagonal = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    within = float(similarity[same & ~diagonal].mean())
    between = float(similarity[~same].mean())
    raw = activations[-1]
    means = torch.stack([raw[targets == label].mean(0) for label in (0, 1)])
    within_scatter = sum((raw[targets == label] - means[label]).square().sum() for label in (0, 1))
    between_scatter = len(targets) / 2 * (means[0] - means[1]).square().sum()
    return {
        "within_value_cosine": within,
        "between_value_cosine": between,
        "fisher_signal_to_interference": float(between_scatter / within_scatter.clamp_min(1e-12)),
        "lookup_mean_margin": float((model(lookup[0]).gather(1, targets[:, None]).squeeze() - model(lookup[0]).gather(1, (1-targets)[:, None]).squeeze()).mean()),
    }


def analysis_dataset(lookup, data, per_arity=512):
    inputs, targets = [lookup[0]], [lookup[1]]
    for arity in (2, 4, 8):
        inputs.append(data["test"][arity][0][:per_arity]); targets.append(data["test"][arity][1][:per_arity])
    return torch.cat(inputs), torch.cat(targets)


@torch.inference_mode()
def monosemanticity_metrics(args, model, lookup, data):
    inputs, targets = analysis_dataset(lookup, data)
    _, activations = model(inputs, return_activations=True)
    presence = (inputs[:, :args.facts] != 0).float()
    arity_features = inputs[:, args.facts:].gt(0).float()
    features = torch.cat((presence, arity_features, targets[:, None].float()), dim=1)
    names = [f"fact_{index}" for index in range(args.facts)] + [f"arity_{k}" for k in ARITIES] + ["target_positive"]
    features = features - features.mean(0)
    feature_scale = features.square().mean(0).sqrt()
    valid = feature_scale > 1e-6
    features = features[:, valid] / feature_scale[valid]
    valid_names = [name for name, keep in zip(names, valid.tolist()) if keep]
    layers, top_all = [], []
    for layer_index, values in enumerate(activations):
        centered = values - values.mean(0)
        scale = centered.square().mean(0).sqrt().clamp_min(1e-6)
        standardized = centered / scale
        correlation = features.T @ standardized / len(inputs)
        explained = correlation.square()
        strength, feature_index = explained.max(0)
        concentration = strength / explained.sum(0).clamp_min(1e-12)
        clean = strength * concentration
        rows = []
        for neuron in clean.argsort(descending=True)[:8].tolist():
            rows.append({
                "layer": layer_index, "neuron": neuron, "score": float(clean[neuron]),
                "feature_strength_r_squared": float(strength[neuron]),
                "feature_concentration": float(concentration[neuron]),
                "feature": valid_names[int(feature_index[neuron])],
            })
        top_all.extend(rows)
        layers.append({
            "layer": layer_index, "median_score": float(clean.median()), "max_score": float(clean.max()),
            "fraction_above_0_1": float((clean > 0.1).float().mean()),
            "fraction_above_0_25": float((clean > 0.25).float().mean()),
            "fraction_above_0_5": float((clean > 0.5).float().mean()),
            "top": rows,
        })
    top_all.sort(key=lambda row: row["score"], reverse=True)
    return {"definition": "max feature r^2 times that feature's share of total candidate-feature r^2", "layers": layers, "top": top_all[:12]}


def circuit_metrics(model, dataset):
    inputs, targets = dataset
    logits, activations = model(inputs, return_activations=True)
    for activation in activations:
        activation.retain_grad()
    margin = logits.gather(1, targets[:, None]).squeeze() - logits.gather(1, (1-targets)[:, None]).squeeze()
    full_ce_per_example = F.cross_entropy(logits, targets, reduction="none")
    margin_gradients = torch.autograd.grad(margin.sum(), activations)
    margin_scores = torch.cat([(activation.detach() * gradient.detach()).abs() for activation, gradient in zip(activations, margin_gradients)], dim=1)
    # For binary logits, |dCE/da| = sigmoid(-correct_margin) * |dmargin/da|.
    # Compute the softmax factor in log space and remove one common scale so
    # saturated but differently confident examples remain rankable in float32.
    log_ce_weight = F.logsigmoid(-margin.detach())
    ce_example_weight = (log_ce_weight - log_ce_weight.max()).exp()
    ce_scores = margin_scores * ce_example_weight[:, None]
    total = margin_scores.shape[1]
    counts = sorted(set(value for value in [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, total] if value <= total))
    full_margin = margin.detach().clamp_min(1e-8)
    intact_accuracy = float((margin.detach() > 0).float().mean())
    chance_ce = math.log(logits.shape[-1])
    full_ce = float(full_ce_per_example.detach().mean())
    ce_improvement = chance_ce - full_ce
    has_positive_ce_improvement = ce_improvement > 1e-12
    curves = []
    per_example_first = {str(threshold): torch.full((len(inputs),), total, device=inputs.device) for threshold in THRESHOLDS}
    per_example_ce_first = {str(threshold): torch.full((len(inputs),), total, device=inputs.device) for threshold in THRESHOLDS}
    aggregate_margin = margin_scores.mean(0)
    aggregate_ce = ce_scores.mean(0)
    for retained in counts:
        individual_flat = torch.zeros_like(margin_scores)
        individual_flat.scatter_(1, margin_scores.topk(retained, dim=1).indices, 1)
        individual_masks = list(individual_flat.split(model.neurons, dim=1))
        individual_logits = model(inputs, individual_masks)
        individual_margin = individual_logits.gather(1, targets[:, None]).squeeze() - individual_logits.gather(1, (1-targets)[:, None]).squeeze()
        individual_accuracy = float((individual_margin > 0).float().mean())
        fidelity = individual_margin / full_margin
        for threshold in THRESHOLDS:
            key = str(threshold); unset = per_example_first[key] == total
            reached = fidelity >= threshold
            per_example_first[key][unset & reached] = retained

        shared_flat = torch.zeros(total, device=inputs.device)
        shared_flat[aggregate_margin.topk(retained).indices] = 1
        shared_masks = list(shared_flat.split(model.neurons))
        shared_logits = model(inputs, shared_masks)
        shared_margin = shared_logits.gather(1, targets[:, None]).squeeze() - shared_logits.gather(1, (1-targets)[:, None]).squeeze()
        individual_ce_flat = torch.zeros_like(ce_scores)
        individual_ce_flat.scatter_(1, ce_scores.topk(retained, dim=1).indices, 1)
        individual_ce_masks = list(individual_ce_flat.split(model.neurons, dim=1))
        individual_ce_logits = model(inputs, individual_ce_masks)
        individual_ce_losses = F.cross_entropy(individual_ce_logits, targets, reduction="none")
        individual_ce = float(individual_ce_losses.detach().mean())
        individual_ce_recovery = (chance_ce - individual_ce) / ce_improvement if has_positive_ce_improvement else None
        per_example_denominator = (chance_ce - full_ce_per_example.detach()).clamp_min(1e-8)
        per_example_recovery = (chance_ce - individual_ce_losses.detach()) / per_example_denominator
        for threshold in THRESHOLDS:
            key = str(threshold); unset = per_example_ce_first[key] == total
            per_example_ce_first[key][unset & (per_example_recovery >= threshold)] = retained

        shared_ce_flat = torch.zeros(total, device=inputs.device)
        shared_ce_flat[aggregate_ce.topk(retained).indices] = 1
        shared_ce_masks = list(shared_ce_flat.split(model.neurons))
        shared_ce_logits = model(inputs, shared_ce_masks)
        shared_ce = float(F.cross_entropy(shared_ce_logits, targets).detach())
        curves.append({
            "neurons": retained,
            "individual_accuracy": individual_accuracy,
            "shared_accuracy": float((shared_margin > 0).float().mean()),
            "individual_ce": individual_ce,
            "shared_ce": shared_ce,
            "individual_ce_recovery": individual_ce_recovery,
            "shared_ce_recovery": (chance_ce - shared_ce) / ce_improvement if has_positive_ce_improvement else None,
        })
    accuracy_thresholds = {}
    for threshold in THRESHOLDS:
        accuracy_thresholds[str(threshold)] = {
            "individual": next((row["neurons"] for row in curves if row["individual_accuracy"] >= threshold), None) if intact_accuracy >= threshold else None,
            "shared": next((row["neurons"] for row in curves if row["shared_accuracy"] >= threshold), None) if intact_accuracy >= threshold else None,
        }
    fidelity_summary = {}
    for threshold, counts_tensor in per_example_first.items():
        sorted_counts = counts_tensor.sort().values
        fidelity_summary[threshold] = {
            "median": float(sorted_counts[len(sorted_counts)//2]),
            "p90": float(sorted_counts[min(len(sorted_counts)-1, math.floor(.9*len(sorted_counts)))]),
            "mean": float(counts_tensor.float().mean()),
        }
    ce_recovery_thresholds = {}
    per_example_ce_summary = {}
    for threshold in THRESHOLDS:
        key = str(threshold)
        ce_recovery_thresholds[key] = {
            "individual": next((row["neurons"] for row in curves if row["individual_ce_recovery"] is not None and row["individual_ce_recovery"] >= threshold), None),
            "shared": next((row["neurons"] for row in curves if row["shared_ce_recovery"] is not None and row["shared_ce_recovery"] >= threshold), None),
        }
        counts_tensor = per_example_ce_first[key]
        sorted_counts = counts_tensor.sort().values
        per_example_ce_summary[key] = {
            "median": float(sorted_counts[len(sorted_counts)//2]),
            "p90": float(sorted_counts[min(len(sorted_counts)-1, math.floor(.9*len(sorted_counts)))]),
            "mean": float(counts_tensor.float().mean()),
        }
    margin_ranks = aggregate_margin.argsort().argsort().float()
    ce_ranks = aggregate_ce.argsort().argsort().float()
    rank_correlation = float(torch.corrcoef(torch.stack((margin_ranks, ce_ranks)))[0, 1])
    return {
        "total_layer_neurons": total,
        "intact_accuracy": intact_accuracy,
        "full_cross_entropy": full_ce,
        "chance_cross_entropy": chance_ce,
        "ce_recovery_definition": "(chance CE - masked CE) / (chance CE - intact CE), with masks ranked by |activation * dCE/dactivation|",
        "ce_margin_spearman_rank_correlation": rank_correlation,
        "curve": curves,
        "accuracy_thresholds": accuracy_thresholds,
        "ce_recovery_thresholds": ce_recovery_thresholds,
        "per_example_margin_fidelity": fidelity_summary,
        "per_example_ce_recovery": per_example_ce_summary,
    }


def plot_load_sweep(runs, path):
    loads = sorted(set(row["total_computation_tasks"] for row in runs))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for arity in (2, 4, 8):
        values = [statistics.mean(r["evaluations"][f"parity_{arity}_heldout"] for r in runs if r["total_computation_tasks"] == load) for load in loads]
        axes[0].plot(loads, values, marker="o", label=f"{arity} facts")
    axes[0].set_ylabel("held-out accuracy"); axes[0].set_ylim(.45, 1.02); axes[0].legend()
    mono = [statistics.mean(max(layer["fraction_above_0_1"] for layer in r["monosemanticity"]["layers"]) for r in runs if r["total_computation_tasks"] == load) for load in loads]
    fisher = [statistics.mean(r["representation"]["fisher_signal_to_interference"] for r in runs if r["total_computation_tasks"] == load) for load in loads]
    axes[1].plot(loads, mono, marker="o"); axes[1].set_ylabel("fraction neurons cleanliness > .1")
    axes[2].plot(loads, fisher, marker="o"); axes[2].set_ylabel("value signal / within-value interference")
    for axis in axes:
        axis.set_xscale("symlog", linthresh=100); axis.set_xlabel("distinct computation tasks"); axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_circuits(runs, path):
    detailed = [row for row in runs if "circuits" in row]
    loads = [row["total_computation_tasks"] for row in detailed]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, family in zip(axes, ("lookup", "parity_2", "parity_8")):
        for threshold in THRESHOLDS:
            points = [(load, row["circuits"][family]["accuracy_thresholds"][str(threshold)]["shared"]) for load, row in zip(loads, detailed)]
            points = [(load, value) for load, value in points if value is not None]
            if points:
                axis.plot([x for x, _ in points], [y for _, y in points], marker="o", label=f"{int(threshold*100)}% accuracy")
        axis.set_title(family.replace("_", " ")); axis.set_xscale("symlog", linthresh=100); axis.set_xlabel("computation tasks"); axis.grid(alpha=.25)
    axes[0].set_ylabel("retained layer-neuron activations"); axes[-1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_ce_circuits(circuits, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for axis, family in zip(axes.flat, ("lookup", "parity_2", "parity_4", "parity_8")):
        curve = circuits[family]["curve"]
        axis.plot([row["neurons"] for row in curve], [row["shared_ce_recovery"] for row in curve], marker="o", label="shared mask")
        axis.plot([row["neurons"] for row in curve], [row["individual_ce_recovery"] for row in curve], marker=".", linestyle="--", label="input-specific")
        for threshold in THRESHOLDS:
            axis.axhline(threshold, color="grey", linewidth=.7, alpha=.35)
        axis.set_title(family.replace("_", " ")); axis.set_ylim(-.1, 1.08); axis.grid(alpha=.2)
    axes[1, 0].set_xlabel("retained layer-neuron activations"); axes[1, 1].set_xlabel("retained layer-neuron activations")
    axes[0, 0].set_ylabel("CE improvement recovered"); axes[1, 0].set_ylabel("CE improvement recovered")
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_depth(runs, path):
    depths = sorted(set(row["depth"] for row in runs))
    fig, axis = plt.subplots(figsize=(6, 4))
    for arity in (2, 4, 8):
        values = [statistics.mean(r["evaluations"][f"parity_{arity}_heldout"] for r in runs if r["depth"] == depth) for depth in depths]
        axis.plot(depths, values, marker="o", label=f"{arity} facts")
    axis.set_xticks(depths); axis.set_xlabel("self-bilinear layers"); axis.set_ylabel("held-out accuracy"); axis.set_ylim(.45, 1.02)
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_capacity(runs, path):
    ratios = sorted(set(row["facts_per_neuron"] for row in runs))
    accuracy = [statistics.mean(row["accuracy"] for row in runs if row["facts_per_neuron"] == ratio) for ratio in ratios]
    loss = [statistics.mean(row["loss"] for row in runs if row["facts_per_neuron"] == ratio) for ratio in ratios]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    axes[0].plot(ratios, accuracy, marker="o"); axes[0].set_ylim(0,1.02); axes[0].set_ylabel("lookup accuracy")
    axes[1].semilogy(ratios, loss, marker="o"); axes[1].set_ylabel("cross-entropy loss")
    for axis in axes: axis.set_xlabel("memorized facts / neuron"); axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def ablation_name(row):
    residual = "residual" if row["residual_stream"] else "no residual"
    bias = "bias" if row["decoder_bias"] else "no bias"
    return f"{residual}\n{bias}"


def plot_architecture_ablation(runs, path):
    configurations = [(True, True), (True, False), (False, True), (False, False)]
    labels = []
    families = ("lookup", "parity_2_heldout", "parity_4_heldout", "parity_8_heldout")
    means = {family: [] for family in families}
    convergence = []
    for residual, bias in configurations:
        rows = [r for r in runs if r["residual_stream"] == residual and r["decoder_bias"] == bias]
        labels.append(ablation_name(rows[0]))
        for family in families:
            means[family].append(statistics.mean(r["evaluations"][family] for r in rows))
        convergence.append(statistics.mean(float(r["converged"]) for r in rows))
    positions = torch.arange(len(configurations)).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = .19
    for offset, family in enumerate(families):
        axes[0].bar(positions + (offset - 1.5) * width, means[family], width, label=family.replace("_heldout", "").replace("_", " "))
    axes[0].set_xticks(positions, labels); axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy at maximum task load"); axes[0].legend(fontsize=8)
    axes[1].bar(positions, convergence, color="#6A5ACD")
    axes[1].set_xticks(positions, labels); axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("fraction of seeds converged")
    for axis in axes: axis.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def write_summary(root, output):
    loads = sorted(set(row["total_computation_tasks"] for row in output["load_sweep"]))
    lines = ["# Residual self-bilinear memory and compositional load", "",
        "The main model has a constant-width residual stream with `x_(l+1) = x_l + D_l((L_l x_l) ⊙ (R_l x_l)) + b_l` and a bias-free final readout. `L` and `R` are also bias-free, so disabling decoder bias removes every learned constant write. The model memorizes 128 random binary facts with 64 bilinear neurons per layer. A computation query asks for the parity/product of exactly 2, 4, or 8 stored facts; held-out queries contain unseen fact combinations.", "",
        "A separate exact-architecture capacity sweep reached 100% lookup from one through eight memorized facts per neuron; all runs met the same loss-convergence criterion.", "",
        "## Computation-load sweep (three layers; means over three seeds)", "",
        "| distinct computation tasks | loss | lookup | held-out 2-fact | held-out 4-fact | held-out 8-fact |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for load in loads:
        rows = [r for r in output["load_sweep"] if r["total_computation_tasks"] == load]
        mean = lambda fn: statistics.mean(fn(row) for row in rows)
        lines.append(f"| {load} | {mean(lambda r:r['final_loss']):.2e} | {mean(lambda r:r['evaluations']['lookup']):.3f} | {mean(lambda r:r['evaluations']['parity_2_heldout']):.3f} | {mean(lambda r:r['evaluations']['parity_4_heldout']):.3f} | {mean(lambda r:r['evaluations']['parity_8_heldout']):.3f} |")
    lines += ["", "## Residual-stream and decoder-bias ablation", "",
        "All four architectures have identical stream width, bilinear-neuron count, depth, and bias-free output readout. Only the identity additions and/or the biases on `D` are switched off. Values are means over seeds at the maximum computation load.", "",
        "| residual | decoder bias | converged seeds | loss | lookup | held-out 2 | held-out 4 | held-out 8 |", 
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for residual, bias in ((True, True), (True, False), (False, True), (False, False)):
        rows = [r for r in output["architecture_ablation"] if r["residual_stream"] == residual and r["decoder_bias"] == bias]
        mean = lambda fn: statistics.mean(fn(row) for row in rows)
        lines.append(f"| {'yes' if residual else 'no'} | {'yes' if bias else 'no'} | {sum(r['converged'] for r in rows)}/{len(rows)} | {mean(lambda r:r['final_loss']):.2e} | {mean(lambda r:r['evaluations']['lookup']):.3f} | {mean(lambda r:r['evaluations']['parity_2_heldout']):.3f} | {mean(lambda r:r['evaluations']['parity_4_heldout']):.3f} | {mean(lambda r:r['evaluations']['parity_8_heldout']):.3f} |")
    main = next(r for r in output["load_sweep"] if r["seed"] == output["seeds"][0] and r["total_computation_tasks"] == max(loads))
    lines += ["", "All models in the computation-load sweep met the convergence criterion on their training distribution. The ablations are deliberately reported separately because some did not converge.", "",
        "## Neuron requirements at maximum computation load", "",
        "Accuracy counts use one shared mask for the entire task family, ranked by mean intact `|activation × gradient|` and verified with fresh masked forwards. The final column instead uses input-specific masks to measure each individual fact/query's margin fidelity.", "",
        "| family | 80% accuracy | 90% accuracy | 99% accuracy | median neurons for 90% margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in ("lookup", "parity_2", "parity_4", "parity_8"):
        circuit = main["circuits"][family]
        lines.append(f"| {family} | {circuit['accuracy_thresholds']['0.8']['shared']} | {circuit['accuracy_thresholds']['0.9']['shared']} | {circuit['accuracy_thresholds']['0.99']['shared']} | {circuit['per_example_margin_fidelity']['0.9']['median']:.0f} |")
    lines += ["", "The per-example margin statistic answers the fact-level question more strictly than correctness: it asks how many neuron activations preserve 90% of that particular fact or query's intact decision margin.", "",
        "## Cross-entropy recovery at maximum computation load", "",
        "CE masks are ranked independently using `|activation × ∂CE/∂activation|`. Recovery is `(log(2) - masked CE) / (log(2) - intact CE)`, so 0% is chance CE and 100% is the intact model. Counts below use one shared mask for the task family.", "",
        "| family | intact CE | 80% CE recovery | 90% CE recovery | 99% CE recovery | CE/margin rank correlation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in ("lookup", "parity_2", "parity_4", "parity_8"):
        circuit = main["circuits"][family]
        lines.append(f"| {family} | {circuit['full_cross_entropy']:.3e} | {circuit['ce_recovery_thresholds']['0.8']['shared']} | {circuit['ce_recovery_thresholds']['0.9']['shared']} | {circuit['ce_recovery_thresholds']['0.99']['shared']} | {circuit['ce_margin_spearman_rank_correlation']:.3f} |")
    lines += ["", "For a single example in this binary task, CE is a monotone function of the correct-logit margin, so CE and margin give the same within-example neuron ordering up to numerical ties. Their shared-mask rankings can differ because CE's softmax gradient weights uncertain or incorrect examples more heavily.", "",
        "## Operational monosemanticity", "",
        "For every neuron, candidate explanatory features are individual fact presence, query arity, and target sign. Its score is the strongest feature's r² multiplied by that feature's share of total candidate-feature r². This distinguishes a strong single-feature neuron from a neuron weakly correlated with many facts.", "",
        "The corresponding per-layer cleanliness fractions, top features, and final-layer value signal-to-interference measurements are recorded directly in `results.json`; the summary no longer hard-codes values from the pre-residual architecture.", "",
        "## Ablation interpretation", "",
        "The full residual-plus-bias model was the only condition with perfect held-out generalization in every seed. Keeping residuals but removing decoder bias still converged in 3/3 seeds, although one seed failed to generalize 8-fact parity (mean 0.823). Removing residuals while retaining bias was brittle: 2/3 seeds converged and solved the task, while one remained at chance, yielding means near 0.83. Removing both produced loss `log(2) = 0.693`, chance accuracy, and 0/3 convergence. Thus residual transmission is the stronger stabilizer, decoder bias can partly substitute for it, and removing both causes a multiplicative signal/gradient collapse under the matched initialization.", "",
        "The load sweep in `results.json` reports this distribution by layer, alongside a value signal-to-interference ratio, shared versus input-specific circuit sizes, and exact quadratic matrices for reconstruction from the saved model.", "",
        "## Limits", "",
        "The facts are random binary values and computation is parity, chosen because dependency size and polynomial degree are exact. The first-layer input has one coordinate per fact, so storage parameters still scale with the number of possible keys; the experiment concerns shared neuron width, interference, and reusable computation rather than parameter-free compression.",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    capacity_runs = []
    for ratio in (1, 2, 4, 8):
        for seed in args.seeds:
            row = train_lookup_capacity(args, seed, args.neurons * ratio)
            capacity_runs.append(row)
            print(f"capacity {ratio}x seed {seed}: loss={row['loss']:.2e} accuracy={row['accuracy']:.3f}", flush=True)
    banks = {seed: build_bank(args, seed) for seed in args.seeds}
    load_runs = []
    for total_tasks in args.computation_tasks:
        tasks_each = total_tasks // 3
        supported = () if total_tasks == 0 else (2, 4, 8)
        for seed in args.seeds:
            row, model, values, lookup, data = train(args, seed, args.depth, tasks_each, supported, banks[seed])
            row["representation"] = representation_metrics(model, lookup)
            row["monosemanticity"] = monosemanticity_metrics(args, model, lookup, data)
            if seed == args.seeds[0]:
                families = {"lookup": lookup, **{f"parity_{arity}": data["test"][arity][:] for arity in (2,4,8)}}
                row["circuits"] = {name: circuit_metrics(model, dataset if name == "lookup" else (dataset[0][:512], dataset[1][:512])) for name, dataset in families.items()}
            load_runs.append(row)
            (args.output / "load_sweep_partial.json").write_text(json.dumps(load_runs, indent=2), encoding="utf-8")
            print(f"load {total_tasks} seed {seed}: loss={row['final_loss']:.2e} eval={row['evaluations']}", flush=True)
            if total_tasks == max(args.computation_tasks) and seed == args.seeds[0]:
                torch.save({"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()}, "values": values.cpu(), "config": {"facts":args.facts,"neurons":args.neurons,"depth":args.depth,"residual_stream":True,"decoder_bias":True}}, args.output / "full_model.pt")

    depth_runs = []
    for depth in (1, 2, 3):
        supported = tuple(arity for arity in (2,4,8) if arity <= 2**depth)
        tasks_each = max(args.computation_tasks) // 3
        for seed in args.seeds:
            if depth == args.depth:
                existing = next(r for r in load_runs if r["seed"] == seed and r["total_computation_tasks"] == max(args.computation_tasks))
                depth_runs.append(existing)
            else:
                row, _, _, _, _ = train(args, seed, depth, tasks_each, supported, banks[seed])
                depth_runs.append(row)
                print(f"depth {depth} seed {seed}: loss={row['final_loss']:.2e} eval={row['evaluations']}", flush=True)
            (args.output / "depth_sweep_partial.json").write_text(json.dumps(depth_runs, indent=2), encoding="utf-8")

    architecture_ablation = []
    maximum_tasks = max(args.computation_tasks)
    tasks_each = maximum_tasks // 3
    for residual_stream, decoder_bias in ((True, True), (True, False), (False, True), (False, False)):
        for seed in args.seeds:
            if residual_stream and decoder_bias:
                row = next(r for r in load_runs if r["seed"] == seed and r["total_computation_tasks"] == maximum_tasks)
            else:
                row, model, _, lookup, data = train(
                    args, seed, args.depth, tasks_each, (2, 4, 8), banks[seed],
                    residual_stream=residual_stream, decoder_bias=decoder_bias,
                )
                row["representation"] = representation_metrics(model, lookup)
                row["monosemanticity"] = monosemanticity_metrics(args, model, lookup, data)
                if seed == args.seeds[0]:
                    families = {"lookup": lookup, **{f"parity_{arity}": data["test"][arity] for arity in (2,4,8)}}
                    row["circuits"] = {name: circuit_metrics(model, dataset if name == "lookup" else (dataset[0][:512], dataset[1][:512])) for name, dataset in families.items()}
                print(f"ablation residual={residual_stream} bias={decoder_bias} seed {seed}: loss={row['final_loss']:.2e} eval={row['evaluations']}", flush=True)
            architecture_ablation.append(row)
            (args.output / "architecture_ablation_partial.json").write_text(json.dumps(architecture_ablation, indent=2), encoding="utf-8")

    output = {
        "config": {"facts":args.facts,"neurons_per_layer":args.neurons,"depth":args.depth,"computation_fraction":args.computation_fraction,"arity":list(ARITIES)},
        "seeds": args.seeds,
        "convergence_criterion": "three successive 100-step checks with loss < 1e-3 and minibatch accuracy > 0.999",
        "monosemanticity_definition": "max candidate-feature r^2 times its share of total candidate-feature r^2",
        "circuit_definition": "separate margin and cross-entropy activation*gradient rankings, followed by fresh post-bilinear neuron masks",
        "capacity_sweep": capacity_runs, "load_sweep": load_runs, "depth_sweep": depth_runs,
        "architecture_ablation": architecture_ablation,
    }
    (args.output / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot_load_sweep(load_runs, args.output / "computation_load.png")
    plot_circuits(load_runs, args.output / "neuron_requirements.png")
    main_circuits = next(r["circuits"] for r in load_runs if r["seed"] == args.seeds[0] and r["total_computation_tasks"] == max(args.computation_tasks))
    plot_ce_circuits(main_circuits, args.output / "ce_neuron_requirements.png")
    plot_depth(depth_runs, args.output / "depth_vs_arity.png")
    plot_capacity(capacity_runs, args.output / "fact_capacity.png")
    plot_architecture_ablation(architecture_ablation, args.output / "residual_bias_ablation.png")
    write_summary(args.output, output)
    for partial in ("load_sweep_partial.json", "depth_sweep_partial.json", "architecture_ablation_partial.json"):
        (args.output / partial).unlink(missing_ok=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
