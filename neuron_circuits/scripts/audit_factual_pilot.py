#!/usr/bin/env python3
"""Held-out explanations, rotation nulls, and sampled exact interventions."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import torch.nn.functional as F

from neuron_circuits.factual_data import ATTRIBUTES, VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.factual_model import FactualTransformer, Intervention


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("artifacts/factual_capacity_pilot"))
    parser.add_argument("--panel-size", type=int, default=8192)
    parser.add_argument("--causal-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()


def cache_for_indices(manifest, indices, template):
    entities, attributes = indices // ATTRIBUTES, indices % ATTRIBUTES
    return encode_prompts(manifest["ids"][entities], manifest["traits"][entities], attributes.astype(np.uint8), template)


@torch.inference_mode()
def collect(model, prompts, device, batch_size=2048):
    layers = [[] for _ in model.blocks]
    logits = []
    for start in range(0, len(prompts), batch_size):
        tokens = torch.from_numpy(prompts[start:start + batch_size].astype(np.int64)).to(device)
        with autocast(device): output, activations, _ = model(tokens, return_activations=True)
        logits.append(output.cpu())
        for index, activation in enumerate(activations): layers[index].append(activation.cpu())
    return torch.cat(logits), [torch.cat(parts) for parts in layers]


def candidate_features(manifest, indices, labels):
    entities, attributes = indices // ATTRIBUTES, indices % ATTRIBUTES
    traits = manifest["traits"][entities]
    ids = manifest["ids"][entities]
    structured = manifest["structured"][entities]
    rules = manifest["rules"][entities]
    current = labels.reshape(-1)[indices]
    features = {"attribute": attributes.astype(np.int64), "correct_answer": current.astype(np.int64)}
    for column in range(4): features[f"z{column + 1}"] = traits[:, column].astype(np.int64)
    for attribute in range(8):
        features[f"rule_a{attribute}"] = structured[:, attribute].astype(np.int64)
    features["branch_a6"] = (traits[:, 2] >= 4).astype(np.int64)
    features["branch_a7"] = (traits[:, 0] >= 4).astype(np.int64)
    for digit in range(ids.shape[1]): features[f"id_digit_{digit}"] = ids[:, digit].astype(np.int64)
    # Locked short conjunction vocabulary. Values are compact categorical IDs.
    originals = dict(features)
    for trait in range(4):
        features[f"attribute&z{trait + 1}"] = attributes * 8 + traits[:, trait]
        features[f"attribute&rule_a{trait}"] = attributes * 8 + structured[:, trait]
    features["attribute&correct_answer"] = attributes * 8 + current
    for left in range(4):
        for right in range(left + 1, 4): features[f"z{left + 1}&z{right + 1}"] = traits[:, left] * 8 + traits[:, right]
    for digit in range(ids.shape[1] - 1): features[f"id_digit_{digit}&{digit + 1}"] = ids[:, digit] * 8 + ids[:, digit + 1]
    # Three-variable rule candidates, fixed without inspecting activations.
    features["attribute&z1&z2"] = attributes * 64 + traits[:, 0] * 8 + traits[:, 1]
    features["attribute&z3&z4"] = attributes * 64 + traits[:, 2] * 8 + traits[:, 3]
    address = {"entity_identity": entities.astype(np.int64), "fact_pair_identity": indices.astype(np.int64)}
    return features, address


def categorical_transfer(train: torch.Tensor, test: torch.Tensor, values: np.ndarray):
    _, values = np.unique(values, return_inverse=True)
    values = torch.from_numpy(values).to(device=train.device, dtype=torch.long)
    categories = int(values.max()) + 1
    sums = torch.zeros(categories, train.shape[1], device=train.device).index_add_(0, values, train)
    counts = torch.bincount(values, minlength=categories).clamp_min(1).to(train.dtype)
    prediction = (sums / counts[:, None])[values]
    variance = ((test - test.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    r2 = 1 - ((test - prediction) ** 2).sum(0) / variance
    return {"r2": r2.cpu().tolist(), "median_r2": float(r2.median()), "mean_r2": float(r2.mean())}


def explanation_metrics(train: torch.Tensor, selection: torch.Tensor, test: torch.Tensor, features: dict[str, np.ndarray]):
    device = train.device
    selection_variance = ((selection - selection.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    test_variance = ((test - test.mean(0)) ** 2).sum(0).clamp_min(1e-12)
    selection_best_r2 = torch.full((train.shape[1],), -torch.inf, device=device)
    heldout_r2 = torch.full((train.shape[1],), -torch.inf, device=device)
    best_r2_feature = torch.zeros(train.shape[1], dtype=torch.long, device=device)
    selection_best_positive = torch.zeros(train.shape[1], device=device)
    selection_best_negative = torch.zeros(train.shape[1], device=device)
    heldout_positive = torch.zeros(train.shape[1], device=device)
    heldout_negative = torch.zeros(train.shape[1], device=device)
    best_positive_feature = torch.zeros(train.shape[1], dtype=torch.long, device=device)
    best_negative_feature = torch.zeros(train.shape[1], dtype=torch.long, device=device)
    threshold_high = torch.quantile(train, 0.9, dim=0)
    threshold_low = torch.quantile(train, 0.1, dim=0)
    selection_high, test_high = selection >= threshold_high, test >= threshold_high
    selection_low, test_low = selection <= threshold_low, test <= threshold_low
    names = list(features)
    for feature_index, name in enumerate(names):
        values = torch.from_numpy(features[name]).to(device=device, dtype=torch.long)
        categories = int(values.max()) + 1
        sums = torch.zeros(categories, train.shape[1], device=device).index_add_(0, values, train)
        counts = torch.bincount(values, minlength=categories).clamp_min(1).to(train.dtype)
        means = sums / counts[:, None]
        prediction = means[values]
        selection_r2 = 1 - ((selection - prediction) ** 2).sum(0) / selection_variance
        test_r2 = 1 - ((test - prediction) ** 2).sum(0) / test_variance
        update = selection_r2 > selection_best_r2
        selection_best_r2[update] = selection_r2[update]; heldout_r2[update] = test_r2[update]; best_r2_feature[update] = feature_index
        score_threshold = torch.quantile(prediction, 0.9, dim=0)
        predicted_high = prediction >= score_threshold
        selection_positive = 0.5 * ((predicted_high & selection_high).sum(0) / selection_high.sum(0).clamp_min(1) + ((~predicted_high) & (~selection_high)).sum(0) / (~selection_high).sum(0).clamp_min(1))
        test_positive = 0.5 * ((predicted_high & test_high).sum(0) / test_high.sum(0).clamp_min(1) + ((~predicted_high) & (~test_high)).sum(0) / (~test_high).sum(0).clamp_min(1))
        update = selection_positive > selection_best_positive
        selection_best_positive[update] = selection_positive[update]; heldout_positive[update] = test_positive[update]; best_positive_feature[update] = feature_index
        score_threshold = torch.quantile(prediction, 0.1, dim=0)
        predicted_low = prediction <= score_threshold
        selection_negative = 0.5 * ((predicted_low & selection_low).sum(0) / selection_low.sum(0).clamp_min(1) + ((~predicted_low) & (~selection_low)).sum(0) / (~selection_low).sum(0).clamp_min(1))
        test_negative = 0.5 * ((predicted_low & test_low).sum(0) / test_low.sum(0).clamp_min(1) + ((~predicted_low) & (~test_low)).sum(0) / (~test_low).sum(0).clamp_min(1))
        update = selection_negative > selection_best_negative
        selection_best_negative[update] = selection_negative[update]; heldout_negative[update] = test_negative[update]; best_negative_feature[update] = feature_index
    def names_for(index): return [names[value] for value in index.cpu().tolist()]
    return {
        "best_r2": heldout_r2.cpu().tolist(), "best_r2_feature": names_for(best_r2_feature),
        "positive_tail_balanced_accuracy": heldout_positive.cpu().tolist(),
        "positive_tail_feature": names_for(best_positive_feature),
        "negative_tail_balanced_accuracy": heldout_negative.cpu().tolist(),
        "negative_tail_feature": names_for(best_negative_feature),
    }


def aggregate_explanations(metrics, variance, write_norm):
    r2 = np.asarray(metrics["best_r2"]); pos = np.asarray(metrics["positive_tail_balanced_accuracy"]); neg = np.asarray(metrics["negative_tail_balanced_accuracy"])
    weight = np.asarray(variance) * np.asarray(write_norm) ** 2
    weight = weight / max(weight.sum(), 1e-12)
    return {
        "median_best_r2": float(np.median(r2)), "mean_best_r2": float(np.mean(r2)),
        "causal_mass_weighted_best_r2": float(np.sum(weight * r2)),
        "median_positive_tail_balanced_accuracy": float(np.median(pos)),
        "median_negative_tail_balanced_accuracy": float(np.median(neg)),
        "inactive_fraction": float(np.mean(np.asarray(variance) < 1e-8)),
    }


def margin_and_loss(logits, targets):
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    masked = logits.clone(); masked.scatter_(1, targets[:, None], -torch.inf)
    return correct - masked.max(1).values, F.cross_entropy(logits, targets, reduction="none")


@torch.inference_mode()
def intervention_effect(model, prompts, targets, intervention, device):
    changed_logits = []
    for start in range(0, len(prompts), 512):
        tokens = torch.from_numpy(prompts[start:start + 512].astype(np.int64)).to(device)
        with autocast(device): changed_logits.append(model(tokens, intervention=intervention).cpu())
    changed_logits = torch.cat(changed_logits)
    changed_margin, changed_loss = margin_and_loss(changed_logits, targets)
    return changed_margin, changed_loss


def support(values):
    values = np.asarray(values, dtype=float)
    return float(values.sum() ** 2 / max(np.square(values).sum(), 1e-20))


def causal_audit(model, prompts, targets, attributes, answers, selected, activations, rotations, device):
    tokens = torch.from_numpy(prompts.astype(np.int64)).to(device)
    with torch.inference_mode(), autocast(device): base = model(tokens).cpu()
    base_margin, base_loss = margin_and_loss(base, targets)
    rows, components = [], {"mlp_layers": [], "attention_heads": []}
    rng = np.random.default_rng(12345 + model.width)
    for layer, activation in enumerate(activations):
        means = activation.mean(0)
        write = model.blocks[layer].down.weight.detach().float().norm(dim=0).cpu().numpy()
        chosen = sorted(set(np.argsort(write)[-4:].tolist() + rng.choice(len(write), 4, replace=False).tolist()))
        for basis_name, basis in (("native", None), ("rotated", rotations[layer])):
            basis_write = write if basis is None else torch.linalg.vector_norm(model.blocks[layer].down.weight.detach().float().cpu() @ basis, dim=0).numpy()
            directions = chosen if basis is None else sorted(set(np.argsort(basis_write)[-4:].tolist() + rng.choice(len(write), 4, replace=False).tolist()))
            for channel in directions:
                if basis is None:
                    intervention = Intervention(layer=layer, channel=channel, replacement=float(means[channel]))
                else:
                    direction = basis[:, channel]
                    replacement = float(means @ direction)
                    intervention = Intervention(layer=layer, kind="mlp_direction", direction=direction.to(device), replacement=replacement)
                margin, loss = intervention_effect(model, prompts, targets, intervention, device)
                absolute = (margin - base_margin).abs().numpy()
                grouped = {}
                for name, values, count in (("attribute", attributes, 8), ("answer", answers, 8), ("selected", selected.astype(int), 2)):
                    effects = [float(absolute[values == group].mean()) if np.any(values == group) else 0.0 for group in range(count)]
                    grouped[name] = {"mean_absolute_effect": effects, "effective_support": support(effects)}
                rows.append({"basis": basis_name, "layer": layer, "channel": channel,
                             "mean_absolute_margin_effect": float(np.mean(absolute)),
                             "mean_loss_change": float((loss - base_loss).mean()), "groups": grouped})
        margin, loss = intervention_effect(model, prompts, targets, Intervention(layer=layer, channel=None, replacement=0.0), device)
        components["mlp_layers"].append({"layer": layer, "mean_absolute_margin_effect": float((margin-base_margin).abs().mean()), "mean_loss_change": float((loss-base_loss).mean())})
        for head in range(model.heads):
            margin, loss = intervention_effect(model, prompts, targets, Intervention(layer=layer, kind="attention_head", head=head), device)
            components["attention_heads"].append({"layer": layer, "head": head, "mean_absolute_margin_effect": float((margin-base_margin).abs().mean()), "mean_loss_change": float((loss-base_loss).mean())})
    return {"sampled_directions": rows, "components": components}


def main():
    args = parse_args(); manifest = load_manifest(args.root / "manifest.npz")
    runs = json.loads((args.root / "pilot_results.json").read_text())["runs"]
    for run in runs:
        destination = args.root / f"rho_{run['rho']:g}" / f"width_{run['width']}"
        if (destination / "audit.json").exists() and not args.force:
            print(f"SKIP {destination}", flush=True); continue
        labels, selected, changed = labels_for_rho(manifest, run["rho"])
        train_indices = np.flatnonzero(manifest["train_pairs"].reshape(-1))
        rng = np.random.default_rng(7000 + run["width"] + round(run["rho"] * 100))
        panel = rng.choice(train_indices, min(args.panel_size, len(train_indices)), replace=False)
        targets = labels.reshape(-1)[panel]
        features, address = candidate_features(manifest, panel, labels)
        checkpoint = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = FactualTransformer(VOCAB.size, run["width"], run["heads"]).to(args.device)
        model.load_state_dict(checkpoint["model"]); model.eval()
        _, discovery = collect(model, cache_for_indices(manifest, panel, 0), args.device)
        _, selection = collect(model, cache_for_indices(manifest, panel, 1), args.device)
        _, transfer = collect(model, cache_for_indices(manifest, panel, 6), args.device)
        layers, rotations = [], []
        for layer, (train_activation, selection_activation, test_activation) in enumerate(zip(discovery, selection, transfer)):
            generator = torch.Generator().manual_seed(8100 + layer + run["width"])
            rotation = torch.linalg.qr(torch.randn(model.channels, model.channels, generator=generator)).Q
            rotations.append(rotation)
            native = explanation_metrics(train_activation.to(args.device), selection_activation.to(args.device), test_activation.to(args.device), features)
            rotated = explanation_metrics((train_activation @ rotation).to(args.device), (selection_activation @ rotation).to(args.device), (test_activation @ rotation).to(args.device), features)
            variance = test_activation.var(0).numpy()
            write = model.blocks[layer].down.weight.detach().float().norm(dim=0).cpu().numpy()
            rotated_write = torch.linalg.vector_norm(model.blocks[layer].down.weight.detach().float().cpu() @ rotation, dim=0).numpy()
            layers.append({
                "layer": layer, "native": native, "rotated": rotated,
                "native_summary": aggregate_explanations(native, variance, write),
                "rotated_summary": aggregate_explanations(rotated, (test_activation @ rotation).var(0).numpy(), rotated_write),
                "address_transfer": {
                    "native_entity": categorical_transfer(train_activation.to(args.device), test_activation.to(args.device), address["entity_identity"]),
                    "native_fact_pair": categorical_transfer(train_activation.to(args.device), test_activation.to(args.device), address["fact_pair_identity"]),
                    "rotated_entity": categorical_transfer((train_activation @ rotation).to(args.device), (test_activation @ rotation).to(args.device), address["entity_identity"]),
                    "rotated_fact_pair": categorical_transfer((train_activation @ rotation).to(args.device), (test_activation @ rotation).to(args.device), address["fact_pair_identity"]),
                },
                "activation_quantiles": {str(q): torch.quantile(test_activation, q, dim=0).tolist() for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
                "activation_variance": variance.tolist(), "write_norm": write.tolist(),
            })
        causal_panel = panel[:min(args.causal_size, len(panel))]
        _, causal_activations = collect(model, cache_for_indices(manifest, causal_panel, 0), args.device)
        causal = causal_audit(
            model, cache_for_indices(manifest, causal_panel, 0),
            torch.from_numpy(labels.reshape(-1)[causal_panel].astype(np.int64)),
            causal_panel % ATTRIBUTES, labels.reshape(-1)[causal_panel], selected.reshape(-1)[causal_panel],
            causal_activations, rotations, args.device,
        )
        output = {"panel_size": len(panel), "causal_panel_size": len(causal_panel), "candidate_features": list(features), "layers": layers, "causal": causal}
        (destination / "audit.json").write_text(json.dumps(output), encoding="utf-8")
        print(f"DONE audit rho={run['rho']:g} width={run['width']}", flush=True)


if __name__ == "__main__": main()
