#!/usr/bin/env python3
"""Trace finite layer-1 channel ablations into layer-2/3 product channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_layer1_channels import freeze, positional_product_mean, rows129
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, compare_logits, load_full_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--mean-start", type=int, default=12000)
    parser.add_argument("--mean-rows", type=int, default=400)
    parser.add_argument("--top-targets", type=int, default=64)
    parser.add_argument("--restore-targets", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--qkov", type=Path,
                        default=Path("artifacts/bilin18_layer1_deep/iteration3_qkov/results.json"))
    parser.add_argument("--atlas", type=Path,
                        default=Path("artifacts/bilin18_full_mlp_atlas"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/bilin18_layer1_deep/iteration3_propagation.json"))
    return parser.parse_args()


@torch.inference_mode()
def run(model, ids, destination, source_unit, source_value=None, restore=None):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0, value_bus = x, None
    captured = {}
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1 and source_value is not None:
            product = product.clone()
            product[0, destination, source_unit] = source_value
        if layer in (2, 3):
            captured[layer] = product[0, destination].float().clone()
        if restore is not None and layer in restore:
            indices, values = restore[layer]
            product = product.clone()
            product[0, destination, indices] = values.to(product.dtype)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return 30 * torch.tanh(raw / 30), captured


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(first, second):
    if np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(first), rankdata(second))[0, 1])


def effective_support(values):
    values = np.abs(np.asarray(values, dtype=float))
    return float(values.sum() ** 2 / max(np.square(values).sum(), 1e-30))


def subspace_summary(discovery, transfer):
    """Fit a propagation subspace on selection sites and test held-out sites."""
    discovery = np.asarray(discovery, dtype=np.float64)
    transfer = np.asarray(transfer, dtype=np.float64)
    # Centering would erase a reusable common shift; the object of interest is
    # the span of causal changes themselves, so use the uncentered SVD.
    _, singular, vh = np.linalg.svd(discovery, full_matrices=False)
    energy = singular ** 2
    total = max(float(energy.sum()), 1e-30)
    result = {
        "selection_singular_values": singular.tolist(),
        "selection_effective_rank": float(total ** 2 / max(float(np.square(energy).sum()), 1e-30)),
        "selection_variance_fraction": {},
        "transfer_projection_fraction_mean": {},
        "transfer_projection_fraction_median": {},
        "isotropic_expected_fraction": {},
    }
    denominator = np.square(transfer).sum(1).clip(1e-30)
    for k in (1, 2, 4, 8, 16):
        use = min(k, len(singular))
        basis = vh[:use]
        projected = np.square(transfer @ basis.T).sum(1) / denominator
        result["selection_variance_fraction"][str(k)] = float(energy[:use].sum() / total)
        result["transfer_projection_fraction_mean"][str(k)] = float(projected.mean())
        result["transfer_projection_fraction_median"][str(k)] = float(np.median(projected))
        result["isotropic_expected_fraction"][str(k)] = use / transfer.shape[1]
    return result


def source_specific_summary(grouped):
    """Does a source neuron's discovery span predict its own transfer effects?"""
    sources = sorted(grouped["selection"])
    bases = {}
    for source in sources:
        matrix = np.asarray(grouped["selection"][source], dtype=np.float64)
        _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = max(1, int((singular > singular[0] * 1e-8).sum()))
        bases[source] = vh[:rank]
    rows = []
    for source in sources:
        transfer = np.asarray(grouped["transfer"][source], dtype=np.float64)
        denominator = np.square(transfer).sum(1).clip(1e-30)
        matched = np.square(transfer @ bases[source].T).sum(1) / denominator
        mismatched = []
        for other in sources:
            if other == source:
                continue
            mismatched.append(np.square(transfer @ bases[other].T).sum(1) / denominator)
        mismatch = np.stack(mismatched)
        rows.append({
            "source_unit": source,
            "matched_projection_mean": float(matched.mean()),
            "mismatched_projection_mean": float(mismatch.mean()),
            "matched_exceeds_mismatched_mean": bool(matched.mean() > mismatch.mean()),
        })
    return {
        "sources": rows,
        "matched_projection_mean": float(np.mean([x["matched_projection_mean"] for x in rows])),
        "mismatched_projection_mean": float(np.mean([x["mismatched_projection_mean"] for x in rows])),
        "matched_win_fraction": float(np.mean([
            x["matched_exceeds_mismatched_mean"] for x in rows])),
    }


def weight_scores(model, source_unit, target_layer):
    source = model.blocks[1].mlp.Down.weight[:, source_unit].float()
    target = model.blocks[target_layer].mlp
    left = target.Left.weight.float() @ source
    right = target.Right.weight.float() @ source
    return torch.sqrt(left.square() + right.square()).cpu().numpy()


def overlap(first, second, k):
    return len(set(np.argsort(first)[-k:]) & set(np.argsort(second)[-k:])) / k


def main():
    args = parse_args()
    qkov = json.loads(args.qkov.read_text())
    corpus = np.load(args.corpus, mmap_mode="r")
    model = load_full_model(args.device)
    freeze(model)
    # Reuse the exact positional-mean construction from iterations 1--2.
    holder = argparse.Namespace(**{
        "layer": 1, "sequence_length": args.sequence_length,
        "batch_size": 2, "device": args.device,
    })
    mean_rows = rows129(corpus, args.mean_start, args.mean_rows, args.sequence_length)
    mean = positional_product_mean(model, mean_rows, holder)
    std = {}
    for layer in (2, 3):
        atlas = json.loads((args.atlas / f"layer_{layer}.json").read_text())
        std[layer] = np.asarray([row["product_std"] for row in atlas["channels"]]).clip(1e-6)

    # Rank-1 positive and negative tail sites provide one discovery and one
    # transfer example per sign and source channel.
    sites = [site for site in qkov["sites"] if site["rank"] == 1]
    records = []
    aggregate = {}
    propagation_vectors = {2: {"selection": [], "transfer": []},
                           3: {"selection": [], "transfer": []}}
    propagation_by_source = {
        layer: {split: {} for split in ("selection", "transfer")}
        for layer in (2, 3)
    }
    for index, site in enumerate(sites):
        row = np.asarray(corpus[site["row"]], dtype=np.int64)
        position = site["position"]
        ids = torch.as_tensor(row[:position + 1], device=args.device)[None]
        target_id = int(row[position + 1])
        source_unit = site["unit"]
        native_logits, native_products = run(model, ids, position, source_unit)
        source_value = mean[position, source_unit]
        changed_logits, changed_products = run(
            model, ids, position, source_unit, source_value=source_value)
        base_effect = compare_logits(native_logits[0], changed_logits[0], target_id)
        layer_rows = {}
        for layer in (2, 3):
            delta = (changed_products[layer] - native_products[layer]).cpu().numpy()
            standardized = np.abs(delta) / std[layer]
            signed_standardized = delta / std[layer]
            propagation_vectors[layer][site["split"]].append(signed_standardized)
            propagation_by_source[layer][site["split"]].setdefault(
                source_unit, []).append(signed_standardized)
            top = np.argsort(standardized)[::-1][:args.top_targets]
            weights = weight_scores(model, source_unit, layer)
            restore_indices = torch.as_tensor(
                top[:args.restore_targets].copy(), device=args.device)
            restored_logits, _ = run(
                model, ids, position, source_unit, source_value=source_value,
                restore={layer: (restore_indices, native_products[layer][restore_indices])},
            )
            full_delta = changed_logits[0] - native_logits[0]
            residual_delta = restored_logits[0] - native_logits[0]
            recovered = 1 - float(residual_delta.norm() / full_delta.norm().clamp_min(1e-12))
            layer_rows[str(layer)] = {
                "standardized_delta_effective_support": effective_support(standardized),
                "weight_to_finite_spearman": spearman(weights, standardized),
                "weight_finite_top64_overlap": overlap(weights, standardized, 64),
                "top_targets": [{
                    "unit": int(unit),
                    "product_delta": float(delta[unit]),
                    "standardized_abs_delta": float(standardized[unit]),
                    "direct_weight_score": float(weights[unit]),
                } for unit in top],
                "restore_top_k": args.restore_targets,
                "logit_l2_recovered_fraction": recovered,
                "restored_kl_from_native": compare_logits(
                    native_logits[0], restored_logits[0], target_id)["kl_base_to_changed"],
            }
            key = (site["split"], source_unit, layer)
            aggregate.setdefault(key, []).append(standardized)
        records.append({
            **site,
            "target_id": target_id,
            "source_replacement": float(source_value),
            "source_delta": float(source_value - site["selected_product"]),
            "full_effect": base_effect,
            "layers": layer_rows,
        })
        print(f"propagation {index + 1}/{len(sites)}", flush=True)

    summaries = []
    by_source_layer = {}
    for (split, source, layer), arrays in aggregate.items():
        mean_effect = np.mean(arrays, axis=0)
        by_source_layer[(split, source, layer)] = mean_effect
        summaries.append({
            "split": split,
            "source_unit": source,
            "target_layer": layer,
            "sites": len(arrays),
            "mean_standardized_delta_effective_support": effective_support(mean_effect),
            "top_targets": [int(x) for x in np.argsort(mean_effect)[::-1][:64]],
        })
    transfer = []
    sources = sorted({site["unit"] for site in sites})
    for source in sources:
        for layer in (2, 3):
            selection = by_source_layer[("selection", source, layer)]
            heldout = by_source_layer[("transfer", source, layer)]
            transfer.append({
                "source_unit": source,
                "target_layer": layer,
                "selection_transfer_spearman": spearman(selection, heldout),
                "selection_transfer_top64_overlap": overlap(selection, heldout, 64),
            })
    result = {
        "intervention": "replace one layer-1 product at one token position by its positional FineWeb mean; recompute downstream layers",
        "sites": len(sites),
        "records": records,
        "aggregate": summaries,
        "transfer": transfer,
        "propagation_subspaces": {
            str(layer): subspace_summary(
                propagation_vectors[layer]["selection"],
                propagation_vectors[layer]["transfer"],
            ) for layer in (2, 3)
        },
        "source_specific_subspaces": {
            str(layer): source_specific_summary(propagation_by_source[layer])
            for layer in (2, 3)
        },
        "limitations": [
            "A single-channel intervention probes a local route inside a broadly redundant layer and need not have a large output effect.",
            "The direct weight score ignores RMSNorm Jacobians and all intervening nonlinear computation; its correlation with finite propagation is an empirical diagnostic, not an identity.",
            "Restoring downstream products can create a hybrid state, so recovered logit norm is evidence of mediation but not a unique path attribution.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "sites": len(records),
        "median_transfer_top64_overlap": float(np.median([
            row["selection_transfer_top64_overlap"] for row in transfer])),
    }, indent=2))


if __name__ == "__main__":
    main()
