#!/usr/bin/env python3
"""Falsification tests for circuits frozen by isolated-token/weight discovery."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc, token_flags
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path,
                        default=Path("artifacts/bilin18_weight_only_circuits/discovery.json"))
    parser.add_argument("--fineweb", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ood-dir", type=Path, default=Path("artifacts/bilin18_ood_corpora"))
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--calibration-start", type=int, default=10000)
    parser.add_argument("--calibration-rows", type=int, default=400)
    parser.add_argument("--panel-rows", type=int, default=600)
    parser.add_argument("--sites", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/bilin18_weight_only_circuits/validation.json"))
    return parser.parse_args()


@torch.inference_mode()
def layer1_selected(model, ids, unit_index):
    d = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (d,))
    x0, value_bus = x, None
    for layer in range(2):
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (d,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (d,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1:
            return product[..., unit_index].float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raise AssertionError


@torch.inference_mode()
def calibration(model, rows, units, args):
    unit_index = torch.as_tensor(units, device=args.device)
    sums = torch.zeros(args.sequence_length, len(units), device=args.device)
    squares = torch.zeros_like(sums)
    l2_sum = torch.zeros(4608, device=args.device)
    l2_square = torch.zeros_like(l2_sum)
    count = 0
    for start in range(0, len(rows), args.batch_size):
        raw = np.asarray(rows[start:start + args.batch_size, :args.sequence_length], dtype=np.int64)
        ids = torch.as_tensor(raw, device=args.device)
        selected = layer1_selected(model, ids, unit_index)
        sums += selected.sum(0)
        squares += selected.square().sum(0)
        # Re-run only the first three layers to obtain scale for downstream deltas.
        _, product2 = run_model(model, ids, capture_layer2=True, stop_after_layer2=True)
        l2_sum += product2.sum((0, 1))
        l2_square += product2.square().sum((0, 1))
        count += len(ids)
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(1e-6).sqrt()
    token_count = count * args.sequence_length
    l2_mean = l2_sum / token_count
    l2_std = (l2_square / token_count - l2_mean.square()).clamp_min(1e-6).sqrt()
    return mean, std, l2_std


@torch.inference_mode()
def run_model(model, ids, source_override=None, target_restore=None,
              capture_layer2=False, stop_after_layer2=False):
    """Run exact model; overrides apply at final sequence position."""
    d = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (d,))
    x0, value_bus = x, None
    captured2 = None
    position = ids.shape[1] - 1
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (d,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (d,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1 and source_override:
            product = product.clone()
            for unit, value in source_override.items():
                product[:, position, unit] = value
        if layer == 2:
            captured2 = product.float().clone()
            if target_restore is not None:
                target_ids, native_values = target_restore
                product = product.clone()
                product[:, position, target_ids] = native_values.to(product.dtype)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
        if stop_after_layer2 and layer == 2:
            return None, captured2
    raw = model.lm_head(F.rms_norm(x, (d,))[:, -1]).float()
    logits = 30 * torch.tanh(raw / 30)
    return logits, captured2 if capture_layer2 else None


def panel(model, rows, tokenizer_flags, units, circuit_rows, args):
    unit_index = torch.as_tensor(units, device=args.device)
    features, labels, token_ids = [], [], []
    for start in range(0, len(rows), args.batch_size):
        raw = np.asarray(rows[start:start + args.batch_size, :args.sequence_length], dtype=np.int64)
        ids = torch.as_tensor(raw, device=args.device)
        features.append(layer1_selected(model, ids, unit_index).cpu().to(torch.float16).numpy())
        labels.append(tokenizer_flags[raw])
        token_ids.append(raw)
    features = np.concatenate(features).reshape(-1, len(units)).astype(np.float32)
    labels = np.concatenate(labels).reshape(-1, tokenizer_flags.shape[1])
    token_ids = np.concatenate(token_ids).reshape(-1)
    scores = np.empty((len(features), len(circuit_rows)), dtype=np.float32)
    unit_to_col = {unit: index for index, unit in enumerate(units)}
    for ci, row in enumerate(circuit_rows):
        columns = [unit_to_col[unit] for unit in row["units"]]
        scores[:, ci] = ((features[:, columns] - np.asarray(row["decoder_mean"])) /
                          np.asarray(row["decoder_std"])) @ np.asarray(row["decoder_weight"])
    return features, labels, scores, token_ids


def kl_from_logits(reference, changed):
    log_p = F.log_softmax(reference, -1)
    p = log_p.exp()
    return float((p * (log_p - F.log_softmax(changed, -1))).sum())


def main():
    args = parse_args()
    started = time.time()
    discovery = json.loads(args.discovery.read_text())
    names = discovery["frozen_names"]
    by_name = {row["name"]: row for row in discovery["categories"]}
    circuit_rows = [by_name[name] for name in names]
    units = [unit for row in circuit_rows for unit in row["units"]]
    unit_to_col = {unit: index for index, unit in enumerate(units)}
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    all_flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    category_indices = [list(discovery_name["name"] for discovery_name in discovery["categories"]).index(name)
                        for name in names]
    frozen_flags = all_flags[:, category_indices]
    model = load_full_model(args.device)
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    fineweb = np.load(args.fineweb, mmap_mode="r")
    cal_rows = fineweb[args.calibration_start:args.calibration_start + args.calibration_rows,
                       :args.sequence_length]
    positional_mean, positional_std, layer2_std = calibration(model, cal_rows, units, args)
    panels = {
        "fineweb": fineweb[args.calibration_start + args.calibration_rows:
                            args.calibration_start + args.calibration_rows + args.panel_rows,
                            :args.sequence_length],
        "wikitext103": np.load(args.ood_dir / "wikitext103_1200x129.npy", mmap_mode="r")[:args.panel_rows, :args.sequence_length],
        "tinystories": np.load(args.ood_dir / "tinystories_1200x129.npy", mmap_mode="r")[:args.panel_rows, :args.sequence_length],
    }
    results = {"semantic": {}, "causal": {}, "path": {}}
    panel_cache = {}
    for panel_name, rows in panels.items():
        features, labels, scores, token_ids = panel(
            model, rows, frozen_flags, units, circuit_rows, args)
        panel_cache[panel_name] = (features, labels, scores, token_ids)
        matrix = {}
        for source_index, source_name in enumerate(names):
            matrix[source_name] = {
                target_name: auc(labels[:, target_index], scores[:, source_index])
                for target_index, target_name in enumerate(names)
            }
        results["semantic"][panel_name] = {
            "auc_matrix": matrix,
            "prevalence": {name: float(labels[:, i].mean()) for i, name in enumerate(names)},
            "tokens": int(len(labels)),
        }
        print("panel", panel_name, "diagonal",
              {name: round(matrix[name][name], 3) for name in names}, flush=True)

    # Sites are chosen from the natural FineWeb panel by the frozen score among
    # true positives, not by intervention outcomes.
    _, labels, scores, _ = panel_cache["fineweb"]
    fw_rows = panels["fineweb"]
    site_positions = []
    for ci, name in enumerate(names):
        flat = np.flatnonzero(labels[:, ci])
        flat = flat[(flat % args.sequence_length) >= 8]
        order = flat[np.argsort(scores[flat, ci])[::-1]][:args.sites]
        site_positions.append(order)

    cross_kl = np.zeros((len(names), len(names), args.sites), dtype=np.float64)
    path_records = {name: [] for name in names}
    individual_records = {name: [] for name in names}
    path_ks = (32, 128, 512, 2048)
    rng = np.random.default_rng(20261019)
    down = model.blocks[1].mlp.Down.weight.detach().float()
    left2 = model.blocks[2].mlp.Left.weight.detach().float()
    right2 = model.blocks[2].mlp.Right.weight.detach().float()
    path_rankings, random_rankings = {}, {}
    for row in circuit_rows:
        source = torch.as_tensor(row["units"], device=args.device)
        writes = down[:, source]
        score = ((left2 @ writes).square() + (right2 @ writes).square()).mean(1).sqrt()
        path_rankings[row["name"]] = torch.argsort(score, descending=True)
        # Fixed before interventions; unlike the oracle below, it sees no data.
        bottom = np.arange(len(score))
        random_rankings[row["name"]] = torch.as_tensor(
            rng.permutation(bottom), device=args.device)
    for semantic_index, flats in enumerate(site_positions):
        for site_number, flat in enumerate(flats):
            row_index, position = divmod(int(flat), args.sequence_length)
            ids = torch.as_tensor(np.asarray(fw_rows[row_index, :position + 1], dtype=np.int64),
                                  device=args.device)[None]
            base_logits, base_l2 = run_model(model, ids, capture_layer2=True)
            for ablation_index, ablation_name in enumerate(names):
                row = circuit_rows[ablation_index]
                replacement = {
                    unit: positional_mean[position, unit_to_col[unit]]
                    for unit in row["units"]
                }
                changed_logits, changed_l2 = run_model(
                    model, ids, source_override=replacement, capture_layer2=True)
                cross_kl[semantic_index, ablation_index, site_number] = kl_from_logits(
                    base_logits[0], changed_logits[0])
                if ablation_index != semantic_index:
                    continue
                full_delta = base_logits[0] - changed_logits[0]
                delta2 = (base_l2[0, position] - changed_l2[0, position]).abs() / layer2_std
                individual_deltas, individual_kls = [], {}
                for unit in row["units"]:
                    one_logits, _ = run_model(
                        model, ids,
                        source_override={unit: replacement[unit]}, capture_layer2=False)
                    one_delta = base_logits[0] - one_logits[0]
                    individual_deltas.append(one_delta)
                    individual_kls[str(unit)] = kl_from_logits(base_logits[0], one_logits[0])
                linear_sum = torch.stack(individual_deltas).sum(0)
                individual_records[ablation_name].append({
                    "unit_kl": individual_kls,
                    "group_kl": float(cross_kl[semantic_index, ablation_index, site_number]),
                    "sum_individual_kl": float(sum(individual_kls.values())),
                    "linear_sum_cosine": float(F.cosine_similarity(linear_sum, full_delta, dim=0)),
                    "linear_sum_norm_ratio": float(linear_sum.norm() /
                                                   full_delta.norm().clamp_min(1e-12)),
                })
                oracle_ranking = torch.argsort(delta2, descending=True)
                recovery_curves = {"weight": {}, "mismatched": {}, "random": {}, "oracle": {}}
                enrichment_curve = {}
                global_delta = float(delta2.mean())
                for k in path_ks:
                    mismatch_name = names[(semantic_index + 1) % len(names)]
                    for kind, ranking in (("weight", path_rankings[ablation_name]),
                                          ("mismatched", path_rankings[mismatch_name]),
                                          ("random", random_rankings[ablation_name]),
                                          ("oracle", oracle_ranking)):
                        targets = ranking[:k]
                        restored_logits, _ = run_model(
                            model, ids, source_override=replacement,
                            target_restore=(targets, base_l2[:, position, targets]))
                        residual = base_logits[0] - restored_logits[0]
                        recovery_curves[kind][str(k)] = float(
                            1 - residual.norm() / full_delta.norm().clamp_min(1e-12))
                    targets = path_rankings[ablation_name][:k]
                    enrichment_curve[str(k)] = float(delta2[targets].mean() /
                                                       max(global_delta, 1e-12))
                top_changed = torch.argsort(full_delta.abs(), descending=True)[:8]
                path_records[ablation_name].append({
                    "row": args.calibration_start + args.calibration_rows + row_index,
                    "position": position,
                    "token": tokenizer.decode([int(ids[0, -1])]),
                    "context": tokenizer.decode(ids[0, max(0, position - 40):position + 1].tolist()),
                    "weight_target_enrichment": enrichment_curve,
                    "recovery": recovery_curves,
                    "full_logit_delta_norm": float(full_delta.norm()),
                    "top_logit_changes": [
                        {"token": tokenizer.decode([int(token)]),
                         "token_id": int(token), "base_minus_ablated": float(full_delta[token])}
                        for token in top_changed
                    ],
                })
            print(f"causal {names[semantic_index]} {site_number + 1}/{len(flats)}", flush=True)

    for si, name in enumerate(names):
        matching = cross_kl[si, si]
        other = np.delete(cross_kl[si], si, axis=0).mean(0)
        results["causal"][name] = {
            "matching_kl_mean": float(matching.mean()),
            "other_kl_mean": float(other.mean()),
            "diagonal_ratio": float(matching.mean() / max(other.mean(), 1e-20)),
            "matching_wins_fraction": float((matching > other).mean()),
            "cross_kl_means": {names[ai]: float(cross_kl[si, ai].mean())
                               for ai in range(len(names))},
        }
        records = path_records[name]
        results["path"][name] = {
            "sites": records,
            "mean_weight_enrichment": {
                str(k): float(np.mean([r["weight_target_enrichment"][str(k)] for r in records]))
                for k in path_ks
            },
            "median_recovery": {
                kind: {str(k): float(np.median([r["recovery"][kind][str(k)] for r in records]))
                       for k in path_ks}
                for kind in ("weight", "mismatched", "random", "oracle")
            },
            "mean_recovery": {
                kind: {str(k): float(np.mean([r["recovery"][kind][str(k)] for r in records]))
                       for k in path_ks}
                for kind in ("weight", "mismatched", "random", "oracle")
            },
        }
        individual = individual_records[name]
        unit_ids = circuit_rows[si]["units"]
        results["causal"][name]["individual_neurons"] = {
            "mean_kl": {
                str(unit): float(np.mean([record["unit_kl"][str(unit)] for record in individual]))
                for unit in unit_ids
            },
            "mean_group_over_sum_individual_kl": float(np.mean([
                record["group_kl"] / max(record["sum_individual_kl"], 1e-20)
                for record in individual
            ])),
            "median_linear_sum_cosine": float(np.median([
                record["linear_sum_cosine"] for record in individual
            ])),
            "median_linear_sum_norm_ratio": float(np.median([
                record["linear_sum_norm_ratio"] for record in individual
            ])),
        }
    results["cross_kl_matrix"] = cross_kl.mean(2).tolist()
    results["names"] = names
    results["settings"] = vars(args) | {"discovery": str(args.discovery), "fineweb": str(args.fineweb),
                                        "ood_dir": str(args.ood_dir), "output": str(args.output)}
    results["elapsed_seconds"] = time.time() - started
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
