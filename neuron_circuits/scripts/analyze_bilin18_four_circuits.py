#!/usr/bin/env python3
"""Fit and causally validate propagation subspaces for four Bilin18 circuits."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_four_circuit_subspaces import CIRCUITS, UNITS, token_flags
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, compare_logits, load_full_model


ROOT = Path("artifacts/bilin18_four_circuits")
LAYERS = (2, 3)
KS = (1, 2, 4, 8, 16, 32)


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2
        start = stop
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def fit_basis(matrix, k):
    matrix = np.asarray(matrix, dtype=np.float32)
    gram = matrix @ matrix.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    keep = min(k, int((values > max(values[0], 1e-20) * 1e-8).sum()))
    basis = (vectors[:, :keep].T @ matrix) / np.sqrt(values[:keep, None].clip(1e-20))
    return basis.astype(np.float32), values


def projection_fractions(matrix, basis):
    matrix = np.asarray(matrix, dtype=np.float32)
    projected = matrix @ basis.T
    return np.square(projected).sum(1) / np.square(matrix).sum(1).clip(1e-20)


def cosine(first, second):
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / max(denominator, 1e-20))


def build_indices(manifest):
    by = {}
    for index, site in enumerate(manifest["sites"]):
        by.setdefault((site["split"], site["source_unit"]), []).append(index)
    return by


def semantic_analysis(arrays, manifest):
    unit_order = manifest["unit_order"]
    semantic_order = manifest["semantic_order"]
    train_x = arrays["fineweb_train_features"].astype(np.float32)
    train_y = arrays["fineweb_train_labels"]
    result = {"panels": {}, "decoders": {}}
    decoders = {}
    for concept_index, (name, units) in enumerate(CIRCUITS.items()):
        columns = [unit_order.index(unit) for unit in units]
        features = train_x[:, columns]
        mean, std = features.mean(0), features.std(0).clip(1e-5)
        z = (features - mean) / std
        labels = train_y[:, concept_index].astype(bool)
        weight = z[labels].mean(0) - z[~labels].mean(0)
        decoders[name] = (columns, mean, std, weight)
        result["decoders"][name] = {
            "units": units,
            "weight": weight.tolist(),
            "train_prevalence": float(labels.mean()),
        }
    for panel in ("fineweb_train", "fineweb_test", "wikitext103", "tinystories"):
        features = arrays[f"{panel}_features"].astype(np.float32)
        labels = arrays[f"{panel}_labels"]
        panel_result = {
            "prevalence": {semantic_order[i]: float(labels[:, i].mean())
                           for i in range(len(semantic_order))},
            "decoder_auc_matrix": {},
            "individual_auc": {},
        }
        for name, (columns, mean, std, weight) in decoders.items():
            score = ((features[:, columns] - mean) / std) @ weight
            panel_result["decoder_auc_matrix"][name] = {
                semantic_order[i]: auc(labels[:, i], score)
                for i in range(len(semantic_order))
            }
        for column, unit in enumerate(unit_order):
            panel_result["individual_auc"][str(unit)] = {
                semantic_order[i]: auc(labels[:, i], features[:, column])
                for i in range(len(semantic_order))
            }
        result["panels"][panel] = panel_result
    return result


def subspace_analysis(propagation, manifest):
    indices = build_indices(manifest)
    unit_to_circuit = {unit: name for name, units in CIRCUITS.items() for unit in units}
    atlas_std = {}
    for layer in LAYERS:
        atlas = json.loads(Path(f"artifacts/bilin18_full_mlp_atlas/layer_{layer}.json").read_text())
        atlas_std[layer] = np.asarray([row["product_std"] for row in atlas["channels"]],
                                     dtype=np.float32).clip(1e-5)
    standardized = {
        layer: propagation[f"layer{layer}"].astype(np.float32) / atlas_std[layer][None]
        for layer in LAYERS
    }
    bases = {}
    result = {"per_source": [], "leave_source_out": [], "global_leave_source_out": []}
    for layer in LAYERS:
        values = standardized[layer]
        for source in UNITS:
            discovery = values[indices[("discovery", source)]]
            transfer = values[indices[("transfer", source)]]
            row = {"layer": layer, "source_unit": source,
                   "circuit": unit_to_circuit[source], "k": {}}
            for k in KS:
                basis, spectrum = fit_basis(discovery, k)
                bases[("source", layer, source, k)] = basis
                fraction = projection_fractions(transfer, basis)
                row["k"][str(k)] = {
                    "projection_mean": float(fraction.mean()),
                    "projection_median": float(np.median(fraction)),
                    "selection_energy_fraction": float(spectrum[:len(basis)].sum() /
                                                        spectrum.clip(0).sum()),
                }
            result["per_source"].append(row)

        for circuit, sources in CIRCUITS.items():
            other_circuits = [name for name in CIRCUITS if name != circuit]
            for heldout in sources:
                train_sources = [source for source in sources if source != heldout]
                discovery = np.concatenate([
                    values[indices[("discovery", source)]] for source in train_sources])
                transfer = values[indices[("transfer", heldout)]]
                row = {"layer": layer, "circuit": circuit,
                       "heldout_source": heldout, "k": {}}
                for k in KS:
                    basis, _ = fit_basis(discovery, k)
                    bases[("circuit_lso", layer, heldout, k)] = basis
                    matched = projection_fractions(transfer, basis)
                    mismatched_values = []
                    for other in other_circuits:
                        other_discovery = np.concatenate([
                            values[indices[("discovery", source)]]
                            for source in CIRCUITS[other][:3]])
                        other_basis, _ = fit_basis(other_discovery, k)
                        mismatched_values.append(projection_fractions(transfer, other_basis))
                    mismatch = np.stack(mismatched_values).mean(0)
                    row["k"][str(k)] = {
                        "matched_mean": float(matched.mean()),
                        "matched_median": float(np.median(matched)),
                        "mismatched_mean": float(mismatch.mean()),
                        "matched_site_win_fraction": float((matched > mismatch).mean()),
                    }
                result["leave_source_out"].append(row)

        for heldout in UNITS:
            discovery = np.concatenate([
                values[indices[("discovery", source)]]
                for source in UNITS if source != heldout])
            transfer = values[indices[("transfer", heldout)]]
            basis, _ = fit_basis(discovery, 32)
            fraction = projection_fractions(transfer, basis)
            result["global_leave_source_out"].append({
                "layer": layer, "heldout_source": heldout,
                "projection_mean": float(fraction.mean()),
                "projection_median": float(np.median(fraction)),
            })
    return result, bases, atlas_std


@torch.inference_mode()
def model_run(model, ids, position, replacements=None, target_override=None,
              capture_layer=2):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0, value_bus = x, None
    captured = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1 and replacements:
            product = product.clone()
            for unit, value in replacements.items():
                product[0, position, unit] = value
        if layer == capture_layer:
            captured = product[0, position].float().clone()
            if target_override is not None:
                product = product.clone()
                product[0, position] = target_override.to(product.dtype)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    raw = model.lm_head(F.rms_norm(x, (dimension,))[:, -1]).float()
    return (30 * torch.tanh(raw / 30))[0], captured


def delta_metrics(reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    return {
        "rms": float(candidate.square().mean().sqrt()),
        "cosine_to_full": float(F.cosine_similarity(candidate, reference, dim=0)),
        "norm_ratio_to_full": float(candidate.norm() / reference.norm().clamp_min(1e-12)),
    }


@torch.inference_mode()
def causal_subspace_test(model, corpus, propagation, manifest, bases, atlas_std, device):
    indices = build_indices(manifest)
    means = propagation["positional_mean"]
    unit_to_column = {unit: index for index, unit in enumerate(UNITS)}
    circuit_names = list(CIRCUITS)
    records = []
    for source in UNITS:
        circuit = next(name for name, units in CIRCUITS.items() if source in units)
        mismatch_circuit = circuit_names[(circuit_names.index(circuit) + 1) % len(circuit_names)]
        mismatch_train = np.concatenate([
            propagation["layer2"][indices[("discovery", unit)]].astype(np.float32) /
            atlas_std[2][None] for unit in CIRCUITS[mismatch_circuit][:3]])
        mismatch_basis, _ = fit_basis(mismatch_train, 16)
        source_basis = bases[("source", 2, source, 16)]
        circuit_basis = bases[("circuit_lso", 2, source, 16)]
        transfer_indices = indices[("transfer", source)]
        # Four moderate positive and four moderate negative sites per source.
        chosen = transfer_indices[:4] + transfer_indices[len(transfer_indices) // 2:
                                                        len(transfer_indices) // 2 + 4]
        for site_index in chosen:
            site = manifest["sites"][site_index]
            row = np.asarray(corpus[site["row"]], dtype=np.int64)
            position = site["position"]
            ids = torch.as_tensor(row[:position + 1], device=device)[None]
            target = int(row[position + 1])
            replacement = float(means[position, unit_to_column[source]])
            native, base_product = model_run(model, ids, position)
            changed, changed_product = model_run(
                model, ids, position, replacements={source: replacement})
            full_logit_delta = changed - native
            standardized_delta = (
                propagation["layer2"][site_index].astype(np.float32) / atlas_std[2]
            )
            variants = {
                "source_specific": source_basis,
                "circuit_leave_source_out": circuit_basis,
                "mismatched_circuit": mismatch_basis,
            }
            variant_rows = {}
            for name, basis in variants.items():
                projected_standardized = standardized_delta @ basis.T @ basis
                projected = torch.as_tensor(projected_standardized * atlas_std[2],
                                            device=device)
                restored, _ = model_run(
                    model, ids, position, replacements={source: replacement},
                    target_override=changed_product - projected)
                replay, _ = model_run(
                    model, ids, position, target_override=base_product + projected)
                remaining = restored - native
                replay_delta = replay - native
                variant_rows[name] = {
                    "projection_fraction": float(
                        np.square(projected_standardized).sum() /
                        max(np.square(standardized_delta).sum(), 1e-20)),
                    "restoration_recovered_norm_fraction": 1 - float(
                        remaining.norm() / full_logit_delta.norm().clamp_min(1e-12)),
                    "restoration": delta_metrics(full_logit_delta, remaining),
                    "replay": delta_metrics(full_logit_delta, replay_delta),
                    "replay_kl_from_native": compare_logits(native, replay, target)["kl_base_to_changed"],
                }
            records.append({
                "source_unit": source, "circuit": circuit,
                "site_index": site_index, "sign": site["sign"],
                "full_effect": compare_logits(native, changed, target),
                "variants": variant_rows,
            })
        print(f"causal subspace patches: source {source}", flush=True)
    return records


def semantic_sites(corpus, flags, circuit_index, start=11000, rows=1000, count=64,
                   min_position=8, seed=20261019):
    rng = np.random.default_rng(seed + circuit_index)
    positive, negative = [], []
    raw = np.asarray(corpus[start:start + rows, :129], dtype=np.int64)
    for row_offset, tokens in enumerate(raw):
        for position in range(min_position, 128):
            bucket = positive if flags[tokens[position], circuit_index] else negative
            priority = float(rng.random())
            bucket.append((priority, start + row_offset, position))
    positive.sort(reverse=True)
    negative.sort(reverse=True)
    return positive[:count], negative[:count]


@torch.inference_mode()
def circuit_causal_test(model, corpus, tokenizer, flags, propagation, device):
    means = propagation["positional_mean"]
    unit_to_column = {unit: index for index, unit in enumerate(UNITS)}
    results = {}
    for circuit_index, (name, units) in enumerate(CIRCUITS.items()):
        sites = semantic_sites(corpus, flags, circuit_index)
        rows = {"positive": [], "negative": []}
        average_native_contribution = {
            "positive": None,
            "negative": None,
        }
        for label, selected in zip(("positive", "negative"), sites):
            for _, row_index, position in selected:
                row = np.asarray(corpus[row_index], dtype=np.int64)
                ids = torch.as_tensor(row[:position + 1], device=device)[None]
                target = int(row[position + 1])
                replacements = {
                    unit: float(means[position, unit_to_column[unit]]) for unit in units
                }
                native, _ = model_run(model, ids, position)
                changed, _ = model_run(model, ids, position, replacements=replacements)
                rows[label].append(compare_logits(native, changed, target))
                contribution = (native - changed).cpu()
                if average_native_contribution[label] is None:
                    average_native_contribution[label] = torch.zeros_like(contribution)
                average_native_contribution[label] += contribution
        result = {"units": units, "conditions": {}}
        for label in ("positive", "negative"):
            current = rows[label]
            contribution = average_native_contribution[label] / len(current)
            top = torch.topk(contribution, 12)
            bottom = torch.topk(contribution, 12, largest=False)
            result["conditions"][label] = {
                "sites": len(current),
                "mean_kl": float(np.mean([row["kl_base_to_changed"] for row in current])),
                "median_kl": float(np.median([row["kl_base_to_changed"] for row in current])),
                "mean_abs_nll_delta": float(np.mean([abs(row["nll_delta"]) for row in current])),
                "top1_change_rate": float(np.mean([row["top1_changed"] for row in current])),
                "normally_promoted_tokens": [(tokenizer.decode([int(i)]) if int(i) < tokenizer.vocab_size
                                                else f"<padded:{int(i)}>" ).replace("\n", "\\n")
                                               for i in top.indices],
                "normally_promoted_values": [float(x) for x in top.values],
                "normally_suppressed_tokens": [(tokenizer.decode([int(i)]) if int(i) < tokenizer.vocab_size
                                                  else f"<padded:{int(i)}>" ).replace("\n", "\\n")
                                                 for i in bottom.indices],
                "normally_suppressed_values": [float(x) for x in bottom.values],
            }
        results[name] = result
        print(f"circuit causal test: {name}", flush=True)
    # Isolation matrix: on each concept-positive panel, ablate every circuit
    # group.  A diagonal advantage is stronger evidence than positive-versus-
    # arbitrary-negative comparisons alone.
    cross = {}
    for concept_index, concept_name in enumerate(CIRCUITS):
        positive, _ = semantic_sites(corpus, flags, concept_index)
        cross[concept_name] = {}
        for ablated_name, ablated_units in CIRCUITS.items():
            current = []
            for _, row_index, position in positive:
                row = np.asarray(corpus[row_index], dtype=np.int64)
                ids = torch.as_tensor(row[:position + 1], device=device)[None]
                target = int(row[position + 1])
                replacements = {
                    unit: float(means[position, unit_to_column[unit]])
                    for unit in ablated_units
                }
                native, _ = model_run(model, ids, position)
                changed, _ = model_run(model, ids, position, replacements=replacements)
                current.append(compare_logits(native, changed, target))
            cross[concept_name][ablated_name] = {
                "mean_kl": float(np.mean([row["kl_base_to_changed"] for row in current])),
                "mean_abs_nll_delta": float(np.mean([abs(row["nll_delta"]) for row in current])),
                "top1_change_rate": float(np.mean([row["top1_changed"] for row in current])),
            }
        print(f"cross-circuit isolation: {concept_name}", flush=True)
    return {"matched_tests": results, "positive_cross_ablation": cross}


def aggregate_causal(records):
    result = {}
    for variant in ("source_specific", "circuit_leave_source_out", "mismatched_circuit"):
        rows = [row["variants"][variant] for row in records]
        result[variant] = {
            "projection_fraction_mean": float(np.mean([row["projection_fraction"] for row in rows])),
            "restoration_recovered_norm_fraction_mean": float(np.mean([
                row["restoration_recovered_norm_fraction"] for row in rows])),
            "restoration_recovered_norm_fraction_median": float(np.median([
                row["restoration_recovered_norm_fraction"] for row in rows])),
            "replay_cosine_mean": float(np.mean([row["replay"]["cosine_to_full"] for row in rows])),
            "replay_cosine_median": float(np.median([row["replay"]["cosine_to_full"] for row in rows])),
            "replay_norm_ratio_mean": float(np.mean([row["replay"]["norm_ratio_to_full"] for row in rows])),
        }
    return result


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    propagation = np.load(ROOT / "ordinary_propagation.npz")
    semantics = np.load(ROOT / "semantic_panels.npz")
    semantic_result = semantic_analysis(semantics, manifest)
    subspaces, bases, atlas_std = subspace_analysis(propagation, manifest)

    corpus = np.load(DEFAULT_CORPUS, mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = token_flags(tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_full_model(device)
    for module in [model.embedding, model.blocks, model.lm_head]:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    causal_records = causal_subspace_test(
        model, corpus, propagation, manifest, bases, atlas_std, device)
    circuits = circuit_causal_test(model, corpus, tokenizer, flags, propagation, device)
    result = {
        "semantic_validation": semantic_result,
        "subspaces": subspaces,
        "causal_subspace_summary": aggregate_causal(causal_records),
        "causal_subspace_records": causal_records,
        "circuit_causal_tests": circuits,
        "red_team": [
            "Semantic labels are deliberately simple token-level proxies and cannot exhaust a neuron's meaning.",
            "Projection uses the recipient's observed finite delta; it tests whether a learned span is causally relevant, not whether coefficients can yet be predicted from the prompt alone.",
            "Leave-source-out transfer tests shared implementation within a preselected circuit family, not generalization to arbitrary unstudied neurons.",
            "Group mean ablations create hybrid states and establish causal dependence, not a unique circuit decomposition.",
        ],
    }
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "output": str(ROOT / "analysis.json"),
        "causal_subspace_summary": result["causal_subspace_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
