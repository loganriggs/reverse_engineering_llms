#!/usr/bin/env python3
"""Disjoint scale sweep and downstream-aware ranking of spectral circuit modes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from redteam_bilin18_spectral_spikes import select_transfer_sites
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_atlas import ARRAYS as ATLAS_ARRAYS, OUTPUT as ATLAS_OUTPUT
from run_bilin18_spectral_spike_pilot import (
    SPARSE,
    TENSOR_ROOT,
    fisher_fidelity,
    load_feature_directions,
    safe_name,
)
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_downstream_ranking.json")
OUTPUT = TENSOR_ROOT / "spectral_downstream_ranking.json"
REPORT = TENSOR_ROOT / "SPECTRAL_DOWNSTREAM_RANKING.md"
FIGURE = TENSOR_ROOT / "spectral_downstream_ranking.png"


def greedy_order(contributions, target, weights, modes):
    """Order exact modes by greedy held-out weighted value reconstruction."""
    chosen = []
    remaining = set(range(modes))
    current = np.zeros_like(target)
    for _ in range(modes):
        best = min(remaining, key=lambda index: float(
            np.average(np.square(current + contributions[:, index] - target), weights=weights)))
        chosen.append(best)
        remaining.remove(best)
        current = current + contributions[:, best]
    return chosen


@torch.inference_mode()
def model_state_at_site(model, row, position):
    prefix = np.array(row[:position + 1], dtype=np.int64, copy=True)
    ids = torch.as_tensor(prefix, device="cuda")[None]
    x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
    product = model.blocks[1].mlp.Left(normalized) * model.blocks[1].mlp.Right(normalized)
    return x0, value_bus, pre, normalized, product


@torch.inference_mode()
def logits_with_coordinate(model, state, coefficients, desired, original):
    x0, value_bus, pre, _, product = state
    changed = product.clone()
    norm_sq = coefficients.square().sum().clamp_min(1e-20)
    changed[0, -1] += coefficients.to(product.dtype) * (
        (desired - original) / norm_sq).to(product.dtype)
    write = model.blocks[1].mlp.Down(changed) + model.blocks[1].mlp.Down_bias
    return suffix_from_layer1(model, x0, value_bus, pre + write)[0, -1]


@torch.inference_mode()
def discovery_statistics(model, rows, sites, direction_rows, arrays,
                         inverse_square, config):
    result = {}
    fraction = config["local_sensitivity_fraction"]
    for name, selected in sites.items():
        coefficients = direction_rows[name]["coefficients"].cuda()
        key = safe_name(f"{name}_whitened")
        values = torch.from_numpy(arrays[f"{key}_eigenvalues"][:config["candidate_modes"]]).cuda()
        vectors = torch.from_numpy(arrays[f"{key}_eigenvectors"][:, :config["candidate_modes"]]).cuda()
        tau = float(arrays[f"{key}_tau"])
        records = []
        for site in selected:
            state = model_state_at_site(model, rows[site["row"]], site["position"])
            normalized, product = state[-2], state[-1]
            x = normalized[0, -1].float()
            z = x @ inverse_square
            original = coefficients @ product[0, -1].float()
            isotropic = tau * z.square().sum()
            target = original - isotropic
            contributions = (z @ vectors).square() * values
            baseline = logits_with_coordinate(model, state, coefficients, isotropic, original)
            local = logits_with_coordinate(
                model, state, coefficients, isotropic + fraction * target, original)
            weights = F.softmax(baseline.float(), -1)
            effect = local.float() - baseline.float()
            effect = effect - (weights * effect).sum()
            per_q_energy = float((weights * effect.square()).sum() /
                                 (fraction * target).square().clamp_min(1e-20))
            records.append({
                "target": float(target),
                "contributions": contributions.cpu().tolist(),
                "per_q_fisher_energy": per_q_energy,
                "site": site,
            })
        matrix = np.asarray([record["contributions"] for record in records])
        target = np.asarray([record["target"] for record in records])
        fisher = np.asarray([record["per_q_fisher_energy"] for record in records])
        fisher = np.clip(fisher, np.median(fisher) * 1e-4 + 1e-30,
                         np.median(fisher) * 1e4 + 1e-30)
        result[name] = {
            "records": records,
            "orders": {
                "magnitude": list(range(config["candidate_modes"])),
                "value_greedy": greedy_order(matrix, target, np.ones_like(target),
                                               config["candidate_modes"]),
                "fisher_greedy": greedy_order(matrix, target, fisher,
                                                config["candidate_modes"]),
            },
            "fisher_weight_range": [float(fisher.min()), float(fisher.max())],
        }
    return result


@torch.inference_mode()
def validation_sweep(model, rows, sites, direction_rows, arrays,
                     inverse_square, discovery, config, tokenizer):
    result = {}
    for name, selected in sites.items():
        coefficients = direction_rows[name]["coefficients"].cuda()
        key = safe_name(f"{name}_whitened")
        values = torch.from_numpy(arrays[f"{key}_eigenvalues"][:config["candidate_modes"]]).cuda()
        vectors = torch.from_numpy(arrays[f"{key}_eigenvectors"][:, :config["candidate_modes"]]).cuda()
        tau = float(arrays[f"{key}_tau"])
        records = []
        for site in selected:
            state = model_state_at_site(model, rows[site["row"]], site["position"])
            normalized, product = state[-2], state[-1]
            z = normalized[0, -1].float() @ inverse_square
            original = coefficients @ product[0, -1].float()
            isotropic = tau * z.square().sum()
            target_value = original - isotropic
            contributions = (z @ vectors).square() * values
            baseline = logits_with_coordinate(model, state, coefficients, isotropic, original)
            site_result = {**site, "token": tokenizer.decode([site["token_id"]]),
                           "target_value": float(target_value), "rankings": {}}
            target_logits = {}
            for scale in config["scales"]:
                target_logits[str(scale)] = logits_with_coordinate(
                    model, state, coefficients, isotropic + scale * target_value, original)
            for ranking, order in discovery[name]["orders"].items():
                site_result["rankings"][ranking] = {}
                for rank in config["ranks"]:
                    indices = order[:rank]
                    piece = contributions[indices].sum()
                    complement = target_value - piece
                    scale_rows = {}
                    for scale in config["scales"]:
                        target_logits_scale = target_logits[str(scale)]
                        target_effect = target_logits_scale - baseline
                        predicted = logits_with_coordinate(
                            model, state, coefficients, isotropic + scale * piece, original) - baseline
                        complement_effect = logits_with_coordinate(
                            model, state, coefficients, isotropic + scale * complement, original) - baseline
                        fidelity, cosine, energy = fisher_fidelity(
                            target_logits_scale, target_effect, predicted)
                        bulk_fidelity, bulk_cosine, _ = fisher_fidelity(
                            target_logits_scale, target_effect, complement_effect)
                        scale_rows[str(scale)] = {
                            "fidelity": fidelity, "cosine": cosine,
                            "complement_fidelity": bulk_fidelity,
                            "complement_cosine": bulk_cosine,
                            "target_effect_energy": energy,
                        }
                    site_result["rankings"][ranking][str(rank)] = {
                        "indices": indices,
                        "piece": float(piece),
                        "scalar_fraction": float(piece / target_value.clamp(min=-1e30, max=1e30))
                            if abs(float(target_value)) > 1e-12 else None,
                        "scales": scale_rows,
                    }
            records.append(site_result)
        summary = {}
        for ranking in discovery[name]["orders"]:
            summary[ranking] = {}
            for rank in config["ranks"]:
                summary[ranking][str(rank)] = {}
                for scale in config["scales"]:
                    rows_for_metric = [record["rankings"][ranking][str(rank)]["scales"][str(scale)]
                                       for record in records]
                    summary[ranking][str(rank)][str(scale)] = {
                        "median_fidelity": float(np.median([row["fidelity"] for row in rows_for_metric])),
                        "median_complement_fidelity": float(np.median(
                            [row["complement_fidelity"] for row in rows_for_metric])),
                        "median_cosine": float(np.median([row["cosine"] for row in rows_for_metric])),
                    }
        result[name] = {"orders": discovery[name]["orders"],
                        "summary": summary, "sites": records}
    return result


def aggregate(validation, rankings, ranks, scales):
    output = {}
    for ranking in rankings:
        output[ranking] = {}
        for rank in ranks:
            output[ranking][str(rank)] = {}
            for scale in scales:
                category_rows = [row["summary"][ranking][str(rank)][str(scale)]
                                 for row in validation.values()]
                output[ranking][str(rank)][str(scale)] = {
                    "median_fidelity": float(np.median([r["median_fidelity"] for r in category_rows])),
                    "median_complement_fidelity": float(np.median(
                        [r["median_complement_fidelity"] for r in category_rows])),
                    "category_fidelities": [r["median_fidelity"] for r in category_rows],
                    "category_complement_fidelities": [r["median_complement_fidelity"] for r in category_rows],
                }
    return output


def main():
    config = json.loads(CONFIG.read_text())
    atlas = json.loads(ATLAS_OUTPUT.read_text())
    arrays = np.load(ATLAS_ARRAYS)
    inverse_square = torch.from_numpy(arrays["covariance_inverse_square"]).cuda()
    all_categories = atlas["config"]["reference_categories"] + atlas["config"]["prospective_categories"]
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    down = state["transformer.h.1.mlp.Down.weight"].float().cuda()
    del state
    archive = np.load(SPARSE / "natural_large.npz")
    h_std = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32)).std(0).clamp_min(1e-4)
    del archive
    directions, metadata = load_feature_directions(
        {"strong_categories": all_categories, "sae_seeds": atlas["config"]["feature_seeds"],
         "seed": atlas["config"]["seed"]}, h_std, down)
    names = [f"semantic/seed{config['feature_seed']}/{category}" for category in config["categories"]]
    direction_rows = {name: {"metadata": metadata[name], "coefficients": directions[name]}
                      for name in names}
    pilot_like = {"spectra": {name: atlas["spectra"][name] for name in names},
                  "reconstruction": {name: atlas["reconstruction"][name] for name in names}}
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")

    discovery_spec = config["discovery"]
    discovery_rows = all_rows[discovery_spec["start"]:
                              discovery_spec["start"] + discovery_spec["rows"]]
    discovery_sites = select_transfer_sites(
        model, discovery_rows, direction_rows, pilot_like, arrays, inverse_square,
        {"causal_transfer": {"positions": config["positions"]},
         "causal_sites_per_direction": discovery_spec["sites_per_category"]}, tokenizer)
    discovery = discovery_statistics(model, discovery_rows, discovery_sites,
                                     direction_rows, arrays, inverse_square, config)

    validation_spec = config["validation"]
    validation_rows = all_rows[validation_spec["start"]:
                               validation_spec["start"] + validation_spec["rows"]]
    validation_sites = select_transfer_sites(
        model, validation_rows, direction_rows, pilot_like, arrays, inverse_square,
        {"causal_transfer": {"positions": config["positions"]},
         "causal_sites_per_direction": validation_spec["sites_per_category"]}, tokenizer)
    validation = validation_sweep(model, validation_rows, validation_sites,
                                  direction_rows, arrays, inverse_square,
                                  discovery, config, tokenizer)
    rankings = ["magnitude", "value_greedy", "fisher_greedy"]
    combined = aggregate(validation, rankings, config["ranks"], config["scales"])

    magnitude_rank2 = combined["magnitude"]["2"]["1.0"]["median_fidelity"]
    fisher_rank2 = combined["fisher_greedy"]["2"]["1.0"]["median_fidelity"]
    fisher_rank4 = combined["fisher_greedy"]["4"]["1.0"]
    rank4_wins = sum(a > b for a, b in zip(
        fisher_rank4["category_fidelities"], fisher_rank4["category_complement_fidelities"]))
    magnitude_small = combined["magnitude"]["1"]["0.1"]["median_fidelity"]
    magnitude_full = combined["magnitude"]["1"]["1.0"]["median_fidelity"]
    value_rank4 = combined["value_greedy"]["4"]["1.0"]["median_fidelity"]
    scale_values = [combined["fisher_greedy"]["4"][str(scale)]["median_fidelity"]
                    for scale in config["scales"]]
    hypotheses = {
        "D1_downstream_rank2": {"supported": fisher_rank2 - magnitude_rank2 >= .10,
                                "fisher": fisher_rank2, "magnitude": magnitude_rank2,
                                "improvement": fisher_rank2 - magnitude_rank2},
        "D2_downstream_rank4": {"supported": fisher_rank4["median_fidelity"] >= .30
            and rank4_wins >= 2, "median_fidelity": fisher_rank4["median_fidelity"],
            "category_wins": rank4_wins},
        "D3_small_scale_rescue": {"supported": magnitude_small - magnitude_full >= .20,
                                   "scale_0.1": magnitude_small, "scale_1.0": magnitude_full,
                                   "improvement": magnitude_small - magnitude_full},
        "D4_fisher_over_value": {"supported": fisher_rank4["median_fidelity"] - value_rank4 >= .05,
                                  "fisher": fisher_rank4["median_fidelity"],
                                  "value": value_rank4,
                                  "improvement": fisher_rank4["median_fidelity"] - value_rank4},
        "D5_cross_scale_robustness": {"supported": min(scale_values) >= .30,
                                      "minimum": min(scale_values),
                                      "by_scale": dict(zip(map(str, config["scales"]), scale_values))},
    }
    result = {"config": config, "discovery": discovery, "validation": validation,
              "aggregate": combined, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))

    lines = ["# Downstream-aware spectral mode ranking", "",
             "Mode orders were fitted on one FineWeb panel and evaluated on a disjoint panel. All finite interventions follow the same native feature-coordinate path; scale 1 reproduces the original atlas intervention.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Downstream ranking](spectral_downstream_ranking.png)", "",
              "| Ranking | Rank | Scale 0.1 | Scale 0.25 | Scale 0.5 | Scale 1.0 |",
              "|---|---:|---:|---:|---:|---:|"]
    for ranking in rankings:
        for rank in config["ranks"]:
            row = combined[ranking][str(rank)]
            lines.append(f"| {ranking} | {rank} | {row['0.1']['median_fidelity']:.3f} | "
                         f"{row['0.25']['median_fidelity']:.3f} | {row['0.5']['median_fidelity']:.3f} | "
                         f"{row['1.0']['median_fidelity']:.3f} |")
    lines += ["", "`value_greedy` minimizes natural scalar reconstruction error. `fisher_greedy` uses the same exact spectral modes but weights discovery examples by local downstream softmax-Fisher sensitivity. Raw sites and per-category results are in `spectral_downstream_ranking.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ranking in rankings:
        axes[0].plot(config["ranks"], [combined[ranking][str(rank)]["1.0"]["median_fidelity"]
                                      for rank in config["ranks"]], marker="o", label=ranking)
    axes[0].set_xlabel("spectral modes"); axes[0].set_ylabel("median causal fidelity")
    axes[0].set_title("Disjoint full-scale validation"); axes[0].legend(fontsize=8)
    for ranking in rankings:
        axes[1].plot(config["scales"], [combined[ranking]["4"][str(scale)]["median_fidelity"]
                                       for scale in config["scales"]], marker="o", label=ranking)
    axes[1].set_xlabel("intervention scale"); axes[1].set_ylabel("median causal fidelity")
    axes[1].set_title("Rank-4 scale sweep"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
