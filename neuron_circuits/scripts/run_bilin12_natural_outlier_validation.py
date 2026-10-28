#!/usr/bin/env python3
"""Test whether independent-checkpoint raw spikes are naturally used and causal."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import collect_states_writes, final_logits, load_snapshot_model
from run_bilin18_depth_spectral_redteam import output_map_eigensystem
from run_bilin18_layer17_tail_cleaning_sweep import reconstruct_quadratic
from run_bilin18_layer8_restricted_tensor_program import ridge_covariance_transforms
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_natural_outlier_validation.json")
OUTPUT = TENSOR_ROOT / "bilin12_natural_outlier_validation.json"
ARRAYS = TENSOR_ROOT / "bilin12_natural_outlier_validation.npz"
REPORT = TENSOR_ROOT / "BILIN12_NATURAL_OUTLIER_VALIDATION.md"
FIGURE = TENSOR_ROOT / "bilin12_natural_outlier_validation.png"


def subspace_overlap(first, second):
    rank = min(first.shape[1], second.shape[1])
    return float((first.T @ second).square().sum() / rank)


def variance_capture(centered, basis):
    return float((centered @ basis).square().sum() / centered.square().sum().clamp_min(1e-30))


def correlation(first, second):
    first = first.double() - first.double().mean(); second = second.double() - second.double().mean()
    return float(first @ second / (first.norm() * second.norm()).clamp_min(1e-30))


def exact_kl(reference, changed):
    probability = F.softmax(reference, -1)
    return (probability * (F.log_softmax(reference, -1) - F.log_softmax(changed, -1))).sum(-1)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    behavior_source = np.load(config["behavior"]["path"], mmap_mode="r")
    behavior_rows = behavior_source[config["behavior"]["start"]:config["behavior"]["start"] + config["behavior"]["rows"]]
    results = {}; arrays = {}
    for layer in config["layers"]:
        print("natural outlier layer", layer, flush=True)
        block = model.blocks[layer]
        down = block.mlp.Down.weight.float(); left = block.mlp.Left.weight.float(); right = block.mlp.Right.weight.float()
        raw_singular, raw_basis = output_map_eigensystem(down, left, right)
        discovery_x, discovery_write = collect_states_writes(model, layer, config["discovery"], config["positions"], config["batch_size"])
        evaluation_x, evaluation_write = collect_states_writes(model, layer, config["evaluation"], config["positions"], config["batch_size"])
        mean = discovery_write.mean(0)
        # collect_states_writes returns the pre-bias down projection, whereas
        # final_logits intervenes on the complete MLP write. Center causal
        # projections around the corresponding post-bias mean.
        causal_mean = mean + block.mlp.Down_bias.float()
        centered_discovery = discovery_write - mean
        centered_evaluation = evaluation_write - mean
        _, _, natural_vh = torch.linalg.svd(centered_discovery, full_matrices=False)
        natural_basis = natural_vh.T
        raw_capture = {str(rank): variance_capture(centered_evaluation, raw_basis[:, :rank]) for rank in (1, 2, 4)}
        optimal_capture = {str(rank): variance_capture(centered_evaluation, natural_basis[:, :rank]) for rank in (1, 2, 4)}
        square, inverse, covariance = ridge_covariance_transforms(discovery_x, config["covariance_ridge_fraction"])
        raw_correlations = []; clean_correlations = []; clean_top4 = []
        z = evaluation_x @ inverse
        for component in range(config["rank"]):
            coefficients = down.T @ raw_basis[:, component]
            form = left.T @ (coefficients[:, None] * right)
            form = 0.5 * (form + form.T)
            raw_full, raw_top, _ = reconstruct_quadratic(evaluation_x, form, config["rank"], preserve_trace=True)
            weighted = square @ form @ square
            clean_full, clean_top, clean_stats = reconstruct_quadratic(z, weighted, config["rank"], preserve_trace=True)
            native_score = evaluation_write @ raw_basis[:, component]
            raw_correlations.append(correlation(native_score, torch.from_numpy(raw_top).cuda()))
            clean_correlations.append(correlation(native_score, torch.from_numpy(clean_top).cuda()))
            clean_top4.append(clean_stats["clean_top4_energy"])
        random_basis = torch.linalg.qr(torch.randn(model.config.n_embd, 4, generator=generator, device="cuda"), mode="reduced").Q
        bases = {"raw1": raw_basis[:, :1], "raw2": raw_basis[:, :2], "raw4": raw_basis[:, :4], "natural4": natural_basis[:, :4], "random4": random_basis, "full": raw_basis}
        kl_rows = {name: [] for name in bases}
        for start in range(0, len(behavior_rows), config["batch_size"]):
            ids = torch.as_tensor(np.asarray(behavior_rows[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
            reference = final_logits(model, ids)
            for name, basis in bases.items():
                changed = final_logits(model, ids, layer, -1, causal_mean, basis)
                kl_rows[name].extend(exact_kl(reference, changed).cpu().tolist())
        behavior = {name: {"median_kl": float(np.median(values)), "mean_kl": float(np.mean(values)), "p90_kl": float(np.quantile(values, .9))} for name, values in kl_rows.items()}
        results[str(layer)] = {
            "raw_spectrum_top4": raw_singular[:4].cpu().tolist(),
            "raw_capture": raw_capture, "optimal_capture": optimal_capture,
            "variance_efficiency": {rank: raw_capture[rank] / max(optimal_capture[rank], 1e-30) for rank in raw_capture},
            "raw_natural_overlap4": subspace_overlap(raw_basis[:, :4], natural_basis[:, :4]),
            "covariance": covariance,
            "raw_rank4_score_correlations": raw_correlations,
            "clean_rank4_score_correlations": clean_correlations,
            "clean_top4_form_energy": clean_top4,
            "behavior": behavior,
        }
        arrays[f"layer{layer}_raw_basis"] = raw_basis[:, :16].cpu().numpy()
        arrays[f"layer{layer}_natural_basis"] = natural_basis[:, :16].cpu().numpy()
        arrays[f"layer{layer}_write_mean"] = mean.cpu().numpy()
        arrays[f"layer{layer}_causal_write_mean"] = causal_mean.cpu().numpy()
    layers = [str(value) for value in config["layers"]]
    hypotheses = {
        "N1_raw_top4_tracks_natural_write": {"supported": all(results[layer]["variance_efficiency"]["4"] >= .5 for layer in layers)},
        "N2_middle_leading_spike_is_used": {"supported": results["5"]["variance_efficiency"]["1"] >= .25, "efficiency": results["5"]["variance_efficiency"]["1"]},
        "N3_late_raw4_is_behaviorally_compact": {"supported": results["11"]["behavior"]["raw4"]["median_kl"] <= .02, "median_kl": results["11"]["behavior"]["raw4"]["median_kl"]},
        "N4_raw4_beats_random_behaviorally": {"supported": all(results[layer]["behavior"]["raw4"]["median_kl"] < results[layer]["behavior"]["random4"]["median_kl"] for layer in layers)},
        "N5_covariance_cleaning_shortens_forms": {"supported": all(float(np.median(results[layer]["clean_rank4_score_correlations"])) - float(np.median(results[layer]["raw_rank4_score_correlations"])) >= .1 for layer in layers)},
        "N6_cleaned_forms_predict_heldout_scores": {"supported": all(float(np.median(results[layer]["clean_rank4_score_correlations"])) >= .9 for layer in layers)},
        "N7_full_projection_identity": {"supported": all(results[layer]["behavior"]["full"]["median_kl"] <= 1e-7 for layer in layers)},
    }
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Natural-state and causal validation of independent-checkpoint outliers", "",
        "Raw full-tensor output directions from the 12-layer checkpoint are compared with centered FineWeb MLP writes, covariance-cleaned input forms, and exact final-distribution effects when only the final-position MLP write is retained in a frozen subspace. Discovery, evaluation, and behavioral rows are disjoint. Causal projections use the post-bias write mean; an earlier pre-bias centering bug was corrected before these reported values.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Natural outlier validation](bilin12_natural_outlier_validation.png)", "", "| Layer | Raw/optimal variance@4 | Efficiency | Raw-natural overlap | Raw/clean score corr. | Clean form top-4 | KL raw4 / natural4 / random4 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for layer, row in results.items():
        lines.append(f"| {layer} | {row['raw_capture']['4']:.3f} / {row['optimal_capture']['4']:.3f} | {row['variance_efficiency']['4']:.3f} | {row['raw_natural_overlap4']:.3f} | {np.median(row['raw_rank4_score_correlations']):.3f} / {np.median(row['clean_rank4_score_correlations']):.3f} | {np.median(row['clean_top4_form_energy']):.3f} | {row['behavior']['raw4']['median_kl']:.4f} / {row['behavior']['natural4']['median_kl']:.4f} / {row['behavior']['random4']['median_kl']:.4f} |")
    lines += ["", "Raw spectral energy, natural-use variance, covariance-short score prediction, and final-distribution sufficiency are distinct validation levels. A spike can pass the first and fail any of the latter three.", "", "**Cleaning caveat.** The covariance-cleaning rows here used only 1,024 discovery states. The later [sample-size and ridge stability red team](BILIN12_CLEANING_STABILITY_REDTEAM.md) supersedes N5/N6 as evidence about cleaning: at this width, cleaning is a regularized covariance-estimation problem, and the small-sample result is not stable enough to support a fourth-moment interpretation."]
    REPORT.write_text("\n".join(lines) + "\n")
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.2)); x = np.arange(len(layers)); width = .25
    plots[0].bar(x - width/2, [results[layer]["raw_capture"]["4"] for layer in layers], width, label="raw4")
    plots[0].bar(x + width/2, [results[layer]["optimal_capture"]["4"] for layer in layers], width, label="optimal4")
    plots[1].bar(x - width/2, [np.median(results[layer]["raw_rank4_score_correlations"]) for layer in layers], width, label="raw")
    plots[1].bar(x + width/2, [np.median(results[layer]["clean_rank4_score_correlations"]) for layer in layers], width, label="clean")
    for offset, name in zip((-width, 0, width), ("raw4", "natural4", "random4")):
        plots[2].bar(x + offset, [results[layer]["behavior"][name]["median_kl"] for layer in layers], width, label=name)
    for plot, title in zip(plots, ("Centered write variance", "Rank-4 score correlation", "Final-distribution KL")):
        plot.set_xticks(x, layers); plot.set_xlabel("layer"); plot.set_title(title); plot.grid(alpha=.2); plot.legend()
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
