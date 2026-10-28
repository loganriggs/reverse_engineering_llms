#!/usr/bin/env python3
"""Prospective spectral-circuit discovery on previously untested token categories."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from redteam_bilin18_spectral_spikes import causal_curves, select_transfer_sites
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_spike_pilot import (
    SPARSE,
    TENSOR_ROOT,
    center_matrix,
    eigensystem,
    interaction_matrix,
    load_feature_directions,
    safe_name,
)


CONFIG = Path("configs/bilin18_spectral_atlas.json")
PILOT_ARRAYS = TENSOR_ROOT / "spectral_spike_pilot.npz"
OUTPUT = TENSOR_ROOT / "spectral_atlas.json"
ARRAYS = TENSOR_ROOT / "spectral_atlas.npz"
REPORT = TENSOR_ROOT / "SPECTRAL_ATLAS.md"
FIGURE = TENSOR_ROOT / "spectral_atlas.png"


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


@torch.inference_mode()
def leading_mode(left, right, coefficients, square):
    matrix = interaction_matrix(left, right, coefficients.cuda())
    centered, tau = center_matrix(square @ matrix @ square)
    values, vectors = eigensystem(centered)
    return float(values[0]), vectors[:, 0].cpu(), float(tau)


def assignment_test(matrix):
    count = len(matrix)
    observed = float(np.trace(matrix) / count)
    distribution = [float(np.mean([matrix[i, p[i]] for i in range(count)]))
                    for p in itertools.permutations(range(count))]
    return {
        "observed_diagonal_mean": observed,
        "permutation_p_value": float(np.mean(np.asarray(distribution) >= observed)),
        "diagonal_top_matches": int(sum(np.argmax(matrix[i]) == i for i in range(count))),
        "permutations": len(distribution),
    }


@torch.inference_mode()
def complete_spectrum(name, coefficients, left, right, square, config, saved):
    raw = interaction_matrix(left, right, coefficients.cuda())
    result = {}
    generator = torch.Generator(device="cuda").manual_seed(
        config["seed"] + int.from_bytes(name.encode(), "little") % 1_000_003)
    signs = torch.where(torch.rand(config["null_sign_flips"], len(coefficients),
                                  generator=generator, device="cuda") < .5, -1., 1.)
    for cleaning, matrix in (("raw", raw), ("whitened", square @ raw @ square)):
        centered, tau = center_matrix(matrix)
        values, vectors = eigensystem(centered)
        null_maxima = []
        for sign in signs:
            null_raw = interaction_matrix(left, right, coefficients.cuda() * sign)
            null_matrix = null_raw if cleaning == "raw" else square @ null_raw @ square
            null_centered, _ = center_matrix(null_matrix)
            null_maxima.append(float(torch.linalg.eigvalsh(null_centered).abs().max()))
        threshold = float(np.quantile(null_maxima, config["null_quantile"]))
        spike_count = int((values.abs() > threshold).sum())
        key = safe_name(f"{name}_{cleaning}")
        saved[f"{key}_eigenvalues"] = values[:16].cpu().numpy().astype(np.float32)
        saved[f"{key}_eigenvectors"] = vectors[:, :16].cpu().numpy().astype(np.float32)
        saved[f"{key}_tau"] = np.asarray(float(tau), dtype=np.float32)
        result[cleaning] = {
            "spike_count": spike_count,
            "null_threshold": threshold,
            "top_eigenvalues": values[:16].cpu().tolist(),
        }
    return result


def semantic_reconstruction(selected, directions, saved, inverse_square,
                            input_archive, token_archive, tokenizer):
    flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    category_index = {name: i for i, name in enumerate(CATEGORIES)}
    output = {}
    for category in selected:
        name = f"semantic/seed101/{category}"
        key = safe_name(f"{name}_whitened")
        value = torch.from_numpy(saved[f"{key}_eigenvalues"][:1]).cuda()
        vector = torch.from_numpy(saved[f"{key}_eigenvectors"][:, :1]).cuda()
        tau = float(saved[f"{key}_tau"])
        coefficient = directions[name].cuda()

        def scores(split, limit=None):
            h = torch.from_numpy(token_archive[f"{split}__h1"][:limit].astype(np.float32)).cuda()
            x = torch.from_numpy(input_archive[f"{split}__input1"][:limit].astype(np.float32)).cuda()
            z = x @ inverse_square
            full = h @ coefficient - tau * z.square().sum(1)
            top = (z @ vector).square().squeeze(1) * value[0]
            labels = flags[token_archive[f"{split}__token_ids"][:limit], category_index[category]]
            return labels, full.cpu().numpy(), top.cpu().numpy()

        train_labels, train_full, _ = scores("fineweb_train", 16384)
        orientation = 1. if auc(train_labels, train_full) >= .5 else -1.
        splits = {}
        for split in ("fineweb_test", "wikitext103", "tinystories"):
            labels, full, top = scores(split)
            splits[split] = {
                "positives": int(labels.sum()),
                "full_auc": auc(labels, orientation * full),
                "top1_auc": auc(labels, orientation * top),
            }
        output[name] = {"orientation": orientation, "splits": splits}
    return output


def main():
    config = json.loads(CONFIG.read_text())
    all_categories = config["reference_categories"] + config["prospective_categories"]
    direction_config = {"strong_categories": all_categories,
                        "sae_seeds": config["feature_seeds"], "seed": config["seed"]}
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    left = state["transformer.h.1.mlp.Left.weight"].float().cuda()
    right = state["transformer.h.1.mlp.Right.weight"].float().cuda()
    down = state["transformer.h.1.mlp.Down.weight"].float().cuda()
    del state
    token_archive = np.load(SPARSE / "natural_large.npz")
    h_std = torch.from_numpy(token_archive["fineweb_train__h1"].astype(np.float32)).std(0).clamp_min(1e-4)
    directions, metadata = load_feature_directions(direction_config, h_std, down)
    del h_std
    pilot_arrays = np.load(PILOT_ARRAYS)
    square = torch.from_numpy(pilot_arrays["covariance_square"]).cuda()
    inverse_square = torch.from_numpy(pilot_arrays["covariance_inverse_square"]).cuda()

    leading = {}
    for seed in config["feature_seeds"]:
        for category in config["prospective_categories"]:
            name = f"semantic/seed{seed}/{category}"
            value, vector, tau = leading_mode(left, right, directions[name], square)
            leading[name] = {"eigenvalue": value, "vector": vector, "tau": tau}
            print("screened", name, flush=True)
    categories = config["prospective_categories"]
    match = np.zeros((len(categories), len(categories)))
    for i, first in enumerate(categories):
        a = leading[f"semantic/seed101/{first}"]["vector"]
        for j, second in enumerate(categories):
            b = leading[f"semantic/seed102/{second}"]["vector"]
            match[i, j] = float(torch.dot(a, b).square())
    stability = {"matrix": match.tolist(), **assignment_test(match)}
    diagonal = {category: float(match[i, i]) for i, category in enumerate(categories)}
    selected = sorted(categories, key=diagonal.get, reverse=True)[:config["select_count"]]
    print("frozen selection", selected, flush=True)

    saved = {"covariance_square": square.cpu().numpy().astype(np.float32),
             "covariance_inverse_square": inverse_square.cpu().numpy().astype(np.float32)}
    spectra = {}
    selected_names = []
    for category in selected:
        for kind in ("semantic", "control"):
            name = f"{kind}/seed{config['causal_seed']}/{category}"
            spectra[name] = {"metadata": metadata[name], "cleanings": complete_spectrum(
                name, directions[name], left, right, square, config, saved)}
            selected_names.append(name)
    np.savez_compressed(ARRAYS, **saved)

    input_archive = np.load(SPARSE / "layer1_inputs_large.npz")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    reconstruction = semantic_reconstruction(selected, directions, saved, inverse_square,
                                              input_archive, token_archive, tokenizer)
    pilot_like = {"spectra": spectra, "reconstruction": reconstruction}
    direction_rows = {name: {"metadata": metadata[name], "coefficients": directions[name]}
                      for name in selected_names}
    model = load_full_model("cuda")
    all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    panel = config["causal_panel"]
    rows = all_rows[panel["start"]:panel["start"] + panel["rows"]]
    sites = select_transfer_sites(
        model, rows, direction_rows, pilot_like, saved, inverse_square,
        {"causal_transfer": {"positions": panel["positions"]},
         "causal_sites_per_direction": config["causal_sites_per_direction"]}, tokenizer)
    causal = causal_curves(model, rows, sites, direction_rows, pilot_like, saved,
                           inverse_square, {"ranks": [1]}, tokenizer)

    semantic_names = [f"semantic/seed{config['causal_seed']}/{c}" for c in selected]
    control_names = [f"control/seed{config['causal_seed']}/{c}" for c in selected]
    def values(names, key, field):
        return [causal[name]["cleanings"]["whitened"]["summary"][key][field] for name in names]
    semantic = values(semantic_names, "1", "median_spike_fidelity")
    bulk = values(semantic_names, "1", "median_bulk_fidelity")
    controls = values(control_names, "1", "median_spike_fidelity")
    significant = values(semantic_names, "significant", "median_spike_fidelity")
    control_significant = values(control_names, "significant", "median_spike_fidelity")
    available_auc = []
    available_full = []
    for name in semantic_names:
        for row in reconstruction[name]["splits"].values():
            if np.isfinite(row["top1_auc"]) and np.isfinite(row["full_auc"]):
                available_auc.append(row["top1_auc"]); available_full.append(row["full_auc"])
    causal_wins = sum(s >= .20 and s > b for s, b in zip(semantic, bulk))
    hypotheses = {
        "A1_prospective_causality": {"supported": causal_wins >= 2, "category_wins": causal_wins},
        "A2_prospective_control_advantage": {"supported": float(np.median(semantic) - np.median(controls)) >= .10,
            "semantic_median": float(np.median(semantic)), "control_median": float(np.median(controls)),
            "advantage": float(np.median(semantic) - np.median(controls))},
        "A3_prospective_semantics": {"supported": float(np.median(available_auc)) >= .70
            and float(np.median(available_full) - np.median(available_auc)) <= .10,
            "top1_auc_median": float(np.median(available_auc)), "full_auc_median": float(np.median(available_full)),
            "auc_loss": float(np.median(available_full) - np.median(available_auc))},
        "A4_all_category_matching": {"supported": stability["permutation_p_value"] <= .05
            and stability["diagonal_top_matches"] >= 6, **stability},
        "A5_significant_only": {"supported": float(np.median(significant)) >= .20
            and float(np.median(significant) - np.median(control_significant)) >= .10,
            "semantic_median": float(np.median(significant)),
            "control_median": float(np.median(control_significant)),
            "advantage": float(np.median(significant) - np.median(control_significant))},
    }
    result = {"config": config, "selection_scores": diagonal, "selected": selected,
              "stability": stability, "spectra": spectra, "reconstruction": reconstruction,
              "causal": causal, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(json_finite(result), indent=2, allow_nan=False))

    lines = ["# Prospective spectral-circuit atlas", "", "The causal candidates below were frozen from cross-seed leading-mode recurrence before causal evaluation.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Prospective atlas](spectral_atlas.png)", "",
              "| Selected category | Cross-seed overlap | Top-1 causal | Bulk | Matched control | Full/top-1 AUROC (FineWeb) |",
              "|---|---:|---:|---:|---:|---:|"]
    for category, spike, complement, control in zip(selected, semantic, bulk, controls):
        name = f"semantic/seed{config['causal_seed']}/{category}"
        rec = reconstruction[name]["splits"]["fineweb_test"]
        lines.append(f"| {category} | {diagonal[category]:.3f} | {spike:.3f} | {complement:.3f} | {control:.3f} | {rec['full_auc']:.3f}/{rec['top1_auc']:.3f} |")
    lines += ["", "All nine prospective categories and the exact assignment test are retained in `spectral_atlas.json`; eigensystems are in `spectral_atlas.npz`."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    image = axes[0].imshow(match, vmin=0, vmax=max(.5, match.max()), cmap="viridis")
    axes[0].set_xticks(range(len(categories)), categories, rotation=55, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(categories)), categories, fontsize=8)
    axes[0].set_title("Prospective cross-seed leading-mode overlap")
    fig.colorbar(image, ax=axes[0], fraction=.046)
    x = np.arange(len(selected)); width = .25
    axes[1].bar(x - width, semantic, width, label="semantic spike")
    axes[1].bar(x, bulk, width, label="bulk")
    axes[1].bar(x + width, controls, width, label="matched control")
    axes[1].set_xticks(x, selected, rotation=25, ha="right")
    axes[1].set_ylabel("median causal fidelity"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
