#!/usr/bin/env python3
"""Disjoint, rank-resolved red-team of Bilin18 spectral spike circuits."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_spike_pilot import (
    SPARSE,
    TENSOR_ROOT,
    correlation,
    fisher_fidelity,
    load_feature_directions,
    safe_name,
    subspace_overlap,
)
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_spike_redteam.json")
PILOT = TENSOR_ROOT / "spectral_spike_pilot.json"
ARRAYS = TENSOR_ROOT / "spectral_spike_pilot.npz"
OUTPUT = TENSOR_ROOT / "spectral_spike_redteam.json"
REPORT = TENSOR_ROOT / "SPECTRAL_SPIKE_REDTEAM.md"
FIGURE = TENSOR_ROOT / "spectral_spike_redteam.png"


def load_eigensystem(arrays, name, cleaning, rank):
    key = safe_name(f"{name}_{cleaning}")
    values = torch.from_numpy(arrays[f"{key}_eigenvalues"][:rank]).cuda()
    vectors = torch.from_numpy(arrays[f"{key}_eigenvectors"][:, :rank]).cuda()
    tau = float(arrays[f"{key}_tau"])
    return values, vectors, tau


def coordinates(x, cleaning, inverse_square):
    return x if cleaning == "raw" else x @ inverse_square


def reconstruction_curves(pilot, arrays, config, inverse_square, input_archive,
                          token_archive, tokenizer):
    flags = np.stack([token_flags(tokenizer.decode([index]))
                      for index in range(tokenizer.vocab_size)])
    category_index = {name: index for index, name in enumerate(CATEGORIES)}
    result = {}
    for name, pilot_row in pilot["spectra"].items():
        if pilot_row["metadata"]["kind"] != "semantic":
            continue
        category = pilot_row["metadata"]["category"]
        orientation = pilot["reconstruction"][name]["orientation"]
        result[name] = {}
        for cleaning in ("whitened",):
            all_values, all_vectors, _ = load_eigensystem(arrays, name, cleaning, 16)
            result[name][cleaning] = {}
            for split in ("fineweb_test", "wikitext103", "tinystories"):
                x = torch.from_numpy(input_archive[f"{split}__input1"].astype(np.float32)).cuda()
                z = coordinates(x, cleaning, inverse_square)
                projection = z @ all_vectors
                # The centered full score is reconstructed from all eigenmodes only
                # in the pilot; use its exact full AUC as the reference here.
                labels = flags[token_archive[f"{split}__token_ids"], category_index[category]]
                rows = {"full_auc": pilot["reconstruction"][name]["splits"][split]["full_auc"]}
                for rank in config["ranks"]:
                    top = (projection[:, :rank].square() * all_values[:rank]).sum(1).cpu().numpy()
                    rows[str(rank)] = {"auc": auc(labels, orientation * top)}
                result[name][cleaning][split] = rows
    return result


@torch.inference_mode()
def select_transfer_sites(model, rows, direction_rows, pilot, arrays,
                          inverse_square, config, tokenizer):
    positions_list = config["causal_transfer"]["positions"]
    positions = torch.as_tensor(positions_list, device="cuda")
    flags = np.stack([token_flags(tokenizer.decode([index]))
                      for index in range(tokenizer.vocab_size)])
    category_index = {name: index for index, name in enumerate(CATEGORIES)}
    candidates = {name: [] for name in direction_rows}
    for start in range(0, len(rows), 4):
        raw = np.asarray(rows[start:start + 4, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, _, _, normalized = prefix_to_layer1(model, ids)
        product = (model.blocks[1].mlp.Left(normalized[:, positions])
                   * model.blocks[1].mlp.Right(normalized[:, positions])).float()
        z = normalized[:, positions].float() @ inverse_square
        token_values = raw[:, positions_list]
        for name, row in direction_rows.items():
            coefficients = row["coefficients"].cuda()
            _, _, tau = load_eigensystem(arrays, name, "whitened", 1)
            full_centered = (product @ coefficients) - tau * z.square().sum(-1)
            if row["metadata"]["kind"] == "semantic":
                labels = flags[token_values, category_index[row["metadata"]["category"]]].astype(bool)
                orientation = pilot["reconstruction"][name]["orientation"]
                score = orientation * full_centered.cpu().numpy()
                keep = labels
            else:
                score = full_centered.abs().cpu().numpy()
                keep = np.ones_like(score, dtype=bool)
            for batch_index, position_index in zip(*np.nonzero(keep)):
                candidates[name].append({
                    "row": start + int(batch_index), "position": int(positions_list[position_index]),
                    "score": float(score[batch_index, position_index]),
                    "token_id": int(token_values[batch_index, position_index]),
                })
    selected = {}
    for name, records in candidates.items():
        records.sort(key=lambda row: row["score"], reverse=True)
        selected[name] = records[:config["causal_sites_per_direction"]]
    return selected


@torch.inference_mode()
def causal_curves(model, rows, sites, direction_rows, pilot, arrays,
                  inverse_square, config, tokenizer):
    result = {}
    for name, selected in sites.items():
        coefficients = direction_rows[name]["coefficients"].cuda()
        coefficient_norm_sq = coefficients.square().sum().clamp_min(1e-20)
        row = {"metadata": direction_rows[name]["metadata"], "cleanings": {}}
        for cleaning in ("raw", "whitened"):
            all_values, all_vectors, tau = load_eigensystem(arrays, name, cleaning, 16)
            ranks = list(config["ranks"])
            significant = pilot["spectra"][name]["cleanings"][cleaning]["spike_count"]
            labels = [str(rank) for rank in ranks] + ["significant"]
            accumulators = {label: {"spike": [], "bulk": [], "cosine": []}
                            for label in labels}
            records = []
            for site in selected:
                prefix = np.asarray(rows[site["row"], :site["position"] + 1], dtype=np.int64)
                ids = torch.as_tensor(prefix, device="cuda")[None]
                x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
                product = model.blocks[1].mlp.Left(normalized) * model.blocks[1].mlp.Right(normalized)
                native_write = model.blocks[1].mlp.Down(product) + model.blocks[1].mlp.Down_bias
                native = suffix_from_layer1(model, x0, value_bus, pre + native_write)[0, -1]
                x = normalized[0, -1].float()
                z = coordinates(x, cleaning, inverse_square)
                q = coefficients @ product[0, -1].float()
                isotropic = tau * z.square().sum()
                projection = z @ all_vectors

                def logits_for(desired):
                    changed = product.clone()
                    changed[0, -1] += coefficients.to(product.dtype) * (
                        (desired - q) / coefficient_norm_sq).to(product.dtype)
                    write = model.blocks[1].mlp.Down(changed) + model.blocks[1].mlp.Down_bias
                    return suffix_from_layer1(model, x0, value_bus, pre + write)[0, -1]

                isotropic_logits = logits_for(isotropic)
                target_effect = native - isotropic_logits
                site_record = {**site, "token": tokenizer.decode([site["token_id"]]),
                               "ranks": {}}
                for label, rank in [(str(rank), rank) for rank in ranks] + [("significant", significant)]:
                    spike = ((projection[:rank].square() * all_values[:rank]).sum()
                             if rank else torch.zeros((), device="cuda"))
                    bulk = q - isotropic - spike
                    spike_logits = logits_for(isotropic + spike)
                    bulk_logits = logits_for(isotropic + bulk)
                    spike_fidelity, spike_cosine, energy = fisher_fidelity(
                        native, target_effect, spike_logits - isotropic_logits)
                    bulk_fidelity, _, _ = fisher_fidelity(
                        native, target_effect, bulk_logits - isotropic_logits)
                    accumulators[label]["spike"].append(spike_fidelity)
                    accumulators[label]["bulk"].append(bulk_fidelity)
                    accumulators[label]["cosine"].append(spike_cosine)
                    site_record["ranks"][label] = {
                        "spike_fidelity": spike_fidelity,
                        "bulk_fidelity": bulk_fidelity,
                        "spike_cosine": spike_cosine,
                        "target_effect_energy": energy,
                    }
                records.append(site_record)
            row["cleanings"][cleaning] = {
                "significant_count": significant,
                "summary": {label: {
                    "median_spike_fidelity": float(np.median(values["spike"])),
                    "median_bulk_fidelity": float(np.median(values["bulk"])),
                    "median_spike_cosine": float(np.median(values["cosine"])),
                } for label, values in accumulators.items()},
                "sites": records,
            }
        result[name] = row
    return result


def stability_permutation(pilot, arrays, categories):
    matrix = np.zeros((len(categories), len(categories)))
    for i, first_category in enumerate(categories):
        key_first = safe_name(f"semantic/seed101/{first_category}_whitened")
        vectors_first = torch.from_numpy(arrays[f"{key_first}_eigenvectors"][:, :4])
        for j, second_category in enumerate(categories):
            key_second = safe_name(f"semantic/seed102/{second_category}_whitened")
            vectors_second = torch.from_numpy(arrays[f"{key_second}_eigenvectors"][:, :4])
            matrix[i, j] = subspace_overlap(vectors_first, vectors_second)
    observed = float(np.trace(matrix) / len(categories))
    permuted = []
    for permutation in itertools.permutations(range(len(categories))):
        permuted.append(float(np.mean([matrix[i, permutation[i]] for i in range(len(categories))])))
    p_value = float(np.mean(np.asarray(permuted) >= observed))
    diagonal_top = int(sum(np.argmax(matrix[i]) == i for i in range(len(categories))))
    return {"matrix": matrix.tolist(), "observed_diagonal_mean": observed,
            "permutation_p_value": p_value, "diagonal_top_matches": diagonal_top,
            "permutations": len(permuted)}


def main():
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(PILOT.read_text())
    arrays = np.load(ARRAYS)
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    down = state["transformer.h.1.mlp.Down.weight"].float().cuda(); del state
    token_archive = np.load(SPARSE / "natural_large.npz")
    h_train = torch.from_numpy(token_archive["fineweb_train__h1"].astype(np.float32))
    h_std = h_train.std(0).clamp_min(1e-4); del h_train
    pilot_config = pilot["config"]
    directions, metadata = load_feature_directions(pilot_config, h_std, down)
    direction_rows = {name: {"metadata": metadata[name], "coefficients": value}
                      for name, value in directions.items() if metadata[name]["seed"] == 101}
    inverse_square = torch.from_numpy(arrays["covariance_inverse_square"]).cuda()
    input_archive = np.load(SPARSE / "layer1_inputs_large.npz")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    reconstruction = reconstruction_curves(
        pilot, arrays, config, inverse_square, input_archive, token_archive, tokenizer)

    model = load_full_model("cuda")
    all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    spec = config["causal_transfer"]
    transfer_rows = all_rows[spec["start"]:spec["start"] + spec["rows"]]
    sites = select_transfer_sites(model, transfer_rows, direction_rows, pilot,
                                  arrays, inverse_square, config, tokenizer)
    causal = causal_curves(model, transfer_rows, sites, direction_rows, pilot,
                           arrays, inverse_square, config, tokenizer)
    stability = stability_permutation(pilot, arrays, config["strong_categories"])

    semantic_names = [f"semantic/seed101/{category}" for category in config["strong_categories"]]
    control_names = [f"control/seed101/{category}" for category in config["strong_categories"]]
    recon_rows = [split for name in reconstruction for split in reconstruction[name]["whitened"].values()]
    rank2_auc = float(np.median([row["2"]["auc"] for row in recon_rows]))
    full_auc = float(np.median([row["full_auc"] for row in recon_rows]))

    def med(names, cleaning, rank, field="median_spike_fidelity"):
        return float(np.median([causal[name]["cleanings"][cleaning]["summary"][str(rank)][field]
                                for name in names]))

    rank2_spike = med(semantic_names, "whitened", 2)
    rank2_wins = sum(causal[name]["cleanings"]["whitened"]["summary"]["2"]["median_spike_fidelity"]
                     > causal[name]["cleanings"]["whitened"]["summary"]["2"]["median_bulk_fidelity"]
                     for name in semantic_names)
    significant_semantic = med(semantic_names, "whitened", "significant")
    significant_control = med(control_names, "whitened", "significant")
    rank4 = med(semantic_names, "whitened", 4)
    rank16 = med(semantic_names, "whitened", 16)
    raw_semantic = med(semantic_names, "raw", 4)
    raw_control = med(control_names, "raw", 4)
    hypotheses = {
        "R1_rank2_semantics": {"supported": bool(rank2_auc >= .75 and full_auc - rank2_auc <= .08),
                               "rank2_auc_median": rank2_auc, "full_auc_median": full_auc,
                               "auc_loss": full_auc - rank2_auc},
        "R2_rank2_causality": {"supported": bool(rank2_spike >= .30 and rank2_wins >= 4),
                               "rank2_spike_median": rank2_spike,
                               "category_wins": rank2_wins},
        "R3_significant_only": {"supported": bool(significant_semantic >= .20
                                    and significant_semantic - significant_control >= .10),
                                "semantic_median": significant_semantic,
                                "control_median": significant_control},
        "R4_rank4_plateau": {"supported": bool(rank4 >= .50 and rank16 - rank4 <= .10),
                             "rank4_median": rank4, "rank16_median": rank16,
                             "rank16_minus_rank4": rank16 - rank4},
        "R5_raw_robustness": {"supported": bool(raw_semantic > .30
                                  and raw_semantic - raw_control >= .10),
                              "semantic_median": raw_semantic,
                              "control_median": raw_control},
        "R6_category_specific_stability": {"supported": bool(
            stability["permutation_p_value"] <= .05 and stability["diagonal_top_matches"] >= 4),
            **stability},
    }
    result = {"config": config, "reconstruction": reconstruction, "causal": causal,
              "stability": stability, "hypotheses": hypotheses,
              "selection_rule": "disjoint panel; semantic true positives ranked by full centered feature form, controls by absolute full centered form"}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))

    lines = ["# Spectral-spike disjoint red-team", "", "## Hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Spectral red-team](spectral_spike_redteam.png)", "",
              "| Category | Significant modes | Rank-1 | Rank-2 | Rank-4 | Rank-8 | Rank-16 | Bulk@4 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for category in config["strong_categories"]:
        name = f"semantic/seed101/{category}"
        rows = causal[name]["cleanings"]["whitened"]
        summary = rows["summary"]
        lines.append(f"| {category} | {rows['significant_count']} "
                     + " ".join(f"| {summary[str(rank)]['median_spike_fidelity']:.3f}" for rank in config["ranks"])
                     + f" | {summary['4']['median_bulk_fidelity']:.3f} |")
    lines += ["", "Sites are disjoint from the pilot and are selected by the complete centered feature form, never by a spike-only score. Raw and covariance-whitened bases are evaluated separately. `significant` retains only modes above the feature's frozen sign-flip-null threshold.", "",
              "Raw results: `spectral_spike_redteam.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    xs = config["ranks"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for category in config["strong_categories"]:
        name = f"semantic/seed101/{category}"
        axes[0].plot(xs, [causal[name]["cleanings"]["whitened"]["summary"][str(rank)]["median_spike_fidelity"] for rank in xs], marker="o", label=category)
        axes[1].plot(xs, [causal[name]["cleanings"]["raw"]["summary"][str(rank)]["median_spike_fidelity"] for rank in xs], marker="o")
    axes[0].set_title("Whitened modes"); axes[0].set_ylabel("causal fidelity"); axes[0].legend(fontsize=8)
    axes[1].set_title("Raw modes"); axes[1].set_ylabel("causal fidelity")
    matrix = np.asarray(stability["matrix"])
    image = axes[2].imshow(matrix, vmin=0, vmax=max(.5, matrix.max()), cmap="viridis")
    axes[2].set_title("Cross-seed top-4 overlap")
    axes[2].set_xticks(range(len(config["strong_categories"])), config["strong_categories"], rotation=45, ha="right")
    axes[2].set_yticks(range(len(config["strong_categories"])), config["strong_categories"])
    fig.colorbar(image, ax=axes[2], fraction=.046)
    for axis in axes[:2]: axis.set_xlabel("spectral rank")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
