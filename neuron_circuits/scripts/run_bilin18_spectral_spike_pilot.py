#!/usr/bin/env python3
"""Output-conditioned spectral-spike pilot for Bilin18 layer 1."""

from __future__ import annotations

import hashlib
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
from run_bilin18_tensor_decomposition import ROOT as TENSOR_ROOT
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_spike_pilot.json")
SPARSE = Path("artifacts/bilin18_sparse_codes")
OUTPUT = TENSOR_ROOT / "spectral_spike_pilot.json"
ARRAYS = TENSOR_ROOT / "spectral_spike_pilot.npz"
REPORT = TENSOR_ROOT / "SPECTRAL_SPIKE_PILOT.md"
FIGURE = TENSOR_ROOT / "spectral_spike_pilot.png"


def safe_name(value):
    return value.replace("/", "_").replace(":", "_")


def correlation(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    first, second = first - first.mean(), second - second.mean()
    return float(np.dot(first, second) /
                 max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def effective_rank(values):
    values = np.abs(np.asarray(values, dtype=np.float64))
    return float(values.sum() ** 2 / max(np.square(values).sum(), 1e-30))


def power_law(values, start, stop):
    values = np.sort(np.abs(np.asarray(values, dtype=np.float64)))[::-1]
    ranks = np.arange(1, len(values) + 1)
    keep = slice(start - 1, min(stop, len(values)))
    x, y = np.log(ranks[keep]), np.log(values[keep].clip(1e-30))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    r2 = 1 - np.square(y - predicted).sum() / max(np.square(y - y.mean()).sum(), 1e-30)
    return {"alpha": float(-slope), "r_squared": float(r2),
            "rank_start": start, "rank_stop": min(stop, len(values))}


def subspace_overlap(first, second):
    return float((first.T @ second).square().sum() / first.shape[1])


def interaction_matrix(left, right, coefficients):
    matrix = left.T @ (coefficients[:, None] * right)
    return .5 * (matrix + matrix.T)


def center_matrix(matrix):
    tau = torch.trace(matrix) / len(matrix)
    return matrix - tau * torch.eye(len(matrix), device=matrix.device), tau


def eigensystem(matrix):
    values, vectors = torch.linalg.eigh(matrix)
    order = values.abs().argsort(descending=True)
    return values[order], vectors[:, order]


def load_feature_directions(config, h_std, down):
    category_results = json.loads((SPARSE / "large_sae_results.json").read_text())
    semantic_indices = {
        str(seed): {category: category_results["seeds"][str(seed)]["categories"][category]["feature"]
                    for category in config["strong_categories"]}
        for seed in config["sae_seeds"]}
    directions, metadata = {}, {}
    for seed in config["sae_seeds"]:
        state = torch.load(SPARSE / f"activation_sae_large_seed{seed}.pt",
                           map_location="cpu", weights_only=True)
        encoder = state["encoder.weight"].float()
        decoder = state["decoder.weight"].float()
        excluded = {row["feature"] for row in category_results["seeds"][str(seed)]["categories"].values()}
        generator = torch.Generator().manual_seed(config["seed"] + seed)
        pool = [int(value) for value in torch.randperm(encoder.shape[0], generator=generator).tolist()
                if int(value) not in excluded][:1024]
        all_indices = list(semantic_indices[str(seed)].values()) + pool
        coefficient_rows = encoder[all_indices] / h_std[None]
        coefficient_norms = coefficient_rows.norm(dim=1)
        decoder_rows = (h_std[:, None] * decoder[:, all_indices])
        write_norms = (down @ decoder_rows.cuda()).norm(dim=0).cpu()
        offset = len(config["strong_categories"])
        available = set(range(len(pool)))
        for category_index, category in enumerate(config["strong_categories"]):
            feature = semantic_indices[str(seed)][category]
            key = f"semantic/seed{seed}/{category}"
            directions[key] = (encoder[feature] / h_std).clone()
            target = torch.stack((coefficient_norms[category_index].log(),
                                  write_norms[category_index].log()))
            candidates = torch.stack((coefficient_norms[offset:].log(),
                                      write_norms[offset:].log()), dim=1)
            distance = (candidates - target[None]).square().sum(1)
            chosen_pool = min(available, key=lambda index: float(distance[index]))
            available.remove(chosen_pool)
            control_feature = pool[chosen_pool]
            control_key = f"control/seed{seed}/{category}"
            directions[control_key] = (encoder[control_feature] / h_std).clone()
            metadata[key] = {"kind": "semantic", "seed": seed, "category": category,
                             "feature": feature,
                             "coefficient_norm": float(coefficient_norms[category_index]),
                             "decoder_write_norm": float(write_norms[category_index])}
            metadata[control_key] = {"kind": "control", "seed": seed, "category": category,
                                     "feature": control_feature,
                                     "coefficient_norm": float(coefficient_norms[offset + chosen_pool]),
                                     "decoder_write_norm": float(write_norms[offset + chosen_pool])}
        del state, encoder, decoder, decoder_rows
    return directions, metadata


@torch.inference_mode()
def covariance_transforms(x_cpu, examples):
    x = x_cpu[:examples].cuda().float()
    covariance = x.T @ x / len(x)
    values, vectors = torch.linalg.eigh(covariance)
    floor = values.median() * 1e-4
    values = values.clamp_min(floor)
    square = (vectors * values.sqrt()[None]) @ vectors.T
    inverse_square = (vectors * values.rsqrt()[None]) @ vectors.T
    return square, inverse_square, {
        "minimum_eigenvalue": float(values.min()), "maximum_eigenvalue": float(values.max()),
        "condition_number": float(values.max() / values.min())}


@torch.inference_mode()
def spectral_analysis(directions, metadata, left, right, square, config):
    result, saved = {}, {}
    eye = torch.eye(left.shape[1], device="cuda")
    for number, (name, coefficients_cpu) in enumerate(directions.items()):
        print("spectrum", number + 1, "/", len(directions), name, flush=True)
        coefficients = coefficients_cpu.cuda()
        raw = interaction_matrix(left, right, coefficients)
        variants = {"raw": raw, "whitened": square @ raw @ square}
        row = {"metadata": metadata[name], "cleanings": {}}
        seed = config["seed"] + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        signs = torch.where(torch.rand(
            config["null_sign_flips"], len(coefficients), generator=generator,
            device="cuda") < .5, -1., 1.)
        for variant_name, matrix in variants.items():
            centered, tau = center_matrix(matrix)
            values, vectors = eigensystem(centered)
            null_maxima, null_power = [], []
            for null_index in range(config["null_sign_flips"]):
                null_raw = interaction_matrix(left, right, coefficients * signs[null_index])
                null_matrix = null_raw if variant_name == "raw" else square @ null_raw @ square
                null_centered, _ = center_matrix(null_matrix)
                null_values = torch.linalg.eigvalsh(null_centered).cpu().numpy()
                null_maxima.append(float(np.abs(null_values).max()))
                null_power.append(power_law(
                    null_values, config["power_law_rank_start"],
                    config["power_law_rank_stop"]))
            threshold = float(np.quantile(null_maxima, config["null_quantile"]))
            values_cpu = values.cpu().numpy()
            spike_count = int((np.abs(values_cpu) > threshold).sum())
            row["cleanings"][variant_name] = {
                "tau": float(tau), "max_abs_eigenvalue": float(np.abs(values_cpu).max()),
                "null_max_median": float(np.median(null_maxima)),
                "null_max_threshold": threshold,
                "outlier_ratio": float(np.abs(values_cpu).max() / max(np.median(null_maxima), 1e-30)),
                "spike_count": spike_count, "effective_rank": effective_rank(values_cpu),
                "power_law": power_law(values_cpu, config["power_law_rank_start"],
                                        config["power_law_rank_stop"]),
                "null_power_law_alpha_median": float(np.median([item["alpha"] for item in null_power])),
                "null_power_law_r2_median": float(np.median([item["r_squared"] for item in null_power])),
                "top_eigenvalues": values_cpu[:16].tolist(),
            }
            key = safe_name(f"{name}_{variant_name}")
            saved[f"{key}_eigenvalues"] = values_cpu.astype(np.float32)
            saved[f"{key}_eigenvectors"] = vectors[:, :16].cpu().numpy().astype(np.float32)
            saved[f"{key}_tau"] = np.asarray(float(tau), dtype=np.float32)
        result[name] = row
        del raw, variants
    return result, saved


@torch.inference_mode()
def reconstruction_analysis(spectra, saved, directions, left, right, square,
                            inverse_square, config, input_archive, token_archive):
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([token_flags(tokenizer.decode([index]))
                      for index in range(tokenizer.vocab_size)])
    category_to_index = {name: index for index, name in enumerate(CATEGORIES)}
    splits = ("fineweb_test", "wikitext103", "tinystories")
    result = {}
    for name, row in spectra.items():
        if row["metadata"]["kind"] != "semantic":
            continue
        category = row["metadata"]["category"]
        key = safe_name(f"{name}_whitened")
        values = torch.from_numpy(saved[f"{key}_eigenvalues"][:config["top_eigenvectors"]]).cuda()
        vectors = torch.from_numpy(saved[f"{key}_eigenvectors"][:, :config["top_eigenvectors"]]).cuda()
        tau = float(saved[f"{key}_tau"])
        matrix = interaction_matrix(left, right, directions[name].cuda())
        whitened, _ = center_matrix(square @ matrix @ square)
        train_x = torch.from_numpy(input_archive["fineweb_train__input1"][:16384].astype(np.float32)).cuda()
        train_z = train_x @ inverse_square
        train_full = torch.einsum("bi,ij,bj->b", train_z, whitened, train_z)
        train_labels = flags[token_archive["fineweb_train__token_ids"][:16384],
                             category_to_index[category]]
        train_auc = auc(train_labels, train_full.cpu().numpy())
        orientation = 1. if train_auc >= .5 else -1.
        category_result = {"train_full_auc": max(train_auc, 1 - train_auc),
                           "orientation": orientation, "splits": {}}
        for split in splits:
            x = torch.from_numpy(input_archive[f"{split}__input1"].astype(np.float32)).cuda()
            z = x @ inverse_square
            full = torch.einsum("bi,ij,bj->b", z, whitened, z)
            projected = z @ vectors
            top = (projected.square() * values[None]).sum(1)
            full_np, top_np = full.cpu().numpy(), top.cpu().numpy()
            tail = np.abs(full_np) >= np.quantile(np.abs(full_np), .9)
            labels = flags[token_archive[f"{split}__token_ids"], category_to_index[category]]
            category_result["splits"][split] = {
                "full_auc": auc(labels, orientation * full_np),
                "top_auc": auc(labels, orientation * top_np),
                "correlation": correlation(full_np, top_np),
                "tail_correlation": correlation(full_np[tail], top_np[tail]),
                "relative_mse": float(np.square(full_np - top_np).sum() /
                                      max(np.square(full_np).sum(), 1e-30)),
            }
        result[name] = category_result
        del matrix, whitened, train_x, train_z
    return result


@torch.inference_mode()
def collect_sites(model, rows, directions, reconstruction, saved, inverse_square,
                  config, tokenizer):
    positions = torch.as_tensor(config["causal_selection"]["positions"], device="cuda")
    flags = np.stack([token_flags(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    category_to_index = {name: index for index, name in enumerate(CATEGORIES)}
    candidates = {name: [] for name, row in directions.items()
                  if row["metadata"]["seed"] == 101}
    for start in range(0, len(rows), 4):
        raw = np.asarray(rows[start:start + 4, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, _, _, normalized = prefix_to_layer1(model, ids)
        z = normalized[:, positions].float() @ inverse_square
        for name, row in candidates.items():
            info = directions[name]["metadata"]
            key = safe_name(f"{name}_whitened")
            values = torch.from_numpy(saved[f"{key}_eigenvalues"][:config["top_eigenvectors"]]).cuda()
            vectors = torch.from_numpy(saved[f"{key}_eigenvectors"][:, :config["top_eigenvectors"]]).cuda()
            projected = z @ vectors
            spike = (projected.square() * values).sum(-1)
            token_values = raw[:, config["causal_selection"]["positions"]]
            if info["kind"] == "semantic":
                labels = flags[token_values, category_to_index[info["category"]]].astype(bool)
                score = reconstruction[name]["orientation"] * spike.cpu().numpy()
                keep = labels
            else:
                score = spike.abs().cpu().numpy()
                keep = np.ones_like(score, dtype=bool)
            for batch_index, position_index in zip(*np.nonzero(keep)):
                candidates[name].append({
                    "row": start + int(batch_index),
                    "position": int(config["causal_selection"]["positions"][position_index]),
                    "score": float(score[batch_index, position_index]),
                    "token_id": int(token_values[batch_index, position_index]),
                })
    selected = {}
    for name, rows_for_name in candidates.items():
        rows_for_name.sort(key=lambda item: item["score"], reverse=True)
        selected[name] = rows_for_name[:config["causal_sites_per_direction"]]
    return selected


def fisher_fidelity(reference_logits, target_effect, predicted_effect):
    weights = F.softmax(reference_logits.float(), -1)
    target = target_effect.float() - (weights * target_effect.float()).sum()
    predicted = predicted_effect.float() - (weights * predicted_effect.float()).sum()
    energy = (weights * target.square()).sum().clamp_min(1e-30)
    fidelity = 1 - (weights * (predicted - target).square()).sum() / energy
    cosine = (weights * predicted * target).sum() / torch.sqrt(
        (weights * predicted.square()).sum() * energy).clamp_min(1e-30)
    return float(fidelity), float(cosine), float(energy)


@torch.inference_mode()
def causal_analysis(model, rows, sites, direction_rows, saved, inverse_square,
                    down, down_bias, config, tokenizer):
    result = {}
    for name, selected in sites.items():
        info = direction_rows[name]["metadata"]
        coefficients = direction_rows[name]["coefficients"].cuda()
        coefficient_norm_sq = coefficients.square().sum().clamp_min(1e-20)
        key = safe_name(f"{name}_whitened")
        values = torch.from_numpy(saved[f"{key}_eigenvalues"][:config["top_eigenvectors"]]).cuda()
        vectors = torch.from_numpy(saved[f"{key}_eigenvectors"][:, :config["top_eigenvectors"]]).cuda()
        tau = float(saved[f"{key}_tau"])
        records = []
        for site in selected:
            prefix = np.asarray(rows[site["row"], :site["position"] + 1], dtype=np.int64)
            ids = torch.as_tensor(prefix, device="cuda")[None]
            x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
            product = model.blocks[1].mlp.Left(normalized) * model.blocks[1].mlp.Right(normalized)
            native_write = F.linear(product, down.to(product.dtype),
                                    down_bias.to(product.dtype))
            x = normalized[0, -1].float()
            z = x @ inverse_square
            q = coefficients @ product[0, -1].float()
            isotropic = tau * z.square().sum()
            projected = z @ vectors
            spike = (projected.square() * values).sum()
            bulk = q - isotropic - spike
            target_q = {"isotropic": isotropic, "spike": isotropic + spike,
                        "bulk": isotropic + bulk}
            logits = {}
            native = suffix_from_layer1(model, x0, value_bus, pre + native_write)[0, -1]
            logits["native"] = native
            for variant, desired in target_q.items():
                changed_product = product.clone()
                changed_product[0, -1] += coefficients.to(product.dtype) * (
                    (desired - q) / coefficient_norm_sq).to(product.dtype)
                write = F.linear(changed_product, down.to(changed_product.dtype),
                                 down_bias.to(changed_product.dtype))
                logits[variant] = suffix_from_layer1(
                    model, x0, value_bus, pre + write)[0, -1]
            target_effect = logits["native"] - logits["isotropic"]
            spike_effect = logits["spike"] - logits["isotropic"]
            bulk_effect = logits["bulk"] - logits["isotropic"]
            spike_fidelity, spike_cosine, energy = fisher_fidelity(
                native, target_effect, spike_effect)
            bulk_fidelity, bulk_cosine, _ = fisher_fidelity(
                native, target_effect, bulk_effect)
            records.append({**site, "token": tokenizer.decode([site["token_id"]]),
                            "quadratic_identity_error": float(abs(q - (isotropic + spike + bulk))),
                            "spike_fidelity": spike_fidelity, "spike_cosine": spike_cosine,
                            "bulk_fidelity": bulk_fidelity, "bulk_cosine": bulk_cosine,
                            "target_effect_energy": energy})
        result[name] = {"metadata": info, "sites": records,
                        "median_spike_fidelity": float(np.median([r["spike_fidelity"] for r in records])) if records else float("nan"),
                        "median_bulk_fidelity": float(np.median([r["bulk_fidelity"] for r in records])) if records else float("nan"),
                        "median_effect_energy": float(np.median([r["target_effect_energy"] for r in records])) if records else float("nan")}
    return result


def main():
    config = json.loads(CONFIG.read_text())
    TENSOR_ROOT.mkdir(parents=True, exist_ok=True)
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    left = state["transformer.h.1.mlp.Left.weight"].float().cuda()
    right = state["transformer.h.1.mlp.Right.weight"].float().cuda()
    down = state["transformer.h.1.mlp.Down.weight"].float().cuda()
    down_bias = state["transformer.h.1.mlp.Down_bias"].float().cuda()
    del state
    token_archive = np.load(SPARSE / "natural_large.npz")
    h_train = torch.from_numpy(token_archive["fineweb_train__h1"].astype(np.float32))
    h_std = h_train.std(0).clamp_min(1e-4); del h_train
    directions, metadata = load_feature_directions(config, h_std, down)
    input_archive = np.load(SPARSE / "layer1_inputs_large.npz")
    x_train = torch.from_numpy(input_archive["fineweb_train__input1"].astype(np.float32))
    square, inverse_square, covariance = covariance_transforms(
        x_train, config["covariance_examples"])
    spectra, saved = spectral_analysis(
        directions, metadata, left, right, square, config)
    saved["covariance_square"] = square.cpu().numpy().astype(np.float32)
    saved["covariance_inverse_square"] = inverse_square.cpu().numpy().astype(np.float32)
    np.savez_compressed(ARRAYS, **saved)
    reconstruction = reconstruction_analysis(
        spectra, saved, directions, left, right, square, inverse_square,
        config, input_archive, token_archive)

    top_vectors = {name: torch.from_numpy(saved[
        f"{safe_name(name + '_whitened')}_eigenvectors"][:, :config["top_eigenvectors"]])
        for name in spectra}
    same, mismatched = [], []
    for category in config["strong_categories"]:
        first = top_vectors[f"semantic/seed101/{category}"]
        second = top_vectors[f"semantic/seed102/{category}"]
        same.append(subspace_overlap(first, second))
        for other in config["strong_categories"]:
            if other != category:
                mismatched.append(subspace_overlap(
                    first, top_vectors[f"semantic/seed102/{other}"]))
    stability = {"same_category": same, "mismatched_category": mismatched,
                 "same_median": float(np.median(same)),
                 "mismatched_median": float(np.median(mismatched))}

    model = load_full_model("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    spec = config["causal_selection"]
    selection_rows = all_rows[spec["start"]:spec["start"] + spec["rows"]]
    direction_rows = {name: {"metadata": metadata[name], "coefficients": value}
                      for name, value in directions.items()
                      if metadata[name]["seed"] == 101}
    sites = collect_sites(model, selection_rows, direction_rows, reconstruction,
                          saved, inverse_square, config, tokenizer)
    causal = causal_analysis(model, selection_rows, sites, direction_rows, saved,
                             inverse_square, down, down_bias, config, tokenizer)

    semantic_ratios = [spectra[name]["cleanings"]["whitened"]["outlier_ratio"]
                       for name in spectra if metadata[name]["kind"] == "semantic"]
    control_ratios = [spectra[name]["cleanings"]["whitened"]["outlier_ratio"]
                      for name in spectra if metadata[name]["kind"] == "control"]
    reconstruction_rows = [split for row in reconstruction.values()
                           for split in row["splits"].values()]
    top_auc = float(np.median([row["top_auc"] for row in reconstruction_rows]))
    full_auc = float(np.median([row["full_auc"] for row in reconstruction_rows]))
    semantic_causal = [row for row in causal.values() if row["metadata"]["kind"] == "semantic"]
    control_causal = [row for row in causal.values() if row["metadata"]["kind"] == "control"]
    causal_wins = sum(row["median_spike_fidelity"] > row["median_bulk_fidelity"]
                      for row in semantic_causal)
    semantic_spike_median = float(np.nanmedian([row["median_spike_fidelity"] for row in semantic_causal]))
    control_spike_median = float(np.nanmedian([row["median_spike_fidelity"] for row in control_causal]))
    hypotheses = {
        "H1_semantic_spikes": {"supported": bool(np.median(semantic_ratios) > 1.2
                               and np.median(semantic_ratios) - np.median(control_ratios) >= .1),
                               "semantic_median": float(np.median(semantic_ratios)),
                               "control_median": float(np.median(control_ratios))},
        "H2_cross_seed_subspaces": {"supported": bool(stability["same_median"] > .02
                                    and stability["same_median"] >= 3 * stability["mismatched_median"]),
                                    **stability},
        "H3_semantic_reconstruction": {"supported": bool(top_auc >= .70 and full_auc - top_auc <= .05),
                                       "top4_auc_median": top_auc,
                                       "full_auc_median": full_auc,
                                       "auc_loss": full_auc - top_auc},
        "H4_causal_spikes": {"supported": bool(causal_wins >= 4 and semantic_spike_median > .30),
                             "category_wins": causal_wins,
                             "semantic_spike_median": semantic_spike_median},
        "H5_semantic_control_advantage": {"supported": bool(semantic_spike_median
                                           - control_spike_median >= .10),
                                           "semantic_spike_median": semantic_spike_median,
                                           "control_spike_median": control_spike_median},
    }
    result = {"config": config, "covariance": covariance, "spectra": spectra,
              "reconstruction": reconstruction, "stability": stability,
              "causal": causal, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))

    lines = ["# Output-conditioned spectral-spike pilot", "", "## Hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Spectral spike pilot](spectral_spike_pilot.png)", "",
              "| Category | Seed 101 spike ratio | Seed 102 spike ratio | Top-4 overlap | FW full/top-4 AUC | Spike/bulk causal fidelity |",
              "|---|---:|---:|---:|---:|---:|"]
    for index, category in enumerate(config["strong_categories"]):
        first = f"semantic/seed101/{category}"; second = f"semantic/seed102/{category}"
        fw = reconstruction[first]["splits"]["fineweb_test"]
        cr = causal[first]
        lines.append(f"| {category} | {spectra[first]['cleanings']['whitened']['outlier_ratio']:.2f} "
                     f"| {spectra[second]['cleanings']['whitened']['outlier_ratio']:.2f} "
                     f"| {same[index]:.3f} | {fw['full_auc']:.3f}/{fw['top_auc']:.3f} "
                     f"| {cr['median_spike_fidelity']:.3f}/{cr['median_bulk_fidelity']:.3f} |")
    lines += ["", "The interaction form is exact for each SAE encoder preactivation before ReLU/Top-K. Input covariance whitening and removal of the isotropic trace component are fixed before eigendecomposition. Sign-flipped coefficient vectors provide a per-feature empirical spectral null.", "",
              "Matched controls are other SAE features selected to match encoder-coefficient and decoded-write norms. Causal interventions make the minimum-norm change to the native product vector needed to replace the chosen feature coordinate.", "",
              "Raw results: `spectral_spike_pilot.json`; eigensystems: `spectral_spike_pilot.npz`."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    xlabels = config["strong_categories"]
    axes[0].bar(np.arange(len(xlabels)) - .18,
                [spectra[f"semantic/seed101/{c}"]["cleanings"]["whitened"]["outlier_ratio"] for c in xlabels], .36, label="semantic")
    axes[0].bar(np.arange(len(xlabels)) + .18,
                [spectra[f"control/seed101/{c}"]["cleanings"]["whitened"]["outlier_ratio"] for c in xlabels], .36, label="matched control")
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("max eigenvalue / null median"); axes[0].legend()
    axes[1].bar(np.arange(len(xlabels)), same)
    axes[1].axhline(stability["mismatched_median"], color="black", linestyle="--", label="mismatched median")
    axes[1].set_ylabel("cross-seed top-4 overlap"); axes[1].legend()
    axes[2].bar(np.arange(len(xlabels)) - .18,
                [causal[f"semantic/seed101/{c}"]["median_spike_fidelity"] for c in xlabels], .36, label="spikes")
    axes[2].bar(np.arange(len(xlabels)) + .18,
                [causal[f"semantic/seed101/{c}"]["median_bulk_fidelity"] for c in xlabels], .36, label="bulk")
    axes[2].set_ylabel("downstream effect fidelity"); axes[2].legend()
    for axis in axes:
        axis.set_xticks(np.arange(len(xlabels)), xlabels, rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
