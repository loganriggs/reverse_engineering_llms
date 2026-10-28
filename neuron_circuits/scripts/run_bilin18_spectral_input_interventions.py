#!/usr/bin/env python3
"""True residual-eigenmode ablations and feature-path mediation tests."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_atlas import ARRAYS as ATLAS_ARRAYS, OUTPUT as ATLAS_OUTPUT
from run_bilin18_spectral_spike_pilot import (
    ARRAYS as PILOT_ARRAYS,
    OUTPUT as PILOT_OUTPUT,
    SPARSE,
    TENSOR_ROOT,
    fisher_fidelity,
    load_feature_directions,
    safe_name,
)
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_input_intervention.json")
OUTPUT = TENSOR_ROOT / "spectral_input_interventions.json"
REPORT = TENSOR_ROOT / "SPECTRAL_INPUT_INTERVENTIONS.md"
FIGURE = TENSOR_ROOT / "spectral_input_interventions.png"


def euclidean_fidelity(target, predicted):
    target = target.float()
    predicted = predicted.float()
    energy = target.square().sum().clamp_min(1e-30)
    fidelity = 1 - (predicted - target).square().sum() / energy
    cosine = (target * predicted).sum() / torch.sqrt(
        energy * predicted.square().sum()).clamp_min(1e-30)
    return float(fidelity), float(cosine), float(energy)


def arrays_for(category, known, pilot_arrays, atlas_arrays):
    return pilot_arrays if category in known else atlas_arrays


@torch.inference_mode()
def select_sites(model, rows, categories, known, directions, reports,
                 pilot_arrays, atlas_arrays, inverse_square, positions,
                 count, tokenizer):
    flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    category_index = {name: i for i, name in enumerate(CATEGORIES)}
    candidates = {category: [] for category in categories}
    positions_tensor = torch.as_tensor(positions, device="cuda")
    for start in range(0, len(rows), 4):
        raw = np.asarray(rows[start:start + 4, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, _, _, normalized = prefix_to_layer1(model, ids)
        product = model.blocks[1].mlp.Left(normalized[:, positions_tensor]) * model.blocks[1].mlp.Right(normalized[:, positions_tensor])
        z = normalized[:, positions_tensor].float() @ inverse_square
        token_ids = raw[:, positions]
        for category in categories:
            name = f"semantic/seed101/{category}"
            arrays = arrays_for(category, known, pilot_arrays, atlas_arrays)
            tau = float(arrays[f"{safe_name(name + '_whitened')}_tau"])
            full = product.float() @ directions[name].cuda() - tau * z.square().sum(-1)
            orientation = reports[category]
            score = orientation * full.cpu().numpy()
            labels = flags[token_ids, category_index[category]].astype(bool)
            for batch_index, position_index in zip(*np.nonzero(labels)):
                candidates[category].append({
                    "row": start + int(batch_index),
                    "position": int(positions[position_index]),
                    "score": float(score[batch_index, position_index]),
                    "token_id": int(token_ids[batch_index, position_index]),
                })
    for category in candidates:
        candidates[category].sort(key=lambda row: row["score"], reverse=True)
        candidates[category] = candidates[category][:count]
    return candidates


@torch.inference_mode()
def run_interventions(model, rows, sites, categories, known, directions,
                      decoders, pilot_arrays, atlas_arrays, square,
                      inverse_square, ranks, tokenizer):
    result = {}
    for category in categories:
        name = f"semantic/seed101/{category}"
        arrays = arrays_for(category, known, pilot_arrays, atlas_arrays)
        key = safe_name(name + "_whitened")
        vectors = torch.from_numpy(arrays[f"{key}_eigenvectors"][:, :16]).cuda()
        coefficient = directions[name].cuda()
        decoder = decoders[name].cuda()
        coefficient_norm_sq = coefficient.square().sum().clamp_min(1e-20)
        decoder_duality = coefficient @ decoder
        records = []
        for site in sites[category]:
            prefix = np.array(rows[site["row"], :site["position"] + 1], dtype=np.int64, copy=True)
            ids = torch.as_tensor(prefix, device="cuda")[None]
            x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
            product = model.blocks[1].mlp.Left(normalized) * model.blocks[1].mlp.Right(normalized)
            native_write = model.blocks[1].mlp.Down(product) + model.blocks[1].mlp.Down_bias
            native_logits = suffix_from_layer1(model, x0, value_bus, pre + native_write)[0, -1]
            x = normalized[0, -1].float()
            z = x @ inverse_square
            original_q = coefficient @ product[0, -1].float()
            site_result = {**site, "token": tokenizer.decode([site["token_id"]]), "ranks": {}}
            for rank in ranks:
                bases = {"top": vectors[:, :rank], "low": vectors[:, 16-rank:16]}
                deltas = {}
                for variant, basis in bases.items():
                    delta_z = -(z @ basis) @ basis.T
                    deltas[variant] = delta_z @ square
                top_norm = deltas["top"].norm()
                low_norm = deltas["low"].norm().clamp_min(1e-20)
                deltas["low"] = deltas["low"] * (top_norm / low_norm)
                rank_result = {}
                for variant, delta_x in deltas.items():
                    modified_x = x + delta_x
                    modified_product = product.clone()
                    modified_product[0, -1] = (
                        model.blocks[1].mlp.Left(modified_x)
                        * model.blocks[1].mlp.Right(modified_x))
                    actual_write = model.blocks[1].mlp.Down(modified_product) + model.blocks[1].mlp.Down_bias
                    actual_logits = suffix_from_layer1(model, x0, value_bus, pre + actual_write)[0, -1]
                    actual_effect = actual_logits - native_logits
                    changed_q = coefficient @ modified_product[0, -1].float()
                    delta_q = changed_q - original_q

                    product_minimum = product.clone()
                    product_minimum[0, -1] += coefficient * (delta_q / coefficient_norm_sq)
                    minimum_write = model.blocks[1].mlp.Down(product_minimum) + model.blocks[1].mlp.Down_bias
                    minimum_logits = suffix_from_layer1(model, x0, value_bus, pre + minimum_write)[0, -1]

                    product_decoder = product.clone()
                    product_decoder[0, -1] += decoder * (delta_q / decoder_duality)
                    decoder_write = model.blocks[1].mlp.Down(product_decoder) + model.blocks[1].mlp.Down_bias
                    decoder_logits = suffix_from_layer1(model, x0, value_bus, pre + decoder_write)[0, -1]

                    minimum_fidelity, minimum_cosine, effect_energy = fisher_fidelity(
                        actual_logits, actual_effect, minimum_logits - native_logits)
                    decoder_fidelity, decoder_cosine, _ = fisher_fidelity(
                        actual_logits, actual_effect, decoder_logits - native_logits)
                    direct_target = actual_write[0, -1] - native_write[0, -1]
                    minimum_direct = minimum_write[0, -1] - native_write[0, -1]
                    decoder_direct = decoder_write[0, -1] - native_write[0, -1]
                    minimum_direct_fidelity, minimum_direct_cosine, direct_energy = euclidean_fidelity(
                        direct_target, minimum_direct)
                    decoder_direct_fidelity, decoder_direct_cosine, _ = euclidean_fidelity(
                        direct_target, decoder_direct)
                    rank_result[variant] = {
                        "input_delta_norm": float(delta_x.norm()),
                        "feature_delta": float(delta_q),
                        "feature_change_per_input_norm": float(abs(delta_q) / delta_x.norm().clamp_min(1e-20)),
                        "downstream_effect_energy": effect_energy,
                        "direct_write_energy": direct_energy,
                        "minimum_dual": {
                            "downstream_fidelity": minimum_fidelity,
                            "downstream_cosine": minimum_cosine,
                            "direct_fidelity": minimum_direct_fidelity,
                            "direct_cosine": minimum_direct_cosine,
                        },
                        "sae_decoder": {
                            "downstream_fidelity": decoder_fidelity,
                            "downstream_cosine": decoder_cosine,
                            "direct_fidelity": decoder_direct_fidelity,
                            "direct_cosine": decoder_direct_cosine,
                        },
                    }
                site_result["ranks"][str(rank)] = rank_result
            records.append(site_result)
        summary = {}
        for rank in ranks:
            summary[str(rank)] = {}
            for variant in ("top", "low"):
                variants = [row["ranks"][str(rank)][variant] for row in records]
                summary[str(rank)][variant] = {
                    "median_feature_change_per_input_norm": float(np.median(
                        [row["feature_change_per_input_norm"] for row in variants])),
                    "median_downstream_effect_energy": float(np.median(
                        [row["downstream_effect_energy"] for row in variants])),
                    "median_minimum_dual_fidelity": float(np.median(
                        [row["minimum_dual"]["downstream_fidelity"] for row in variants])),
                    "median_sae_decoder_fidelity": float(np.median(
                        [row["sae_decoder"]["downstream_fidelity"] for row in variants])),
                    "median_minimum_direct_fidelity": float(np.median(
                        [row["minimum_dual"]["direct_fidelity"] for row in variants])),
                    "median_decoder_direct_fidelity": float(np.median(
                        [row["sae_decoder"]["direct_fidelity"] for row in variants])),
                }
        result[name] = {
            "group": "known" if category in known else "prospective",
            "decoder_duality": float(decoder_duality),
            "summary": summary,
            "sites": records,
        }
    return result


def main():
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(PILOT_OUTPUT.read_text())
    atlas = json.loads(ATLAS_OUTPUT.read_text())
    pilot_arrays = np.load(PILOT_ARRAYS)
    atlas_arrays = np.load(ATLAS_ARRAYS)
    square = torch.from_numpy(pilot_arrays["covariance_square"]).cuda()
    inverse_square = torch.from_numpy(pilot_arrays["covariance_inverse_square"]).cuda()
    known = config["known_categories"]
    prospective = config["prospective_categories"]
    categories = known + prospective
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
    sae = torch.load(SPARSE / "activation_sae_large_seed101.pt",
                     map_location="cpu", weights_only=True)
    decoder = sae["decoder.weight"].float()
    decoders = {}
    for category in categories:
        name = f"semantic/seed101/{category}"
        decoders[name] = h_std * decoder[:, metadata[name]["feature"]]
    del sae, decoder, h_std

    reports = {}
    for category in known:
        reports[category] = pilot["reconstruction"][f"semantic/seed101/{category}"]["orientation"]
    for category in prospective:
        reports[category] = atlas["reconstruction"][f"semantic/seed101/{category}"]["orientation"]

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    panel = config["panel"]
    rows = all_rows[panel["start"]:panel["start"] + panel["rows"]]
    sites = select_sites(model, rows, categories, set(known), directions, reports,
                         pilot_arrays, atlas_arrays, inverse_square, config["positions"],
                         panel["sites_per_category"], tokenizer)
    results = run_interventions(model, rows, sites, categories, set(known), directions,
                                decoders, pilot_arrays, atlas_arrays, square,
                                inverse_square, config["ranks"], tokenizer)

    def rows(group, rank, variant="top"):
        return [row["summary"][str(rank)][variant] for row in results.values()
                if row["group"] == group]
    selectivity_ratios = {}
    for rank in config["ranks"]:
        top = np.median([r["median_feature_change_per_input_norm"] for r in rows("known", rank)])
        low = np.median([r["median_feature_change_per_input_norm"] for r in rows("known", rank, "low")])
        selectivity_ratios[str(rank)] = float(top / max(low, 1e-30))
    top_energy = np.median([r["median_downstream_effect_energy"] for r in rows("known", 4)])
    low_energy = np.median([r["median_downstream_effect_energy"] for r in rows("known", 4, "low")])
    energy_ratio = float(top_energy / max(low_energy, 1e-30))
    known_rank4 = rows("known", 4)
    mediation_wins = sum(max(r["median_minimum_dual_fidelity"], r["median_sae_decoder_fidelity"]) >= .30
                         for r in known_rank4)
    minimum_known = float(np.median([r["median_minimum_dual_fidelity"] for r in known_rank4]))
    decoder_known = float(np.median([r["median_sae_decoder_fidelity"] for r in known_rank4]))
    known_best = float(np.median([max(r["median_minimum_dual_fidelity"], r["median_sae_decoder_fidelity"])
                                  for r in known_rank4]))
    prospective_rank4 = rows("prospective", 4)
    prospective_best = float(np.median([max(r["median_minimum_dual_fidelity"], r["median_sae_decoder_fidelity"])
                                        for r in prospective_rank4]))
    hypotheses = {
        "I1_feature_selectivity": {"supported": all(value >= 2 for value in selectivity_ratios.values()),
                                   "ratios": selectivity_ratios},
        "I2_downstream_specificity": {"supported": energy_ratio >= 1.5,
                                      "top_to_low_energy_ratio": energy_ratio},
        "I3_feature_mediation": {"supported": mediation_wins >= 4,
                                 "known_category_wins": mediation_wins},
        "I4_decoder_advantage": {"supported": decoder_known - minimum_known >= .10,
                                 "decoder_median": decoder_known,
                                 "minimum_dual_median": minimum_known,
                                 "advantage": decoder_known - minimum_known},
        "I5_prospective_gap": {"supported": known_best - prospective_best >= .10,
                               "known_best_median": known_best,
                               "prospective_best_median": prospective_best,
                               "gap": known_best - prospective_best},
    }
    output = {"config": config, "sites": sites, "results": results,
              "hypotheses": hypotheses,
              "intervention_definition": "ablate residual eigencomponents at the layer-1 post-RMSNorm MLP input; recompute every native product channel and the full suffix"}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = ["# True spectral input-subspace interventions", "",
             "Unlike the earlier coordinate test, this experiment removes eigenvector components from the actual post-RMSNorm residual input, recomputes every bilinear product channel, and runs the complete downstream model. It then asks whether the effect is mediated by the proposed SAE feature coordinate.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Input interventions](spectral_input_interventions.png)", "",
              "| Category | Group | Rank-4 feature selectivity | Effect energy | Minimum-dual mediation | SAE-decoder mediation |",
              "|---|---|---:|---:|---:|---:|"]
    for category in categories:
        name = f"semantic/seed101/{category}"
        row = results[name]; top = row["summary"]["4"]["top"]; low = row["summary"]["4"]["low"]
        ratio = top["median_feature_change_per_input_norm"] / max(low["median_feature_change_per_input_norm"], 1e-30)
        lines.append(f"| {category} | {row['group']} | {ratio:.2f}× | {top['median_downstream_effect_energy']:.3g} | {top['median_minimum_dual_fidelity']:.3f} | {top['median_sae_decoder_fidelity']:.3f} |")
    lines += ["", "Low-mode controls use modes 13–16 at rank 4 and are rescaled per example to exactly match the top-mode input-edit norm. Computation is float32 to avoid the bfloat16 coordinate quantization found in the precision red-team."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = categories; x = np.arange(len(labels))
    min_values = [results[f"semantic/seed101/{c}"]["summary"]["4"]["top"]["median_minimum_dual_fidelity"] for c in labels]
    dec_values = [results[f"semantic/seed101/{c}"]["summary"]["4"]["top"]["median_sae_decoder_fidelity"] for c in labels]
    axes[0].bar(x - .18, min_values, .36, label="minimum dual")
    axes[0].bar(x + .18, dec_values, .36, label="SAE decoder")
    axes[0].set_xticks(x, labels, rotation=45, ha="right"); axes[0].set_ylabel("downstream mediation fidelity")
    axes[0].legend(fontsize=8)
    top_energies = [results[f"semantic/seed101/{c}"]["summary"]["4"]["top"]["median_downstream_effect_energy"] for c in labels]
    low_energies = [results[f"semantic/seed101/{c}"]["summary"]["4"]["low"]["median_downstream_effect_energy"] for c in labels]
    axes[1].bar(x - .18, top_energies, .36, label="top modes")
    axes[1].bar(x + .18, low_energies, .36, label="norm-matched low modes")
    axes[1].set_yscale("log"); axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_ylabel("downstream Fisher effect energy"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
