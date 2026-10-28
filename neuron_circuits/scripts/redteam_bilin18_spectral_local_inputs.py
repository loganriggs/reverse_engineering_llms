#!/usr/bin/env python3
"""Local, norm-controlled raw/whitened spectral input intervention red-team."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_input_interventions import (
    CONFIG as FULL_CONFIG,
    OUTPUT as FULL_OUTPUT,
    arrays_for,
    euclidean_fidelity,
)
from run_bilin18_spectral_atlas import ARRAYS as ATLAS_ARRAYS, OUTPUT as ATLAS_OUTPUT
from run_bilin18_spectral_spike_pilot import (
    ARRAYS as PILOT_ARRAYS,
    SPARSE,
    TENSOR_ROOT,
    fisher_fidelity,
    load_feature_directions,
    safe_name,
)
from audit_bilin18_mlp_discriminators import SNAPSHOT
from train_bilin18_causal_cp import prefix_to_layer1, suffix_from_layer1


CONFIG = Path("configs/bilin18_spectral_local_input_redteam.json")
OUTPUT = TENSOR_ROOT / "spectral_local_input_redteam.json"
REPORT = TENSOR_ROOT / "SPECTRAL_LOCAL_INPUT_REDTEAM.md"
FIGURE = TENSOR_ROOT / "spectral_local_input_redteam.png"


@torch.inference_mode()
def evaluate(model, rows, sites, categories, known, directions, decoders,
             pilot_arrays, atlas_arrays, square, inverse_square, config):
    output = {}
    rank = config["rank"]
    for category in categories:
        name = f"semantic/seed101/{category}"
        arrays = arrays_for(category, known, pilot_arrays, atlas_arrays)
        coefficient = directions[name].cuda()
        decoder = decoders[name].cuda()
        coefficient_norm_sq = coefficient.square().sum().clamp_min(1e-20)
        decoder_duality = (coefficient @ decoder).clamp(min=-1e30, max=1e30)
        records = []
        for site in sites[category]:
            prefix = np.array(rows[site["row"], :site["position"] + 1], dtype=np.int64, copy=True)
            ids = torch.as_tensor(prefix, device="cuda")[None]
            x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
            product = model.blocks[1].mlp.Left(normalized) * model.blocks[1].mlp.Right(normalized)
            native_write = model.blocks[1].mlp.Down(product) + model.blocks[1].mlp.Down_bias
            native_logits = suffix_from_layer1(model, x0, value_bus, pre + native_write)[0, -1]
            x = normalized[0, -1].float()
            original_q = coefficient @ product[0, -1].float()
            site_row = {**site, "input_norm": float(x.norm()), "bases": {}}
            for basis_name in config["bases"]:
                key = safe_name(f"{name}_{basis_name}")
                vectors = torch.from_numpy(arrays[f"{key}_eigenvectors"][:, :16]).cuda()
                if basis_name == "whitened":
                    z = x @ inverse_square
                    def direction(v):
                        return (-(z @ v) @ v.T) @ square
                else:
                    def direction(v):
                        return -(x @ v) @ v.T
                base_deltas = {"top": direction(vectors[:, :rank]),
                               "low": direction(vectors[:, 16-rank:16])}
                basis_row = {}
                for fraction in config["input_norm_fractions"]:
                    fraction_row = {}
                    target_norm = fraction * x.norm()
                    for variant, base_delta in base_deltas.items():
                        delta_x = base_delta * (target_norm / base_delta.norm().clamp_min(1e-20))
                        modified_x = x + delta_x
                        modified_product = product.clone()
                        modified_product[0, -1] = model.blocks[1].mlp.Left(modified_x) * model.blocks[1].mlp.Right(modified_x)
                        actual_write = model.blocks[1].mlp.Down(modified_product) + model.blocks[1].mlp.Down_bias
                        actual_logits = suffix_from_layer1(model, x0, value_bus, pre + actual_write)[0, -1]
                        actual_effect = actual_logits - native_logits
                        delta_q = coefficient @ modified_product[0, -1].float() - original_q

                        product_minimum = product.clone()
                        product_minimum[0, -1] += coefficient * (delta_q / coefficient_norm_sq)
                        minimum_write = model.blocks[1].mlp.Down(product_minimum) + model.blocks[1].mlp.Down_bias
                        minimum_logits = suffix_from_layer1(model, x0, value_bus, pre + minimum_write)[0, -1]
                        product_decoder = product.clone()
                        product_decoder[0, -1] += decoder * (delta_q / decoder_duality)
                        decoder_write = model.blocks[1].mlp.Down(product_decoder) + model.blocks[1].mlp.Down_bias
                        decoder_logits = suffix_from_layer1(model, x0, value_bus, pre + decoder_write)[0, -1]

                        min_fidelity, min_cosine, energy = fisher_fidelity(
                            actual_logits, actual_effect, minimum_logits - native_logits)
                        dec_fidelity, dec_cosine, _ = fisher_fidelity(
                            actual_logits, actual_effect, decoder_logits - native_logits)
                        min_direct, _, direct_energy = euclidean_fidelity(
                            actual_write[0, -1] - native_write[0, -1],
                            minimum_write[0, -1] - native_write[0, -1])
                        dec_direct, _, _ = euclidean_fidelity(
                            actual_write[0, -1] - native_write[0, -1],
                            decoder_write[0, -1] - native_write[0, -1])
                        fraction_row[variant] = {
                            "input_delta_norm": float(delta_x.norm()),
                            "feature_delta": float(delta_q),
                            "feature_change_per_input_norm": float(abs(delta_q) / delta_x.norm()),
                            "downstream_effect_energy": energy,
                            "direct_write_energy": direct_energy,
                            "minimum_dual_fidelity": min_fidelity,
                            "minimum_dual_cosine": min_cosine,
                            "sae_decoder_fidelity": dec_fidelity,
                            "sae_decoder_cosine": dec_cosine,
                            "minimum_direct_fidelity": min_direct,
                            "decoder_direct_fidelity": dec_direct,
                        }
                    basis_row[str(fraction)] = fraction_row
                site_row["bases"][basis_name] = basis_row
            records.append(site_row)
        summary = {}
        for basis_name in config["bases"]:
            summary[basis_name] = {}
            for fraction in config["input_norm_fractions"]:
                summary[basis_name][str(fraction)] = {}
                for variant in ("top", "low"):
                    selected = [row["bases"][basis_name][str(fraction)][variant] for row in records]
                    summary[basis_name][str(fraction)][variant] = {
                        key: float(np.median([row[key] for row in selected]))
                        for key in (
                            "feature_change_per_input_norm", "downstream_effect_energy",
                            "minimum_dual_fidelity", "sae_decoder_fidelity",
                            "minimum_direct_fidelity", "decoder_direct_fidelity")
                    }
        output[name] = {"group": "known" if category in known else "prospective",
                        "summary": summary, "sites": records}
    return output


def main():
    config = json.loads(CONFIG.read_text())
    full_config = json.loads(FULL_CONFIG.read_text())
    full = json.loads(FULL_OUTPUT.read_text())
    atlas = json.loads(ATLAS_OUTPUT.read_text())
    pilot_arrays = np.load(PILOT_ARRAYS); atlas_arrays = np.load(ATLAS_ARRAYS)
    square = torch.from_numpy(pilot_arrays["covariance_square"]).cuda()
    inverse_square = torch.from_numpy(pilot_arrays["covariance_inverse_square"]).cuda()
    known = full_config["known_categories"]; prospective = full_config["prospective_categories"]
    categories = known + prospective; all_categories = atlas["config"]["reference_categories"] + atlas["config"]["prospective_categories"]
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    down = state["transformer.h.1.mlp.Down.weight"].float().cuda(); del state
    archive = np.load(SPARSE / "natural_large.npz")
    h_std = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32)).std(0).clamp_min(1e-4); del archive
    directions, metadata = load_feature_directions(
        {"strong_categories": all_categories, "sae_seeds": atlas["config"]["feature_seeds"], "seed": atlas["config"]["seed"]}, h_std, down)
    sae = torch.load(SPARSE / "activation_sae_large_seed101.pt", map_location="cpu", weights_only=True)
    decoder = sae["decoder.weight"].float(); decoders = {}
    for category in categories:
        name = f"semantic/seed101/{category}"
        decoders[name] = h_std * decoder[:, metadata[name]["feature"]]
    del sae, decoder, h_std
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    panel = full_config["panel"]; all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    rows = all_rows[panel["start"]:panel["start"] + panel["rows"]]
    results = evaluate(model, rows, full["sites"], categories, set(known), directions,
                       decoders, pilot_arrays, atlas_arrays, square, inverse_square, config)

    fraction = str(config["primary_fraction"])
    basis_metrics = {}
    for basis in config["bases"]:
        known_rows = [row["summary"][basis][fraction] for row in results.values() if row["group"] == "known"]
        prospective_rows = [row["summary"][basis][fraction] for row in results.values() if row["group"] == "prospective"]
        feature_ratio = float(np.median([r["top"]["feature_change_per_input_norm"] for r in known_rows]) /
                              max(np.median([r["low"]["feature_change_per_input_norm"] for r in known_rows]), 1e-30))
        energy_ratio = float(np.median([r["top"]["downstream_effect_energy"] for r in known_rows]) /
                             max(np.median([r["low"]["downstream_effect_energy"] for r in known_rows]), 1e-30))
        known_best_each = [max(r["top"]["minimum_dual_fidelity"], r["top"]["sae_decoder_fidelity"]) for r in known_rows]
        known_low_best_each = [max(r["low"]["minimum_dual_fidelity"], r["low"]["sae_decoder_fidelity"]) for r in known_rows]
        prospective_best_each = [max(r["top"]["minimum_dual_fidelity"], r["top"]["sae_decoder_fidelity"]) for r in prospective_rows]
        known_minimum = float(np.median([r["top"]["minimum_dual_fidelity"] for r in known_rows]))
        known_decoder = float(np.median([r["top"]["sae_decoder_fidelity"] for r in known_rows]))
        basis_metrics[basis] = {
            "feature_selectivity_ratio": feature_ratio,
            "effect_energy_ratio": energy_ratio,
            "known_mediation_wins": sum(value >= .30 for value in known_best_each),
            "known_best_median": float(np.median(known_best_each)),
            "known_low_best_median": float(np.median(known_low_best_each)),
            "top_minus_low_mediation": float(np.median(known_best_each) - np.median(known_low_best_each)),
            "known_minimum_dual_median": known_minimum,
            "known_sae_decoder_median": known_decoder,
            "decoder_advantage": known_decoder - known_minimum,
            "prospective_best_median": float(np.median(prospective_best_each)),
            "prospective_gap": float(np.median(known_best_each) - np.median(prospective_best_each)),
        }
    hypotheses = {
        "L1_local_feature_selectivity": {"supported": max(r["feature_selectivity_ratio"] for r in basis_metrics.values()) >= 2, "bases": basis_metrics},
        "L2_local_downstream_specificity": {"supported": max(r["effect_energy_ratio"] for r in basis_metrics.values()) >= 1.5, "bases": basis_metrics},
        "L3_local_feature_mediation": {"supported": max(r["known_mediation_wins"] for r in basis_metrics.values()) >= 4, "bases": basis_metrics},
        "L4_basis_robustness": {"supported": abs(basis_metrics["raw"]["known_best_median"] - basis_metrics["whitened"]["known_best_median"]) <= .10,
                                "absolute_difference": abs(basis_metrics["raw"]["known_best_median"] - basis_metrics["whitened"]["known_best_median"])},
        "L5_local_prospective_gap": {"supported": max(r["prospective_gap"] for r in basis_metrics.values()) >= .10, "bases": basis_metrics},
    }
    output = {"config": config, "results": results, "basis_metrics": basis_metrics,
              "hypotheses": hypotheses,
              "control": "top and low spectral directions are separately normalized to the same fixed fraction of each example's input norm"}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = ["# Local spectral-input intervention red-team", "",
             "The full-component ablation sometimes moved the MLP input by several times its original norm. This red-team instead makes top- and low-mode edits with exactly 5%, 10%, or 25% of the input norm, and compares raw Euclidean with covariance-whitened spectral bases.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Local input red-team](spectral_local_input_redteam.png)", "",
              "| Basis | Feature selectivity | Effect-energy ratio | Known top/low mediation | Decoder advantage | Prospective best median |",
              "|---|---:|---:|---:|---:|---:|"]
    for basis, row in basis_metrics.items():
        lines.append(f"| {basis} | {row['feature_selectivity_ratio']:.2f}× | {row['effect_energy_ratio']:.2f}× | {row['known_best_median']:.3f}/{row['known_low_best_median']:.3f} | {row['decoder_advantage']:.3f} | {row['prospective_best_median']:.3f} |")
    lines += ["", "The primary table uses 10%-of-input-norm edits. Complete 5% and 25% results and both direct-write and downstream mediation scores are in `spectral_local_input_redteam.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = categories; x = np.arange(len(labels)); fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for offset, basis in zip((-.18, .18), config["bases"]):
        values = [max(results[f"semantic/seed101/{c}"]["summary"][basis][fraction]["top"]["minimum_dual_fidelity"],
                      results[f"semantic/seed101/{c}"]["summary"][basis][fraction]["top"]["sae_decoder_fidelity"]) for c in labels]
        axes[0].bar(x + offset, values, .36, label=basis)
    axes[0].set_xticks(x, labels, rotation=45, ha="right"); axes[0].set_ylabel("best feature-path mediation fidelity"); axes[0].legend(fontsize=8)
    for basis in config["bases"]:
        axes[1].plot(config["input_norm_fractions"], [float(np.median([
            max(row["summary"][basis][str(f)]["top"]["minimum_dual_fidelity"], row["summary"][basis][str(f)]["top"]["sae_decoder_fidelity"])
            for row in results.values() if row["group"] == "known"])) for f in config["input_norm_fractions"]], marker="o", label=basis)
    axes[1].set_xlabel("input edit / input norm"); axes[1].set_ylabel("known-category median mediation"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
