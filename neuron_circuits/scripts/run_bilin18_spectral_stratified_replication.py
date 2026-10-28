#!/usr/bin/env python3
"""Large stratified FineWeb/OOD replication of local spectral input paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_atlas import ARRAYS as ATLAS_ARRAYS, OUTPUT as ATLAS_OUTPUT
from run_bilin18_spectral_input_interventions import arrays_for
from redteam_bilin18_spectral_local_inputs import evaluate
from run_bilin18_spectral_spike_pilot import (
    ARRAYS as PILOT_ARRAYS,
    OUTPUT as PILOT_OUTPUT,
    SPARSE,
    TENSOR_ROOT,
    load_feature_directions,
    safe_name,
)
from train_bilin18_causal_cp import prefix_to_layer1


CONFIG = Path("configs/bilin18_spectral_stratified_replication.json")
OUTPUT = TENSOR_ROOT / "spectral_stratified_replication.json"
REPORT = TENSOR_ROOT / "SPECTRAL_STRATIFIED_REPLICATION.md"
FIGURE = TENSOR_ROOT / "spectral_stratified_replication.png"


def stable_rng(seed, text):
    offset = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed + offset)


@torch.inference_mode()
def collect_sites(model, rows, categories, known, directions, orientations,
                  pilot_arrays, atlas_arrays, inverse_square, config, tokenizer):
    flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    category_index = {name: i for i, name in enumerate(CATEGORIES)}
    candidates = {category: {"positive": [], "negative": []} for category in categories}
    positions = config["positions"]
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
            score = orientations[category] * (
                product.float() @ directions[name].cuda() - tau * z.square().sum(-1))
            labels = flags[token_ids, category_index[category]].astype(bool)
            score = score.cpu().numpy()
            for batch_index in range(len(raw)):
                for position_index, position in enumerate(positions):
                    label = "positive" if labels[batch_index, position_index] else "negative"
                    candidates[category][label].append({
                        "row": start + batch_index, "position": int(position),
                        "score": float(score[batch_index, position_index]),
                        "token_id": int(token_ids[batch_index, position_index]),
                    })
    selected, excluded = {}, {}
    size = config["stratum_size"]
    for category, pools in candidates.items():
        positives = sorted(pools["positive"], key=lambda row: row["score"], reverse=True)
        negatives = sorted(pools["negative"], key=lambda row: row["score"], reverse=True)
        if len(positives) < config["minimum_positives"]:
            excluded[category] = {"positives": len(positives), "reason": "below minimum_positives"}
            continue
        high_positive = positives[:size]
        hard_negative = negatives[:size]
        positive_pool = positives[size:]
        negative_pool = negatives[size:]
        rng = stable_rng(config["seed"], category)
        random_positive = [positive_pool[i] for i in rng.choice(len(positive_pool), size=size, replace=False)]
        random_negative = [negative_pool[i] for i in rng.choice(len(negative_pool), size=size, replace=False)]
        strata = (("high_positive", high_positive), ("random_positive", random_positive),
                  ("hard_negative", hard_negative), ("random_negative", random_negative))
        selected[category] = []
        for stratum, records in strata:
            selected[category].extend([{**row, "stratum": stratum} for row in records])
    return selected, excluded


def summarize_strata(results, bases, fraction):
    output = {}
    for basis in bases:
        output[basis] = {}
        for group in ("known", "prospective"):
            output[basis][group] = {}
            for stratum in ("high_positive", "random_positive", "hard_negative", "random_negative"):
                category_rows = []
                for row in results.values():
                    if row["group"] != group:
                        continue
                    sites = [site["bases"][basis][fraction] for site in row["sites"]
                             if site["stratum"] == stratum]
                    if not sites:
                        continue
                    category_rows.append({variant: {
                        key: float(np.median([site[variant][key] for site in sites]))
                        for key in ("feature_change_per_input_norm", "downstream_effect_energy",
                                    "sae_decoder_fidelity", "minimum_dual_fidelity")}
                        for variant in ("top", "low")})
                summary_row = {"eligible_categories": len(category_rows)}
                for variant in ("top", "low"):
                    summary_row[variant] = {
                        key: float(np.median([row[variant][key] for row in category_rows]))
                        for key in category_rows[0][variant]
                    } if category_rows else {}
                output[basis][group][stratum] = summary_row
    return output


def main():
    config = json.loads(CONFIG.read_text())
    pilot = json.loads(PILOT_OUTPUT.read_text()); atlas = json.loads(ATLAS_OUTPUT.read_text())
    pilot_arrays = np.load(PILOT_ARRAYS); atlas_arrays = np.load(ATLAS_ARRAYS)
    square = torch.from_numpy(pilot_arrays["covariance_square"]).cuda()
    inverse_square = torch.from_numpy(pilot_arrays["covariance_inverse_square"]).cuda()
    known = config["known_categories"]; prospective = config["prospective_categories"]
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
    orientations = {}
    for category in known: orientations[category] = pilot["reconstruction"][f"semantic/seed101/{category}"]["orientation"]
    for category in prospective: orientations[category] = atlas["reconstruction"][f"semantic/seed101/{category}"]["orientation"]
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)

    corpora = {}
    local_config = {"rank": config["rank"], "bases": config["bases"],
                    "input_norm_fractions": [config["input_norm_fraction"]]}
    for corpus, specification in config["corpora"].items():
        print("selecting", corpus, flush=True)
        path = DEFAULT_CORPUS if specification["path"] == "default_fineweb" else Path(specification["path"])
        all_rows = np.load(path, mmap_mode="r")
        rows = all_rows[specification["start"]:specification["start"] + specification["rows"]]
        sites, excluded = collect_sites(model, rows, categories, set(known), directions,
                                        orientations, pilot_arrays, atlas_arrays,
                                        inverse_square, config, tokenizer)
        eligible = list(sites)
        print("intervening", corpus, "categories", eligible, flush=True)
        results = evaluate(model, rows, sites, eligible, set(known), directions,
                           decoders, pilot_arrays, atlas_arrays, square,
                           inverse_square, local_config)
        summaries = summarize_strata(results, config["bases"], str(config["input_norm_fraction"]))
        corpora[corpus] = {"excluded": excluded, "sites": sites,
                           "results": results, "summaries": summaries}

    fraction = str(config["input_norm_fraction"])
    fineweb_checks = []
    for basis in config["bases"]:
        for stratum in ("high_positive", "random_positive"):
            row = corpora["fineweb"]["summaries"][basis]["known"][stratum]
            fineweb_checks.append(row["top"]["sae_decoder_fidelity"] >= .30 and
                                  row["top"]["sae_decoder_fidelity"] - row["low"]["sae_decoder_fidelity"] >= .20)
    ood_checks = {}
    for corpus in ("wikitext103", "tinystories"):
        ood_checks[corpus] = any(
            (lambda row: row["top"]["sae_decoder_fidelity"] >= .25 and
             row["top"]["sae_decoder_fidelity"] - row["low"]["sae_decoder_fidelity"] >= .15)(
                corpora[corpus]["summaries"][basis]["known"]["random_positive"])
            for basis in config["bases"])

    selectivity = {}
    for basis in config["bases"]:
        feature_top, feature_low, energy_top, energy_low = [], [], [], []
        for corpus in corpora.values():
            for stratum in ("high_positive", "random_positive"):
                row = corpus["summaries"][basis]["known"][stratum]
                feature_top.append(row["top"]["feature_change_per_input_norm"]); feature_low.append(row["low"]["feature_change_per_input_norm"])
                energy_top.append(row["top"]["downstream_effect_energy"]); energy_low.append(row["low"]["downstream_effect_energy"])
        selectivity[basis] = {"feature_ratio": float(np.median(feature_top) / max(np.median(feature_low), 1e-30)),
                              "energy_ratio": float(np.median(energy_top) / max(np.median(energy_low), 1e-30))}
    positive_specificity, known_gap, high_bias = {}, {}, {}
    for basis in config["bases"]:
        known_random, known_hard, prospective_positive, known_high = [], [], [], []
        for corpus in corpora.values():
            summaries = corpus["summaries"][basis]
            known_random.append(summaries["known"]["random_positive"]["top"]["sae_decoder_fidelity"])
            known_hard.append(summaries["known"]["hard_negative"]["top"]["sae_decoder_fidelity"])
            known_high.append(summaries["known"]["high_positive"]["top"]["sae_decoder_fidelity"])
            if summaries["prospective"]["random_positive"]["eligible_categories"]:
                prospective_positive.append(summaries["prospective"]["random_positive"]["top"]["sae_decoder_fidelity"])
        positive_specificity[basis] = float(np.median(known_random) - np.median(known_hard))
        known_gap[basis] = float(np.median(known_random) - np.median(prospective_positive))
        high_bias[basis] = float(np.median(known_high) - np.median(known_random))
    hypotheses = {
        "S1_fineweb_replication": {"supported": all(fineweb_checks), "checks": fineweb_checks},
        "S2_ood_positive_transfer": {"supported": all(ood_checks.values()), "corpora": ood_checks},
        "S3_local_selectivity": {"supported": all(row["feature_ratio"] >= 2 and row["energy_ratio"] >= 1.5 for row in selectivity.values()), "bases": selectivity},
        "S4_positive_specificity": {"supported": max(positive_specificity.values()) >= .10, "bases": positive_specificity},
        "S5_known_prospective_gap": {"supported": max(known_gap.values()) >= .10, "bases": known_gap},
        "S6_high_example_bias": {"supported": all(value <= .15 for value in high_bias.values()), "high_minus_random": high_bias},
    }
    output = {"config": config, "corpora": corpora, "selectivity": selectivity,
              "positive_specificity": positive_specificity, "known_prospective_gap": known_gap,
              "high_example_bias": high_bias, "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = ["# Stratified local spectral-path replication", "",
             "Each eligible category contributes 16 high positives, 16 random positives, 16 hard negatives, and 16 random negatives per corpus. Interventions are rank-4, float32, and exactly 10% of the MLP-input norm.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Stratified replication](spectral_stratified_replication.png)", "",
              "| Corpus | Basis | Group | High positive | Random positive | Hard negative | Random negative |",
              "|---|---|---|---:|---:|---:|---:|"]
    for corpus, corpus_row in corpora.items():
        for basis in config["bases"]:
            for group in ("known", "prospective"):
                summary = corpus_row["summaries"][basis][group]
                values = [summary[s]["top"].get("sae_decoder_fidelity", float("nan")) for s in ("high_positive", "random_positive", "hard_negative", "random_negative")]
                lines.append(f"| {corpus} | {basis} | {group} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines += ["", "Values are median SAE-decoder mediation fidelity across category medians. TinyStories categories with fewer than 32 positives are excluded and disclosed in the JSON."]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7)); strata = ["high_positive", "random_positive", "hard_negative", "random_negative"]
    x = np.arange(len(strata)); width = .12
    offset = -2.5 * width
    for corpus in corpora:
        for basis in config["bases"]:
            summary = corpora[corpus]["summaries"][basis]["known"]
            values = [summary[s]["top"]["sae_decoder_fidelity"] for s in strata]
            axes[0].bar(x + offset, values, width, label=f"{corpus}/{basis}"); offset += width
    axes[0].set_xticks(x, strata, rotation=20, ha="right"); axes[0].set_ylabel("known mediation fidelity"); axes[0].legend(fontsize=7)
    for basis in config["bases"]:
        axes[1].scatter([selectivity[basis]["feature_ratio"]], [selectivity[basis]["energy_ratio"]], s=100, label=basis)
    axes[1].axvline(2, color="grey", linestyle="--"); axes[1].axhline(1.5, color="grey", linestyle="--")
    axes[1].set_xlabel("top/low feature-change ratio"); axes[1].set_ylabel("top/low effect-energy ratio"); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
