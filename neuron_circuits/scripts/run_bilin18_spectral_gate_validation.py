#!/usr/bin/env python3
"""Fresh-panel validation stratified by the SAE's actual Top-K gate state."""

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
from discover_bilin18_weight_only_circuits import CATEGORIES, token_flags
from redteam_bilin18_spectral_local_inputs import evaluate
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_spectral_atlas import ARRAYS as ATLAS_ARRAYS, OUTPUT as ATLAS_OUTPUT
from run_bilin18_spectral_spike_pilot import (
    ARRAYS as PILOT_ARRAYS,
    SPARSE,
    TENSOR_ROOT,
    load_feature_directions,
)
from train_bilin18_causal_cp import prefix_to_layer1


CONFIG = Path("configs/bilin18_spectral_gate_validation.json")
OUTPUT = TENSOR_ROOT / "spectral_gate_validation.json"
REPORT = TENSOR_ROOT / "SPECTRAL_GATE_VALIDATION.md"
FIGURE = TENSOR_ROOT / "spectral_gate_validation.png"


def stable_rng(seed, text):
    return np.random.default_rng(seed + int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))


@torch.inference_mode()
def collect_gate_sites(model, rows, categories, feature_indices, encoder_weight,
                       encoder_bias, mean, std, config, tokenizer):
    flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    category_index = {name: i for i, name in enumerate(CATEGORIES)}
    candidates = {category: {"active": [], "inactive": []} for category in categories}
    positions = config["positions"]
    positions_tensor = torch.as_tensor(positions, device="cuda")
    for start in range(0, len(rows), 8):
        raw = np.asarray(rows[start:start + 8, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        _, _, _, normalized = prefix_to_layer1(model, ids)
        product = model.blocks[1].mlp.Left(normalized[:, positions_tensor]) * model.blocks[1].mlp.Right(normalized[:, positions_tensor])
        flat = product.float().flatten(0, 1)
        preactivation = F.linear((flat - mean) / std, encoder_weight, encoder_bias)
        positive = F.relu(preactivation)
        top = torch.topk(positive, 128, dim=1).indices
        token_ids = raw[:, positions].reshape(-1)
        for category in categories:
            feature = feature_indices[category]
            active = (top == feature).any(1) & (preactivation[:, feature] > 0)
            values = preactivation[:, feature]
            surface = flags[token_ids, category_index[category]].astype(bool)
            for flat_index in range(len(flat)):
                batch_index, position_index = divmod(flat_index, len(positions))
                state = "active" if bool(active[flat_index]) else "inactive"
                candidates[category][state].append({
                    "row": start + batch_index,
                    "position": int(positions[position_index]),
                    "score": float(values[flat_index]),
                    "token_id": int(token_ids[flat_index]),
                    "surface_label": bool(surface[flat_index]),
                    "feature_active": bool(active[flat_index]),
                    "feature_preactivation": float(values[flat_index]),
                })
        print("gate scan", min(start + 8, len(rows)), "/", len(rows), flush=True) if start % 256 == 0 else None
    selected, excluded = {}, {}
    size = config["stratum_size"]
    for category, pools in candidates.items():
        active = sorted(pools["active"], key=lambda row: row["score"], reverse=True)
        inactive = sorted(pools["inactive"], key=lambda row: row["score"], reverse=True)
        if len(active) < 2 * size or len(inactive) < 2 * size:
            excluded[category] = {"active": len(active), "inactive": len(inactive)}
            continue
        high_active = active[:size]
        near_inactive = inactive[:size]
        rng = stable_rng(config["seed"], category)
        random_active = [active[size:][i] for i in rng.choice(len(active) - size, size=size, replace=False)]
        random_inactive = [inactive[size:][i] for i in rng.choice(len(inactive) - size, size=size, replace=False)]
        selected[category] = []
        for stratum, records in (("high_active", high_active), ("random_active", random_active),
                                 ("near_inactive", near_inactive), ("random_inactive", random_inactive)):
            selected[category].extend([{**row, "stratum": stratum} for row in records])
    counts = {category: {key: len(value) for key, value in pools.items()}
              for category, pools in candidates.items()}
    return selected, excluded, counts


def summarize(results, bases, fraction):
    output = {}
    for basis in bases:
        output[basis] = {}
        for group in ("known", "prospective"):
            group_rows = [row for row in results.values() if row["group"] == group]
            output[basis][group] = {}
            for stratum in ("high_active", "random_active", "near_inactive", "random_inactive"):
                category_values = []
                surface_values = {True: [], False: []}
                for row in group_rows:
                    sites = [site for site in row["sites"] if site["stratum"] == stratum]
                    if not sites:
                        continue
                    category_values.append({variant: float(np.median([
                        site["bases"][basis][fraction][variant]["sae_decoder_fidelity"]
                        for site in sites])) for variant in ("top", "low")})
                    for label in (True, False):
                        labeled = [site["bases"][basis][fraction]["top"]["sae_decoder_fidelity"]
                                   for site in sites if site["surface_label"] is label]
                        if labeled:
                            surface_values[label].append(float(np.median(labeled)))
                output[basis][group][stratum] = {
                    "eligible_categories": len(category_values),
                    "top_mediation": float(np.median([row["top"] for row in category_values])) if category_values else None,
                    "low_mediation": float(np.median([row["low"] for row in category_values])) if category_values else None,
                    "surface_positive_mediation": float(np.median(surface_values[True])) if surface_values[True] else None,
                    "surface_negative_mediation": float(np.median(surface_values[False])) if surface_values[False] else None,
                    "surface_positive_categories": len(surface_values[True]),
                    "surface_negative_categories": len(surface_values[False]),
                }
    return output


def main():
    config = json.loads(CONFIG.read_text()); atlas = json.loads(ATLAS_OUTPUT.read_text())
    pilot_arrays = np.load(PILOT_ARRAYS); atlas_arrays = np.load(ATLAS_ARRAYS)
    square = torch.from_numpy(pilot_arrays["covariance_square"]).cuda()
    inverse_square = torch.from_numpy(pilot_arrays["covariance_inverse_square"]).cuda()
    known = config["known_categories"]; prospective = config["prospective_categories"]
    categories = known + prospective; all_categories = atlas["config"]["reference_categories"] + atlas["config"]["prospective_categories"]
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    down = state["transformer.h.1.mlp.Down.weight"].float().cuda(); del state
    archive = np.load(SPARSE / "natural_large.npz")
    train = torch.from_numpy(archive["fineweb_train__h1"].astype(np.float32))
    mean = train.mean(0).cuda(); std = train.std(0).clamp_min(1e-4).cuda(); del train, archive
    directions, metadata = load_feature_directions(
        {"strong_categories": all_categories, "sae_seeds": atlas["config"]["feature_seeds"], "seed": atlas["config"]["seed"]}, std.cpu(), down)
    sae = torch.load(SPARSE / "activation_sae_large_seed101.pt", map_location="cpu", weights_only=True)
    encoder_weight = sae["encoder.weight"].float().cuda(); encoder_bias = sae["encoder.bias"].float().cuda()
    decoder = sae["decoder.weight"].float(); decoders = {}; feature_indices = {}
    for category in categories:
        name = f"semantic/seed101/{category}"; feature_indices[category] = metadata[name]["feature"]
        decoders[name] = std.cpu() * decoder[:, feature_indices[category]]
    del sae, decoder
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    all_rows = np.load(DEFAULT_CORPUS, mmap_mode="r"); panel = config["panel"]
    rows = all_rows[panel["start"]:panel["start"] + panel["rows"]]
    sites, excluded, gate_counts = collect_gate_sites(
        model, rows, categories, feature_indices, encoder_weight, encoder_bias,
        mean, std, config, tokenizer)
    eligible = list(sites); print("gate intervention categories", eligible, flush=True)
    local_config = {"rank": config["rank"], "bases": config["bases"],
                    "input_norm_fractions": [config["input_norm_fraction"]]}
    results = evaluate(model, rows, sites, eligible, set(known), directions, decoders,
                       pilot_arrays, atlas_arrays, square, inverse_square, local_config)
    fraction = str(config["input_norm_fraction"]); summaries = summarize(results, config["bases"], fraction)

    high_checks, gate_gaps, monotonic, surface_differences, group_gaps = {}, {}, {}, {}, {}
    for basis in config["bases"]:
        k = summaries[basis]["known"]
        high_checks[basis] = (k["high_active"]["top_mediation"] >= .30 and
                              k["high_active"]["top_mediation"] - k["high_active"]["low_mediation"] >= .20)
        gate_gaps[basis] = k["random_active"]["top_mediation"] - k["random_inactive"]["top_mediation"]
        sequence = [k[s]["top_mediation"] for s in ("high_active", "random_active", "near_inactive", "random_inactive")]
        monotonic[basis] = all(a >= b for a, b in zip(sequence, sequence[1:]))
        pos, neg = k["random_active"]["surface_positive_mediation"], k["random_active"]["surface_negative_mediation"]
        surface_differences[basis] = abs(pos - neg) if pos is not None and neg is not None else None
        p = summaries[basis]["prospective"]["random_active"]["top_mediation"]
        group_gaps[basis] = k["random_active"]["top_mediation"] - p
    hypotheses = {
        "G1_active_high_replication": {"supported": all(high_checks.values()), "bases": high_checks},
        "G2_gate_specificity": {"supported": all(value >= .15 for value in gate_gaps.values()), "bases": gate_gaps},
        "G3_activation_gradient": {"supported": any(monotonic.values()), "bases": monotonic},
        "G4_surface_label_residual": {"supported": any(value is not None and value <= .10 for value in surface_differences.values()), "absolute_differences": surface_differences},
        "G5_known_prospective_gate_gap": {"supported": any(value >= .10 for value in group_gaps.values()), "bases": group_gaps},
    }
    output = {"config": config, "excluded": excluded, "gate_counts": gate_counts,
              "sites": sites, "results": results, "summaries": summaries,
              "hypotheses": hypotheses}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False))

    lines = ["# SAE-gate-stratified spectral-path validation", "",
             "This fresh FineWeb panel ignores the hand-written token category when selecting interventions. Sites are stratified by whether the exact seed-101 SAE Top-128 gate selects the feature and by feature preactivation. Surface labels are retained only for diagnosis.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Gate validation](spectral_gate_validation.png)", "",
              "| Basis | Group | High active | Random active | Near inactive | Random inactive |",
              "|---|---|---:|---:|---:|---:|"]
    for basis in config["bases"]:
        for group in ("known", "prospective"):
            row = summaries[basis][group]
            lines.append(f"| {basis} | {group} | " + " | ".join(
                f"{row[s]['top_mediation']:.3f}" for s in ("high_active", "random_active", "near_inactive", "random_inactive")) + " |")
    lines += ["", "Values are category-median SAE-decoder mediation fidelities for rank-4, 10%-of-input-norm edits. Exact gate counts, surface-label subdivisions, and site records are in `spectral_gate_validation.json`."]
    REPORT.write_text("\n".join(lines) + "\n")

    strata = ["high_active", "random_active", "near_inactive", "random_inactive"]
    x = np.arange(len(strata)); fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for basis in config["bases"]:
        axes[0].plot(x, [summaries[basis]["known"][s]["top_mediation"] for s in strata], marker="o", label=basis)
        axes[1].plot(x, [summaries[basis]["prospective"][s]["top_mediation"] for s in strata], marker="o", label=basis)
    for axis, title in zip(axes, ("Known categories", "Prospective categories")):
        axis.set_xticks(x, strata, rotation=25, ha="right"); axis.set_ylabel("SAE-decoder mediation fidelity"); axis.set_title(title); axis.legend()
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
