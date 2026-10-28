#!/usr/bin/env python3
"""Principal-angle-matched null for crossed chart factors."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from redteam_bilin18_stratified_chart_factor_symmetry import orthogonal_grid
from run_bilin18_stratified_natural_route_census import all_strata, token_class_from_flags, weighted_basis


CONFIG = Path("configs/bilin18_chart_factor_geometry_null.json")
OUTPUT = TENSOR_ROOT / "bilin18_chart_factor_geometry_null.json"
REPORT = TENSOR_ROOT / "BILIN18_CHART_FACTOR_GEOMETRY_NULL.md"


def principal_coordinates(first, second):
    u, singular, vh = np.linalg.svd(first @ second.T)
    return u.T @ first, vh @ second, singular


def matched_alternative(reference, cosines, rng):
    values = rng.standard_normal((reference.shape[1], reference.shape[0]))
    values = values - reference.T @ (reference @ values)
    q, _ = np.linalg.qr(values)
    complement = q[:, :reference.shape[0]].T
    return cosines[:, None] * reference + np.sqrt(np.maximum(1 - np.square(cosines), 0))[:, None] * complement


def transform(values, read, write):
    return (values @ read.T) @ write


def effect_fidelity(values, gradients, read, write):
    target = np.einsum("ni,ni->n", values, gradients)
    predicted = np.einsum("ni,ni->n", transform(values, read, write), gradients)
    return float(1 - np.square(target - predicted).sum() / max(np.square(target).sum(), 1e-30))


def oracle_hybrid(values, gradients, matching, alternative, grid):
    scores = []
    for _, rotation in grid:
        rotated = rotation @ alternative
        scores.append(effect_fidelity(values, gradients, matching, rotated))
        scores.append(effect_fidelity(values, gradients, rotated, matching))
    return max(scores), min(scores)


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"]); rng = np.random.default_rng(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    classes = token_class_from_flags(flags); grid = orthogonal_grid(config["angles"]); results = []; max_error = 0.0
    for target in config["targets"]:
        layer = target["layer"]
        panels = {}
        for split in ("discovery", "replication"):
            panel = {key: arrays[f"{split}_{key}_L{layer}"] for key in ("residuals", "gradients", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss")}
            panel["source_class"] = classes[panel["source_id"]]; panel["next_class"] = classes[panel["next_id"]]; panels[split] = panel
        discovery = panels["discovery"]; replication = panels["replication"]
        d_masks = all_strata(discovery, discovery["loss"]); r_masks = all_strata(replication, discovery["loss"])
        effects = discovery["full_effect"] - discovery["semantic_effect"]
        matching_raw = weighted_basis(discovery["residuals"][d_masks[target["target"]]], effects[d_masks[target["target"]]], config["rank"])
        alternative_raw = weighted_basis(discovery["residuals"][d_masks[target["alternative"]]], effects[d_masks[target["alternative"]]], config["rank"])
        matching, alternative, cosines = principal_coordinates(matching_raw, alternative_raw)
        selected = r_masks[target["target"]]; values = replication["residuals"][selected]; gradients = replication["gradients"][selected]
        matching_fidelity = effect_fidelity(values, gradients, matching, matching)
        actual_best, actual_worst = oracle_hybrid(values, gradients, matching, alternative, grid)
        null_best = []; angle_errors = []
        for _ in range(config["null_replicates"]):
            null = matched_alternative(matching, cosines, rng)
            angle_errors.append(float(np.max(np.abs(np.linalg.svd(matching @ null.T, compute_uv=False) - cosines))))
            best, _ = oracle_hybrid(values, gradients, matching, null, grid); null_best.append(best)
        actual_gap = matching_fidelity - actual_best; null_gaps = [matching_fidelity - value for value in null_best]
        results.append({
            "layer": layer, "target": target["target"], "alternative": target["alternative"], "principal_cosines": cosines.tolist(),
            "matching_fidelity": matching_fidelity, "actual_best_hybrid": actual_best, "actual_worst_hybrid": actual_worst,
            "actual_gap": actual_gap, "null_best_mean": float(np.mean(null_best)), "null_best_p95": float(np.quantile(null_best, .95)),
            "null_gap_mean": float(np.mean(null_gaps)), "null_gap_p95": float(np.quantile(null_gaps, .95)),
            "actual_hybrid_above_null_p95": bool(actual_best > np.quantile(null_best, .95)),
            "actual_gap_above_null_p95": bool(actual_gap > np.quantile(null_gaps, .95)),
        })
        max_error = max(max_error, max(angle_errors))
    special = [row["actual_hybrid_above_null_p95"] for row in results]; excess = [row["actual_gap_above_null_p95"] for row in results]
    hypotheses = {
        "G0_nulls_match_principal_angles": {"supported": max_error <= 1e-6, "maximum_error": max_error},
        "G1_actual_cross_factor_is_behaviorally_special": {"supported": any(special), "by_target": special},
        "G2_actual_interaction_gap_exceeds_geometry": {"supported": any(excess), "by_target": excess},
        "G3_factor_cooperation_claim_survives_geometry_null": {"supported": all(s or e for s, e in zip(special, excess)), "by_target": [s or e for s, e in zip(special, excess)]},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Principal-angle-matched geometry null for chart factors", "", "Random alternative subspaces are constructed with exactly the same principal cosines to the matching chart as the observed wrong chart. Each receives the same oracle O(2) hybrid search. This asks whether hybrid failure contains behavioral evidence beyond the geometry of crossing different projectors.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | Principal cosines | Matching | Actual best hybrid / gap | Null best mean/p95 | Null gap mean/p95 | Actual special / excess gap |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| L{row['layer']} {row['target']} | {[round(value, 3) for value in row['principal_cosines']]} | {row['matching_fidelity']:.3f} | {row['actual_best_hybrid']:.3f}/{row['actual_gap']:+.3f} | {row['null_best_mean']:.3f}/{row['null_best_p95']:.3f} | {row['null_gap_mean']:+.3f}/{row['null_gap_p95']:+.3f} | {row['actual_hybrid_above_null_p95']}/{row['actual_gap_above_null_p95']} |")
    lines += ["", "## Interpretation boundary", "", "This is a frozen-route local CE metric on the earlier replication panel, not a second finite intervention. The null is deliberately matched only on subspace angles; it does not match token-conditioned route covariance or contextual OV payloads. Failure against this null means the previous finite hybrid gap cannot by itself establish learned factor cooperation beyond projector geometry."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
