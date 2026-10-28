#!/usr/bin/env python3
"""Token-identity and repeated-row red team for the L12 punctuation chart."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features, within_token_auc
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_stratified_natural_route_census import all_strata, distance_class, token_class_from_flags, weighted_basis


CONFIG = Path("configs/bilin18_l12_punctuation_lookup_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_l12_punctuation_lookup_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_L12_PUNCTUATION_LOOKUP_REDTEAM.md"
DISTANCES = ("self", "local1-4", "near5-16", "mid17-64", "far65+")


def r2(target, prediction):
    return float(1 - np.square(target - prediction).sum() / max(np.square(target - target.mean()).sum(), 1e-30))


def token_lookup(discovery_ids, discovery_values, replication_ids):
    global_mean = float(discovery_values.mean()); sums = {}; counts = {}
    for token, value in zip(discovery_ids.tolist(), discovery_values.tolist()):
        sums[token] = sums.get(token, 0.0) + value; counts[token] = counts.get(token, 0) + 1
    means = {token: sums[token] / counts[token] for token in sums}
    return np.asarray([means.get(int(token), global_mean) for token in replication_ids])


def pole_profile(data, indices):
    labels = distance_class(data["distance"][indices])
    values = [np.mean(labels == name) for name in DISTANCES]
    values += [np.mean(data["source_class"][indices] == name) for name in ("punctuation", "function", "numeric", "capitalized", "specialized", "continuation", "other")]
    values += [np.mean((data["full_effect"] - data["semantic_effect"])[indices] > 0)]
    return np.asarray(values, dtype=float)


def cosine(first, second):
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def select(score, eligible, count, unique_rows=False, rows=None):
    ordered = eligible[np.argsort(-score[eligible])]
    if not unique_rows:
        return ordered[:count]
    selected = []; seen = set()
    for index in ordered:
        row = int(rows[index])
        if row in seen: continue
        seen.add(row); selected.append(index)
        if len(selected) == count: break
    return np.asarray(selected, dtype=int)


def panel(arrays, split, layer, classes):
    result = {key: arrays[f"{split}_{key}_L{layer}"] for key in ("residuals", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss", "global_row")}
    result["source_class"] = classes[result["source_id"]]; result["next_class"] = classes[result["next_id"]]
    return result


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"]); layer = config["layer"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    classes = token_class_from_flags(flags)
    discovery = panel(arrays, "discovery", layer, classes); replication = panel(arrays, "replication", layer, classes)
    dmask = all_strata(discovery, discovery["loss"])[config["stratum"]]; rmask = all_strata(replication, discovery["loss"])[config["stratum"]]
    deffect = discovery["full_effect"] - discovery["semantic_effect"]; reffect = replication["full_effect"] - replication["semantic_effect"]
    basis = weighted_basis(discovery["residuals"][dmask], deffect[dmask], config["rank"])
    dcoord = discovery["residuals"] @ basis.T; rcoord = replication["residuals"] @ basis.T
    axes = []; profile_cosines = []
    eligible_d = np.where(dmask)[0]; eligible_r = np.where(rmask)[0]
    for axis in range(config["rank"]):
        prediction = token_lookup(discovery["source_id"][dmask], dcoord[dmask, axis], replication["source_id"][rmask])
        lookup_r2 = r2(rcoord[rmask, axis], prediction)
        far_d = discovery["distance"][dmask] >= 17; far_r = replication["distance"][rmask] >= 17
        raw_auc, _, _ = within_token_auc(far_d, dcoord[dmask, axis], discovery["source_id"][dmask])
        orientation = 1 if raw_auc is not None and raw_auc >= .5 else -1
        heldout_auc, groups, pairs = within_token_auc(far_r, orientation * rcoord[rmask, axis], replication["source_id"][rmask])
        poles = []
        for sign in (1, -1):
            score = sign * rcoord[:, axis] * np.abs(reffect)
            ordinary = select(score, eligible_r, config["top_examples"])
            unique = select(score, eligible_r, config["top_examples"], True, replication["global_row"])
            similarity = cosine(pole_profile(replication, ordinary), pole_profile(replication, unique)); profile_cosines.append(similarity)
            poles.append({"sign": sign, "ordinary_median_distance": float(np.median(replication["distance"][ordinary])), "unique_row_median_distance": float(np.median(replication["distance"][unique])), "ordinary_distinct_rows": len(set(replication["global_row"][ordinary].tolist())), "unique_rows": len(unique), "profile_cosine": similarity})
        axes.append({"axis": axis, "source_token_lookup_r2": lookup_r2, "distance_orientation": orientation, "within_token_long_range_auc": heldout_auc, "within_token_groups": groups, "within_token_pairs": pairs, "poles": poles})
    axis0 = axes[0]
    positive = next(row for row in axis0["poles"] if row["sign"] == 1); negative = next(row for row in axis0["poles"] if row["sign"] == -1)
    hypotheses = {
        "T0_axes_are_frozen_from_discovery": {"supported": True},
        "T1_source_token_lookup_is_insufficient": {"supported": all(row["source_token_lookup_r2"] < .50 for row in axes), "r2": [row["source_token_lookup_r2"] for row in axes]},
        "T2_axis0_tracks_distance_within_token": {"supported": axis0["within_token_long_range_auc"] is not None and axis0["within_token_long_range_auc"] >= .70, "auc": axis0["within_token_long_range_auc"], "groups": axis0["within_token_groups"], "pairs": axis0["within_token_pairs"]},
        "T3_profiles_survive_one_edge_per_row": {"supported": float(np.mean(profile_cosines)) >= .80, "mean_cosine": float(np.mean(profile_cosines)), "cosines": profile_cosines},
        "T4_axis0_separates_local_and_long_range_poles": {"supported": positive["unique_row_median_distance"] <= 8 and negative["unique_row_median_distance"] >= 17, "positive_median": positive["unique_row_median_distance"], "negative_median": negative["unique_row_median_distance"]},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "axes": axes}, indent=2, allow_nan=False))
    lines = ["# L12 punctuation chart: token-lookup and repeated-row red team", "", "The frozen punctuation axes are tested against source-token lookup, within-token distance discrimination, and a one-edge-per-FineWeb-row selection rule.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Axis | Token-lookup R2 | Within-token long-range AUROC | + pole distance ordinary/unique | - pole distance ordinary/unique | Profile cosine +/- |", "|---:|---:|---:|---:|---:|---:|"]
    for row in axes:
        positive = next(pole for pole in row["poles"] if pole["sign"] == 1); negative = next(pole for pole in row["poles"] if pole["sign"] == -1)
        lines.append(f"| {row['axis']} | {row['source_token_lookup_r2']:.3f} | {row['within_token_long_range_auc']:.3f} | {positive['ordinary_median_distance']:.1f}/{positive['unique_row_median_distance']:.1f} | {negative['ordinary_median_distance']:.1f}/{negative['unique_row_median_distance']:.1f} | {positive['profile_cosine']:.3f}/{negative['profile_cosine']:.3f} |")
    lines += ["", "## Guardrail", "", "A within-token distance signal rejects literal source-token lookup, but does not uniquely identify a punctuation concept. Distance, repeated delimiter structure, and document layout remain entangled in this observational chart."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
