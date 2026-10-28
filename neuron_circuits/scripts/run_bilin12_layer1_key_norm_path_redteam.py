#!/usr/bin/env python3
"""Test whether QK normalization creates the sharp layer-1 his->her path."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import trace
from run_bilin12_his_her_layer1_qkov_path import factor_row
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer1_key_norm_path_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer1_key_norm_path_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER1_KEY_NORM_PATH_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_layer1_key_norm_path_redteam.png"


def main():
    red = json.loads(CONFIG.read_text())
    config = json.loads(Path(red["source_config"]).read_text())
    prior = json.loads(Path(red["source_output"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    sources = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"] : spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    selected = [record for record in prior["records"] if record["family"] == red["family"]]
    locations = np.linspace(0, 1, red["dense_points"])
    records = []

    for prior_record in selected:
        corpus = prior_record["corpus"]
        row_index, start = prior_record["row"], prior_record["start"]
        _, target_token = (int(value) for value in prior_record["pair"].split("->"))
        raw = np.asarray(sources[corpus][row_index, start : start + config["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy()
        changed[config["edit_position"]] = target_token
        base_ids = torch.as_tensor(raw, device="cuda")[None]
        changed_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            trace(model, base_ids)
            trace(model, changed_ids)
            base_x0 = F.rms_norm(model.embedding(base_ids), (model.config.n_embd,))
            changed_x0 = F.rms_norm(model.embedding(changed_ids), (model.config.n_embd,))
            tangent = changed_x0 - base_x0
        with torch.enable_grad():
            path = [factor_row(model, base_x0, tangent, float(location), red["layer"], config["prediction_position"]) for location in locations]

        source_norm = np.asarray([float(row["source"].norm()) for row in path])
        peak_index = int(source_norm.argmax())
        peak = path[peak_index]
        routing_heads = peak["factors"]["query_routing"] + peak["factors"]["key_routing"]
        head_norms = routing_heads.flatten(2).norm(dim=-1)[0]
        dominant_head = int(head_norms.argmax())
        raw_key_norms = np.asarray([float(row["raw_key"][0, config["edit_position"], dominant_head].norm()) for row in path])
        raw_key_derivatives = np.asarray([float(row["raw_key_derivative"][0, config["edit_position"], dominant_head].norm()) for row in path])
        normalized_key_derivatives = np.asarray([float(row["key_derivative"][0, config["edit_position"], dominant_head].norm()) for row in path])
        amplification = normalized_key_derivatives / np.maximum(raw_key_derivatives, 1e-30)
        minimum_index = int(raw_key_norms.argmin())
        endpoint_norm = float(np.median([raw_key_norms[0], raw_key_norms[-1]]))
        endpoint_amplification = float(np.median([amplification[0], amplification[-1]]))
        all_head_key_norms = peak["raw_key"][0, config["edit_position"]].norm(dim=-1)
        correlation = float(np.corrcoef(np.log(source_norm + 1e-30), -np.log(raw_key_norms + 1e-30))[0, 1])
        records.append({
            "corpus": corpus,
            "pair": prior_record["pair"],
            "peak_location": float(locations[peak_index]),
            "minimum_location": float(locations[minimum_index]),
            "peak_minimum_distance": float(abs(locations[peak_index] - locations[minimum_index])),
            "minimum_endpoint_ratio": float(raw_key_norms.min() / max(endpoint_norm, 1e-30)),
            "source_inverse_key_correlation": correlation,
            "peak_amplification": float(amplification[peak_index]),
            "endpoint_amplification": endpoint_amplification,
            "peak_to_endpoint_amplification": float(amplification[peak_index] / max(endpoint_amplification, 1e-30)),
            "dominant_head": dominant_head,
            "minimum_norm_head": int(all_head_key_norms.argmin()),
            "maximum_normalized_key_derivative": float(normalized_key_derivatives.max()),
            "maximum_raw_key_derivative": float(raw_key_derivatives.max()),
            "paths": {
                "locations": locations.tolist(),
                "source_derivative": source_norm.tolist(),
                "raw_key_norm": raw_key_norms.tolist(),
                "raw_key_derivative": raw_key_derivatives.tolist(),
                "normalized_key_derivative": normalized_key_derivatives.tolist(),
                "amplification": amplification.tolist(),
            },
        })
        print(corpus, prior_record["pair"], flush=True)

    anomaly = [record for record in records if record["pair"] == red["anomalous_pair"]]
    controls = [record for record in records if record["pair"] != red["anomalous_pair"]]
    anomaly_min_ratio = min(record["minimum_endpoint_ratio"] for record in anomaly)
    control_median_ratio = float(np.median([record["minimum_endpoint_ratio"] for record in controls]))
    anomaly_max_normalized = max(record["maximum_normalized_key_derivative"] for record in anomaly)
    control_median_normalized = float(np.median([record["maximum_normalized_key_derivative"] for record in controls]))
    anomaly_max_raw = max(record["maximum_raw_key_derivative"] for record in anomaly)
    control_median_raw = float(np.median([record["maximum_raw_key_derivative"] for record in controls]))
    hypotheses = {
        "N1_key_norm_minimum_is_near_the_source_derivative_peak": {"supported": all(record["peak_minimum_distance"] <= 0.10 for record in anomaly), "distances": {record["corpus"]: record["peak_minimum_distance"] for record in anomaly}},
        "N2_the_raw_key_has_a_large_norm_dip": {"supported": all(record["minimum_endpoint_ratio"] <= 0.50 for record in anomaly), "ratios": {record["corpus"]: record["minimum_endpoint_ratio"] for record in anomaly}},
        "N3_inverse_key_norm_tracks_source_sharpness": {"supported": all(record["source_inverse_key_correlation"] >= 0.70 for record in anomaly), "correlations": {record["corpus"]: record["source_inverse_key_correlation"] for record in anomaly}},
        "N4_qk_normalization_amplifies_the_peak": {"supported": all(record["peak_to_endpoint_amplification"] >= 2 for record in anomaly), "ratios": {record["corpus"]: record["peak_to_endpoint_amplification"] for record in anomaly}},
        "N5_the_norm_dip_is_specific_to_the_anomalous_pair": {"supported": bool(anomaly_min_ratio <= 0.5 * control_median_ratio), "anomaly_minimum": anomaly_min_ratio, "control_median": control_median_ratio},
        "N6_the_dominant_routing_head_is_the_lowest_norm_key_head": {"supported": all(record["dominant_head"] == record["minimum_norm_head"] for record in anomaly), "heads": {record["corpus"]: {"dominant": record["dominant_head"], "minimum_norm": record["minimum_norm_head"]} for record in anomaly}},
        "N7_the_anomaly_has_larger_normalized_key_derivatives_than_controls": {"supported": bool(anomaly_max_normalized >= 2 * control_median_normalized), "anomaly_maximum": anomaly_max_normalized, "control_median": control_median_normalized},
        "N8_raw_key_derivatives_are_not_equally_anomalous": {"supported": bool(anomaly_max_raw < 1.5 * control_median_raw), "anomaly_maximum": anomaly_max_raw, "control_median": control_median_raw},
        "N9_head_identity_remains_context_dependent": {"supported": len({record["dominant_head"] for record in anomaly}) > 1, "heads": {record["corpus"]: record["dominant_head"] for record in anomaly}},
        "N10_qk_normalization_explains_more_than_attention_mass": {"supported": bool(max(record["peak_to_endpoint_amplification"] for record in anomaly) >= 2), "maximum_amplification_ratio": max(record["peak_to_endpoint_amplification"] for record in anomaly), "prior_attention_mass_ratio": 0.9849241340240336},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "records": records}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for record in anomaly:
        axes[0].plot(record["paths"]["locations"], record["paths"]["source_derivative"], label=record["corpus"])
        axes[1].plot(record["paths"]["locations"], record["paths"]["raw_key_norm"], label=record["corpus"])
        axes[2].plot(record["paths"]["locations"], record["paths"]["amplification"], label=record["corpus"])
    axes[0].set_title("his→her source derivative")
    axes[1].set_title("Dominant raw key norm")
    axes[2].set_title("QK-normalization derivative gain")
    for axis in axes:
        axis.set_xlabel("path location")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# QK-normalization red team of the layer-1 key-routing spike",
        "",
        "For every gender edit in FineWeb and TinyStories, the layer-1 token path is sampled at 129 points. The dominant routing head's pre-QK-normalization key norm, raw key derivative, normalized key derivative, and full source derivative are compared. This tests a concrete replacement for the falsified attention-denominator-collapse explanation.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Key norm path red team](bilin12_layer1_key_norm_path_redteam.png)",
        "",
        "| Corpus | Pair | Peak/min distance | Min/endpoint key norm | Inverse-norm correlation | Peak/endpoint gain | Dominant/min-norm head |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(f"| {record['corpus']} | {record['pair']} | {record['peak_minimum_distance']:.3f} | {record['minimum_endpoint_ratio']:.3f} | {record['source_inverse_key_correlation']:.3f} | {record['peak_to_endpoint_amplification']:.3f} | H{record['dominant_head']} / H{record['minimum_norm_head']} |")
    lines += [
        "",
        "QK normalization can amplify directional motion when a projected query or key has low raw norm. A norm dip would explain sharp but continuous routing without either a raw score zero-crossing or collapse of the attention row normalizer.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
