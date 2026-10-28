#!/usr/bin/env python3
"""Diagnose the discovered his->her path-integration outlier."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_attention_qkov_depth_audit import logits_from_attention_writes, trace
from run_bilin12_boundary_carrier_depth_audit import fisher_terms
from run_bilin12_final_destination_source_program import source_write_contributions
from run_bilin12_semantic_source_jvp_predictor import differentiable_source_trace_with_masses
from run_bilin12_semantic_source_program_transfer import jaccard
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_gender_path_singularity_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_gender_path_singularity_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_GENDER_PATH_SINGULARITY_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_gender_path_singularity_redteam.png"


def path_state(base_x0, changed_x0, edit_position, location, spherical):
    point = base_x0.clone()
    tangent = torch.zeros_like(base_x0)
    left = base_x0[:, edit_position]
    right = changed_x0[:, edit_position]
    if not spherical:
        point[:, edit_position] = left + location * (right - left)
        tangent[:, edit_position] = right - left
        return point, tangent
    left_norm = left.norm(dim=-1, keepdim=True)
    right_norm = right.norm(dim=-1, keepdim=True)
    unit_left = left / left_norm
    unit_right = right / right_norm
    angle = torch.acos((unit_left * unit_right).sum(-1, keepdim=True).clamp(-1 + 1e-7, 1 - 1e-7))
    sine = torch.sin(angle)
    scale = (left_norm + location * (right_norm - left_norm))
    direction = torch.sin((1 - location) * angle) / sine * unit_left + torch.sin(location * angle) / sine * unit_right
    direction_derivative = -angle * torch.cos((1 - location) * angle) / sine * unit_left + angle * torch.cos(location * angle) / sine * unit_right
    scale_derivative = right_norm - left_norm
    point[:, edit_position] = scale * direction
    tangent[:, edit_position] = scale_derivative * direction + scale * direction_derivative
    return point, tangent


def evaluate_node(model, base_x0, changed_x0, config, location, spherical):
    point, tangent = path_state(base_x0, changed_x0, config["edit_position"], location, spherical)
    function = lambda value: differentiable_source_trace_with_masses(model, value, config["layers"], config["prediction_position"])
    (sources, masses), (derivatives, _) = torch.func.jvp(function, (point,), (tangent,))
    return tuple(value.float() for value in derivatives), tuple(value.float() for value in masses)


def integrate(values, weights):
    return tuple(sum(float(weight) * row[layer] for row, weight in zip(values, weights)) for layer in range(len(values[0])))


def finite_metric(base_logits, target_delta, prediction_delta):
    target, _, _, error = fisher_terms(base_logits, target_delta, prediction_delta)
    return float(target.sum()), float(error.sum()), float(1 - error.sum() / target.sum().clamp_min(1e-30))


def aggregate(rows):
    target = sum(row[0] for row in rows)
    error = sum(row[1] for row in rows)
    return {"target": target, "error": error, "fisher_fidelity": 1 - error / max(target, 1e-30)}


def main():
    red = json.loads(CONFIG.read_text())
    path_config = json.loads(Path(red["source_config"]).read_text())
    config = json.loads(Path(path_config["source_config"]).read_text())
    prior = json.loads(Path(red["source_output"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    sources = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"] : spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    gender_records = [record for record in prior["records"] if record["family"] == red["family"]]
    dense_nodes = np.linspace(0, 1, red["dense_points"])
    dense_weights = np.ones(red["dense_points"]) / (red["dense_points"] - 1)
    dense_weights[[0, -1]] *= 0.5
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(8)
    gauss_nodes = (gauss_nodes + 1) / 2
    gauss_weights = gauss_weights / 2
    records = []
    method_rows = defaultdict(list)

    for prior_record in gender_records:
        corpus = prior_record["corpus"]
        row_index, start = prior_record["row"], prior_record["start"]
        _, target_token = (int(value) for value in prior_record["pair"].split("->"))
        raw = np.asarray(sources[corpus][row_index, start : start + config["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy()
        changed[config["edit_position"]] = target_token
        base_ids = torch.as_tensor(raw, device="cuda")[None]
        changed_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            base_rows, _, _ = trace(model, base_ids)
            changed_rows, _, _ = trace(model, changed_ids)
            base_x0 = F.rms_norm(model.embedding(base_ids), (model.config.n_embd,))
            changed_x0 = F.rms_norm(model.embedding(changed_ids), (model.config.n_embd,))
        with torch.enable_grad():
            straight_dense = [evaluate_node(model, base_x0, changed_x0, config, float(node), False) for node in dense_nodes]
            spherical_dense = [evaluate_node(model, base_x0, changed_x0, config, float(node), True) for node in dense_nodes]
            straight_gauss = [evaluate_node(model, base_x0, changed_x0, config, float(node), False) for node in gauss_nodes]
            spherical_gauss = [evaluate_node(model, base_x0, changed_x0, config, float(node), True) for node in gauss_nodes]
        predictions = {
            "straight_dense": integrate([row[0] for row in straight_dense], dense_weights),
            "spherical_dense": integrate([row[0] for row in spherical_dense], dense_weights),
            "straight_gauss8": integrate([row[0] for row in straight_gauss], gauss_weights),
            "spherical_gauss8": integrate([row[0] for row in spherical_gauss], gauss_weights),
        }
        layer_records = {}
        for layer_index, layer in enumerate(config["layers"]):
            base = base_rows[layer]
            changed_row = changed_rows[layer]
            block = model.blocks[layer]
            exact_delta = source_write_contributions(block, changed_row["details"], config["prediction_position"]).float() - source_write_contributions(block, base["details"], config["prediction_position"]).float()
            full_write = base["details"]["write"].clone()
            full_write[:, config["prediction_position"]] = changed_row["details"]["write"][:, config["prediction_position"]]
            base_logits, target_logits = logits_from_attention_writes(model, layer, base, torch.stack((base["details"]["write"], full_write)))
            target_delta = target_logits - base_logits
            worlds = []
            names = []
            for name, predicted in predictions.items():
                write = base["details"]["write"].clone()
                write[:, config["prediction_position"]] += predicted[layer_index].sum(1).to(write.dtype)
                worlds.append(write)
                names.append(name)
            logits = logits_from_attention_writes(model, layer, base, torch.stack(worlds))
            metrics = {}
            for world_index, name in enumerate(names):
                metric = finite_metric(base_logits, target_delta, logits[world_index] - base_logits)
                metrics[name] = metric
                method_rows[(prior_record["pair"], name)].append(metric)
            derivative_norms = np.asarray([float(row[0][layer_index].norm()) for row in straight_dense])
            minimum_masses = np.asarray([float(row[1][layer_index].min()) for row in straight_dense])
            endpoint_mass_median = float(np.median([minimum_masses[0], minimum_masses[-1]]))
            correlation = float(np.corrcoef(np.log(derivative_norms + 1e-30), -np.log(minimum_masses + 1e-30))[0, 1])
            base_order = straight_dense[0][0][layer_index].norm(dim=-1).argsort(-1, descending=True)[0, :16].cpu().tolist()
            endpoint_order = straight_dense[-1][0][layer_index].norm(dim=-1).argsort(-1, descending=True)[0, :16].cpu().tolist()
            layer_records[str(layer)] = {
                "metrics": {name: list(value) for name, value in metrics.items()},
                "derivative_peak_ratio": float(derivative_norms.max() / max(np.median(derivative_norms), 1e-30)),
                "minimum_mass_endpoint_ratio": float(minimum_masses.min() / max(endpoint_mass_median, 1e-30)),
                "mass_derivative_correlation": correlation,
                "endpoint_top16_jaccard": jaccard(base_order, endpoint_order),
                "straight_dense_relative_error": float((predictions["straight_dense"][layer_index] - exact_delta).norm() / exact_delta.norm().clamp_min(1e-30)),
                "spherical_dense_relative_error": float((predictions["spherical_dense"][layer_index] - exact_delta).norm() / exact_delta.norm().clamp_min(1e-30)),
                "path": {
                    "locations": dense_nodes.tolist(),
                    "derivative_norms": derivative_norms.tolist(),
                    "minimum_attention_masses": minimum_masses.tolist(),
                },
            }
        records.append({"corpus": corpus, "pair": prior_record["pair"], "layers": layer_records})
        print(corpus, prior_record["pair"], flush=True)

    by_pair_method = {pair: {method: aggregate(method_rows[(pair, method)]) for method in predictions} for pair in sorted({record["pair"] for record in records})}
    anomaly = red["anomalous_pair"]
    control_pairs = [pair for pair in by_pair_method if pair != anomaly]
    anomaly_layers = [layer for record in records if record["pair"] == anomaly for layer in record["layers"].values()]
    control_layers = [layer for record in records if record["pair"] != anomaly for layer in record["layers"].values()]
    total_gauss_error = sum(row[1] for rows in method_rows.values() for row in rows if False)
    pair_gauss_error = {pair: by_pair_method[pair]["straight_gauss8"]["error"] for pair in by_pair_method}
    anomaly_error_share = pair_gauss_error[anomaly] / max(sum(pair_gauss_error.values()), 1e-30)
    anomaly_peak = max(row["derivative_peak_ratio"] for row in anomaly_layers)
    control_peak_median = float(np.median([row["derivative_peak_ratio"] for row in control_layers]))
    anomaly_gauss = by_pair_method[anomaly]["straight_gauss8"]["fisher_fidelity"]
    anomaly_dense = by_pair_method[anomaly]["straight_dense"]["fisher_fidelity"]
    anomaly_spherical_gauss = by_pair_method[anomaly]["spherical_gauss8"]["fisher_fidelity"]
    anomaly_spherical_dense = by_pair_method[anomaly]["spherical_dense"]["fisher_fidelity"]
    hypotheses = {
        "D1_gauss8_regression_is_concentrated_in_the_discovered_pair": {"supported": bool(anomaly_error_share >= 0.75), "error_share": anomaly_error_share},
        "D2_the_discovered_path_has_an_unusually_sharp_derivative": {"supported": bool(anomaly_peak >= 2 * control_peak_median), "anomaly_max_ratio": anomaly_peak, "control_median_ratio": control_peak_median},
        "D3_sharpness_coincides_with_attention_normalizer_collapse": {"supported": bool(min(row["minimum_mass_endpoint_ratio"] for row in anomaly_layers) <= 0.25), "minimum_ratio": min(row["minimum_mass_endpoint_ratio"] for row in anomaly_layers)},
        "D4_dense_straight_path_integration_resolves_the_outlier": {"supported": bool(anomaly_dense >= 0.99), "fidelity": anomaly_dense},
        "D5_constant_norm_path_improves_low_order_quadrature": {"supported": bool(anomaly_spherical_gauss - anomaly_gauss >= 0.20), "improvement": anomaly_spherical_gauss - anomaly_gauss, "spherical": anomaly_spherical_gauss, "straight": anomaly_gauss},
        "D6_gauss8_is_robust_on_matched_gender_controls": {"supported": bool(min(by_pair_method[pair]["straight_gauss8"]["fisher_fidelity"] for pair in control_pairs) >= 0.98), "minimum": min(by_pair_method[pair]["straight_gauss8"]["fisher_fidelity"] for pair in control_pairs)},
        "D7_attention_mass_predicts_derivative_spikes": {"supported": bool(max(row["mass_derivative_correlation"] for row in anomaly_layers) >= 0.50), "maximum_correlation": max(row["mass_derivative_correlation"] for row in anomaly_layers)},
        "D8_source_support_remains_stable_through_the_sharp_path": {"supported": bool(np.median([row["endpoint_top16_jaccard"] for row in anomaly_layers]) >= 0.60), "median_jaccard": float(np.median([row["endpoint_top16_jaccard"] for row in anomaly_layers]))},
        "D9_the_regression_is_quadrature_under_resolution_not_endpoint_mismatch": {"supported": bool(anomaly_dense - anomaly_gauss >= 0.20), "improvement": anomaly_dense - anomaly_gauss},
        "D10_dense_integrals_are_path_independent_in_behavior": {"supported": bool(abs(anomaly_dense - anomaly_spherical_dense) <= 0.01), "difference": anomaly_dense - anomaly_spherical_dense, "straight": anomaly_dense, "spherical": anomaly_spherical_dense},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "by_pair_method": by_pair_method, "records": records}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for record in records:
        if record["pair"] != anomaly:
            continue
        for layer, row in record["layers"].items():
            axes[0].plot(row["path"]["locations"], row["path"]["derivative_norms"], label=f"{record['corpus']} L{layer}")
            axes[1].plot(row["path"]["locations"], row["path"]["minimum_attention_masses"], label=f"{record['corpus']} L{layer}")
    axes[0].set_title("his→her source-derivative norm")
    axes[0].set_xlabel("path location")
    axes[1].set_title("Minimum final-destination QK² mass")
    axes[1].set_xlabel("path location")
    methods = ["straight_gauss8", "spherical_gauss8", "straight_dense", "spherical_dense"]
    x = np.arange(len(by_pair_method))
    width = 0.18
    for index, method in enumerate(methods):
        axes[2].bar(x + (index - 1.5) * width, [by_pair_method[pair][method]["fisher_fidelity"] for pair in by_pair_method], width, label=method)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(list(by_pair_method), rotation=30, ha="right")
    axes[2].set_title("Gender-pair path reconstruction")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Red team of the his→her path-integration outlier",
        "",
        "Gauss-8 was nearly exact on most semantic edits but regressed on `his→her`, especially at layer 5 in TinyStories. This discovery-locked follow-up compares that pair with the other gender edits, samples 129 path points, records the final-destination squared-QK normalizer, and compares straight embedding chords with constant-norm spherical paths.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Gender path singularity red team](bilin12_gender_path_singularity_redteam.png)",
        "",
        "| Pair | Straight Gauss-8 | Spherical Gauss-8 | Straight dense | Spherical dense |",
        "|---|---:|---:|---:|---:|",
    ]
    for pair, methods_row in by_pair_method.items():
        lines.append(f"| {pair} | {methods_row['straight_gauss8']['fisher_fidelity']:.3f} | {methods_row['spherical_gauss8']['fisher_fidelity']:.3f} | {methods_row['straight_dense']['fisher_fidelity']:.3f} | {methods_row['spherical_dense']['fisher_fidelity']:.3f} |")
    lines += [
        "",
        f"The anomalous pair contributes **{anomaly_error_share:.1%}** of matched gender Gauss-8 Fisher error. Its largest derivative peak ratio is **{anomaly_peak:.2f}×**, versus a control median of **{control_peak_median:.2f}×**.",
        "",
        "This audit distinguishes smooth-but-sharp routing curvature from numerical recomposition error. Path choice may change low-order quadrature efficiency, but a correctly resolved line integral must recover the same endpoint difference.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
