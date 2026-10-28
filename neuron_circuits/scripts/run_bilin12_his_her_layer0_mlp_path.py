#!/usr/bin/env python3
"""Decompose the upstream layer-0 path and its native bilinear channels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_semantic_source_jvp_predictor import differentiable_attention
from run_bilin12_semantic_source_program_transfer import jaccard
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_his_her_layer0_mlp_path.json")
OUTPUT = TENSOR_ROOT / "bilin12_his_her_layer0_mlp_path.json"
REPORT = TENSOR_ROOT / "BILIN12_HIS_HER_LAYER0_MLP_PATH.md"
FIGURE = TENSOR_ROOT / "bilin12_his_her_layer0_mlp_path.png"


def layer0_components(model, x0, position):
    block = model.blocks[0]
    mixed = block.lambdas[0] * x0 + block.lambdas[1] * x0
    normalized = F.rms_norm(mixed, (mixed.shape[-1],))
    attention, _, _, _, _ = differentiable_attention(model, block, normalized, None)
    pre = mixed + attention
    state = F.rms_norm(pre, (pre.shape[-1],))
    left = block.mlp.Left(state)
    right = block.mlp.Right(state)
    product = left * right
    mlp = block.mlp.Down(product) + block.mlp.Down_bias
    output = pre + mlp
    return mixed[:, position], attention[:, position], pre[:, position], state[:, position], left[:, position], right[:, position], product[:, position], mlp[:, position], output[:, position]


def path_row(model, base_x0, tangent, location, position):
    point = base_x0 + location * tangent
    function = lambda value: layer0_components(model, value, position)
    primal, derivative = torch.func.jvp(function, (point,), (tangent,))
    mixed, attention, pre, state, left, right, product, mlp, output = primal
    dmixed, dattention, dpre, dstate, dleft, dright, dproduct, dmlp, doutput = derivative
    block = model.blocks[0]
    left_product = dleft * right
    right_product = left * dright
    weight = block.mlp.Down.weight
    channel = dproduct[:, :, None] * weight.T[None]
    left_channel = left_product[:, :, None] * weight.T[None]
    right_channel = right_product[:, :, None] * weight.T[None]
    return {
        "primal": {"pre": pre.float(), "mlp": mlp.float(), "output": output.float()},
        "derivative": {
            "embedding": dmixed.float(),
            "attention": dattention.float(),
            "pre": dpre.float(),
            "state": dstate.float(),
            "product": dproduct.float(),
            "left_product": (left_channel.sum(1)).float(),
            "right_product": (right_channel.sum(1)).float(),
            "mlp": dmlp.float(),
            "output": doutput.float(),
            "channels": channel.float(),
        },
    }


def squared_fidelity(target, approximation):
    denominator = target.square().sum().clamp_min(1e-30)
    return float(1 - (target - approximation).square().sum() / denominator)


def minimum_budget(target, contributions, order, threshold=0.90):
    approximation = torch.zeros_like(target)
    for count, index in enumerate(order, 1):
        approximation += contributions[:, int(index)]
        if squared_fidelity(target, approximation) >= threshold:
            return count
    return contributions.shape[1]


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
    weights = np.ones(red["dense_points"]) / (red["dense_points"] - 1)
    weights[[0, -1]] *= 0.5
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
            base_x0 = F.rms_norm(model.embedding(base_ids), (model.config.n_embd,))
            changed_x0 = F.rms_norm(model.embedding(changed_ids), (model.config.n_embd,))
            tangent = changed_x0 - base_x0
            base_components = layer0_components(model, base_x0, config["edit_position"])
            changed_components = layer0_components(model, changed_x0, config["edit_position"])
        with torch.enable_grad():
            path = [path_row(model, base_x0, tangent, float(location), config["edit_position"]) for location in locations]

        norm_paths = {
            name: np.asarray([float(row["derivative"][name].norm()) for row in path])
            for name in ("embedding", "attention", "pre", "state", "left_product", "right_product", "mlp", "output")
        }
        peak_index = int(norm_paths["output"].argmax())
        peak = path[peak_index]["derivative"]
        component_norms = {name: float(peak[name].norm()) for name in ("embedding", "attention", "mlp")}
        mlp_share = component_norms["mlp"] / max(sum(component_norms.values()), 1e-30)
        left_norm = float(peak["left_product"].norm())
        right_norm = float(peak["right_product"].norm())
        channel_norms = peak["channels"].norm(dim=-1)[0]
        order = channel_norms.argsort(descending=True)
        channel_fidelity = {}
        for budget in red["channel_budgets"]:
            approximation = peak["channels"][:, order[:budget]].sum(1)
            channel_fidelity[str(budget)] = squared_fidelity(peak["mlp"], approximation)
        channels90 = minimum_budget(peak["mlp"], peak["channels"], order)
        integrated = {
            name: sum(float(weight) * row["derivative"][name] for row, weight in zip(path, weights))
            for name in ("embedding", "attention", "mlp", "output", "channels")
        }
        finite_mlp = changed_components[7].float() - base_components[7].float()
        finite_output = changed_components[8].float() - base_components[8].float()
        channel_integral_error = float((integrated["channels"].sum(1) - finite_mlp).norm() / finite_mlp.norm().clamp_min(1e-30))
        mlp_projection = float((integrated["mlp"] * finite_output).sum() / finite_output.square().sum().clamp_min(1e-30))
        pre_norms = np.asarray([float(row["primal"]["pre"].norm()) for row in path])
        records.append({
            "corpus": corpus,
            "pair": prior_record["pair"],
            "peak_location": float(locations[peak_index]),
            "peak_ratios": {name: float(values.max() / max(np.median(values), 1e-30)) for name, values in norm_paths.items()},
            "mlp_component_share": mlp_share,
            "left_product_share": left_norm / max(left_norm + right_norm, 1e-30),
            "right_product_share": right_norm / max(left_norm + right_norm, 1e-30),
            "minimum_pre_norm_endpoint_ratio": float(pre_norms.min() / max(np.median([pre_norms[0], pre_norms[-1]]), 1e-30)),
            "channel_fidelity": channel_fidelity,
            "channels_to_90": channels90,
            "top64_channels": order[:64].cpu().tolist(),
            "channel_integral_relative_error": channel_integral_error,
            "mlp_finite_output_projection": mlp_projection,
            "paths": {"locations": locations.tolist(), **{name: values.tolist() for name, values in norm_paths.items()}},
        })
        print(corpus, prior_record["pair"], flush=True)

    anomaly = [record for record in records if record["pair"] == red["anomalous_pair"]]
    controls = [record for record in records if record["pair"] != red["anomalous_pair"]]
    anomaly_max_mlp = max(record["peak_ratios"]["mlp"] for record in anomaly)
    control_median_max_mlp = float(np.median([record["peak_ratios"]["mlp"] for record in controls]))
    anomaly_channel_median = float(np.median([record["channels_to_90"] for record in anomaly]))
    control_channel_median = float(np.median([record["channels_to_90"] for record in controls]))
    overlap = jaccard(anomaly[0]["top64_channels"], anomaly[1]["top64_channels"])
    hypotheses = {
        "M1_layer0_mlp_derivative_is_unusually_sharp": {"supported": bool(anomaly_max_mlp >= 2 * control_median_max_mlp), "anomaly_maximum": anomaly_max_mlp, "control_median": control_median_max_mlp},
        "M2_the_mlp_dominates_the_sharp_layer0_update": {"supported": all(record["mlp_component_share"] >= 0.60 for record in anomaly), "shares": {record["corpus"]: record["mlp_component_share"] for record in anomaly}},
        "M3_both_bilinear_sides_matter_at_the_peak": {"supported": all(min(record["left_product_share"], record["right_product_share"]) >= 0.20 for record in anomaly), "shares": {record["corpus"]: [record["left_product_share"], record["right_product_share"]] for record in anomaly}},
        "M4_the_pre_mlp_state_does_not_collapse": {"supported": all(record["minimum_pre_norm_endpoint_ratio"] >= 0.80 for record in anomaly), "ratios": {record["corpus"]: record["minimum_pre_norm_endpoint_ratio"] for record in anomaly}},
        "M5_sixtyfour_native_channels_are_sufficient_for_the_peak_mlp_derivative": {"supported": all(record["channel_fidelity"]["64"] >= 0.90 for record in anomaly), "fidelities": {record["corpus"]: record["channel_fidelity"]["64"] for record in anomaly}},
        "M6_the_peak_mlp_derivative_is_not_single_neuron_aligned": {"supported": all(record["channels_to_90"] >= 8 for record in anomaly), "counts": {record["corpus"]: record["channels_to_90"] for record in anomaly}},
        "M7_the_same_native_channels_recur_across_corpora": {"supported": bool(overlap >= 0.25), "top64_jaccard": overlap},
        "M8_dense_channel_integrals_recompose_the_finite_mlp_change": {"supported": all(record["channel_integral_relative_error"] <= 1e-3 for record in records), "maximum_error": max(record["channel_integral_relative_error"] for record in records)},
        "M9_the_mlp_is_a_major_finite_layer0_component": {"supported": all(record["mlp_finite_output_projection"] >= 0.50 for record in anomaly), "projections": {record["corpus"]: record["mlp_finite_output_projection"] for record in anomaly}},
        "M10_controls_require_no_more_channels_than_the_anomaly": {"supported": bool(anomaly_channel_median >= control_channel_median), "anomaly_median": anomaly_channel_median, "control_median": control_channel_median},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "records": records}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    for record in anomaly:
        axes[0].plot(record["paths"]["locations"], record["paths"]["attention"], label=f"attention {record['corpus']}")
        axes[0].plot(record["paths"]["locations"], record["paths"]["mlp"], ls="--", label=f"MLP {record['corpus']}")
    axes[0].set_title("Layer-0 edited-position derivatives")
    budgets = red["channel_budgets"]
    for record in anomaly:
        axes[1].plot(budgets, [record["channel_fidelity"][str(budget)] for budget in budgets], marker="o", label=record["corpus"])
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Native-channel derivative frontier")
    pairs = sorted({record["pair"] for record in records})
    axes[2].bar(np.arange(len(pairs)), [np.median([record["channels_to_90"] for record in records if record["pair"] == pair]) for pair in pairs])
    axes[2].set_xticks(np.arange(len(pairs)))
    axes[2].set_xticklabels(pairs, rotation=30, ha="right")
    axes[2].set_title("Channels needed for 90% peak fidelity")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Layer-0 bilinear-MLP path behind the his→her key-routing spike",
        "",
        "The edited token's layer-0 update is differentiated into direct embedding, attention, and bilinear-MLP paths. The MLP derivative is split exactly into left-factor and right-factor product rules and into native post-product channel writes. Matched gender substitutions provide controls for curvature and channel count.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-0 MLP path](bilin12_his_her_layer0_mlp_path.png)",
        "",
        "| Corpus | Pair | MLP peak ratio | MLP component share | Left/right share | Channels to 90% | Top-64 fidelity | Finite MLP projection |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(f"| {record['corpus']} | {record['pair']} | {record['peak_ratios']['mlp']:.2f} | {record['mlp_component_share']:.3f} | {record['left_product_share']:.3f}/{record['right_product_share']:.3f} | {record['channels_to_90']} | {record['channel_fidelity']['64']:.3f} | {record['mlp_finite_output_projection']:.3f} |")
    lines += [
        "",
        f"Anomalous top-64 channel Jaccard across corpora: **{overlap:.3f}**. Channel counts describe causal reconstruction of the local MLP derivative, not semantic purity of those channels.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
