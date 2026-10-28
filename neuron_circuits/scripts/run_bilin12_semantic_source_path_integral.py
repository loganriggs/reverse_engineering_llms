#!/usr/bin/env python3
"""Integrate source-contribution JVPs along finite semantic edit paths."""

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
from run_bilin12_semantic_source_jvp_predictor import cosine, differentiable_source_trace
from run_bilin12_semantic_source_program_transfer import jaccard
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_semantic_source_path_integral.json")
OUTPUT = TENSOR_ROOT / "bilin12_semantic_source_path_integral.json"
REPORT = TENSOR_ROOT / "BILIN12_SEMANTIC_SOURCE_PATH_INTEGRAL.md"
FIGURE = TENSOR_ROOT / "bilin12_semantic_source_path_integral.png"


def quadrature_rules():
    rules = {
        "base_jvp": ([0.0], [1.0]),
        "midpoint": ([0.5], [1.0]),
        "trapezoid": ([0.0, 1.0], [0.5, 0.5]),
        "simpson": ([0.0, 0.5, 1.0], [1 / 6, 4 / 6, 1 / 6]),
    }
    for count in (2, 4, 8):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        rules[f"gauss{count}"] = (((nodes + 1) / 2).tolist(), (weights / 2).tolist())
    return rules


def tangent_at(model, base_x0, tangent, layers, destination, location):
    point = base_x0 + location * tangent
    function = lambda value: differentiable_source_trace(model, value, layers, destination)
    _, linear = torch.func.jvp(function, (point,), (tangent,))
    return tuple(value.float() for value in linear)


def finite_metric(base_logits, target_delta, prediction_delta):
    target, _, _, error = fisher_terms(base_logits, target_delta, prediction_delta)
    return float(target.sum()), float(error.sum()), float(1 - error.sum() / target.sum().clamp_min(1e-30))


def finish(rows):
    target = sum(row[0] for row in rows)
    error = sum(row[1] for row in rows)
    values = [row[2] for row in rows]
    return {
        "examples": len(rows),
        "fisher_fidelity": 1 - error / max(target, 1e-30),
        "mean_individual_fidelity": float(np.mean(values)),
        "q05_individual_fidelity": float(np.quantile(values, 0.05)),
        "minimum_individual_fidelity": float(np.min(values)),
    }


def aggregate(records, labels, selector=lambda record: True, layer_filter=None):
    rows = defaultdict(list)
    for record in records:
        if not selector(record):
            continue
        for layer_name, layer in record["layers"].items():
            if layer_filter is not None and layer_name != str(layer_filter):
                continue
            for label in labels:
                rows[label].append(tuple(layer["metrics"][label]))
    return {label: finish(values) for label, values in rows.items()}


def main():
    red = json.loads(CONFIG.read_text())
    config = json.loads(Path(red["source_config"]).read_text())
    prior = json.loads(Path(red["source_output"]).read_text())
    semantic = json.loads(Path(config["semantic_source"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    sources = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"] : spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    rules = quadrature_rules()
    unique_nodes = sorted({float(node) for nodes, _ in rules.values() for node in nodes})
    labels = [method for method in red["methods"]]
    labels += [f"select/{method}/{budget}" for method in red["methods"] for budget in red["selection_budgets"]]
    records = []

    for prior_record in prior["records"]:
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
            input_tangent = changed_x0 - base_x0
        with torch.enable_grad():
            local = {
                node: tangent_at(model, base_x0, input_tangent, config["layers"], config["prediction_position"], node)
                for node in unique_nodes
            }

        predictions = {}
        for method, (nodes, weights) in rules.items():
            predictions[method] = tuple(
                sum(float(weight) * local[float(node)][layer_index] for node, weight in zip(nodes, weights))
                for layer_index in range(len(config["layers"]))
            )

        layer_records = {}
        for layer_index, layer in enumerate(config["layers"]):
            base = base_rows[layer]
            changed_row = changed_rows[layer]
            block = model.blocks[layer]
            exact_base = source_write_contributions(block, base["details"], config["prediction_position"]).float()
            exact_changed = source_write_contributions(block, changed_row["details"], config["prediction_position"]).float()
            exact_delta = exact_changed - exact_base
            full_write = base["details"]["write"].clone()
            full_write[:, config["prediction_position"]] = changed_row["details"]["write"][:, config["prediction_position"]]
            base_logits, target_logits = logits_from_attention_writes(model, layer, base, torch.stack((base["details"]["write"], full_write)))
            target_delta = target_logits - base_logits

            worlds = []
            world_labels = []
            for method in red["methods"]:
                predicted_delta = predictions[method][layer_index]
                write = base["details"]["write"].clone()
                write[:, config["prediction_position"]] += predicted_delta.sum(1).to(write.dtype)
                worlds.append(write)
                world_labels.append(method)
                order = predicted_delta.norm(dim=-1).argsort(-1, descending=True)
                for budget in red["selection_budgets"]:
                    mask = torch.zeros(1, config["sequence_length"], device="cuda")
                    mask.scatter_(1, order[:, :budget], 1)
                    selected = (mask[:, :, None] * exact_delta).sum(1)
                    selected_write = base["details"]["write"].clone()
                    selected_write[:, config["prediction_position"]] += selected.to(selected_write.dtype)
                    worlds.append(selected_write)
                    world_labels.append(f"select/{method}/{budget}")
            logits = logits_from_attention_writes(model, layer, base, torch.stack(worlds))
            metrics = {
                label: finite_metric(base_logits, target_delta, logits[index] - base_logits)
                for index, label in enumerate(world_labels)
            }
            tensor_rows = {
                method: {
                    "cosine": float(cosine(predictions[method][layer_index], exact_delta)[0]),
                    "relative_error": float((predictions[method][layer_index] - exact_delta).norm() / exact_delta.norm().clamp_min(1e-30)),
                }
                for method in red["methods"]
            }
            base_order = local[0.0][layer_index].norm(dim=-1).argsort(-1, descending=True)[0, :16].cpu().tolist()
            endpoint_order = local[1.0][layer_index].norm(dim=-1).argsort(-1, descending=True)[0, :16].cpu().tolist()
            layer_records[str(layer)] = {
                "metrics": {label: list(value) for label, value in metrics.items()},
                "tensor": tensor_rows,
                "endpoint_top16_jaccard": jaccard(base_order, endpoint_order),
            }
        records.append({
            "corpus": corpus,
            "family": prior_record["family"],
            "pair": prior_record["pair"],
            "layers": layer_records,
        })
        print(corpus, prior_record["family"], prior_record["pair"], flush=True)

    overall = aggregate(records, labels)
    by_layer = {str(layer): aggregate(records, labels, layer_filter=layer) for layer in config["layers"]}
    by_corpus = {corpus: aggregate(records, labels, lambda row, corpus=corpus: row["corpus"] == corpus) for corpus in config["evaluation"]}
    by_family = {family: aggregate(records, labels, lambda row, family=family: row["family"] == family) for family in semantic["families"]}
    layer_rows = [layer for record in records for layer in record["layers"].values()]
    gauss2_cosines = np.asarray([layer["tensor"]["gauss2"]["cosine"] for layer in layer_rows])
    path_jaccards = np.asarray([layer["endpoint_top16_jaccard"] for layer in layer_rows])
    subgroup_gauss4 = [
        summary["gauss4"]["fisher_fidelity"]
        for collection in (by_layer, by_corpus, by_family)
        for summary in collection.values()
    ]
    gauss_values = [overall[method]["fisher_fidelity"] for method in ("gauss2", "gauss4", "gauss8")]
    hypotheses = {
        "P1_base_jvp_reproduces_the_prior_failure": {"supported": bool(0.50 <= overall["base_jvp"]["fisher_fidelity"] <= 0.65), "fidelity": overall["base_jvp"]["fisher_fidelity"]},
        "P2_midpoint_substantially_improves_finite_reconstruction": {"supported": bool(overall["midpoint"]["fisher_fidelity"] - overall["base_jvp"]["fisher_fidelity"] >= 0.20), "improvement": overall["midpoint"]["fisher_fidelity"] - overall["base_jvp"]["fisher_fidelity"]},
        "P3_two_point_gauss_is_a_useful_finite_program": {"supported": bool(overall["gauss2"]["fisher_fidelity"] >= 0.85), "fidelity": overall["gauss2"]["fisher_fidelity"]},
        "P4_four_point_gauss_nearly_reconstructs_the_finite_effect": {"supported": bool(overall["gauss4"]["fisher_fidelity"] >= 0.97 and overall["gauss4"]["q05_individual_fidelity"] >= 0.85), "aggregate": overall["gauss4"]["fisher_fidelity"], "q05": overall["gauss4"]["q05_individual_fidelity"]},
        "P5_quadrature_error_converges": {"supported": bool(gauss_values[0] <= gauss_values[1] <= gauss_values[2]), "fidelities": gauss_values},
        "P6_gauss2_recovers_source_tensor_direction": {"supported": bool(np.median(gauss2_cosines) >= 0.90), "median_cosine": float(np.median(gauss2_cosines)), "q05_cosine": float(np.quantile(gauss2_cosines, 0.05))},
        "P7_support_was_easier_than_amplitude": {"supported": bool(overall["gauss4"]["fisher_fidelity"] - overall["base_jvp"]["fisher_fidelity"] >= 0.25 and overall["select/gauss4/32"]["fisher_fidelity"] - overall["select/base_jvp/32"]["fisher_fidelity"] <= 0.02), "reconstruction_gain": overall["gauss4"]["fisher_fidelity"] - overall["base_jvp"]["fisher_fidelity"], "selection_gain": overall["select/gauss4/32"]["fisher_fidelity"] - overall["select/base_jvp/32"]["fisher_fidelity"]},
        "P8_path_integral_transfers_across_layers_corpora_and_families": {"supported": bool(min(subgroup_gauss4) >= 0.85), "minimum": min(subgroup_gauss4)},
        "P9_local_support_is_stable_along_the_path": {"supported": bool(np.median(path_jaccards) >= 0.60), "median_jaccard": float(np.median(path_jaccards))},
        "P10_four_points_are_close_to_eight": {"supported": bool(overall["gauss8"]["fisher_fidelity"] - overall["gauss4"]["fisher_fidelity"] <= 0.01), "improvement": overall["gauss8"]["fisher_fidelity"] - overall["gauss4"]["fisher_fidelity"]},
    }
    OUTPUT.write_text(json.dumps({"config": red, "source_config": config, "hypotheses": hypotheses, "overall": overall, "by_layer": by_layer, "by_corpus": by_corpus, "by_family": by_family, "records": records}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), layout="constrained")
    x = np.arange(len(red["methods"]))
    axes[0].plot(x, [overall[method]["fisher_fidelity"] for method in red["methods"]], marker="o")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(red["methods"], rotation=30, ha="right")
    axes[0].set_title("Finite reconstruction from path JVPs")
    axes[0].set_ylabel("Fisher fidelity")
    for budget in red["selection_budgets"]:
        axes[1].plot(x, [overall[f"select/{method}/{budget}"]["fisher_fidelity"] for method in red["methods"]], marker="o", label=f"top {budget}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(red["methods"], rotation=30, ha="right")
    axes[1].set_title("Exact-edge selection by path predictor")
    axes[2].hist(path_jaccards, bins=14, alpha=0.7, label="base/endpoint support")
    axes[2].hist(gauss2_cosines, bins=14, alpha=0.5, label="Gauss2/exact cosine")
    axes[2].set_title("Individual path diagnostics")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    fig.savefig(FIGURE, dpi=180)
    plt.close(fig)

    lines = [
        "# Path-integrated tensor JVPs for finite semantic edits",
        "",
        "The normalized source-token embedding is moved along the straight chord to the replacement-token embedding. Source-contribution Jacobian-vector products are integrated along this path with midpoint, trapezoid, Simpson, and Gauss-Legendre rules. This tests whether the failed base-point amplitude prediction reflects smooth curvature or a discontinuous change of computational regime.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Path-integrated JVPs](bilin12_semantic_source_path_integral.png)",
        "",
        "| Method | Full predicted-delta fidelity | Top-16 selection | Top-32 selection | Individual q05 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in red["methods"]:
        lines.append(f"| {method} | {overall[method]['fisher_fidelity']:.3f} | {overall[f'select/{method}/16']['fisher_fidelity']:.3f} | {overall[f'select/{method}/32']['fisher_fidelity']:.3f} | {overall[method]['q05_individual_fidelity']:.3f} |")
    lines += [
        "",
        f"Gauss-2 median predicted/exact source-tensor cosine: **{np.median(gauss2_cosines):.3f}**. Base-versus-endpoint derivative top-16 Jaccard: **{np.median(path_jaccards):.3f}**. Worst layer/corpus/family Gauss-4 aggregate fidelity: **{min(subgroup_gauss4):.3f}**.",
        "",
        "Path integration is an explicit finite-effect explanation in the model's native tensor coordinates. It uses the proposed input edit and several local linear evaluations; it is not a context-free feature dictionary or a claim that a single neuron carries the edit.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
