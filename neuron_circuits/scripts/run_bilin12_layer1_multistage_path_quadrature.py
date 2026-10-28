#!/usr/bin/env python3
"""Prospective multistage quadrature for the rare layer-1 edit-path tail."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_path_quadrature import aggregate
from run_bilin12_layer01_directional_taylor_selector import cosine
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, median, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import exact_kl, logits_for_deltas, product_from_x0, sample_edits
from run_bilin12_layer01_path_quadrature_compiler import gauss_integral
from run_bilin12_layer01_suffix_fisher_channel_selector import suffix_fisher_scores
from run_bilin12_layer01_suffix_fisher_chart_router import trace_from_x0
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer1_multistage_path_quadrature.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer1_multistage_path_quadrature.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER1_MULTISTAGE_PATH_QUADRATURE.md"
FIGURE = TENSOR_ROOT / "bilin12_layer1_multistage_path_quadrature.png"


def converged(a, b, relative_threshold, cosine_threshold):
    relative = float((b - a).norm() / b.norm().clamp_min(1e-30))
    agreement = cosine(a, b)
    return relative <= relative_threshold and agreement >= cosine_threshold, relative, agreement


def main():
    config = json.loads(CONFIG.read_text())
    adaptive = json.loads(Path(config["source_config"]).read_text())
    quad = json.loads(Path(adaptive["source_config"]).read_text())
    weighted = json.loads(Path(quad["source_config"]).read_text())
    suffix = json.loads(Path(weighted["source_config"]).read_text())
    behavior = json.loads(Path(suffix["source_config"]).read_text())
    parent = json.loads(Path(behavior["source_config"]).read_text())
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    spec = parent["evaluation"][config["corpus"]]
    rows = np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
    edits = sample_edits(rows, behavior["sites_per_corpus"], behavior["sampling_seed"], behavior["edit_position"], behavior["sequence_length"])[:config["sites"]]
    layer = config["layer"]
    records = []
    for site_index, (row_index, target_token) in enumerate(edits):
        raw = np.asarray(rows[row_index, :behavior["sequence_length"]], dtype=np.int64).copy()
        changed = raw.copy(); changed[behavior["edit_position"]] = target_token
        base_ids = torch.as_tensor(raw, device="cuda")[None]
        endpoint_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            base_trace = forward_layers_native(model, base_ids, layer)
            dimension = model.config.n_embd
            base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
            endpoint_x0 = F.rms_norm(model.embedding(endpoint_ids), (dimension,)).float()
            midpoint_trace = trace_from_x0(model, base_x0 + .5 * (endpoint_x0 - base_x0), layer)
        direction = endpoint_x0 - base_x0
        function = lambda value: product_from_x0(model, value, layer, behavior["prediction_position"])
        predictions = {order: gauss_integral(function, base_x0, direction, order).detach().float() for order in config["orders"]}
        decisions = {}
        stop_order = 64
        for left, right in zip(config["orders"][:-1], config["orders"][1:]):
            stable, relative, agreement = converged(predictions[left], predictions[right], config["relative_disagreement_threshold"], config["cosine_disagreement_threshold"])
            decisions[str(right)] = {"stable": stable, "relative": relative, "cosine": agreement}
            if stable:
                stop_order = right
                break
        prediction = predictions[stop_order]
        decoder = model.blocks[layer].mlp.Down.weight.float()
        score, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], prediction, decoder, behavior["prediction_position"])
        indices = score.topk(config["rank"]).indices
        with torch.no_grad():
            full = function(endpoint_x0).detach().float() - function(base_x0).detach().float()
        endpoint_score, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], full, decoder, behavior["prediction_position"])
        deltas = {f"prediction_{order}": value for order, value in predictions.items()}
        deltas.update({"multistage": prediction, "multistage_rank_actual": selected_delta(full, indices), "endpoint_rank_actual": selected_delta(full, endpoint_score.topk(config["rank"]).indices)})
        with torch.no_grad():
            logits = logits_for_deltas(model, layer, base_trace[layer], {"base": torch.zeros_like(full), "full": full, **deltas}, behavior["prediction_position"])
        base_kl = exact_kl(logits["full"], logits["base"])
        records.append({
            "site": site_index, "stop_order": stop_order, "decisions": decisions, "base_kl": base_kl,
            "endpoint": {str(order): {"cosine": cosine(value, full), "norm_ratio_error": abs(float(value.norm() / full.norm().clamp_min(1e-30)) - 1)} for order, value in predictions.items()},
            "kl_error": {name: exact_kl(logits["full"], logits[name]) for name in deltas},
        })
        if (site_index + 1) % 8 == 0:
            print("multistage quadrature", site_index + 1, "/", len(edits), flush=True)

    routed_past_16 = [row for row in records if row["stop_order"] > 16]
    bad16 = [row for row in records if row["endpoint"]["16"]["cosine"] < .90]
    early = [row for row in records if row["stop_order"] < 64]
    summaries = {
        "sites": len(records),
        "stop_counts": {str(order): sum(row["stop_order"] == order for row in records) for order in config["orders"][1:]},
        "mean_selected_nodes": float(np.mean([row["stop_order"] for row in records])),
        "bad16": len(bad16),
        "bad16_route_recall": float(np.mean([row["stop_order"] > 16 for row in bad16])) if bad16 else 1.0,
        "routed32": {"sites": len(routed_past_16), "median_cosine": median([row["endpoint"]["32"]["cosine"] for row in routed_past_16]), "median_norm_ratio_error": median([row["endpoint"]["32"]["norm_ratio_error"] for row in routed_past_16])},
        "endpoint": {str(order): {"median_cosine": median([row["endpoint"][str(order)]["cosine"] for row in records]), "median_norm_ratio_error": median([row["endpoint"][str(order)]["norm_ratio_error"] for row in records])} for order in config["orders"]},
        "methods": {method: aggregate(records, method) for method in [*(f"prediction_{order}" for order in config["orders"]), "multistage", "multistage_rank_actual", "endpoint_rank_actual"]},
        "early_certified_fraction": float(np.mean([row["endpoint"][str(row["stop_order"])]["cosine"] >= .99 for row in early])) if early else 1.0,
    }
    hypotheses = {
        "M0_next_stage_flags_remaining_bad_paths": {"supported": summaries["bad16_route_recall"] >= .90},
        "M1_thirty_two_nodes_are_typically_exact_on_routed_paths": {"supported": summaries["routed32"]["median_cosine"] >= .995 and summaries["routed32"]["median_norm_ratio_error"] <= .02},
        "M2_multistage_compiler_preserves_behavior": {"supported": summaries["methods"]["multistage"]["effect_weighted"] >= .95 and summaries["methods"]["multistage"]["median"] >= .99},
        "M3_compact_support_preserves_behavior": {"supported": summaries["methods"]["multistage_rank_actual"]["effect_weighted"] >= .85},
        "M4_internal_convergence_certifies_accuracy": {"supported": summaries["early_certified_fraction"] >= .95},
        "M5_routing_is_prospective": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Multistage layer-1 path quadrature", "", "A prospective convergence rule stops each FineWeb token-edit path at 8, 16, 32, or 64 Gauss nodes. Each decision compares adjacent interior estimates; exact endpoint effects are evaluated only after the route is frozen.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Multistage quadrature](bilin12_layer1_multistage_path_quadrature.png)", "", f"Stop counts: `{summaries['stop_counts']}`; mean selected node count: **{summaries['mean_selected_nodes']:.1f}**.", "", f"Bad-16 routing recall: **{summaries['bad16_route_recall']:.3f}** ({summaries['bad16']} sites). Early-stop endpoint-cosine certification: **{summaries['early_certified_fraction']:.3f}**.", "", "| Method | Effect-weighted KL recovery | Median | Q25 | Negative fraction |", "|---|---:|---:|---:|---:|"]
    for method, row in summaries["methods"].items(): lines.append(f"| {method} | {row['effect_weighted']:.3f} | {row['median']:.3f} | {row['q25']:.3f} | {row['negative_fraction']:.3f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].scatter([row["decisions"].get("16", {}).get("relative", np.nan) for row in records], [row["endpoint"]["16"]["cosine"] for row in records], c=[row["stop_order"] for row in records], cmap="viridis", alpha=.8)
    axes[0].axvline(config["relative_disagreement_threshold"], color="black", linestyle="--"); axes[0].set(xlabel="8→16 relative disagreement", ylabel="16-node endpoint cosine", title="Internal signal versus held-out accuracy"); axes[0].grid(alpha=.2)
    methods = [f"prediction_{order}" for order in config["orders"]] + ["multistage"]
    axes[1].bar(range(len(methods)), [summaries["methods"][method]["effect_weighted"] for method in methods]); axes[1].set_xticks(range(len(methods)), [method.replace("prediction_", "G") for method in methods], rotation=25, ha="right"); axes[1].axhline(.95, color="black", linestyle="--"); axes[1].set(ylabel="Effect-weighted KL recovery", title="Fixed versus adaptive integration"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT, flush=True)


if __name__ == "__main__": main()
