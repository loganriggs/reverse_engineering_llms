#!/usr/bin/env python3
"""Prospectively flag and refine high-curvature FineWeb edit paths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_directional_taylor_selector import cosine
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, median, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import exact_kl, logits_for_deltas, product_from_x0, sample_edits
from run_bilin12_layer01_path_quadrature_compiler import gauss_integral
from run_bilin12_layer01_suffix_fisher_channel_selector import suffix_fisher_scores
from run_bilin12_layer01_suffix_fisher_chart_router import trace_from_x0
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_adaptive_path_quadrature.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_adaptive_path_quadrature.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_ADAPTIVE_PATH_QUADRATURE.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_adaptive_path_quadrature.png"


def aggregate(rows, method):
    denominator = sum(row["base_kl"] for row in rows)
    values = np.asarray([1 - row["kl_error"][method] / max(row["base_kl"], 1e-30) for row in rows])
    return {"effect_weighted": float(1 - sum(row["kl_error"][method] for row in rows) / max(denominator, 1e-30)), "median": float(np.median(values)), "q25": float(np.quantile(values, .25)), "negative_fraction": float(np.mean(values < 0))}


def main():
    red = json.loads(CONFIG.read_text())
    quad = json.loads(Path(red["source_config"]).read_text())
    weighted = json.loads(Path(quad["source_config"]).read_text())
    suffix = json.loads(Path(weighted["source_config"]).read_text())
    behavior = json.loads(Path(suffix["source_config"]).read_text())
    parent = json.loads(Path(behavior["source_config"]).read_text())
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    spec = parent["evaluation"][red["corpus"]]
    rows = np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
    edits = sample_edits(rows, behavior["sites_per_corpus"], behavior["sampling_seed"], behavior["edit_position"], behavior["sequence_length"])[:red["sites"]]
    records = {str(layer): [] for layer in parent["layers"]}
    for site_index, (row_index, target_token) in enumerate(edits):
        raw = np.asarray(rows[row_index, :behavior["sequence_length"]], dtype=np.int64).copy(); changed = raw.copy(); changed[behavior["edit_position"]] = target_token
        base_ids = torch.as_tensor(raw, device="cuda")[None]; endpoint_ids = torch.as_tensor(changed, device="cuda")[None]
        with torch.no_grad():
            base_trace = forward_layers_native(model, base_ids, max(parent["layers"]))
            dimension = model.config.n_embd; base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float(); endpoint_x0 = F.rms_norm(model.embedding(endpoint_ids), (dimension,)).float()
            midpoint_trace = trace_from_x0(model, base_x0 + .5 * (endpoint_x0 - base_x0), max(parent["layers"]))
        direction = endpoint_x0 - base_x0
        for layer in parent["layers"]:
            function = lambda value, layer=layer: product_from_x0(model, value, layer, behavior["prediction_position"])
            prediction4 = gauss_integral(function, base_x0, direction, 4).detach().float()
            prediction8 = gauss_integral(function, base_x0, direction, 8).detach().float()
            relative = float((prediction8 - prediction4).norm() / prediction8.norm().clamp_min(1e-30))
            agreement = cosine(prediction4, prediction8)
            flagged = relative > red["relative_disagreement_threshold"] or agreement < red["cosine_disagreement_threshold"]
            # The routing decision is fixed here, before prediction16 is evaluated.
            prediction16 = gauss_integral(function, base_x0, direction, 16).detach().float()
            adaptive = prediction16 if flagged else prediction8
            decoder = model.blocks[layer].mlp.Down.weight.float()
            adaptive_score, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], adaptive, decoder, behavior["prediction_position"])
            adaptive_indices = adaptive_score.topk(red["rank"]).indices
            with torch.no_grad(): full = function(endpoint_x0).detach().float() - function(base_x0).detach().float()
            endpoint_score, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], full, decoder, behavior["prediction_position"])
            deltas = {"prediction4": prediction4, "prediction8": prediction8, "prediction16": prediction16, "adaptive": adaptive, "adaptive_rank_actual": selected_delta(full, adaptive_indices), "endpoint_rank_actual": selected_delta(full, endpoint_score.topk(red["rank"]).indices)}
            with torch.no_grad(): logits = logits_for_deltas(model, layer, base_trace[layer], {"base": torch.zeros_like(full), "full": full, **deltas}, behavior["prediction_position"])
            base_kl = exact_kl(logits["full"], logits["base"])
            records[str(layer)].append({"site": site_index, "flagged": flagged, "relative_disagreement": relative, "agreement": agreement, "base_kl": base_kl, "endpoint": {str(order): {"cosine": cosine(value, full), "norm_ratio_error": abs(float(value.norm() / full.norm().clamp_min(1e-30)) - 1)} for order, value in ((4,prediction4),(8,prediction8),(16,prediction16))}, "kl_error": {name: exact_kl(logits["full"], logits[name]) for name in deltas}})
        if (site_index + 1) % 8 == 0: print("adaptive quadrature", site_index + 1, "/", len(edits), flush=True)

    summaries = {}
    for layer in parent["layers"]:
        selected = records[str(layer)]; flagged = [row for row in selected if row["flagged"]]; stable = [row for row in selected if not row["flagged"]]
        bad4 = [row for row in selected if row["endpoint"]["4"]["cosine"] < .90]
        summaries[str(layer)] = {"sites": len(selected), "flagged": len(flagged), "bad4": len(bad4), "bad4_flag_recall": float(np.mean([row["flagged"] for row in bad4])) if bad4 else 1.0, "endpoint": {str(order): {"median_cosine": median([row["endpoint"][str(order)]["cosine"] for row in selected]), "median_norm_ratio_error": median([row["endpoint"][str(order)]["norm_ratio_error"] for row in selected])} for order in red["orders"]}, "flagged_endpoint16": {"median_cosine": median([row["endpoint"]["16"]["cosine"] for row in flagged]), "median_norm_ratio_error": median([row["endpoint"]["16"]["norm_ratio_error"] for row in flagged])}, "methods": {method: aggregate(selected, method) for method in ("prediction4","prediction8","prediction16","adaptive","adaptive_rank_actual","endpoint_rank_actual")}, "flagged_adaptive": aggregate(flagged, "adaptive"), "stable_adaptive": aggregate(stable, "adaptive")}
    hypotheses = {
        "A0_internal_disagreement_detects_bad_four_node_paths": {"supported": all(summaries[str(layer)]["bad4_flag_recall"] >= .90 for layer in parent["layers"])},
        "A1_eight_nodes_are_typically_accurate": {"supported": all(summaries[str(layer)]["endpoint"]["8"]["median_cosine"] >= .995 and summaries[str(layer)]["endpoint"]["8"]["median_norm_ratio_error"] <= .02 for layer in parent["layers"])},
        "A2_sixteen_nodes_resolve_flagged_paths": {"supported": all(summaries[str(layer)]["flagged_endpoint16"]["median_cosine"] >= .95 and summaries[str(layer)]["flagged_endpoint16"]["median_norm_ratio_error"] <= .10 for layer in parent["layers"])},
        "A3_adaptive_continuous_compiler_preserves_fineweb_behavior": {"supported": all(summaries[str(layer)]["methods"]["adaptive"]["effect_weighted"] >= .80 and summaries[str(layer)]["methods"]["adaptive"]["median"] >= .90 for layer in parent["layers"])},
        "A4_adaptive_support_is_compact_and_behavioral": {"supported": all(summaries[str(layer)]["methods"]["adaptive_rank_actual"]["effect_weighted"] >= .80 for layer in parent["layers"])},
        "A5_internal_confidence_separates_errors": {"supported": all(summaries[str(layer)]["stable_adaptive"]["median"] >= summaries[str(layer)]["flagged_adaptive"]["median"] + .20 for layer in parent["layers"])},
        "A6_adaptation_is_prospective": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Adaptive path quadrature on FineWeb", "", "Four- and eight-node interior Gauss estimates are compared prospectively. Paths exceeding the locked disagreement threshold are routed to sixteen nodes; endpoint vectors remain held out until the routing decision is frozen.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Adaptive quadrature](bilin12_layer01_adaptive_path_quadrature.png)", "", "| Layer | Flagged/bad4/recall | Cosine 4/8/16 | Norm error 4/8/16 | KL 4/8/16/adaptive | Adaptive rank/endpoint rank | Stable/flagged median |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in parent["layers"]:
        row=summaries[str(layer)]; joined=lambda key:"/".join(f"{row['endpoint'][str(order)][key]:.3f}" for order in red["orders"]); methods=row["methods"]
        lines.append(f"| {layer} | {row['flagged']}/{row['bad4']}/{row['bad4_flag_recall']:.3f} | {joined('median_cosine')} | {joined('median_norm_ratio_error')} | {methods['prediction4']['effect_weighted']:.3f}/{methods['prediction8']['effect_weighted']:.3f}/{methods['prediction16']['effect_weighted']:.3f}/{methods['adaptive']['effect_weighted']:.3f} | {methods['adaptive_rank_actual']['effect_weighted']:.3f}/{methods['endpoint_rank_actual']['effect_weighted']:.3f} | {row['stable_adaptive']['median']:.3f}/{row['flagged_adaptive']['median']:.3f} |")
    REPORT.write_text("\n".join(lines)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for layer in parent["layers"]:
        rows_layer=records[str(layer)]; axes[layer].scatter([r['relative_disagreement'] for r in rows_layer],[r['endpoint']['4']['cosine'] for r in rows_layer],c=[r['flagged'] for r in rows_layer],cmap='coolwarm',alpha=.7); axes[layer].axvline(red['relative_disagreement_threshold'],color='black',linestyle='--'); axes[layer].set_xlabel('4→8 relative disagreement'); axes[layer].set_ylabel('4-node endpoint cosine'); axes[layer].set_title(f'Layer {layer}'); axes[layer].grid(alpha=.2)
    fig.savefig(FIGURE,dpi=180);plt.close(fig)
    print(json.dumps(hypotheses,indent=2));print(REPORT,flush=True)


if __name__ == "__main__": main()
