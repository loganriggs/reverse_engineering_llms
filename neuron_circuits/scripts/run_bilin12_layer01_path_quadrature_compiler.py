#!/usr/bin/env python3
"""Compile finite product changes by Gaussian integration of exact path JVPs."""

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
from run_bilin12_layer01_suffix_fisher_channel_selector import suffix_fisher_scores
from run_bilin12_layer01_suffix_fisher_chart_router import trace_from_x0
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_path_quadrature_compiler.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_path_quadrature_compiler.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_PATH_QUADRATURE_COMPILER.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_path_quadrature_compiler.png"


def gauss_integral(function, base, direction, order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    total = None
    for node, weight in zip(nodes, weights):
        location = .5 * (float(node) + 1)
        with torch.enable_grad():
            _, tangent = torch.func.jvp(function, (base + location * direction,), (direction,))
        term = .5 * float(weight) * tangent
        total = term if total is None else total + term
    return total


def aggregate(rows, method):
    denominator = sum(row["base_kl"] for row in rows)
    values = np.asarray([1 - row["kl_error"][method] / max(row["base_kl"], 1e-30) for row in rows])
    return {
        "effect_weighted_kl_recovery": float(1 - sum(row["kl_error"][method] for row in rows) / max(denominator, 1e-30)),
        "median_per_example_recovery": float(np.median(values)),
        "q25_per_example_recovery": float(np.quantile(values, .25)),
        "negative_recovery_fraction": float(np.mean(values < 0)),
    }


def main():
    red = json.loads(CONFIG.read_text())
    weighted = json.loads(Path(red["source_config"]).read_text())
    suffix = json.loads(Path(weighted["source_config"]).read_text())
    behavior = json.loads(Path(suffix["source_config"]).read_text())
    parent = json.loads(Path(behavior["source_config"]).read_text())
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]] for corpus, spec in parent["evaluation"].items()}
    records = {str(layer): [] for layer in parent["layers"]}
    continuous_methods = [f"gauss{order}_full_predicted" for order in red["quadrature_orders"]]
    methods = continuous_methods + ["gauss1_rank_actual", "gauss3_rank_predicted", "gauss3_rank_actual", "endpoint_rank_actual"]

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        edits = sample_edits(rows, behavior["sites_per_corpus"], behavior["sampling_seed"] + corpus_index, behavior["edit_position"], behavior["sequence_length"])[:red["sites_per_corpus"]]
        for site_index, (row_index, target_token) in enumerate(edits):
            raw = np.asarray(rows[row_index, :behavior["sequence_length"]], dtype=np.int64).copy()
            changed = raw.copy(); changed[behavior["edit_position"]] = target_token
            base_ids = torch.as_tensor(raw, device="cuda")[None]; endpoint_ids = torch.as_tensor(changed, device="cuda")[None]
            with torch.no_grad():
                base_trace = forward_layers_native(model, base_ids, max(parent["layers"]))
                dimension = model.config.n_embd
                base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                endpoint_x0 = F.rms_norm(model.embedding(endpoint_ids), (dimension,)).float()
                midpoint_trace = trace_from_x0(model, base_x0 + .5 * (endpoint_x0 - base_x0), max(parent["layers"]))
            direction = endpoint_x0 - base_x0
            for layer in parent["layers"]:
                function = lambda value, layer=layer: product_from_x0(model, value, layer, behavior["prediction_position"])
                predictions = {order: gauss_integral(function, base_x0, direction, order).detach().float() for order in red["quadrature_orders"]}
                decoder = model.blocks[layer].mlp.Down.weight.float()
                score1, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], predictions[1], decoder, behavior["prediction_position"])
                score3, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], predictions[3], decoder, behavior["prediction_position"])
                indices1 = score1.topk(red["rank"]).indices; indices3 = score3.topk(red["rank"]).indices
                # All quadrature vectors and predicted supports are now frozen.
                with torch.no_grad(): full = function(endpoint_x0).detach().float() - function(base_x0).detach().float()
                endpoint_score, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], full, decoder, behavior["prediction_position"])
                endpoint_indices = endpoint_score.topk(red["rank"]).indices
                deltas = {f"gauss{order}_full_predicted": predictions[order] for order in red["quadrature_orders"]}
                deltas.update({
                    "gauss1_rank_actual": selected_delta(full, indices1),
                    "gauss3_rank_predicted": selected_delta(predictions[3], indices3),
                    "gauss3_rank_actual": selected_delta(full, indices3),
                    "endpoint_rank_actual": selected_delta(full, endpoint_indices),
                })
                with torch.no_grad(): logits = logits_for_deltas(model, layer, base_trace[layer], {"base": torch.zeros_like(full), "full": full, **deltas}, behavior["prediction_position"])
                base_kl = exact_kl(logits["full"], logits["base"])
                records[str(layer)].append({
                    "corpus": corpus, "site": site_index, "base_kl": base_kl,
                    "prediction": {str(order): {"cosine": cosine(predictions[order], full), "norm_ratio": float(predictions[order].norm() / full.norm().clamp_min(1e-30))} for order in red["quadrature_orders"]},
                    "kl_error": {method: exact_kl(logits["full"], logits[method]) for method in methods},
                })
            if (site_index + 1) % 8 == 0: print("quadrature", corpus, site_index + 1, "/", len(edits), flush=True)

    summaries = {str(layer): {} for layer in parent["layers"]}
    for layer in parent["layers"]:
        for corpus in arrays:
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summaries[str(layer)][corpus] = {
                "sites": len(selected),
                "prediction": {str(order): {"median_cosine": median([row["prediction"][str(order)]["cosine"] for row in selected]), "median_norm_ratio": median([row["prediction"][str(order)]["norm_ratio"] for row in selected]), "median_norm_ratio_error": median([abs(row["prediction"][str(order)]["norm_ratio"] - 1) for row in selected])} for order in red["quadrature_orders"]},
                "methods": {method: aggregate(selected, method) for method in methods},
            }
    cell = lambda layer, corpus: summaries[str(layer)][corpus]
    value = lambda layer, corpus, method: cell(layer, corpus)["methods"][method]["effect_weighted_kl_recovery"]
    cells = [(layer, corpus) for layer in parent["layers"] for corpus in arrays]
    hypotheses = {
        "Q0_quadrature_direction_converges": {"supported": all(all(cell(layer, corpus)["prediction"][str(order)]["median_cosine"] >= cell(layer, corpus)["prediction"][str(order-1)]["median_cosine"] - .005 for order in red["quadrature_orders"][1:]) for layer, corpus in cells)},
        "Q1_three_nodes_predict_the_finite_vector": {"supported": all(cell(layer, corpus)["prediction"]["3"]["median_cosine"] >= .98 and cell(layer, corpus)["prediction"]["3"]["median_norm_ratio_error"] <= .10 for layer, corpus in cells)},
        "Q2_three_node_continuous_patch_preserves_behavior": {"supported": all(value(layer, corpus, "gauss3_full_predicted") >= .80 and cell(layer, corpus)["methods"]["gauss3_full_predicted"]["median_per_example_recovery"] >= .70 for layer, corpus in cells)},
        "Q3_three_node_support_preserves_behavior": {"supported": all(value(layer, corpus, "gauss3_rank_actual") >= .80 for layer, corpus in cells)},
        "Q4_quadrature_repairs_midpoint_overshoot": {"supported": all(value(layer, corpus, "gauss3_full_predicted") >= value(layer, corpus, "gauss1_full_predicted") + .50 for layer, corpus in cells)},
        "Q5_four_nodes_have_diminishing_returns": {"supported": all(abs(value(layer, corpus, "gauss4_full_predicted") - value(layer, corpus, "gauss3_full_predicted")) <= .03 for layer, corpus in cells)},
        "Q6_quadrature_compiler_transfers_ood": {"supported": all(abs(value(layer, "fineweb", "gauss3_full_predicted") - value(layer, "tinystories", "gauss3_full_predicted")) <= .10 for layer in parent["layers"])},
        "Q7_quadrature_is_prospective": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Path-quadrature compiler", "", "Finite product changes are predicted by Gauss–Legendre integration of exact directional JVPs at strictly interior points of the embedding edit path. Endpoint product values are held out until continuous predictions and suffix-Fisher supports are frozen.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Path quadrature compiler](bilin12_layer01_path_quadrature_compiler.png)", "", "| Layer | Corpus | Cosine 1/2/3/4 | Norm ratio 1/2/3/4 | Continuous KL 1/2/3/4 | Rank actual 1/3/endpoint | Gauss3 rank predicted |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer, corpus in cells:
        row = cell(layer, corpus)
        joined = lambda key: "/".join(f"{row['prediction'][str(order)][key]:.3f}" for order in red["quadrature_orders"])
        continuous = "/".join(f"{value(layer, corpus, f'gauss{order}_full_predicted'):.3f}" for order in red["quadrature_orders"])
        ranks = "/".join(f"{value(layer, corpus, method):.3f}" for method in ("gauss1_rank_actual", "gauss3_rank_actual", "endpoint_rank_actual"))
        lines.append(f"| {layer} | {corpus} | {joined('median_cosine')} | {joined('median_norm_ratio')} | {continuous} | {ranks} | {value(layer, corpus, 'gauss3_rank_predicted'):.3f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for layer in parent["layers"]:
        for corpus in arrays:
            axes[layer].plot(red["quadrature_orders"], [value(layer, corpus, f"gauss{order}_full_predicted") for order in red["quadrature_orders"]], marker="o", label=corpus)
        axes[layer].set_title(f"Layer {layer}"); axes[layer].set_xlabel("Gauss nodes"); axes[layer].set_ylabel("Continuous KL recovery"); axes[layer].grid(alpha=.2); axes[layer].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT, flush=True)


if __name__ == "__main__": main()
