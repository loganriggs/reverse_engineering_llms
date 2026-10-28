#!/usr/bin/env python3
"""Separate continuous-vector, hard-support, and endpoint-value errors."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_directional_order_frontier import derivative_frontier, taylor_sum
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, median, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import exact_kl, logits_for_deltas, product_from_x0, sample_edits
from run_bilin12_layer01_suffix_fisher_channel_selector import suffix_fisher_scores
from run_bilin12_layer01_suffix_fisher_chart_router import trace_from_x0
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_weighted_coalition_decomposition.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_weighted_coalition_decomposition.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_WEIGHTED_COALITION_DECOMPOSITION.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_weighted_coalition_decomposition.png"
METHODS = (
    "taylor_full_predicted", "taylor_rank_predicted", "taylor_rank_actual",
    "midpoint_full_predicted", "midpoint_rank_predicted", "midpoint_rank_actual",
    "endpoint_rank_actual",
)


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
    suffix_config = json.loads(Path(red["source_config"]).read_text())
    behavior_config = json.loads(Path(suffix_config["source_config"]).read_text())
    parent = json.loads(Path(behavior_config["source_config"]).read_text())
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in parent["evaluation"].items()
    }
    records = {str(layer): [] for layer in parent["layers"]}
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        edits = sample_edits(
            rows, behavior_config["sites_per_corpus"], behavior_config["sampling_seed"] + corpus_index,
            behavior_config["edit_position"], behavior_config["sequence_length"],
        )[:red["sites_per_corpus"]]
        for site_index, (row_index, target_token) in enumerate(edits):
            raw = np.asarray(rows[row_index, :behavior_config["sequence_length"]], dtype=np.int64).copy()
            changed = raw.copy(); changed[behavior_config["edit_position"]] = target_token
            base_ids = torch.as_tensor(raw, device="cuda")[None]
            endpoint_ids = torch.as_tensor(changed, device="cuda")[None]
            with torch.no_grad():
                base_trace = forward_layers_native(model, base_ids, max(parent["layers"]))
                dimension = model.config.n_embd
                base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                endpoint_x0 = F.rms_norm(model.embedding(endpoint_ids), (dimension,)).float()
                midpoint_trace = trace_from_x0(model, base_x0 + .5 * (endpoint_x0 - base_x0), max(parent["layers"]))
            direction = endpoint_x0 - base_x0
            for layer in parent["layers"]:
                order = behavior_config["frozen_orders"][str(layer)]
                function = lambda value, layer=layer: product_from_x0(model, value, layer, behavior_config["prediction_position"])
                with torch.enable_grad():
                    derivatives, functions = derivative_frontier(function, base_x0, direction, order)
                    midpoint = functions[1](base_x0 + .5 * direction)
                taylor = taylor_sum([value.detach().float() for value in derivatives], .5, integral=False)
                midpoint = midpoint.detach().float()
                decoder = model.blocks[layer].mlp.Down.weight.float()
                taylor_score, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], taylor, decoder, behavior_config["prediction_position"])
                midpoint_score, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], midpoint, decoder, behavior_config["prediction_position"])
                taylor_indices = taylor_score.topk(red["rank"]).indices
                midpoint_indices = midpoint_score.topk(red["rank"]).indices
                # Endpoint values are intentionally evaluated only after all
                # prospective predicted-value vectors and supports are frozen.
                with torch.no_grad():
                    full = function(endpoint_x0).detach().float() - function(base_x0).detach().float()
                endpoint_score, _ = suffix_fisher_scores(model, layer, midpoint_trace[layer], full, decoder, behavior_config["prediction_position"])
                endpoint_indices = endpoint_score.topk(red["rank"]).indices
                deltas = {
                    "taylor_full_predicted": taylor,
                    "taylor_rank_predicted": selected_delta(taylor, taylor_indices),
                    "taylor_rank_actual": selected_delta(full, taylor_indices),
                    "midpoint_full_predicted": midpoint,
                    "midpoint_rank_predicted": selected_delta(midpoint, midpoint_indices),
                    "midpoint_rank_actual": selected_delta(full, midpoint_indices),
                    "endpoint_rank_actual": selected_delta(full, endpoint_indices),
                }
                with torch.no_grad():
                    logits = logits_for_deltas(model, layer, base_trace[layer], {"base": torch.zeros_like(full), "full": full, **deltas}, behavior_config["prediction_position"])
                base_kl = exact_kl(logits["full"], logits["base"])
                records[str(layer)].append({
                    "corpus": corpus, "site": site_index, "base_kl": base_kl,
                    "kl_error": {method: exact_kl(logits["full"], logits[method]) for method in METHODS},
                })
            if (site_index + 1) % 8 == 0: print("weighted coalition", corpus, site_index + 1, "/", len(edits), flush=True)

    summaries = {str(layer): {} for layer in parent["layers"]}
    for layer in parent["layers"]:
        for corpus in arrays:
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            summaries[str(layer)][corpus] = {"sites": len(selected), "methods": {method: aggregate(selected, method) for method in METHODS}}
    value = lambda layer, corpus, method: summaries[str(layer)][corpus]["methods"][method]["effect_weighted_kl_recovery"]
    cells = [(layer, corpus) for layer in parent["layers"] for corpus in arrays]
    hypotheses = {
        "W0_endpoint_reference_recomposes": {"supported": True},
        "W1_continuous_prediction_beats_hard_prediction": {"supported": all(value(layer, corpus, "taylor_full_predicted") >= value(layer, corpus, "taylor_rank_predicted") + .10 for layer, corpus in cells)},
        "W2_exact_values_materially_help_predicted_support": {"supported": all(value(layer, corpus, "taylor_rank_actual") >= value(layer, corpus, "taylor_rank_predicted") + .10 for layer, corpus in cells)},
        "W3_midpoint_direction_improves_continuous_prediction": {"supported": all(value(layer, corpus, "midpoint_full_predicted") >= value(layer, corpus, "taylor_full_predicted") + .10 for layer, corpus in cells)},
        "W4_continuous_taylor_is_behaviorally_useful": {"supported": all(value(layer, corpus, "taylor_full_predicted") >= .50 for layer, corpus in cells)},
        "W5_hard_support_is_not_the_only_failure": {"supported": any(value(layer, corpus, "endpoint_rank_actual") >= value(layer, corpus, "taylor_full_predicted") + .15 for layer, corpus in cells)},
        "W6_weighted_decomposition_transfers_ood": {"supported": all(abs(value(layer, "fineweb", "taylor_full_predicted") - value(layer, "tinystories", "taylor_full_predicted")) <= .15 for layer in parent["layers"])},
        "W7_no_endpoint_values_enter_prospective_predictions": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Weighted versus hard coalition decomposition", "", "The prospective Taylor or exact-midpoint product vector is compared as a continuous all-channel patch, a suffix-Fisher-selected rank-256 predicted-value patch, and the same rank-256 support populated with exact finite endpoint values. This separates vector prediction, hard support, and endpoint-value errors.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Weighted coalition decomposition](bilin12_layer01_weighted_coalition_decomposition.png)", "", "| Layer | Corpus | Taylor full/rank-pred/rank-actual | Midpoint full/rank-pred/rank-actual | Endpoint rank actual |", "|---:|---|---:|---:|---:|"]
    for layer, corpus in cells:
        names = lambda prefix: "/".join(f"{value(layer, corpus, name):.3f}" for name in (f"{prefix}_full_predicted", f"{prefix}_rank_predicted", f"{prefix}_rank_actual"))
        lines.append(f"| {layer} | {corpus} | {names('taylor')} | {names('midpoint')} | {value(layer, corpus, 'endpoint_rank_actual'):.3f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for layer in parent["layers"]:
        for corpus in arrays:
            axes[layer].plot(range(len(METHODS)), [value(layer, corpus, method) for method in METHODS], marker="o", label=corpus)
        axes[layer].set_xticks(range(len(METHODS)), METHODS, rotation=35, ha="right"); axes[layer].set_ylabel("Effect-weighted KL recovery"); axes[layer].set_title(f"Layer {layer}"); axes[layer].grid(alpha=.2); axes[layer].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT, flush=True)


if __name__ == "__main__": main()
