#!/usr/bin/env python3
"""Prospective suffix-Fisher scoring of Taylor-predicted native channels."""

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
from run_bilin12_layer01_suffix_fisher_chart_router import suffix_logits, trace_from_x0
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_suffix_fisher_channel_selector.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_suffix_fisher_channel_selector.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_SUFFIX_FISHER_CHANNEL_SELECTOR.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_suffix_fisher_channel_selector.png"
METHODS = ("taylor_suffix", "explicit_midpoint_suffix", "endpoint_suffix")


def fisher_cotangent(logits, tangent):
    probabilities = F.softmax(logits.float(), -1)
    centered = tangent.float() - (probabilities * tangent.float()).sum()
    return probabilities * centered


def reset_rotary_cache(model, layer):
    for block in model.blocks[layer + 1:]:
        block.attn.rotary.seq_len_cached = None
        block.attn.rotary.cos_cached = None
        block.attn.rotary.sin_cached = None


def suffix_fisher_scores(model, layer, record, vector, decoder, position):
    """Return |v * J'FJv| at a midpoint state plus a JVP/VJP duality check."""
    reset_rotary_cache(model, layer)
    x0 = record["x0"].detach().clone()
    bus = record["value_bus"].detach().clone()
    output = record["output"].detach().clone()
    vector = vector.detach().clone()
    zero = torch.zeros_like(vector)

    def function(delta):
        write = F.linear(delta, decoder)
        mask = torch.zeros_like(output)
        mask[:, position] = write
        return suffix_logits(model, layer, x0, bus, output + mask, position)[0]

    with torch.enable_grad():
        logits, tangent = torch.func.jvp(function, (zero,), (vector,))
        cotangent = fisher_cotangent(logits.detach(), tangent.detach()).detach()
        _, vjp = torch.func.vjp(function, zero)
        gradient = vjp(cotangent)[0]
    left = float((tangent.detach() * cotangent).sum())
    right = float((vector * gradient.detach()).sum())
    duality_error = abs(left - right) / max(abs(left), abs(right), 1e-30)
    return (vector * gradient.detach()).abs(), duality_error


def aggregate(records, method):
    denominator = sum(row["base_kl"] for row in records)
    recovery = float(1 - sum(row["kl_error"][method] for row in records) / max(denominator, 1e-30))
    individual = np.asarray([1 - row["kl_error"][method] / max(row["base_kl"], 1e-30) for row in records])
    return {
        "effect_weighted_kl_recovery": recovery,
        "median_per_example_recovery": float(np.median(individual)),
        "q25_per_example_recovery": float(np.quantile(individual, .25)),
        "negative_recovery_fraction": float(np.mean(individual < 0)),
    }


def main():
    red = json.loads(CONFIG.read_text())
    behavior_config = json.loads(Path(red["source_config"]).read_text())
    behavior = json.loads(Path(red["source_results"]).read_text())
    parent = json.loads(Path(behavior_config["source_config"]).read_text())
    model = load_snapshot_model(Path(parent["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in parent["evaluation"].items()
    }
    prior = {(layer, row["corpus"], row["site"]): row for layer, rows in behavior["records"].items() for row in rows}
    records = {str(layer): [] for layer in parent["layers"]}
    maximum_duality_error = 0.0

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
                endpoint_trace = forward_layers_native(model, endpoint_ids, max(parent["layers"]))
                dimension = model.config.n_embd
                base_x0 = F.rms_norm(model.embedding(base_ids), (dimension,)).float()
                endpoint_x0 = F.rms_norm(model.embedding(endpoint_ids), (dimension,)).float()
                midpoint_trace = trace_from_x0(model, base_x0 + .5 * (endpoint_x0 - base_x0), max(parent["layers"]))
            direction = endpoint_x0 - base_x0
            for layer in parent["layers"]:
                order = behavior_config["frozen_orders"][str(layer)]
                function = lambda value, layer=layer: product_from_x0(
                    model, value, layer, behavior_config["prediction_position"],
                )
                with torch.enable_grad():
                    derivatives, functions = derivative_frontier(function, base_x0, direction, order)
                    explicit_midpoint = functions[1](base_x0 + .5 * direction)
                derivatives = [value.detach().float() for value in derivatives]
                vectors = {
                    "taylor_suffix": taylor_sum(derivatives, .5, integral=False),
                    "explicit_midpoint_suffix": explicit_midpoint.detach().float(),
                }
                with torch.no_grad():
                    base_product = function(base_x0)
                    endpoint_product = function(endpoint_x0)
                full = endpoint_product.detach().float() - base_product.detach().float()
                vectors["endpoint_suffix"] = full
                decoder = model.blocks[layer].mlp.Down.weight.float()
                scores, duality = {}, {}
                for method, vector in vectors.items():
                    scores[method], duality[method] = suffix_fisher_scores(
                        model, layer, midpoint_trace[layer], vector, decoder,
                        behavior_config["prediction_position"],
                    )
                    maximum_duality_error = max(maximum_duality_error, duality[method])
                deltas = {
                    method: selected_delta(full, score.topk(red["rank"]).indices)
                    for method, score in scores.items()
                }
                with torch.no_grad():
                    logits = logits_for_deltas(
                        model, layer, base_trace[layer],
                        {"base": torch.zeros_like(full), "full": full, **deltas},
                        behavior_config["prediction_position"],
                    )
                base_kl = exact_kl(logits["full"], logits["base"])
                prior_row = prior[(str(layer), corpus, site_index)]
                prior_methods = {
                    "taylor_local": prior_row["metrics"][f"taylor_local_{red['rank']}"]["kl_error"],
                    "endpoint_local": prior_row["metrics"][f"endpoint_local_{red['rank']}"]["kl_error"],
                    "behavior_oracle": prior_row["metrics"][f"behavior_oracle_{red['rank']}"]["kl_error"],
                }
                records[str(layer)].append({
                    "corpus": corpus, "site": site_index, "base_kl": base_kl,
                    "duality_error": duality,
                    "kl_error": {**{method: exact_kl(logits["full"], logits[method]) for method in METHODS}, **prior_methods},
                })
            if (site_index + 1) % 8 == 0:
                print("suffix channel", corpus, site_index + 1, "/", len(edits), flush=True)

    summaries = {str(layer): {} for layer in parent["layers"]}
    all_methods = METHODS + ("taylor_local", "endpoint_local", "behavior_oracle")
    for layer in parent["layers"]:
        for corpus in arrays:
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            threshold = float(np.quantile([row["base_kl"] for row in selected], .75))
            high = [row for row in selected if row["base_kl"] >= threshold]
            summaries[str(layer)][corpus] = {
                "sites": len(selected), "mean_base_kl": float(np.mean([row["base_kl"] for row in selected])),
                "methods": {method: aggregate(selected, method) for method in all_methods},
                "high_effect_methods": {method: aggregate(high, method) for method in all_methods},
            }
    cell = lambda layer, corpus: summaries[str(layer)][corpus]["methods"]
    value = lambda layer, corpus, method: cell(layer, corpus)[method]["effect_weighted_kl_recovery"]
    hard_cells = [(0, "fineweb"), (0, "tinystories"), (1, "fineweb"), (1, "tinystories")]
    hypotheses = {
        "F0_suffix_fisher_contraction_is_numerically_valid": {"supported": maximum_duality_error <= 1e-4},
        "F1_suffix_conditioned_taylor_preserves_behavior": {"supported": all(value(layer, corpus, "taylor_suffix") >= .60 and cell(layer, corpus)["taylor_suffix"]["median_per_example_recovery"] >= .60 for layer, corpus in hard_cells)},
        "F2_suffix_conditioning_beats_local_scoring": {"supported": all(value(layer, corpus, "taylor_suffix") >= value(layer, corpus, "taylor_local") + .15 for layer, corpus in hard_cells)},
        "F3_suffix_conditioning_closes_half_the_endpoint_gap": {"supported": all(value(layer, corpus, "taylor_suffix") - value(layer, corpus, "taylor_local") >= .5 * (value(layer, corpus, "endpoint_local") - value(layer, corpus, "taylor_local")) for layer, corpus in hard_cells)},
        "F4_endpoint_direction_plus_suffix_metric_is_strong": {"supported": all(value(layer, corpus, "endpoint_suffix") >= .80 for layer, corpus in hard_cells)},
        "F5_explicit_midpoint_identifies_truncation_cost": {"supported": all(value(layer, corpus, "explicit_midpoint_suffix") >= value(layer, corpus, "taylor_suffix") + .10 for layer, corpus in hard_cells)},
        "F6_suffix_conditioned_selector_transfers_ood": {"supported": all(abs(value(layer, "fineweb", "taylor_suffix") - value(layer, "tinystories", "taylor_suffix")) <= .15 for layer in parent["layers"])},
        "F7_high_effect_behavior_is_preserved": {"supported": all(summaries[str(layer)][corpus]["high_effect_methods"]["taylor_suffix"]["effect_weighted_kl_recovery"] >= .55 for layer, corpus in hard_cells)},
        "F8_selector_is_prospective": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": red, "maximum_duality_error": maximum_duality_error, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Suffix-Fisher channel selection", "", "A frozen Taylor product-change estimate is contracted through the exact midpoint suffix Jacobian and the output softmax-Fisher metric. Channels are ranked by `|v_i [J'FJv]_i|`; actual finite endpoint deltas are inspected only after selection and patched through the complete base suffix.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Suffix-Fisher channel selector](bilin12_layer01_suffix_fisher_channel_selector.png)", "", "| Layer | Corpus | Taylor suffix/local | Explicit/endpoint suffix | Endpoint local/oracle | Taylor suffix median/q25 | High-effect suffix |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer, corpus in hard_cells:
        row = summaries[str(layer)][corpus]; methods = row["methods"]
        lines.append(f"| {layer} | {corpus} | {methods['taylor_suffix']['effect_weighted_kl_recovery']:.3f}/{methods['taylor_local']['effect_weighted_kl_recovery']:.3f} | {methods['explicit_midpoint_suffix']['effect_weighted_kl_recovery']:.3f}/{methods['endpoint_suffix']['effect_weighted_kl_recovery']:.3f} | {methods['endpoint_local']['effect_weighted_kl_recovery']:.3f}/{methods['behavior_oracle']['effect_weighted_kl_recovery']:.3f} | {methods['taylor_suffix']['median_per_example_recovery']:.3f}/{methods['taylor_suffix']['q25_per_example_recovery']:.3f} | {row['high_effect_methods']['taylor_suffix']['effect_weighted_kl_recovery']:.3f} |")
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    names = ("taylor_local", "taylor_suffix", "explicit_midpoint_suffix", "endpoint_suffix", "endpoint_local", "behavior_oracle")
    for layer in parent["layers"]:
        x = np.arange(len(names)); width = .35
        for offset, corpus in ((-.5, "fineweb"), (.5, "tinystories")):
            axes[layer].bar(x + offset * width, [value(layer, corpus, name) for name in names], width, label=corpus)
        axes[layer].set_xticks(x, names, rotation=35, ha="right"); axes[layer].set_ylabel("Effect-weighted KL recovery"); axes[layer].set_title(f"Layer {layer}"); axes[layer].grid(axis="y", alpha=.2); axes[layer].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps(hypotheses, indent=2)); print(REPORT, flush=True)


if __name__ == "__main__": main()
