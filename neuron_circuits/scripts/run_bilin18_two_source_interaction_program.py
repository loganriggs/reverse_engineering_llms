#!/usr/bin/env python3
"""Differential tensor program for a two-token interaction circuit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_competent_feature_prospective_tensor_selector import selector_metrics
from run_bilin18_layer1_cross_position_role_redteam import cross_gradient, native_logits
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_layers0_3_two_source_selector import (
    gradients_at_positions, logits_for_position_deltas, masks, selected_vector,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_two_source_interaction_program.json")
OUTPUT = TENSOR_ROOT / "bilin18_two_source_interaction_program.json"
REPORT = TENSOR_ROOT / "BILIN18_TWO_SOURCE_INTERACTION_PROGRAM.md"
FIGURE = TENSOR_ROOT / "bilin18_two_source_interaction_program.png"


def summarize(records, layer, pair, method):
    rows = [row for row in records if row["layer"] == layer and row["pair"] == pair]
    values = [row["metrics"][method] for row in rows]
    effects = np.asarray([row["finite_mobius_effects"][method] for row in rows])
    exact = np.asarray([row["finite_mobius_effects"]["all_attention"] for row in rows])
    return {
        "gradient_cosine_median": float(np.median([row["gradient_cosine"] for row in values])),
        "gradient_cosine_q10": float(np.quantile([row["gradient_cosine"] for row in values], .1)),
        "support_jaccard_median": float(np.median([row["support_jaccard"] for row in values])),
        "finite_directional_fraction": float(np.mean(effects > 0)),
        "finite_mean_effect": float(effects.mean()),
        "finite_exact_effect_ratio": float(effects.mean() / max(exact.mean(), 1e-30)),
    }


@torch.inference_mode()
def full_logits(model, ids, position):
    trace = forward_layers_native(model, ids, len(model.blocks) - 1)
    return trace, native_logits(model, trace, position)


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source_results"]).read_text()); source_config = source["config"]
    corpus = np.load(config["corpus"], mmap_mode="r")
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    sites = [row for row in source["records"] if row["layer"] == config["source_layers"][0]]
    records, maximum_autograd_error = [], 0.0
    for site_index, site in enumerate(sites):
        early, late = site["positions"]; early_token, late_token = site["replacement_tokens"]
        raw = np.asarray(corpus[source_config["start_row"] + site["row"], :source_config["sequence_length"]], dtype=np.int64).copy()
        variants_np = {"base": raw.copy(), "a": raw.copy(), "b": raw.copy(), "ab": raw.copy()}
        variants_np["a"][early] = early_token; variants_np["b"][late] = late_token
        variants_np["ab"][early] = early_token; variants_np["ab"][late] = late_token
        variants = {name: torch.as_tensor(value, device="cuda")[None] for name, value in variants_np.items()}
        with torch.no_grad():
            full = {name: full_logits(model, ids, config["prediction_position"]) for name, ids in variants.items()}
        traces = {name: row[0] for name, row in full.items()}; logits = {name: row[1] for name, row in full.items()}
        probabilities = {name: F.softmax(value.float(), -1) for name, value in logits.items()}
        interaction = (probabilities["ab"] - probabilities["a"] - probabilities["b"] + probabilities["base"]).detach()
        for layer in config["source_layers"]:
            # The later edit cannot affect an earlier-position representation in
            # a causal model.  Thus the same early-token delta can be inserted
            # into base and B contexts, isolating downstream context dependence.
            delta = product_at(model, traces["a"], layer, early) - product_at(model, traces["base"], layer, early)
            programs = masks(layer, len(model.blocks), config["near_widths"])
            base_gradients = gradients_at_positions(model, traces["base"], layer, (early,), config["prediction_position"], interaction, programs)
            b_gradients = gradients_at_positions(model, traces["b"], layer, (early,), config["prediction_position"], interaction, programs)
            differential = {name: b_gradients[name] - base_gradients[name] for name in programs}
            exact = differential["all_attention"]
            if site_index < config["autograd_check_sites"]:
                reference = cross_gradient(model, traces["b"], layer, early, config["prediction_position"], interaction) - cross_gradient(model, traces["base"], layer, early, config["prediction_position"], interaction)
                maximum_autograd_error = max(maximum_autograd_error, float((exact - reference).norm() / reference.norm().clamp_min(1e-30)))
            selected = {}
            for name, prediction in differential.items(): selected[name], _ = selected_vector(delta, prediction * delta, config["support_rank"])
            vectors = {"base": torch.zeros_like(delta), **selected}
            with torch.no_grad():
                logits_base = logits_for_position_deltas(model, layer, traces["base"][layer], vectors, (early,), config["prediction_position"])
                logits_b = logits_for_position_deltas(model, layer, traces["b"][layer], vectors, (early,), config["prediction_position"])
            finite = {}
            for name in selected:
                effect_b = float((logits_b[name] - logits_b["base"]).float() @ interaction)
                effect_base = float((logits_base[name] - logits_base["base"]).float() @ interaction)
                finite[name] = effect_b - effect_base
            records.append({"pair": site["pair"], "row": site["row"], "layer": layer, "early_position": early, "late_position": late, "interaction_norm": float(interaction.norm()), "differential_gradient_norm": float(exact.norm()), "metrics": {name: selector_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in differential.items()}, "finite_mobius_effects": finite})
        print(f"interaction program {site_index + 1}/{len(sites)} {site['pair']}", flush=True)
    pairs = sorted({row["pair"] for row in records}); methods = list(records[0]["metrics"])
    summaries = {str(layer): {pair: {method: summarize(records, layer, pair, method) for method in methods} for pair in pairs} for layer in config["source_layers"]}
    cells = [summaries[str(layer)][pair] for layer in config["source_layers"] for pair in pairs]
    hypotheses = {
        "I0_exact_differential_composition": {"supported": maximum_autograd_error <= 1e-5},
        "I1_near12_interaction_gradient": {"supported": all(cell["near_12"]["gradient_cosine_median"] >= .80 for cell in cells)},
        "I2_near12_interaction_tail": {"supported": all(cell["near_12"]["gradient_cosine_q10"] >= .50 for cell in cells)},
        "I3_near12_recovers_interaction_support": {"supported": all(cell["near_12"]["support_jaccard_median"] >= .40 for cell in cells)},
        "I4_exact_interaction_patch_has_finite_direction": {"supported": all(cell["all_attention"]["finite_directional_fraction"] >= .75 for cell in cells)},
        "I5_near12_retains_interaction_effect": {"supported": all(cell["near_12"]["finite_exact_effect_ratio"] >= .70 for cell in cells)},
    }
    result = {"config": config, "hypotheses": hypotheses, "maximum_autograd_error": maximum_autograd_error, "summaries": summaries, "records": records}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# Differential tensor program for two-token interactions", "", "For each strength-balanced two-token edit, this assay asks how the earlier token's causal effect changes when the later token is present. Because causal masking prevents the later token from changing the earlier-position source product, subtracting the two suffix gradients isolates downstream contextual interaction. The finite endpoint is a Möbius difference: `(patch effect in B context) - (patch effect in base context)`.", "", f"Maximum differential all-attention error against independent autograd: `{maximum_autograd_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Interaction program](bilin18_two_source_interaction_program.png)", "", "| Layer | Pair | Near 8 cosine | Near 12 cosine/Q10 | Near 12 support | Exact finite direction | Near 12/exact Möbius effect |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for pair in pairs:
            cell = summaries[str(layer)][pair]; n8 = cell["near_8"]; n12 = cell["near_12"]
            lines.append(f"| {layer} | {pair} | {n8['gradient_cosine_median']:.3f} | {n12['gradient_cosine_median']:.3f}/{n12['gradient_cosine_q10']:.3f} | {n12['support_jaccard_median']:.3f} | {cell['all_attention']['finite_directional_fraction']:.3f} | {n12['finite_exact_effect_ratio']:.3f} |")
    lines += ["", "## Guardrails", "", "This is a local interaction explanation around two discrete edit worlds. Subtracting gradients is vulnerable to cancellation when the interaction is tiny, so absolute differential norms and finite Möbius effects are retained. A successful differential program explains contextual modulation of one source; it does not prove that a persistent conjunction feature exists in either source layer."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["source_layers"])); fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for pair in pairs:
        axes[0].plot(x, [summaries[str(layer)][pair]["near_12"]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=pair)
        axes[1].plot(x, [summaries[str(layer)][pair]["near_12"]["finite_exact_effect_ratio"] for layer in config["source_layers"]], marker="o", label=pair)
    for axis in axes: axis.set_xticks(x, config["source_layers"]); axis.grid(alpha=.2); axis.legend()
    axes[0].set(xlabel="Source layer", ylabel="Differential-gradient cosine", title="Interaction recurrence")
    axes[1].set(xlabel="Source layer", ylabel="Möbius effect / exact", title="Prospective interaction coalition")
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "maximum_autograd_error": maximum_autograd_error}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
