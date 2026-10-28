#!/usr/bin/env python3
"""Two-source cross-position test of the compositional tensor selector."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_competent_feature_prospective_tensor_selector import (
    masked_attention_program,
    selector_metrics,
)
from run_bilin18_layer1_cross_position_role_redteam import cross_gradient, native_logits
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layers0_3_two_source_selector.json")
OUTPUT = TENSOR_ROOT / "bilin18_layers0_3_two_source_selector.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYERS0_3_TWO_SOURCE_SELECTOR.md"
FIGURE = TENSOR_ROOT / "bilin18_layers0_3_two_source_selector.png"
CHECKPOINT = TENSOR_ROOT / "bilin18_layers0_3_two_source_selector.partial.json"


def masks(source_layer: int, block_count: int, widths: list[int]) -> dict[str, set[int]]:
    suffix = list(range(source_layer + 1, block_count))
    result = {f"near_{width}": set(suffix[:width]) for width in widths}
    result["late_8"] = set(suffix[-8:])
    result["all_attention"] = set(suffix)
    return result


def gradients_at_positions(model, trace, source_layer, positions, prediction_position, contrast, programs):
    """Reuse the exact compositional recurrence, extracting multiple source positions."""
    rows = {}
    for position in positions:
        rows[position] = masked_attention_program(
            model, trace, source_layer, position, contrast, programs,
            prediction_position=prediction_position,
        )
    return {
        name: torch.cat([rows[position][name] for position in positions])
        for name in programs
    }


def split_vector(vector, positions, width):
    return {position: vector[index * width:(index + 1) * width] for index, position in enumerate(positions)}


def selected_vector(delta, score, rank):
    selected = score.topk(rank).indices
    result = torch.zeros_like(delta)
    result[selected] = delta[selected]
    return result, selected


@torch.inference_mode()
def logits_for_position_deltas(model, source_layer, base, vectors, positions, prediction_position):
    decoder = model.blocks[source_layer].mlp.Down.weight.float()
    width = decoder.shape[1]
    names, outputs, x0s, buses = list(vectors), [], [], []
    for vector in vectors.values():
        output = base["output"].clone()
        pieces = split_vector(vector, positions, width)
        for position in positions:
            output[:, position] += F.linear(pieces[position], decoder).to(output.dtype)
        outputs.append(output); x0s.append(base["x0"]); buses.append(base["value_bus"])
    logits = suffix_from_layer(
        model, source_layer, torch.cat(x0s), torch.cat(buses),
        torch.cat(outputs), prediction_position,
    )
    return {name: logits[index] for index, name in enumerate(names)}


def cosine(left, right):
    return float((left @ right) / (left.norm() * right.norm()).clamp_min(1e-30))


def summarize(records, layer, pair_name, method):
    rows = [row for row in records if row["layer"] == layer and row["pair"] == pair_name]
    values = [row["metrics"][method] for row in rows]
    shifts = np.asarray([row["finite_shifts"][method] for row in rows])
    exact = np.asarray([row["finite_shifts"]["all_attention"] for row in rows])
    return {
        "gradient_cosine_median": float(np.median([v["gradient_cosine"] for v in values])),
        "gradient_cosine_q10": float(np.quantile([v["gradient_cosine"] for v in values], .1)),
        "support_jaccard_median": float(np.median([v["support_jaccard"] for v in values])),
        "true_leverage_ratio_median": float(np.median([v["true_leverage_ratio"] for v in values])),
        "finite_directional_fraction": float(np.mean(shifts > 0)),
        "finite_mean_shift": float(shifts.mean()),
        "finite_exact_effect_ratio": float(shifts.mean() / max(exact.mean(), 1e-30)),
    }


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")
    rows = corpus[config["start_row"]:config["start_row"] + config["rows"]]
    signature = {
        "start_row": config["start_row"], "rows": config["rows"],
        "pairs": config["edit_position_pairs"], "layers": config["source_layers"],
        "widths": config["near_widths"], "rank": config["support_rank"],
    }
    records, maximum_autograd_error = [], 0.0
    if CHECKPOINT.exists():
        old = json.loads(CHECKPOINT.read_text())
        if old.get("signature") == signature:
            records = old.get("records", [])
            maximum_autograd_error = old.get("maximum_autograd_error", 0.0)
    completed = {(row["pair"], row["row"], row["layer"]) for row in records}
    for pair_index, positions_list in enumerate(config["edit_position_pairs"]):
        positions = tuple(positions_list); pair_name = f"{positions[0]}+{positions[1]}"
        for row_index, raw_row in enumerate(rows):
            raw = np.asarray(raw_row[:config["sequence_length"]], dtype=np.int64).copy()
            if any(raw[position] == newline for position in positions): continue
            variants = {}
            for name, selected in (("base", ()), ("a", (positions[0],)), ("b", (positions[1],)), ("ab", positions)):
                value = raw.copy()
                for position in selected: value[position] = newline
                variants[name] = torch.as_tensor(value, device="cuda")[None]
            with torch.no_grad():
                traces = {
                    name: forward_layers_native(model, ids, len(model.blocks) - 1 if name == "base" else max(config["source_layers"]))
                    for name, ids in variants.items()
                }
                logits = {
                    "base": native_logits(model, traces["base"], config["prediction_position"]),
                }
                for name in ("a", "b", "ab"):
                    full = forward_layers_native(model, variants[name], len(model.blocks) - 1)
                    logits[name] = native_logits(model, full, config["prediction_position"])
            probabilities = {name: F.softmax(value.float(), -1) for name, value in logits.items()}
            contrast = (probabilities["ab"] - probabilities["base"]).detach()
            interaction = probabilities["ab"] - probabilities["a"] - probabilities["b"] + probabilities["base"]
            interaction_ratio = float(interaction.norm() / contrast.norm().clamp_min(1e-30))
            behavior_score = float((logits["ab"] - logits["base"]) @ contrast)
            for layer in config["source_layers"]:
                key = (pair_name, row_index, layer)
                if key in completed: continue
                width = model.blocks[layer].mlp.Left.out_features
                delta = torch.cat([
                    product_at(model, traces["ab"], layer, position)
                    - product_at(model, traces["base"], layer, position)
                    for position in positions
                ])
                programs = masks(layer, len(model.blocks), config["near_widths"])
                predictions = gradients_at_positions(
                    model, traces["base"], layer, positions,
                    config["prediction_position"], contrast, programs,
                )
                exact = predictions["all_attention"]
                if row_index < config["autograd_check_sites"]:
                    references = torch.cat([
                        cross_gradient(model, traces["base"], layer, position,
                                       config["prediction_position"], contrast)
                        for position in positions
                    ])
                    maximum_autograd_error = max(
                        maximum_autograd_error,
                        float((exact - references).norm() / references.norm().clamp_min(1e-30)),
                    )
                selected, supports = {}, {}
                for name, prediction in predictions.items():
                    selected[name], supports[name] = selected_vector(
                        delta, prediction * delta, config["support_rank"],
                    )
                zero = torch.zeros_like(delta)
                patch_vectors = {"base": zero, "full_product": delta, **selected}
                with torch.no_grad():
                    patched_logits = logits_for_position_deltas(
                        model, layer, traces["base"][layer], patch_vectors,
                        positions, config["prediction_position"],
                    )
                base_score = float(patched_logits["base"].float() @ contrast)
                finite = {
                    name: float(value.float() @ contrast) - base_score
                    for name, value in patched_logits.items() if name != "base"
                }
                exact_support = supports["all_attention"]
                counts = [int(((exact_support >= i * width) & (exact_support < (i + 1) * width)).sum()) for i in range(2)]
                records.append({
                    "pair": pair_name, "positions": list(positions), "row": row_index,
                    "layer": layer, "source_tokens": [int(raw[p]) for p in positions],
                    "source_text": [tokenizer.decode([int(raw[p])]).replace("\n", "\\n") for p in positions],
                    "behavior_score": behavior_score, "interaction_ratio": interaction_ratio,
                    "exact_support_counts": counts,
                    "metrics": {
                        name: selector_metrics(prediction, exact, delta, config["support_rank"])
                        for name, prediction in predictions.items()
                    },
                    "finite_shifts": finite,
                })
                CHECKPOINT.write_text(json.dumps({
                    "signature": signature, "maximum_autograd_error": maximum_autograd_error,
                    "records": records,
                }, allow_nan=False))
            print(f"two-source {pair_name} row {row_index + 1}/{len(rows)}", flush=True)

    pair_names = [f"{a}+{b}" for a, b in config["edit_position_pairs"]]
    methods = list(records[0]["metrics"])
    summaries = {
        str(layer): {
            pair: {method: summarize(records, layer, pair, method) for method in methods}
            for pair in pair_names
        } for layer in config["source_layers"]
    }
    cells = [summaries[str(layer)][pair] for layer in config["source_layers"] for pair in pair_names]
    minority_fractions = [min(row["exact_support_counts"]) / config["support_rank"] for row in records]
    interaction_ratios = [row["interaction_ratio"] for row in records if row["layer"] == config["source_layers"][0]]
    hypotheses = {
        "M0_exact_composition": {"supported": maximum_autograd_error <= 1e-5},
        "M1_near12_gradient_fidelity": {"supported": all(cell["near_12"]["gradient_cosine_median"] >= .90 for cell in cells)},
        "M2_near8_not_universal": {"supported": any(cell["near_8"]["gradient_cosine_median"] < .90 or cell["near_8"]["gradient_cosine_q10"] < .75 for cell in cells)},
        "M3_fixed_budget_finite_direction": {"supported": all(cell["near_12"]["finite_directional_fraction"] >= .90 for cell in cells)},
        "M4_both_sources_are_used": {"supported": float(np.median(minority_fractions)) >= .10},
        "M5_output_interaction_is_material": {"supported": float(np.median(interaction_ratios)) >= .05},
        "M6_near12_retains_exact_effect": {"supported": all(cell["near_12"]["finite_exact_effect_ratio"] >= .80 for cell in cells)},
    }
    result = {
        "config": config, "hypotheses": hypotheses,
        "maximum_autograd_error": maximum_autograd_error,
        "median_output_interaction_ratio": float(np.median(interaction_ratios)),
        "median_minority_support_fraction": float(np.median(minority_fractions)),
        "summaries": summaries, "records": records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))
    CHECKPOINT.write_text(json.dumps({"complete": True, "signature": signature, "maximum_autograd_error": maximum_autograd_error, "records": records}, allow_nan=False))
    lines = [
        "# Layers 0–3 two-source compositional selector", "",
        "Two natural FineWeb tokens are replaced by newline together. One fixed top-64 product-channel budget is allocated across both earlier positions, while the behavioral direction is the full model's own joint probability change at Position 127. This tests whether the local tensor recurrence composes across source positions; it does not assume a human semantic label.", "",
        f"Maximum all-attention recurrence error against independent two-position autograd: `{maximum_autograd_error:.2e}`.",
        f"Median output interaction norm / joint-change norm: `{np.median(interaction_ratios):.3f}`.",
        f"Median minority-position share of the exact top-64 support: `{np.median(minority_fractions):.3f}`.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Two-source selector](bilin18_layers0_3_two_source_selector.png)", "", "| Layer | Positions | Near 8 cosine/Q10 | Near 12 cosine/Q10 | Near 12 support | Near 12 finite direction | Near 12/exact effect |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for pair in pair_names:
            cell = summaries[str(layer)][pair]; n8 = cell["near_8"]; n12 = cell["near_12"]
            lines.append(f"| {layer} | {pair} | {n8['gradient_cosine_median']:.3f}/{n8['gradient_cosine_q10']:.3f} | {n12['gradient_cosine_median']:.3f}/{n12['gradient_cosine_q10']:.3f} | {n12['support_jaccard_median']:.3f} | {n12['finite_directional_fraction']:.3f} | {n12['finite_exact_effect_ratio']:.3f} |")
    lines += ["", "## Guardrails", "", "The behavioral objective is automatically induced by the chosen two-token edit, but the edit itself is analyst supplied. The assay therefore tests multi-source compositional execution, not autonomous concept discovery. The interaction ratio distinguishes a genuinely nonlinear two-edit output from a purely additive pair, but the top-64 intervention targets the complete joint change rather than isolating only its Möbius interaction term."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    x = np.arange(len(config["source_layers"]))
    for pair in pair_names:
        axes[0].plot(x, [summaries[str(layer)][pair]["near_12"]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=pair)
        axes[1].plot(x, [summaries[str(layer)][pair]["near_12"]["finite_exact_effect_ratio"] for layer in config["source_layers"]], marker="o", label=pair)
    for axis in axes: axis.set_xticks(x, config["source_layers"]); axis.grid(alpha=.2); axis.legend()
    axes[0].set(xlabel="Source layer", ylabel="Median gradient cosine", title="Two-source near-12 recurrence")
    axes[1].set(xlabel="Source layer", ylabel="Mean effect / exact selector", title="Fixed top-64 causal payload")
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({key: result[key] for key in ("hypotheses", "maximum_autograd_error", "median_output_interaction_ratio", "median_minority_support_fraction")}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
