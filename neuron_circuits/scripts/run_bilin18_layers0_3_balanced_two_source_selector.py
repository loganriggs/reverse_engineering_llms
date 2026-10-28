#!/usr/bin/env python3
"""Strength-balanced two-source red team of the compositional selector."""

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
from run_bilin18_competent_feature_prospective_tensor_selector import selector_metrics
from run_bilin18_layer1_cross_position_role_redteam import cross_gradient, native_logits
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_layers0_3_two_source_selector import (
    gradients_at_positions, logits_for_position_deltas, masks, selected_vector,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layers0_3_balanced_two_source_selector.json")
OUTPUT = TENSOR_ROOT / "bilin18_layers0_3_balanced_two_source_selector.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYERS0_3_BALANCED_TWO_SOURCE_SELECTOR.md"
FIGURE = TENSOR_ROOT / "bilin18_layers0_3_balanced_two_source_selector.png"
CHECKPOINT = TENSOR_ROOT / "bilin18_layers0_3_balanced_two_source_selector.partial.json"


@torch.inference_mode()
def complete_logits(model, ids, prediction_position):
    trace = forward_layers_native(model, ids, len(model.blocks) - 1)
    return native_logits(model, trace, prediction_position)


@torch.inference_mode()
def choose_balanced_edits(model, raw, positions, candidates, prediction_position):
    base_ids = torch.as_tensor(raw, device="cuda")[None]
    base_logits = complete_logits(model, base_ids, prediction_position)
    base_probability = F.softmax(base_logits.float(), -1)
    choices = {}
    for position in positions:
        rows = []
        for token in candidates:
            if raw[position] == token: continue
            changed = raw.copy(); changed[position] = token
            logits = complete_logits(model, torch.as_tensor(changed, device="cuda")[None], prediction_position)
            probability = F.softmax(logits.float(), -1)
            rows.append({"token": int(token), "norm": float((probability - base_probability).norm()), "logits": logits})
        choices[position] = rows
    candidates_pairs = []
    for left in choices[positions[0]]:
        for right in choices[positions[1]]:
            small, large = sorted((left["norm"], right["norm"]))
            ratio = small / max(large, 1e-30)
            # Prefer both material and matched. The geometric mean prevents a
            # pair of equally negligible edits from winning on ratio alone.
            score = np.sqrt(max(small * large, 0.0)) * ratio
            candidates_pairs.append((score, ratio, left, right))
    _, ratio, left, right = max(candidates_pairs, key=lambda row: row[0])
    return base_logits, left, right, float(ratio)


def summarize(records, layer, pair, method):
    rows = [row for row in records if row["layer"] == layer and row["pair"] == pair]
    metrics = [row["metrics"][method] for row in rows]
    shifts = np.asarray([row["finite_shifts"][method] for row in rows])
    exact = np.asarray([row["finite_shifts"]["all_attention"] for row in rows])
    return {
        "gradient_cosine_median": float(np.median([row["gradient_cosine"] for row in metrics])),
        "gradient_cosine_q10": float(np.quantile([row["gradient_cosine"] for row in metrics], .1)),
        "support_jaccard_median": float(np.median([row["support_jaccard"] for row in metrics])),
        "finite_directional_fraction": float(np.mean(shifts > 0)),
        "finite_mean_shift": float(shifts.mean()),
        "finite_exact_effect_ratio": float(shifts.mean() / max(exact.mean(), 1e-30)),
    }


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    candidate_ids = []
    for text in config["candidate_replacements"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1: raise RuntimeError(f"replacement {text!r} is not one token: {ids}")
        candidate_ids.append(ids[0])
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")
    source = corpus[config["start_row"]:config["start_row"] + config["rows"]]
    signature = {key: config[key] for key in ("start_row", "rows", "edit_position_pairs", "candidate_replacements", "source_layers", "near_widths", "support_rank")}
    records, maximum_autograd_error = [], 0.0
    if CHECKPOINT.exists():
        old = json.loads(CHECKPOINT.read_text())
        if old.get("signature") == signature:
            records = old.get("records", []); maximum_autograd_error = old.get("maximum_autograd_error", 0.0)
    completed = {(row["pair"], row["row"], row["layer"]) for row in records}
    for positions_list in config["edit_position_pairs"]:
        positions = tuple(positions_list); pair_name = f"{positions[0]}+{positions[1]}"
        for row_index, raw_row in enumerate(source):
            raw = np.asarray(raw_row[:config["sequence_length"]], dtype=np.int64).copy()
            base_logits, left, right, balance = choose_balanced_edits(
                model, raw, positions, candidate_ids, config["prediction_position"],
            )
            variants_np = {"base": raw.copy()}
            for name, selected in (("a", ((positions[0], left["token"]),)), ("b", ((positions[1], right["token"]),)), ("ab", ((positions[0], left["token"]), (positions[1], right["token"])))):
                value = raw.copy()
                for position, token in selected: value[position] = token
                variants_np[name] = value
            variants = {name: torch.as_tensor(value, device="cuda")[None] for name, value in variants_np.items()}
            with torch.no_grad():
                traces = {name: forward_layers_native(model, ids, len(model.blocks) - 1 if name == "base" else max(config["source_layers"])) for name, ids in variants.items()}
                logits = {"base": base_logits, "a": left["logits"], "b": right["logits"], "ab": complete_logits(model, variants["ab"], config["prediction_position"])}
            probabilities = {name: F.softmax(value.float(), -1) for name, value in logits.items()}
            contrast = (probabilities["ab"] - probabilities["base"]).detach()
            interaction = probabilities["ab"] - probabilities["a"] - probabilities["b"] + probabilities["base"]
            interaction_ratio = float(interaction.norm() / contrast.norm().clamp_min(1e-30))
            for layer in config["source_layers"]:
                if (pair_name, row_index, layer) in completed: continue
                channel_width = model.blocks[layer].mlp.Left.out_features
                delta = torch.cat([product_at(model, traces["ab"], layer, position) - product_at(model, traces["base"], layer, position) for position in positions])
                programs = masks(layer, len(model.blocks), config["near_widths"])
                predictions = gradients_at_positions(model, traces["base"], layer, positions, config["prediction_position"], contrast, programs)
                exact = predictions["all_attention"]
                if row_index < config["autograd_check_sites"]:
                    reference = torch.cat([cross_gradient(model, traces["base"], layer, position, config["prediction_position"], contrast) for position in positions])
                    maximum_autograd_error = max(maximum_autograd_error, float((exact - reference).norm() / reference.norm().clamp_min(1e-30)))
                selected, supports = {}, {}
                for name, prediction in predictions.items():
                    selected[name], supports[name] = selected_vector(delta, prediction * delta, config["support_rank"])
                patch_vectors = {"base": torch.zeros_like(delta), "full_product": delta, **selected}
                with torch.no_grad(): patched = logits_for_position_deltas(model, layer, traces["base"][layer], patch_vectors, positions, config["prediction_position"])
                base_score = float(patched["base"].float() @ contrast)
                finite = {name: float(value.float() @ contrast) - base_score for name, value in patched.items() if name != "base"}
                support = supports["all_attention"]
                counts = [int(((support >= index * channel_width) & (support < (index + 1) * channel_width)).sum()) for index in range(2)]
                records.append({
                    "pair": pair_name, "positions": list(positions), "row": row_index, "layer": layer,
                    "source_tokens": [int(raw[p]) for p in positions],
                    "replacement_tokens": [left["token"], right["token"]],
                    "replacement_text": [tokenizer.decode([left["token"]]).replace("\n", "\\n"), tokenizer.decode([right["token"]]).replace("\n", "\\n")],
                    "individual_effect_norms": [left["norm"], right["norm"]], "individual_balance_ratio": balance,
                    "interaction_ratio": interaction_ratio, "exact_support_counts": counts,
                    "metrics": {name: selector_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in predictions.items()},
                    "finite_shifts": finite,
                })
                CHECKPOINT.write_text(json.dumps({"signature": signature, "maximum_autograd_error": maximum_autograd_error, "records": records}, allow_nan=False))
            print(f"balanced two-source {pair_name} row {row_index + 1}/{len(source)} balance={balance:.3f}", flush=True)
    pairs = [f"{a}+{b}" for a, b in config["edit_position_pairs"]]
    methods = list(records[0]["metrics"])
    summaries = {str(layer): {pair: {method: summarize(records, layer, pair, method) for method in methods} for pair in pairs} for layer in config["source_layers"]}
    cells = [summaries[str(layer)][pair] for layer in config["source_layers"] for pair in pairs]
    independent = [row for row in records if row["layer"] == config["source_layers"][0]]
    balances = [row["individual_balance_ratio"] for row in independent]
    interactions = [row["interaction_ratio"] for row in independent]
    minority = [min(row["exact_support_counts"]) / config["support_rank"] for row in records]
    reference = json.loads(Path(config["reference_results"]).read_text())
    old_minority = reference["median_minority_support_fraction"]
    med_minority = float(np.median(minority))
    hypotheses = {
        "B0_single_edit_effects_are_balanced": {"supported": float(np.median(balances)) >= .50},
        "B1_balancing_increases_two_source_support": {"supported": med_minority >= old_minority + .05},
        "B2_both_sources_are_material": {"supported": med_minority >= .10},
        "B3_near12_gradient_fidelity": {"supported": all(cell["near_12"]["gradient_cosine_median"] >= .90 for cell in cells)},
        "B4_near12_finite_direction": {"supported": all(cell["near_12"]["finite_directional_fraction"] >= .90 for cell in cells)},
        "B5_near12_retains_exact_effect": {"supported": all(cell["near_12"]["finite_exact_effect_ratio"] >= .80 for cell in cells)},
        "B6_output_interaction_is_material": {"supported": float(np.median(interactions)) >= .05},
    }
    result = {"config": config, "hypotheses": hypotheses, "maximum_autograd_error": maximum_autograd_error, "median_individual_balance_ratio": float(np.median(balances)), "median_output_interaction_ratio": float(np.median(interactions)), "median_minority_support_fraction": med_minority, "reference_minority_support_fraction": old_minority, "summaries": summaries, "records": records}
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)); CHECKPOINT.write_text(json.dumps({"complete": True, "signature": signature, "maximum_autograd_error": maximum_autograd_error, "records": records}, allow_nan=False))
    lines = ["# Strength-balanced two-source selector", "", "For each FineWeb example and source-position pair, two single-token replacements are selected from a fixed, preregistered vocabulary using only their individual output-effect norms. The selection objective favors both materiality and equal strength. The joint edit, tensor selector, and finite intervention are then evaluated separately.", "", f"Median smaller/larger single-edit norm ratio: **{np.median(balances):.3f}**.", f"Median output interaction/joint-change norm: **{np.median(interactions):.3f}**.", f"Median minority-position top-64 share: **{med_minority:.3f}**, versus **{old_minority:.3f}** in the fixed-newline assay.", f"Maximum exact-recurrence error: `{maximum_autograd_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Balanced two-source selector](bilin18_layers0_3_balanced_two_source_selector.png)", "", "| Layer | Positions | Near 8 cosine | Near 12 cosine/Q10 | Near 12 support | Finite direction | Effect/exact |", "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for pair in pairs:
            cell = summaries[str(layer)][pair]; n8 = cell["near_8"]; n12 = cell["near_12"]
            lines.append(f"| {layer} | {pair} | {n8['gradient_cosine_median']:.3f} | {n12['gradient_cosine_median']:.3f}/{n12['gradient_cosine_q10']:.3f} | {n12['support_jaccard_median']:.3f} | {n12['finite_directional_fraction']:.3f} | {n12['finite_exact_effect_ratio']:.3f} |")
    lines += ["", "## Guardrails", "", "The replacement search deliberately conditions on individual output-effect magnitude, so it is an adversarial balance control—not a natural estimate of how often two equally important sources occur. It does not use gradients, channel identities, joint effects, or semantic labels. A positive result would establish that the tensor recurrence can allocate one sparse causal budget across multiple material sources; it would not establish a persistent multi-token concept neuron."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    x = np.arange(len(config["source_layers"]))
    for pair in pairs:
        axes[0].plot(x, [summaries[str(layer)][pair]["near_12"]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=pair)
        subset = [row for row in records if row["pair"] == pair]
        axes[1].plot(x, [np.median([min(row["exact_support_counts"]) / config["support_rank"] for row in subset if row["layer"] == layer]) for layer in config["source_layers"]], marker="o", label=pair)
    for axis in axes: axis.set_xticks(x, config["source_layers"]); axis.grid(alpha=.2); axis.legend()
    axes[0].set(xlabel="Source layer", ylabel="Gradient cosine", title="Balanced two-source recurrence")
    axes[1].set(xlabel="Source layer", ylabel="Minority top-64 share", title="Causal budget allocation")
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({key: result[key] for key in ("hypotheses", "maximum_autograd_error", "median_individual_balance_ratio", "median_output_interaction_ratio", "median_minority_support_fraction", "reference_minority_support_fraction")}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
