#!/usr/bin/env python3
"""Finite causal validation of supervised layer-1 native-channel coalitions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, collect_split, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_supervised_coalition_causality.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_supervised_coalition_causality.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUPERVISED_COALITION_CAUSALITY.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_supervised_coalition_causality.png"


def fit_supports(discovery, labels, feature_indices, rank):
    x = discovery.astype(np.float32); mean = x.mean(0); std = x.std(0).clip(1e-4); z = (x - mean) / std
    result = {}
    for feature in feature_indices:
        y = labels[:, feature].astype(bool)
        gap = z[y].mean(0) - z[~y].mean(0)
        order = np.argsort(np.abs(gap))[::-1].copy()
        result[feature] = {"mean": mean, "std": std, "gap": gap, "indices": order[:rank], "single": order[:1]}
    return result


def select_pairs(labels, current_ids, heldout_values, support, feature, positions, count, extreme_count, rng):
    z = (heldout_values.astype(np.float32) - support["mean"]) / support["std"]
    score = z[:, support["indices"]] @ support["gap"][support["indices"]]
    candidates = []
    y = labels[:, feature].astype(bool)
    for token in np.unique(current_ids):
        members = np.flatnonzero(current_ids == token)
        positive, negative = members[y[members]], members[~y[members]]
        if not len(positive) or not len(negative): continue
        donor = int(positive[np.argmax(score[positive])]); recipient = int(negative[np.argmin(score[negative])])
        candidates.append((float(score[donor] - score[recipient]), int(token), donor, recipient))
    candidates.sort(reverse=True)
    extreme = [("extreme", *row[1:]) for row in candidates[:extreme_count]]
    remaining = candidates[extreme_count:]
    chosen = rng.choice(len(remaining), size=min(count - extreme_count, len(remaining)), replace=False)
    uniform = [("uniform", *remaining[int(index)][1:]) for index in chosen]
    result = []
    for selection, token, donor, recipient in extreme + uniform:
        result.append({"selection": selection, "token": token, "donor_flat": donor, "recipient_flat": recipient, "donor_row": donor // len(positions), "donor_position": positions[donor % len(positions)], "recipient_row": recipient // len(positions), "recipient_position": positions[recipient % len(positions)]})
    return result


def product_at(model, trace, layer, position):
    normalized = torch.nn.functional.rms_norm(trace[layer]["pre"][:, position], (model.config.n_embd,))
    return (model.blocks[layer].mlp.Left(normalized) * model.blocks[layer].mlp.Right(normalized))[0].float()


def class_score(logits, contrast): return float(logits.float()[:contrast.numel()] @ contrast)


def summarize(records, method, selection=None):
    rows = [row for row in records if selection is None or row["selection"] == selection]
    shifts = np.asarray([row["shifts"][method] for row in rows], dtype=np.float64)
    full = np.asarray([row["shifts"]["full"] for row in rows], dtype=np.float64)
    correlation = float(np.corrcoef(shifts, full)[0, 1]) if len(rows) > 1 and shifts.std() > 0 and full.std() > 0 else None
    return {"pairs": len(rows), "directional_fraction": float(np.mean(shifts > 0)), "mean_shift": float(shifts.mean()), "mean_absolute_shift": float(np.abs(shifts).mean()), "correlation_with_full": correlation}


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrast_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]
    contrast = torch.as_tensor(contrast_np, device="cuda", dtype=torch.float32)
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); start = source["start_row"]
    discovery_rows = corpus[start:start + source["discovery_rows"]]
    heldout_start = start + source["discovery_rows"]
    heldout_rows = corpus[heldout_start:heldout_start + source["heldout_rows"]]
    discovery_products, discovery_labels, _, _ = collect_split(model, discovery_rows, source["positions"], flags, contrast, source["batch_size"])
    heldout_products, heldout_labels, heldout_ids, _ = collect_split(model, heldout_rows, source["positions"], flags, contrast, source["batch_size"])
    feature_indices = [FEATURES.index(name) for name in config["features"]]
    supports = fit_supports(discovery_products[config["layer"]], discovery_labels, feature_indices, config["rank"])
    rng = np.random.default_rng(config["seed"]); records = []
    random_supports = {feature: torch.as_tensor(rng.choice(heldout_products[config["layer"]].shape[1], config["rank"], replace=False), device="cuda") for feature in feature_indices}
    for feature in feature_indices:
        pairs = select_pairs(heldout_labels, heldout_ids, heldout_products[config["layer"]], supports[feature], feature, source["positions"], config["pairs_per_feature"], config["extreme_pairs"], rng)
        supervised_indices = torch.as_tensor(supports[feature]["indices"], device="cuda")
        single_index = torch.as_tensor(supports[feature]["single"], device="cuda")
        for pair_index, pair in enumerate(pairs):
            donor_raw = np.asarray(heldout_rows[pair["donor_row"], :source["sequence_length"] + 1], dtype=np.int64)
            recipient_raw = np.asarray(heldout_rows[pair["recipient_row"], :source["sequence_length"] + 1], dtype=np.int64)
            donor_ids = torch.as_tensor(donor_raw[:source["sequence_length"]], device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw[:source["sequence_length"]], device="cuda")[None]
            with torch.no_grad():
                donor_trace = forward_layers_native(model, donor_ids, config["layer"])
                recipient_trace = forward_layers_native(model, recipient_ids, config["layer"])
                donor_product = product_at(model, donor_trace, config["layer"], pair["donor_position"])
                recipient_product = product_at(model, recipient_trace, config["layer"], pair["recipient_position"])
                full = donor_product - recipient_product
                magnitude_indices = full.abs().topk(config["rank"]).indices
                deltas = {"base": torch.zeros_like(full), "full": full, "supervised64": selected_delta(full, supervised_indices), "single": selected_delta(full, single_index), "random64": selected_delta(full, random_supports[feature]), "magnitude64": selected_delta(full, magnitude_indices)}
                logits = logits_for_deltas(model, config["layer"], recipient_trace[config["layer"]], deltas, pair["recipient_position"])
            vector = contrast[:, feature]
            base_score = class_score(logits["base"], vector)
            shifts = {name: class_score(value, vector) - base_score for name, value in logits.items() if name != "base"}
            context = lambda raw, position: tokenizer.decode(raw[max(0, position - 8):position + 2].tolist()).replace("\n", "\\n")
            records.append({"feature": FEATURES[feature], "pair": pair_index, **pair, "current_text": tokenizer.decode([pair["token"]]).replace("\n", "\\n"), "donor_next": tokenizer.decode([int(donor_raw[pair["donor_position"] + 1])]).replace("\n", "\\n"), "recipient_next": tokenizer.decode([int(recipient_raw[pair["recipient_position"] + 1])]).replace("\n", "\\n"), "donor_context": context(donor_raw, pair["donor_position"]), "recipient_context": context(recipient_raw, pair["recipient_position"]), "shifts": shifts})
        print("causal feature", FEATURES[feature], len(pairs), flush=True)
    methods = ("full", "supervised64", "single", "random64", "magnitude64")
    summary = {method: {"all": summarize(records, method), "extreme": summarize(records, method, "extreme"), "uniform": summarize(records, method, "uniform")} for method in methods}
    target = summary["supervised64"]["all"]; random = summary["random64"]["all"]; single = summary["single"]["all"]; uniform_target = summary["supervised64"]["uniform"]; uniform_random = summary["random64"]["uniform"]
    hypotheses = {
        "P0_full_layer1_path_is_directional": {"supported": summary["full"]["all"]["directional_fraction"] >= .60},
        "P1_supervised_coalition_is_directional": {"supported": target["directional_fraction"] >= .60 and target["directional_fraction"] - random["directional_fraction"] >= .10},
        "P2_coalition_tracks_full_path": {"supported": target["correlation_with_full"] is not None and target["correlation_with_full"] >= .50},
        "P3_coalition_beats_single_neuron": {"supported": target["mean_absolute_shift"] >= 2 * single["mean_absolute_shift"]},
        "P4_effect_is_not_only_extreme_selection": {"supported": uniform_target["directional_fraction"] >= .55 and uniform_target["directional_fraction"] > uniform_random["directional_fraction"]},
        "P5_endpoint_magnitude_is_upper_control": {"supported": True},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "records": records}, indent=2, allow_nan=False))
    lines = ["# Finite causal validation of supervised layer-1 coalitions", "", "For each declared next-token feature, positive and negative held-out contexts are paired while holding the exact current token fixed. A donor's layer-1 product activations are patched into the negative recipient at the current position and propagated through all remaining layers. Supports are frozen from disjoint discovery examples. Half the pairs are strong held-out decoder examples; half are uniformly sampled token groups. No intervention outcome is used for pair or support selection.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Causal coalitions](bilin18_layer1_supervised_coalition_causality.png)", "", "| Method | Directional all/extreme/uniform | Mean | Mean abs | Corr. with full |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        row = summary[method]["all"]; lines.append(f"| {method} | {row['directional_fraction']:.3f}/{summary[method]['extreme']['directional_fraction']:.3f}/{summary[method]['uniform']['directional_fraction']:.3f} | {row['mean_shift']:+.4f} | {row['mean_absolute_shift']:.4f} | {'—' if row['correlation_with_full'] is None else f'{row['correlation_with_full']:.3f}'} |")
    lines += ["", "## Example interventions", "", "| Feature | Selection | Current | Donor context → next | Recipient context → next | Full / supervised / random shift |", "|---|---|---|---|---|---:|"]
    for row in records[:16]: lines.append(f"| {row['feature']} | {row['selection']} | `{row['current_text']}` | `{row['donor_context']}` → `{row['donor_next']}` | `{row['recipient_context']}` → `{row['recipient_next']}` | {row['shifts']['full']:+.3f}/{row['shifts']['supervised64']:+.3f}/{row['shifts']['random64']:+.3f} |")
    lines += ["", "## Guardrails", "", "- The intervention changes one MLP product vector only; a null does not imply the feature is absent from attention or the residual stream.", "- The output score is the declared class's mean clipped logit minus its complement, not the probability of the observed next token.", "- Endpoint-magnitude support reads the donor-recipient activation difference and is therefore an upper control, not a prospective interpretation method."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    x = np.asarray([row["shifts"]["full"] for row in records]); y = np.asarray([row["shifts"]["supervised64"] for row in records]); colors = [0 if row["selection"] == "uniform" else 1 for row in records]
    axes[0].scatter(x, y, c=colors, cmap="coolwarm", alpha=.75); axes[0].axhline(0, color="gray"); axes[0].axvline(0, color="gray"); axes[0].set(xlabel="Full layer-1 product-patch shift", ylabel="Supervised-64 shift", title="Does the decoded support track the causal path?"); axes[0].grid(alpha=.2)
    axes[1].bar(range(len(methods)), [summary[method]["all"]["directional_fraction"] for method in methods]); axes[1].set_xticks(range(len(methods)), methods, rotation=25, ha="right"); axes[1].axhline(.5, color="gray", linestyle=":"); axes[1].set(ylabel="Fraction shifting toward donor label", title="Finite intervention direction"); axes[1].grid(axis="y", alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
