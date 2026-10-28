#!/usr/bin/env python3
"""A held-out 50-feature census of native MLP products on FineWeb."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import (
    AUXILIARIES, CONJUNCTIONS, DETERMINERS, PREPOSITIONS, PRONOUNS, TOKENIZER,
)
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_supervised_feature_census.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer01_supervised_feature_census.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER01_SUPERVISED_FEATURE_CENSUS.md"
FIGURE = TENSOR_ROOT / "bilin18_layer01_supervised_feature_census.png"

FEATURES = json.loads(CONFIG.read_text())["features"]
MODALS = frozenset("can could may might must shall should will would".split())
NEGATIONS = frozenset("no not never neither nor none nobody nothing nowhere cannot".split())
MONTHS = frozenset("january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec".split())
WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun".split())
TIME_UNITS = frozenset("second seconds minute minutes hour hours day days week weeks month months year years decade decades century centuries ms sec mins hrs".split())
QUANTITY_UNITS = frozenset("kg g mg lb lbs oz km m cm mm mile miles meter meters litre liter liters ml mph percent hz khz mhz ghz px mb gb tb".split())
BOOLEANS = frozenset("true false yes no null none nil on off".split())
PROGRAMMING = frozenset("def class function return import from const let var if else elif for while try catch except async await yield public private static void int float string bool new this self".split())
OPERATORS = frozenset(("=", "==", "+", "-", "*", "/", "%", "=>", "!=", "<=", ">=", "&&", "||", "+=", "-=", "::", "->", "&", "|", "^", "~"))


def measured_auc(labels, scores):
    value = auc(labels, scores)
    return float(value) if np.isfinite(value) else None


def vocabulary_features(text: str) -> np.ndarray:
    stripped = text.strip()
    lower = stripped.lower()
    alpha = bool(stripped) and stripped.isalpha()
    numeric = bool(stripped) and all(char.isdigit() or char in ".," for char in stripped) and any(char.isdigit() for char in stripped)
    punctuation = bool(stripped) and not any(char.isalnum() for char in stripped)
    quote_chars = set("'\"`\u2018\u2019\u201c\u201d")
    mapping = {
        "newline": "\n" in text,
        "tab": "\t" in text,
        "leading_space": text[:1].isspace() and "\n" not in text and "\t" not in text,
        "word_continuation": alpha and not text[:1].isspace(),
        "alphabetic": alpha,
        "numeric": numeric,
        "alphanumeric": bool(stripped) and any(c.isalpha() for c in stripped) and any(c.isdigit() for c in stripped),
        "punctuation": punctuation,
        "non_ascii": any(ord(char) > 127 for char in text),
        "capitalized": alpha and stripped[0].isupper(),
        "all_caps": alpha and len(stripped) >= 2 and stripped.isupper(),
        "lowercase": alpha and stripped.islower(),
        "long_piece": len(stripped) >= 8,
        "determiner": lower in DETERMINERS,
        "preposition": lower in PREPOSITIONS,
        "conjunction": lower in CONJUNCTIONS,
        "pronoun": lower in PRONOUNS,
        "auxiliary": lower in AUXILIARIES,
        "negation": lower in NEGATIONS,
        "modal": lower in MODALS,
        "suffix_ing": alpha and len(lower) > 4 and lower.endswith("ing"),
        "suffix_ed": alpha and len(lower) > 3 and lower.endswith("ed"),
        "suffix_ly": alpha and len(lower) > 3 and lower.endswith("ly"),
        "suffix_s": alpha and len(lower) > 3 and lower.endswith("s"),
        "suffix_ion": alpha and len(lower) > 4 and lower.endswith("ion"),
        "suffix_ment": alpha and len(lower) > 5 and lower.endswith("ment"),
        "suffix_ness": alpha and len(lower) > 5 and lower.endswith("ness"),
        "suffix_ity": alpha and len(lower) > 4 and lower.endswith("ity"),
        "suffix_ive": alpha and len(lower) > 4 and lower.endswith("ive"),
        "suffix_ous": alpha and len(lower) > 4 and lower.endswith("ous"),
        "period": stripped in {".", "..."},
        "comma": stripped == ",",
        "question": "?" in stripped,
        "exclamation": "!" in stripped,
        "colon_semicolon": bool(stripped) and set(stripped) <= set(":;"),
        "opening_delimiter": bool(stripped) and set(stripped) <= set("([{<"),
        "closing_delimiter": bool(stripped) and set(stripped) <= set(")]}>"),
        "quote": bool(stripped) and set(stripped) <= quote_chars,
        "dash": bool(stripped) and set(stripped) <= set("-–—_"),
        "slash": bool(stripped) and set(stripped) <= set("/\\"),
        "code_operator": stripped in OPERATORS,
        "url_web": any(fragment in lower for fragment in ("http", "www", ".com", ".org", "://")),
        "currency": any(char in text for char in "$€£¥₹") or lower in {"usd", "eur", "gbp"},
        "percent": "%" in text or lower in {"percent", "percentage"},
        "month": lower in MONTHS,
        "weekday": lower in WEEKDAYS,
        "time_unit": lower in TIME_UNITS,
        "quantity_unit": lower in QUANTITY_UNITS,
        "boolean": lower in BOOLEANS,
        "programming_keyword": lower in PROGRAMMING,
    }
    return np.asarray([mapping[name] for name in FEATURES], dtype=np.uint8)


@torch.inference_mode()
def forward_census(model, ids, positions, contrast, layers):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x = x0
    value_bus = None
    products = {}
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer in layers:
            products[layer] = product[:, positions].flatten(0, 1).float()
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    selected = F.rms_norm(x[:, positions], (dimension,)).flatten(0, 1)
    # This is the checkpoint's exact clipped output logit, followed by the
    # declared positive-class mean minus negative-class mean for each label.
    logits = 30 * torch.tanh(model.lm_head(selected).float()[:, :contrast.shape[0]] / 30)
    output_scores = logits @ contrast
    return products, output_scores


def collect_split(model, rows, positions, flags, contrast, batch_size, layers):
    total = len(rows) * len(positions)
    width = model.blocks[0].mlp.Left.out_features
    product_arrays = {layer: np.empty((total, width), dtype=np.float16) for layer in layers}
    labels = np.empty((total, len(FEATURES)), dtype=np.uint8)
    current_ids = np.empty(total, dtype=np.int32)
    output_scores = np.empty((total, len(FEATURES)), dtype=np.float32)
    offset = 0
    positions_np = np.asarray(positions)
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        products, scores = forward_census(model, ids, positions, contrast, set(layers))
        count = len(raw) * len(positions)
        for layer in layers:
            product_arrays[layer][offset:offset + count] = products[layer].cpu().to(torch.float16).numpy()
        next_ids = raw[:, positions_np + 1].reshape(-1)
        labels[offset:offset + count] = flags[next_ids]
        current_ids[offset:offset + count] = raw[:, positions_np].reshape(-1)
        output_scores[offset:offset + count] = scores.cpu().numpy()
        offset += count
        if (start + batch_size) % 64 == 0:
            print("supervised census rows", min(start + batch_size, len(rows)), "/", len(rows), flush=True)
    return product_arrays, labels, current_ids, output_scores


def unigram_scores(train_ids, train_labels, test_ids, vocab_size):
    counts = np.bincount(train_ids, minlength=vocab_size).astype(np.float64)
    positives = np.zeros((vocab_size, train_labels.shape[1]), dtype=np.float64)
    np.add.at(positives, train_ids, train_labels)
    prevalence = train_labels.mean(0)
    probabilities = (positives + prevalence[None] * 4) / (counts[:, None] + 4)
    return probabilities[test_ids]


def within_token_auc(labels, scores, token_ids):
    """Pair-weighted AUROC among examples with the identical current token."""
    weighted, total_pairs, groups = 0.0, 0, 0
    for token_id in np.unique(token_ids):
        selected = token_ids == token_id
        group_labels = labels[selected]
        positive = int(group_labels.sum())
        negative = int(len(group_labels) - positive)
        if positive == 0 or negative == 0:
            continue
        pairs = positive * negative
        weighted += pairs * auc(group_labels, scores[selected])
        total_pairs += pairs
        groups += 1
    return (float(weighted / total_pairs) if total_pairs else None, groups, total_pairs)


def coalition_metrics(discovery, heldout, discovery_labels, heldout_labels, heldout_ids, sizes):
    discovery = discovery.astype(np.float32)
    heldout = heldout.astype(np.float32)
    mean = discovery.mean(0)
    std = discovery.std(0).clip(1e-4)
    z_discovery = (discovery - mean) / std
    z_heldout = (heldout - mean) / std
    rows = []
    for feature_index, name in enumerate(FEATURES):
        y = discovery_labels[:, feature_index].astype(bool)
        if not y.any() or y.all():
            rows.append({"feature": name, "aucs": {str(size): None for size in sizes}, "within_token_aucs": {str(size): None for size in sizes}, "within_token_groups": 0, "within_token_pairs": 0, "units": {str(size): [] for size in sizes}, "ninety_percent_size": None})
            continue
        gap = z_discovery[y].mean(0) - z_discovery[~y].mean(0)
        order = np.argsort(np.abs(gap))[::-1]
        aucs, within, units = {}, {}, {}
        within_groups, within_pairs = 0, 0
        for size in sizes:
            selected = order[:size]
            score = z_heldout[:, selected] @ gap[selected]
            aucs[str(size)] = measured_auc(heldout_labels[:, feature_index], score)
            within[str(size)], within_groups, within_pairs = within_token_auc(
                heldout_labels[:, feature_index], score, heldout_ids,
            )
            units[str(size)] = selected[:min(size, 16)].astype(int).tolist()
        best = aucs[str(sizes[-1])]
        threshold = None if best is None else .5 + .9 * max(best - .5, 0)
        sufficient = None if threshold is None else next((size for size in sizes if aucs[str(size)] is not None and aucs[str(size)] >= threshold), sizes[-1])
        rows.append({"feature": name, "aucs": aucs, "within_token_aucs": within, "within_token_groups": within_groups, "within_token_pairs": within_pairs, "units": units, "ninety_percent_size": sufficient})
    return rows


def main(config_path=CONFIG):
    requested = json.loads(Path(config_path).read_text())
    if "base_config" in requested:
        config = json.loads(Path(requested["base_config"]).read_text())
        config.update({key: value for key, value in requested.items() if key != "base_config"})
    else:
        config = requested
    artifact_stem = config.get("artifact_stem", "bilin18_layer01_supervised_feature_census")
    output = TENSOR_ROOT / f"{artifact_stem}.json"
    report = TENSOR_ROOT / f"{artifact_stem.upper()}.md"
    figure = TENSOR_ROOT / f"{artifact_stem}.png"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    vocab_text = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    flags = np.stack([vocabulary_features(text) for text in vocab_text])
    positive_vocab = flags.sum(0).astype(np.float32)
    negative_vocab = len(flags) - positive_vocab
    contrast_np = flags / positive_vocab.clip(1)[None] - (1 - flags) / negative_vocab.clip(1)[None]
    contrast = torch.as_tensor(contrast_np, device="cuda", dtype=torch.float32)

    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.eval()
        for parameter in module.parameters(): parameter.requires_grad_(False)
    source = np.load(config["corpus"], mmap_mode="r")
    start = config["start_row"]
    discovery_rows = source[start:start + config["discovery_rows"]]
    heldout_start = start + config["discovery_rows"]
    heldout_rows = source[heldout_start:heldout_start + config["heldout_rows"]]
    discovery_products, discovery_labels, discovery_ids, discovery_output = collect_split(model, discovery_rows, config["positions"], flags, contrast, config["batch_size"], config["layers"])
    heldout_products, heldout_labels, heldout_ids, heldout_output = collect_split(model, heldout_rows, config["positions"], flags, contrast, config["batch_size"], config["layers"])

    unigram = unigram_scores(discovery_ids, discovery_labels, heldout_ids, tokenizer.vocab_size)
    layers = {str(layer): coalition_metrics(discovery_products[layer], heldout_products[layer], discovery_labels, heldout_labels, heldout_ids, config["coalition_sizes"]) for layer in config["layers"]}
    feature_rows = []
    for index, name in enumerate(FEATURES):
        discovery_positive = int(discovery_labels[:, index].sum())
        heldout_positive = int(heldout_labels[:, index].sum())
        eligible = discovery_positive >= config["minimum_positive_per_split"] and heldout_positive >= config["minimum_positive_per_split"]
        model_auc = measured_auc(heldout_labels[:, index], heldout_output[:, index])
        unigram_auc = measured_auc(heldout_labels[:, index], unigram[:, index])
        competent = eligible and model_auc is not None and model_auc >= config["competence_auc"]
        feature_rows.append({
            "feature": name, "positive_vocabulary_tokens": int(positive_vocab[index]),
            "discovery_positive": discovery_positive, "heldout_positive": heldout_positive,
            "eligible": eligible, "model_output_auc": model_auc, "current_token_unigram_auc": unigram_auc,
            "competent": competent,
            "layers": {str(layer): layers[str(layer)][index] for layer in config["layers"]},
        })

    eligible = [row for row in feature_rows if row["eligible"]]
    competent = [row for row in feature_rows if row["competent"]]
    def med(values):
        finite = [value for value in values if value is not None and np.isfinite(value)]
        return float(np.median(finite)) if finite else None
    layer_summary = {}
    for layer in config["layers"]:
        key = str(layer)
        layer_summary[key] = {str(size): med([row["layers"][key]["aucs"][str(size)] for row in competent]) for size in config["coalition_sizes"]}
        layer_summary[key]["within_token"] = {str(size): med([row["layers"][key]["within_token_aucs"][str(size)] for row in competent]) for size in config["coalition_sizes"]}
        layer_summary[key]["median_ninety_percent_size"] = med([row["layers"][key]["ninety_percent_size"] for row in competent])
    best_layer = max(config["layers"], key=lambda layer: layer_summary[str(layer)]["64"] if competent else -1)
    single_close = [max(row["layers"][str(layer)]["aucs"]["1"] for layer in config["layers"]) >= max(row["layers"][str(layer)]["aucs"]["64"] for layer in config["layers"]) - .03 for row in competent]
    best64 = [max(row["layers"][str(layer)]["aucs"]["64"] for layer in config["layers"]) for row in competent]
    summary = {
        "examples_per_split": len(discovery_labels), "eligible_features": len(eligible), "competent_features": len(competent),
        "median_model_output_auc": med([row["model_output_auc"] for row in competent]),
        "median_unigram_auc": med([row["current_token_unigram_auc"] for row in competent]),
        "layer": layer_summary, "best_layer_at_64": best_layer,
        "single_within_003_fraction": float(np.mean(single_close)) if single_close else None,
        "median_best64_auc": med(best64),
    }
    hypotheses = {
        "S0_label_coverage": {"supported": len(eligible) >= 35},
        "S1_model_competence": {"supported": len(competent) >= 25},
        "S2_fixed_coalitions_recover_known_features": {"supported": bool(competent) and max(layer_summary[str(layer)]["64"] for layer in config["layers"]) >= .70},
        "S3_coalitions_beat_single_neurons": {"supported": bool(competent) and layer_summary[str(best_layer)]["64"] - layer_summary[str(best_layer)]["1"] >= .05},
        "S4_depth_adds_context": {"supported": bool(competent) and layer_summary[str(config["layers"][-1])]["64"] - layer_summary[str(config["layers"][0])]["64"] >= .02},
        "S5_neuron_aligned_ontology_is_sufficient": {"supported": bool(single_close) and float(np.mean(single_close)) >= .75},
        "S6_embedding_reinjection_control": {"supported": bool(competent) and med(best64) - med([row["current_token_unigram_auc"] for row in competent]) >= .03},
        "S7_evaluation_is_held_out": {"supported": True},
        "S8_depth_signal_survives_exact_current_token_control": {"supported": bool(competent) and layer_summary[str(config["layers"][-1])]["within_token"]["64"] >= .60},
    }
    output.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summary": summary, "features": feature_rows}, indent=2, allow_nan=False))

    layer_range = f"layers {min(config['layers'])}–{max(config['layers'])}"
    lines = [f"# Supervised FineWeb feature census: {layer_range}", "", "Fifty token-level target features were declared before activation collection. Native product-channel coalitions are selected on disjoint discovery sites and evaluated on disjoint held-out sites. The competence gate uses the full model's exact clipped output logits; the unigram control asks how much is predictable from the current token alone, which is especially important because this architecture reinjects the embedding at every layer.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    layer_headers = [f"L{layer} one / 64" for layer in config["layers"]]
    lines += ["", f"![Supervised census]({figure.name})", "", f"Eligible features: **{len(eligible)}/50**. Competent features: **{len(competent)}/50**. Median model-output AUROC among competent labels: **{summary['median_model_output_auc']:.3f}**; current-token unigram: **{summary['median_unigram_auc']:.3f}**.", "", "The exact-current-token control computes AUROC only between positive and negative sites whose current token ID is identical, then weights token groups by their number of positive-negative pairs. Chance is 0.5 even though the embedding is reintroduced at every layer.", "", "| Feature | Positives (D/H) | Model | Unigram | " + " | ".join(layer_headers) + " | Best 90%-size |", "|---|---:|---:|---:|" + "---:|" * (len(config["layers"]) + 1)]
    for row in sorted(feature_rows, key=lambda item: item["model_output_auc"] if item["model_output_auc"] is not None else -1, reverse=True):
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        layer_cells = [f"{fmt(row['layers'][str(layer)]['aucs']['1'])}/{fmt(row['layers'][str(layer)]['aucs']['64'])}" for layer in config["layers"]]
        ninety_sizes = [row["layers"][str(layer)]["ninety_percent_size"] for layer in config["layers"] if row["layers"][str(layer)]["ninety_percent_size"] is not None]
        best_ninety = min(ninety_sizes) if ninety_sizes else "—"
        lines.append(f"| {row['feature']} | {row['discovery_positive']}/{row['heldout_positive']} | {fmt(row['model_output_auc'])} | {fmt(row['current_token_unigram_auc'])} | " + " | ".join(layer_cells) + f" | {best_ninety} |")
    lines += ["", "## Exact-current-token control", "", "| Layer | 1 channel | 4 | 16 | 64 | 256 |", "|---:|---:|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        values = layer_summary[str(layer)]["within_token"]
        lines.append(f"| {layer} | " + " | ".join(f"{values[str(size)]:.3f}" for size in config["coalition_sizes"]) + " |")
    lines += ["", "## Interpretation guardrails", "", "- These labels are supervised probes of a declared ontology, not proof that the model internally uses those exact concepts.", "- A coalition is a fixed linear discriminator over native product channels. It is not yet the input-conditioned causal coalition tested by edit compilation.", "- Exact-current-token AUROC removes direct token identity but not longer n-gram correlations or all lexical confounds.", "- The output-logit competence score tests whether the trained model ranks the feature; it is not an exact class-probability calibration metric."]
    report.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    sizes = config["coalition_sizes"]
    for layer in config["layers"]:
        axes[0].plot(sizes, [layer_summary[str(layer)][str(size)] for size in sizes], marker="o", label=f"layer {layer}")
    axes[0].axhline(summary["median_unigram_auc"], color="black", linestyle="--", label="current-token unigram"); axes[0].set_xscale("log", base=2); axes[0].set(xlabel="Native product channels", ylabel="Median held-out AUROC", title="Feature recovery versus coalition size"); axes[0].legend(); axes[0].grid(alpha=.2)
    ordered = sorted(competent, key=lambda row: row["layers"][str(best_layer)]["aucs"]["64"] - row["layers"][str(best_layer)]["aucs"]["1"], reverse=True)
    names = [row["feature"] for row in ordered]
    gaps = [row["layers"][str(best_layer)]["aucs"]["64"] - row["layers"][str(best_layer)]["aucs"]["1"] for row in ordered]
    axes[1].barh(range(len(names)), gaps); axes[1].set_yticks(range(len(names)), names, fontsize=7); axes[1].invert_yaxis(); axes[1].axvline(.05, color="black", linestyle="--"); axes[1].set(xlabel="AUROC gain: 64 channels − best native channel", title=f"Layer {best_layer}: where coalitions matter"); axes[1].grid(axis="x", alpha=.2)
    fig.savefig(figure, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summary": summary}, indent=2)); print(report, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else CONFIG)
