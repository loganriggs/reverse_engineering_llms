#!/usr/bin/env python3
"""Discover sparse context refinements of the human source-variable grammar."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_contextual_source_variables import make_schemes
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate, select_supports
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, discovery_scores, token_categories


CONFIG = Path("configs/bilin18_selective_context_features.json")
CONTEXT_SOURCE = TENSOR_ROOT / "bilin18_contextual_source_variables.json"
OUTPUT = TENSOR_ROOT / "bilin18_selective_context_features.json"
REPORT = TENSOR_ROOT / "BILIN18_SELECTIVE_CONTEXT_FEATURES.md"
PARTIAL = TENSOR_ROOT / "bilin18_selective_context_features.partial.json"


def selective_scheme(base, crossed, selections):
    """Retain 21 base IDs and append context branches only for selected bases."""
    offsets = {}
    next_id = len(CATEGORIES)
    for category, name, cardinality in selections:
        offsets[(category, name)] = next_id
        next_id += cardinality - 1

    def assign(ids):
        current = base[ids]
        output = current.copy()
        crossed_values = {name: crossed[name][0](ids) for _, name, _ in selections}
        for category, name, cardinality in selections:
            context = crossed_values[name] % cardinality
            mask = (current == category) & (context > 0)
            output[mask] = offsets[(category, name)] + context[mask] - 1
        return output

    return assign, next_id


def rank_splits(model, raw, base_fn, schemes, config, seed_offset):
    totals = {name: np.zeros(len(CATEGORIES), dtype=np.float64) for name in config["candidate_schemes"]}
    heads = model.config.n_head
    distances = len(config["distance_bins"]) - 1
    for layer in config["layers"]:
        probes, _, _ = operator_probe_vectors(model.blocks[layer], config["operator_probes"], config["seed"] + seed_offset + layer)
        human_config = {**config, "batch_size": config["selection_batch_size"], "category_count": len(CATEGORIES)}
        human, _ = discovery_scores(model, layer, raw, base_fn, human_config, probes)
        human = human.cpu().numpy().reshape(heads, distances, len(CATEGORIES))
        for name in config["candidate_schemes"]:
            function, count = schemes[name]
            cardinality = count // len(CATEGORIES)
            local = {**config, "batch_size": config["selection_batch_size"], "category_count": count}
            split, _ = discovery_scores(model, layer, raw, function, local, probes)
            split = split.cpu().numpy().reshape(heads, distances, len(CATEGORIES), cardinality)
            totals[name] += split.sum((0, 1, 3)) - human.sum((0, 1))
    rows = []
    cardinalities = {name: schemes[name][1] // len(CATEGORIES) for name in config["candidate_schemes"]}
    for category in range(len(CATEGORIES)):
        name = max(config["candidate_schemes"], key=lambda key: totals[key][category])
        rows.append({
            "category": category,
            "category_name": CATEGORIES[category],
            "scheme": name,
            "cardinality": cardinalities[name],
            "gain": float(totals[name][category]),
        })
    return sorted(rows, key=lambda row: row["gain"], reverse=True)


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    base = token_categories(flags)
    schemes = make_schemes(base, config["seed"])
    base_fn = schemes["human"][0]
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    discovery_halves = {"a": discovery[:len(discovery) // 2], "b": discovery[len(discovery) // 2:]}
    evaluation_halves = {"a": evaluation[:len(evaluation) // 2], "b": evaluation[len(evaluation) // 2:]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    print("rank selective context splits A", flush=True)
    ranks_a = rank_splits(model, discovery_halves["a"], base_fn, schemes, config, 0)
    print("rank selective context splits B", flush=True)
    ranks_b = rank_splits(model, discovery_halves["b"], base_fn, schemes, config, 100)
    positive_b = {(row["category"], row["scheme"]) for row in ranks_b if row["gain"] > 0}
    top_a = {(row["category"], row["scheme"]) for row in ranks_a[:8]}
    top_b = {(row["category"], row["scheme"]) for row in ranks_b[:8]}
    selected = {}
    for count in config["split_counts"]:
        rows = [row for row in ranks_a if row["gain"] > 0][:count]
        triples = [(row["category"], row["scheme"], row["cardinality"]) for row in rows]
        selected[str(count)] = {"rows": rows, "program": selective_scheme(base, schemes, triples)}
    context = json.loads(CONTEXT_SOURCE.read_text())
    results = {}
    maximum_error = 0.0
    saved = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    if saved.get("config") == config:
        results = saved.get("results", {}); maximum_error = float(saved.get("maximum_error", 0.0))
    for count in config["split_counts"]:
        category_fn, category_count = selected[str(count)]["program"]
        local = {**config, "category_count": category_count}
        print("select sparse context supports", count, "splits", flush=True)
        supports = select_supports(model, discovery, category_fn, local)
        for half, panel in evaluation_halves.items():
            key = f"selective_{count}_{half}"
            if key in results:
                print("evaluate", key, "(resumed)", flush=True); continue
            print("evaluate", key, flush=True)
            row, error = evaluate(model, panel, category_fn, local, supports)
            results[key] = row["5"]; maximum_error = max(maximum_error, error)
            PARTIAL.write_text(json.dumps({"config": config, "results": results, "maximum_error": maximum_error}, indent=2))

    def value(rows, scheme, half, budget, metric="kl_recovery"):
        return rows[f"{scheme}_{half}"]["methods"][f"fine{budget}"][metric]

    static_results = context["results"]
    winners = []
    efficiency = []
    for count in config["split_counts"]:
        scheme = f"selective_{count}"
        for budget in config["support_budgets"]:
            if all(value(results, scheme, half, budget) >= value(static_results, "human", half, budget) + .002 for half in evaluation_halves):
                winners.append((count, budget))
            full_best = max(
                min(value(static_results, candidate, half, budget) for half in evaluation_halves)
                for candidate in config["candidate_schemes"]
            )
            if all(value(results, scheme, half, budget) > full_best for half in evaluation_halves):
                efficiency.append((count, budget))
    both_positive = sum((row["category"], row["scheme"]) in positive_b for row in ranks_a if row["gain"] > 0)
    hypotheses = {
        "S1_context_cancellation_is_concentrated": {"supported": both_positive >= 4, "positive_on_both": both_positive},
        "S2_selected_splits_are_stable": {"supported": len(top_a & top_b) >= 4, "top8_overlap": len(top_a & top_b)},
        "S3_selective_context_beats_static_human": {"supported": bool(winners), "winners": [list(row) for row in winners]},
        "S4_selective_context_is_more_efficient_than_full_crossing": {"supported": bool(efficiency), "winners": [list(row) for row in efficiency]},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    serial_selected = {key: {"rows": value_["rows"], "category_count": value_["program"][1]} for key, value_ in selected.items()}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "rank_a": ranks_a, "rank_b": ranks_b, "selected": serial_selected, "results": results}, indent=2, allow_nan=False))
    PARTIAL.unlink(missing_ok=True)
    lines = [
        "# Selective context-feature discovery", "",
        "A fully crossed context vocabulary was informative but inefficient. Here, a split is nominated only when separating a human source category by a readable context increases the summed squared operator-effect signal, indicating cancellation in the unsplit route. Splits are ranked on discovery half A, checked on discovery half B, then frozen for causal evaluation.", "",
        f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "Top split candidates from discovery A:", ""]
    for row in ranks_a[:10]:
        marker = "replicated-positive" if (row["category"], row["scheme"]) in positive_b else "not replicated"
        lines.append(f"- `{row['category_name']}` by `{row['scheme']}`: gain {row['gain']:.3e} ({marker})")
    lines += ["", "| Selective grammar | Categories | Recovery 128 A/B | Recovery 256 A/B |", "|---|---:|---:|---:|"]
    for count in config["split_counts"]:
        scheme = f"selective_{count}"; category_count = serial_selected[str(count)]["category_count"]
        r128 = "/".join(f"{value(results, scheme, half, 128):.4f}" for half in evaluation_halves)
        r256 = "/".join(f"{value(results, scheme, half, 256):.4f}" for half in evaluation_halves)
        lines.append(f"| {count} splits | {category_count} | {r128} | {r256} |")
    lines += [
        "", "## Guardrails", "",
        "The split score is a native-state local operator sketch, while evaluation is closed-loop final-logit behavior; disagreement is expected. Selecting on one discovery half and validating ranking signs on another limits but does not eliminate researcher degrees of freedom. The predicate library is small, and an unsuccessful selective grammar would not falsify context-dependent latent variables outside this library.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "selected": serial_selected}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
