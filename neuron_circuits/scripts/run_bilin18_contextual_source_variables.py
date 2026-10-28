#!/usr/bin/env python3
"""Search for source variables whose identity depends on causal prefix context."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate, select_supports
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, token_categories


CONFIG = Path("configs/bilin18_contextual_source_variables.json")
OUTPUT = TENSOR_ROOT / "bilin18_contextual_source_variables.json"
REPORT = TENSOR_ROOT / "BILIN18_CONTEXTUAL_SOURCE_VARIABLES.md"
PARTIAL = TENSOR_ROOT / "bilin18_contextual_source_variables.partial.json"


def make_schemes(base, seed):
    punctuation = np.asarray([
        CATEGORIES.index(name) for name in (
            "period", "comma", "colon_semicolon", "question_exclamation",
            "opening_delimiter", "closing_delimiter", "quote", "dash", "slash",
            "other_punctuation",
        )
    ])
    word = np.asarray([
        CATEGORIES.index(name) for name in (
            "programming_keyword", "function_word", "suffix", "capitalized", "lowercase",
        )
    ])
    newline = CATEGORIES.index("newline")
    sentence_end = np.asarray([CATEGORIES.index("period"), CATEGORIES.index("question_exclamation")])
    rng = np.random.default_rng(seed)
    random_role = rng.integers(0, 4, size=len(base), dtype=np.int64)

    def static(ids):
        return base[ids]

    def previous_role(ids):
        current = base[ids]
        previous = np.full_like(current, newline)
        previous[:, 1:] = current[:, :-1]
        role = np.full_like(current, 3)
        role[np.isin(previous, word)] = 1
        role[np.isin(previous, punctuation)] = 2
        role[previous == newline] = 0
        return current * 4 + role

    def line_phase(ids):
        current = base[ids]
        previous = np.full_like(current, newline)
        previous[:, 1:] = current[:, :-1]
        phase = np.full_like(current, 2)
        phase[np.isin(previous, sentence_end)] = 1
        phase[previous == newline] = 0
        return current * 3 + phase

    def recent_repeat(ids):
        current = base[ids]
        repeated = np.zeros_like(current)
        for offset in range(1, 9):
            repeated[:, offset:] |= ids[:, offset:] == ids[:, :-offset]
        return current * 2 + repeated

    def random_previous(ids):
        current = base[ids]
        previous_ids = np.zeros_like(ids)
        previous_ids[:, 1:] = ids[:, :-1]
        return current * 4 + random_role[previous_ids]

    return {
        "human": (static, len(CATEGORIES)),
        "previous_role": (previous_role, len(CATEGORIES) * 4),
        "line_phase": (line_phase, len(CATEGORIES) * 3),
        "recent_repeat": (recent_repeat, len(CATEGORIES) * 2),
        "random_previous_role": (random_previous, len(CATEGORIES) * 4),
    }


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    base = token_categories(flags)
    schemes = make_schemes(base, config["seed"])
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(
        corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]],
        dtype=np.int64,
    )
    evaluation = np.asarray(
        corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1],
        dtype=np.int64,
    )
    halves = {"a": evaluation[:len(evaluation) // 2], "b": evaluation[len(evaluation) // 2:]}
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    saved = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    results = saved.get("results", {}) if saved.get("config") == config else {}
    maximum_error = float(saved.get("maximum_error", 0.0)) if saved.get("config") == config else 0.0
    for name in config["schemes"]:
        if all(f"{name}_{half}" in results for half in halves):
            print("contextual scheme", name, "(fully resumed)", flush=True)
            continue
        category_fn, count = schemes[name]
        local = {**config, "category_count": count}
        print("select contextual supports", name, flush=True)
        supports = select_supports(model, discovery, category_fn, local)
        for half, panel in halves.items():
            key = f"{name}_{half}"
            if key in results:
                print("evaluate contextual", key, "(resumed)", flush=True)
                continue
            print("evaluate contextual", key, flush=True)
            row, error = evaluate(model, panel, category_fn, local, supports)
            results[key] = row["5"]
            maximum_error = max(maximum_error, error)
            PARTIAL.write_text(json.dumps({"config": config, "results": results, "maximum_error": maximum_error}, indent=2))

    def value(scheme, half, budget, metric="kl_recovery"):
        return results[f"{scheme}_{half}"]["methods"][f"fine{budget}"][metric]

    previous_wins = []
    context_wins = []
    for budget in config["support_budgets"]:
        previous_wins.append(all(value("previous_role", half, budget) >= value("random_previous_role", half, budget) + .01 for half in halves))
        for scheme in ("previous_role", "line_phase", "recent_repeat"):
            if all(value(scheme, half, budget) >= value("human", half, budget) + .005 for half in halves):
                context_wins.append((scheme, budget))
    random_control = all(
        all(value(scheme, half, budget) > value("random_previous_role", half, budget) for half in halves)
        for scheme, budget in context_wins
    ) if context_wins else False
    best_top1 = max(value(scheme, half, budget, "top1_agreement") for scheme in config["schemes"] for half in halves for budget in config["support_budgets"])
    hypotheses = {
        "C1_previous_role_beats_random_context": {"supported": any(previous_wins), "by_budget": dict(zip(map(str, config["support_budgets"]), previous_wins))},
        "C2_context_beats_static_human": {"supported": bool(context_wins), "winners": [list(row) for row in context_wins]},
        "C3_context_advantage_is_not_only_more_types": {"supported": random_control},
        "C4_context_program_is_prediction_useful": {"supported": best_top1 >= .85, "best_top1": best_top1},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    PARTIAL.unlink(missing_ok=True)
    lines = [
        "# Context-conditioned source variables", "",
        "Static token classes cannot represent a token changing role with its causal prefix. This experiment crosses the 21 human token categories with three readable contextual predicates and compares them with static human categories and a matched random previous-token role. Typed routes are discovered once, then evaluated on two disjoint FineWeb panels.", "",
        f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Scheme | Categories | Recovery 128 A/B | Recovery 256 A/B | Top-1 256 A/B |", "|---|---:|---:|---:|---:|"]
    for name in config["schemes"]:
        count = schemes[name][1]
        r128 = "/".join(f"{value(name, half, 128):.4f}" for half in halves)
        r256 = "/".join(f"{value(name, half, 256):.4f}" for half in halves)
        t256 = "/".join(f"{value(name, half, 256, 'top1_agreement'):.3f}" for half in halves)
        lines.append(f"| {name} | {count} | {r128} | {r256} | {t256} |")
    lines += [
        "", "## Guardrails", "",
        "The contextual predicates are hand-written probes, not a complete grammar. Crossing variables increases the type inventory and can fragment route evidence, so failure does not show that contextual state is absent. Conversely, a gain would show that the predicate predicts useful routing, not that it is a unique or causally atomic feature. The random control is matched in category count but not exactly in empirical group sizes.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
