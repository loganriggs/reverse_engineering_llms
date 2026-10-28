#!/usr/bin/env python3
"""Budget-matched red team for the scaled readable source grammar."""

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
from run_bilin18_typed_attention_edge_vocabulary import token_categories


CONFIG = Path("configs/bilin18_human_ontology_budget_control.json")
SCALED = TENSOR_ROOT / "bilin18_readable_grammar_scaled_budget.json"
OUTPUT = TENSOR_ROOT / "bilin18_human_ontology_budget_control.json"
REPORT = TENSOR_ROOT / "BILIN18_HUMAN_ONTOLOGY_BUDGET_CONTROL.md"


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        vocabulary_features(tokenizer.decode([index]))
        for index in range(tokenizer.vocab_size)
    ])
    categories = token_categories(flags)
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(
        corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]],
        dtype=np.int64,
    )
    evaluation = np.asarray(
        corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1],
        dtype=np.int64,
    )
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    supports = select_supports(model, discovery, categories, config)
    results, maximum_error = evaluate(model, evaluation, categories, config, supports)
    methods = results["5"]["methods"]
    scaled = json.loads(SCALED.read_text())["results"]
    grammar_42 = max(
        scaled[split]["methods"]["fine256"]["kl_recovery"]
        for split in ("42_256_a", "42_256_b")
    )
    ordered = [methods[f"fine{budget}"]["kl_recovery"] for budget in config["support_budgets"]]
    hypotheses = {
        "R1_budget_alone_explains_scaled_grammar_gain": {
            "supported": methods["fine256"]["kl_recovery"] >= grammar_42,
            "human_minus_readable_42": methods["fine256"]["kl_recovery"] - grammar_42,
        },
        "R2_more_human_routes_improve_recovery": {
            "supported": all(left < right for left, right in zip(ordered, ordered[1:])),
        },
        "R3_readable_42_adds_behavior_beyond_human_categories": {
            "supported": grammar_42 >= methods["fine256"]["kl_recovery"] + .01,
            "readable_42_minus_human": grammar_42 - methods["fine256"]["kl_recovery"],
        },
        "R4_budget_matched_human_reaches_high_prediction_fidelity": {
            "supported": methods["fine256"]["top1_agreement"] >= .75,
        },
        "native_branch_matches_checkpoint": {
            "supported": maximum_error <= 1e-4,
            "maximum_error": maximum_error,
        },
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "readable_42_fine256_recovery": grammar_42,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = [
        "# Human ontology route-budget control", "",
        "This post-result red team gives the original 21 human source categories the same 192/256 route budgets used by the larger readable grammars. It separates a better variable vocabulary from simply retaining more typed edges.", "",
        f"Maximum native-branch error: `{maximum_error:.2e}`. Readable-42/256 recovery: `{grammar_42:.4f}`.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Human variables / routes | KL recovery | CE delta | Top-1 |", "|---|---:|---:|---:|"]
    for budget in config["support_budgets"]:
        row = methods[f"fine{budget}"]
        lines.append(f"| 21 / {budget} | {row['kl_recovery']:.4f} | {row['mean_ce_delta']:+.4f} | {row['top1_agreement']:.3f} |")
    lines += [
        "", "## Guardrails", "",
        "This is a post-result red team, not an independent preregistered discovery. Both ontologies use the same discovery/evaluation panels and route selector. A human win would show that the scaled grammar result does not establish novel variables; a readable-grammar win would still require targeted intervention examples before naming those extra distinctions as model features.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "methods": methods}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
