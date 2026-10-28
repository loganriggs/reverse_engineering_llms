#!/usr/bin/env python3
"""Test the discovery-frozen L17H2 value echo on eleven new spelling pairs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import make_contrasts, select_records
from run_bilin18_boundary_style_H2_value_echo import (
    collect_value_deltas,
    evaluate,
    fit_bases,
)
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_H2_value_echo_lexical_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_H2_value_echo_lexical_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_H2_VALUE_ECHO_LEXICAL_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_H2_value_echo_lexical_redteam.png"


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    lexical = json.loads(Path(config["lexical_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack(
        [vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]
    )
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[
        :, FEATURES.index("period")
    ].astype(bool)
    old_pairs = token_pairs(tokenizer, base["spelling_pairs"])
    new_pairs = token_pairs(tokenizer, lexical["new_spelling_pairs"])

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    # Reproduce the old-lexeme discovery basis without looking at new lexeme effects.
    targets = set(range(4))
    discovery_cues = [pair for index, pair in enumerate(old_pairs) if index not in targets]
    begin, end = base["splits"][0]
    discovery_raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
    discovery_records = select_records(
        discovery_raw,
        boundary_flags,
        discovery_cues,
        counts,
        127,
        64,
        20261432,
    )
    print("fit old-lexeme basis", len(discovery_records), flush=True)
    discovery_values = collect_value_deltas(model, discovery_raw, discovery_records, config)
    pca_basis, random_bases = fit_bases(discovery_values, config)

    cases = (
        ("new_cues_old_targets", new_pairs, old_pairs),
        ("old_cues_new_targets", old_pairs, new_pairs),
    )
    results = []
    for case_index, (case, cues, targets) in enumerate(cases):
        contrasts = make_contrasts(
            model.lm_head.weight.shape[0],
            targets,
            0,
            config["seed"] + 100 * case_index,
            "cuda",
        )
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(
                raw,
                boundary_flags,
                cues,
                counts,
                127,
                config["examples_per_label_per_split"],
                config["seed"] + 100 * case_index + split,
            )
            print("lexical echo", case, split, len(records), flush=True)
            results.append(
                {
                    "case": case,
                    "split": split,
                    **evaluate(model, raw, records, contrasts, pca_basis, random_bases, config),
                }
            )

    new_cues = [row for row in results if row["case"] == "new_cues_old_targets"]
    new_targets = [row for row in results if row["case"] == "old_cues_new_targets"]
    o0 = True
    o1 = all(row["value_fidelity"]["1"] >= 0.50 for row in new_cues)
    o2 = all(row["causal_value_fidelity"]["1"] >= 0.95 for row in new_cues)
    o3 = all(row["causal_value_fidelity"]["1"] >= 0.95 for row in new_targets)
    o4 = all(max(row["random_causal_value_fidelity"]["1"]) < 0.25 for row in results)
    o5 = all(row["routing_fraction"] < 0.50 for row in results)
    hypotheses = {
        "O0_basis_is_old_lexeme_discovery_only": {"supported": o0},
        "O1_rank1_raw_echo_generalizes_to_new_cues": {"supported": o1},
        "O2_rank1_causal_echo_generalizes_to_new_cues": {"supported": o2},
        "O3_rank1_generalizes_to_new_output_lexemes": {"supported": o3},
        "O4_random_rank1_does_not_explain_the_effect": {"supported": o4},
        "O5_routing_alone_remains_incomplete": {"supported": o5},
        "O6_lexically_general_literal_echo": {"supported": all((o0, o1, o2, o3, o4, o5))},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(
        json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2)
    )

    lines = [
        "# New-lexeme red team of the L17H2 value echo",
        "",
        "The rank-1 value basis is refit only on the original discovery cell, then frozen. It is tested on eleven spelling pairs absent from circuit and basis discovery, both as cue swaps and as output targets.",
        "",
        "## Locked results",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += [
        "",
        "| Case/split | raw R1 | causal R1 | routing/full | best random R1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['case']}/{row['split']} | {row['value_fidelity']['1']:.3f} | "
            f"{row['causal_value_fidelity']['1']:.3f} | {row['routing_fraction']:.3f} | "
            f"{max(row['random_causal_value_fidelity']['1']):.3f} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Passing rejects the explanation that rank-1 transfer came only from cue-token overlap among the original folds. It remains a within-family spelling intervention: a generic token-swap direction or a broader lexical-source variable requires unrelated matched swap families as controls.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = [f"{'new-cue' if row['case'].startswith('new') else 'new-target'}-s{row['split']}" for row in results]
    x = np.arange(len(results))
    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.bar(x - 0.18, [row["value_fidelity"]["1"] for row in results], 0.36, label="raw value")
    axis.bar(x + 0.18, [row["causal_value_fidelity"]["1"] for row in results], 0.36, label="causal value")
    axis.set(xticks=x, xticklabels=labels, ylim=(0, 1.05), ylabel="Rank-1 fidelity", title="Old-discovery H2 value direction on new spelling lexemes")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
