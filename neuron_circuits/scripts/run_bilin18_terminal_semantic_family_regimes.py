#!/usr/bin/env python3
"""Replicate terminal representation regimes across five coherent families."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import select_records
from run_bilin18_boundary_style_H2_weight_outlier import spectral_atlas
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_terminal_edit_dynamic_selector import collect_examples, summarize
from run_bilin18_terminal_head_k90 import collect as collect_k90


CONFIG = Path("configs/bilin18_terminal_semantic_family_regimes.json")
OUTPUT = TENSOR_ROOT / "bilin18_terminal_semantic_family_regimes.json"
REPORT = TENSOR_ROOT / "BILIN18_TERMINAL_SEMANTIC_FAMILY_REGIMES.md"
FIGURE = TENSOR_ROOT / "bilin18_terminal_semantic_family_regimes.png"


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    family_source = json.loads(Path(config["family_config"]).read_text())
    dynamic_config = json.loads(Path(config["dynamic_config"]).read_text())
    k90_config = json.loads(Path(config["k90_config"]).read_text())
    for child in (dynamic_config, k90_config):
        child["batch_size"] = config["batch_size"]
        child["examples_per_label_per_split"] = config["examples_per_label_per_split"]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[:, FEATURES.index("period")].astype(bool)
    families = {name: [tuple(pair) for pair in pairs] for name, pairs in family_source["families"].items()}

    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    _, bases = spectral_atlas(model)

    panels = {}
    for family_index, (family, pairs) in enumerate(families.items()):
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(raw, boundary_flags, pairs, counts, 127, config["examples_per_label_per_split"], config["seed"] + 100 * family_index + split)
            print("semantic family", family, split, len(records), flush=True)
            dynamic = collect_examples(model, raw, records, bases, dynamic_config)
            k90 = collect_k90(model, raw, records, k90_config)
            panels[(family, split)] = {"records": records, "dynamic": dynamic, "k90": k90}

    static_heads = {family: int(panels[(family, 0)]["dynamic"]["finite_rank1"].mean(0).argmax()) for family in families}
    results = []
    for family in families:
        for split in range(2):
            panel = panels[(family, split)]
            labels = [int(row[4]) for row in panel["records"]]
            results.append({
                "family": family,
                "split": split,
                "label_counts": {str(label): labels.count(label) for label in (0, 1)},
                "static_head": static_heads[family],
                **summarize(panel["dynamic"], static_heads[family]),
                "joint": panel["k90"],
            })
    lookup = {(row["family"], row["split"]): row for row in results}
    minimum = config["minimum_families"]
    m0 = all(row["label_counts"]["0"] == 40 and row["label_counts"]["1"] == 40 for row in results)
    short = sum(all(lookup[(family, split)]["joint"]["quotient_k90_median"] <= 3 for split in range(2)) for family in families)
    m1 = short >= minimum
    m2 = all(abs(lookup[(family, 0)]["joint"]["quotient_k90_median"] - lookup[(family, 1)]["joint"]["quotient_k90_median"]) <= 1 for family in families)
    rank1 = sum(all(lookup[(family, split)]["rank1_full_correlation_median"] >= .7 and lookup[(family, split)]["rank1_full_winner_agreement"] >= .5 for split in range(2)) for family in families)
    m3 = rank1 >= minimum
    static = sum(lookup[(family, 1)]["static_winner_accuracy"] >= .5 for family in families)
    m4 = static >= minimum
    routing = sum(all(lookup[(family, split)]["context_oracle_effect_fraction"] >= .7 for split in range(2)) for family in families)
    m5 = routing >= minimum
    all_k90 = [row["joint"]["quotient_k90_median"] for row in results]
    m6 = float(np.median(all_k90)) <= 3
    hypotheses = {
        "M0_all_family_panels_are_balanced": {"supported": m0},
        "M1_coherent_families_have_short_terminal_programs": {"supported": m1, "families": short},
        "M2_family_program_length_replicates": {"supported": m2},
        "M3_weight_rank1_preserves_family_head_order": {"supported": m3, "families": rank1},
        "M4_one_static_head_dominates_each_family": {"supported": m4, "families": static},
        "M5_routing_only_selector_is_family_useful": {"supported": m5, "families": routing},
        "M6_coherent_families_are_shorter_than_heterogeneous_control": {"supported": m6, "median_k90": float(np.median(all_k90))},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "static_heads": static_heads, "results": results}, indent=2))

    lines = ["# Terminal regimes across five semantic families", "", "Five lexical relation families fixed in an earlier experiment—sentiment, gender, tense, increasing scale, and polarity—are evaluated with the unchanged natural-edit, per-example selector, complete-head, and joint-k90 assays. Each cell has 80 orientation-balanced FineWeb examples.", "", "## Locked results", ""]
    for name, value in hypotheses.items():
        suffix = f" ({value['families']}/5 families)" if "families" in value else ""
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**{suffix}")
    lines += ["", "| Family/split | static H/winner | routing effect | rank1↔full r/winner | joint k90/p90 | full support/top share |", "|---|---:|---:|---:|---:|---:|"]
    for row in results:
        joint = row["joint"]
        lines.append(f"| {row['family']}/{row['split']} | H{row['static_head']}/{row['static_winner_accuracy']:.3f} | {row['context_oracle_effect_fraction']:.3f} | {row['rank1_full_correlation_median']:.3f}/{row['rank1_full_winner_agreement']:.3f} | {joint['quotient_k90_median']:.1f}/{joint['quotient_k90_p90']:.1f} | {row['full_positive_effective_support_median']:.2f}/{row['full_top_positive_share_median']:.3f} |")
    lines += ["", "## Interpretation boundary", "", "These human-declared families were fixed before this terminal experiment, but they differ in frequency, grammatical role, pair count, and contextual naturalness. A family-level k90 advantage supports a relation-conditioned regime; it does not show that human semantic coherence itself causes sparsity. The terminal post-boundary branch remains only one part of each full natural edit."]
    REPORT.write_text("\n".join(lines) + "\n")

    family_names = list(families); x = np.arange(len(family_names)); figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for split in range(2):
        axes[0].plot(x, [lookup[(family, split)]["joint"]["quotient_k90_median"] for family in family_names], marker="o", label=f"split {split}")
        axes[1].plot(x, [lookup[(family, split)]["static_winner_accuracy"] for family in family_names], marker="o", label=f"split {split}")
    axes[0].axhline(5, color="black", linestyle="--", linewidth=.8, label="heterogeneous control"); axes[0].set(xticks=x, xticklabels=family_names, ylabel="Median joint k90", title="Conditional program length"); axes[0].tick_params(axis="x", rotation=25); axes[0].legend()
    axes[1].set(xticks=x, xticklabels=family_names, ylim=(0, 1.05), ylabel="Static-head winner rate", title="Does one operation dominate?"); axes[1].tick_params(axis="x", rotation=25); axes[1].legend()
    for axis in axes: axis.grid(alpha=.2)
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "static_heads": static_heads, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
