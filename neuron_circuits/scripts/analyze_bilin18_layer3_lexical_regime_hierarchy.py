#!/usr/bin/env python3
"""Post-hoc hierarchical red-team of shared-versus-residual regime labels."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_lexical_regime_hierarchy.json")


def regime(record: dict) -> float:
    effects = record["effects"]
    return float(
        effects["shared_like8"]["family_patch_fidelity"]
        - effects["family_residual24"]["family_patch_fidelity"]
    )


def interval(values: list[float], rng: np.random.Generator, replicates: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [float("nan"), float("nan")]
    draws = rng.choice(array, size=(replicates, len(array)), replace=True)
    boot = np.median(draws, axis=1)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def grouped(records: list[dict]) -> dict:
    result = defaultdict(list)
    for record in records:
        pair = record.get("pair") or f"{record['source_id']}->{record['target_id']}"
        result[(record["corpus"], record["family"], pair)].append(regime(record))
    return result


def main() -> None:
    config = json.loads(CONFIG.read_text())
    seed_source = json.loads(Path(config["seed_results"]).read_text())
    unseen_source = json.loads(Path(config["unseen_results"]).read_text())
    seed_config = json.loads(Path(config["seed_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    rng = np.random.default_rng(config["bootstrap_seed"])
    families = list(seed_config["families"])
    corpora = ("fineweb", "tinystories")
    unseen_groups = grouped(unseen_source["records"])
    seed_groups = grouped(seed_source["causal_records"])

    def summarize(groups: dict, corpus: str, family: str) -> dict:
        selected = {pair: values for (c, f, pair), values in groups.items()
                    if c == corpus and f == family}
        pair_medians = {pair: float(np.median(values)) for pair, values in selected.items()}
        all_values = [value for values in selected.values() for value in values]
        consistencies = [
            max(sum(value > 0 for value in values), sum(value < 0 for value in values)) / len(values)
            for values in selected.values()
        ]
        balanced = float(np.median(list(pair_medians.values())))
        occurrence = float(np.median(all_values))
        return {
            "contexts": len(all_values),
            "pairs": len(selected),
            "occurrence_weighted_median": occurrence,
            "pair_balanced_median": balanced,
            "pair_bootstrap_95": interval(
                list(pair_medians.values()), rng, config["bootstrap_replicates"],
            ),
            "aggregation_sign_agrees": bool(np.sign(occurrence) == np.sign(balanced)),
            "positive_pair_fraction": float(np.mean([value > 0 for value in pair_medians.values()])),
            "median_within_pair_sign_consistency": float(np.median(consistencies)),
            "pair_medians": pair_medians,
        }

    unseen = {
        corpus: {family: summarize(unseen_groups, corpus, family) for family in families}
        for corpus in corpora
    }
    seed = {
        corpus: {family: summarize(seed_groups, corpus, family) for family in families}
        for corpus in corpora
    }
    cross_corpus = {}
    for family in families:
        fine = unseen["fineweb"][family]["pair_medians"]
        tiny = unseen["tinystories"][family]["pair_medians"]
        common = sorted(set(fine) & set(tiny))
        agreements = [np.sign(fine[pair]) == np.sign(tiny[pair]) for pair in common]
        cross_corpus[family] = {
            "common_pairs": len(common),
            "sign_agreement_fraction": float(np.mean(agreements)) if agreements else float("nan"),
            "disagreeing_pairs": [pair for pair in common if np.sign(fine[pair]) != np.sign(tiny[pair])],
        }

    token_names = {}
    for record in unseen_source["records"]:
        pair = f"{record['source_id']}->{record['target_id']}"
        token_names[pair] = (
            tokenizer.decode([record["source_id"]]), tokenizer.decode([record["target_id"]])
        )

    diagnostic_results = {
        "D1_aggregation_sign_agreement": {
            f"{corpus}/{family}": unseen[corpus][family]["aggregation_sign_agrees"]
            for corpus in corpora for family in families
        },
        "D2_unanimous_pair_sign": {
            f"{corpus}/{family}": unseen[corpus][family]["positive_pair_fraction"] in (0.0, 1.0)
            for corpus in corpora for family in families
        },
        "D3_cross_corpus_pair_sign_agreement": cross_corpus,
        "D4_context_sign_consistency": {
            f"{corpus}/{family}": unseen[corpus][family]["median_within_pair_sign_consistency"]
            for corpus in corpora for family in families
        },
    }

    output_stem = config["output_stem"]
    output = TENSOR_ROOT / f"{output_stem}.json"
    report = TENSOR_ROOT / f"{output_stem.upper()}.md"
    figure_path = TENSOR_ROOT / f"{output_stem}.png"
    output.write_text(json.dumps({
        "config": config,
        "diagnostics": diagnostic_results,
        "unseen": unseen,
        "seed": seed,
        "cross_corpus": cross_corpus,
        "token_names": token_names,
    }, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(families))
    width = .19
    for index, (corpus, field) in enumerate((
        ("fineweb", "occurrence_weighted_median"),
        ("fineweb", "pair_balanced_median"),
        ("tinystories", "occurrence_weighted_median"),
        ("tinystories", "pair_balanced_median"),
    )):
        axes[0].bar(
            x + (index - 1.5) * width,
            [unseen[corpus][family][field] for family in families],
            width, label=f"{corpus} {field.replace('_median', '')}",
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x, families, rotation=25, ha="right")
    axes[0].set_ylabel("Shared-like minus residual fidelity")
    axes[0].set_title("Aggregation changes the estimand")
    axes[0].legend(fontsize=7)

    scale_pairs = unseen["fineweb"]["increasing_scale"]["pair_medians"]
    labels = [f"{token_names[p][0]}→{token_names[p][1]}" for p in scale_pairs]
    y = np.arange(len(labels))
    axes[1].barh(y - .18, list(scale_pairs.values()), .36, label="FineWeb")
    tiny_scale = unseen["tinystories"]["increasing_scale"]["pair_medians"]
    axes[1].barh(y + .18, [tiny_scale.get(pair, np.nan) for pair in scale_pairs], .36, label="TinyStories")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("Pair-balanced regime score")
    axes[1].set_title("Scale is lexical- and corpus-conditional")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    lines = [
        "# Hierarchical red-team of Layer-3 causal regime labels",
        "",
        "This is a post-hoc aggregation diagnostic. Positive values mean the frozen shared-like eight-dimensional route better reproduces the family's top-32 patch; negative values favor the residual 24. Occurrence weighting asks about a typical occurrence in the sampled corpus. Pair balancing asks about a typical frozen lexical contrast. Neither is universally privileged.",
        "",
        f"![Hierarchical regime audit]({figure_path.name})",
        "",
        "| Corpus / family | Contexts / pairs | Occurrence median | Pair-balanced median [95% pair bootstrap] | Positive pairs | Within-pair sign consistency |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpora:
        for family in families:
            row = unseen[corpus][family]
            low, high = row["pair_bootstrap_95"]
            lines.append(
                f"| {corpus} / {family} | {row['contexts']} / {row['pairs']} | "
                f"{row['occurrence_weighted_median']:.3f} | {row['pair_balanced_median']:.3f} "
                f"[{low:.3f}, {high:.3f}] | {row['positive_pair_fraction']:.2f} | "
                f"{row['median_within_pair_sign_consistency']:.2f} |"
            )
    lines += [
        "",
        "## What survives the red-team",
        "",
        "FineWeb sentiment and polarity are unanimously shared-like across lexical pairs; gender and tense are unanimously residual-like. These four signs also hold for every available TinyStories pair. Scale is different: all ten FineWeb pairs are shared-like in median, but TinyStories contains both shared-like and residual-like scale pairs. Its occurrence-weighted and pair-balanced summaries differ sharply because frequent words dominate the former.",
        "",
        "The family label is therefore a useful prior, not a complete causal type. A regime label needs an explicit sampling unit: token occurrences for expected corpus behavior, lexical pairs for vocabulary-level claims, and individual contexts for mechanistic universality. The bootstrap intervals quantify lexical sampling only; they do not turn the manually selected lexicon into a population-random sample.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
