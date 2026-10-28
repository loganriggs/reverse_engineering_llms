#!/usr/bin/env python3
"""Test frozen Layer-3 semantic modes on unseen words and later positions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_layer3_input_mode_semantics import (
    auc_from_ranks,
    collect_coordinates,
    score_ranks,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_unseen_lexical_transfer.json")
OUTPUT = TENSOR_ROOT / "layer3_unseen_lexical_transfer.json"
REPORT = TENSOR_ROOT / "LAYER3_UNSEEN_LEXICAL_TRANSFER.md"
FIGURE = TENSOR_ROOT / "layer3_unseen_lexical_transfer.png"


def finite_median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")


def apply_signal(values: np.ndarray, label: dict) -> np.ndarray:
    values = values if label["signal"] == "signed" else np.abs(values)
    return label["orientation"] * values


def safe_auc(labels: np.ndarray, scores: np.ndarray, minimum: int) -> dict:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    value = auc(labels, scores) if min(positives, negatives) >= minimum else float("nan")
    return {"auc": float(value), "positives": positives, "negatives": negatives}


def safe_auc_from_ranks(labels: np.ndarray, ranks: np.ndarray, minimum: int) -> dict:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    value = (
        auc_from_ranks(labels, ranks)
        if min(positives, negatives) >= minimum else float("nan")
    )
    return {"auc": float(value), "positives": positives, "negatives": negatives}


def lexicon_ids(tokenizer, lexicon: dict, seed_ids: dict[str, set[int]]) -> dict:
    result = {}
    for family, poles in lexicon.items():
        result[family] = {}
        for pole, words in poles.items():
            ids = []
            for word in words:
                encoded = tokenizer.encode(word, add_special_tokens=False)
                if len(encoded) != 1:
                    raise ValueError(f"unseen lexicon item is not one token: {word!r} -> {encoded}")
                ids.append(encoded[0])
            overlap = sorted(set(ids) & seed_ids[family])
            if overlap:
                raise ValueError(f"unseen lexicon overlaps seed_ids for {family}: {overlap}")
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate unseen lexicon IDs for {family}/{pole}")
            result[family][pole] = sorted(ids)
    return result


def representative_column(source: dict, family: str, basis: str) -> tuple[int, dict]:
    choice = source["representatives"][family][basis]
    for column, record in enumerate(source["records"]):
        if (
            record["family"] == family
            and record["basis"] == basis
            and record["mode"] == choice["mode"]
        ):
            return column, choice["labels"]["own"]
    raise KeyError((family, basis, choice["mode"]))


def pole_from_label(label: dict) -> str:
    pole = label["predicate"].rsplit(":", 1)[-1]
    if pole not in ("source", "target", "any"):
        raise ValueError(f"not a family pole: {label['predicate']}")
    return pole


def pole_mask(ids: np.ndarray, family: str, pole: str, lexicon: dict) -> np.ndarray:
    selected = (
        lexicon[family]["source"] + lexicon[family]["target"]
        if pole == "any" else lexicon[family][pole]
    )
    return np.isin(ids, selected)


def prior_ids(panel: dict, corpus: dict, offset: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(
        np.load(corpus["path"], mmap_mode="r")[
            corpus["start"]:corpus["start"] + corpus["rows"],
            :,
        ],
        dtype=np.int64,
    )
    positions = panel["position_ids"].astype(np.int64) - offset
    valid = positions >= 0
    result = np.zeros(len(positions), dtype=np.int64)
    result[valid] = raw[panel["row_ids"][valid], positions[valid]]
    return result, valid


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_results"]).read_text())
    source_config = source["config"]
    seed_config = json.loads(Path(config["seed_config"]).read_text())
    family_names = list(config["lexicon"])
    seed_ids = {
        family: {token for pair in seed_config["families"][family] for token in pair}
        for family in family_names
    }
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    unseen = lexicon_ids(tokenizer, config["lexicon"], seed_ids)

    saved = np.load(config["source_arrays"])
    directions_np = saved["directions"].astype(np.float32)
    embedding_scores = saved["embedding_scores"].astype(np.float32)
    panels = {
        corpus: {
            "coordinates": saved[f"{corpus}_coordinates"],
            "token_ids": saved[f"{corpus}_token_ids"],
            "row_ids": saved[f"{corpus}_row_ids"],
            "position_ids": saved[f"{corpus}_position_ids"],
        }
        for corpus in ("fineweb", "tinystories")
    }
    corpora = dict(source_config["corpora"])
    corpora["wikitext"] = config["wikitext"]

    model = load_full_model("cuda")
    wiki_config = dict(source_config)
    panels["wikitext"] = collect_coordinates(
        model,
        config["wikitext"],
        torch.from_numpy(directions_np).cuda(),
        wiki_config,
    )

    basis_names = ("native", "rotated", "random")
    representative = {
        family: {
            basis: representative_column(source, family, basis)
            for basis in basis_names
        }
        for family in family_names
    }
    records = []
    for corpus, panel in panels.items():
        for offset in config["offsets"]:
            labels_at_offset, valid = prior_ids(panel, corpora[corpus], offset)
            current_ids = panel["token_ids"][valid]
            for family in family_names:
                for basis in basis_names:
                    column, frozen_label = representative[family][basis]
                    pole = pole_from_label(frozen_label)
                    scores = apply_signal(panel["coordinates"][valid, column], frozen_label)
                    ranks = score_ranks(scores)
                    own = pole_mask(labels_at_offset[valid], family, pole, unseen)
                    own_result = safe_auc_from_ranks(
                        own, ranks, config["minimum_positive_count"],
                    )
                    wrong_rows = {}
                    for wrong in family_names:
                        if wrong == family:
                            continue
                        wrong_mask = pole_mask(labels_at_offset[valid], wrong, pole, unseen)
                        wrong_result = safe_auc_from_ranks(
                            wrong_mask, ranks, config["minimum_positive_count"],
                        )
                        if np.isfinite(wrong_result["auc"]):
                            wrong_result["oriented_auc"] = max(
                                wrong_result["auc"], 1 - wrong_result["auc"],
                            )
                        else:
                            wrong_result["oriented_auc"] = float("nan")
                        wrong_rows[wrong] = wrong_result
                    valid_wrong = {
                        name: value for name, value in wrong_rows.items()
                        if np.isfinite(value["oriented_auc"])
                    }
                    strongest_wrong = max(
                        valid_wrong,
                        key=lambda name: valid_wrong[name]["oriented_auc"],
                    ) if valid_wrong else None
                    embedding_baseline = safe_auc_from_ranks(
                        own,
                        score_ranks(apply_signal(
                            embedding_scores[current_ids, column], frozen_label,
                        )),
                        config["minimum_positive_count"],
                    )
                    records.append({
                        "corpus": corpus,
                        "offset": offset,
                        "family": family,
                        "basis": basis,
                        "mode": source["representatives"][family][basis]["mode"],
                        "pole": pole,
                        "own": own_result,
                        "embedding_baseline": embedding_baseline,
                        "strongest_wrong": strongest_wrong,
                        "strongest_wrong_auc": (
                            valid_wrong[strongest_wrong]["oriented_auc"]
                            if strongest_wrong is not None else float("nan")
                        ),
                        "wrong": wrong_rows,
                    })
                print("unseen", corpus, offset, family, flush=True)

    def find(corpus: str, offset: int, family: str, basis: str = "native") -> dict:
        return next(
            row for row in records
            if row["corpus"] == corpus and row["offset"] == offset
            and row["family"] == family and row["basis"] == basis
        )

    ood = ("tinystories", "wikitext")
    current = {
        corpus: {
            basis: [find(corpus, 0, family, basis)["own"]["auc"] for family in family_names]
            for basis in basis_names
        }
        for corpus in panels
    }
    transfer_counts = {
        corpus: sum(value >= 0.70 for value in current[corpus]["native"])
        for corpus in ood
    }
    persistence_counts = {
        corpus: sum(
            find(corpus, 1, family)["own"]["auc"] >= 0.60 for family in family_names
        )
        for corpus in ood
    }
    contextual_counts = {
        corpus: sum(
            find(corpus, 1, family)["own"]["auc"]
            - find(corpus, 1, family)["embedding_baseline"]["auc"] >= 0.05
            for family in family_names
        )
        for corpus in ood
    }
    specificity_counts = {
        corpus: sum(
            find(corpus, 0, family)["own"]["auc"]
            - find(corpus, 0, family)["strongest_wrong_auc"] >= 0.05
            for family in family_names
        )
        for corpus in ood
    }
    embedding_gaps = {
        corpus: finite_median([
            find(corpus, 0, family)["own"]["auc"]
            - find(corpus, 0, family)["embedding_baseline"]["auc"]
            for family in family_names
        ])
        for corpus in ood
    }
    hypotheses = {
        "U1_unseen_words_transfer": {
            "supported": all(value >= 4 for value in transfer_counts.values()),
            "counts_at_least_0.70": transfer_counts,
        },
        "U2_native_modes_beat_random_on_unseen_words": {
            "supported": all(
                finite_median(current[corpus]["native"])
                - finite_median(current[corpus]["random"]) >= 0.05
                for corpus in ood
            ),
            "median_gaps": {
                corpus: finite_median(current[corpus]["native"])
                - finite_median(current[corpus]["random"])
                for corpus in ood
            },
        },
        "U3_native_axes_beat_rotated_subspace": {
            "supported": all(
                finite_median(current[corpus]["native"])
                - finite_median(current[corpus]["rotated"]) >= 0.05
                for corpus in ood
            ),
            "median_gaps": {
                corpus: finite_median(current[corpus]["native"])
                - finite_median(current[corpus]["rotated"])
                for corpus in ood
            },
        },
        "U4_semantics_persist_one_token": {
            "supported": all(value >= 3 for value in persistence_counts.values()),
            "counts_at_least_0.60": persistence_counts,
        },
        "U5_contextual_state_beats_current_embedding_after_token": {
            "supported": all(value >= 3 for value in contextual_counts.values()),
            "counts_with_gap_at_least_0.05": contextual_counts,
        },
        "U6_unseen_family_labels_are_specific": {
            "supported": all(value >= 4 for value in specificity_counts.values()),
            "counts_with_gap_at_least_0.05": specificity_counts,
        },
        "U7_current_token_abstraction_exceeds_embedding": {
            "supported": all(value >= 0.05 for value in embedding_gaps.values()),
            "median_gaps": embedding_gaps,
        },
        "U8_increasing_scale_generalizes": {
            "supported": all(
                find(corpus, 0, "increasing_scale")["own"]["auc"] >= 0.65
                for corpus in panels
            ),
            "values": {
                corpus: find(corpus, 0, "increasing_scale")["own"]["auc"]
                for corpus in panels
            },
        },
        "U9_unseen_transfer_is_corpus_stable": {
            "supported": sum(
                abs(find("tinystories", 0, family)["own"]["auc"]
                    - find("wikitext", 0, family)["own"]["auc"]) <= 0.15
                for family in family_names
            ) >= 4,
            "absolute_differences": {
                family: abs(find("tinystories", 0, family)["own"]["auc"]
                            - find("wikitext", 0, family)["own"]["auc"])
                for family in family_names
            },
        },
    }

    summary = []
    for family in family_names:
        row = {"family": family, "pole": find("fineweb", 0, family)["pole"]}
        for corpus in panels:
            native_row = find(corpus, 0, family)
            row[f"{corpus}_auc"] = native_row["own"]["auc"]
            row[f"{corpus}_embedding"] = native_row["embedding_baseline"]["auc"]
            row[f"{corpus}_offset1"] = find(corpus, 1, family)["own"]["auc"]
            row[f"{corpus}_wrong"] = native_row["strongest_wrong_auc"]
            row[f"{corpus}_rotated"] = find(corpus, 0, family, "rotated")["own"]["auc"]
            row[f"{corpus}_random"] = find(corpus, 0, family, "random")["own"]["auc"]
        summary.append(row)

    result = {
        "config": config,
        "lexicon_ids": unseen,
        "hypotheses": hypotheses,
        "summary": summary,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for family in family_names:
        axes[0].plot(
            config["offsets"],
            [find("tinystories", offset, family)["own"]["auc"] for offset in config["offsets"]],
            marker="o", label=family,
        )
    axes[0].axhline(0.5, color="black", linewidth=1)
    axes[0].set_xticks(config["offsets"])
    axes[0].set_xlabel("Tokens after unseen lexicon item")
    axes[0].set_ylabel("TinyStories AUROC")
    axes[0].set_ylim(0.20, 1.0)
    axes[0].set_title("Semantic coordinate persistence")
    axes[0].legend(fontsize=8)
    x = np.arange(len(family_names))
    width = 0.25
    axes[1].bar(x - width, current["wikitext"]["native"], width, label="native")
    axes[1].bar(x, current["wikitext"]["rotated"], width, label="rotated")
    axes[1].bar(x + width, current["wikitext"]["random"], width, label="random")
    axes[1].axhline(0.5, color="black", linewidth=1)
    axes[1].set_xticks(x, family_names, rotation=25, ha="right")
    axes[1].set_ylim(0.35, 1.0)
    axes[1].set_ylabel("WikiText unseen-word AUROC")
    axes[1].set_title("Axes versus subspace and random control")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Unseen-word and context-persistence test for Layer-3 modes",
        "",
        "The native, rotated-subspace, and isotropic-random representative axes, including their "
        "signal type, sign, and source/target pole, are frozen from the exact-seed-token experiment. "
        "This follow-up evaluates disjoint, manually declared one-token lexicons and adds WikiText-103 "
        "as a third corpus. Offset tests ask whether the coordinate remains informative after the "
        "lexical token is no longer the current repeatedly injected embedding.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Unseen lexical transfer](layer3_unseen_lexical_transfer.png)",
        "",
        "| Family/pole | Fine current | Tiny current / +1 | Wiki current / +1 | Tiny embed | Wiki embed | Tiny wrong | Wiki wrong | Wiki rotated/random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['family']} / {row['pole']} | {row['fineweb_auc']:.3f} | "
            f"{row['tinystories_auc']:.3f} / {row['tinystories_offset1']:.3f} | "
            f"{row['wikitext_auc']:.3f} / {row['wikitext_offset1']:.3f} | "
            f"{row['tinystories_embedding']:.3f} | {row['wikitext_embedding']:.3f} | "
            f"{row['tinystories_wrong']:.3f} | {row['wikitext_wrong']:.3f} | "
            f"{row['wikitext_rotated']:.3f} / {row['wikitext_random']:.3f} |"
        )
    lines += [
        "",
        "Offset-zero success beyond the direct embedding baseline would support a transformed semantic "
        "coordinate; equality instead supports inherited lexical geometry. Offset-one success with a "
        "weak current-token embedding baseline is stronger evidence of contextual transport. Rotated "
        "axes matching native axes means the explanatory object is the family subspace, not its exact "
        "weight eigenbasis. Wrong-family AUROC is oriented adversarially and therefore detects shared "
        "or multiplexed axes even when the sign differs.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
