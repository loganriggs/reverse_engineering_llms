#!/usr/bin/env python3
"""Decode seed semantics from nested weight-defined Layer-3 input subspaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import auc
from run_bilin18_layer3_frozen_channel_transfer import (
    channel_state,
    recover_configs,
    top_indices,
)
from run_bilin18_layer3_input_mode_semantics import collect_coordinates
from run_bilin18_layer3_native_bank_input_forms import input_metric
from run_bilin18_layer3_unseen_lexical_transfer import pole_from_label, prior_ids
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_semantic_subspace_rank.json")
OUTPUT = TENSOR_ROOT / "layer3_semantic_subspace_rank.json"
REPORT = TENSOR_ROOT / "LAYER3_SEMANTIC_SUBSPACE_RANK.md"
FIGURE = TENSOR_ROOT / "layer3_semantic_subspace_rank.png"


def stable_rng(seed: int, key: str) -> np.random.Generator:
    offset = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed + offset)


def sampled_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    maximum_negative_examples: int,
    minimum_positive_count: int,
    seed: int,
    key: str,
) -> dict:
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    if len(positive) < minimum_positive_count or not len(negative):
        return {"auc": float("nan"), "positives": len(positive), "negatives": 0}
    if len(negative) > maximum_negative_examples:
        negative = stable_rng(seed, key).choice(
            negative, size=maximum_negative_examples, replace=False,
        )
    selected = np.concatenate((positive, negative))
    return {
        "auc": float(auc(labels[selected], scores[selected])),
        "positives": int(len(positive)),
        "negatives": int(len(negative)),
    }


def finite_median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")


def rank_80(curve: dict[int, float]) -> int | None:
    return next((rank for rank in sorted(curve) if curve[rank] >= 0.80), None)


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    semantic_source = json.loads(Path(config["source_results"]).read_text())
    semantic_config = semantic_source["config"]
    unseen_config = json.loads(Path(config["unseen_config"]).read_text())
    unseen_result = json.loads(Path(config["unseen_results"]).read_text())
    seed_config = json.loads(Path(unseen_config["seed_config"]).read_text())
    families = seed_config["families"]
    family_names = list(families)
    unseen_ids = unseen_result["lexicon_ids"]

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    block = model.blocks[semantic_config["layer"]]

    # Recover inclusive family channel banks from the original frozen discovery
    # panel, then construct a nested rotation-invariant family_subspace basis.
    input_config = json.loads(Path(semantic_config["source_config"]).read_text())
    specificity_config = json.loads(Path(input_config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    fixed, _ = recover_configs(frozen_config)
    native_saved = json.loads(Path(semantic_config["native_results"]).read_text())
    discovery_corpus = frozen_config["discovery_corpus"]
    semantic_rows = [
        row for row in native_saved["semantic_records"]
        if row["corpus"] == discovery_corpus
        and row["context_index"] == frozen_config["discovery_context_index"]
    ]
    fixed_array = np.load(fixed["evaluation"][discovery_corpus]["path"], mmap_mode="r")
    family_scores = {family: [] for family in family_names}
    for row in semantic_rows:
        raw = np.asarray(
            fixed_array[
                row["row"],
                row["source_start"]:row["source_start"] + fixed["sequence_length"],
            ], dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[fixed["edit_position"]] = int(row["pair"].split("->")[1])
        base = forward_layers(
            model, torch.as_tensor(raw, device="cuda")[None],
            last_layer=semantic_config["layer"],
        )[semantic_config["layer"]]
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None],
            last_layer=semantic_config["layer"],
        )[semantic_config["layer"]]
        family_scores[row["family"]].append(channel_state(
            model, semantic_config["layer"], base, counterfactual,
            fixed["causal_position"],
        )["scores"].abs().cpu())

    family_subspace = {}
    for family in family_names:
        indices = top_indices(
            torch.stack(family_scores[family]).mean(0), config["channel_rank"],
        ).cuda()
        family_subspace[family] = input_metric(
            block, indices, config["output_rank"],
        )["vectors"][:, : config["basis_rank"]]
        print("family_subspace", family, flush=True)
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    random_subspace = torch.linalg.qr(torch.randn(
        model.config.n_embd, config["basis_rank"], generator=generator, device="cuda",
    ), mode="reduced").Q

    corpora = dict(semantic_config["corpora"])
    corpora["wikitext"] = unseen_config["wikitext"]
    seed_ids = {
        family: {
            "source": [pair[0] for pair in pairs],
            "target": [pair[1] for pair in pairs],
            "any": [token for pair in pairs for token in pair],
        }
        for family, pairs in families.items()
    }
    frozen_labels = {
        family: semantic_source["representatives"][family]["native"]["labels"]["own"]
        for family in family_names
    }

    all_records = []
    basis_map = dict(family_subspace)
    basis_map["random"] = random_subspace
    for basis_name, basis in basis_map.items():
        panels = {
            corpus: collect_coordinates(model, spec, basis, semantic_config)
            for corpus, spec in corpora.items()
        }
        fine = panels["fineweb"]
        mean = fine["coordinates"].mean(0)
        std = fine["coordinates"].std(0).clip(1e-5)
        normalized_fine = (fine["coordinates"] - mean) / std
        for target_family in family_names:
            pole = pole_from_label(frozen_labels[target_family])
            seed_mask = np.isin(fine["token_ids"], seed_ids[target_family][pole])
            mean_positive = normalized_fine[seed_mask].mean(0)
            mean_negative = normalized_fine[~seed_mask].mean(0)
            probe = mean_positive - mean_negative
            for corpus, panel in panels.items():
                normalized = (panel["coordinates"] - mean) / std
                for offset in config["evaluation_offsets"]:
                    token_ids, valid = prior_ids(panel, corpora[corpus], offset)
                    selected_unseen = unseen_ids[target_family][pole]
                    labels = np.isin(token_ids[valid], selected_unseen)
                    for rank in config["ranks"]:
                        scores = normalized[valid, :rank] @ probe[:rank]
                        result = sampled_auc(
                            labels, scores,
                            config["maximum_negative_examples"],
                            config["minimum_positive_count"],
                            config["seed"],
                            f"{basis_name}/{target_family}/{corpus}/{offset}/{rank}",
                        )
                        all_records.append({
                            "basis": basis_name,
                            "target_family": target_family,
                            "corpus": corpus,
                            "offset": offset,
                            "rank": rank,
                            "pole": pole,
                            **result,
                        })
            print("probe", basis_name, target_family, flush=True)
        del panels, normalized_fine

    def value(basis: str, target: str, corpus: str, rank: int, offset: int = 0) -> float:
        return next(
            row["auc"] for row in all_records
            if row["basis"] == basis and row["target_family"] == target
            and row["corpus"] == corpus and row["rank"] == rank
            and row["offset"] == offset
        )

    ood = ("tinystories", "wikitext")
    correct = {
        corpus: {
            rank: [value(family, family, corpus, rank) for family in family_names]
            for rank in config["ranks"]
        }
        for corpus in corpora
    }
    random_values = {
        corpus: {
            rank: [value("random", family, corpus, rank) for family in family_names]
            for rank in config["ranks"]
        }
        for corpus in corpora
    }
    best_wrong = {
        corpus: {
            rank: {
                family: max(
                    value(wrong, family, corpus, rank)
                    for wrong in family_names if wrong != family
                )
                for family in family_names
            }
            for rank in config["ranks"]
        }
        for corpus in corpora
    }
    semantic_rank80 = {
        corpus: {
            family: rank_80({
                rank: value(family, family, corpus, rank) for rank in config["ranks"]
            })
            for family in family_names
        }
        for corpus in corpora
    }

    hypotheses = {
        "R1_scale_exists_in_high_rank_family_subspace": {
            "supported": all(
                value("increasing_scale", "increasing_scale", corpus, 256) >= 0.70
                for corpus in ood
            ),
            "values": {
                corpus: value("increasing_scale", "increasing_scale", corpus, 256)
                for corpus in ood
            },
        },
        "R2_scale_subspace_beats_random": {
            "supported": all(
                value("increasing_scale", "increasing_scale", corpus, 256)
                - value("random", "increasing_scale", corpus, 256) >= 0.05
                for corpus in ood
            ),
            "gaps": {
                corpus: value("increasing_scale", "increasing_scale", corpus, 256)
                - value("random", "increasing_scale", corpus, 256)
                for corpus in ood
            },
        },
        "R3_scale_subspace_is_family_specific": {
            "supported": all(
                value("increasing_scale", "increasing_scale", corpus, 256)
                - best_wrong[corpus][256]["increasing_scale"] >= 0.05
                for corpus in ood
            ),
            "gaps": {
                corpus: value("increasing_scale", "increasing_scale", corpus, 256)
                - best_wrong[corpus][256]["increasing_scale"]
                for corpus in ood
            },
        },
        "R4_correct_family_subspaces_beat_random": {
            "supported": all(
                finite_median(correct[corpus][64])
                - finite_median(random_values[corpus][64]) >= 0.05
                for corpus in ood
            ),
            "median_gaps": {
                corpus: finite_median(correct[corpus][64])
                - finite_median(random_values[corpus][64]) for corpus in ood
            },
        },
        "R5_correct_family_subspaces_are_specific": {
            "supported": all(
                sum(
                    value(family, family, corpus, 64)
                    > best_wrong[corpus][64][family]
                    for family in family_names
                ) >= 4
                for corpus in ood
            ),
            "counts_beating_best_wrong": {
                corpus: sum(
                    value(family, family, corpus, 64)
                    > best_wrong[corpus][64][family]
                    for family in family_names
                ) for corpus in ood
            },
        },
        "R6_scale_context_persists_in_high_rank_code": {
            "supported": all(
                value("increasing_scale", "increasing_scale", corpus, 256, 1) >= 0.60
                for corpus in ood
            ),
            "values": {
                corpus: value("increasing_scale", "increasing_scale", corpus, 256, 1)
                for corpus in ood
            },
        },
        "R7_semantic_rank_is_compact": {
            "supported": all(
                any(
                    semantic_rank80[corpus][family] is not None
                    and semantic_rank80[corpus][family] <= 64
                    for corpus in ood
                )
                for family in family_names
            ),
            "rank_80": semantic_rank80,
        },
        "R8_high_rank_random_projection_does_not_close_gap": {
            "supported": all(
                finite_median(correct[corpus][256])
                > finite_median(random_values[corpus][256])
                for corpus in ood
            ),
            "median_gaps": {
                corpus: finite_median(correct[corpus][256])
                - finite_median(random_values[corpus][256]) for corpus in ood
            },
        },
    }

    summary = []
    for family in family_names:
        summary.append({
            "family": family,
            "rank80_tinystories": semantic_rank80["tinystories"][family],
            "rank80_wikitext": semantic_rank80["wikitext"][family],
            "tinystories_correct_64": value(family, family, "tinystories", 64),
            "tinystories_correct_256": value(family, family, "tinystories", 256),
            "tinystories_random_256": value("random", family, "tinystories", 256),
            "tinystories_wrong_256": best_wrong["tinystories"][256][family],
            "wikitext_correct_64": value(family, family, "wikitext", 64),
            "wikitext_correct_256": value(family, family, "wikitext", 256),
            "wikitext_random_256": value("random", family, "wikitext", 256),
            "wikitext_wrong_256": best_wrong["wikitext"][256][family],
            "tinystories_offset1_256": value(family, family, "tinystories", 256, 1),
            "wikitext_offset1_256": value(family, family, "wikitext", 256, 1),
        })

    result = {
        "config": config,
        "hypotheses": hypotheses,
        "summary": summary,
        "records": all_records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for corpus, linestyle in (("tinystories", "-"), ("wikitext", "--")):
        axes[0].plot(
            config["ranks"],
            [value("increasing_scale", "increasing_scale", corpus, rank) for rank in config["ranks"]],
            marker="o", linestyle=linestyle, label=f"correct {corpus}",
        )
        axes[0].plot(
            config["ranks"],
            [value("random", "increasing_scale", corpus, rank) for rank in config["ranks"]],
            marker=".", linestyle=linestyle, alpha=.65, label=f"random {corpus}",
        )
    axes[0].set_xscale("log", base=2)
    axes[0].axhline(0.5, color="black", linewidth=1)
    axes[0].set_xlabel("Input subspace rank")
    axes[0].set_ylabel("Unseen increasing-scale AUROC")
    axes[0].set_title("Does high rank recover the failed family?")
    axes[0].legend(fontsize=7)
    x = np.arange(len(family_names))
    width = .25
    axes[1].bar(x - width, correct["wikitext"][64], width, label="correct family")
    axes[1].bar(
        x, [best_wrong["wikitext"][64][family] for family in family_names],
        width, label="best wrong family",
    )
    axes[1].bar(x + width, random_values["wikitext"][64], width, label="random")
    axes[1].axhline(0.5, color="black", linewidth=1)
    axes[1].set_xticks(x, family_names, rotation=25, ha="right")
    axes[1].set_ylabel("WikiText unseen-word AUROC at rank 64")
    axes[1].set_title("Subspace specificity")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Rank of semantic information in Layer-3 family input subspaces",
        "",
        "For each weight-defined family input eigenspace, a rotation-invariant linear mean-contrast "
        "probe is fit on only the original seed-token occurrences in FineWeb. Nested ranks are then "
        "evaluated on the previously frozen unseen lexicons in TinyStories and WikiText-103. Other "
        "family eigenspaces and an isotropic random subspace are controls. AUROC uses every positive "
        "and a fixed 20,000-negative sample per locked key.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Semantic subspace rank](layer3_semantic_subspace_rank.png)",
        "",
        "| Target family | K80 Tiny/Wiki | Tiny correct 64/256 | Tiny wrong/random 256 | Wiki correct 64/256 | Wiki wrong/random 256 | +1 Tiny/Wiki 256 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        kt = row["rank80_tinystories"] if row["rank80_tinystories"] is not None else "∞"
        kw = row["rank80_wikitext"] if row["rank80_wikitext"] is not None else "∞"
        lines.append(
            f"| {row['family']} | {kt}/{kw} | "
            f"{row['tinystories_correct_64']:.3f}/{row['tinystories_correct_256']:.3f} | "
            f"{row['tinystories_wrong_256']:.3f}/{row['tinystories_random_256']:.3f} | "
            f"{row['wikitext_correct_64']:.3f}/{row['wikitext_correct_256']:.3f} | "
            f"{row['wikitext_wrong_256']:.3f}/{row['wikitext_random_256']:.3f} | "
            f"{row['tinystories_offset1_256']:.3f}/{row['wikitext_offset1_256']:.3f} |"
        )
    lines += [
        "",
        "A successful correct-family curve would show that a failed leading axis hid a distributed "
        "semantic code. Matching wrong-family or random curves instead means high-rank linear "
        "decodability is generic residual-stream information, not evidence that the selected family "
        "tensor module owns that concept. This distinction is essential because sufficiently wide "
        "random projections preserve many lexical distinctions.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
