#!/usr/bin/env python3
"""Finite unseen-word causal validation of the frozen Layer-3 quotient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer3_family_interaction_modes import evaluate_writes
from run_bilin18_layer3_frozen_channel_transfer import channel_state, recover_configs, top_indices
from run_bilin18_layer3_native_bank_input_forms import bank_delta, input_metric
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_layer8_semantic_attention_causal import median
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


DEFAULT_CONFIG = Path("configs/bilin18_layer3_unseen_causal_quotient.json")


def encode_single(tokenizer, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if len(encoded) != 1:
        raise ValueError(f"lexicon item is not a single token: {text!r} -> {encoded}")
    return encoded[0]


def select_panel(
    source: np.ndarray,
    lexicon: dict,
    tokenizer,
    config: dict,
    seed_ids: dict[str, set[int]],
) -> list[dict]:
    records = []
    edit = config["edit_position"]
    maximum_position = source.shape[1] - (config["sequence_length"] - edit)
    for family, poles in lexicon.items():
        source_ids = [encode_single(tokenizer, text) for text in poles["source"]]
        target_ids = [encode_single(tokenizer, text) for text in poles["target"]]
        overlap = (set(source_ids) | set(target_ids)) & seed_ids[family]
        if overlap:
            raise ValueError(f"unseen lexicon overlap with seed_ids for {family}: {overlap}")
        for source_id, target_id in zip(source_ids, target_ids):
            positions = np.argwhere(source[:, edit:maximum_position + 1] == source_id)
            for row, relative in positions[: config["contexts_per_pair"]]:
                position = int(relative + edit)
                records.append({
                    "family": family,
                    "source_id": source_id,
                    "target_id": target_id,
                    "row": int(row),
                    "source_start": position - edit,
                })
    return records


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text())
    output_stem = config.get("output_stem", "layer3_unseen_causal_quotient")
    output = TENSOR_ROOT / f"{output_stem}.json"
    report = TENSOR_ROOT / f"{output_stem.upper()}.md"
    figure_path = TENSOR_ROOT / f"{output_stem}.png"
    quotient_config = json.loads(Path(config["source_config"]).read_text())
    quotient_source = json.loads(Path(config["source_results"]).read_text())
    unseen_config = json.loads(Path(config["unseen_config"]).read_text())
    seed_config = json.loads(Path(unseen_config["seed_config"]).read_text())
    semantic_config = json.loads(Path(quotient_config["source_config"]).read_text())
    input_config = json.loads(Path(semantic_config["source_config"]).read_text())
    specificity_config = json.loads(Path(input_config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    family_names = list(parent["families"])
    native_saved = json.loads(Path(config["native_results"]).read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    seed_ids = {
        family: {token for pair in seed_config["families"][family] for token in pair}
        for family in family_names
    }
    corpora = {
        corpus: semantic_config["corpora"][corpus]
        for corpus in ("fineweb", "tinystories")
    }
    arrays = {
        corpus: np.asarray(
            np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]],
            dtype=np.int64,
        )
        for corpus, spec in corpora.items()
    }
    panels = {
        corpus: select_panel(
            rows, unseen_config["lexicon"], tokenizer, config, seed_ids,
        )
        for corpus, rows in arrays.items()
    }
    for corpus, records in panels.items():
        for record in records:
            record["corpus"] = corpus

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    layer, final_position = semantic_config["layer"], config["causal_position"]
    block = model.blocks[layer]

    def reconstruct(record):
        raw = arrays[record["corpus"]][
            record["row"],
            record["source_start"]:record["source_start"] + config["sequence_length"],
        ].copy()
        if int(raw[config["edit_position"]]) != record["source_id"]:
            raise AssertionError("unseen panel edit position misaligned")
        changed = raw.copy()
        changed[config["edit_position"]] = record["target_id"]
        base = forward_layers(
            model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer,
        )[layer]
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )[layer]
        return base, counterfactual

    discovery = [
        record for record in native_saved["semantic_records"]
        if record["corpus"] == frozen_config["discovery_corpus"]
        and record["context_index"] == frozen_config["discovery_context_index"]
    ]
    discovery_arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")
        for corpus, spec in fixed["evaluation"].items()
    }

    def reconstruct_discovery(record):
        spec = fixed["evaluation"][record["corpus"]]
        raw = np.asarray(
            discovery_arrays[record["corpus"]][
                spec["start"] + record["row"],
                record["source_start"]:record["source_start"] + fixed["sequence_length"],
            ], dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[fixed["edit_position"]] = int(record["pair"].split("->")[1])
        base = forward_layers(
            model, torch.as_tensor(raw, device="cuda")[None], last_layer=layer,
        )[layer]
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=layer,
        )[layer]
        return base, counterfactual

    family_scores = {family: [] for family in family_names}
    for record in discovery:
        base, counterfactual = reconstruct_discovery(record)
        family_scores[record["family"]].append(channel_state(
            model, layer, base, counterfactual, fixed["causal_position"],
        )["scores"].abs().cpu())

    saved = np.load(config["source_arrays"])
    common = torch.from_numpy(saved["common"]).cuda()
    family_cache = {}
    for family in family_names:
        indices = top_indices(
            torch.stack(family_scores[family]).mean(0), quotient_config["channel_rank"],
        ).cuda()
        metric = input_metric(block, indices, quotient_config["output_rank"])
        family_cache[family] = {
            "indices": indices,
            "output": metric["output"],
            "basis": torch.from_numpy(saved[f"{family}_basis"]).cuda(),
            "shared_like": torch.from_numpy(saved[f"{family}_shared_like"]).cuda(),
            "family_residual": torch.from_numpy(saved[f"{family}_family_residual"]).cuda(),
        }

    evaluated = []
    for corpus, records in panels.items():
        for record in records:
            base, counterfactual = reconstruct(record)
            state = channel_state(model, layer, base, counterfactual, final_position)
            family = record["family"]
            cache = family_cache[family]
            dimension = base["pre"].shape[-1]
            xb = F.rms_norm(base["pre"], (dimension,)).float()
            xc = F.rms_norm(counterfactual["pre"], (dimension,)).float()
            delta = xc - xb

            def route(basis):
                changed = xb + (delta @ basis) @ basis.T
                predicted = bank_delta(block, xb, changed, cache["indices"])
                return (predicted @ cache["output"]) @ cache["output"].T

            union = route(cache["basis"])
            shared_like = route(cache["shared_like"])
            family_residual = route(cache["family_residual"])
            cross_term = union - shared_like - family_residual
            writes = {
                "full_mlp": state["target_write"],
                "family_exact": union,
                "shared_like8": shared_like,
                "family_residual24": family_residual,
                "cross_term": cross_term,
                "global_common8": route(common),
            }
            wrong_names = []
            for wrong in family_names:
                if wrong == family:
                    continue
                name = f"wrong_residual:{wrong}"
                writes[name] = route(family_cache[wrong]["family_residual"])
                wrong_names.append(name)
            effects = evaluate_writes(
                model, layer, base, counterfactual, writes, final_position,
            )
            strongest_wrong = max(
                wrong_names, key=lambda name: effects[name]["family_patch_fidelity"],
            )
            evaluated.append({
                **record,
                "effects": effects,
                "strongest_wrong": strongest_wrong.split(":", 1)[1],
                "strongest_wrong_fidelity": effects[strongest_wrong]["family_patch_fidelity"],
                "cross_norm_share": float(
                    cross_term.norm() / union.norm().clamp_min(1e-30)
                ),
            })
            print(
                "unseen causal", corpus, family,
                f"{record['source_id']}->{record['target_id']}", flush=True,
            )

    summary = {}
    for corpus in corpora:
        summary[corpus] = {}
        for family in family_names:
            selected = [
                row for row in evaluated
                if row["corpus"] == corpus and row["family"] == family
            ]
            summary[corpus][family] = {
                "count": len(selected),
                "unique_source_tokens": len({row["source_id"] for row in selected}),
                "shared_like": median([
                    row["effects"]["shared_like8"]["family_patch_fidelity"] for row in selected
                ]),
                "family_residual": median([
                    row["effects"]["family_residual24"]["family_patch_fidelity"] for row in selected
                ]),
                "global_common": median([
                    row["effects"]["global_common8"]["family_patch_fidelity"] for row in selected
                ]),
                "strongest_wrong": median([
                    row["strongest_wrong_fidelity"] for row in selected
                ]),
                "cross_norm_share": median([
                    row["cross_norm_share"] for row in selected
                ]),
                "union_semantic_projection": median([
                    row["effects"]["family_exact"]["semantic_target_projection"] for row in selected
                ]),
            }
        selected = [row for row in evaluated if row["corpus"] == corpus]
        summary[corpus]["aggregate"] = {
            key: median([
                row["cross_norm_share"] if key == "cross_norm_share"
                else row["effects"][key]["family_patch_fidelity"]
                for row in selected
            ])
            for key in ("shared_like8", "family_residual24", "global_common8", "cross_norm_share")
        }

    seed_regime = {}
    for corpus in corpora:
        seed_regime[corpus] = {}
        for family in family_names:
            selected = [
                row for row in quotient_source["causal_records"]
                if row["corpus"] == corpus and row["family"] == family
            ]
            seed_regime[corpus][family] = median([
                row["effects"]["shared_like8"]["family_patch_fidelity"]
                - row["effects"]["family_residual24"]["family_patch_fidelity"]
                for row in selected
            ])

    unseen_regime = {
        corpus: {
            family: summary[corpus][family]["shared_like"]
            - summary[corpus][family]["family_residual"]
            for family in family_names
        }
        for corpus in corpora
    }
    def has_coverage(corpus: str, family: str) -> bool:
        return summary[corpus][family]["count"] >= config["minimum_records_per_family"]

    hypotheses = {
        "U1_unseen_causal_panel_has_coverage": {
            "supported": all(
                has_coverage(corpus, family)
                for corpus in corpora for family in family_names
            ),
            "counts": {
                f"{corpus}/{family}": summary[corpus][family]["count"]
                for corpus in corpora for family in family_names
            },
        },
        "U2_sentiment_and_polarity_remain_shared_dominant": {
            "supported": all(
                has_coverage(corpus, family) and unseen_regime[corpus][family] > 0
                for corpus in corpora for family in ("sentiment", "polarity")
            ),
        },
        "U3_gender_and_tense_remain_residual_dominant": {
            "supported": all(
                has_coverage(corpus, family) and -unseen_regime[corpus][family] >= 0.20
                for corpus in corpora for family in ("gender", "tense")
            ),
        },
        "U4_scale_remains_a_mixed_causal_regime": {
            "supported": all(
                has_coverage(corpus, "increasing_scale")
                and summary[corpus]["increasing_scale"][route] >= 0.20
                for corpus in corpora for route in ("shared_like", "family_residual")
            ),
        },
        "U5_unseen_residuals_are_family_specific": {
            "supported": all(
                has_coverage(corpus, family)
                and summary[corpus][family]["family_residual"]
                - summary[corpus][family]["strongest_wrong"] >= 0.10
                for corpus in corpora
                for family in ("gender", "tense", "increasing_scale")
            ),
            "gaps": {
                f"{corpus}/{family}": summary[corpus][family]["family_residual"]
                - summary[corpus][family]["strongest_wrong"]
                for corpus in corpora
                for family in ("gender", "tense", "increasing_scale")
            },
        },
        "U6_seed_and_unseen_regime_labels_agree": {
            "supported": all(
                has_coverage(corpus, family)
                and np.sign(seed_regime[corpus][family]) == np.sign(unseen_regime[corpus][family])
                for corpus in corpora
                for family in ("sentiment", "gender", "tense", "polarity")
            ),
            "seed": seed_regime,
            "unseen": unseen_regime,
        },
        "U7_unseen_cross_term_is_nontrivial": {
            "supported": all(
                all(has_coverage(corpus, family) for family in family_names)
                and summary[corpus]["aggregate"]["cross_norm_share"] >= 0.05
                for corpus in corpora
            ),
            "values": {
                corpus: summary[corpus]["aggregate"]["cross_norm_share"]
                for corpus in corpora
            },
        },
        "U8_unseen_regimes_transfer_ood": {
            "supported": all(
                has_coverage("fineweb", family)
                and has_coverage("tinystories", family)
                and np.sign(unseen_regime["fineweb"][family])
                == np.sign(unseen_regime["tinystories"][family])
                for family in family_names
            ),
            "values": unseen_regime,
        },
    }

    output.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summary": summary,
        "records": evaluated,
    }, indent=2))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(family_names))
    width = .18
    for index, (corpus, route) in enumerate((
        ("fineweb", "shared_like"), ("fineweb", "family_residual"),
        ("tinystories", "shared_like"), ("tinystories", "family_residual"),
    )):
        axes[0].bar(
            x + (index - 1.5) * width,
            [summary[corpus][family][route] for family in family_names],
            width, label=f"{corpus} {route}",
        )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x, family_names, rotation=25, ha="right")
    axes[0].set_ylabel("Fidelity to frozen family top-32 patch")
    axes[0].set_title("Unseen-word causal regimes")
    axes[0].legend(fontsize=7)
    width = .35
    axes[1].bar(
        x - width / 2,
        [unseen_regime["fineweb"][family] for family in family_names],
        width, label="FineWeb",
    )
    axes[1].bar(
        x + width / 2,
        [unseen_regime["tinystories"][family] for family in family_names],
        width, label="TinyStories",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xticks(x, family_names, rotation=25, ha="right")
    axes[1].set_ylabel("Shared-like minus residual fidelity")
    axes[1].set_title("Regime sign transfer")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    lines = [
        "# Unseen-word causal validation of the Layer-3 quotient",
        "",
        "Ten manually frozen source/target word pairs per family are disjoint from every seed pair "
        "used to construct the channel banks and input metrics. Natural occurrences are placed at "
        "the same edit and causal positions as the original test, the target token is substituted, "
        "and the model suffix is recomputed. The rank-eight shared/residual quotient is never refit.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        f"![Unseen causal quotient]({figure_path.name})",
        "",
        "| Corpus / family | N | Unique source tokens | Shared-like | Residual | Shared−residual | Best wrong residual | Cross norm | Semantic projection |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in corpora:
        for family in family_names:
            row = summary[corpus][family]
            lines.append(
                f"| {corpus} / {family} | {row['count']} | {row['unique_source_tokens']} | "
                f"{row['shared_like']:.3f} | "
                f"{row['family_residual']:.3f} | {unseen_regime[corpus][family]:.3f} | "
                f"{row['strongest_wrong']:.3f} | {row['cross_norm_share']:.3f} | "
                f"{row['union_semantic_projection']:.3f} |"
            )
    lines += [
        "",
        "Fidelity is relative to the frozen family's own top-32 causal patch, not to the entire MLP. "
        "This asks whether the internal regime transfers to new lexical realizations. Semantic "
        "projection is reported separately because an internally faithful family patch can still be "
        "a small or opposing contributor to the complete final-logit counterfactual.",
        "",
        "Repeated contexts for the same source token improve context coverage but do not improve lexical "
        "coverage. Any sensitivity run with few unique source tokens must therefore be read as a context-"
        "robustness test, not as independent evidence for family-wide lexical generalization.",
    ]
    report.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(report)


if __name__ == "__main__":
    main()
