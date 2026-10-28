#!/usr/bin/env python3
"""Prospectively label weight-defined Layer-3 family input modes on natural text.

The native modes are frozen from family-specific bilinear weights and semantic
counterfactual channel discovery.  Natural FineWeb coordinates are used only to
choose a small token predicate, signal type, orientation, and one mode per
family.  Those choices are then evaluated unchanged on TinyStories.  Isotropic
random directions, rotations of the same family subspace, wrong-family token
predicates, and generic tokenizer predicates are explicit controls.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER, target_layer_inputs
from discover_bilin18_weight_only_circuits import CATEGORIES, auc, token_flags
from run_bilin18_layer3_frozen_channel_transfer import (
    channel_state,
    recover_configs,
    top_indices,
)
from run_bilin18_layer3_native_bank_input_forms import input_metric
from run_bilin18_layer8_semantic_attention_atlas import forward_layers
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer3_input_mode_semantics.json")
OUTPUT = TENSOR_ROOT / "layer3_input_mode_semantics.json"
ARRAYS = TENSOR_ROOT / "layer3_input_mode_semantics.npz"
REPORT = TENSOR_ROOT / "LAYER3_INPUT_MODE_SEMANTICS.md"
FIGURE = TENSOR_ROOT / "layer3_input_mode_semantics.png"


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.sqrt(np.square(first).sum() * np.square(second).sum())
    return float((first * second).sum() / max(denominator, 1e-30))


def oriented_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, int]:
    value = auc(labels, scores)
    if not np.isfinite(value):
        return float("nan"), 1
    return (float(value), 1) if value >= 0.5 else (float(1 - value), -1)


def score_ranks(scores: np.ndarray) -> np.ndarray:
    """Tie-aware one-indexed ranks, computed once for many binary predicates."""
    scores = np.asarray(scores)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        ranks[order[start:stop]] = (start + stop + 1) / 2
    return ranks


def auc_from_ranks(labels: np.ndarray, ranks: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        return float("nan")
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def fit_label(
    coordinates: np.ndarray,
    masks: dict[str, np.ndarray],
    candidates: list[str],
    minimum_positive_count: int,
    rank_cache: dict[str, np.ndarray] | None = None,
) -> dict:
    """Select predicate, signed/magnitude signal, and orientation on discovery."""
    best = None
    if rank_cache is None:
        rank_cache = {
            "signed": score_ranks(coordinates),
            "magnitude": score_ranks(np.abs(coordinates)),
        }
    for name in candidates:
        labels = masks[name]
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if min(positives, negatives) < minimum_positive_count:
            continue
        for signal in ("signed", "magnitude"):
            raw_value = auc_from_ranks(labels, rank_cache[signal])
            value, orientation = (
                (float(raw_value), 1)
                if raw_value >= 0.5 else (float(1 - raw_value), -1)
            )
            candidate = {
                "predicate": name,
                "signal": signal,
                "orientation": orientation,
                "discovery_auc": value,
                "discovery_positive_count": positives,
                "discovery_negative_count": negatives,
                "selected_on": "fineweb",
            }
            if best is None or candidate["discovery_auc"] > best["discovery_auc"]:
                best = candidate
    if best is None:
        return {
            "predicate": None,
            "signal": None,
            "orientation": 1,
            "discovery_auc": float("nan"),
            "discovery_positive_count": 0,
            "discovery_negative_count": 0,
            "selected_on": "fineweb",
        }
    return best


def evaluate_label(
    label: dict,
    coordinates: np.ndarray,
    masks: dict[str, np.ndarray],
    rank_cache: dict[str, np.ndarray] | None = None,
) -> dict:
    if label["predicate"] is None:
        return {"evaluation_auc": float("nan"), "positive_count": 0, "negative_count": 0}
    labels = masks[label["predicate"]]
    if rank_cache is None:
        raw = coordinates if label["signal"] == "signed" else np.abs(coordinates)
        value = auc(labels, label["orientation"] * raw)
    else:
        value = auc_from_ranks(labels, rank_cache[label["signal"]])
        if label["orientation"] < 0:
            value = 1 - value
    return {
        "evaluation_auc": float(value),
        "positive_count": int(labels.sum()),
        "negative_count": int(len(labels) - labels.sum()),
    }


def predicate_masks(
    token_ids: np.ndarray,
    families: dict[str, list[list[int]]],
    flag_lookup: np.ndarray,
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for family, pairs in families.items():
        source = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        target = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
        masks[f"family:{family}:source"] = np.isin(token_ids, source)
        masks[f"family:{family}:target"] = np.isin(token_ids, target)
        masks[f"family:{family}:any"] = np.isin(token_ids, np.concatenate((source, target)))
    for column, name in enumerate(CATEGORIES):
        masks[f"generic:{name}"] = flag_lookup[token_ids, column].astype(bool)
    return masks


@torch.inference_mode()
def collect_coordinates(
    model,
    corpus: dict,
    directions: torch.Tensor,
    config: dict,
) -> dict[str, np.ndarray]:
    source = np.load(corpus["path"], mmap_mode="r")
    rows = source[corpus["start"]: corpus["start"] + corpus["rows"]]
    positions = np.arange(config["position_start"], config["position_stop"])
    total = len(rows) * len(positions)
    coordinates = np.empty((total, directions.shape[1]), dtype=np.float32)
    token_ids = np.empty(total, dtype=np.int64)
    row_ids = np.empty(total, dtype=np.int32)
    position_ids = np.empty(total, dtype=np.int16)
    offset = 0
    blocks = model.blocks[: config["layer"] + 1]
    for start in range(0, len(rows), config["batch_size"]):
        stop = min(start + config["batch_size"], len(rows))
        raw = np.asarray(rows[start:stop, : config["sequence_length"]], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        normalized = target_layer_inputs(
            model.embedding, blocks, ids, model.config, model.tt,
        )
        selected = normalized[:, positions].float().flatten(0, 1)
        values = selected @ directions
        count = values.shape[0]
        coordinates[offset:offset + count] = values.cpu().numpy()
        token_ids[offset:offset + count] = raw[:, positions].reshape(-1)
        row_ids[offset:offset + count] = np.repeat(
            np.arange(start, stop, dtype=np.int32), len(positions),
        )
        position_ids[offset:offset + count] = np.tile(positions, stop - start)
        offset += count
        if start == 0 or stop % 256 == 0 or stop == len(rows):
            print(f"natural coordinates {corpus['path']} {stop}/{len(rows)}", flush=True)
    return {
        "coordinates": coordinates,
        "token_ids": token_ids,
        "row_ids": row_ids,
        "position_ids": position_ids,
    }


def context_tail(
    tokenizer,
    corpus: dict,
    panel: dict[str, np.ndarray],
    values: np.ndarray,
    tail_size: int,
) -> dict[str, list[dict]]:
    source = np.load(corpus["path"], mmap_mode="r")
    rows = source[corpus["start"]: corpus["start"] + corpus["rows"]]

    def records(indices: np.ndarray) -> list[dict]:
        result = []
        for index in indices:
            row_index = int(panel["row_ids"][index])
            position = int(panel["position_ids"][index])
            raw = np.asarray(rows[row_index, :], dtype=np.int64)
            left = max(0, position - 10)
            right = min(len(raw), position + 6)
            result.append({
                "score": float(values[index]),
                "token_id": int(raw[position]),
                "token": tokenizer.decode([int(raw[position])]),
                "before": tokenizer.decode(raw[left:position].tolist()),
                "after": tokenizer.decode(raw[position + 1:right].tolist()),
                "row": row_index,
                "position": position,
            })
        return result

    order = np.argsort(values)
    return {
        "highest": records(order[-tail_size:][::-1]),
        "lowest": records(order[:tail_size]),
    }


def token_mean_correlation(
    coordinates: np.ndarray,
    token_ids: np.ndarray,
    embedding_scores: np.ndarray,
    vocabulary_size: int,
    minimum_count: int,
) -> np.ndarray:
    count = np.bincount(token_ids, minlength=vocabulary_size)[:vocabulary_size]
    sums = np.zeros((vocabulary_size, coordinates.shape[1]), dtype=np.float64)
    np.add.at(sums, token_ids, coordinates)
    valid = count >= minimum_count
    means = sums[valid] / count[valid, None]
    return np.asarray([
        correlation(means[:, column], embedding_scores[:vocabulary_size][valid, column])
        for column in range(coordinates.shape[1])
    ])


def finite_median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")


def format_metric(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "n/a"


@torch.inference_mode()
def main() -> None:
    config = json.loads(CONFIG.read_text())
    input_config = json.loads(Path(config["source_config"]).read_text())
    specificity_config = json.loads(Path(input_config["source_config"]).read_text())
    bank_config = json.loads(Path(specificity_config["source_config"]).read_text())
    rank_config = json.loads(Path(bank_config["source_config"]).read_text())
    form_config = json.loads(Path(rank_config["source_config"]).read_text())
    lopo_config = json.loads(Path(form_config["source_config"]).read_text())
    frozen_config = json.loads(Path(lopo_config["source_config"]).read_text())
    fixed, parent = recover_configs(frozen_config)
    families = parent["families"]
    family_names = list(families)
    native_saved = json.loads(Path(config["native_results"]).read_text())

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    flag_lookup = np.stack([token_flags(text) for text in texts])
    model = load_full_model("cuda")
    block = model.blocks[config["layer"]]

    # Recover the inclusive channel bank only from the previously frozen
    # FineWeb semantic-edit discovery panel; natural labeling data is untouched.
    semantic_rows = [
        row for row in native_saved["semantic_records"]
        if row["corpus"] == frozen_config["discovery_corpus"]
        and row["context_index"] == frozen_config["discovery_context_index"]
    ]
    fixed_array = np.load(
        fixed["evaluation"][frozen_config["discovery_corpus"]]["path"], mmap_mode="r",
    )
    family_scores = {family: [] for family in family_names}
    for row in semantic_rows:
        raw = np.asarray(
            fixed_array[
                row["row"],
                row["source_start"]:row["source_start"] + fixed["sequence_length"],
            ],
            dtype=np.int64,
        ).copy()
        changed = raw.copy()
        changed[fixed["edit_position"]] = int(row["pair"].split("->")[1])
        base = forward_layers(
            model, torch.as_tensor(raw, device="cuda")[None], last_layer=config["layer"],
        )[config["layer"]]
        counterfactual = forward_layers(
            model, torch.as_tensor(changed, device="cuda")[None], last_layer=config["layer"],
        )[config["layer"]]
        score = channel_state(
            model, config["layer"], base, counterfactual, fixed["causal_position"],
        )["scores"].abs().cpu()
        family_scores[row["family"]].append(score)
        print("frozen bank", row["family"], row["pair"], flush=True)

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    native, random, rotated, eigenvalues = {}, {}, {}, {}
    for family in family_names:
        indices = top_indices(
            torch.stack(family_scores[family]).mean(0), config["channel_rank"],
        ).cuda()
        metric = input_metric(block, indices, config["output_rank"])
        native[family] = metric["vectors"][:, : config["modes_per_family"]]
        eigenvalues[family] = metric["values"][: config["modes_per_family"]].cpu().numpy()
        random[family] = torch.linalg.qr(torch.randn(
            model.config.n_embd,
            config["modes_per_family"],
            generator=generator,
            device="cuda",
        ), mode="reduced").Q
        rotation = torch.linalg.qr(torch.randn(
            config["modes_per_family"],
            config["modes_per_family"],
            generator=generator,
            device="cuda",
        )).Q
        rotated[family] = native[family] @ rotation
        print("frozen metric", family, flush=True)

    basis_order = ("native", "random", "rotated")
    basis_map = {"native": native, "random": random, "rotated": rotated}
    direction_columns = []
    column_metadata = []
    for basis in basis_order:
        for family in family_names:
            direction_columns.append(basis_map[basis][family])
            for mode in range(config["modes_per_family"]):
                column_metadata.append({"basis": basis, "family": family, "mode": mode})
    directions = torch.cat(direction_columns, dim=1)

    panels = {
        corpus: collect_coordinates(model, spec, directions, config)
        for corpus, spec in config["corpora"].items()
    }
    masks = {
        corpus: predicate_masks(panel["token_ids"], families, flag_lookup)
        for corpus, panel in panels.items()
    }
    discovery_name = config["discovery_corpus"]
    evaluation_name = config["evaluation_corpus"]

    embedding = F.normalize(model.embedding.weight.float(), dim=1)
    # The checkpoint has 47 padding rows beyond the GPT-2 tokenizer.  They are
    # valid model parameters but not decodable vocabulary items, so exclude
    # them from all semantic ranking and correlation tests.
    embedding_scores = (embedding @ directions).cpu().numpy()[: tokenizer.vocab_size]
    site_embedding_correlations = {}
    token_embedding_correlations = {}
    for corpus, panel in panels.items():
        site_embedding_correlations[corpus] = np.asarray([
            correlation(
                panel["coordinates"][:, column],
                embedding_scores[panel["token_ids"], column],
            )
            for column in range(directions.shape[1])
        ])
        token_embedding_correlations[corpus] = token_mean_correlation(
            panel["coordinates"], panel["token_ids"], embedding_scores,
            tokenizer.vocab_size, config["minimum_token_count_for_correlation"],
        )

    generic_candidates = [f"generic:{name}" for name in CATEGORIES]
    records = []
    for column, metadata in enumerate(column_metadata):
        family = metadata["family"]
        own_candidates = [
            f"family:{family}:source", f"family:{family}:target", f"family:{family}:any",
        ]
        wrong_candidates = [
            f"family:{other}:{kind}"
            for other in family_names if other != family
            for kind in ("source", "target", "any")
        ]
        discovery_coordinates = panels[discovery_name]["coordinates"][:, column]
        evaluation_coordinates = panels[evaluation_name]["coordinates"][:, column]
        discovery_ranks = {
            "signed": score_ranks(discovery_coordinates),
            "magnitude": score_ranks(np.abs(discovery_coordinates)),
        }
        evaluation_ranks = {
            "signed": score_ranks(evaluation_coordinates),
            "magnitude": score_ranks(np.abs(evaluation_coordinates)),
        }
        labels = {
            "own": fit_label(
                discovery_coordinates, masks[discovery_name], own_candidates,
                config["minimum_positive_count"], discovery_ranks,
            ),
            "wrong": fit_label(
                discovery_coordinates, masks[discovery_name], wrong_candidates,
                config["minimum_positive_count"], discovery_ranks,
            ),
            "generic": fit_label(
                discovery_coordinates, masks[discovery_name], generic_candidates,
                config["minimum_positive_count"], discovery_ranks,
            ),
        }
        for label in labels.values():
            label.update(evaluate_label(
                label,
                evaluation_coordinates, masks[evaluation_name], evaluation_ranks,
            ))

        scores = embedding_scores[:, column]
        high = np.argsort(scores)[-config["nearest_tokens"]:][::-1]
        low = np.argsort(scores)[:config["nearest_tokens"]]
        seed_ids = np.asarray(sorted({token for pair in families[family] for token in pair}))
        absolute_order = np.argsort(np.abs(scores))[::-1]
        ranks = np.empty(tokenizer.vocab_size, dtype=np.int64)
        ranks[absolute_order] = np.arange(1, tokenizer.vocab_size + 1)
        embedding_evidence = {
            "highest": [
                {"id": int(index), "text": texts[index], "score": float(scores[index])}
                for index in high
            ],
            "lowest": [
                {"id": int(index), "text": texts[index], "score": float(scores[index])}
                for index in low
            ],
            "best_family_seed_rank": int(ranks[seed_ids].min()),
            "family_seed_ranks": {
                str(int(index)): int(ranks[index]) for index in seed_ids
            },
        }
        tails = {}
        if metadata["basis"] == "native":
            for corpus, panel in panels.items():
                tails[corpus] = context_tail(
                    tokenizer, config["corpora"][corpus], panel,
                    panel["coordinates"][:, column], config["tail_size"],
                )
        records.append({
            **metadata,
            "eigenvalue": (
                float(eigenvalues[family][metadata["mode"]])
                if metadata["basis"] == "native" else None
            ),
            "labels": labels,
            "embedding": embedding_evidence,
            "site_embedding_correlation": {
                corpus: float(site_embedding_correlations[corpus][column]) for corpus in panels
            },
            "token_mean_embedding_correlation": {
                corpus: float(token_embedding_correlations[corpus][column]) for corpus in panels
            },
            "tails": tails,
        })
        print("label", metadata["basis"], family, metadata["mode"], flush=True)

    # Freeze the family representative by discovery AUROC within each basis,
    # then inspect its already-computed OOD score.  No TinyStories selection.
    representatives = {}
    for family in family_names:
        representatives[family] = {}
        for basis in basis_order:
            candidates = [
                record for record in records
                if record["family"] == family and record["basis"] == basis
            ]
            representatives[family][basis] = max(
                candidates, key=lambda record: record["labels"]["own"]["discovery_auc"],
            )

    def rep_auc(family: str, basis: str, label: str = "own") -> float:
        return representatives[family][basis]["labels"][label]["evaluation_auc"]

    native_ood = [rep_auc(family, "native") for family in family_names]
    random_ood = [rep_auc(family, "random") for family in family_names]
    rotated_ood = [rep_auc(family, "rotated") for family in family_names]
    wrong_ood = [rep_auc(family, "native", "wrong") for family in family_names]
    generic_ood = [rep_auc(family, "native", "generic") for family in family_names]
    native_records = [record for record in records if record["basis"] == "native"]
    retention = [
        (record["labels"]["own"]["evaluation_auc"] - 0.5)
        / max(record["labels"]["own"]["discovery_auc"] - 0.5, 1e-6)
        for record in native_records
        if record["labels"]["own"]["discovery_auc"] > 0.51
    ]
    median_site_correlations = {
        corpus: finite_median([
            abs(record["site_embedding_correlation"][corpus]) for record in native_records
        ])
        for corpus in panels
    }
    native_seed = [
        min(
            record["embedding"]["best_family_seed_rank"]
            for record in records
            if record["basis"] == "native" and record["family"] == family
        )
        for family in family_names
    ]
    random_seed = [
        min(
            record["embedding"]["best_family_seed_rank"]
            for record in records
            if record["basis"] == "random" and record["family"] == family
        )
        for family in family_names
    ]

    hypothesis_results = {
        "S1_every_family_has_semantic_mode": {
            "supported": all(value >= 0.70 for value in native_ood),
            "values": dict(zip(family_names, native_ood)),
        },
        "S2_semantic_modes_beat_random": {
            "supported": finite_median(native_ood) - finite_median(random_ood) >= 0.05,
            "native_median": finite_median(native_ood),
            "random_median": finite_median(random_ood),
        },
        "S3_family_labels_are_specific": {
            "supported": all(a - b >= 0.05 for a, b in zip(native_ood, wrong_ood)),
            "gaps": dict(zip(family_names, [a - b for a, b in zip(native_ood, wrong_ood)])),
        },
        "S4_weight_axes_predict_natural_coordinates": {
            "supported": all(value >= 0.30 for value in median_site_correlations.values()),
            "median_absolute_site_correlations": median_site_correlations,
        },
        "S5_labels_transfer_ood": {
            "supported": finite_median(retention) >= 0.75,
            "median_retained_fraction": finite_median(retention),
        },
        "S6_semantics_exceed_generic_token_form": {
            "supported": all(a > b for a, b in zip(native_ood, generic_ood)),
            "gaps": dict(zip(family_names, [a - b for a, b in zip(native_ood, generic_ood)])),
        },
        "S7_family_tokens_are_weight_neighbours": {
            "supported": all(rank <= 50 for rank in native_seed)
            and finite_median(random_seed) / max(finite_median(native_seed), 1) >= 2,
            "native_best_ranks": dict(zip(family_names, native_seed)),
            "random_best_ranks": dict(zip(family_names, random_seed)),
            "median_random_to_native_ratio": finite_median(random_seed)
            / max(finite_median(native_seed), 1),
        },
        "S8_native_axes_are_more_semantic_than_rotations": {
            "supported": finite_median(native_ood) - finite_median(rotated_ood) >= 0.05,
            "native_median": finite_median(native_ood),
            "rotated_median": finite_median(rotated_ood),
        },
    }

    summary_rows = []
    for family in family_names:
        representative = representatives[family]["native"]
        summary_rows.append({
            "family": family,
            "selected_native_mode": representative["mode"],
            "predicate": representative["labels"]["own"]["predicate"],
            "signal": representative["labels"]["own"]["signal"],
            "fineweb_auc": representative["labels"]["own"]["discovery_auc"],
            "tinystories_auc": representative["labels"]["own"]["evaluation_auc"],
            "wrong_auc": representative["labels"]["wrong"]["evaluation_auc"],
            "generic_auc": representative["labels"]["generic"]["evaluation_auc"],
            "random_auc": rep_auc(family, "random"),
            "rotated_auc": rep_auc(family, "rotated"),
            "fineweb_embedding_corr": representative["site_embedding_correlation"]["fineweb"],
            "tinystories_embedding_corr": representative["site_embedding_correlation"]["tinystories"],
            "best_seed_rank": representative["embedding"]["best_family_seed_rank"],
        })

    serializable_representatives = {
        family: {
            basis: {"mode": record["mode"], "labels": record["labels"]}
            for basis, record in choices.items()
        }
        for family, choices in representatives.items()
    }
    result = {
        "config": config,
        "panel_sizes": {corpus: len(panel["token_ids"]) for corpus, panel in panels.items()},
        "hypotheses": hypothesis_results,
        "summary": summary_rows,
        "representatives": serializable_representatives,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        ARRAYS,
        directions=directions.cpu().numpy().astype(np.float32),
        embedding_scores=embedding_scores.astype(np.float16),
        **{
            f"{corpus}_{name}": value
            for corpus, panel in panels.items()
            for name, value in panel.items()
        },
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(family_names))
    width = 0.25
    axes[0].bar(x - width, native_ood, width, label="native")
    axes[0].bar(x, rotated_ood, width, label="rotated same subspace")
    axes[0].bar(x + width, random_ood, width, label="isotropic random")
    axes[0].axhline(0.5, color="black", linewidth=1)
    axes[0].set_ylim(0.45, 1.0)
    axes[0].set_ylabel("TinyStories AUROC\n(FineWeb-frozen label and mode)")
    axes[0].set_xticks(x, family_names, rotation=25, ha="right")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Prospective semantic transfer")
    fine_corr = [abs(row["fineweb_embedding_corr"]) for row in summary_rows]
    tiny_corr = [abs(row["tinystories_embedding_corr"]) for row in summary_rows]
    axes[1].bar(x - width / 2, fine_corr, width, label="FineWeb")
    axes[1].bar(x + width / 2, tiny_corr, width, label="TinyStories")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("|corr(mode coordinate, embedding projection)|")
    axes[1].set_xticks(x, family_names, rotation=25, ha="right")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Direct token identity versus contextualization")
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)

    lines = [
        "# Natural semantic labels for frozen Layer-3 input modes",
        "",
        "The family channel banks and their bilinear input metrics are frozen before this run. "
        "FineWeb selects a current-token predicate, signed-versus-magnitude signal, orientation, "
        "and one of eight modes. The unchanged explanation is evaluated on TinyStories. A random "
        "rotation tests whether semantics belong to the family subspace rather than its native "
        "eigenvectors; isotropic random directions, wrong families, and generic token-form labels "
        "test weaker alternatives.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, value in hypothesis_results.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Layer-3 input-mode semantics](layer3_input_mode_semantics.png)",
        "",
        f"FineWeb sites: **{len(panels['fineweb']['token_ids']):,}**; TinyStories sites: "
        f"**{len(panels['tinystories']['token_ids']):,}**.",
        "",
        "| Family | Native mode | Frozen label | Signal | Fine AUROC | Tiny AUROC | Wrong | Generic | Rotated | Random | |embed corr| Fine/Tiny | Seed rank |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['family']} | {row['selected_native_mode']} | {row['predicate']} | "
            f"{row['signal']} | {row['fineweb_auc']:.3f} | {row['tinystories_auc']:.3f} | "
            f"{format_metric(row['wrong_auc'])} | {format_metric(row['generic_auc'])} | "
            f"{row['rotated_auc']:.3f} | {row['random_auc']:.3f} | "
            f"{abs(row['fineweb_embedding_corr']):.3f}/{abs(row['tinystories_embedding_corr']):.3f} | "
            f"{row['best_seed_rank']} |"
        )
    lines += [
        "",
        "A high current-token AUROC identifies a lexical coordinate, not the complete family "
        "interaction program. A rotated-subspace match would mean the family subspace is semantic "
        "but its individual eigenvectors are not uniquely privileged. Low embedding correlation "
        "with preserved AUROC would instead indicate a contextual transformation rather than "
        "simple survival of the repeatedly injected token embedding. Full nearest-token lists, "
        "positive/negative natural-context tails, all mode records, and frozen controls are in "
        "`layer3_input_mode_semantics.json`.",
        "",
        "The locked generic-token-form hypothesis is not supported and is partly non-adjudicative: "
        "for four representatives FineWeb selected a rare generic category with zero TinyStories "
        "positives, shown as `n/a`. This is retained as a failed prospective control rather than "
        "silently replacing the label after inspecting the evaluation corpus.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypothesis_results, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
