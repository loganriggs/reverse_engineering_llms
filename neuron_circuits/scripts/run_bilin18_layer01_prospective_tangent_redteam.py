#!/usr/bin/env python3
"""Curvature, wrong-target, and clustered red-team of the prospective router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_layer01_final_bilinear_readout import downstream_fidelities
from run_bilin18_layer01_prospective_tangent_router import (
    cosine,
    finite_example,
    jaccard,
    normalized_embedding,
    prefix_pre_from_x0,
    score_rows,
    select_indices,
    tangent_product,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer01_prospective_tangent_redteam.json")
OUTPUT = TENSOR_ROOT / "layer01_prospective_tangent_redteam.json"
REPORT = TENSOR_ROOT / "LAYER01_PROSPECTIVE_TANGENT_REDTEAM.md"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def rank(values):
    order = np.argsort(values)
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def spearman(left, right):
    a, b = rank(np.asarray(left)), rank(np.asarray(right))
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(1e-30, np.linalg.norm(a) * np.linalg.norm(b)))


def soft_product(model, block, ids, target_token, layer, edit_position, final_position, dose):
    x0 = normalized_embedding(model, ids)
    target_id = torch.tensor([target_token], device=ids.device)
    target = normalized_embedding(model, target_id)[0]
    direction = torch.zeros_like(x0)
    direction[:, edit_position] = target - x0[:, edit_position]
    base_pre = prefix_pre_from_x0(model, x0, layer, final_position)[0]
    dose_pre = prefix_pre_from_x0(model, x0 + dose * direction, layer, final_position)[0]
    base_state = F.rms_norm(base_pre, (model.config.n_embd,))
    dose_state = F.rms_norm(dose_pre, (model.config.n_embd,))
    return (
        block.mlp.Left(dose_state) * block.mlp.Right(dose_state)
        - block.mlp.Left(base_state) * block.mlp.Right(base_state)
    )


def pair_bootstrap(rows, field_left, field_right, samples, seed):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair"], []).append(row[field_left] - row[field_right])
    pair_values = np.asarray([np.median(values) for values in grouped.values()])
    generator = np.random.default_rng(seed)
    draws = np.median(
        pair_values[generator.integers(0, len(pair_values), size=(samples, len(pair_values)))], axis=1,
    )
    return float(np.median(pair_values)), [float(value) for value in np.quantile(draws, [0.025, 0.975])]


@torch.no_grad()
def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source_config"]).read_text())
    source_results = json.loads(Path(config["source_results"]).read_text())
    data_config = json.loads(Path(source["data_config"]).read_text())
    evaluation_pairs = json.loads(Path(source["evaluation_pair_config"]).read_text())
    edit_position, final_position = data_config["edit_position"], data_config["causal_position"]
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in data_config["evaluation"].items()
    }
    all_targets = [target for pairs in evaluation_pairs["new_families"].values() for _, target in pairs]
    target_cycle = {target: all_targets[(index + 7) % len(all_targets)] for index, target in enumerate(all_targets)}
    records = {str(layer): [] for layer in source["layers"]}

    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        generator = np.random.default_rng(source["seed"] + corpus_index)
        for family, pairs in evaluation_pairs["new_families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, final_position,
                    source["evaluation_contexts_per_pair"], generator,
                )
                wrong_target = target_cycle[target_token]
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(
                        rows[row_index, start:start + data_config["sequence_length"]], dtype=np.int64,
                    ).copy()
                    ids = torch.as_tensor(raw, device="cuda")[None]
                    for layer in source["layers"]:
                        block = model.blocks[layer]
                        decoder = block.mlp.Down.weight.float()
                        base, _, full, base_product = finite_example(
                            model, ids, target_token, layer, edit_position, final_position,
                        )
                        base_tangent, _ = tangent_product(
                            model, block, ids, target_token, layer, edit_position, final_position,
                        )
                        midpoint_tangent, _ = tangent_product(
                            model, block, ids, target_token, layer, edit_position, final_position,
                            path_point=config["midpoint"],
                        )
                        wrong_tangent, _ = tangent_product(
                            model, block, ids, wrong_target, layer, edit_position, final_position,
                        )
                        actual_score = score_rows(full, decoder)["local"]
                        actual_indices = select_indices(actual_score, source["rank"])

                        candidates = {
                            "base_activity": base_product.abs() * decoder.norm(dim=0),
                            "base_tangent": score_rows(base_tangent, decoder)["local"],
                            "midpoint_tangent": score_rows(midpoint_tangent, decoder)["local"],
                            "wrong_tangent": score_rows(wrong_tangent, decoder)["local"],
                        }
                        indices = {
                            name: select_indices(score, source["rank"])
                            for name, score in candidates.items()
                        }
                        deltas = {"full": full}
                        for name, selected_indices in indices.items():
                            selected = torch.zeros_like(full)
                            selected[selected_indices] = full[selected_indices]
                            deltas[name] = selected
                        effects = downstream_fidelities(model, layer, base, deltas, final_position)
                        dose_cosines = {
                            str(dose): cosine(
                                base_tangent,
                                soft_product(
                                    model, block, ids, target_token, layer,
                                    edit_position, final_position, dose,
                                ),
                            )
                            for dose in config["doses"]
                        }
                        records[str(layer)].append({
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "base_activity": effects["base_activity"],
                            "base_tangent": effects["base_tangent"],
                            "midpoint_tangent": effects["midpoint_tangent"],
                            "wrong_tangent": effects["wrong_tangent"],
                            "base_support_jaccard": jaccard(indices["base_tangent"], actual_indices),
                            "midpoint_support_jaccard": jaccard(indices["midpoint_tangent"], actual_indices),
                            "wrong_support_jaccard": jaccard(indices["wrong_tangent"], actual_indices),
                            "base_product_cosine": cosine(base_tangent, full),
                            "midpoint_product_cosine": cosine(midpoint_tangent, full),
                            "wrong_product_cosine": cosine(wrong_tangent, full),
                            "dose_cosines": dose_cosines,
                        })
                print("tangent redteam", corpus, family, source_token, target_token, flush=True)

    summaries = {}
    for layer in source["layers"]:
        summaries[str(layer)] = {}
        for corpus in arrays:
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            bootstrap, interval = pair_bootstrap(
                selected, "base_tangent", "base_activity", config["bootstrap_samples"],
                config["seed"] + layer * 100 + list(arrays).index(corpus),
            )
            summaries[str(layer)][corpus] = {
                name: median([row[name] for row in selected])
                for name in (
                    "base_activity", "base_tangent", "midpoint_tangent", "wrong_tangent",
                    "base_support_jaccard", "midpoint_support_jaccard", "wrong_support_jaccard",
                    "base_product_cosine", "midpoint_product_cosine", "wrong_product_cosine",
                )
            }
            summaries[str(layer)][corpus].update({
                "dose_cosines": {
                    str(dose): median([row["dose_cosines"][str(dose)] for row in selected])
                    for dose in config["doses"]
                },
                "tangent_minus_base_pair_median": bootstrap,
                "tangent_minus_base_pair_ci": interval,
                "cosine_fidelity_spearman": spearman(
                    [row["base_product_cosine"] for row in selected],
                    [row["base_tangent"] for row in selected],
                ),
            })

    def row(layer, corpus):
        return summaries[str(layer)][corpus]

    hypotheses = {
        "R0_tangent_is_correct_in_the_small_edit_limit": {
            "supported": all(row(layer, corpus)["dose_cosines"]["0.1"] >= 0.90 for layer in source["layers"] for corpus in arrays)
        },
        "R1_full_edit_failure_is_curvature": {
            "supported": all(row(layer, corpus)["dose_cosines"]["0.1"] - row(layer, corpus)["dose_cosines"]["1.0"] >= 0.25 for layer in source["layers"] for corpus in arrays)
        },
        "R2_midpoint_tangent_rescues_full_direction": {
            "supported": all(row(layer, corpus)["midpoint_product_cosine"] >= 0.80 for layer in source["layers"] for corpus in arrays)
        },
        "R3_midpoint_router_is_causally_strong": {
            "supported": all(row(layer, corpus)["midpoint_tangent"] >= 0.60 for layer in source["layers"] for corpus in arrays)
        },
        "R4_target_identity_is_causally_specific": {
            "supported": all(row(layer, corpus)["base_tangent"] - row(layer, corpus)["wrong_tangent"] >= 0.10 for layer in source["layers"] for corpus in arrays)
        },
        "R5_target_identity_predicts_native_support": {
            "supported": all(row(layer, corpus)["base_support_jaccard"] - row(layer, corpus)["wrong_support_jaccard"] >= 0.05 for layer in source["layers"] for corpus in arrays)
        },
        "R6_base_tangent_beats_base_activity_under_pair_bootstrap": {
            "supported": all(row(layer, corpus)["tangent_minus_base_pair_ci"][0] > 0 for layer in source["layers"] for corpus in arrays)
        },
        "R7_midpoint_router_beats_base_activity": {
            "supported": all(row(layer, corpus)["midpoint_tangent"] - row(layer, corpus)["base_activity"] >= 0.15 for layer in source["layers"] for corpus in arrays)
        },
        "R8_direction_accuracy_predicts_causal_fidelity": {
            "supported": all(row(layer, corpus)["cosine_fidelity_spearman"] >= 0.30 for layer in source["layers"] for corpus in arrays)
        },
    }

    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": records,
    }, indent=2))
    lines = [
        "# Curvature and wrong-target red-team of the prospective tangent router",
        "",
        "The base JVP is checked against continuous embedding edits of increasing dose. A midpoint JVP tests whether one edit-conditioned relinearization rescues the finite circuit. A cyclic wrong target tests whether support prediction is target-specific. All causal scores patch the actual correct-target finite channel deltas, so a wrong-target router cannot win by changing the intervention being scored.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "| Layer | Corpus | Dose cosine .1/.25/.5/1 | Product cosine base/mid/wrong | Fidelity base-active/base/mid/wrong | Support Jaccard base/mid/wrong | Base tangent - activity pair median (95% CI) | Cosine↔fidelity Spearman |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in source["layers"]:
        for corpus in arrays:
            item = row(layer, corpus)
            doses = item["dose_cosines"]
            ci = item["tangent_minus_base_pair_ci"]
            lines.append(
                f"| {layer} | {corpus} | {doses['0.1']:.3f}/{doses['0.25']:.3f}/{doses['0.5']:.3f}/{doses['1.0']:.3f} | "
                f"{item['base_product_cosine']:.3f}/{item['midpoint_product_cosine']:.3f}/{item['wrong_product_cosine']:.3f} | "
                f"{item['base_activity']:.3f}/{item['base_tangent']:.3f}/{item['midpoint_tangent']:.3f}/{item['wrong_tangent']:.3f} | "
                f"{item['base_support_jaccard']:.3f}/{item['midpoint_support_jaccard']:.3f}/{item['wrong_support_jaccard']:.3f} | "
                f"{item['tangent_minus_base_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) | {item['cosine_fidelity_spearman']:.3f} |"
            )
    lines += [
        "",
        "The midpoint router is still prospective relative to the endpoint effect, but it evaluates the network at a soft embedding halfway to the requested token. It is therefore an executable nonlinear approximation, not a static feature dictionary. Pair bootstraps treat lexical pairs, not contexts or neurons, as replication units.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
