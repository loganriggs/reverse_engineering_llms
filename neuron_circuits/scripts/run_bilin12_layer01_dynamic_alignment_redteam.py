#!/usr/bin/env python3
"""Red-team 12-layer dynamic alignment with rotation and bank-data controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_dynamic_alignment_replication import (
    evaluate_deltas,
    final_product_delta,
    forward_layers_native,
    median,
    pair_cluster_interval,
    selected_delta,
)
from run_bilin18_layer01_native_rotation_tournament import (
    make_rotation,
    rotate,
    selected_rotated,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_dynamic_alignment_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_dynamic_alignment_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_DYNAMIC_ALIGNMENT_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_layer01_dynamic_alignment_redteam.png"


def jaccard(left, right):
    left, right = set(left.tolist()), set(right.tolist())
    return len(left & right) / len(left | right)


@torch.inference_mode()
def main():
    red = json.loads(CONFIG.read_text())
    config = json.loads(Path(red["source_config"]).read_text())
    primary = json.loads(Path(red["source_results"]).read_text())
    discovery_spec = json.loads(Path(config["discovery_pair_config"]).read_text())
    evaluation_spec = json.loads(Path(config["evaluation_pair_config"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    layers = config["layers"]
    families = list(discovery_spec["families"])
    ranks = red["bank_ranks"]
    edit_position, position = config["edit_position"], config["prediction_position"]

    rotations = {layer: {} for layer in layers}
    rotated_decoders = {layer: {} for layer in layers}
    for layer in layers:
        decoder = model.blocks[layer].mlp.Down.weight.float()
        for seed in red["rotation_seeds"]:
            rotations[layer][seed] = make_rotation(
                decoder.shape[1], config["rotation_rounds"], seed + layer, "cuda",
            )
            rotated_decoders[layer][seed] = rotate(decoder, rotations[layer][seed])

    # Eight contexts per pair, retaining discovery-half membership for stability.
    score_rows = {
        layer: {family: [] for family in families} for layer in layers
    }
    global_rows = {layer: [] for layer in layers}
    rng = np.random.default_rng(config["window_seed"])
    rows = arrays[config["discovery_corpus"]]
    for family, pairs in discovery_spec["families"].items():
        for source_token, target_token in pairs:
            windows = select_windows(
                rows, source_token, edit_position, position,
                red["discovery_contexts_per_pair"], rng,
            )
            for context_index, (row_index, start) in enumerate(windows):
                raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy()
                changed = raw.copy(); changed[edit_position] = target_token
                base = forward_layers_native(model, torch.as_tensor(raw, device="cuda")[None], max(layers))
                counter = forward_layers_native(model, torch.as_tensor(changed, device="cuda")[None], max(layers))
                for layer in layers:
                    delta = final_product_delta(model, layer, base[layer], counter[layer], position)
                    decoder = model.blocks[layer].mlp.Down.weight.float()
                    write = F.linear(delta, decoder)
                    score = (delta * (write @ decoder)).abs().cpu()
                    item = {"score": score, "half": context_index // 4}
                    score_rows[layer][family].append(item)
                    global_rows[layer].append(item)
            print("redteam discovery", family, source_token, target_token, flush=True)

    banks = {layer: {} for layer in layers}
    support_stability = {}
    for layer in layers:
        support_stability[str(layer)] = {}
        for family in families:
            matrix = torch.stack([row["score"] for row in score_rows[layer][family]])
            banks[layer][family] = {
                rank: matrix.mean(0).topk(rank).indices.cuda() for rank in ranks
            }
            half_banks = []
            for half in (0, 1):
                half_matrix = torch.stack([
                    row["score"] for row in score_rows[layer][family] if row["half"] == half
                ])
                half_banks.append(half_matrix.mean(0).topk(red["comparison_rank"]).indices)
            support_stability[str(layer)][family] = jaccard(*half_banks)
        global_matrix = torch.stack([row["score"] for row in global_rows[layer]])
        banks[layer]["global"] = {
            rank: global_matrix.mean(0).topk(rank).indices.cuda() for rank in ranks
        }

    records = {str(layer): [] for layer in layers}
    maximum_rotation_error = 0.0
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family, pairs in evaluation_spec["new_families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, edit_position, position,
                    config["evaluation_contexts_per_pair"], rng,
                )
                for context_index, (row_index, start) in enumerate(windows):
                    raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy()
                    changed = raw.copy(); changed[edit_position] = target_token
                    base = forward_layers_native(model, torch.as_tensor(raw, device="cuda")[None], max(layers))
                    counter = forward_layers_native(model, torch.as_tensor(changed, device="cuda")[None], max(layers))
                    for layer in layers:
                        decoder = model.blocks[layer].mlp.Down.weight.float()
                        full = final_product_delta(model, layer, base[layer], counter[layer], position)
                        write = F.linear(full, decoder)
                        native_score = (full * (write @ decoder)).abs()
                        deltas = {
                            "native": selected_delta(full, native_score.topk(red["comparison_rank"]).indices),
                        }
                        for seed in red["rotation_seeds"]:
                            rotated_full = rotate(full, rotations[layer][seed])
                            rotated_score = (rotated_full * (write @ rotated_decoders[layer][seed])).abs()
                            deltas[f"rotation_{seed}"] = selected_rotated(
                                full, rotated_score.topk(red["comparison_rank"]).indices, rotations[layer][seed],
                            )
                            restored = selected_rotated(
                                full, torch.arange(len(full), device="cuda"), rotations[layer][seed],
                            )
                            maximum_rotation_error = max(
                                maximum_rotation_error,
                                float((restored - full).norm() / full.norm().clamp_min(1e-30)),
                            )
                        for rank in ranks:
                            for bank_family in families:
                                deltas[f"bank_{bank_family}_{rank}"] = selected_delta(
                                    full, banks[layer][bank_family][rank],
                                )
                            deltas[f"global_{rank}"] = selected_delta(full, banks[layer]["global"][rank])
                        effects = evaluate_deltas(model, layer, base[layer], deltas, full, position)
                        row = {
                            "corpus": corpus,
                            "family": family,
                            "pair": f"{source_token}->{target_token}",
                            "context": context_index,
                            "native": effects["native"],
                            "rotations": {str(seed): effects[f"rotation_{seed}"] for seed in red["rotation_seeds"]},
                            "best_rotation": max(effects[f"rotation_{seed}"] for seed in red["rotation_seeds"]),
                            "banks": {},
                        }
                        for rank in ranks:
                            correct = effects[f"bank_{family}_{rank}"]
                            wrong = median([
                                effects[f"bank_{other}_{rank}"] for other in families if other != family
                            ])
                            row["banks"][str(rank)] = {
                                "correct": correct,
                                "wrong": wrong,
                                "global": effects[f"global_{rank}"],
                            }
                        records[str(layer)].append(row)
                print("redteam evaluation", corpus, family, source_token, target_token, flush=True)

    summaries = {str(layer): {} for layer in layers}
    for layer in layers:
        for corpus_index, corpus in enumerate(config["evaluation"]):
            selected = [row for row in records[str(layer)] if row["corpus"] == corpus]
            robust_center, robust_ci = pair_cluster_interval(
                selected, lambda row: row["native"] - row["best_rotation"],
                red["bootstrap_samples"], red["bootstrap_seed"] + layer * 100 + corpus_index,
            )
            bank_summary = {}
            for rank_index, rank in enumerate(ranks):
                advantage, advantage_ci = pair_cluster_interval(
                    selected, lambda row, rank=rank: row["banks"][str(rank)]["correct"] - row["banks"][str(rank)]["wrong"],
                    red["bootstrap_samples"], red["bootstrap_seed"] + 10000 + layer * 1000 + corpus_index * 100 + rank_index,
                )
                bank_summary[str(rank)] = {
                    "correct": median([row["banks"][str(rank)]["correct"] for row in selected]),
                    "wrong": median([row["banks"][str(rank)]["wrong"] for row in selected]),
                    "global": median([row["banks"][str(rank)]["global"] for row in selected]),
                    "advantage_pair_median": advantage,
                    "advantage_pair_ci": advantage_ci,
                }
            summaries[str(layer)][corpus] = {
                "native": median([row["native"] for row in selected]),
                "rotation_medians": {
                    str(seed): median([row["rotations"][str(seed)] for row in selected])
                    for seed in red["rotation_seeds"]
                },
                "best_rotation": median([row["best_rotation"] for row in selected]),
                "native_minus_best_pair_median": robust_center,
                "native_minus_best_pair_ci": robust_ci,
                "banks": bank_summary,
            }

    def cell(layer, corpus):
        return summaries[str(layer)][corpus]

    multiple_rotations = all(
        cell(layer, corpus)["native_minus_best_pair_median"] >= 0.03
        and cell(layer, corpus)["native_minus_best_pair_ci"][0] > 0
        for layer in layers for corpus in config["evaluation"]
    )
    rescue_ranks = []
    label_ranks = []
    global_ranks = []
    for rank in ranks:
        rescue = all(
            cell(layer, corpus)["banks"][str(rank)]["correct"] >= 0.50
            and cell(layer, corpus)["banks"][str(rank)]["advantage_pair_median"] >= 0.05
            and cell(layer, corpus)["banks"][str(rank)]["advantage_pair_ci"][0] > 0
            and cell(layer, corpus)["banks"][str(rank)]["correct"] - cell(layer, corpus)["banks"][str(rank)]["global"] >= 0.03
            for layer in layers for corpus in config["evaluation"]
        )
        labels = all(
            cell(layer, corpus)["banks"][str(rank)]["advantage_pair_ci"][0] > 0
            for layer in layers for corpus in config["evaluation"]
        )
        beats_global = all(
            cell(layer, corpus)["banks"][str(rank)]["correct"] - cell(layer, corpus)["banks"][str(rank)]["global"] >= 0.03
            for layer in layers for corpus in config["evaluation"]
        )
        if rescue: rescue_ranks.append(rank)
        if labels: label_ranks.append(rank)
        if beats_global: global_ranks.append(rank)
    source_ranks = primary["config"]["ranks"]
    fixed_weaker = all(
        cell(layer, corpus)["banks"][str(rank)]["correct"]
        <= primary["summaries"][str(layer)][corpus]["curves"]["native_local"][source_ranks.index(rank)] - 0.15
        for rank in ranks for layer in layers for corpus in config["evaluation"]
    )
    primary_256 = {
        (layer, corpus): primary["summaries"][str(layer)][corpus]["correct_bank"]
        for layer in layers for corpus in config["evaluation"]
    }
    discovery_noise = all(
        cell(layer, corpus)["banks"]["256"]["correct"] - primary_256[(layer, corpus)] < 0.10
        for layer in layers for corpus in config["evaluation"]
    ) and not rescue_ranks
    stability_medians = {
        str(layer): median(list(support_stability[str(layer)].values())) for layer in layers
    }
    hypotheses = {
        "M0_rotation_round_trips_are_exact": {"supported": maximum_rotation_error <= 1e-5, "maximum_relative_error": maximum_rotation_error},
        "M1_native_privilege_survives_multiple_rotations": {"supported": multiple_rotations},
        "M2_more_discovery_data_rescues_a_fixed_semantic_code": {"supported": bool(rescue_ranks), "passing_ranks": rescue_ranks},
        "M3_any_fixed_family_label_signal_is_universal": {"supported": bool(label_ranks), "passing_ranks": label_ranks},
        "M4_family_banks_beat_generic_banks": {"supported": bool(global_ranks), "passing_ranks": global_ranks},
        "M5_fixed_banks_remain_weaker_than_dynamic_selection": {"supported": fixed_weaker},
        "M6_two_context_failure_was_not_just_estimation_noise": {"supported": discovery_noise},
        "M7_family_bank_support_is_reproducible": {"supported": all(value >= 0.20 for value in stability_medians.values()), "median_jaccard": stability_medians},
    }
    OUTPUT.write_text(json.dumps({
        "config": red,
        "hypotheses": hypotheses,
        "maximum_rotation_error": maximum_rotation_error,
        "support_stability": support_stability,
        "summaries": summaries,
        "records": records,
    }, indent=2, allow_nan=False))

    lines = [
        "# Rotation and discovery-data red-team of Layer-0/1 dynamic alignment", "",
        "Seven independently randomized dense orthogonal coordinate systems are compared with the native bilinear basis at rank 256. Frozen semantic banks are refit from eight rather than two discovery contexts per old lexical pair and swept from rank 64 to 1,024. Evaluation remains on the seventeen untouched lexical pairs in FineWeb and TinyStories.", "", "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "![Red-team curves](bilin12_layer01_dynamic_alignment_redteam.png)", "",
        "| Layer | Corpus | Native / best rotation | Native-best pair median (95% CI) | Bank rank 64/128/256/512/1024 correct | Corresponding wrong | Corresponding global |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for layer in layers:
        for corpus in config["evaluation"]:
            row = cell(layer, corpus); ci = row["native_minus_best_pair_ci"]
            series = lambda key: "/".join(f"{row['banks'][str(rank)][key]:.3f}" for rank in ranks)
            lines.append(
                f"| {layer} | {corpus} | {row['native']:.3f}/{row['best_rotation']:.3f} | "
                f"{row['native_minus_best_pair_median']:+.3f} ({ci[0]:+.3f}, {ci[1]:+.3f}) | "
                f"{series('correct')} | {series('wrong')} | {series('global')} |"
            )
    lines += ["", "Discovery-half rank-256 support Jaccard:", ""]
    for layer in layers:
        values = support_stability[str(layer)]
        lines.append(f"- Layer {layer}: median **{stability_medians[str(layer)]:.3f}**; " + ", ".join(f"{family}={value:.3f}" for family, value in values.items()))
    lines += [
        "", f"Maximum inverse-rotation relative error: **{maximum_rotation_error:.2e}**.", "",
        "The best-of-seven rotation is an intentionally adversarial comparator: it gives the rotated basis seven chances per example. A fixed family bank is credited only if one preregistered rank works in every layer/corpus cell; isolated positive cells are retained in the raw table but do not establish a universal semantic neuron dictionary.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (layer, corpus) in zip(axes.flat, [(l, c) for l in layers for c in config["evaluation"]]):
        row = cell(layer, corpus)
        axis.plot(ranks, [row["banks"][str(rank)]["correct"] for rank in ranks], marker="o", label="correct family")
        axis.plot(ranks, [row["banks"][str(rank)]["wrong"] for rank in ranks], marker="o", label="wrong-family median")
        axis.plot(ranks, [row["banks"][str(rank)]["global"] for rank in ranks], marker="o", label="global")
        axis.plot(ranks, [primary["summaries"][str(layer)][corpus]["curves"]["native_local"][source_ranks.index(rank)] for rank in ranks], marker="o", label="dynamic native")
        axis.set_xscale("log", base=2); axis.set_ylim(-0.1, 1.05); axis.grid(alpha=0.25)
        axis.set_title(f"Layer {layer} / {corpus}"); axis.set_xlabel("channel budget"); axis.set_ylabel("Fisher fidelity"); axis.legend(fontsize=7)
    figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
