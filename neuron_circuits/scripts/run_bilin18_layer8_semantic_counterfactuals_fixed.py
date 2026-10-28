#!/usr/bin/env python3
"""Fixed-position red-team for natural Layer-8 semantic counterfactuals."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_semantic_counterfactuals import analyze_edit, median, summarize_comparisons
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_semantic_counterfactuals_fixed.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_counterfactuals_fixed.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_COUNTERFACTUALS_FIXED.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_counterfactuals_fixed.png"


def select_windows(rows, token, edit_position, causal_position, count, rng):
    minimum_source = edit_position
    maximum_source = rows.shape[1] - (causal_position - edit_position) - 1
    locations = np.argwhere(np.asarray(rows[:, minimum_source:maximum_source + 1]) == token)
    rng.shuffle(locations)
    selected, used_rows = [], set()
    for row, relative_position in locations:
        row = int(row)
        source_position = int(relative_position + minimum_source)
        if row in used_rows:
            continue
        start = source_position - edit_position
        selected.append((row, start))
        used_rows.add(row)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"token {token} has only {len(selected)} usable independent windows")
    return selected


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    parent = json.loads(Path(config["source_config"]).read_text())
    tensor_source = json.loads(Path(parent["source_config"]).read_text())
    baseline = json.loads(Path(config["baseline_results"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    tensor_config = {**tensor_source, "positions": tensor_source["discovery_positions"]}
    discovery_x, _ = collect(model, 8, tensor_source["tensor_discovery"], tensor_config)
    _, _, tensor_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = tensor_vh.T[:, :tensor_source["dimension"]]
    block = model.blocks[8]
    results = {}

    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(config["seed"] + corpus_index)
        records = []
        for family, pairs in parent["families"].items():
            for source_token, target_token in pairs:
                windows = select_windows(
                    rows, source_token, config["edit_position"], config["causal_position"],
                    config["contexts_per_pair"], rng,
                )
                for row_index, start in windows:
                    base_raw = np.asarray(
                        rows[row_index, start:start + config["sequence_length"]], dtype=np.int64,
                    ).copy()
                    counterfactual_raw = base_raw.copy()
                    counterfactual_raw[config["edit_position"]] = target_token
                    record = analyze_edit(
                        model, block, basis,
                        torch.as_tensor(base_raw, device="cuda")[None],
                        torch.as_tensor(counterfactual_raw, device="cuda")[None],
                        config["causal_position"], config["axes"],
                    )
                    record.update({
                        "family": family,
                        "pair": f"{source_token}->{target_token}",
                        "source_token": tokenizer.decode([source_token]),
                        "target_token": tokenizer.decode([target_token]),
                        "row": row_index,
                        "source_start": start,
                        "context": tokenizer.decode(base_raw[config["edit_position"] - 8:].tolist()),
                    })
                    records.append(record)
                print(corpus, family, tokenizer.decode([source_token]), "->", tokenizer.decode([target_token]), flush=True)
        comparisons = summarize_comparisons(records)
        high_effect = []
        for pair in sorted({record["pair"] for record in records}):
            pair_records = sorted(
                (record for record in records if record["pair"] == pair),
                key=lambda record: record["full_kl"], reverse=True,
            )
            high_effect.extend(pair_records[:config["high_effect_contexts_per_pair"]])
        high_effect_comparisons = summarize_comparisons(high_effect)
        public_records = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in records
        ]
        summary = {
            key: median([record[key] for record in records])
            for key in (
                "full_kl", "patch_kl", "patch_to_full_kl", "patch_fidelity",
                "patch_cosine", "state_basis_retained_fraction", "top_axis_projection_fraction",
                "input_axis_overlap", "logit_axis_overlap", "fisher_axis_overlap",
            )
        }
        summary["top_prediction_changed"] = float(np.mean([
            record["top_prediction_changed"] for record in records
        ]))
        high_effect_summary = {
            key: median([record[key] for record in high_effect])
            for key in (
                "full_kl", "patch_to_full_kl", "patch_fidelity",
                "top_axis_projection_fraction",
            )
        }
        results[corpus] = {
            "summary": summary,
            "comparisons": comparisons,
            "high_effect_summary": high_effect_summary,
            "high_effect_comparisons": high_effect_comparisons,
            "records": public_records,
        }

    def all_advantage(key, threshold):
        return all(
            results[c]["comparisons"]["same_family_different_pair"][key]
            - results[c]["comparisons"]["different_family"][key] >= threshold
            for c in results
        )

    hypotheses = {
        "G1_fixed_distance_lexical_directions_recur": {"supported": all(results[c]["comparisons"]["same_pair"]["state_delta_cosine"] >= 0.20 for c in results)},
        "G2_fixed_distance_semantic_abstraction": {"supported": all_advantage("state_delta_cosine", 0.05)},
        "G3_fixed_distance_input_reconfiguration_recurs": {"supported": all_advantage("input_projector_delta_cosine", 0.05)},
        "G4_fixed_distance_behavior_reconfiguration_recurs": {"supported": all_advantage("fisher_projector_delta_cosine", 0.05)},
        "G5_fixed_distance_deltas_hit_top_axes": {"supported": all(results[c]["summary"]["top_axis_projection_fraction"] >= 0.05 for c in results)},
        "G6_fixed_distance_final_mlp_mediates": {"supported": all(results[c]["summary"]["patch_fidelity"] >= 0.25 for c in results)},
        "G7_position_control_rescues_lexical_recurrence": {"supported": all(
            results[c]["comparisons"]["same_pair"]["state_delta_cosine"]
            - baseline["results"][c]["comparisons"]["same_pair"]["state_delta_cosine"] >= 0.05
            for c in results
        )},
        "G8_high_effect_semantic_state_abstraction": {"supported": all(
            results[c]["high_effect_comparisons"]["same_family_different_pair"]["state_delta_cosine"]
            - results[c]["high_effect_comparisons"]["different_family"]["state_delta_cosine"] >= 0.05
            for c in results
        )},
        "G9_high_effect_input_reconfiguration_recurs": {"supported": all(
            results[c]["high_effect_comparisons"]["same_family_different_pair"]["input_projector_delta_cosine"]
            - results[c]["high_effect_comparisons"]["different_family"]["input_projector_delta_cosine"] >= 0.05
            for c in results
        )},
        "G10_high_effect_behavior_reconfiguration_recurs": {"supported": all(
            results[c]["high_effect_comparisons"]["same_family_different_pair"]["fisher_projector_delta_cosine"]
            - results[c]["high_effect_comparisons"]["different_family"]["fisher_projector_delta_cosine"] >= 0.05
            for c in results
        )},
        "G11_high_effect_deltas_hit_top_axes": {"supported": all(
            results[c]["high_effect_summary"]["top_axis_projection_fraction"] >= 0.05 for c in results
        )},
        "G12_high_effect_final_mlp_mediates": {"supported": all(
            results[c]["high_effect_summary"]["patch_fidelity"] >= 0.25 for c in results
        )},
        "G13_fixed_distance_input_operator_is_edit_stable": {"supported": all(
            results[c]["summary"]["input_axis_overlap"] >= 0.90 for c in results
        )},
        "G14_fixed_distance_behavioral_operator_is_edit_stable": {"supported": all(
            results[c]["summary"]["fisher_axis_overlap"] >= 0.75 for c in results
        )},
    }
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "results": results,
        "baseline_same_pair_state_cosine": {
            c: baseline["results"][c]["comparisons"]["same_pair"]["state_delta_cosine"] for c in results
        },
    }, indent=2))

    lines = [
        "# Fixed-position semantic counterfactual red-team",
        "",
        "This repeats the natural one-token counterfactual experiment on cropped 64-token windows. Every edited token is at position 47 and every effect is evaluated at position 63, exactly 16 tokens later. This removes edit distance, absolute position, and sequence length as explanations of the earlier non-recurrence.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Fixed semantic counterfactuals](layer8_semantic_counterfactuals_fixed.png)",
        "",
        "| Corpus | Full KL | Prediction changed | Top-axis delta energy | L8 patch fidelity / KL ratio | Input/logit/Fisher preservation | Same-pair state cosine (variable→fixed) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        row = result["summary"]
        old = baseline["results"][corpus]["comparisons"]["same_pair"]["state_delta_cosine"]
        new = result["comparisons"]["same_pair"]["state_delta_cosine"]
        lines.append(
            f"| {corpus} | {row['full_kl']:.3e} | {row['top_prediction_changed']:.3f} | "
            f"{row['top_axis_projection_fraction']:.3f} | {row['patch_fidelity']:.3f} / {row['patch_to_full_kl']:.3f} | "
            f"{row['input_axis_overlap']:.3f} / {row['logit_axis_overlap']:.3f} / {row['fisher_axis_overlap']:.3f} | {old:.3f}→{new:.3f} |"
        )
    lines += [
        "",
        "## High-effect strength control",
        "",
        "The two largest full-KL contexts per lexical pair were selected after the aggregate fixed-position result but before inspecting their internal recurrence metrics.",
        "",
        "| Corpus | Full KL | Top-axis delta energy | Patch fidelity / KL ratio | Same-family state/input/Fisher advantage |",
        "|---|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        row = result["high_effect_summary"]
        same = result["high_effect_comparisons"]["same_family_different_pair"]
        different = result["high_effect_comparisons"]["different_family"]
        lines.append(
            f"| {corpus} | {row['full_kl']:.3e} | {row['top_axis_projection_fraction']:.3f} | "
            f"{row['patch_fidelity']:.3f} / {row['patch_to_full_kl']:.3f} | "
            f"{same['state_delta_cosine'] - different['state_delta_cosine']:.3f} / "
            f"{same['input_projector_delta_cosine'] - different['input_projector_delta_cosine']:.3f} / "
            f"{same['fisher_projector_delta_cosine'] - different['fisher_projector_delta_cosine']:.3f} |"
        )
    lines += [
        "",
        "| Corpus | Comparison | State delta | Input-projector Δ | Logit-projector Δ | Fisher-projector Δ | Probability delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        for comparison, row in result["comparisons"].items():
            lines.append(
                f"| {corpus} | {comparison} | {row['state_delta_cosine']:.3f} | "
                f"{row['input_projector_delta_cosine']:.3f} | {row['logit_projector_delta_cosine']:.3f} | "
                f"{row['fisher_projector_delta_cosine']:.3f} | {row['probability_delta_cosine']:.3f} |"
            )
    lines += [
        "",
        "If same-pair recurrence remains weak, position was not the main reason semantic edit directions rotated. If lexical recurrence improves without across-word family recurrence, the reusable object is token-pair-specific rather than abstractly semantic.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes_plot = plt.subplots(1, 2, figsize=(10.5, 4.2))
    corpora = list(results)
    x = np.arange(len(corpora))
    old = [baseline["results"][c]["comparisons"]["same_pair"]["state_delta_cosine"] for c in corpora]
    new = [results[c]["comparisons"]["same_pair"]["state_delta_cosine"] for c in corpora]
    axes_plot[0].bar(x - 0.18, old, 0.36, label="variable distance")
    axes_plot[0].bar(x + 0.18, new, 0.36, label="fixed distance")
    axes_plot[0].set_xticks(x, corpora)
    axes_plot[0].set_title("Same-token-pair state recurrence")
    metrics = ("state_delta_cosine", "input_projector_delta_cosine", "fisher_projector_delta_cosine")
    width = 0.35
    for offset, corpus in ((-width / 2, "fineweb"), (width / 2, "tinystories")):
        same = results[corpus]["comparisons"]["same_family_different_pair"]
        different = results[corpus]["comparisons"]["different_family"]
        axes_plot[1].bar(
            np.arange(len(metrics)) + offset,
            [same[key] - different[key] for key in metrics], width, label=corpus,
        )
    axes_plot[1].set_xticks(np.arange(len(metrics)), ["state", "input Δ", "Fisher Δ"])
    axes_plot[1].set_title("Semantic-family advantage")
    for axis in axes_plot:
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
