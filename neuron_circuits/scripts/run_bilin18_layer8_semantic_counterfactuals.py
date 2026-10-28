#!/usr/bin/env python3
"""Test recurring tensor signatures for minimal semantic token counterfactuals."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer8_behavioral_bottleneck import normalized_rows, orthonormal_row_space
from run_bilin18_layer8_direct_tensor_jacobian import (
    analytic_suffix,
    bilinear_mlp_jvp,
    output_fisher_response,
)
from run_bilin18_layer8_full_tensor_gram_rank import gram_eigensystem
from run_bilin18_layer8_router_causal_validation import exact_kl, prefix_to_layer, suffix_from_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT, fisher_fidelity


CONFIG = Path("configs/bilin18_layer8_semantic_counterfactuals.json")
OUTPUT = TENSOR_ROOT / "layer8_semantic_counterfactuals.json"
REPORT = TENSOR_ROOT / "LAYER8_SEMANTIC_COUNTERFACTUALS.md"
FIGURE = TENSOR_ROOT / "layer8_semantic_counterfactuals.png"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def projector_inner(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.square(first @ second.T).sum())


def projector_delta_cosine(first, second, key):
    a0, a1 = first[f"_{key}_base"], first[f"_{key}_counterfactual"]
    b0, b1 = second[f"_{key}_base"], second[f"_{key}_counterfactual"]
    numerator = (
        projector_inner(a1, b1) - projector_inner(a1, b0)
        - projector_inner(a0, b1) + projector_inner(a0, b0)
    )
    norm_a = projector_inner(a1, a1) - 2 * projector_inner(a1, a0) + projector_inner(a0, a0)
    norm_b = projector_inner(b1, b1) - 2 * projector_inner(b1, b0) + projector_inner(b0, b0)
    return numerator / max(np.sqrt(max(norm_a, 0) * max(norm_b, 0)), 1e-30)


def vector_cosine(first, second):
    return float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def comparison_metrics(first, second):
    return {
        "state_delta_cosine": vector_cosine(first["_state_delta"], second["_state_delta"]),
        "probability_delta_cosine": vector_cosine(first["_probability_delta"], second["_probability_delta"]),
        "input_projector_delta_cosine": projector_delta_cosine(first, second, "input"),
        "logit_projector_delta_cosine": projector_delta_cosine(first, second, "logit"),
        "fisher_projector_delta_cosine": projector_delta_cosine(first, second, "fisher"),
    }


def summarize_comparisons(records):
    buckets = {"same_pair": [], "same_family_different_pair": [], "different_family": []}
    for first_index in range(len(records)):
        for second_index in range(first_index + 1, len(records)):
            first, second = records[first_index], records[second_index]
            if first["pair"] == second["pair"]:
                bucket = "same_pair"
            elif first["family"] == second["family"]:
                bucket = "same_family_different_pair"
            else:
                bucket = "different_family"
            buckets[bucket].append(comparison_metrics(first, second))
    return {
        bucket: {
            key: median([row[key] for row in values])
            for key in values[0]
        } | {"pairs": len(values)}
        for bucket, values in buckets.items()
    }


def select_locations(rows, token, minimum, maximum, count, rng):
    locations = np.argwhere(np.asarray(rows[:, minimum:maximum + 1]) == token)
    rng.shuffle(locations)
    selected, used_rows = [], set()
    for row, relative_position in locations:
        if int(row) in used_rows:
            continue
        selected.append((int(row), int(relative_position + minimum)))
        used_rows.add(int(row))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"token {token} has only {len(selected)} unique contexts")
    return selected


@torch.inference_mode()
def analyze_state(model, block, basis, ids, position, axes):
    x0, value_bus, pre, normalized = prefix_to_layer(model, ids, 8)
    write = block.mlp.Down(block.mlp.Left(normalized) * block.mlp.Right(normalized)) + block.mlp.Down_bias
    layer_output = pre + write
    x = normalized[0, position].float()
    write_tangents = bilinear_mlp_jvp(block, x, basis.T)
    logits, logit_tangents, _ = analytic_suffix(
        model, layer_output, x0, value_bus, position, write_tangents, include_attention=True,
    )
    response = output_fisher_response(logits, logit_tangents)
    _, vectors = gram_eigensystem(response)
    probability = F.softmax(logits.float(), -1)
    centered = logit_tangents - (logit_tangents * probability[None]).sum(-1, keepdim=True)
    return {
        "x0": x0,
        "value_bus": value_bus,
        "pre": pre,
        "normalized": normalized,
        "layer_output": layer_output,
        "logits": logits,
        "probability": probability,
        "input_axes": vectors[:, :axes].T.cpu().numpy(),
        "logit_axes": orthonormal_row_space(vectors[:, :axes].T @ centered).cpu().numpy(),
        "fisher_axes": normalized_rows((response @ vectors[:, :axes]).T).cpu().numpy(),
    }


@torch.inference_mode()
def analyze_edit(model, block, basis, base_ids, counterfactual_ids, position, axes):
    base = analyze_state(model, block, basis, base_ids, position, axes)
    counterfactual = analyze_state(model, block, basis, counterfactual_ids, position, axes)
    raw_delta = counterfactual["normalized"][0, position].float() - base["normalized"][0, position].float()
    coordinate_delta = basis.T @ raw_delta
    retained = float(coordinate_delta.square().sum() / raw_delta.square().sum().clamp_min(1e-30))
    coordinate_norm = coordinate_delta.norm().clamp_min(1e-30)
    coordinate_unit = coordinate_delta / coordinate_norm
    projection = 0.5 * (
        np.square(base["input_axes"] @ coordinate_unit.cpu().numpy()).sum()
        + np.square(counterfactual["input_axes"] @ coordinate_unit.cpu().numpy()).sum()
    )

    patched_normalized = base["normalized"].clone()
    patched_normalized[0, position] = counterfactual["normalized"][0, position]
    patched_write = block.mlp.Down(
        block.mlp.Left(patched_normalized) * block.mlp.Right(patched_normalized)
    ) + block.mlp.Down_bias
    patched_logits = suffix_from_layer(
        model, 8, base["x0"], base["value_bus"], base["pre"] + patched_write, position,
    )[0]
    target_effect = counterfactual["logits"] - base["logits"]
    patch_effect = patched_logits - base["logits"]
    fidelity, cosine, _ = fisher_fidelity(base["logits"], target_effect, patch_effect)
    full_kl = exact_kl(base["logits"], counterfactual["logits"])
    patch_kl = exact_kl(base["logits"], patched_logits)
    probability_delta = (counterfactual["probability"] - base["probability"]).cpu().numpy()
    input_axis_overlap = projector_inner(base["input_axes"], counterfactual["input_axes"]) / axes
    logit_axis_overlap = projector_inner(base["logit_axes"], counterfactual["logit_axes"]) / axes
    fisher_axis_overlap = projector_inner(base["fisher_axes"], counterfactual["fisher_axes"]) / axes
    return {
        "full_kl": full_kl,
        "patch_kl": patch_kl,
        "patch_to_full_kl": patch_kl / max(full_kl, 1e-30),
        "patch_fidelity": fidelity,
        "patch_cosine": cosine,
        "top_prediction_changed": int(base["logits"].argmax() != counterfactual["logits"].argmax()),
        "state_basis_retained_fraction": retained,
        "top_axis_projection_fraction": float(projection),
        "input_axis_overlap": input_axis_overlap,
        "logit_axis_overlap": logit_axis_overlap,
        "fisher_axis_overlap": fisher_axis_overlap,
        "_state_delta": coordinate_unit.cpu().numpy(),
        "_probability_delta": probability_delta,
        "_input_base": base["input_axes"],
        "_input_counterfactual": counterfactual["input_axes"],
        "_logit_base": base["logit_axes"],
        "_logit_counterfactual": counterfactual["logit_axes"],
        "_fisher_base": base["fisher_axes"],
        "_fisher_counterfactual": counterfactual["fisher_axes"],
    }


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    source_config = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    tensor_config = {**source_config, "positions": source_config["discovery_positions"]}
    discovery_x, _ = collect(model, 8, source_config["tensor_discovery"], tensor_config)
    _, _, tensor_vh = torch.linalg.svd(discovery_x, full_matrices=False)
    basis = tensor_vh.T[:, :source_config["dimension"]]
    block = model.blocks[8]
    position = source_config["causal_position"]
    results = {}

    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        source = np.load(corpus_config["path"], mmap_mode="r")
        rows = source[corpus_config["start"]:corpus_config["start"] + corpus_config["rows"]]
        rng = np.random.default_rng(config["seed"] + corpus_index)
        records = []
        for family, pairs in config["families"].items():
            for source_token, target_token in pairs:
                locations = select_locations(
                    rows, source_token, config["replacement_position_min"],
                    config["replacement_position_max"], config["contexts_per_pair"], rng,
                )
                for row_index, edit_position in locations:
                    base_raw = np.asarray(rows[row_index, :position + 1], dtype=np.int64).copy()
                    counterfactual_raw = base_raw.copy()
                    counterfactual_raw[edit_position] = target_token
                    base_ids = torch.as_tensor(base_raw, device="cuda")[None]
                    counterfactual_ids = torch.as_tensor(counterfactual_raw, device="cuda")[None]
                    record = analyze_edit(
                        model, block, basis, base_ids, counterfactual_ids, position, config["axes"],
                    )
                    record.update({
                        "family": family,
                        "pair": f"{source_token}->{target_token}",
                        "source_token": tokenizer.decode([source_token]),
                        "target_token": tokenizer.decode([target_token]),
                        "row": row_index,
                        "edit_position": edit_position,
                        "distance": position - edit_position,
                        "context": tokenizer.decode(base_raw[max(0, edit_position - 8):min(position + 1, edit_position + 16)].tolist()),
                    })
                    records.append(record)
                print(corpus, family, tokenizer.decode([source_token]), "->", tokenizer.decode([target_token]), flush=True)
        comparisons = summarize_comparisons(records)
        public_records = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in records
        ]
        summary = {
            key: median([record[key] for record in records])
            for key in (
                "full_kl", "patch_kl", "patch_to_full_kl", "patch_fidelity",
                "patch_cosine", "state_basis_retained_fraction",
                "top_axis_projection_fraction", "input_axis_overlap",
                "logit_axis_overlap", "fisher_axis_overlap",
            )
        }
        summary["top_prediction_changed"] = float(np.mean([
            record["top_prediction_changed"] for record in records
        ]))
        results[corpus] = {
            "summary": summary,
            "comparisons": comparisons,
            "records": public_records,
        }

    def all_summary(key, threshold):
        return all(results[corpus]["summary"][key] >= threshold for corpus in results)

    def all_advantage(key, threshold):
        return all(
            results[corpus]["comparisons"]["same_family_different_pair"][key]
            - results[corpus]["comparisons"]["different_family"][key] >= threshold
            for corpus in results
        )

    hypotheses = {
        "F1_lexical_edit_directions_recur": {"supported": all(
            results[c]["comparisons"]["same_pair"]["state_delta_cosine"] >= 0.20 for c in results
        )},
        "F2_semantic_families_abstract_across_words": {"supported": all_advantage("state_delta_cosine", 0.05)},
        "F3_semantic_families_recurrently_reconfigure_input_sensitivity": {"supported": all_advantage("input_projector_delta_cosine", 0.05)},
        "F4_semantic_families_recurrently_reconfigure_behavior": {"supported": all_advantage("fisher_projector_delta_cosine", 0.05)},
        "F5_semantic_deltas_hit_top_local_axes": {"supported": all_summary("top_axis_projection_fraction", 0.05)},
        "F6_final_layer8_mlp_state_mediates_semantic_edit": {"supported": all_summary("patch_fidelity", 0.25) and all_summary("patch_to_full_kl", 0.10)},
        "F7_minimal_edits_change_behavior": {"supported": all_summary("full_kl", 1e-4)},
        "F8_local_input_operator_is_edit_stable": {"supported": all_summary("input_axis_overlap", 0.90)},
        "F9_local_behavioral_operator_is_edit_stable": {"supported": all_summary("fisher_axis_overlap", 0.75)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2))

    lines = [
        "# Natural semantic counterfactuals at Layer 8",
        "",
        "Natural FineWeb and TinyStories contexts are edited at one token 7–31 positions before the final prediction. Each semantic family contains three or four distinct lexical substitutions, allowing same-word recurrence to be separated from abstraction across words. The analysis compares Layer-8 state deltas, changes in the four leading input/logit/Fisher sensitivity projectors, and a finite patch of only the final-position Layer-8 MLP input.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Semantic counterfactuals](layer8_semantic_counterfactuals.png)",
        "",
        "| Corpus | Full KL | Prediction changed | Top-axis delta energy | L8 patch fidelity / KL ratio | Input/logit/Fisher axis preservation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for corpus, result in results.items():
        row = result["summary"]
        lines.append(
            f"| {corpus} | {row['full_kl']:.3e} | {row['top_prediction_changed']:.3f} | "
            f"{row['top_axis_projection_fraction']:.3f} | {row['patch_fidelity']:.3f} / {row['patch_to_full_kl']:.3f} | "
            f"{row['input_axis_overlap']:.3f} / {row['logit_axis_overlap']:.3f} / {row['fisher_axis_overlap']:.3f} |"
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
        "Same-pair recurrence can reflect lexical identity. Evidence for an abstract semantic mechanism requires same-family recurrence across different token pairs, above different-family controls, and preferably OOD replication. Final-position patching measures one mediation route; low fidelity does not imply the semantic edit is absent from earlier positions, attention, or the value bus.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, axes_plot = plt.subplots(1, 2, figsize=(11, 4.2))
    categories = ("same_pair", "same_family_different_pair", "different_family")
    x = np.arange(len(categories))
    width = 0.35
    for offset, corpus in ((-width / 2, "fineweb"), (width / 2, "tinystories")):
        axes_plot[0].bar(x + offset, [results[corpus]["comparisons"][category]["state_delta_cosine"] for category in categories], width, label=corpus)
        axes_plot[1].bar(x + offset, [results[corpus]["comparisons"][category]["input_projector_delta_cosine"] for category in categories], width, label=corpus)
    axes_plot[0].set_title("State-delta recurrence")
    axes_plot[1].set_title("Input-sensitivity change recurrence")
    for axis in axes_plot:
        axis.set_xticks(x, ["same pair", "same family", "different"], rotation=15)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
