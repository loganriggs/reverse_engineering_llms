#!/usr/bin/env python3
"""Test a 32-feature direction dictionary against matched random objectives."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_dynamic_mlp_jacobian_spectrum import exact_incoming_adjoints
from run_bilin18_layer01_supervised_feature_census import FEATURES, forward_census, vocabulary_features
from run_bilin18_layer1_suffix_degree_pilot import mlp_vjp
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_competent_feature_direction_dictionary.json")
OUTPUT = TENSOR_ROOT / "bilin18_competent_feature_direction_dictionary.json"
ARRAYS = TENSOR_ROOT / "bilin18_competent_feature_direction_dictionary.npz"
REPORT = TENSOR_ROOT / "BILIN18_COMPETENT_FEATURE_DIRECTION_DICTIONARY.md"
FIGURE = TENSOR_ROOT / "bilin18_competent_feature_direction_dictionary.png"


def unit_rows(array):
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-30)


def prototype_metrics(array, labels):
    array = unit_rows(array.astype(np.float64)); classes = np.unique(labels); ranks = []
    for target in range(len(labels)):
        prototypes = []
        for label in classes:
            indices = np.flatnonzero(labels == label); indices = indices[indices != target]
            prototype = array[indices].mean(0); prototype /= max(np.linalg.norm(prototype), 1e-30); prototypes.append(prototype)
        scores = np.asarray(prototypes) @ array[target]; order = np.argsort(scores)[::-1]; true_index = int(np.flatnonzero(classes == labels[target])[0]); ranks.append(int(np.flatnonzero(order == true_index)[0]) + 1)
    ranks = np.asarray(ranks)
    return {"top1": float(np.mean(ranks == 1)), "top5": float(np.mean(ranks <= 5)), "mean_reciprocal_rank": float(np.mean(1 / ranks)), "median_rank": float(np.median(ranks))}


def recurrence_metrics(array, labels, positions):
    unit = unit_rows(array.astype(np.float64)); matrix = np.square(unit @ unit.T); same = []; different = []; nearest = []
    for target in range(len(labels)):
        same_indices = np.flatnonzero(labels == labels[target]); same_indices = same_indices[same_indices != target]
        different_indices = np.flatnonzero(labels != labels[target]); distance = np.min(np.abs(positions[different_indices] - positions[target])); nearest_indices = different_indices[np.abs(positions[different_indices] - positions[target]) == distance]
        same.append(matrix[target, same_indices].mean()); different.append(matrix[target, different_indices].mean()); nearest.append(matrix[target, nearest_indices].mean())
    same = np.asarray(same); different = np.asarray(different); nearest = np.asarray(nearest)
    return {"same_mean": float(same.mean()), "different_mean": float(different.mean()), "nearest_position_different_mean": float(nearest.mean()), "same_to_different_ratio": float(same.mean() / max(different.mean(), 1e-30)), "same_to_position_control_ratio": float(same.mean() / max(nearest.mean(), 1e-30)), "paired_positive_fraction": float(np.mean(same > different))}


def collect_candidate_scores(model, rows, positions, flags, contrasts, batch_size):
    scores = []; labels = []; current_ids = []; next_ids = []
    position_array = np.asarray(positions)
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :129], dtype=np.int64); ids = torch.as_tensor(raw[:, :128], device="cuda")
        with torch.no_grad(): _, batch_scores = forward_census(model, ids, positions, contrasts)
        scores.append(batch_scores.cpu().numpy()); next_batch = raw[:, position_array + 1].reshape(-1); labels.append(flags[next_batch]); current_ids.append(raw[:, position_array].reshape(-1)); next_ids.append(next_batch)
        if start % 64 == 0: print(f"dictionary candidate scan {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(current_ids), np.concatenate(next_ids)


def choose_sites(features, feature_indices, scores, labels, positions, extreme_count, uniform_count, rng):
    chosen = []
    for local_label, feature_index in enumerate(feature_indices):
        candidates = np.flatnonzero(labels[:, feature_index].astype(bool)); order = candidates[np.argsort(scores[candidates, feature_index])[::-1]]; extreme = order[:extreme_count]; remaining = np.setdiff1d(candidates, extreme, assume_unique=False); uniform = rng.choice(remaining, size=uniform_count, replace=False)
        for selection, indices in (("extreme", extreme), ("uniform", uniform)):
            for flat in indices:
                chosen.append({"feature": features[local_label], "feature_index": int(feature_index), "label": local_label, "selection": selection, "flat_index": int(flat), "row": int(flat // len(positions)), "position": int(positions[flat % len(positions)]), "model_score": float(scores[flat, feature_index])})
    return chosen


def main():
    config = json.loads(CONFIG.read_text()); census_config = json.loads(Path(config["census_config"]).read_text()); census = json.loads(Path(config["census_results"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]; flags = np.stack([vocabulary_features(text) for text in texts]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive
    contrasts_np = flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None]; contrasts = torch.as_tensor(contrasts_np, device="cuda", dtype=torch.float32)
    competent = [row["feature"] for row in census["features"] if row["competent"]]; feature_indices = [FEATURES.index(name) for name in competent]
    site_rng = np.random.default_rng(config["seed"]); independent_rng = np.random.default_rng(config["seed"] + 1); shared_rng = np.random.default_rng(config["seed"] + 2); semantic_contrasts_np = contrasts_np[:, feature_indices]
    random_contrasts_np = np.stack([contrasts_np[independent_rng.permutation(len(flags)), index] for index in feature_indices], axis=1); random_contrasts = torch.as_tensor(random_contrasts_np, device="cuda", dtype=torch.float32)
    shared_permutation_np = semantic_contrasts_np[shared_rng.permutation(len(flags))]; shared_permutation = torch.as_tensor(shared_permutation_np, device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(census_config["corpus"], mmap_mode="r"); heldout_start = census_config["start_row"] + census_config["discovery_rows"]; rows = corpus[heldout_start:heldout_start + census_config["heldout_rows"]]
    candidate_scores, candidate_labels, current_ids, next_ids = collect_candidate_scores(model, rows, census_config["positions"], flags, contrasts, config["batch_size"])
    sites = choose_sites(competent, feature_indices, candidate_scores, candidate_labels, census_config["positions"], config["extreme_sites_per_feature"], config["uniform_sites_per_feature"], site_rng); labels = np.asarray([site["label"] for site in sites]); positions = np.asarray([site["position"] for site in sites])
    objectives_used = ("semantic", "random", "shared_permutation")
    vectors = {objective: {family: {str(block): [] for block in config["blocks"]} for family in ("incoming", "response")} for objective in objectives_used}
    records = []
    for site_index, site in enumerate(sites):
        raw = np.asarray(rows[site["row"], :census_config["sequence_length"]], dtype=np.int64); ids = torch.as_tensor(raw, device="cuda")[None]
        with torch.no_grad(): trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        objectives = {"semantic": contrasts[:, site["feature_index"]], "random": random_contrasts[:, site["label"]], "shared_permutation": shared_permutation[:, site["label"]]}
        record = dict(site); record.update({"current_token": tokenizer.decode([int(raw[site["position"]])]), "next_token": tokenizer.decode([int(rows[site["row"], site["position"] + 1])])})
        for objective, contrast in objectives.items():
            incoming = exact_incoming_adjoints(model, trace, site["position"], contrast)
            for block in config["blocks"]:
                vector = incoming[block][0, 0, site["position"]]; response = mlp_vjp(model.blocks[block], trace[block]["pre"].float(), incoming[block])[0, 0, site["position"]]
                vectors[objective]["incoming"][str(block)].append(vector.cpu().numpy().astype(np.float32)); vectors[objective]["response"][str(block)].append(response.cpu().numpy().astype(np.float32))
        records.append(record); print(f"dictionary site {site_index + 1}/{len(sites)} {site['feature']} {site['selection']}", flush=True)
    arrays = {}; summaries = {}
    for block in config["blocks"]:
        summaries[str(block)] = {}
        for objective in objectives_used:
            summaries[str(block)][objective] = {}
            for family in ("incoming", "response"):
                array = np.stack(vectors[objective][family][str(block)]); arrays[f"{objective}_{family}_block{block}"] = array
                summaries[str(block)][objective][family] = {"prototype": prototype_metrics(array, labels), "recurrence": recurrence_metrics(array, labels, positions)}
    np.savez_compressed(ARRAYS, labels=labels, positions=positions, **arrays)
    hypotheses = {
        "D0_feature_coverage": {"supported": len(competent) >= 25},
        "D1_late_semantic_dictionary": {"supported": all(summaries[str(block)]["semantic"]["response"]["prototype"]["top1"] >= .25 for block in (12, 17))},
        "D2_semantics_beat_random_objectives": {"supported": all(summaries[str(block)]["semantic"]["response"]["prototype"]["top1"] >= 2 * summaries[str(block)]["random"]["response"]["prototype"]["top1"] for block in (12, 17))},
        "D3_semantic_recurrence_beats_random": {"supported": all(summaries[str(block)]["semantic"]["response"]["recurrence"]["same_to_different_ratio"] >= 1.25 * summaries[str(block)]["random"]["response"]["recurrence"]["same_to_different_ratio"] for block in (12, 17))},
        "D4_late_directions_are_cleaner": {"supported": summaries["17"]["semantic"]["response"]["prototype"]["top1"] - summaries["1"]["semantic"]["response"]["prototype"]["top1"] >= .10},
        "D5_input_conditioning_remains": {"supported": any(summaries[str(block)]["semantic"]["incoming"]["prototype"]["top1"] - summaries[str(block)]["semantic"]["response"]["prototype"]["top1"] >= .10 for block in (8, 12, 17))},
    }
    gram_relative_error = float(np.linalg.norm(semantic_contrasts_np.T @ semantic_contrasts_np - shared_permutation_np.T @ shared_permutation_np) / np.linalg.norm(semantic_contrasts_np.T @ semantic_contrasts_np))
    posthoc = {"shared_permutation_gram_relative_error": gram_relative_error, "semantic_response_top1_exceeds_shared_permutation": {str(block): summaries[str(block)]["semantic"]["response"]["prototype"]["top1"] > summaries[str(block)]["shared_permutation"]["response"]["prototype"]["top1"] for block in config["blocks"]}, "semantic_response_recurrence_exceeds_shared_permutation": {str(block): summaries[str(block)]["semantic"]["response"]["recurrence"]["same_to_different_ratio"] > summaries[str(block)]["shared_permutation"]["response"]["recurrence"]["same_to_different_ratio"] for block in config["blocks"]}}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "posthoc_gram_matched_control": posthoc, "competent_features": competent, "site_count": len(sites), "summaries": summaries, "records": records, "arrays": str(ARRAYS)}, indent=2, allow_nan=False))
    lines = ["# Competent-feature direction dictionary", "", f"This held-out audit uses {len(competent)} of 50 predeclared token properties that passed the prior model-competence gate, with four FineWeb contexts per feature. Each semantic output contrast is compared with a vocabulary-permuted random contrast having the same positive/negative mass.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "## Post-hoc Gram-matched red team", "", f"A single shared vocabulary permutation preserves the complete 32-objective Gram matrix to relative error `{gram_relative_error:.2e}` while breaking the association between labels and token identities.", "", "![Direction dictionary](bilin18_competent_feature_direction_dictionary.png)", "", "| Block | Semantic response top1/top5 | Independent/shared-permutation top1 | Semantic response recurrence | Independent/shared-permutation recurrence | Semantic incoming top1 |", "|---:|---:|---:|---:|---:|---:|"]
    for block in config["blocks"]:
        row = summaries[str(block)]; sp = row["semantic"]["response"]["prototype"]; rp = row["random"]["response"]["prototype"]; gp = row["shared_permutation"]["response"]["prototype"]; sr = row["semantic"]["response"]["recurrence"]; rr = row["random"]["response"]["recurrence"]; gr = row["shared_permutation"]["response"]["recurrence"]; ip = row["semantic"]["incoming"]["prototype"]
        lines.append(f"| {block} | {sp['top1']:.3f}/{sp['top5']:.3f} | {rp['top1']:.3f}/{gp['top1']:.3f} | {sr['same_to_different_ratio']:.2f} | {rr['same_to_different_ratio']:.2f}/{gr['same_to_different_ratio']:.2f} | {ip['top1']:.3f} |")
    lines += ["", "## Guardrails", "", "The labels overlap semantically and lexically, so 32-way top-1 accuracy is deliberately strict. The independent random-objective control preserves each contrast's balance but not inter-feature overlap. The shared-permutation control preserves both. Site selection and the two null generators use separate deterministic streams, so adding a null cannot change the evaluation sites. One draw of each null is still insufficient for a precise null distribution. These are output-defined supervised features, not a claim about the model's complete native ontology."]
    REPORT.write_text("\n".join(lines) + "\n")
    x = np.arange(len(config["blocks"])); fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for objective in objectives_used: axes[0].plot(x, [summaries[str(block)][objective]["response"]["prototype"]["top1"] for block in config["blocks"]], marker="o", label=objective)
    axes[0].axhline(1 / len(competent), color="black", linestyle="--", linewidth=1); axes[0].set_xticks(x, config["blocks"]); axes[0].set(xlabel="Block", ylabel="Leave-one-out top-1", title=f"{len(competent)}-way response direction dictionary"); axes[0].legend(); axes[0].grid(alpha=.2)
    for objective in objectives_used: axes[1].plot(x, [summaries[str(block)][objective]["response"]["recurrence"]["same_to_different_ratio"] for block in config["blocks"]], marker="o", label=objective)
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1); axes[1].set_xticks(x, config["blocks"]); axes[1].set(xlabel="Block", ylabel="Same/different squared-cosine ratio", title="Feature recurrence"); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "site_count": len(sites), "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
