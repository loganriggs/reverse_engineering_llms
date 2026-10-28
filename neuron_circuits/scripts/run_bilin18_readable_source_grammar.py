#!/usr/bin/env python3
"""Induce readable source-variable rules directly from routing profiles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_model_derived_source_ontology import weighted_nmi
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate, select_supports


CONFIG = Path("configs/bilin18_readable_source_grammar.json")
PROFILE_SOURCE = TENSOR_ROOT / "bilin18_causal_source_profiles.pt"
CAUSAL_SOURCE = TENSOR_ROOT / "bilin18_causal_source_ontology.json"
OUTPUT = TENSOR_ROOT / "bilin18_readable_source_grammar.json"
ASSIGNMENTS = TENSOR_ROOT / "bilin18_readable_source_grammar_assignments.npz"
REPORT = TENSOR_ROOT / "BILIN18_READABLE_SOURCE_GRAMMAR.md"


def readable_predicates(tokenizer, flags, frequencies):
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    columns = [flags[:, index].astype(bool) for index in range(len(FEATURES))]
    names = [f"known:{name}" for name in FEATURES]
    new = {
        "lex:leading_space": [text.startswith(" ") for text in texts],
        "lex:space_word": [text.startswith(" ") and text.strip().isalpha() for text in texts],
        "lex:continuation_piece": [(not text.startswith(" ")) and text.isalpha() for text in texts],
        "lex:replacement_byte": ["�" in text for text in texts],
        "lex:starts_upper": [bool(text.strip()) and text.strip()[0].isupper() for text in texts],
    }
    for name, values in new.items(): names.append(name); columns.append(np.asarray(values, dtype=bool))
    lengths = np.asarray([len(text) for text in texts])
    for threshold in (1, 2, 3, 4, 6, 8, 12):
        names.append(f"lex:length>={threshold}"); columns.append(lengths >= threshold)
    positive = frequencies[frequencies > 0]
    for quantile in (.25, .5, .75, .9):
        threshold = float(np.quantile(positive, quantile)) if len(positive) else 0
        names.append(f"corpus:frequency>={threshold:g}"); columns.append(frequencies >= threshold)
    return names, np.stack(columns, axis=1)


def best_split(indices, matrix, weights, predicates, minimum_mass):
    node_weights = weights[indices]
    total_mass = node_weights.sum()
    if total_mass < 2 * minimum_mass: return None
    weighted_matrix = matrix[indices] * node_weights[:, None]
    total_sum = weighted_matrix.sum(0)
    choices = predicates[indices].float()
    left_mass = choices.T @ node_weights
    left_sum = choices.T @ weighted_matrix
    right_mass = total_mass - left_mass; right_sum = total_sum[None] - left_sum
    valid = (left_mass >= minimum_mass) & (right_mass >= minimum_mass)
    gain = left_sum.square().sum(1) / left_mass.clamp_min(1e-30) + right_sum.square().sum(1) / right_mass.clamp_min(1e-30) - total_sum.square().sum() / total_mass.clamp_min(1e-30)
    gain[~valid] = -torch.inf
    feature = int(gain.argmax())
    if not torch.isfinite(gain[feature]): return None
    return {"feature": feature, "gain": float(gain[feature])}


def fit_readable_tree(matrix, weights, predicates, names, leaves, minimum_fraction):
    device = matrix.device; predicates = predicates.to(device); weights = weights.to(device)
    nodes = {0: {"indices": torch.where(weights > 0)[0], "path": []}}
    assignments = torch.zeros(len(matrix), dtype=torch.long, device=device)
    next_id = 1; splits = []
    minimum_mass = float(weights.sum()) * minimum_fraction
    while len(nodes) < leaves:
        candidates = []
        for node_id, node in nodes.items():
            candidate = best_split(node["indices"], matrix, weights, predicates, minimum_mass)
            if candidate is not None: candidates.append((candidate["gain"], node_id, candidate["feature"]))
        if not candidates: break
        gain, node_id, feature = max(candidates)
        node = nodes.pop(node_id); indices = node["indices"]; positive = predicates[indices, feature]
        left_id, right_id = next_id, next_id + 1; next_id += 2
        nodes[left_id] = {"indices": indices[~positive], "path": node["path"] + [(feature, False)]}
        nodes[right_id] = {"indices": indices[positive], "path": node["path"] + [(feature, True)]}
        splits.append({"node": node_id, "predicate": names[feature], "gain": gain, "false_child": left_id, "true_child": right_id})
    ordered = sorted(nodes.items())
    default_leaf = max(ordered, key=lambda item: float(weights[item[1]["indices"]].sum()))[0]
    for category, (node_id, node) in enumerate(ordered): assignments[node["indices"]] = category; node["category"] = category
    unseen = weights == 0
    for index in torch.where(unseen)[0].tolist():
        matches = [(node_id, node) for node_id, node in ordered if all(bool(predicates[index, feature]) == value for feature, value in node["path"])]
        chosen = matches[0][1]["category"] if matches else nodes[default_leaf]["category"]
        assignments[index] = chosen
    rules = []
    for node_id, node in ordered:
        rule = " AND ".join(("" if value else "NOT ") + names[feature] for feature, value in node["path"]) or "TRUE"
        rules.append({"category": node["category"], "rule": rule, "weighted_mass": float(weights[node["indices"]].sum() / weights.sum())})
    return assignments.cpu().numpy().astype(np.int64), splits, rules


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]).astype(np.float32)
    profiles = torch.load(PROFILE_SOURCE, map_location="cpu", weights_only=True)
    frequency = (profiles["a"]["counts"] + profiles["b"]["counts"]).numpy()
    names, predicate_array = readable_predicates(tokenizer, flags, frequency)
    predicates = torch.as_tensor(predicate_array, device="cuda")
    assignments = {}; tree_data = {}
    for split in ("a", "b"):
        print("fit readable grammar", split, flush=True)
        assignment, splits, rules = fit_readable_tree(
            profiles[split]["routing"].float().to("cuda"), profiles[split]["counts"].float().to("cuda"), predicates, names,
            config["leaves"], config["minimum_leaf_mass_fraction"],
        )
        assignments[split] = assignment; tree_data[split] = {"splits": splits, "rules": rules}
    np.savez_compressed(ASSIGNMENTS, **assignments)
    stability = weighted_nmi(assignments["a"], assignments["b"], frequency, config["leaves"])
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; maximum_error = 0.0
    for split in ("a", "b"):
        print("evaluate readable grammar", split, flush=True)
        supports = select_supports(model, discovery, assignments[split], config)
        row, error = evaluate(model, evaluation, assignments[split], config, supports)
        results[split] = row["5"]; maximum_error = max(maximum_error, error)
    score = {split: results[split]["methods"]["fine128"]["kl_recovery"] for split in results}
    top1 = {split: results[split]["methods"]["fine128"]["top1_agreement"] for split in results}
    prior = json.loads(CAUSAL_SOURCE.read_text()); human = prior["results"]["human"]["methods"]["fine128"]["kl_recovery"]; embedding = prior["results"]["embedding_exact"]["methods"]["fine128"]["kl_recovery"]
    new_splits = sum(not row["predicate"].startswith("known:") for split in tree_data.values() for row in split["splits"])
    hypotheses = {
        "G1_readable_grammar_is_causally_useful": {"supported": max(score.values()) >= .93},
        "G2_readable_grammar_is_competitive_with_human": {"supported": max(score.values()) >= human - .03, "best_difference": max(score.values()) - human},
        "G3_new_lexical_primitives_are_selected": {"supported": new_splits >= 3, "new_split_count_across_fits": new_splits},
        "G4_grammar_variables_are_split_stable": {"supported": stability >= .60, "weighted_nmi": stability},
        "G5_readable_grammar_is_prediction_useful": {"supported": max(top1.values()) >= .70},
        "G6_readable_grammar_beats_embedding_clusters": {"supported": max(score.values()) >= embedding + .05, "best_difference": max(score.values()) - embedding},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "predicate_names": names, "trees": tree_data, "results": results}, indent=2, allow_nan=False))
    lines = ["# Readable source-variable grammar", "", "A 21-leaf decision tree is fit directly to routing-profile variance. Every leaf is an executable source variable generated by a conjunction of preregistered readable predicates. Independent trees are fit on disjoint FineWeb profile panels and evaluated with the same 128 route types per layer.", "", f"Maximum native-branch error: `{maximum_error:.2e}`. Independent assignment NMI: `{stability:.3f}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Grammar | KL recovery | CE delta | Top-1 |", "|---|---:|---:|---:|"]
    for split in ("a", "b"):
        value = results[split]["methods"]["fine128"]; lines.append(f"| {split} | {value['kl_recovery']:.4f} | {value['mean_ce_delta']:+.4f} | {value['top1_agreement']:.4f} |")
    for split in ("a", "b"):
        lines += ["", f"## Grammar {split.upper()}", "", "Selected splits:", ""]
        for row in tree_data[split]["splits"]: lines.append(f"- `{row['predicate']}` (variance gain {row['gain']:.3g})")
        lines += ["", "Leaf variables:", ""]
        for row in tree_data[split]["rules"]: lines.append(f"- **V{row['category']}** ({row['weighted_mass']:.1%}): `{row['rule']}`")
    lines += ["", "## Guardrails", "", "The predicate language encodes human prior knowledge and is therefore not a label-free discovery method. Frequency predicates are corpus-relative. Greedy variance reduction is not guaranteed to find the shortest or behaviorally optimal tree, and the leaves are evaluated after a separate route-selection stage. Split stability and causal fidelity are both required: a readable rule that changes completely with the sample is not a reusable model variable."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "score": score, "top1": top1}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
