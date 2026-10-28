#!/usr/bin/env python3
"""Measure readable-variable count versus six-layer causal fidelity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_model_derived_source_ontology import weighted_nmi
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_readable_source_grammar import fit_readable_tree, readable_predicates
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate, select_supports


CONFIG = Path("configs/bilin18_readable_grammar_size_frontier.json")
PROFILE_SOURCE = TENSOR_ROOT / "bilin18_causal_source_profiles.pt"
CAUSAL_SOURCE = TENSOR_ROOT / "bilin18_causal_source_ontology.json"
OUTPUT = TENSOR_ROOT / "bilin18_readable_grammar_size_frontier.json"
REPORT = TENSOR_ROOT / "BILIN18_READABLE_GRAMMAR_SIZE_FRONTIER.md"
PARTIAL = TENSOR_ROOT / "bilin18_readable_grammar_size_frontier.partial.json"


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]).astype(np.float32)
    profiles = torch.load(PROFILE_SOURCE, map_location="cpu", weights_only=True)
    frequency = (profiles["a"]["counts"] + profiles["b"]["counts"]).numpy()
    names, predicate_array = readable_predicates(tokenizer, flags, frequency)
    predicates = torch.as_tensor(predicate_array, device="cuda")
    assignments, trees, stability = {}, {}, {}
    for leaves in config["leaf_counts"]:
        assignments[str(leaves)] = {}
        for split in ("a", "b"):
            print("fit readable frontier", leaves, split, flush=True)
            assignment, split_rows, rules = fit_readable_tree(
                profiles[split]["routing"].float().to("cuda"), profiles[split]["counts"].float().to("cuda"), predicates, names,
                leaves, config["minimum_leaf_mass_fraction"],
            )
            assignments[str(leaves)][split] = assignment
            trees[f"{leaves}_{split}"] = {"splits": split_rows, "rules": rules}
        stability[str(leaves)] = weighted_nmi(assignments[str(leaves)]["a"], assignments[str(leaves)]["b"], frequency, leaves)
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]], dtype=np.int64)
    evaluation = np.asarray(corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; maximum_error = 0.0
    saved = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else {}
    if saved.get("config") == config: results = saved.get("results", {}); maximum_error = float(saved.get("maximum_error", 0))
    for leaves in config["leaf_counts"]:
        for split in ("a", "b"):
            name = f"{leaves}_{split}"
            if name in results: print("evaluate readable frontier", name, "(resumed)", flush=True); continue
            print("evaluate readable frontier", name, flush=True)
            assignment = assignments[str(leaves)][split]
            supports = select_supports(model, discovery, assignment, config)
            row, error = evaluate(model, evaluation, assignment, config, supports)
            results[name] = row["5"]; maximum_error = max(maximum_error, error)
            PARTIAL.write_text(json.dumps({"config": config, "maximum_error": maximum_error, "results": results}, indent=2))
    score = {name: row["methods"]["fine128"]["kl_recovery"] for name, row in results.items()}
    top1 = {name: row["methods"]["fine128"]["top1_agreement"] for name, row in results.items()}
    best = {str(leaves): max(score[f"{leaves}_a"], score[f"{leaves}_b"]) for leaves in config["leaf_counts"]}
    prior = json.loads(CAUSAL_SOURCE.read_text()); human = prior["results"]["human"]["methods"]["fine128"]["kl_recovery"]
    hypotheses = {
        "L1_larger_readable_grammar_matches_human": {"supported": max(best["32"], best["42"]) >= human - .01, "best_difference": max(best["32"], best["42"]) - human},
        "L2_more_readable_variables_improve_behavior": {"supported": best["42"] >= best["21"] + .02, "difference": best["42"] - best["21"]},
        "L3_larger_readable_grammar_is_prediction_useful": {"supported": max(top1[name] for name in top1 if name.startswith("32_") or name.startswith("42_")) >= .72},
        "L4_grammar_stability_persists": {"supported": all(stability[str(leaves)] >= .80 for leaves in config["leaf_counts"]), "stability": stability},
        "L5_frontier_has_diminishing_returns": {"supported": best["42"] - best["32"] < best["32"] - best["21"], "gain_21_32": best["32"] - best["21"], "gain_32_42": best["42"] - best["32"]},
        "L6_readable_grammar_does_not_artificially_surpass_human": {"supported": max(best.values()) <= human + .03},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for value in hypotheses.values(): value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "stability": stability, "trees": trees, "results": results}, indent=2, allow_nan=False)); PARTIAL.unlink(missing_ok=True)
    lines = ["# Readable grammar size frontier", "", "Readable routing grammars with 21, 32, and 42 leaf variables are fit independently on two profile panels. Every program still receives only 128 selected `(head, distance, variable)` predicates per layer, so increased variable vocabulary competes for the same execution budget.", "", f"Maximum native-branch error: `{maximum_error:.2e}`. Human-21 recovery reference: `{human:.4f}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Leaves | Split A recovery/top1 | Split B recovery/top1 | Assignment NMI |", "|---:|---:|---:|---:|"]
    for leaves in config["leaf_counts"]:
        a=results[f"{leaves}_a"]["methods"]["fine128"]; b=results[f"{leaves}_b"]["methods"]["fine128"]
        lines.append(f"| {leaves} | {a['kl_recovery']:.4f}/{a['top1_agreement']:.3f} | {b['kl_recovery']:.4f}/{b['top1_agreement']:.3f} | {stability[str(leaves)]:.3f} |")
    lines += ["", "## Guardrails", "", "The route budget is fixed, but selecting among more possible types increases support-description cost. Trees optimize routing-profile variance rather than output KL, and split B can choose different predicates. This frontier tests whether readable variable vocabulary size was the main limitation; it does not optimize a full end-to-end MDL objective."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "best": best, "top1": top1}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
