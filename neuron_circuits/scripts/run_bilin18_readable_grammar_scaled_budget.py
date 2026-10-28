#!/usr/bin/env python3
"""Test readable variable vocabularies with proportional route budgets."""

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


CONFIG = Path("configs/bilin18_readable_grammar_scaled_budget.json")
PROFILE_SOURCE = TENSOR_ROOT / "bilin18_causal_source_profiles.pt"
CAUSAL_SOURCE = TENSOR_ROOT / "bilin18_causal_source_ontology.json"
FIXED_SOURCE = TENSOR_ROOT / "bilin18_readable_grammar_size_frontier.json"
OUTPUT = TENSOR_ROOT / "bilin18_readable_grammar_scaled_budget.json"
REPORT = TENSOR_ROOT / "BILIN18_READABLE_GRAMMAR_SCALED_BUDGET.md"
PARTIAL = TENSOR_ROOT / "bilin18_readable_grammar_scaled_budget.partial.json"


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]).astype(np.float32)
    profiles = torch.load(PROFILE_SOURCE, map_location="cpu", weights_only=True)
    frequency = (profiles["a"]["counts"] + profiles["b"]["counts"]).numpy()
    names, predicate_array = readable_predicates(tokenizer, flags, frequency); predicates = torch.as_tensor(predicate_array, device="cuda")
    assignments, stability = {}, {}
    for leaves, _ in config["leaf_budget_pairs"]:
        assignments[str(leaves)] = {}
        for split in ("a", "b"):
            print("fit scaled grammar", leaves, split, flush=True)
            assignment, _, _ = fit_readable_tree(profiles[split]["routing"].float().to("cuda"), profiles[split]["counts"].float().to("cuda"), predicates, names, leaves, config["minimum_leaf_mass_fraction"])
            assignments[str(leaves)][split] = assignment
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
    for leaves, budget in config["leaf_budget_pairs"]:
        local_config = {**config, "methods": [f"fine{budget}"]}
        for split in ("a", "b"):
            name = f"{leaves}_{budget}_{split}"
            if name in results: print("evaluate scaled grammar", name, "(resumed)", flush=True); continue
            print("evaluate scaled grammar", name, flush=True)
            assignment = assignments[str(leaves)][split]; supports = select_supports(model, discovery, assignment, local_config)
            row, error = evaluate(model, evaluation, assignment, local_config, supports)
            results[name] = row["5"]; maximum_error = max(maximum_error, error)
            PARTIAL.write_text(json.dumps({"config": config, "maximum_error": maximum_error, "results": results}, indent=2))
    score = {}; top1 = {}
    for leaves, budget in config["leaf_budget_pairs"]:
        key=f"{leaves}_{budget}"; method=f"fine{budget}"
        score[key]=max(results[f"{key}_a"]["methods"][method]["kl_recovery"],results[f"{key}_b"]["methods"][method]["kl_recovery"])
        top1[key]=max(results[f"{key}_a"]["methods"][method]["top1_agreement"],results[f"{key}_b"]["methods"][method]["top1_agreement"])
    human=json.loads(CAUSAL_SOURCE.read_text())["results"]["human"]["methods"]["fine128"]["kl_recovery"]
    fixed=json.loads(FIXED_SOURCE.read_text())["results"]
    fixed_best={str(leaves):max(fixed[f"{leaves}_a"]["methods"]["fine128"]["kl_recovery"],fixed[f"{leaves}_b"]["methods"]["fine128"]["kl_recovery"]) for leaves,_ in config["leaf_budget_pairs"]}
    ordered=[score[f"{leaves}_{budget}"] for leaves,budget in config["leaf_budget_pairs"]]
    hypotheses={
        "B1_scaled_budget_closes_human_gap":{"supported":max(ordered[1:])>=human-.02,"best_difference":max(ordered[1:])-human},
        "B2_scaled_budget_restores_prediction_fidelity":{"supported":max(list(top1.values())[1:])>=.72},
        "B3_more_variables_help_at_proportional_budget":{"supported":all(a<b for a,b in zip(ordered,ordered[1:]))},
        "B4_grammar_remains_stable":{"supported":all(value>=.80 for value in stability.values()),"stability":stability},
        "B5_extra_variables_are_not_free":{"supported":score["32_192"]>=fixed_best["32"]+.10 and score["42_256"]>=fixed_best["42"]+.10,"gain32":score["32_192"]-fixed_best["32"],"gain42":score["42_256"]-fixed_best["42"]},
        "native_branch_matches_checkpoint":{"supported":maximum_error<=1e-4,"maximum_error":maximum_error},
    }
    for value in hypotheses.values(): value["supported"]=bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config":config,"hypotheses":hypotheses,"stability":stability,"fixed_budget_best":fixed_best,"results":results},indent=2,allow_nan=False)); PARTIAL.unlink(missing_ok=True)
    lines=["# Readable grammar with proportional route budgets","","Variable count and invocation budget are increased together: 21/128, 32/192, and 42/256. This tests whether larger readable grammars failed because they fragmented useful routes across more types.","",f"Maximum native-branch error: `{maximum_error:.2e}`. Human-21/128 recovery: `{human:.4f}`.","","## Locked hypotheses",""]
    for name,value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["","| Variables / predicates | Best scaled recovery | Best top-1 | Fixed-128 recovery | Assignment NMI |","|---|---:|---:|---:|---:|"]
    for leaves,budget in config["leaf_budget_pairs"]:
        key=f"{leaves}_{budget}"; lines.append(f"| {leaves} / {budget} | {score[key]:.4f} | {top1[key]:.3f} | {fixed_best[str(leaves)]:.4f} | {stability[str(leaves)]:.3f} |")
    lines += ["","## Guardrails","","Budgets are approximately proportional rather than MDL-optimal. More selected predicates increase both support-description and execution cost. The trees still optimize profile variance, not final KL. A positive result would establish a variable/invocation tradeoff, not that larger grammars are universally preferable."]
    REPORT.write_text("\n".join(lines)+"\n"); print(json.dumps({"hypotheses":hypotheses,"score":score,"top1":top1},indent=2)); print(REPORT)


if __name__ == "__main__": main()
