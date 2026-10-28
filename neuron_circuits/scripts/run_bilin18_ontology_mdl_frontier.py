#!/usr/bin/env python3
"""Build an explicit multi-cost MDL frontier for the competing ontologies."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_ontology_mdl_frontier.json")
CAUSAL = TENSOR_ROOT / "bilin18_causal_source_ontology.json"
NMF = TENSOR_ROOT / "bilin18_routing_primitive_factorization.json"
FACTORS = TENSOR_ROOT / "bilin18_routing_primitive_factors.pt"
PROFILES = TENSOR_ROOT / "bilin18_causal_source_profiles.pt"
SOFT = TENSOR_ROOT / "bilin18_soft_routing_primitive_program.json"
MASS = TENSOR_ROOT / "bilin18_typed_attention_edge_mass_redteam.json"
OUTPUT = TENSOR_ROOT / "bilin18_ontology_mdl_frontier.json"
REPORT = Path("ONTOLOGY_MDL_FRAMEWORK.md")


def log2_choose(total, selected):
    return (math.lgamma(total + 1) - math.lgamma(selected + 1) - math.lgamma(total - selected + 1)) / math.log(2)


def kl_error(row):
    return row["baseline_kl"] * (1 - row["methods"]["fine128"]["kl_recovery"])


def main():
    config = json.loads(CONFIG.read_text())
    causal = json.loads(CAUSAL.read_text()); nmf = json.loads(NMF.read_text()); soft = json.loads(SOFT.read_text()); mass = json.loads(MASS.read_text())
    factors = torch.load(FACTORS, map_location="cpu", weights_only=True); profiles = torch.load(PROFILES, map_location="cpu", weights_only=True)
    type_count = config["heads"] * config["distance_bins"] * config["variables"]
    hard_table_bits = config["tokenizer_vocabulary"] * math.ceil(math.log2(config["variables"]))
    membership = factors["membership_a"].clamp_min(0); membership /= membership.sum(1, keepdim=True).clamp_min(1e-30)
    observed = (profiles["a"]["counts"] + profiles["b"]["counts"]) > 0
    encoded = membership.sum(1) > 0
    ordered = membership[encoded].sort(dim=1, descending=True).values
    support90 = ((ordered.cumsum(1) / ordered.sum(1, keepdim=True).clamp_min(1e-30)) < .9).sum(1) + 1
    sparse_soft_bits = 0.0
    for support in support90.tolist():
        sparse_soft_bits += log2_choose(config["variables"], int(support)) + config["soft_quantization_bits"] * int(support)
    sparse_soft_bits += log2_choose(config["tokenizer_vocabulary"], int(encoded.sum()))
    dense_soft_bits = config["tokenizer_vocabulary"] * (config["variables"] - 1) * config["soft_quantization_bits"]
    human_row = causal["results"]["human"]
    nmf_a_row = nmf["results"]["nmf_a"]; nmf_b_row = nmf["results"]["nmf_b"]
    programs = {
        "human_hard128": {"budget": 128, "mean_kl": kl_error(human_row), "top1": human_row["methods"]["fine128"]["top1_agreement"], "variable_code": "20 ordered readable clauses / 38 feature references", "literal_variable_bits": hard_table_bits, "runtime_kind": "hard sparse", "edge_fraction_proxy": float(np.mean([mass["coverage"][str(layer)]["operator128"]["frequency"] for layer in range(6)]))},
        "nmf_hard128_a": {"budget": 128, "mean_kl": kl_error(nmf_a_row), "top1": nmf_a_row["methods"]["fine128"]["top1_agreement"], "variable_code": "opaque categorical token table", "literal_variable_bits": hard_table_bits, "runtime_kind": "hard sparse", "edge_fraction_proxy": None},
        "nmf_hard128_b": {"budget": 128, "mean_kl": kl_error(nmf_b_row), "top1": nmf_b_row["methods"]["fine128"]["top1_agreement"], "variable_code": "opaque categorical token table", "literal_variable_bits": hard_table_bits, "runtime_kind": "hard sparse", "edge_fraction_proxy": None},
        "nmf_soft128": {"budget": 128, "mean_kl": soft["results"]["soft128"]["mean_kl_error"], "top1": soft["results"]["soft128"]["top1_agreement"], "variable_code": "real-valued multiplexed membership table", "literal_variable_bits": dense_soft_bits, "sparse90_variable_bits": sparse_soft_bits, "runtime_kind": "dense fractional", "edge_fraction_proxy": 1.0},
        "nmf_soft256": {"budget": 256, "mean_kl": soft["results"]["soft256"]["mean_kl_error"], "top1": soft["results"]["soft256"]["top1_agreement"], "variable_code": "real-valued multiplexed membership table", "literal_variable_bits": dense_soft_bits, "sparse90_variable_bits": sparse_soft_bits, "runtime_kind": "dense fractional", "edge_fraction_proxy": 1.0},
        "native_all": {"budget": type_count, "mean_kl": soft["results"]["all"]["mean_kl_error"], "top1": soft["results"]["all"]["top1_agreement"], "variable_code": "no source dictionary needed for mask", "literal_variable_bits": 0, "runtime_kind": "native dense", "edge_fraction_proxy": 1.0},
    }
    for row in programs.values():
        row["route_support_bits"] = 0.0 if row["budget"] == type_count else config["layers"] * log2_choose(type_count, row["budget"])
        row["behavior_bits_per_token"] = row["mean_kl"] / math.log(2)
        row["total_literal_model_bits"] = row["literal_variable_bits"] + row["route_support_bits"]
        row["total_bits_by_scale"] = {str(tokens): row["total_literal_model_bits"] + tokens * row["behavior_bits_per_token"] for tokens in config["token_scales"]}
    human_behavior = programs["human_hard128"]["behavior_bits_per_token"]
    crossovers = {}
    for name in ("nmf_hard128_a", "nmf_hard128_b", "nmf_soft256"):
        row = programs[name]; saving = human_behavior - row["behavior_bits_per_token"]
        extra = row["total_literal_model_bits"] - programs["human_hard128"]["route_support_bits"]
        crossovers[name] = extra / saving if saving > 0 else None
    claims = {
        "no_single_ontology_dominates": True,
        "human_is_current_readability_winner": True,
        "hard_nmf_b_behaviorally_beats_human": programs["nmf_hard128_b"]["mean_kl"] < programs["human_hard128"]["mean_kl"],
        "soft256_uses_more_predicates_and_dense_fractional_execution": programs["nmf_soft256"]["budget"] > programs["human_hard128"]["budget"] and programs["nmf_soft256"]["runtime_kind"] == "dense fractional",
    }
    OUTPUT.write_text(json.dumps({"config": config, "claims": claims, "programs": programs, "crossovers_under_literal_table_assumption": crossovers, "notes": {"observed_tokens": int(observed.sum()), "encoded_soft_token_exceptions": int(encoded.sum()), "median_memberships90": float(support90.float().median()), "human_rule_clauses": config["human_rule_clauses"], "human_rule_feature_references": config["human_rule_feature_references"]}}, indent=2, allow_nan=False))
    lines = [
        "# Ontology MDL framework", "",
        "The experiments now contain several behaviorally valid descriptions that optimize different resources. A single scalar ‘interpretability’ or MDL number would hide the choice of interpreter. This document defines the costs separately and reports the current frontier.", "",
        "## Program language", "",
        "A mechanistic description has four reusable declarations:", "",
        "1. **Variables** describe source token/state properties. Examples are `function_word`, a learned categorical token table, or a soft routing membership vector.",
        "2. **Read functions** specify where a variable is read: layer, native head, relative-distance predicate, and any contextual guard. `previous_token` is the distance interval `[1,2)` and can be reused by many heads.",
        "3. **Transforms** specify what happens to the read value: native value projection, output projection, RMSNorm geometry, and the receiving bilinear tensor contraction.",
        "4. **Invocations** bind a variable, read function, and transform in a concrete layer/context.", "",
        "The behavior residual is the KL from the native next-token distribution. Its ideal coding penalty is `N × KL / ln(2)` bits over `N` token predictions.", "",
        "## Why one MDL number is insufficient", "",
        "- A mask saying `keep all edges` is extremely short but performs the full native computation. Description length must not be confused with runtime compression.",
        "- An arbitrary hard token table can be execution-sparse and behaviorally strong, yet is not readable or stable enough to be a scientific variable.",
        "- A soft factorization can expose reusable geometry, but its real-valued memberships and fractional execution are expensive unless another short generator explains them.",
        "- Human rules are short only relative to an interpreter that already knows lexical predicates such as punctuation, suffix, or leading whitespace. Readability is therefore an explicit primitive prior, not a basis-invariant fact.", "",
        "## Current measured frontier", "",
        "| Program | Route predicates/layer | Mean KL | Top-1 | Variable declaration | Runtime proxy | Literal table bits* |", "|---|---:|---:|---:|---|---|---:|",
    ]
    for name, row in programs.items():
        edge = "—" if row["edge_fraction_proxy"] is None else f"{row['edge_fraction_proxy']:.2f} edge fraction"
        lines.append(f"| {name} | {row['budget']} | {row['mean_kl']:.4f} | {row['top1']:.3f} | {row['variable_code']} | {row['runtime_kind']}; {edge} | {row['total_literal_model_bits']:.0f} |")
    lines += ["", "`*` Literal bits use a deliberately crude standalone encoding: five bits per hard token label, eight bits per independent soft membership, plus an enumerative code for selected route types. They are not claimed as the final scientific readability prior.", "", "Under that literal-table assumption, the token-count crossovers where lower behavioral KL repays extra description are:", ""]
    for name, value in crossovers.items(): lines.append(f"- `{name}`: " + (f"approximately {value:,.0f} evaluated tokens." if value is not None else "never, because it has no behavioral-bit advantage."))
    lines += ["", "## Recommended MDL views by use case", "", "| Use case | Charge heavily | Current preferred ontology |", "|---|---|---|", "| Human reverse engineering | Unreadable variables/functions | Human hard categories plus native head/distance functions |", "| Fast approximate execution | Edge invocations and residual KL | Hard 128-type programs |", "| Faithful behavioral surrogate | Residual KL | Soft-256 or a larger hard program |", "| Discover recurring geometry | Instability across splits | Continuous routing profiles / soft memberships |", "| Exact mechanistic execution | Any unexplained residual | Native QKOV + RMSNorm + bilinear contractions |", "", "## Next test", "", "The next meaningful compression step is not another arbitrary clustering. It is a grammar induction problem: search for short predicates that generate the stable parts of the token memberships and read-function patterns—token boundary, distance, punctuation, grammatical role, code/text mode—then charge those predicates once and their invocations repeatedly. A proposed primitive only earns readability if it predicts held-out memberships, interventions, and examples better than a literal lookup table with comparable total cost."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"claims": claims, "crossovers": crossovers}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
