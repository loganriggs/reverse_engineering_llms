#!/usr/bin/env python3
"""Prospectively validate semantic profiles of task-agnostic route modes."""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_unseeded_route_semantic_validation.json")
OUTPUT = TENSOR_ROOT / "bilin18_unseeded_route_semantic_validation.json"
REPORT = TENSOR_ROOT / "BILIN18_UNSEEDED_ROUTE_SEMANTIC_VALIDATION.md"
FEATURE_NAMES = ("determiner", "preposition", "conjunction", "pronoun", "auxiliary", "modal", "punctuation", "opening_delimiter", "closing_delimiter", "quote", "numeric", "capitalized")


def normalized(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def profile(row):
    examples = row["examples"]
    source = np.stack([vocabulary_features(item["source_token"]) for item in examples])
    outputs = np.stack([vocabulary_features(token) for item in examples for token in item["output_positive"]])
    values = {
        "source_echo_rate": float(np.mean([any(normalized(token) == normalized(item["source_token"]) and normalized(token) for token in item["output_positive"]) for item in examples])),
        "copy_rate": float(np.mean([item["source_equals_next"] for item in examples])),
        "self_rate": float(np.mean([item["source_equals_destination"] for item in examples])),
        "eot_source_rate": float(np.mean([item["source_token"] == "<|endoftext|>" for item in examples])),
        "median_distance": float(np.median([item["distance"] for item in examples])),
        "median_route_cosine": float(np.median([item["route_cosine"] for item in examples])),
    }
    for name in FEATURE_NAMES:
        index = FEATURES.index(name)
        values[f"source_{name}"] = float(source[:, index].mean())
        values[f"output_{name}"] = float(outputs[:, index].mean())
    return values


def profile_vector(row):
    keys = ["source_echo_rate", "copy_rate", "self_rate", "eot_source_rate"]
    keys += [f"source_{name}" for name in FEATURE_NAMES]
    keys += [f"output_{name}" for name in FEATURE_NAMES]
    vector = [row[key] for key in keys]
    vector += [np.log1p(row["median_distance"]) / np.log(128), row["median_route_cosine"]]
    return np.asarray(vector, dtype=np.float64)


def cosine(first, second):
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def main():
    config = json.loads(CONFIG.read_text()); discovery = json.loads(Path(config["discovery"]).read_text()); heldout = json.loads(Path(config["heldout"]).read_text())
    first = {row["name"]: profile(row) for row in discovery["results"]}; second = {row["name"]: profile(row) for row in heldout["results"]}
    cosines = {name: cosine(profile_vector(first[name]), profile_vector(second[name])) for name in first}
    checks = {
        "S0_L17C1_capitalized_output_mode": second["L17C1"]["output_capitalized"] >= .60 and second["L17C1"]["source_echo_rate"] <= .10,
        "S1_L2C0_current_entity_to_syntax": second["L2C0"]["self_rate"] >= .90 and second["L2C0"]["source_capitalized"] >= .35 and second["L2C0"]["output_capitalized"] <= .15,
        "S2_L6C2_document_boundary_formatting": second["L6C2"]["eot_source_rate"] >= .25 and second["L6C2"]["output_punctuation"] >= .15,
        "S3_L8C1_short_range_lexical_echo": second["L8C1"]["source_echo_rate"] >= .70 and second["L8C1"]["median_distance"] <= 4,
        "S4_L11C0_determiner_led_association": second["L11C0"]["source_determiner"] >= .40 and second["L11C0"]["source_echo_rate"] <= .10,
        "S5_L8C2_long_range_lexical_echo": second["L8C2"]["source_echo_rate"] >= .80 and second["L8C2"]["copy_rate"] >= .15 and second["L8C2"]["median_distance"] >= 8,
        "S6_L13C0_punctuation_closure": second["L13C0"]["source_punctuation"] >= .50 and second["L13C0"]["output_punctuation"] >= .50,
        "S7_L4C0_preposition_led_relation": second["L4C0"]["source_preposition"] >= .30 and second["L4C0"]["source_echo_rate"] <= .10,
        "S8_L14C1_name_valued_association": second["L14C1"]["output_capitalized"] >= .35 and second["L14C1"]["source_echo_rate"] <= .10 and second["L14C1"]["median_distance"] <= 3,
        "S9_L1C2_current_capitalized_gate": second["L1C2"]["self_rate"] >= .90 and second["L1C2"]["source_capitalized"] >= .60,
    }
    hypothesis_candidates = {
        "S0_L17C1_capitalized_output_mode": "L17C1",
        "S1_L2C0_current_entity_to_syntax": "L2C0",
        "S2_L6C2_document_boundary_formatting": "L6C2",
        "S3_L8C1_short_range_lexical_echo": "L8C1",
        "S4_L11C0_determiner_led_association": "L11C0",
        "S5_L8C2_long_range_lexical_echo": "L8C2",
        "S6_L13C0_punctuation_closure": "L13C0",
        "S7_L4C0_preposition_led_relation": "L4C0",
        "S8_L14C1_name_valued_association": "L14C1",
        "S9_L1C2_current_capitalized_gate": "L1C2",
    }
    passed = [name for name, value in checks.items() if value]
    candidates = tuple(hypothesis_candidates.values())
    profile_pass = [candidate for candidate in candidates if cosines[candidate] >= .95]
    hypotheses = {
        name: {
            "supported": bool(value),
            "candidate": hypothesis_candidates[name],
            "heldout_profile": second[hypothesis_candidates[name]],
        }
        for name, value in checks.items()
    }
    hypotheses["S10_task_agnostic_labels_validate_broadly"] = {"supported": len(passed) >= 7, "passed": passed, "count": len(passed)}
    hypotheses["S11_full_profiles_transfer"] = {"supported": len(profile_pass) >= 8, "passing_candidates": profile_pass, "cosines": {name: cosines[name] for name in candidates}}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "discovery_profiles": first, "heldout_profiles": second, "profile_cosines": cosines}, indent=2, allow_nan=False))
    lines = ["# Held-out semantics of task-agnostic route modes", "", "Ten candidate descriptions were frozen after a fresh semantic-discovery panel. The candidate geometry itself was selected earlier from two token-blind panels. This report tests the named operational predicates on another disjoint FineWeb panel.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**" for name, row in hypotheses.items())
    lines += ["", "| Candidate | Echo | Copy | Self | EOT source | Determiner | Preposition | Source punct. | Output punct. | Source cap. | Output cap. | Distance | Profile cosine |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in candidates:
        row = second[name]
        lines.append(f"| {name} | {row['source_echo_rate']:.3f} | {row['copy_rate']:.3f} | {row['self_rate']:.3f} | {row['eot_source_rate']:.3f} | {row['source_determiner']:.3f} | {row['source_preposition']:.3f} | {row['source_punctuation']:.3f} | {row['output_punctuation']:.3f} | {row['source_capitalized']:.3f} | {row['output_capitalized']:.3f} | {row['median_distance']:.1f} | {cosines[name]:.3f} |")
    lines += ["", "## Guardrail", "", "These labels describe the top high-magnitude invocations of a route mode, not every edge assigned to the cluster and not every use of its constituent heads. Failures remain unlabeled geometric modes; passing profiles still require finite causal tests before being called sufficient natural-language circuits."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps({name: {"supported": row["supported"]} for name, row in hypotheses.items()}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
