#!/usr/bin/env python3
"""Validate frozen semantic labels for natural operator invocations."""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_natural_operator_semantic_validation.json")
OUTPUT = TENSOR_ROOT / "bilin18_natural_operator_semantic_validation.json"
REPORT = TENSOR_ROOT / "BILIN18_NATURAL_OPERATOR_SEMANTIC_VALIDATION.md"


def normalized(text):
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def capitalized(text):
    text = text.strip()
    return bool(text) and text[0].isupper() and any(character.isalpha() for character in text)


def role_profile(role, auxiliaries):
    examples = role["examples"]
    count = max(len(examples), 1)
    auxiliary_rate = sum(normalized(row["next_token"]) in auxiliaries for row in examples) / count
    local_rate = sum(row["distance"] <= 2 for row in examples) / count
    echo_rate = sum(
        any(normalized(output) == normalized(row["source_token"]) and normalized(output) for output in row["output_positive"])
        for row in examples
    ) / count
    capitalized_output_fraction = sum(capitalized(output) for row in examples for output in row["output_positive"]) / max(sum(len(row["output_positive"]) for row in examples), 1)
    copy_rate = sum(row["source_equals_next"] for row in examples) / count
    median_distance = float(np.median([row["distance"] for row in examples]))
    return {
        "auxiliary_rate": float(auxiliary_rate), "local_rate": float(local_rate),
        "source_echo_rate": float(echo_rate), "capitalized_output_fraction": float(capitalized_output_fraction),
        "copy_rate": float(copy_rate), "median_distance": median_distance,
    }


def profile_vector(profile):
    return np.asarray([
        profile["auxiliary_rate"], profile["local_rate"], profile["source_echo_rate"],
        profile["capitalized_output_fraction"], profile["copy_rate"],
        np.log1p(profile["median_distance"]) / np.log(128),
    ], dtype=np.float64)


def cosine(first, second):
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def profiles(payload, auxiliaries):
    return {
        f"{family}:{role['role']}": role_profile(role, auxiliaries)
        for family, row in payload["results"].items() for role in row["roles"]
    }


def main():
    config = json.loads(CONFIG.read_text()); discovery = json.loads(Path(config["discovery"]).read_text()); heldout = json.loads(Path(config["heldout"]).read_text())
    auxiliaries = set(config["auxiliary_tokens"]); discovery_profiles = profiles(discovery, auxiliaries); heldout_profiles = profiles(heldout, auxiliaries)
    other_aux = float(np.mean([value["auxiliary_rate"] for key, value in heldout_profiles.items() if not key.startswith("agreement_attractor:")]))
    agreement = heldout_profiles["agreement_attractor:0"]
    ioi = heldout_profiles["ioi:0"]; capital = heldout_profiles["capital_recall:0"]
    repetition = [heldout_profiles["repeated_sequence:0"], heldout_profiles["repeated_sequence:1"]]
    transfer = {key: cosine(profile_vector(discovery_profiles[key]), profile_vector(heldout_profiles[key])) for key in heldout_profiles}
    hypotheses = {
        "V0_agreement_route_predicts_auxiliaries": {"supported": agreement["auxiliary_rate"] >= .15 and agreement["auxiliary_rate"] >= 3 * other_aux, "agreement_rate": agreement["auxiliary_rate"], "other_rate": other_aux, "ratio": agreement["auxiliary_rate"] / max(other_aux, 1e-30)},
        "V1_ioi_route_is_local_name_valued_without_echo": {"supported": ioi["local_rate"] >= .6 and ioi["capitalized_output_fraction"] >= .6 and ioi["source_echo_rate"] <= .1, "profile": ioi},
        "V2_capital_route_is_source_linked_entity_lexicalization": {"supported": capital["source_echo_rate"] >= .7 and capital["capitalized_output_fraction"] >= .5, "profile": capital},
        "V3_repetition_route_is_long_range_copy": {"supported": all(row["source_echo_rate"] >= .7 and row["copy_rate"] >= .3 and row["median_distance"] >= 4 for row in repetition), "profiles": repetition},
        "V4_semantic_profiles_transfer": {"supported": all(value >= .95 for value in transfer.values()), "cosines": transfer},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "discovery_profiles": discovery_profiles, "heldout_profiles": heldout_profiles, "profile_cosines": transfer}, indent=2, allow_nan=False))
    lines = ["# Held-out semantic validation of natural operator invocations", "", "The semantic predicates below were frozen after inspecting the first 512-row FineWeb atlas and before scanning the disjoint held-out rows. They describe broader natural functions, not the names of the synthetic tasks that originally exposed each route template.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Route role | Aux next | Local d≤2 | Source echo | Capitalized outputs | Copy-next | Median distance | Profile cosine |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in heldout_profiles.items():
        lines.append(f"| {key} | {row['auxiliary_rate']:.3f} | {row['local_rate']:.3f} | {row['source_echo_rate']:.3f} | {row['capitalized_output_fraction']:.3f} | {row['copy_rate']:.3f} | {row['median_distance']:.1f} | {transfer[key]:.3f} |")
    lines += ["", "## Interpretation boundary", "", "These are operational semantic profiles of high-scoring invocations. They do not show that every use of a head has the same meaning, and they do not yet show that the route templates exhaust each natural function. The validated objects are broader operators—agreement-like auxiliary control, local name-valued association, source-linked entity lexicalization, and long-range copy—not one concept per head."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
