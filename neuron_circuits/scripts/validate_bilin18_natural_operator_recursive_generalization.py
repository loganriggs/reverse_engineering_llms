#!/usr/bin/env python3
"""Test broader labels on a third panel after narrow labels failed."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from validate_bilin18_natural_operator_semantics import (
    capitalized, cosine, normalized, profile_vector, profiles,
)


CONFIG = Path("configs/bilin18_natural_operator_recursive_validation.json")
OUTPUT = TENSOR_ROOT / "bilin18_natural_operator_recursive_validation.json"
REPORT = TENSOR_ROOT / "BILIN18_NATURAL_OPERATOR_RECURSIVE_VALIDATION.md"


def predicate_rate(role, lexicon):
    outputs = [output for row in role["examples"] for output in row["output_positive"]]
    return float(np.mean([normalized(output) in lexicon for output in outputs]))


def role_map(payload):
    return {f"{family}:{role['role']}": role for family, row in payload["results"].items() for role in row["roles"]}


def main():
    config = json.loads(CONFIG.read_text()); first = json.loads(Path(config["discovery"]).read_text()); second = json.loads(Path(config["heldout"]).read_text()); third = json.loads(Path(config["third"]).read_text())
    lexicon = set(config["predicate_lexicon"])
    # The generic profile uses the same frozen auxiliary list as the prior audit;
    # predicate-rate is an additional preregistered broader semantic coordinate.
    auxiliaries = set(config["predicate_lexicon"][:24])
    first_profiles, second_profiles, third_profiles = profiles(first, auxiliaries), profiles(second, auxiliaries), profiles(third, auxiliaries)
    third_roles = role_map(third)
    predicate_rates = {key: predicate_rate(role, lexicon) for key, role in third_roles.items()}
    other_predicate = float(np.mean([value for key, value in predicate_rates.items() if not key.startswith("agreement_attractor:")]))
    agreement = third_profiles["agreement_attractor:0"]; ioi = third_profiles["ioi:0"]; capital = third_profiles["capital_recall:0"]
    repetition = [third_profiles["repeated_sequence:0"], third_profiles["repeated_sequence:1"]]
    reference = {key: .5 * (profile_vector(first_profiles[key]) + profile_vector(second_profiles[key])) for key in third_profiles}
    transfer = {key: cosine(reference[key], profile_vector(third_profiles[key])) for key in third_profiles}
    hypotheses = {
        "G0_agreement_route_is_nearby_predicate_lexicalization": {"supported": agreement["median_distance"] <= 3 and predicate_rates["agreement_attractor:0"] >= .25 and predicate_rates["agreement_attractor:0"] >= 10 * other_predicate, "median_distance": agreement["median_distance"], "predicate_rate": predicate_rates["agreement_attractor:0"], "other_rate": other_predicate, "ratio": predicate_rates["agreement_attractor:0"] / max(other_predicate, 1e-30)},
        "G1_ioi_route_is_short_range_name_valued_association": {"supported": ioi["median_distance"] <= 2 and ioi["capitalized_output_fraction"] >= .6 and ioi["source_echo_rate"] <= .1, "profile": ioi},
        "G2_capital_route_remains_source_linked_entity_lexicalization": {"supported": capital["source_echo_rate"] >= .7 and capital["capitalized_output_fraction"] >= .5, "profile": capital},
        "G3_repetition_route_remains_long_range_copy": {"supported": all(row["source_echo_rate"] >= .7 and row["copy_rate"] >= .3 and row["median_distance"] >= 4 for row in repetition), "profiles": repetition},
        "G4_richer_profiles_remain_stable": {"supported": all(value >= .95 for value in transfer.values()), "cosines": transfer},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "third_profiles": third_profiles, "predicate_rates": predicate_rates, "profile_cosines": transfer}, indent=2, allow_nan=False))
    lines = ["# Recursive semantic generalization on a third FineWeb panel", "", "The first held-out panel rejected narrow auxiliary and strict-locality labels. Broader labels were then frozen: nearby predicate lexicalization and short-range name-valued association. This third disjoint panel tests those generalizations while replicating the two previously successful functions.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Route role | Predicate outputs | Local d≤2 | Echo | Capitalized outputs | Copy-next | Median distance | Profile cosine |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in third_profiles.items():
        lines.append(f"| {key} | {predicate_rates[key]:.3f} | {row['local_rate']:.3f} | {row['source_echo_rate']:.3f} | {row['capitalized_output_fraction']:.3f} | {row['copy_rate']:.3f} | {row['median_distance']:.1f} | {transfer[key]:.3f} |")
    lines += ["", "## Interpretation boundary", "", "Passing a broader label after a narrow one fails is evidence only because the broader predicate was frozen before this third panel. The result supports recursive semantic organization when related uses share a stable operator profile; it does not license unlimited post-hoc broadening."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
