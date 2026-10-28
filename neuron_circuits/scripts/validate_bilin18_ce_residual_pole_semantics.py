#!/usr/bin/env python3
"""Prospectively validate the few interpretable CE-residual pole descriptions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_ce_residual_pole_semantic_validation.json")
OUTPUT = TENSOR_ROOT / "bilin18_ce_residual_pole_semantic_validation.json"
REPORT = TENSOR_ROOT / "BILIN18_CE_RESIDUAL_POLE_SEMANTIC_VALIDATION.md"


def vector(profile):
    keys = sorted(key for key, value in profile.items() if isinstance(value, (int, float)) and key != "distinct_source_tokens")
    return np.asarray([np.log1p(profile[key]) / np.log(128) if key == "median_distance" else profile[key] for key in keys], dtype=np.float64)


def cosine(first, second):
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def main():
    config = json.loads(CONFIG.read_text()); discovery = json.loads(Path(config["discovery"]).read_text()); replication = json.loads(Path(config["replication"]).read_text())
    first = {row["name"]: row["profile"] for row in discovery["results"]}; second = {row["name"]: row["profile"] for row in replication["results"]}
    cosines = {name: cosine(vector(first[name]), vector(second[name])) for name in first}
    checks = {
        "V0_L1N0n_is_local_structural_punctuation": second["L1N0n"]["median_distance"] <= 2 and second["L1N0n"]["source_punctuation"] >= .25 and second["L1N0n"]["self_rate"] >= .20,
        "V1_L2N0n_is_local_current_token_routing": second["L2N0n"]["median_distance"] <= 2 and second["L2N0n"]["self_rate"] >= .35,
        "V2_L4N0n_is_current_punctuation_influenced_routing": second["L4N0n"]["median_distance"] <= 1 and second["L4N0n"]["self_rate"] >= .40 and second["L4N0n"]["source_punctuation"] >= .15,
        "V3_all_L5_poles_are_long_range_routing": all(second[name]["median_distance"] >= 32 for name in ("L5N0p", "L5N0n", "L5N1p", "L5N1n")),
        "V4_broad_pole_profiles_transfer": sum(value >= .85 for value in cosines.values()) >= 10,
    }
    hypotheses = {name: {"supported": bool(value)} for name, value in checks.items()}
    hypotheses["V4_broad_pole_profiles_transfer"]["passing"] = [name for name, value in cosines.items() if value >= .85]
    hypotheses["V4_broad_pole_profiles_transfer"]["cosines"] = cosines
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "discovery_profiles": first, "replication_profiles": second}, indent=2, allow_nan=False))
    lines = ["# Held-out semantics of novel CE-residual route poles", "", "Only four operational descriptions were frozen from discovery; heterogeneous poles remained unnamed. The same discovery-fitted axes rank examples on the previously unopened replication panel.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**" for name, row in hypotheses.items())
    lines += ["", "| Pole | Profile cosine | Distance | Self | Source punctuation | Distinct sources |", "|---|---:|---:|---:|---:|---:|"]
    for name in first:
        row = second[name]
        lines.append(f"| {name} | {cosines[name]:.3f} | {row['median_distance']:.1f} | {row['self_rate']:.3f} | {row['source_punctuation']:.3f} | {row['distinct_source_tokens']} |")
    lines += ["", "## Guardrail", "", "Passing labels describe recurring routing situations, not sufficient end-to-end semantic circuits. Failure of a narrow label with a high full-profile cosine means the axis recurs but the proposed human boundary was wrong. No new label is minted from replication failures without another fresh panel."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
