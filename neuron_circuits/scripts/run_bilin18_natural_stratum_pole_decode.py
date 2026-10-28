#!/usr/bin/env python3
"""Decode both poles of causally validated natural-stratum route charts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_natural_operator_invocation_atlas import clean, context_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_stratified_natural_route_census import all_strata, distance_class, token_class_from_flags, weighted_basis


CONFIG = Path("configs/bilin18_natural_stratum_pole_decode.json")
OUTPUT = TENSOR_ROOT / "bilin18_natural_stratum_pole_decode.json"
REPORT = TENSOR_ROOT / "BILIN18_NATURAL_STRATUM_POLE_DECODE.md"
TOKEN_CLASSES = ("punctuation", "function", "numeric", "capitalized", "specialized", "continuation", "other")
DISTANCES = ("self", "local1-4", "near5-16", "mid17-64", "far65+")


def cosine(first, second):
    first = np.asarray(first, dtype=float); second = np.asarray(second, dtype=float)
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def effective_support(axis):
    values = np.abs(np.asarray(axis, dtype=float))
    return float(values.sum() ** 2 / max(np.square(values).sum(), 1e-30))


def profile(rows):
    values = []
    for label in TOKEN_CLASSES:
        values.append(np.mean([row["source_class"] == label for row in rows]))
    for label in TOKEN_CLASSES:
        values.append(np.mean([row["destination_class"] == label for row in rows]))
    for label in DISTANCES:
        values.append(np.mean([row["distance_class"] == label for row in rows]))
    values += [np.mean([row["residual_effect"] > 0 for row in rows]), np.mean([row["source_id"] == row["next_id"] for row in rows])]
    return [float(value) for value in values]


def panel(arrays, split, layer, classes):
    result = {key: arrays[f"{split}_{key}_L{layer}"] for key in ("residuals", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss", "global_row", "destination", "source_position")}
    result["source_class"] = classes[result["source_id"]]; result["next_class"] = classes[result["next_id"]]
    return result


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"]); corpus = np.load(config["corpus"], mmap_mode="r")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    classes = token_class_from_flags(flags); results = []
    for target in config["targets"]:
        layer = target["layer"]; stratum = target["stratum"]
        discovery = panel(arrays, "discovery", layer, classes); replication = panel(arrays, "replication", layer, classes)
        dmask = all_strata(discovery, discovery["loss"])[stratum]; rmask = all_strata(replication, discovery["loss"])[stratum]
        effects_d = discovery["full_effect"] - discovery["semantic_effect"]
        effects_r = replication["full_effect"] - replication["semantic_effect"]
        basis = weighted_basis(discovery["residuals"][dmask], effects_d[dmask], config["rank"])
        target_rows = []
        for axis_index, axis in enumerate(basis):
            pole_rows = []
            for sign in (1, -1):
                panel_rows = []
                for split, data, mask, effects in (("discovery", discovery, dmask, effects_d), ("replication", replication, rmask, effects_r)):
                    coordinate = data["residuals"] @ axis
                    eligible = np.where(mask)[0]
                    score = sign * coordinate[eligible] * np.abs(effects[eligible])
                    chosen = eligible[np.argsort(-score)[:config["examples_per_pole"]]]
                    examples = []
                    for index in chosen:
                        global_row = int(data["global_row"][index]); destination = int(data["destination"][index]); source = int(data["source_position"][index])
                        ids = np.asarray(corpus[global_row], dtype=np.int64)
                        source_context, destination_context = context_windows(ids.tolist(), source, destination, tokenizer)
                        destination_id = int(ids[destination]); source_id = int(data["source_id"][index]); next_id = int(data["next_id"][index])
                        examples.append({
                            "global_row": global_row, "source": source, "destination": destination, "distance": int(data["distance"][index]),
                            "distance_class": str(distance_class(np.asarray([data["distance"][index]]))[0]),
                            "source_id": source_id, "source_token": clean(tokenizer, source_id), "source_class": str(classes[source_id]),
                            "destination_id": destination_id, "destination_token": clean(tokenizer, destination_id), "destination_class": str(classes[destination_id]),
                            "next_id": next_id, "next_token": clean(tokenizer, next_id), "next_class": str(classes[next_id]),
                            "loss": float(data["loss"][index]), "coordinate": float(coordinate[index]), "residual_effect": float(effects[index]),
                            "source_context": source_context, "destination_context": destination_context,
                        })
                    panel_rows.append({"split": split, "profile": profile(examples), "distinct_sources": len(set(row["source_id"] for row in examples)), "examples": examples})
                pole_rows.append({"sign": sign, "discovery": panel_rows[0], "replication": panel_rows[1], "profile_cosine": cosine(panel_rows[0]["profile"], panel_rows[1]["profile"])})
            target_rows.append({"axis": axis_index, "coefficients": axis.tolist(), "effective_head_support": effective_support(axis), "poles": pole_rows})
        results.append({"layer": layer, "stratum": stratum, "axes": target_rows})
    poles = [pole for target in results for axis in target["axes"] for pole in axis["poles"]]
    axes = [axis for target in results for axis in target["axes"]]
    hypotheses = {
        "D0_decoding_uses_frozen_axes": {"supported": True},
        "D1_poles_are_not_single_token_lookups": {"supported": all(min(pole["discovery"]["distinct_sources"], pole["replication"]["distinct_sources"]) >= 16 for pole in poles), "minimum_distinct_sources": min(min(pole["discovery"]["distinct_sources"], pole["replication"]["distinct_sources"]) for pole in poles)},
        "D2_route_axes_use_multiple_heads": {"supported": all(axis["effective_head_support"] >= 2 for axis in axes), "supports": [axis["effective_head_support"] for axis in axes]},
        "D3_pole_profiles_replicate": {"supported": float(np.mean([pole["profile_cosine"] for pole in poles])) >= .80, "mean_cosine": float(np.mean([pole["profile_cosine"] for pole in poles])), "cosines": [pole["profile_cosine"] for pole in poles]},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Natural examples for specific stratum route charts", "", "Both poles of discovery-fitted axes are decoded on discovery and untouched replication edges. Selection ranks a frozen coordinate by absolute residual CE effect within the validated natural stratum.", "", "## Locked checks", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    for target in results:
        lines += ["", f"## L{target['layer']} {target['stratum']}", ""]
        for axis in target["axes"]:
            lines += [f"### Axis {axis['axis']} (effective head support {axis['effective_head_support']:.2f})", "", f"Head coefficients: `{[round(value, 3) for value in axis['coefficients']]}`", ""]
            for pole in axis["poles"]:
                lines += [f"#### {'Positive' if pole['sign'] > 0 else 'Negative'} pole (profile cosine {pole['profile_cosine']:.3f})", ""]
                for example in pole["replication"]["examples"][:config["report_examples"]]:
                    context = example["source_context"] if example["source_context"] == example["destination_context"] else example["source_context"] + " … " + example["destination_context"]
                    lines.append(f"- coord={example['coordinate']:+.3f}, residual CE={example['residual_effect']:+.3g}, d={example['distance']}, loss={example['loss']:.2f}: `{context}`; source `{example['source_token']}` → next `{example['next_token']}`.")
                lines.append("")
    lines += ["## Guardrail", "", "These examples decode conditional route coordinates, not neurons and not complete tasks. Related examples can motivate a prospective semantic hypothesis, but heterogeneous poles remain valid engineering coordinates without receiving a concept label."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__": main()
