#!/usr/bin/env python3
"""Decode discovery examples for CE-important route geometry beyond generic PCA."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_ce_adjoint_residual_atlas import fit_basis
from redteam_bilin18_ce_adjoint_residual_novelty import normalized_complement
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_natural_operator_invocation_atlas import clean, context_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_ce_residual_pole_semantic_discovery.json")
OUTPUT = TENSOR_ROOT / "bilin18_ce_residual_pole_semantic_discovery.json"
REPORT = TENSOR_ROOT / "BILIN18_CE_RESIDUAL_POLE_SEMANTIC_DISCOVERY.md"
PANEL = "discovery"
PROFILE_FEATURES = ("determiner", "preposition", "conjunction", "pronoun", "auxiliary", "modal", "punctuation", "opening_delimiter", "closing_delimiter", "quote", "numeric", "capitalized")


def profile(examples):
    result = {
        "median_distance": float(np.median([row["distance"] for row in examples])),
        "self_rate": float(np.mean([row["source_token_id"] == row["destination_token_id"] for row in examples])),
        "copy_next_rate": float(np.mean([row["source_token_id"] == row["next_token_id"] for row in examples])),
        "positive_residual_effect_rate": float(np.mean([row["residual_ce_effect"] > 0 for row in examples])),
        "distinct_source_tokens": len(set(row["source_token_id"] for row in examples)),
    }
    for role in ("source", "destination", "next"):
        values = np.stack([vocabulary_features(row[f"{role}_token"]) for row in examples])
        for name in PROFILE_FEATURES:
            result[f"{role}_{name}"] = float(values[:, FEATURES.index(name)].mean())
    return result


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text())
    arrays = np.load(config["arrays"]); generic = np.load(config["generic_arrays"])
    corpus = np.load(config["corpus"], mmap_mode="r"); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    results = []
    for layer in config["layers"]:
        routes = arrays[f"{PANEL}_residual_routes_L{layer}"]; metadata = arrays[f"{PANEL}_metadata_L{layer}"]; effects = arrays[f"{PANEL}_effects_L{layer}"]
        generic_basis = fit_basis(generic[f"discovery_routes_L{layer}"], config["generic_rank"])
        discovery_complement = normalized_complement(arrays[f"discovery_residual_routes_L{layer}"], generic_basis)
        novel_basis = fit_basis(discovery_complement, config["novel_rank"])
        complement = normalized_complement(routes, generic_basis)
        coordinates = complement @ novel_basis.T
        for axis in range(config["novel_rank"]):
            for sign in (1, -1):
                score = sign * coordinates[:, axis] * np.abs(effects[:, 2])
                chosen = np.argsort(-score)[:config["examples_per_pole"]]
                examples = []
                for index in chosen:
                    global_row, destination, source_position, distance, source_id, destination_id, next_id, semantic_member = metadata[index]
                    ids = np.asarray(corpus[int(global_row)], dtype=np.int64)
                    source_context, destination_context = context_windows(ids.tolist(), int(source_position), int(destination), tokenizer)
                    examples.append({
                        "global_row": int(global_row), "source": int(source_position), "destination": int(destination), "distance": int(distance),
                        "source_token_id": int(source_id), "source_token": clean(tokenizer, int(source_id)),
                        "destination_token_id": int(destination_id), "destination_token": clean(tokenizer, int(destination_id)),
                        "next_token_id": int(next_id), "next_token": clean(tokenizer, int(next_id)),
                        "semantic_member": bool(semantic_member), "pole_coordinate": float(coordinates[index, axis]),
                        "selection_score": float(score[index]), "native_ce_effect": float(effects[index, 0]),
                        "semantic_ce_effect": float(effects[index, 1]), "residual_ce_effect": float(effects[index, 2]),
                        "source_context": source_context, "destination_context": destination_context,
                    })
                name = f"L{layer}N{axis}{'p' if sign > 0 else 'n'}"
                results.append({"name": name, "layer": layer, "axis": axis, "sign": sign, "basis": novel_basis[axis].tolist(), "profile": profile(examples), "examples": examples})
    diverse = [row["name"] for row in results if row["profile"]["distinct_source_tokens"] >= 16]
    complete = all(all({"source_context", "destination_context", "source_token", "destination_token", "next_token", "pole_coordinate", "residual_ce_effect"} <= set(example) for example in row["examples"]) for row in results)
    hypotheses = {
        "P0_selection_precedes_decoding": {"supported": True, "selection_rule": config["selection_rule"]},
        "P1_each_pole_has_diverse_lexical_support": {"supported": len(diverse) == len(results), "passing": diverse, "total": len(results)},
        "P2_discovery_artifact_supports_frozen_labels": {"supported": complete},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [f"# Semantic {PANEL} panel for novel CE-residual route poles", "", "Four layers were frozen from the novelty red team before decoding. For each, the earlier generic rank-four route space is removed; both signs of both rank-two complement axes are ranked by coordinate times absolute residual CE effect. The discovery axes are used unchanged on replication.", "", "## Locked gates", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Pole | Distance | Self | Copy-next | Positive CE pole | Distinct sources | Source punct/cap/num | Destination punct/cap | Next punct/cap |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        p = row["profile"]
        lines.append(f"| {row['name']} | {p['median_distance']:.1f} | {p['self_rate']:.3f} | {p['copy_next_rate']:.3f} | {p['positive_residual_effect_rate']:.3f} | {p['distinct_source_tokens']} | {p['source_punctuation']:.2f}/{p['source_capitalized']:.2f}/{p['source_numeric']:.2f} | {p['destination_punctuation']:.2f}/{p['destination_capitalized']:.2f} | {p['next_punctuation']:.2f}/{p['next_capitalized']:.2f} |")
    lines += [""]
    for row in results:
        lines += [f"### {row['name']}", ""]
        for example in row["examples"][:config["report_examples"]]:
            context = example["source_context"] if example["source_context"] == example["destination_context"] else example["source_context"] + " … " + example["destination_context"]
            lines.append(f"- coord={example['pole_coordinate']:+.3f}, residual CE={example['residual_ce_effect']:+.3g}, d={example['distance']}: `{context}`; source `{example['source_token']}` → next `{example['next_token']}`.")
        lines.append("")
    lines += ["## Guardrail", "", "These poles are discovery coordinates, not yet semantic features. A short name is warranted only when a concrete profile can be frozen here and then predicts the untouched replication panel. Related examples may justify a broader recursive label; heterogeneous examples remain unlabeled."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
