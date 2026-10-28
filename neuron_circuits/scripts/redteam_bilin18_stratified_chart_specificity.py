#!/usr/bin/env python3
"""Cross-stratum red team for natural-route chart labels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_ce_adjoint_residual_atlas import subspace_cosine
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_stratified_natural_route_census import (
    all_strata,
    effect_fidelity,
    token_class_from_flags,
    weighted_basis,
)
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_stratified_chart_specificity_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_stratified_chart_specificity_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_STRATIFIED_CHART_SPECIFICITY_REDTEAM.md"


def panel(arrays, split: str, layer: int, token_classes: np.ndarray) -> dict[str, np.ndarray]:
    result = {key: arrays[f"{split}_{key}_L{layer}"] for key in ("residuals", "gradients", "full_effect", "semantic_effect", "distance", "source_id", "next_id", "loss")}
    result["source_class"] = token_classes[result["source_id"]]
    result["next_class"] = token_classes[result["next_id"]]
    return result


def main():
    config = json.loads(CONFIG.read_text()); arrays = np.load(config["arrays"])
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    token_classes = token_class_from_flags(flags)
    results = []
    for target in config["targets"]:
        layer = target["layer"]; target_stratum = target["stratum"]; family = target_stratum.split(":", 1)[0]
        discovery = panel(arrays, "discovery", layer, token_classes); replication = panel(arrays, "replication", layer, token_classes)
        d_masks = all_strata(discovery, discovery["loss"]); r_masks = all_strata(replication, discovery["loss"])
        candidates = sorted(name for name in d_masks if name.startswith(family + ":") and d_masks[name].sum() >= config["minimum_edges"] and r_masks[name].sum() >= config["minimum_edges"])
        bases = {
            name: weighted_basis(discovery["residuals"][d_masks[name]], (discovery["full_effect"] - discovery["semantic_effect"])[d_masks[name]], config["rank"])
            for name in candidates
        }
        target_mask = r_masks[target_stratum]
        fidelities = {
            name: effect_fidelity(replication["residuals"][target_mask], replication["gradients"][target_mask], basis)
            for name, basis in bases.items()
        }
        diagonal = fidelities[target_stratum]
        off_name, off_value = max(((name, value) for name, value in fidelities.items() if name != target_stratum), key=lambda item: item[1])
        ranking = sorted(fidelities, key=fidelities.get, reverse=True)
        overlaps = {first: {second: subspace_cosine(bases[first], bases[second]) for second in candidates} for first in candidates}
        results.append({
            "layer": layer, "target_stratum": target_stratum, "candidates": candidates,
            "fidelities": fidelities, "diagonal_fidelity": diagonal,
            "best_off_stratum": off_name, "best_off_fidelity": off_value,
            "specificity_margin": diagonal - off_value,
            "diagonal_rank": ranking.index(target_stratum) + 1,
            "basis_overlaps": overlaps,
        })

    wins = [row for row in results if row["diagonal_rank"] == 1]
    material = [row for row in results if row["specificity_margin"] >= .05]
    distance = next(row for row in results if row["target_stratum"].startswith("distance:"))
    order = ["distance:self", "distance:local1-4", "distance:near5-16", "distance:mid17-64", "distance:far65+"]
    adjacent = [distance["basis_overlaps"][order[index]][order[index + 1]] for index in range(len(order) - 1)]
    nonadjacent = [distance["basis_overlaps"][order[i]][order[j]] for i in range(len(order)) for j in range(i + 2, len(order))]
    hypotheses = {
        "X0_cross_transfer_is_held_out": {"supported": True},
        "X1_declared_stratum_selects_its_best_chart": {"supported": len(wins) >= 3, "wins": [row["target_stratum"] for row in wins]},
        "X2_chart_specificity_is_material": {"supported": len(material) >= 3, "passing": [row["target_stratum"] for row in material]},
        "X3_distance_geometry_is_ordered": {"supported": float(np.mean(adjacent)) > float(np.mean(nonadjacent)), "adjacent_mean": float(np.mean(adjacent)), "nonadjacent_mean": float(np.mean(nonadjacent))},
        "X4_basis_difference_is_not_required_for_every_regime": {"supported": len(material) < len(results), "failing": [row["target_stratum"] for row in results if row["specificity_margin"] < .05]},
    }
    for row in hypotheses.values():
        row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Cross-stratum specificity red team", "", "Each natural-stratum rank-two chart is fitted on discovery data and evaluated on the frozen target stratum in replication. This asks whether the successful finite chart is genuinely selected by its proposed observational label or is merely one interchangeable alternative to the global chart.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Target | Matching fidelity | Best nonmatching chart | Nonmatching fidelity | Margin | Rank |", "|---|---:|---|---:|---:|---:|"]
    for row in results:
        lines.append(f"| L{row['layer']} {row['target_stratum']} | {row['diagonal_fidelity']:.3f} | {row['best_off_stratum']} | {row['best_off_fidelity']:.3f} | {row['specificity_margin']:+.3f} | {row['diagonal_rank']}/{len(row['candidates'])} |")
    lines += ["", "## Guardrail", "", "Specificity to a coarse stratum supports a conditional routing regime, not a semantic feature. Token-form classes may proxy for syntax, position, frequency, or shared QK state. A failed diagonal comparison invalidates the proposed label as the unique selector but does not invalidate the chart's finite causal usefulness."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
