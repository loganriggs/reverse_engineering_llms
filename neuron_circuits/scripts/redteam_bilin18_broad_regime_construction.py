#!/usr/bin/env python3
"""Materiality, crossing stability, and exhaustive equal-size subset controls."""

from __future__ import annotations

from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_broad_regime_construction import (
    TENSOR_ROOT, candidate_pairs, head_program_batch, length_batches, metric,
)
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_broad_regime_construction_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_broad_regime_construction_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_BROAD_REGIME_CONSTRUCTION_REDTEAM.md"


def first_crossing(summary, threshold):
    for layer in sorted(map(int, summary)):
        if summary[str(layer)]["destination"]["projection_share"] >= threshold:
            return layer
    return None


def reconstruct_selected(tokenizer, source):
    pools = candidate_pairs(tokenizer, source["config"]["families"], source["config"]["maximum_token_edits"], source["config"]["seed"])
    output = {}
    for family in source["config"]["families"]:
        lookup = {(row["base"].example_id, row["source"].example_id): row for row in pools[family]}
        keys = [(row["base"], row["source"]) for row in source["selected"][family]["pairs"]]
        rows = [lookup[key] for key in keys]
        midpoint = source["selected"][family]["discovery_count"]
        output[family] = {"discovery": rows[:midpoint], "heldout": rows[midpoint:]}
    return output


def exhaustive_subset_metrics(model, tokenizer, pairs, layer, frozen, subset_size, chunk_size, batch_size):
    subsets = list(itertools.combinations(range(model.config.n_head), subset_size))
    effects = defaultdict(list); targets = []
    for batch in length_batches(pairs, batch_size):
        batch_target = None
        for start in range(0, len(subsets), chunk_size):
            chunk = subsets[start:start + chunk_size]
            programs = {"subset_" + "_".join(map(str, subset)): list(subset) for subset in chunk}
            programs["all"] = list(range(model.config.n_head))
            target, values, _ = head_program_batch(model, tokenizer, batch, layer, programs)
            if batch_target is None:
                batch_target = target
            for name in programs:
                if name != "all": effects[name].append(values[name])
        targets.append(batch_target)
    target = torch.cat(targets)
    metrics = {name: metric(target, torch.cat(values)) for name, values in effects.items()}
    frozen_name = "subset_" + "_".join(map(str, sorted(frozen)))
    frozen_metric = metrics[frozen_name]
    fidelities = np.asarray([row["energy_fidelity"] for row in metrics.values()])
    return {
        "subsets": len(subsets),
        "frozen_subset": sorted(frozen),
        "frozen": frozen_metric,
        "percentile": float(np.mean(fidelities <= frozen_metric["energy_fidelity"])),
        "null_median": float(np.median(fidelities)),
        "null_p90": float(np.quantile(fidelities, .9)),
        "null_best": float(fidelities.max()),
    }


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(Path(config["source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    selected = reconstruct_selected(tokenizer, source)
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)

    results = {}
    for family in source["config"]["families"]:
        discovery_crossing = first_crossing(source["layer_summary"][family]["discovery"], source["config"]["destination_crossing"])
        heldout_crossing = first_crossing(source["layer_summary"][family]["heldout"], source["config"]["destination_crossing"])
        head = source["head_summary"][family]
        all_metric = head["heldout"]["all"]
        site = str(head["site"])
        native_rms = source["layer_summary"][family]["heldout"][site]["destination"]["target_rms"]
        attention_ratio = all_metric["target_rms"] / max(native_rms, 1e-30)
        frozen = head["head_order"][:config["subset_size"]]
        null = exhaustive_subset_metrics(
            model, tokenizer, selected[family]["heldout"], head["site"], frozen,
            config["subset_size"], config["subset_chunk_size"], source["config"]["batch_size"],
        )
        results[family] = {
            "discovery_crossing": discovery_crossing,
            "heldout_crossing": heldout_crossing,
            "crossing_difference": None if discovery_crossing is None or heldout_crossing is None else abs(discovery_crossing - heldout_crossing),
            "heldout_pairs": len(selected[family]["heldout"]),
            "attention_native_rms_ratio": attention_ratio,
            "material_attention": attention_ratio >= config["attention_native_rms_ratio_gate"],
            "subset_null": null,
        }
        print(f"subset null {family} percentile={null['percentile']:.3f}", flush=True)

    material = [family for family, row in results.items() if row["material_attention"]]
    compact = [family for family in material if results[family]["subset_null"]["frozen"]["energy_fidelity"] >= source["config"]["head_compact_fidelity"]]
    high_percentile = [family for family in material if results[family]["subset_null"]["percentile"] >= .9]
    exploratory = [family for family, row in results.items() if row["heldout_pairs"] < 4]
    maximum_head_error = max(row["max_head_sum_abs"] for row in source["head_summary"].values())
    hypotheses = {
        "R0_exactness_survives": {"supported": source["maximum_errors"]["state_relative"] <= 1e-5 and source["maximum_errors"]["shapley_relative"] <= 1e-5 and source["maximum_errors"]["base_logit_abs"] <= 1e-4 and maximum_head_error <= 1e-3, "construction_errors": source["maximum_errors"], "maximum_head_error": maximum_head_error},
        "R1_crossing_depth_generalizes": {"supported": sum(row["crossing_difference"] is not None and row["crossing_difference"] <= config["crossing_layer_tolerance"] for row in results.values()) >= 6, "differences": {family: row["crossing_difference"] for family, row in results.items()}},
        "R2_material_attention_is_not_universal": {"supported": any(not row["material_attention"] for row in results.values()), "ratios": {family: row["attention_native_rms_ratio"] for family, row in results.items()}},
        "R3_material_attention_remains_compact": {"supported": len(compact) >= 4, "families": compact},
        "R4_discovery_rankings_beat_equal_size_subset_nulls": {"supported": len(high_percentile) >= 4, "families": high_percentile, "percentiles": {family: results[family]["subset_null"]["percentile"] for family in material}},
        "R5_small_family_panels_are_disclosed": {"supported": True, "exploratory": exploratory},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Broad-regime construction red team", "", "This audit separates whether an attention branch is behaviorally material from whether that branch is internally compressible. It also compares each frozen discovery-selected top-three program against every one of the 84 possible three-head subsets on held-out pairs.", "", "## Locked results", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name,row in hypotheses.items())
    lines += ["", "| Family | Cross D/H | Heldout n | Attention/native RMS | Material | Frozen top-3 fidelity | All-subset median / p90 | Percentile |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for family, row in results.items():
        null=row["subset_null"]
        lines.append(f"| {family} | {row['discovery_crossing']}/{row['heldout_crossing']} | {row['heldout_pairs']} | {row['attention_native_rms_ratio']:.3f} | {'yes' if row['material_attention'] else 'no'} | {null['frozen']['energy_fidelity']:.3f} | {null['null_median']:.3f} / {null['null_p90']:.3f} | {null['percentile']:.3f} |")
    lines += ["", "## Interpretation", "", "A high subset percentile says the frozen head identity matters beyond merely choosing three of nine heads. It still describes one family-conditioned invocation at one layer. Families with fewer than four held-out pairs are exploratory, and a compact but immaterial branch is not counted as a task mechanism."]
    REPORT.write_text("\n".join(lines)+"\n")
    print(json.dumps(hypotheses,indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
