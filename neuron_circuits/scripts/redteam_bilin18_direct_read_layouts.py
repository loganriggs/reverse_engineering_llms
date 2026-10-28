#!/usr/bin/env python3
"""Test direct-read verbs beyond the original fixed prompt layout."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_bilin18_mlp_discriminators import TOKENIZER
from redteam_bilin18_broad_regime_construction import reconstruct_selected
from redteam_bilin18_broad_task_competence import class_token_ids
from run_bilin18_broad_regime_construction import contrasts, length_batches, trace
from run_bilin18_direct_read_operator_transfer import (
    aggregate_evaluation, collect_templates,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_direct_read_layout_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_direct_read_layout_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_DIRECT_READ_LAYOUT_REDTEAM.md"


def insert_before_destination(pair, filler_ids):
    row = copy.copy(pair)
    row["base_ids"] = list(pair["base_ids"][:-1]) + list(filler_ids) + [pair["base_ids"][-1]]
    row["source_ids"] = list(pair["source_ids"][:-1]) + list(filler_ids) + [pair["source_ids"][-1]]
    return row


@torch.inference_mode()
def exact_endpoint_screen(model, tokenizer, pairs, batch_size):
    retained = []; records = []
    for batch in length_batches(pairs, batch_size):
        size = len(batch)
        ids = torch.as_tensor([pair["base_ids"] for pair in batch] + [pair["source_ids"] for pair in batch], device=model.embedding.weight.device)
        _, _, logits = trace(model, ids)
        contrast = contrasts(tokenizer, batch, ids.device, model.config.vocab_size)
        native = (logits[size:] * contrast).sum(-1) - (logits[:size] * contrast).sum(-1)
        for index, pair in enumerate(batch):
            endpoint_ok = []
            ranks = []
            for endpoint, values in ((pair["base"], logits[index]), (pair["source"], logits[size + index])):
                groups = class_token_ids(tokenizer, endpoint)
                answer = endpoint.candidates.index(endpoint.answer)
                correct_ids = groups[answer]
                best = int(values.argmax())
                endpoint_ok.append(best in correct_ids)
                ranks.append(1 + int((values > values[correct_ids].max()).sum()))
            keep = all(endpoint_ok) and float(native[index]) > 0
            records.append({
                "base": pair["base"].example_id, "source": pair["source"].example_id,
                "base_exact": endpoint_ok[0], "source_exact": endpoint_ok[1],
                "base_rank": ranks[0], "source_rank": ranks[1],
                "native_pair_effect": float(native[index]), "retained": keep,
            })
            if keep: retained.append(pair)
    return retained, records


def cosine(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    return float(first @ second / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-30))


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text()); construction = json.loads(Path(config["construction_source"]).read_text()); operator = json.loads(Path(config["operator_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); selected = reconstruct_selected(tokenizer, construction)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; maximum_error = 0.0
    for family, fillers in config["layouts"].items():
        row = source["results"][family]; layer = row["layer"]; program = row["heldout"]["semantic_causal"]["heads"]
        template, error = collect_templates(model, selected[family]["discovery"], layer, program, construction["config"]["batch_size"])
        maximum_error = max(maximum_error, error); results[family] = {"layer": layer, "program": program, "layouts": {}}
        for filler in fillers:
            filler_ids = tokenizer.encode(filler, add_special_tokens=False)
            variants = [insert_before_destination(pair, filler_ids) for pair in selected[family]["heldout"]]
            retained, screen = exact_endpoint_screen(model, tokenizer, variants, construction["config"]["batch_size"])
            layout = {
                "filler": filler, "filler_ids": filler_ids, "filler_tokens": [tokenizer.decode([item]) for item in filler_ids],
                "declared_pairs": len(variants), "exact_pairs": len(retained), "screen": screen,
            }
            if retained:
                evaluated, error = aggregate_evaluation(model, tokenizer, retained, layer, program, template, [], construction["config"]["batch_size"])
                maximum_error = max(maximum_error, error)
                held_template, _ = collect_templates(model, retained, layer, program, construction["config"]["batch_size"])
                layout.update({
                    "actual_edited": evaluated["actual_edited"],
                    "frozen_route_heldout_value": evaluated["frozen_route_heldout_value"],
                    "shared_route_value_change": evaluated["shared_route_value_change"],
                    "route_cosine": float(np.mean([
                        cosine(template["route0"], held_template["route0"]),
                        cosine(template["route1"], held_template["route1"]),
                    ])),
                    "original_length": len(selected[family]["heldout"][0]["base_ids"]),
                    "altered_length": len(retained[0]["base_ids"]),
                })
            results[family]["layouts"][str(len(filler_ids))] = layout
            print(f"layout {family} +{len(filler_ids)} exact={len(retained)}/{len(variants)} frozen={layout.get('frozen_route_heldout_value',{}).get('program_output',{}).get('energy_fidelity')}", flush=True)
    operational = [(family, key, cell) for family, row in results.items() for key, cell in row["layouts"].items() if cell["exact_pairs"] >= config["minimum_exact_pairs"]]
    families_operational = sorted(set(family for family, _, _ in operational))
    shortest = {}
    for family in families_operational:
        candidates = [(int(key), cell) for key, cell in results[family]["layouts"].items() if cell["exact_pairs"] >= config["minimum_exact_pairs"]]
        shortest[family] = min(candidates)[1]
    native_pass = [family for family, cell in shortest.items() if cell["actual_edited"]["program_output"]["energy_fidelity"] >= .8]
    frozen_pass = [family for family, cell in shortest.items() if cell["frozen_route_heldout_value"]["program_output"]["energy_fidelity"] >= config["minimum_transfer_fidelity"]]
    two_pass = [family for family, row in results.items() if "2" in row["layouts"] and row["layouts"]["2"]["exact_pairs"] >= config["minimum_exact_pairs"] and row["layouts"]["2"]["frozen_route_heldout_value"]["program_output"]["energy_fidelity"] >= config["minimum_transfer_fidelity"]]
    route_cosines = [cell["route_cosine"] for _, _, cell in operational]
    hypotheses = {
        "L0_layouts_retain_exact_behavior": {"supported": len(families_operational) >= 3, "families": families_operational},
        "L1_native_edited_edges_survive_layout_change": {"supported": len(native_pass) >= 3, "families": native_pass},
        "L2_original_frozen_routes_transfer_across_layouts": {"supported": len(frozen_pass) >= 3, "families": frozen_pass},
        "L3_two_token_distance_shift_transfers": {"supported": len(two_pass) >= 2, "families": two_pass},
        "L4_route_coefficients_are_layout_stable": {"supported": bool(route_cosines) and float(np.mean(route_cosines)) >= .9, "mean_cosine": float(np.mean(route_cosines)) if route_cosines else None, "cells": len(route_cosines)},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "maximum_error": maximum_error, "results": results}, indent=2, allow_nan=False))
    lines = ["# Direct-read operator transfer across altered layouts", "", "Neutral material is inserted immediately before the original prediction-position token, increasing the relative distance from the edited operand while preserving the original answer. Mechanistic results are reported only for cells with at least two pairs whose base and source endpoints both retain exact full-vocabulary top-1 behavior.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Family | Added tokens | Exact pairs | Native edited fidelity | Original frozen-route fidelity | Route cosine |", "|---|---:|---:|---:|---:|---:|"]
    for family, row in results.items():
        for key, cell in row["layouts"].items():
            native = cell.get("actual_edited", {}).get("program_output", {}).get("energy_fidelity")
            frozen = cell.get("frozen_route_heldout_value", {}).get("program_output", {}).get("energy_fidelity")
            cosine_value = cell.get("route_cosine")
            fmt = lambda value: "—" if value is None else f"{value:.3f}"
            lines.append(f"| {family} | {key} | {cell['exact_pairs']}/{cell['declared_pairs']} | {fmt(native)} | {fmt(frozen)} | {fmt(cosine_value)} |")
    lines += ["", "## Guardrail", "", "These variants alter relative distance but preserve the same broad syntactic template. Success rules out a single absolute-position lookup; it does not yet establish transfer to arbitrary paraphrases. Cells losing exact behavior are capability failures and are not used as negative mechanistic evidence."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
