#!/usr/bin/env python3
"""Quantize frozen read-route templates to obtain an operational MDL curve."""

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
from run_bilin18_direct_read_operator_transfer import aggregate_evaluation, collect_templates
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_direct_read_route_quantization.json")
OUTPUT = TENSOR_ROOT / "bilin18_direct_read_route_quantization.json"
REPORT = TENSOR_ROOT / "BILIN18_DIRECT_READ_ROUTE_QUANTIZATION.md"


def quantize_symmetric(values, bits):
    values = torch.as_tensor(values, dtype=torch.float32)
    if bits == 1:
        scale = values.abs().mean().clamp_min(1e-30)
        return torch.where(values >= 0, scale, -scale)
    maximum = 2 ** (bits - 1) - 1
    scale = values.abs().max().clamp_min(1e-30) / maximum
    return torch.round(values / scale).clamp(-maximum, maximum) * scale


def main():
    config = json.loads(CONFIG.read_text()); source = json.loads(Path(config["source"]).read_text()); construction = json.loads(Path(config["construction_source"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); selected = reconstruct_selected(tokenizer, construction)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    results = {}; maximum_error = 0.0
    for family in config["families"]:
        row = source["results"][family]; layer = row["layer"]; program = row["heldout"]["semantic_causal"]["heads"]
        template, error = collect_templates(model, selected[family]["discovery"], layer, program, construction["config"]["batch_size"])
        maximum_error = max(maximum_error, error)
        shared = .5 * (template["route0"] + template["route1"])
        variants = {"float": template}
        for bits in config["bits"]:
            quantized = quantize_symmetric(shared, bits)
            variant = copy.copy(template); variant["route0"] = quantized; variant["route1"] = quantized
            variants[f"bits_{bits}"] = variant
        rows = {}
        for name, variant in variants.items():
            evaluated, error = aggregate_evaluation(model, tokenizer, selected[family]["heldout"], layer, program, variant, [], construction["config"]["batch_size"])
            maximum_error = max(maximum_error, error)
            # The float row uses the explicitly value-only shared-route model;
            # quantized rows have route0==route1 so both keys coincide.
            key = "shared_route_value_change" if name == "float" else "frozen_route_heldout_value"
            rows[name] = evaluated[key]
        results[family] = {
            "layer": layer, "program": program, "edited_roles": template["roles"],
            "route_scalars": int(shared.numel()),
            "float_route": shared.tolist(), "variants": rows,
        }
        print(f"quantize {family} scalars={shared.numel()} " + " ".join(f"b{bits}={rows[f'bits_{bits}']['program_output']['energy_fidelity']:.3f}" for bits in config["bits"]), flush=True)
    q4 = {family: row["variants"]["bits_4"]["program_output"]["energy_fidelity"] for family, row in results.items()}
    q2 = {family: row["variants"]["bits_2"]["program_output"]["energy_fidelity"] for family, row in results.items()}
    q1 = {family: row["variants"]["bits_1"]["program_output"]["energy_fidelity"] for family, row in results.items()}
    differences = {family: row["variants"]["float"]["program_output"]["energy_fidelity"] - row["variants"]["bits_4"]["program_output"]["energy_fidelity"] for family, row in results.items()}
    hypotheses = {
        "Q0_route_templates_are_small": {"supported": max(row["route_scalars"] for row in results.values()) <= 6, "scalars": {family: row["route_scalars"] for family, row in results.items()}},
        "Q1_four_bit_routes_retain_function": {"supported": all(value >= .7 for value in q4.values()), "fidelities": q4},
        "Q2_two_bit_routes_often_retain_function": {"supported": sum(value >= .7 for value in q2.values()) >= 3, "fidelities": q2},
        "Q3_one_bit_sign_routes_are_not_assumed_sufficient": {"supported": all(value >= .7 for value in q1.values()), "fidelities": q1},
        "Q4_quantized_routes_preserve_float_operator": {"supported": all(value <= .1 for value in differences.values()), "float_minus_four_bit": differences},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "maximum_error": maximum_error, "results": results}, indent=2, allow_nan=False))
    lines = ["# Direct-read route quantization", "", "The checkpoint weights, contextual values, three-head identity, and edited-source role are treated as shared machinery. Only the discovery-mean head-by-role route coefficients are quantized. This is a conditional description-length curve, not a standalone bit count for the whole circuit.", "", "## Locked hypotheses", ""]
    lines.extend(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.** Observed: `{ {k:v for k,v in row.items() if k!='supported'} }`" for name, row in hypotheses.items())
    lines += ["", "| Family | Route scalars | Float | 1 bit | 2 bit | 3 bit | 4 bit | 8 bit |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for family, row in results.items():
        value = lambda name: row["variants"][name]["program_output"]["energy_fidelity"]
        lines.append(f"| {family} | {row['route_scalars']} | {value('float'):.3f} | " + " | ".join(f"{value(f'bits_{bits}'):.3f}" for bits in config["bits"]) + " |")
    lines += ["", "## Interpretation boundary", "", "This curve measures the extra metadata needed after the native tensor operator and contextual values are given. It should not be compared directly with a standalone compressed network. A short route template explains which typed operation is invoked; it does not encode the variable object carried in the value vector."]
    REPORT.write_text("\n".join(lines) + "\n"); print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
