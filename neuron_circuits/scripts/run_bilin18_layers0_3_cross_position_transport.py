#!/usr/bin/env python3
"""Cross-position transport-role red team at source MLP Layers 0--3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin18_layer1_cross_position_role_redteam import (
    cross_gradient, extended_metrics, native_logits,
)
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_suffix_qk_role_pilot import ROLES, attention_role_vjps
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_layer8_router_causal_validation import suffix_from_layer
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layers0_3_cross_position_transport.json")
OUTPUT = TENSOR_ROOT / "bilin18_layers0_3_cross_position_transport.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYERS0_3_CROSS_POSITION_TRANSPORT.md"
FIGURE = TENSOR_ROOT / "bilin18_layers0_3_cross_position_transport.png"


def specifications(source_layer, block_count):
    suffix = list(range(source_layer + 1, block_count))
    near = set(suffix[:8])
    return {
        "mlp_only": {"layers": set(), "roles": set()},
        "queries": {"layers": set(suffix), "roles": {"q1", "q2"}},
        "keys": {"layers": set(suffix), "roles": {"k1", "k2"}},
        "value": {"layers": set(suffix), "roles": {"value"}},
        "query_key": {"layers": set(suffix), "roles": {"q1", "q2", "k1", "k2"}},
        "query_value": {"layers": set(suffix), "roles": {"q1", "q2", "value"}},
        "key_value": {"layers": set(suffix), "roles": {"k1", "k2", "value"}},
        "near_8": {"layers": near, "roles": set(ROLES)},
        "all_full": {"layers": set(suffix), "roles": set(ROLES)},
    }


def role_program(model, trace, source_layer, edit_position, prediction_position, contrast, specs):
    names = list(specs)
    current = final_adjoint(model, trace[-1]["output"], prediction_position, contrast).repeat(
        len(names), 1, 1, 1,
    )
    maximum_error = 0.0
    for layer in range(len(model.blocks) - 1, source_layer, -1):
        block = model.blocks[layer]
        updated = current + mlp_vjp(block, trace[layer]["pre"].float(), current)
        enabled = [row for row, name in enumerate(names) if layer in specs[name]["layers"] and specs[name]["roles"]]
        if enabled:
            incoming = updated[enabled]
            previous = trace[layer - 1]["output"].float()
            x0 = trace[layer]["x0"].float()
            mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
            bus = trace[layer - 1]["value_bus"]
            pieces = attention_role_vjps(block, mixed, bus, incoming)
            native = attention_vjp(block, mixed, bus, incoming)
            recomposed = sum(pieces.values())
            maximum_error = max(maximum_error, float((recomposed - native).norm() / native.norm().clamp_min(1e-30)))
            for local, row in enumerate(enabled):
                updated[row] += sum((pieces[role][local] for role in specs[names[row]]["roles"]), torch.zeros_like(updated[row]))
        current = block.lambdas[0].float() * updated
    down = model.blocks[source_layer].mlp.Down.weight.float()
    return {name: current[row, 0, edit_position] @ down for row, name in enumerate(names)}, maximum_error


def logits_for_deltas(model, source_layer, base, deltas, edit_position, prediction_position):
    names = list(deltas)
    decoder = model.blocks[source_layer].mlp.Down.weight.float()
    outputs, x0s, buses = [], [], []
    for delta in deltas.values():
        output = base["output"].clone()
        output[:, edit_position] += F.linear(delta, decoder).to(output.dtype)
        outputs.append(output); x0s.append(base["x0"]); buses.append(base["value_bus"])
    logits = suffix_from_layer(model, source_layer, torch.cat(x0s), torch.cat(buses), torch.cat(outputs), prediction_position)
    return {name: logits[index] for index, name in enumerate(names)}


def summarize(records, layer, distance, method):
    rows = [row for row in records if row["layer"] == layer and row["distance"] == distance]
    metrics = [row["metrics"][method] for row in rows]
    shifts = np.asarray([row["finite_shifts"][method] for row in rows])
    return {
        "gradient_cosine_median": float(np.median([value["gradient_cosine"] for value in metrics])),
        "gradient_cosine_q10": float(np.quantile([value["gradient_cosine"] for value in metrics], .1)),
        "gradient_norm_ratio_median": float(np.median([value["gradient_norm_ratio"] for value in metrics])),
        "finite_directional_fraction": float(np.mean(shifts > 0)),
        "finite_mean_shift": float(shifts.mean()),
    }


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(config["corpus"], mmap_mode="r")
    rows = corpus[config["start_row"]:config["start_row"] + config["rows"]]
    records = []
    maximum_error = 0.0
    for edit_position in config["edit_positions"]:
        for row_index, raw_row in enumerate(rows):
            raw = np.asarray(raw_row[:config["sequence_length"]], dtype=np.int64).copy()
            if raw[edit_position] == newline: continue
            changed = raw.copy(); changed[edit_position] = newline
            base_ids = torch.as_tensor(raw, device="cuda")[None]
            changed_ids = torch.as_tensor(changed, device="cuda")[None]
            with torch.no_grad():
                base_trace = forward_layers_native(model, base_ids, len(model.blocks) - 1)
                changed_trace = forward_layers_native(model, changed_ids, max(config["source_layers"]))
                base_logits = native_logits(model, base_trace, config["prediction_position"])
                changed_logits = native_logits(model, forward_layers_native(model, changed_ids, len(model.blocks) - 1), config["prediction_position"])
            contrast = (F.softmax(changed_logits, -1) - F.softmax(base_logits, -1)).detach()
            for source_layer in config["source_layers"]:
                delta = product_at(model, changed_trace, source_layer, edit_position) - product_at(model, base_trace, source_layer, edit_position)
                exact = cross_gradient(model, base_trace, source_layer, edit_position, config["prediction_position"], contrast)
                specs = specifications(source_layer, len(model.blocks))
                predictions, error = role_program(model, base_trace, source_layer, edit_position, config["prediction_position"], contrast, specs)
                maximum_error = max(maximum_error, error)
                predictions["exact"] = exact
                supports = {name: (prediction * delta).topk(config["support_rank"]).indices for name, prediction in predictions.items()}
                patches = {"base": torch.zeros_like(delta), **{name: selected_delta(delta, support) for name, support in supports.items()}}
                with torch.no_grad(): logits = logits_for_deltas(model, source_layer, base_trace[source_layer], patches, edit_position, config["prediction_position"])
                base_score = float(logits["base"].float() @ contrast)
                finite = {name: float(value.float() @ contrast) - base_score for name, value in logits.items() if name != "base"}
                records.append({
                    "layer": source_layer, "distance": config["prediction_position"] - edit_position,
                    "row": row_index, "source_token": int(raw[edit_position]),
                    "source_text": tokenizer.decode([int(raw[edit_position])]).replace("\n", "\\n"),
                    "metrics": {name: extended_metrics(prediction, exact, delta, config["support_rank"]) for name, prediction in predictions.items()},
                    "finite_shifts": finite,
                })
            print(f"position {edit_position} row {row_index + 1}/{len(rows)}", flush=True)

    distances = sorted({row["distance"] for row in records})
    summaries = {
        str(layer): {str(distance): {method: summarize(records, layer, distance, method) for method in records[0]["metrics"]} for distance in distances}
        for layer in config["source_layers"]
    }
    cells = [summaries[str(layer)][str(distance)] for layer in config["source_layers"] for distance in distances]
    hypotheses = {
        "T0_exact_role_composition": {"supported": maximum_error <= 1e-5 and all(cell["all_full"]["gradient_cosine_median"] >= .999 for cell in cells)},
        "T1_nontransport_roles_are_zero": {"supported": all(max(cell[name]["gradient_norm_ratio_median"] for name in ("mlp_only", "queries")) <= 1e-5 for cell in cells)},
        "T2_key_value_is_material": {"supported": all(cell["key_value"]["gradient_cosine_median"] >= .70 for cell in cells)},
        "T3_no_transport_role_is_sufficient": {"supported": not (all(cell["keys"]["gradient_cosine_median"] >= .90 for cell in cells) or all(cell["value"]["gradient_cosine_median"] >= .90 for cell in cells))},
        "T4_near_eight_blocks_are_sufficient": {"supported": all(cell["near_8"]["gradient_cosine_median"] >= .90 for cell in cells)},
        "T5_top64_finite_direction": {"supported": all(cell["exact"]["finite_directional_fraction"] >= .90 for cell in cells)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "maximum_local_recomposition_error": maximum_error, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Layers 0–3 cross-position transport red team", "", "A natural FineWeb token is replaced by newline at distances 8, 32, and 64 from the output position. The full model's probability-change direction defines the behavior. Source-layer MLP-product sensitivity must travel across positions through key or value roles; query-only and MLP-only paths are structurally unable to do so.", "", f"Maximum local role recomposition error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Cross-position depth comparison](bilin18_layers0_3_cross_position_transport.png)", "", "| Layer | Distance | Keys | Value | Key+value | Query+value | Near 8 | Exact top-64 direction |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for layer in config["source_layers"]:
        for distance in distances:
            cell = summaries[str(layer)][str(distance)]
            lines.append(f"| {layer} | {distance} | {cell['keys']['gradient_cosine_median']:.3f} | {cell['value']['gradient_cosine_median']:.3f} | {cell['key_value']['gradient_cosine_median']:.3f} | {cell['query_value']['gradient_cosine_median']:.3f} | {cell['near_8']['gradient_cosine_median']:.3f} | {cell['exact']['finite_directional_fraction']:.3f} |")
    lines += ["", "## Guardrail", "", "This assay explains an already identified behavioral change and does not discover a semantic feature prospectively. Its value is architectural: it prevents same-position query dominance from being mistaken for a universal learned ontology."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained", sharey=True)
    for axis, distance in zip(axes, distances):
        for method in ("keys", "value", "key_value", "query_value", "near_8"):
            axis.plot(config["source_layers"], [summaries[str(layer)][str(distance)][method]["gradient_cosine_median"] for layer in config["source_layers"]], marker="o", label=method)
        axis.set(title=f"Distance {distance}", xlabel="Source MLP layer"); axis.grid(alpha=.2); axis.axhline(.9, color="gray", linestyle=":")
    axes[0].set_ylabel("Median gradient cosine"); axes[-1].legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"maximum_local_recomposition_error": maximum_error, "hypotheses": hypotheses, "summaries": summaries}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
