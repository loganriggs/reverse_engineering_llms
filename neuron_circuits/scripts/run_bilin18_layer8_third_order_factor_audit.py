#!/usr/bin/env python3
"""Decompose singular third-order attention-head paths into QK/value factors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_third_order_factor_audit.json")
OUTPUT = TENSOR_ROOT / "layer8_third_order_factor_audit.json"
REPORT = TENSOR_ROOT / "LAYER8_THIRD_ORDER_FACTOR_AUDIT.md"


@torch.inference_mode()
def response(model, site, position, mask, layer=None, head=None, factor=None):
    logits, tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=bool(mask), attention_layers=mask,
        attention_heads=None if layer is None else {layer: (head,)},
        attention_factors=None if factor is None else {layer: (factor,)},
    )
    return output_fisher_response(logits, tangents)


@torch.inference_mode()
def third_order(model, site, triple, position, head=None, factor=None):
    i, j, k = triple
    empty = response(model, site, position, ())
    ri = response(model, site, position, (i,)); rj = response(model, site, position, (j,))
    if head is None:
        rk = response(model, site, position, (k,))
        rik = response(model, site, position, (i, k)); rjk = response(model, site, position, (j, k))
        rijk = response(model, site, position, triple)
    else:
        rk = response(model, site, position, (k,), k, head, factor)
        rik = response(model, site, position, (i, k), k, head, factor)
        rjk = response(model, site, position, (j, k), k, head, factor)
        rijk = response(model, site, position, triple, k, head, factor)
    rij = response(model, site, position, (i, j))
    return rijk - rij - rik - rjk + ri + rj + rk - empty, rijk


@torch.inference_mode()
def audit(model, site, triple, head, factors, position):
    full, exact = third_order(model, site, triple, position)
    head_full, _ = third_order(model, site, triple, position, head=head)
    components = []
    for factor in factors:
        component, _ = third_order(model, site, triple, position, head=head, factor=factor)
        components.append(component)
    recomposed = torch.stack(components).sum(0)
    head_norm = head_full.norm().clamp_min(1e-30)
    shares = np.asarray([float((component * head_full).sum() / head_norm.square())
                         for component in components])
    absolute = np.abs(shares); normalized = absolute / max(absolute.sum(), 1e-30)
    routing_mass = float(normalized[:4].sum())
    return {
        "full_relative_residual_norm": float(full.norm() / exact.norm().clamp_min(1e-30)),
        "head_relative_residual_norm": float(head_norm / exact.norm().clamp_min(1e-30)),
        "recomposition_error_to_exact_response": float(
            (recomposed - head_full).norm() / exact.norm().clamp_min(1e-30)
        ),
        "factor_projection_shares": shares.tolist(),
        "top_factor": int(np.argmax(absolute)),
        "top_absolute_share": float(normalized.max()),
        "effective_factor_support": float(1 / max(np.square(normalized).sum(), 1e-30)),
        "routing_absolute_mass": routing_mass,
        "current_value_absolute_share": float(normalized[4]),
    }


def main():
    config = json.loads(CONFIG.read_text())
    head_config = json.loads(Path(config["parent_config"]).read_text())
    stability_config = json.loads(Path(head_config["parent_config"]).read_text())
    mobius_config = json.loads(Path(stability_config["parent_config"]).read_text())
    synergy_config = json.loads(Path(mobius_config["parent_config"]).read_text())
    singleton_config = json.loads(Path(synergy_config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"], "sites_per_query": 2,
                     "budget": parent["budget"]}
    head_audit = json.loads((TENSOR_ROOT / "layer8_third_order_head_audit.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    row_banks = {}
    for corpus, corpus_config in parent["evaluation"].items():
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], 2)
        row_banks[corpus] = {(query, row_index): raw for query, row_index, raw in rows}
    records = []
    for number, source_row in enumerate(head_audit["records"]):
        corpus, query = source_row["corpus"], source_row["query"]
        triple = tuple(source_row["mask"]); head = source_row["failure"]["top_head"]
        failure_site = prepare_site(model, row_banks[corpus][(query, source_row["row"])], query,
                                    source_config, tensor_basis, directions, vectors)
        control_site = prepare_site(model, row_banks[corpus][(query, source_row["control_row"])], query,
                                    source_config, tensor_basis, directions, vectors)
        failure = audit(model, failure_site, triple, head, config["factors"], source_config["causal_position"])
        control = audit(model, control_site, triple, head, config["factors"], source_config["causal_position"])
        records.append({"corpus": corpus, "query": query, "row": source_row["row"],
                        "control_row": source_row["control_row"], "mask": list(triple),
                        "head": head, "failure": failure, "control": control})
        print("factor audit", number + 1, "of", len(head_audit["records"]), flush=True)
    max_error = max(row[side]["recomposition_error_to_exact_response"]
                    for row in records for side in ("failure", "control"))
    top = float(np.median([row["failure"]["top_absolute_share"] for row in records]))
    support = float(np.median([row["failure"]["effective_factor_support"] for row in records]))
    routing = float(np.median([row["failure"]["routing_absolute_mass"] for row in records]))
    current = float(np.median([row["failure"]["current_value_absolute_share"] for row in records]))
    hypotheses = {
        "F0_factor_decomposition_recomposes": {"supported": bool(max_error <= 1e-4)},
        "F1_one_factor_dominates": {"supported": bool(top >= .50 and support <= 3)},
        "F2_routing_derivative_dominates": {"supported": bool(routing >= .80)},
        "F3_direct_current_value_dominates": {"supported": bool(current >= .50)},
    }
    summary = {"max_recomposition_error_to_exact": max_error,
               "median_top_factor_share": top, "median_effective_factor_support": support,
               "median_routing_mass": routing, "median_current_value_share": current}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "summary": summary, "records": records}, indent=2))
    lines = ["# QK×OV factor decomposition of singular third-order paths", "",
             "The dominant final head of every causal failure is decomposed into first/second QK routing of current/value-bus codes plus the direct current-value derivative.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum recomposition error relative to exact response: `{max_error:.3e}`. Median top-factor share: `{top:.3f}`. Effective support: `{support:.2f}`. QK-routing mass: `{routing:.3f}`. Direct current-value share: `{current:.3f}`.", "",
              "| Corpus | Query/row | Triple/head | Top factor/share | Effective support | Routing mass | Current-V share |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in records:
        result = row["failure"]; factor = config["factors"][result["top_factor"]]
        lines.append(f"| {row['corpus']} | {row['query']}/{row['row']} | {row['mask']}/H{row['head']} | {factor}/{result['top_absolute_share']:.3f} | {result['effective_factor_support']:.2f} | {result['routing_absolute_mass']:.3f} | {result['current_value_absolute_share']:.3f} |")
    lines += ["", "This is an exact final-factor decomposition. If a QK/value-bus term dominates, the next trace should inspect its source-token routing and the two earlier attention factors in the ordered path."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
