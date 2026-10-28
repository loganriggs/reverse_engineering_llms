#!/usr/bin/env python3
"""Resolve upstream heads in the singular ordered attention path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_singular_head_chain.json")
OUTPUT = TENSOR_ROOT / "layer8_singular_head_chain.json"
REPORT = TENSOR_ROOT / "LAYER8_SINGULAR_HEAD_CHAIN.md"


@torch.inference_mode()
def response(model, site, position, mask, heads, final_layer, factor):
    logits, tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=bool(mask), attention_layers=mask,
        attention_heads={layer: (head,) for layer, head in heads.items()},
        attention_factors={final_layer: (factor,)},
    )
    return output_fisher_response(logits, tangents)


@torch.inference_mode()
def third_order(model, site, triple, position, heads, factor):
    i, j, k = triple
    r0 = response(model, site, position, (), heads, k, factor)
    ri = response(model, site, position, (i,), heads, k, factor)
    rj = response(model, site, position, (j,), heads, k, factor)
    rk = response(model, site, position, (k,), heads, k, factor)
    rij = response(model, site, position, (i, j), heads, k, factor)
    rik = response(model, site, position, (i, k), heads, k, factor)
    rjk = response(model, site, position, (j, k), heads, k, factor)
    rijk = response(model, site, position, triple, heads, k, factor)
    return rijk - rij - rik - rjk + ri + rj + rk - r0, rijk


def decompose(model, site, triple, position, parent_heads, split_layer, factor):
    parent, exact = third_order(model, site, triple, position, parent_heads, factor)
    components = []
    for head in range(model.config.n_head):
        heads = {**parent_heads, split_layer: head}
        component, _ = third_order(model, site, triple, position, heads, factor)
        components.append(component)
    recomposed = torch.stack(components).sum(0)
    norm = parent.norm().clamp_min(1e-30)
    shares = np.asarray([float((component * parent).sum() / norm.square())
                         for component in components])
    absolute = np.abs(shares); normalized = absolute / max(absolute.sum(), 1e-30)
    return {"parent": parent, "exact": exact, "components": components,
            "shares": shares, "top_head": int(np.argmax(absolute)),
            "top_absolute_share": float(normalized.max()),
            "effective_support": float(1 / max(np.square(normalized).sum(), 1e-30)),
            "recomposition_error_to_exact": float((recomposed - parent).norm() /
                                                   exact.norm().clamp_min(1e-30))}


@torch.inference_mode()
def audit(model, site, triple, final_head, factor, position, fixed_chain=None):
    layer11, layer13, layer14 = triple
    stage13 = decompose(model, site, triple, position, {layer14: final_head}, layer13, factor)
    chosen13 = stage13["top_head"] if fixed_chain is None else fixed_chain[1]
    stage11 = decompose(model, site, triple, position,
                        {layer13: chosen13, layer14: final_head}, layer11, factor)
    chosen11 = stage11["top_head"] if fixed_chain is None else fixed_chain[0]
    chain, exact = third_order(model, site, triple, position,
                               {layer11: chosen11, layer13: chosen13, layer14: final_head}, factor)
    return {
        "chain": [chosen11, chosen13, final_head],
        "layer13": {key: value for key, value in stage13.items()
                    if key not in ("parent", "exact", "components", "shares")},
        "layer13_shares": stage13["shares"].tolist(),
        "layer11": {key: value for key, value in stage11.items()
                    if key not in ("parent", "exact", "components", "shares")},
        "layer11_shares": stage11["shares"].tolist(),
        "chain_response_norm": float(chain.norm()),
        "chain_relative_to_exact": float(chain.norm() / exact.norm().clamp_min(1e-30)),
    }


def main():
    config = json.loads(CONFIG.read_text())
    gain_config = json.loads(Path(config["parent_config"]).read_text())
    trace_config = json.loads(Path(gain_config["parent_config"]).read_text())
    factor_config = json.loads(Path(trace_config["parent_config"]).read_text())
    head_config = json.loads(Path(factor_config["parent_config"]).read_text())
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
    trace = json.loads((TENSOR_ROOT / "layer8_singular_source_trace.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    corpus_config = parent["evaluation"][trace_config["corpus"]]
    rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                         parent["queries"], 2)
    bank = {row_index: raw for query, row_index, raw in rows if query == trace_config["query"]}
    failure_site = prepare_site(model, bank[trace_config["row"]], trace_config["query"],
                                source_config, tensor_basis, directions, vectors)
    control_site = prepare_site(model, bank[trace["control_row"]], trace_config["query"],
                                source_config, tensor_basis, directions, vectors)
    triple = tuple(trace_config["triple"])
    failure = audit(model, failure_site, triple, trace_config["head"], trace_config["factor"],
                    source_config["causal_position"])
    control = audit(model, control_site, triple, trace_config["head"], trace_config["factor"],
                    source_config["causal_position"], fixed_chain=failure["chain"])
    max_error = max(failure[stage]["recomposition_error_to_exact"]
                    for stage in ("layer13", "layer11"))
    max_error = max(max_error, *(control[stage]["recomposition_error_to_exact"]
                                 for stage in ("layer13", "layer11")))
    amplification = failure["chain_response_norm"] / max(control["chain_response_norm"], 1e-30)
    hypotheses = {
        "C0_nested_head_decompositions_recompose": {"supported": bool(max_error <= 1e-4)},
        "C1_layer13_is_head_localized": {"supported": bool(
            failure["layer13"]["top_absolute_share"] >= .50 and
            failure["layer13"]["effective_support"] <= 3)},
        "C2_layer11_is_head_localized": {"supported": bool(
            failure["layer11"]["top_absolute_share"] >= .50 and
            failure["layer11"]["effective_support"] <= 3)},
        "C3_full_ordered_head_chain_is_context_amplified": {"supported": bool(amplification >= 10)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "max_recomposition_error": max_error,
                                  "amplification": amplification,
                                  "failure": failure, "control": control}, indent=2))
    lines = ["# Upstream head chain of the singular ordered path", "",
             "Layer-14 head 6 and its QK2/value-bus factor are held fixed. The path is decomposed by layer-13 head, then by layer-11 head after choosing the dominant failure component.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Selected ordered head chain: `{failure['chain']}` for layers `{list(triple)}`. Maximum recomposition error: `{max_error:.3e}`. Fixed-chain failure/control amplification: `{amplification:.1f}x`.", "",
              "| Stage | Top head/share | Effective support | Control top head/share |",
              "|---|---:|---:|---:|",
              f"| Layer 13 | {failure['layer13']['top_head']}/{failure['layer13']['top_absolute_share']:.3f} | {failure['layer13']['effective_support']:.2f} | {control['layer13']['top_head']}/{control['layer13']['top_absolute_share']:.3f} |",
              f"| Layer 11 | {failure['layer11']['top_head']}/{failure['layer11']['top_absolute_share']:.3f} | {failure['layer11']['effective_support']:.2f} | {control['layer11']['top_head']}/{control['layer11']['top_absolute_share']:.3f} |",
              "", "A localized chain is a compact exceptional circuit even though its gain is distributed. A diffuse upstream stage would instead make the final clean head the endpoint of a broad subspace computation."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
