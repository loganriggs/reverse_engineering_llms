#!/usr/bin/env python3
"""Trace a singular third-order QK/value-bus path to source token positions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_singular_source_trace.json")
OUTPUT = TENSOR_ROOT / "layer8_singular_source_trace.json"
REPORT = TENSOR_ROOT / "LAYER8_SINGULAR_SOURCE_TRACE.md"


@torch.inference_mode()
def response(model, site, position, mask, layer=None, head=None, factor=None, sources=None):
    logits, tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=bool(mask), attention_layers=mask,
        attention_heads=None if layer is None else {layer: (head,)},
        attention_factors=None if factor is None else {layer: (factor,)},
        attention_sources=None if sources is None else {layer: tuple(sources)},
    )
    return output_fisher_response(logits, tangents)


@torch.inference_mode()
def factor_residual(model, site, triple, position, head, factor, sources=None):
    i, j, k = triple
    empty = response(model, site, position, ())
    ri = response(model, site, position, (i,)); rj = response(model, site, position, (j,))
    rij = response(model, site, position, (i, j))
    kwargs = {"layer": k, "head": head, "factor": factor, "sources": sources}
    rk = response(model, site, position, (k,), **kwargs)
    rik = response(model, site, position, (i, k), **kwargs)
    rjk = response(model, site, position, (j, k), **kwargs)
    rijk = response(model, site, position, triple, **kwargs)
    return rijk - rij - rik - rjk + ri + rj + rk - empty, rijk


@torch.inference_mode()
def audit(model, site, raw, config, position, tokenizer):
    triple = tuple(config["triple"]); layer = triple[-1]
    full, exact = factor_residual(
        model, site, triple, position, config["head"], config["factor"], None,
    )
    recomposed = torch.zeros_like(full)
    shares = []
    for source in range(position + 1):
        component, _ = factor_residual(
            model, site, triple, position, config["head"], config["factor"], (source,),
        )
        recomposed.add_(component)
        shares.append(float((component * full).sum()))
        del component
    norm = full.norm().clamp_min(1e-30)
    shares = np.asarray(shares) / float(norm.square())
    absolute = np.abs(shares); normalized = absolute / max(absolute.sum(), 1e-30)
    order = np.argsort(absolute)[::-1][:config["top_sources"]]
    top = []
    for source in order:
        start = max(0, int(source) - 6)
        top.append({"position": int(source), "token_id": int(raw[source]),
                    "token": tokenizer.decode([int(raw[source])]),
                    "prefix": tokenizer.decode(raw[start:int(source) + 1].tolist()),
                    "projection_share": float(shares[source]),
                    "absolute_mass": float(normalized[source])})
    return {
        "relative_factor_norm": float(norm / exact.norm().clamp_min(1e-30)),
        "recomposition_error_to_exact": float((recomposed - full).norm() /
                                              exact.norm().clamp_min(1e-30)),
        "top_source": int(np.argmax(absolute)),
        "top_absolute_share": float(normalized.max()),
        "effective_source_support": float(1 / max(np.square(normalized).sum(), 1e-30)),
        "pre_final_absolute_mass": float(normalized[:-1].sum()),
        "self_absolute_mass": float(normalized[-1]),
        "top_sources": top,
    }


def main():
    config = json.loads(CONFIG.read_text())
    factor_config = json.loads(Path(config["parent_config"]).read_text())
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
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    corpus_config = parent["evaluation"][config["corpus"]]
    rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                         parent["queries"], 2)
    candidates = [(row_index, raw) for query, row_index, raw in rows if query == config["query"]]
    failure_row = next(raw for row_index, raw in candidates if row_index == config["row"])
    control_index, control_row = next((row_index, raw) for row_index, raw in candidates
                                      if row_index != config["row"])
    failure_site = prepare_site(model, failure_row, config["query"], source_config,
                                tensor_basis, directions, vectors)
    control_site = prepare_site(model, control_row, config["query"], source_config,
                                tensor_basis, directions, vectors)
    failure = audit(model, failure_site, failure_row, config, source_config["causal_position"], tokenizer)
    control = audit(model, control_site, control_row, config, source_config["causal_position"], tokenizer)
    amplification = failure["relative_factor_norm"] / max(control["relative_factor_norm"], 1e-30)
    max_error = max(failure["recomposition_error_to_exact"], control["recomposition_error_to_exact"])
    hypotheses = {
        "T0_source_decomposition_recomposes": {"supported": bool(max_error <= 1e-4)},
        "T1_source_path_is_sparse": {"supported": bool(
            failure["top_absolute_share"] >= .30 and failure["effective_source_support"] <= 10)},
        "T2_source_path_is_context_amplified": {"supported": bool(amplification >= 10)},
        "T3_path_is_not_self_only": {"supported": bool(failure["pre_final_absolute_mass"] >= .50)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "control_row": control_index, "amplification": amplification,
                                  "failure": failure, "control": control}, indent=2))
    lines = ["# Source-token trace of the singular QK/value-bus path", "",
             f"The representative failure is `{config['corpus']}` query {config['query']} row {config['row']}, triple `{config['triple']}`, head {config['head']}, factor `{config['factor']}`. The matched control is row {control_index}.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum recomposition error relative to exact response: `{max_error:.3e}`. Failure/control factor amplification: `{amplification:.1f}x`. Top-source share: `{failure['top_absolute_share']:.3f}`. Effective support: `{failure['effective_source_support']:.2f}` positions. Pre-final mass: `{failure['pre_final_absolute_mass']:.3f}`.", "",
              "| Position | Token | Signed projection | Absolute mass | Local prefix |",
              "|---:|---|---:|---:|---|"]
    for row in failure["top_sources"]:
        token = row["token"].replace("|", "\\|").replace("\n", "\\n")
        prefix = row["prefix"].replace("|", "\\|").replace("\n", "\\n")
        lines.append(f"| {row['position']} | `{token}` | {row['projection_share']:.3f} | {row['absolute_mass']:.3f} | `{prefix}` |")
    lines += ["", "These source tokens explain one singular path. Semantic interpretation requires checking recurrence on newly discovered states with the same head/factor signature."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
