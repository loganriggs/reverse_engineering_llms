#!/usr/bin/env python3
"""Decompose singular third-order suffix responses by final attention head."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_direct_tensor_jacobian import analytic_suffix, output_fisher_response
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_third_order_head_audit.json")
OUTPUT = TENSOR_ROOT / "layer8_third_order_head_audit.json"
REPORT = TENSOR_ROOT / "LAYER8_THIRD_ORDER_HEAD_AUDIT.md"


@torch.inference_mode()
def get_response(model, site, position, mask, head_layer=None, head=None):
    head_map = None if head_layer is None else {head_layer: (head,)}
    logits, tangents, _ = analytic_suffix(
        model, site["x8"], site["x0"], site["value_bus"], position,
        site["tangents"], include_attention=bool(mask), attention_layers=mask,
        attention_heads=head_map,
    )
    return output_fisher_response(logits, tangents)


@torch.inference_mode()
def audit(model, site, triple, position):
    i, j, k = triple
    empty = get_response(model, site, position, ())
    ri = get_response(model, site, position, (i,)); rj = get_response(model, site, position, (j,))
    rk = get_response(model, site, position, (k,)); rij = get_response(model, site, position, (i, j))
    rik = get_response(model, site, position, (i, k)); rjk = get_response(model, site, position, (j, k))
    rijk = get_response(model, site, position, triple)
    residual = rijk - rij - rik - rjk + ri + rj + rk - empty
    residual_norm = residual.norm().clamp_min(1e-30)
    static = -rij + ri + rj - empty
    head_residuals = []
    for head in range(model.config.n_head):
        rkh = get_response(model, site, position, (k,), k, head)
        rikh = get_response(model, site, position, (i, k), k, head)
        rjkh = get_response(model, site, position, (j, k), k, head)
        rijkh = get_response(model, site, position, triple, k, head)
        head_residuals.append(rijkh - rikh - rjkh + rkh + static)
    recomposed = torch.stack(head_residuals).sum(0)
    shares = np.asarray([float((row * residual).sum() / residual_norm.square())
                         for row in head_residuals])
    absolute = np.abs(shares); normalized = absolute / max(absolute.sum(), 1e-30)
    recomposed_norm = recomposed.norm().clamp_min(1e-30)
    recomposed_shares = np.asarray([
        float((row * recomposed).sum() / recomposed_norm.square()) for row in head_residuals
    ])
    recomposed_absolute = np.abs(recomposed_shares)
    recomposed_normalized = recomposed_absolute / max(recomposed_absolute.sum(), 1e-30)
    return {
        "relative_residual_norm": float(residual_norm / rijk.norm().clamp_min(1e-30)),
        "recomposition_error": float((recomposed - residual).norm() / residual_norm),
        "recomposition_error_to_exact_response": float(
            (recomposed - residual).norm() / rijk.norm().clamp_min(1e-30)
        ),
        "head_projection_shares": shares.tolist(),
        "top_head": int(np.argmax(absolute)),
        "top_absolute_share": float(normalized.max()),
        "effective_head_support": float(1 / max(np.square(normalized).sum(), 1e-30)),
        "recomposed_top_head": int(np.argmax(recomposed_absolute)),
        "recomposed_top_absolute_share": float(recomposed_normalized.max()),
        "recomposed_effective_head_support": float(
            1 / max(np.square(recomposed_normalized).sum(), 1e-30)
        ),
    }


def main():
    config = json.loads(CONFIG.read_text())
    stability_config = json.loads(Path(config["parent_config"]).read_text())
    mobius_config = json.loads(Path(stability_config["parent_config"]).read_text())
    synergy_config = json.loads(Path(mobius_config["parent_config"]).read_text())
    singleton_config = json.loads(Path(synergy_config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": 2, "budget": parent["budget"]}
    stability = json.loads((TENSOR_ROOT / "layer8_program_stability.json").read_text())
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    row_banks = {}
    for corpus, corpus_config in parent["evaluation"].items():
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], 2)
        row_banks[corpus] = {(query, row_index): raw for query, row_index, raw in rows}
    failures = [(corpus, row) for corpus, result in stability["results"].items()
                for row in result["records"] if row["failure"]]
    records = []
    for number, (corpus, failure) in enumerate(failures):
        query, row_index, triple = failure["query"], failure["row"], tuple(failure["mask"])
        candidates = [(key, raw) for key, raw in row_banks[corpus].items()
                      if key[0] == query and key[1] != row_index]
        if not candidates:
            raise RuntimeError(f"no matched control for {corpus} query {query}")
        control_key, control_raw = candidates[0]
        failure_site = prepare_site(model, row_banks[corpus][(query, row_index)], query,
                                    source_config, tensor_basis, directions, vectors)
        control_site = prepare_site(model, control_raw, query, source_config,
                                    tensor_basis, directions, vectors)
        failure_result = audit(model, failure_site, triple, source_config["causal_position"])
        control_result = audit(model, control_site, triple, source_config["causal_position"])
        records.append({"corpus": corpus, "query": query, "row": row_index,
                        "control_row": control_key[1], "mask": list(triple),
                        "finite_effect_fidelity": failure["finite_effect_fidelity"],
                        "failure": failure_result, "control": control_result,
                        "amplification": failure_result["relative_residual_norm"] /
                        max(control_result["relative_residual_norm"], 1e-30)})
        print("head audit", number + 1, "of", len(failures), flush=True)
    max_recomposition = max(max(row[side]["recomposition_error"] for side in ("failure", "control"))
                            for row in records)
    top_share = float(np.median([row["failure"]["top_absolute_share"] for row in records]))
    support = float(np.median([row["failure"]["effective_head_support"] for row in records]))
    amplification = float(np.median([row["amplification"] for row in records]))
    hypotheses = {
        "H0_head_decomposition_recomposes": {"supported": bool(max_recomposition <= 1e-4)},
        "H1_singular_residual_is_head_localized": {"supported": bool(top_share >= .50 and support <= 3)},
        "H2_singular_residual_is_amplified": {"supported": bool(amplification >= 5)},
        "H3_singular_residual_is_diffuse": {"supported": bool(support > 5)},
    }
    summary = {"failures": len(records), "max_recomposition_error": max_recomposition,
               "median_top_absolute_share": top_share,
               "median_effective_head_support": support,
               "median_amplification": amplification}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses,
                                  "summary": summary, "records": records}, indent=2))
    lines = ["# Head decomposition of singular third-order suffix paths", "",
             "Every below-0.8 Mobius failure is paired with the same tensor query and layer triple at the other selected site in its corpus. The third-order response is decomposed exactly by head at the last selected attention layer.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum recomposition error: `{max_recomposition:.3e}`. Median top-head share: `{top_share:.3f}`. Median effective support: `{support:.2f}` heads. Median matched amplification: `{amplification:.1f}x`.", "",
              "| Corpus | Query/row | Triple | Top head/share | Effective support | Relative residual failure/control | Amplification |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in records:
        failure_row = row["failure"]; control = row["control"]
        lines.append(f"| {row['corpus']} | {row['query']}/{row['row']} | {row['mask']} | {failure_row['top_head']}/{failure_row['top_absolute_share']:.3f} | {failure_row['effective_head_support']:.2f} | {failure_row['relative_residual_norm']:.3e}/{control['relative_residual_norm']:.3e} | {row['amplification']:.1f}x |")
    lines += ["", "Head localization identifies only the final factor of the ordered path. A diffuse result would motivate subspace rather than head-level decomposition; a concentrated result motivates tracing the dominant head's QK and value factors backward."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
