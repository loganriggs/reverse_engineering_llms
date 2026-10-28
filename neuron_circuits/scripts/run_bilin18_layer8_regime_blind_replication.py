#!/usr/bin/env python3
"""Blindly replicate Layer-8 regular/singular path regimes on new corpora."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer8_pair_mobius_reconstruction import audit_site
from run_bilin18_layer8_suffix_attention_subset import prepare_site, selected_rows, tensor_program
from run_bilin18_layer8_suffix_route_atlas import summarize
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer8_regime_blind_replication.json")
OUTPUT = TENSOR_ROOT / "layer8_regime_blind_replication.json"
REPORT = TENSOR_ROOT / "LAYER8_REGIME_BLIND_REPLICATION.md"
CHECKPOINT = TENSOR_ROOT / "layer8_regime_blind_replication.partial.json"


def corpus_summary(records, threshold, failure_fidelity, seed):
    triples = [row for site in records for row in site["triples"]]
    fidelity = np.asarray([row["finite_effect_fidelity"] for row in triples])
    accepted = np.asarray([row["whitened_gram_error"] <= threshold for row in triples])
    good = fidelity >= failure_fidelity
    accepted_failure = float((~good[accepted]).mean()) if accepted.any() else float("nan")
    rejected_failure = float((~good[~accepted]).mean()) if (~accepted).any() else float("nan")
    failure_sites = sum(any(row["finite_effect_fidelity"] < failure_fidelity
                            for row in site["triples"]) for site in records)
    return {
        "triples": len(triples), "sites": len(records),
        "fidelity": summarize(fidelity, seed),
        "failures": int((~good).sum()), "failure_fraction": float((~good).mean()),
        "failure_sites": failure_sites, "failure_site_fraction": failure_sites / len(records),
        "certificate_coverage": float(accepted.mean()),
        "certificate_precision": float(good[accepted].mean()) if accepted.any() else float("nan"),
        "accepted_failure_rate": accepted_failure,
        "rejected_failure_rate": rejected_failure,
        "failure_enrichment": rejected_failure / max(accepted_failure, 1e-30)
        if (~accepted).any() else float("nan"),
        "rejected": int((~accepted).sum()),
    }


def main():
    config = json.loads(CONFIG.read_text())
    mobius_config = json.loads(Path(config["parent_config"]).read_text())
    synergy_config = json.loads(Path(mobius_config["parent_config"]).read_text())
    singleton_config = json.loads(Path(synergy_config["parent_config"]).read_text())
    scaling_config = json.loads(Path(singleton_config["parent_config"]).read_text())
    state_config = json.loads(Path(scaling_config["parent_config"]).read_text())
    atlas_config = json.loads(Path(state_config["parent_config"]).read_text())
    parent = json.loads(Path(atlas_config["source_config"]).read_text())
    source_config = json.loads(Path(parent["source_config"]).read_text())
    source_config = {**source_config, "queries": parent["queries"],
                     "sites_per_query": config["sites_per_query"], "budget": parent["budget"]}
    certificate_source = json.loads(Path(config["certificate_source"]).read_text())
    threshold = certificate_source["certificate"]["error"]
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    tensor_basis, directions, values, vectors = tensor_program(model, source_config, parent["queries"])
    partial = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {"results": {}}
    results = partial["results"]
    for corpus_index, (corpus, corpus_config) in enumerate(config["evaluation"].items()):
        rows = selected_rows(model, source_config, corpus_config, tensor_basis, values, vectors,
                             parent["queries"], config["sites_per_query"])
        records = results.get(corpus, {}).get("records", [])
        for number, (query, row_index, raw) in enumerate(rows[len(records):], start=len(records)):
            site = prepare_site(model, raw, query, source_config, tensor_basis, directions, vectors)
            triples = audit_site(
                model, site, parent["suffix_attention_layers"], source_config["causal_position"],
                parent["budget"], source_config["edit_norm_fraction"],
                config["finite_triples_per_batch"],
            )
            records.append({"query": query, "row": row_index, "triples": triples})
            results[corpus] = {"records": records}
            CHECKPOINT.write_text(json.dumps({"results": results}))
            del site, triples; gc.collect(); torch.cuda.empty_cache()
            print(corpus, "site", number + 1, "of", len(rows), flush=True)
        results[corpus]["summary"] = corpus_summary(
            records, threshold, config["failure_fidelity"], config["seed"] + corpus_index,
        )
    bulk = all(result["summary"]["fidelity"]["median"] >= .95
               and result["summary"]["fidelity"]["q10"] >= .80 for result in results.values())
    precision = all(result["summary"]["certificate_coverage"] >= .90
                    and result["summary"]["certificate_precision"] >= .99 for result in results.values())
    enriched = all(result["summary"]["rejected"] > 0
                   and result["summary"]["failure_enrichment"] >= 10 for result in results.values())
    rare = all(result["summary"]["failure_fraction"] < .02
               and result["summary"]["failure_site_fraction"] < .20 for result in results.values())
    hypotheses = {
        "R0_pairwise_bulk_replicates": {"supported": bool(bulk)},
        "R1_frozen_certificate_retains_precision": {"supported": bool(precision)},
        "R2_rejected_region_is_failure_enriched": {"supported": bool(enriched)},
        "R3_no_broad_third_order_regime": {"supported": bool(rare)},
    }
    OUTPUT.write_text(json.dumps({"config": config, "certificate_threshold": threshold,
                                  "hypotheses": hypotheses, "results": results}, indent=2))
    CHECKPOINT.unlink(missing_ok=True)
    lines = ["# Blind replication of Layer-8 regular/singular path regimes", "",
             "The second-order reconstruction, failure threshold, and cost-whitened error certificate were frozen before evaluating new FineWeb rows and TinyStories.",
             "", "## Locked hypotheses", ""]
    for name, item in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if item['supported'] else 'not supported'}.**")
    lines += ["", f"Frozen certificate threshold: `{threshold:.6f}`.", "",
              "| Corpus | Fidelity median/q10 | Failures/triples | Failure sites | Certificate coverage/precision | Accepted/rejected failure rate | Enrichment |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, result in results.items():
        row = result["summary"]
        lines.append(f"| {corpus} | {row['fidelity']['median']:.3f}/{row['fidelity']['q10']:.3f} | {row['failures']}/{row['triples']} | {row['failure_sites']}/{row['sites']} | {row['certificate_coverage']:.3f}/{row['certificate_precision']:.3f} | {row['accepted_failure_rate']:.4f}/{row['rejected_failure_rate']:.4f} | {row['failure_enrichment']:.1f}x |")
    lines += ["", "Bulk replication supports the low-order regular regime. Certificate enrichment is the stronger test of whether the singular boundary transfers. Zero failures or zero rejections cannot establish enrichment and are left unsupported."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
