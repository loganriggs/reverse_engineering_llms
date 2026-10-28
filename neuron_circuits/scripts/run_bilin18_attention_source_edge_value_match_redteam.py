#!/usr/bin/env python3
"""Value-norm and heavy-tail red team for source-feature attention edges."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_attention_source_edge_causality import evaluate, select_sites
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_attention_source_edge_value_match_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_attention_source_edge_value_match_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_ATTENTION_SOURCE_EDGE_VALUE_MATCH_REDTEAM.md"


def finalize(config, records):
    maximum_error = max(row["maximum_native_error"] for row in records)
    hypotheses = {
        "V0_native_branch_is_exact": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
        "V1_local_privilege_survives_value_matching": {"supported": sum(row["local_effect_ratio"] >= 1.5 for row in records) >= 5},
        "V2_local_privilege_is_bootstrap_robust": {"supported": sum(row["local_bootstrap_interval"][0] > 1 for row in records) >= 5},
        "V3_local_privilege_is_not_top_tail_only": {"supported": sum(row["local_trimmed_ratio"] >= 1.5 for row in records) >= 4},
        "V4_downstream_privilege_survives_value_matching": {"supported": sum(row["downstream_kl_ratio"] >= 1.5 and row["downstream_bootstrap_interval"][0] > 1 for row in records) >= 4},
        "V5_broad_sitewise_role_is_supported": {"supported": sum(row["local_site_win_rate"] >= .60 for row in records) >= 5},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "records": records}, indent=2, allow_nan=False))
    lines = ["# Value-norm and heavy-tail red team for attention source roles", "", "The frozen source-feature edges are tested on another untouched FineWeb split. Controls retain source count and relative-distance bins and now preferentially match the native value-vector norm within each bin. Aggregate, sitewise, 95%-trimmed, and site-bootstrap effects are separated.", "", f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Candidate | Local aggregate / trimmed / CI | Local win / median log2 | KL aggregate / trimmed / CI |", "|---|---:|---:|---:|"]
    for row in records:
        local_ci = row["local_bootstrap_interval"]; kl_ci = row["downstream_bootstrap_interval"]
        lines.append(f"| L{row['layer']} H{row['head']} `{row['source_feature']}` | {row['local_effect_ratio']:.2f} / {row['local_trimmed_ratio']:.2f} / [{local_ci[0]:.2f},{local_ci[1]:.2f}] | {row['local_site_win_rate']:.2f} / {row['median_local_log2_ratio']:+.2f} | {row['downstream_kl_ratio']:.2f} / {row['downstream_trimmed_ratio']:.2f} / [{kl_ci[0]:.2f},{kl_ci[1]:.2f}] |")
    lines += ["", "## Guardrails", "", "Value-norm matching controls one source of large edge effects but not value direction, exact token identity, or overlapping categories. Bootstrap intervals resample sites and therefore quantify context variability for this panel, not candidate-selection uncertainty. Aggregate and trimmed ratios can support a sparse conditional role even when median log ratio or win rate rejects a broad role; those are intentionally distinct ontologies."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "records": records}, indent=2))
    print(REPORT)


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    index_text = os.environ.get("BILIN_VALUE_EDGE_CANDIDATE")
    if os.environ.get("BILIN_VALUE_EDGE_AGGREGATE"):
        records = [json.loads((TENSOR_ROOT / f"bilin18_attention_source_edge_value_match_partial_{index}.json").read_text())["record"] for index in range(len(config["candidates"]))]
        finalize(config, records); return
    index = int(index_text) if index_text is not None else 0
    candidate = config["candidates"][index]
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([token])) for token in range(tokenizer.vocab_size)])
    source = np.load(config["corpus"], mmap_mode="r")
    raw = np.asarray(source[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    feature = FEATURES.index(candidate["source_feature"])
    sites = select_sites(raw, config["positions"], flags[:, feature], config, config["seed"] + index)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    print("value-match source edge", index + 1, "/", len(config["candidates"]), candidate, "sites", len(sites), flush=True)
    record = {**candidate, **evaluate(model, raw, sites, candidate, config)}
    if index_text is not None:
        destination = TENSOR_ROOT / f"bilin18_attention_source_edge_value_match_partial_{index}.json"
        destination.write_text(json.dumps({"record": record}, indent=2, allow_nan=False)); print(destination); return
    finalize(config, [record])


if __name__ == "__main__": main()
