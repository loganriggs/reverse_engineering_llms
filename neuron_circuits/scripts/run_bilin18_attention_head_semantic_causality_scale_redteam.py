#!/usr/bin/env python3
"""Red-team the semantic-head causal null across edit scale and larger panels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_attention_head_semantic_causality import evaluate_candidate, summarize
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_attention_head_semantic_causality_scale_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_attention_head_semantic_causality_scale_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_ATTENTION_HEAD_SEMANTIC_CAUSALITY_SCALE_REDTEAM.md"


@torch.inference_mode()
def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)])
    source = np.load(config["corpus"], mmap_mode="r")
    calibration_raw = np.asarray(source[config["calibration_start"]:config["calibration_start"] + config["calibration_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    evaluation_raw = np.asarray(source[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1], dtype=np.int64)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    records = []; maximum_error = 0.0
    for candidate_index, candidate in enumerate(config["candidates"]):
        scale_rows = {}
        for scale in config["removal_scales"]:
            print("scale redteam", candidate_index + 1, "/", len(config["candidates"]), candidate, "scale", scale, flush=True)
            current = {**config, "removal_scale": scale}
            calibration = evaluate_candidate(model, calibration_raw, config["positions"], flags, candidate, current, config["seed"] + candidate_index)
            evaluation = evaluate_candidate(model, evaluation_raw, config["positions"], flags, candidate, current, config["seed"] + 100 + candidate_index)
            maximum_error = max(maximum_error, calibration["maximum_native_error"], evaluation["maximum_native_error"])
            scale_rows[str(scale)] = summarize(calibration, evaluation, candidate["head"])
        signs = [scale_rows[str(scale)]["selected"]["sign"] for scale in config["removal_scales"]]
        records.append({**candidate, "scales": scale_rows, "same_sign": len(set(signs)) == 1})
    tenth = [row["scales"]["0.1"]["selected"] for row in records]
    full = [row["scales"]["1.0"]["selected"] for row in records]
    passes_by_scale = {
        str(scale): sum(row["scales"][str(scale)]["selected"]["causal_auc"] >= .60 for row in records)
        for scale in config["removal_scales"]
    }
    stable = [row["same_sign"] and sum(row["scales"][str(scale)]["selected"]["causal_auc"] >= .55 for scale in config["removal_scales"]) >= 3 for row in records]
    hypotheses = {
        "R0_manual_native_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
        "R1_small_ablation_recovers_probe_roles": {"supported": sum(row["causal_auc"] >= .60 for row in tenth) >= 4},
        "R2_selected_heads_beat_controls_locally": {"supported": sum(row["causal_auc"] >= row["control_median_auc"] + .05 for row in tenth) >= 4},
        "R3_causal_role_is_scale_stable": {"supported": sum(stable) >= 4, "passing_candidates": int(sum(stable))},
        "R4_full_ablation_null_was_low_power": {"supported": sum(row["causal_auc"] >= .60 for row in full) >= 4},
        "R5_broad_head_semantic_ontology_remains_falsified": {"supported": max(passes_by_scale.values()) < 4, "passes_by_scale": passes_by_scale},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "records": records}, indent=2, allow_nan=False))
    lines = ["# Scale and power red team for semantic attention-head causality", "", "The six frozen candidates are retested on larger, later FineWeb calibration and evaluation panels. Exact head-write removal is scaled from 10 to 100 percent. Every scale uses the same examples, while the causal sign is independently fixed on calibration data. This distinguishes a nonlinear whole-head-ablation failure from a genuine lack of stable semantic causal specialization.", "", f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Candidate | AUROC at 0.1/0.25/0.5/1.0 | Rank at 0.1/1.0 | Calibration sign stable |", "|---|---:|---:|---:|"]
    for row in records:
        aucs = "/".join(f"{row['scales'][str(scale)]['selected']['causal_auc']:.3f}" for scale in config["removal_scales"])
        ranks = f"{row['scales']['0.1']['selected']['rank']}/{row['scales']['1.0']['selected']['rank']}"
        lines.append(f"| L{row['layer']} H{row['head']} `{row['feature']}` | {aucs} | {ranks} | {row['same_sign']} |")
    lines += ["", "## Guardrails", "", "Scaling the same ablation checks local-versus-large intervention behavior but does not create an infinitesimal derivative: numerical disagreement and class-score noise become relatively more important at 10 percent. Larger panels improve sign calibration for rare labels but the six candidates remain selected from earlier exploratory probes. A stable source-routing role can survive even if this broad output-class role fails."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "records": [{"layer":r["layer"],"head":r["head"],"feature":r["feature"],"aucs":{s:r["scales"][s]["selected"]["causal_auc"] for s in r["scales"]},"same_sign":r["same_sign"]} for r in records]}, indent=2))
    print(REPORT)


if __name__ == "__main__": main()
