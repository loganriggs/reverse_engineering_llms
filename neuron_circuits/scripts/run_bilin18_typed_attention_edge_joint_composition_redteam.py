#!/usr/bin/env python3
"""Red-team joint typed-edge composition with all-type and random controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate, select_supports
from run_bilin18_typed_attention_edge_vocabulary import token_categories


CONFIG = Path("configs/bilin18_typed_attention_edge_joint_composition_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_typed_attention_edge_joint_composition_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_TYPED_ATTENTION_EDGE_JOINT_COMPOSITION_REDTEAM.md"


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)
    ])
    categories = token_categories(flags)
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery = np.asarray(
        corpus[config["selection_start"]:config["selection_start"] + config["selection_rows"], :config["sequence_length"]],
        dtype=np.int64,
    )
    evaluation = np.asarray(
        corpus[config["evaluation_start"]:config["evaluation_start"] + config["evaluation_rows"], :config["sequence_length"] + 1],
        dtype=np.int64,
    )
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    supports = select_supports(model, discovery, categories, config)
    results, maximum_error = evaluate(model, evaluation, categories, config, supports)
    row = results["5"]
    methods = row["methods"]
    fine128_error = row["baseline_kl"] * (1 - methods["fine128"]["kl_recovery"])
    hypotheses = {
        "R0_all_types_reproduce_native": {
            "supported": methods["all"]["kl_recovery"] >= .9999 and methods["all"]["top1_agreement"] >= .999,
        },
        "R1_selected128_beats_random128": {
            "supported": methods["fine128"]["kl_recovery"] >= methods["random128"]["kl_recovery"] + .20,
            "difference": methods["fine128"]["kl_recovery"] - methods["random128"]["kl_recovery"],
        },
        "R2_selected128_has_small_absolute_error": {
            "supported": fine128_error <= .20, "mean_kl_error": fine128_error,
        },
        "R3_selected128_prediction_agreement_replicates": {
            "supported": methods["fine128"]["top1_agreement"] >= .75,
        },
        "R4_fine32_is_not_enough_for_predictions": {
            "supported": methods["fine32"]["top1_agreement"] < .50,
        },
        "native_branch_matches_checkpoint": {
            "supported": maximum_error <= 1e-4, "maximum_error": maximum_error,
        },
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Red team: joint typed-edge composition", "",
        "The six-layer closed-loop program is repeated on a new FineWeb split. An all-types program tests exact partition/execution, and fixed random 128-type programs test whether the discovery-selected vocabulary is special.", "",
        f"Maximum native-branch error: `{maximum_error:.2e}`. Fine-128 absolute mean KL error: `{fine128_error:.4f}`.", "",
        "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Method | KL recovery | CE delta | Top-1 agreement |", "|---|---:|---:|---:|"]
    for name in config["methods"]:
        value = methods[name]
        lines.append(f"| {name} | {value['kl_recovery']:.4f} | {value['mean_ce_delta']:+.4f} | {value['top1_agreement']:.4f} |")
    lines += [
        "", "## Guardrails", "",
        "Random supports are sampled independently in each layer from types active on discovery, without matching head, distance, or category marginals. Thus they are a coarse null; the earlier coordinate-preserving controls remain the stronger test of individual coordinates. The all-types branch is the implementation invariant. Normalized recovery is accompanied by absolute KL, CE, and top-1 because the no-attention denominator is intentionally severe.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "results": results}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
