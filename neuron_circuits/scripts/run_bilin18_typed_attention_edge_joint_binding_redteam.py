#!/usr/bin/env python3
"""Test coordinate binding and layer assignment in the joint typed-edge program."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate
from run_bilin18_typed_attention_edge_specificity_redteam import randomized_support
from run_bilin18_typed_attention_edge_vocabulary import CATEGORIES, discovery_scores, token_categories


CONFIG = Path("configs/bilin18_typed_attention_edge_joint_binding_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_typed_attention_edge_joint_binding_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_TYPED_ATTENTION_EDGE_JOINT_BINDING_REDTEAM.md"


@torch.inference_mode()
def binding_supports(model, discovery, categories, config):
    heads = model.config.n_head
    distances = len(config["distance_bins"]) - 1
    category_count = len(CATEGORIES)
    selection_config = {**config, "batch_size": config["selection_batch_size"]}
    supports, selected_by_layer = {}, {}
    for layer in config["layers"]:
        print("select binding controls layer", layer, flush=True)
        probes, _, _ = operator_probe_vectors(
            model.blocks[layer], config["operator_probes"], config["seed"] + layer,
        )
        scores, _ = discovery_scores(model, layer, discovery, categories, selection_config, probes)
        selected = scores.argsort(descending=True).cpu().numpy()[:128]
        selected_by_layer[layer] = selected
        generator = np.random.default_rng(config["seed"] + 1000 + layer)
        supports[layer] = {"fine128": selected}
        for axis in ("category", "distance", "head"):
            for index in range(3):
                supports[layer][f"{axis}_{index}"] = randomized_support(
                    selected, axis, heads, distances, category_count, generator,
                )
    for layer in config["layers"]:
        next_layer = config["layers"][(config["layers"].index(layer) + 1) % len(config["layers"])]
        supports[layer]["layer_shift"] = selected_by_layer[next_layer]
    return supports


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
    supports = binding_supports(model, discovery, categories, config)
    results, maximum_error = evaluate(model, evaluation, categories, config, supports)
    methods = results["5"]["methods"]
    selected = methods["fine128"]["kl_recovery"]
    controls = {
        axis: [methods[f"{axis}_{index}"]["kl_recovery"] for index in range(3)]
        for axis in ("category", "distance", "head")
    }
    hypotheses = {}
    for axis, key in (("category", "B1_category_binding_matters"), ("distance", "B2_distance_binding_matters"), ("head", "B3_head_binding_matters")):
        p95 = float(np.quantile(controls[axis], .95))
        hypotheses[key] = {"supported": selected >= p95 + .10, "selected_minus_p95": selected - p95}
    hypotheses["B4_layer_assignment_matters"] = {
        "supported": selected >= methods["layer_shift"]["kl_recovery"] + .10,
        "difference": selected - methods["layer_shift"]["kl_recovery"],
    }
    hypotheses["B5_selected_program_prediction_fidelity_replicates"] = {
        "supported": methods["fine128"]["top1_agreement"] >= .75,
    }
    hypotheses["native_branch_matches_checkpoint"] = {
        "supported": maximum_error <= 1e-4, "maximum_error": maximum_error,
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results, "controls": controls}, indent=2, allow_nan=False))
    lines = [
        "# Red team: coordinate binding in the joint typed-edge program", "",
        "Fine-128 is compared on a third held-out FineWeb split with controls that preserve two coordinates while randomizing the third in every layer, plus a cyclic shift of each layer's selected program to another layer.", "",
        f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Program | KL recovery | CE delta | Top-1 agreement |", "|---|---:|---:|---:|"]
    for name in config["methods"]:
        value = methods[name]
        lines.append(f"| {name} | {value['kl_recovery']:.4f} | {value['mean_ce_delta']:+.4f} | {value['top1_agreement']:.4f} |")
    lines += [
        "", "## Guardrails", "",
        "Only three randomized programs are used per coordinate, so their empirical p95 is a coarse control. Randomization preserves the other-coordinate group counts but may create duplicate types, as in the earlier local test. A layer shift preserves every selected triple globally but changes where it executes. These tests establish coordinate and layer specificity of the executable program, not that the human category partition is uniquely optimal.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "selected": selected, "controls": controls, "layer_shift": methods["layer_shift"]}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
