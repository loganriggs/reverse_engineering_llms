#!/usr/bin/env python3
"""Compare operator-selected typed edges with frequency and magnitude pruning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_early_operator_tensor_vocabulary import operator_probe_vectors
from run_bilin18_layer01_supervised_feature_census import vocabulary_features
from run_bilin18_qkov_causal_edges import attention_details, load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_typed_attention_edge_joint_composition import evaluate
from run_bilin18_typed_attention_edge_vocabulary import (
    CATEGORIES,
    discovery_scores,
    distance_ids,
    edge_type_ids,
    token_categories,
)


CONFIG = Path("configs/bilin18_typed_attention_edge_mass_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_typed_attention_edge_mass_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_TYPED_ATTENTION_EDGE_MASS_REDTEAM.md"


@torch.inference_mode()
def type_mass_statistics(model, layer, raw, categories, config):
    heads = model.config.n_head
    dimension = model.config.n_embd
    type_count = heads * (len(config["distance_bins"]) - 1) * len(CATEGORIES)
    distance = distance_ids(config["positions"], config["sequence_length"], config["distance_bins"], "cuda")
    totals = {
        "frequency": torch.zeros(type_count, device="cuda", dtype=torch.float64),
        "routing_mass": torch.zeros(type_count, device="cuda", dtype=torch.float64),
        "transport_mass": torch.zeros(type_count, device="cuda", dtype=torch.float64),
    }
    for start in range(0, len(raw), config["selection_batch_size"]):
        batch_raw = raw[start:start + config["selection_batch_size"]]
        ids = torch.as_tensor(batch_raw[:, :config["sequence_length"]], device="cuda")
        category = torch.as_tensor(categories[batch_raw[:, :config["sequence_length"]]], device="cuda")
        x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
        for block_index, block in enumerate(model.blocks[:layer + 1]):
            mixed = block.lambdas[0] * x + block.lambdas[1] * x0
            details = attention_details(block.attn, F.rms_norm(mixed, (dimension,)), bus, model.config, model.tt)
            bus = details["value_bus"]
            if block_index == layer:
                ids_by_edge = edge_type_ids(category, distance, heads, len(CATEGORIES)).reshape(-1)
                routing = details["pattern"][:, :, config["positions"]].permute(0, 2, 1, 3).abs()
                value_norm = details["mixed_value"].float().norm(dim=-1).permute(0, 2, 1)
                transport = routing * value_norm[:, None]
                valid = ids_by_edge >= 0
                safe = ids_by_edge[valid]
                totals["frequency"].scatter_add_(0, safe, torch.ones_like(safe, dtype=torch.float64))
                totals["routing_mass"].scatter_add_(0, safe, routing.reshape(-1)[valid].double())
                totals["transport_mass"].scatter_add_(0, safe, transport.reshape(-1)[valid].double())
            pre = mixed + details["write"]
            state = F.rms_norm(pre, (dimension,))
            x = pre + block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
    return totals


@torch.inference_mode()
def select_control_supports(model, discovery, categories, config):
    selection_config = {**config, "batch_size": config["selection_batch_size"]}
    supports, coverage = {}, {}
    for layer in config["layers"]:
        print("select mass controls layer", layer, flush=True)
        probes, _, _ = operator_probe_vectors(
            model.blocks[layer], config["operator_probes"], config["seed"] + layer,
        )
        operator, _ = discovery_scores(model, layer, discovery, categories, selection_config, probes)
        masses = type_mass_statistics(model, layer, discovery, categories, config)
        supports[layer] = {
            "operator128": operator.argsort(descending=True).cpu().numpy()[:128],
            "frequency128": masses["frequency"].argsort(descending=True).cpu().numpy()[:128],
            "routing_mass128": masses["routing_mass"].argsort(descending=True).cpu().numpy()[:128],
            "transport_mass128": masses["transport_mass"].argsort(descending=True).cpu().numpy()[:128],
        }
        coverage[str(layer)] = {}
        for method, support in supports[layer].items():
            index = torch.as_tensor(support, device="cuda")
            coverage[str(layer)][method] = {
                name: float(values[index].sum() / values.sum().clamp_min(1e-30))
                for name, values in masses.items()
            }
    return supports, coverage


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
    supports, coverage = select_control_supports(model, discovery, categories, config)
    results, maximum_error = evaluate(model, evaluation, categories, config, supports)
    methods = results["5"]["methods"]
    selected = methods["operator128"]["kl_recovery"]
    hypotheses = {
        "M1_operator_beats_frequency": {"supported": selected >= methods["frequency128"]["kl_recovery"] + .10, "difference": selected - methods["frequency128"]["kl_recovery"]},
        "M2_operator_beats_routing_mass": {"supported": selected >= methods["routing_mass128"]["kl_recovery"] + .10, "difference": selected - methods["routing_mass128"]["kl_recovery"]},
        "M3_operator_beats_transport_mass": {"supported": selected >= methods["transport_mass128"]["kl_recovery"] + .10, "difference": selected - methods["transport_mass128"]["kl_recovery"]},
        "M4_operator_program_prediction_fidelity_replicates": {"supported": methods["operator128"]["top1_agreement"] >= .75},
        "M5_operator_program_is_instance_sparse": {"supported": all(coverage[str(layer)]["operator128"]["frequency"] <= .50 for layer in config["layers"])},
        "native_branch_matches_checkpoint": {"supported": maximum_error <= 1e-4, "maximum_error": maximum_error},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "coverage": coverage, "results": results}, indent=2, allow_nan=False))
    lines = [
        "# Red team: operator vocabulary versus frequency and routing mass", "",
        "The six-layer closed-loop program is selected four ways on the same discovery data: destination-operator effect, raw occurrence frequency, absolute bilinear QK-pattern mass, or pattern mass times source-value norm. Evaluation uses a fourth untouched FineWeb split.", "",
        f"Maximum native-branch error: `{maximum_error:.2e}`.", "", "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Program | KL recovery | CE delta | Top-1 agreement |", "|---|---:|---:|---:|"]
    for name in config["methods"]:
        value = methods[name]
        lines.append(f"| {name} | {value['kl_recovery']:.4f} | {value['mean_ce_delta']:+.4f} | {value['top1_agreement']:.4f} |")
    lines += ["", "| Layer | Operator-128 edge fraction | Routing-mass fraction | Transport-mass fraction |", "|---:|---:|---:|---:|"]
    for layer in config["layers"]:
        value = coverage[str(layer)]["operator128"]
        lines.append(f"| {layer} | {value['frequency']:.3f} | {value['routing_mass']:.3f} | {value['transport_mass']:.3f} |")
    lines += [
        "", "## Guardrails", "",
        "Coverage is measured on discovery at the same 16 destination positions used for ranking, while causal metrics use a disjoint split and execute all destinations. Frequency counts architectural causal edge instances, not FLOPs after an optimized grouped implementation. QK pattern is signed in this model; magnitude controls use its absolute value. Value norm remains an incomplete proxy for destination effect because it omits output projection, RMSNorm geometry, and the following bilinear tensor.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses, "coverage": coverage, "methods": methods}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
