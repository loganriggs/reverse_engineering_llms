#!/usr/bin/env python3
"""Test whether Layer-6-to-final transport is a context-gated bilinear map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_finite_transport import collect_events, edge_values, fidelity
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_context_transport.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_context_transport.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_CONTEXT_TRANSPORT.md"


def context_encoder(train, ranks):
    count = train["direction_count"] * 2
    sites = train["base_input"][::count]
    mean = sites.mean(0); scale = sites.std(0).clamp_min(1e-6)
    standardized = (sites - mean) / scale
    _, _, vh = torch.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ vh.T
    score_scale = scores.std(0).clamp_min(1e-6)
    return {"mean": mean, "scale": scale, "basis": vh.T,
            "score_scale": score_scale, "ranks": ranks}


def encode(events, encoder, rank):
    standardized = (events["base_input"] - encoder["mean"]) / encoder["scale"]
    return (standardized @ encoder["basis"][:, :rank]) / encoder["score_scale"][:rank]


def positions(events, allowed):
    values = events["position"].long().flatten()
    result = torch.zeros(len(values), len(allowed))
    for index, position in enumerate(allowed): result[:, index] = values == position
    return result


def features(x, context=None):
    if context is None or context.shape[1] == 0: return x
    interaction = torch.einsum("ni,nc->nic", x, context).flatten(1)
    return torch.cat([x, interaction], 1)


def fit(features_value, target, ridge):
    gram = features_value.T @ features_value
    penalty = ridge * float(torch.trace(gram)) / len(gram)
    return torch.linalg.solve(gram + penalty * torch.eye(len(gram)), features_value.T @ target)


def score(events, prediction, pairs):
    edge_target = edge_values(events["base_output"] + events["output"], pairs) - edge_values(events["base_output"], pairs)
    edge_prediction = edge_values(events["base_output"] + prediction, pairs) - edge_values(events["base_output"], pairs)
    return {"coordinate_fidelity": fidelity(events["output"], prediction),
            "edge_change_fidelity": fidelity(edge_target, edge_prediction)}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); graph = json.loads(Path(config["selected_graph"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    states, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(states.float(), full_matrices=False)
    natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, forms, natural_basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (natural_basis @ rotation.detach()).float()
    canonical = torch.eye(32, device="cuda"); layer = config["layer"]
    print("collect train", flush=True)
    train = collect_events(model, layer, config["transport_train"], config["positions"], axes,
                           canonical, config["small_scale"], config["batch_size"])
    encoder = context_encoder(train, config["context_ranks"])
    pairs = [tuple(pair) for pair in graph["selected_pairs"]]
    train_position = positions(train, config["positions"])
    models = {"global": fit(features(train["input"]), train["output"], config["ridge"]),
              "position": fit(features(train["input"], train_position), train["output"], config["ridge"])}
    for rank in config["context_ranks"]:
        context = encode(train, encoder, rank)
        models[f"context_{rank}"] = fit(features(train["input"], context), train["output"], config["ridge"])
        if rank == 16:
            combined = torch.cat([context, train_position], 1)
            models["context_16_position"] = fit(features(train["input"], combined), train["output"], config["ridge"])
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        events = collect_events(model, layer, spec, config["positions"], axes, canonical,
                                config["large_scale"], config["batch_size"])
        position_context = positions(events, config["positions"])
        rows = {}
        for name, model_map in models.items():
            if name == "global": design = features(events["input"])
            elif name == "position": design = features(events["input"], position_context)
            elif name == "context_16_position":
                design = features(events["input"], torch.cat([encode(events, encoder, 16), position_context], 1))
            else:
                rank = int(name.split("_")[-1]); design = features(events["input"], encode(events, encoder, rank))
            rows[name] = score(events, design @ model_map, pairs)
        results[corpus] = rows
    g0 = all(.70 <= rows["global"]["coordinate_fidelity"] <= .85 for rows in results.values())
    g1 = all(rows["context_8"]["coordinate_fidelity"] >= rows["global"]["coordinate_fidelity"] + .05 for rows in results.values())
    g2 = all(rows["context_16"]["coordinate_fidelity"] >= .90 for rows in results.values())
    g3 = all(rows["context_16"]["edge_change_fidelity"] >= .90 for rows in results.values())
    g4 = all(rows["position"]["coordinate_fidelity"] < rows["global"]["coordinate_fidelity"] + .05 for rows in results.values())
    fw = results["fineweb"]["context_16"]["coordinate_fidelity"]
    g5 = all(results[c]["context_16"]["coordinate_fidelity"] >= fw - .05 for c in ("wikitext", "tinystories"))
    g6 = all(rows["context_32"]["coordinate_fidelity"] >= rows["context_16"]["coordinate_fidelity"] - .03 for rows in results.values())
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "G0_global_transport_replicates": g0,
        "G1_low_rank_context_materially_improves": g1,
        "G2_context_gating_explains_transport": g2,
        "G3_context_gating_predicts_graph_changes": g3,
        "G4_position_is_not_the_explanation": g4,
        "G5_context_rule_transfers_ood": g5,
        "G6_more_context_does_not_overfit_badly": g6,
    }.items()}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Context-gated transport from Layer 6", "",
             "A frozen linear transport map explained about four-fifths of Layer-6 perturbation effects. This test asks whether the missing variation is a compact bilinear rule: perturbation coordinates multiplied by a low-rank summary of the unperturbed Layer-6 state. Every map is fit on small FineWeb edits and tested on edits four times larger and on OOD corpora.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Global | Position | Context r8 | Context r16 | Context r32 | r16 edge fidelity |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | {row['global']['coordinate_fidelity']:.3f} | {row['position']['coordinate_fidelity']:.3f} | "
                     f"{row['context_8']['coordinate_fidelity']:.3f} | {row['context_16']['coordinate_fidelity']:.3f} | "
                     f"{row['context_32']['coordinate_fidelity']:.3f} | {row['context_16']['edge_change_fidelity']:.3f} |")
    lines += ["", "This is a deliberately restrictive conditional model. Success would identify a compact tensor-valued routing rule. Failure would mean that local transport depends on information outside the 32 late coordinates, on richer nonlinear context, or on attention/history variables not represented by the local residual state."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
