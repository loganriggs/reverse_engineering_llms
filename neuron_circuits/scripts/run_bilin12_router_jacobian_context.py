#!/usr/bin/env python3
"""Predict sitewise Layer-6 transport Jacobians from local residual context."""

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


CONFIG = Path("configs/bilin12_router_jacobian_context.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_jacobian_context.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_JACOBIAN_CONTEXT.md"


def sites(events):
    count = events["direction_count"] * 2
    return {key: value.reshape(-1, count, value.shape[-1]) for key, value in events.items()
            if isinstance(value, torch.Tensor)}


def site_jacobians(events):
    grouped = sites(events); x, y = grouped["input"], grouped["output"]
    gram = x.transpose(1, 2) @ x
    penalty = 1e-7 * gram.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
    identity = torch.eye(32)[None]
    return torch.linalg.solve(gram + penalty * identity, x.transpose(1, 2) @ y)


def fit_pca_context(values, max_rank):
    mean = values.mean(0); centered = values - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:max_rank].T
    scores = centered @ basis; scale = scores.std(0).clamp_min(1e-6)
    return mean, basis, scale


def context_scores(values, encoder, rank):
    mean, basis, scale = encoder
    return ((values - mean) @ basis[:, :rank]) / scale[:rank]


def design(scores):
    return torch.cat([torch.ones(len(scores), 1), scores], 1)


def fit_jacobians(scores, jacobians, ridge):
    x = design(scores); y = jacobians.flatten(1)
    gram = x.T @ x; penalty = ridge * float(torch.trace(gram)) / len(gram)
    regularizer = torch.eye(len(gram)); regularizer[0, 0] = 0
    return torch.linalg.solve(gram + penalty * regularizer, x.T @ y)


def predict_jacobians(scores, weight):
    return (design(scores) @ weight).reshape(-1, 32, 32)


def apply_jacobians(events, jacobians):
    grouped = sites(events)
    return (grouped["input"] @ jacobians).reshape(-1, 32)


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
    natural, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False)
    natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, forms, natural_basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (natural_basis @ rotation.detach()).float()
    canonical = torch.eye(32, device="cuda"); layer = config["layer"]
    print("collect train", flush=True)
    train = collect_events(model, layer, config["transport_train"], config["positions"], axes,
                           canonical, config["small_scale"], config["batch_size"])
    train_sites = sites(train); train_j = site_jacobians(train)
    full_context = train_sites["base_input_full"][:, 0]
    late_context = train_sites["base_input"][:, 0]
    max_rank = max(config["context_ranks"])
    full_encoder = fit_pca_context(full_context, max_rank)
    late_mean = late_context.mean(0); late_scale = late_context.std(0).clamp_min(1e-6)
    zero_scores = torch.empty(len(train_j), 0)
    models = {"global": fit_jacobians(zero_scores, train_j, config["ridge"]),
              "late32": fit_jacobians((late_context - late_mean) / late_scale, train_j, config["ridge"])}
    for rank in config["context_ranks"]:
        models[f"full_{rank}"] = fit_jacobians(context_scores(full_context, full_encoder, rank), train_j, config["ridge"])
    results = {}; pairs = [tuple(pair) for pair in graph["selected_pairs"]]
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        small = collect_events(model, layer, spec, config["positions"], axes, canonical,
                               config["small_scale"], config["batch_size"])
        large = collect_events(model, layer, spec, config["positions"], axes, canonical,
                               config["large_scale"], config["batch_size"])
        large_sites = sites(large); full = large_sites["base_input_full"][:, 0]
        late = large_sites["base_input"][:, 0]
        rows = {"oracle": score(large, apply_jacobians(large, site_jacobians(small)), pairs)}
        rows["global"] = score(large, apply_jacobians(large, predict_jacobians(torch.empty(len(full), 0), models["global"])), pairs)
        late_scores = (late - late_mean) / late_scale
        rows["late32"] = score(large, apply_jacobians(large, predict_jacobians(late_scores, models["late32"])), pairs)
        for rank in config["context_ranks"]:
            context = context_scores(full, full_encoder, rank)
            prediction = apply_jacobians(large, predict_jacobians(context, models[f"full_{rank}"]))
            rows[f"full_{rank}"] = score(large, prediction, pairs)
        results[corpus] = rows
    j0 = all(rows["oracle"]["coordinate_fidelity"] >= .99 for rows in results.values())
    j1 = all(rows["full_32"]["coordinate_fidelity"] >= rows["global"]["coordinate_fidelity"] + .05 for rows in results.values())
    j2 = all(rows["full_64"]["coordinate_fidelity"] >= .90 for rows in results.values())
    j3 = all(rows["full_64"]["edge_change_fidelity"] >= .90 for rows in results.values())
    j4 = all(rows["full_32"]["coordinate_fidelity"] >= rows["late32"]["coordinate_fidelity"] + .05 for rows in results.values())
    eligible = [rank for rank in config["context_ranks"] if rank <= 128]
    best_rank = max(eligible, key=lambda rank: results["fineweb"][f"full_{rank}"]["coordinate_fidelity"])
    fw = results["fineweb"][f"full_{best_rank}"]["coordinate_fidelity"]
    j5 = all(results[c][f"full_{best_rank}"]["coordinate_fidelity"] >= fw - .05 for c in ("wikitext", "tinystories"))
    j6 = all(any(rows[f"full_{rank}"]["coordinate_fidelity"] >= rows["global"]["coordinate_fidelity"] +
                     .95 * (rows["oracle"]["coordinate_fidelity"] - rows["global"]["coordinate_fidelity"])
                     for rank in eligible) for rows in results.values())
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "J0_sitewise_jacobian_is_the_correct_local_object": j0,
        "J1_full_context_materially_improves": j1,
        "J2_full_context_explains_transport": j2,
        "J3_full_context_explains_graph_changes": j3,
        "J4_full_context_beats_late_coordinates": j4,
        "J5_context_rule_transfers_ood": j5,
        "J6_transport_has_a_low_dimensional_context_code": j6,
    }.items()}
    hypotheses["J5_context_rule_transfers_ood"]["selected_rank"] = best_rank
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Can the local residual predict the Layer-6 transport Jacobian?", "",
             "Small finite edits identify one 32-by-32 causal transport matrix per site. Linear models then predict that matrix from PCA coordinates of the unperturbed 768-dimensional local residual. They are fit on FineWeb and frozen before four-times-larger disjoint and OOD edits. The oracle uses each evaluation site's own small-edit matrix and therefore tests whether a sitewise Jacobian is sufficient at all.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Global | Late-32 context | Full r32 | Full r64 | Full r128 | Oracle | r64 edge |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | {row['global']['coordinate_fidelity']:.3f} | {row['late32']['coordinate_fidelity']:.3f} | "
                     f"{row['full_32']['coordinate_fidelity']:.3f} | {row['full_64']['coordinate_fidelity']:.3f} | "
                     f"{row['full_128']['coordinate_fidelity']:.3f} | {row['oracle']['coordinate_fidelity']:.3f} | "
                     f"{row['full_64']['edge_change_fidelity']:.3f} |")
    lines += ["", "If the oracle succeeds but full-state prediction fails, the computation remains locally simple but its routing rule is not a low-dimensional linear function of the local residual. That would point next to nonlocal attention/value-bus context or a nonlinear partition of state space—not to unstructured noise."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
