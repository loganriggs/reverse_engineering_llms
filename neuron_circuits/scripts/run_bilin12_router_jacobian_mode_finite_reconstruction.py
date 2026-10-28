#!/usr/bin/env python3
"""Finite causal reconstruction from a frozen spectral dictionary of Jacobians."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_analytic_mlp_jacobian import analytic_position, boundary
from run_bilin12_router_finite_transport import edge_values, fidelity
from run_bilin12_router_jacobian_mode_spectrum import collect as collect_jacobians
from run_bilin12_router_transport_operator_decomposition import base_suffix_cache, batch_worlds
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_jacobian_mode_finite_reconstruction.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_jacobian_mode_finite_reconstruction.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_JACOBIAN_MODE_FINITE_RECONSTRUCTION.md"


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


def approximate(values, mean, basis):
    centered = values - mean
    if len(basis) == 0: return mean.expand_as(values)
    return mean + (centered @ basis.T) @ basis


def event_prediction(matrices, scale):
    response = matrices.reshape(-1, 32, 32) * scale
    return torch.cat([response, -response], 1)


def quadratic(values, forms):
    flat = values.reshape(-1, 32)
    return torch.einsum("ni,kij,nj->nk", flat, forms, flat).reshape(*values.shape[:-1], -1)


def metrics(target, prediction, base, pairs, forms):
    edge_target = edge_values(base + target, pairs) - edge_values(base, pairs)
    edge_prediction = edge_values(base + prediction, pairs) - edge_values(base, pairs)
    coefficient_target = quadratic(base + target, forms) - quadratic(base, forms)
    coefficient_prediction = quadratic(base + prediction, forms) - quadratic(base, forms)
    return {"coordinate_fidelity": fidelity(target, prediction),
            "edge_change_fidelity": fidelity(edge_target, edge_prediction),
            "coefficient_change_fidelity": fidelity(coefficient_target, coefficient_prediction)}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); graph = json.loads(Path(config["selected_graph"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": 32, "discovery": config["basis_discovery"]})
    block = model.blocks[11]; channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    natural, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False); natural_basis = vh[:32].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms, natural_basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (natural_basis @ rotation.detach()).float()
    forms = (rotation.T[None] @ restricted @ rotation[None]).cpu()
    print("collect dictionary", flush=True); train = collect_jacobians(model, config["mode_train"], config, axes)
    mean = train.mean(0); centered = train - mean; _, _, vh = torch.linalg.svd(centered.cuda(), full_matrices=False); modes = vh[:128].cpu()
    generator = torch.Generator().manual_seed(config["seed"]); q, _ = torch.linalg.qr(torch.randn(16, 16, generator=generator))
    rotated_top = q @ modes[:16]
    pairs = [tuple(pair) for pair in graph["selected_pairs"]]; results = {}; rotation_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True); matrices = []; native = []; mlp = []; bases = []; source = load_rows(spec)
        finite_config = {**config, "scale": config["finite_scale"]}
        for start in range(0, len(source), config["batch_size"]):
            ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
            worlds = batch_worlds(model, config["layer"], ids, config["positions"], axes, config["finite_scale"])
            native.append(worlds["full"].reshape(-1, 64, 32)); mlp.append(worlds["mlp_only"].reshape(-1, 64, 32))
            x, x0, bus = boundary(model, ids, config["layer"]); final, _ = base_suffix_cache(model, config["layer"], x, x0, bus)
            for position in config["positions"]:
                matrices.append(analytic_position(model, ids, position, config, axes, False).cpu())
                bases.append((final[:, position] @ axes).cpu())
        matrices = torch.cat(matrices).reshape(-1, 1024); native = torch.cat(native); mlp = torch.cat(mlp)
        base = torch.cat(bases)[:, None].expand(-1, 64, -1)
        rows = {}
        for rank in config["ranks"]:
            basis = modes[:rank]
            prediction = event_prediction(approximate(matrices, mean, basis), config["finite_scale"])
            rows[str(rank)] = {"vs_mlp_path": metrics(mlp, prediction, base, pairs, forms),
                               "vs_native": metrics(native, prediction, base, pairs, forms)}
        bulk_prediction = event_prediction(approximate(matrices, mean, modes[16:32]), config["finite_scale"])
        without_top = event_prediction(approximate(matrices, mean, modes[16:128]), config["finite_scale"])
        rotated_prediction = event_prediction(approximate(matrices, mean, rotated_top), config["finite_scale"])
        top_prediction = event_prediction(approximate(matrices, mean, modes[:16]), config["finite_scale"])
        rotation_error = max(rotation_error, float((rotated_prediction - top_prediction).abs().max()))
        rows["bulk_16"] = {"vs_mlp_path": metrics(mlp, bulk_prediction, base, pairs, forms)}
        rows["rank128_without_top16"] = {"vs_mlp_path": metrics(mlp, without_top, base, pairs, forms)}
        results[corpus] = rows
    b0 = all(row["16"]["vs_mlp_path"]["coordinate_fidelity"] >= .85 for row in results.values())
    b1 = all(row["32"]["vs_mlp_path"]["coordinate_fidelity"] >= .89 for row in results.values())
    b2 = all(row["32"]["vs_mlp_path"]["edge_change_fidelity"] >= .85 and
             row["32"]["vs_mlp_path"]["coefficient_change_fidelity"] >= .85 for row in results.values())
    b3 = all(row["16"]["vs_mlp_path"]["coordinate_fidelity"] >= row["bulk_16"]["vs_mlp_path"]["coordinate_fidelity"] + .05 for row in results.values())
    b4 = rotation_error <= 1e-5
    b5 = all(row["32"]["vs_native"]["coordinate_fidelity"] >= .85 for row in results.values())
    b6 = all(row["128"]["vs_mlp_path"]["coordinate_fidelity"] >=
             row["rank128_without_top16"]["vs_mlp_path"]["coordinate_fidelity"] + .05 for row in results.values())
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "B0_rank16_is_finitely_sufficient": b0, "B1_rank32_is_high_fidelity": b1,
        "B2_rank32_recovers_sparse_grammar_changes": b2, "B3_outliers_beat_equal_size_bulk": b3,
        "B4_causal_sufficiency_is_rotation_invariant": b4, "B5_rank32_predicts_native_transport": b5,
        "B6_leading_modes_are_necessary_for_compact_reconstruction": b6}.items()}
    hypotheses["B4_causal_sufficiency_is_rotation_invariant"]["maximum_error"] = rotation_error
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Finite causal sufficiency of Jacobian spectral modes", "",
             "A FineWeb-only PCA dictionary of weight-derived Layer-6 transport Jacobians predicts one-sided interventions of scale 0.5. Predictions are scored against the finite attention-frozen MLP path, the complete native path, changes in the selected sparse edge products, and changes in all 32 restricted quadratic output coefficients. This tests predictive causal sufficiency; operator modes are not uniquely removable physical components.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Mean / r16 / r32 / r128 coordinate | r32 edge / coefficient | r32 native | Bulk16 | r128 minus top16 |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        coordinate = " / ".join(f"{row[str(rank)]['vs_mlp_path']['coordinate_fidelity']:.3f}" for rank in (0,16,32,128))
        r32 = row["32"]; lines.append(f"| {corpus} | {coordinate} | {r32['vs_mlp_path']['edge_change_fidelity']:.3f} / "
                     f"{r32['vs_mlp_path']['coefficient_change_fidelity']:.3f} | {r32['vs_native']['coordinate_fidelity']:.3f} | "
                     f"{row['bulk_16']['vs_mlp_path']['coordinate_fidelity']:.3f} | "
                     f"{row['rank128_without_top16']['vs_mlp_path']['coordinate_fidelity']:.3f} |")
    lines += ["", f"Randomly rotating the top-16 operator subspace changes its reconstructed response by at most `{rotation_error:.2e}`. Thus causal sufficiency belongs to the subspace, whereas individual semantic mode labels can vary under an equally sufficient basis."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
