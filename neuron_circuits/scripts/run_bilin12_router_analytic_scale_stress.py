#!/usr/bin/env python3
"""Stress-test the weight-derived local Jacobian far beyond its discovery scale."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_analytic_mlp_jacobian import analytic_position, boundary, score
from run_bilin12_router_transport_operator_decomposition import batch_worlds, central
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_analytic_scale_stress.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_analytic_scale_stress.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_ANALYTIC_SCALE_STRESS.md"


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": 32, "discovery": config["basis_discovery"]})
    block = model.blocks[11]; channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    natural, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False); basis = vh[:32].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", basis, forms, basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (basis @ rotation.detach()).float()
    # Put edit sizes in units of ordinary Layer-6 coordinate variation.
    coordinate_rows = []
    source = load_rows(config["natural"])
    with torch.inference_mode():
        for start in range(0, len(source), config["batch_size"]):
            ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
            x, _, _ = boundary(model, ids, config["layer"])
            coordinate_rows.append((torch.nn.functional.rms_norm(x, (x.shape[-1],))[:, config["positions"]] @ axes).flatten(0, 1).cpu())
    coordinate_std = torch.cat(coordinate_rows).std(0)
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True); results[corpus] = {}
        source = load_rows(spec)
        for scale in config["scales"]:
            local = {**config, "scale": scale}; finite_mlp = []; finite_native = []; analytic = []
            for start in range(0, len(source), config["batch_size"]):
                ids = torch.as_tensor(np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64), device="cuda")
                worlds = batch_worlds(model, config["layer"], ids, config["positions"], axes, scale)
                finite_mlp.append(worlds["mlp_only"]); finite_native.append(worlds["full"])
                for position in config["positions"]:
                    analytic.append(analytic_position(model, ids, position, local, axes, False).cpu())
            tangent = torch.cat(analytic).reshape(-1, 32, 32)
            mlp_world = torch.cat(finite_mlp).reshape(-1, 64, 32)
            native_world = torch.cat(finite_native).reshape(-1, 64, 32)
            prediction = torch.cat([tangent, -tangent], 1).reshape(-1, 32)
            mlp_target = mlp_world.reshape(-1, 32)
            native_target = native_world.reshape(-1, 32)
            central_prediction = tangent.reshape(-1, 32)
            central_mlp = ((mlp_world[:, :32] - mlp_world[:, 32:]) / 2).reshape(-1, 32)
            results[corpus][str(scale)] = {
                "analytic_vs_mlp_path": score(mlp_target, prediction),
                "analytic_vs_native": score(native_target, prediction),
                "central_difference_diagnostic": score(central_mlp, central_prediction),
            }
    def all_scale(scale, threshold):
        return all(results[c][str(scale)]["analytic_vs_mlp_path"]["fidelity"] >= threshold for c in results)
    s0, s1, s2, s3 = all_scale(.08, .999), all_scale(.25, .95), all_scale(.5, .80), all_scale(1.0, .50)
    s4 = all(all(row[str(config["scales"][i + 1])]["analytic_vs_mlp_path"]["fidelity"] <=
                 row[str(config["scales"][i])]["analytic_vs_mlp_path"]["fidelity"] + 1e-4
                 for i in range(len(config["scales"]) - 1)) for row in results.values())
    s5 = all(abs(results[c][str(scale)]["analytic_vs_mlp_path"]["fidelity"] -
                 results["fineweb"][str(scale)]["analytic_vs_mlp_path"]["fidelity"]) <= .05
             for c in ("wikitext", "tinystories") for scale in config["scales"])
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "S0_small_finite_extension_holds": s0, "S1_quarter_unit_extension_holds": s1,
        "S2_half_unit_extension_holds": s2, "S3_unit_scale_retains_useful_prediction": s3,
        "S4_breakdown_is_monotonic": s4, "S5_scale_profile_transfers_ood": s5}.items()}
    metadata = {"median_coordinate_std": float(coordinate_std.median()),
                "minimum_coordinate_std": float(coordinate_std.min()),
                "maximum_coordinate_std": float(coordinate_std.max()),
                "scale_over_median_std": {str(scale): float(scale / coordinate_std.median()) for scale in config["scales"]}}
    OUTPUT.write_text(json.dumps({"config": config, "metadata": metadata, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Large-edit stress test of the analytic Jacobian", "",
             "The weight-derived Layer-6-to-final MLP Jacobian was tested without refitting from 0.08 to 1.0 coordinate units. The median natural standard deviation of these coordinates is " + f"{metadata['median_coordinate_std']:.3f}; thus the largest intervention is {metadata['scale_over_median_std']['1.0']:.2f} median standard deviations. The primary metric scores positive and negative edits separately. An earlier draft used their central difference, which cancels even-order nonlinear terms; that implementation was caught during red-teaming and is retained only as a diagnostic.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | One-sided 0.08 | 0.25 | 0.50 | 1.00 | Native at 1.00 | Central diagnostic at 1.00 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | " + " | ".join(f"{row[str(scale)]['analytic_vs_mlp_path']['fidelity']:.3f}" for scale in config["scales"]) +
                     f" | {row['1.0']['analytic_vs_native']['fidelity']:.3f} | "
                     f"{row['1.0']['central_difference_diagnostic']['fidelity']:.3f} |")
    lines += ["", "Failure at large scale would not refute the local mechanistic account; it would bound its chart size. Success far beyond a typical coordinate standard deviation would support treating one Jacobian chart as a behaviorally useful regime rather than an infinitesimal description."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
