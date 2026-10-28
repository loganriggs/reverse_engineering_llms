#!/usr/bin/env python3
"""Canonicalize Layer-16 loss effects as forward tangents paired with adjoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_attention_layer_path_program import native_cache
from run_bilin18_layer16_ce_aware_mode_selection import ce_gradient, score
from run_bilin18_layer16_upstream_tangent_channels import (
    autograd_check, build_input_axes, propagate_to_layer16,
)
from run_bilin18_mlp_path_symmetry_replication import load_rows
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_forward_adjoint_atlas.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_forward_adjoint_atlas.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_FORWARD_ADJOINT_ATLAS.md"
BASIS_OUTPUT = TENSOR_ROOT / "bilin18_layer16_forward_adjoint_bases.pt"


@torch.inference_mode(False)
def collect_factors(model, spec, axes, config):
    source = load_rows(spec); stores = {name: [] for name in ("forward", "state", "write_adjoint", "effective_adjoint", "target")}
    maximum_forward_error = maximum_contraction_error = 0.0
    block = model.blocks[16]; left_weight = block.mlp.Left.weight.float()
    right_weight = block.mlp.Right.weight.float(); down = block.mlp.Down.weight.float()
    for start in range(0, len(source), config["batch_size"]):
        array = np.asarray(source[start:start + config["batch_size"]])
        ids = torch.as_tensor(array[:, :128].astype(np.int64), device="cuda")
        x, cache = native_cache(model, ids, config); by_layer = {row["index"]: row for row in cache}
        for position in config["positions"]:
            forward, error = propagate_to_layer16(model, x, cache, position, axes, config)
            maximum_forward_error = max(maximum_forward_error, error)
            labels = torch.as_tensor(array[:, position + 1].astype(np.int64), device="cuda")
            write_adjoint, _ = ce_gradient(model, by_layer[17], position, labels)
            state = by_layer[16]["state"][:, position].float()
            left = state @ left_weight.T; right = state @ right_weight.T
            channel_adjoint = write_adjoint @ down
            effective = ((channel_adjoint * right) @ left_weight
                         + (channel_adjoint * left) @ right_weight)
            contraction = torch.einsum("bnd,bd->bn", forward.cuda(), effective)
            dleft = torch.einsum("bnd,cd->bnc", forward.cuda(), left_weight)
            dright = torch.einsum("bnd,cd->bnc", forward.cuda(), right_weight)
            direct_write = torch.einsum(
                "bnc,dc->bnd", dleft * right[:, None] + dright * left[:, None], down)
            reference = (direct_write * write_adjoint[:, None]).sum(-1)
            relative = float((contraction - reference).double().norm()
                             / reference.double().norm().clamp_min(1e-30))
            maximum_contraction_error = max(maximum_contraction_error, relative)
            stores["forward"].append(forward.cpu()); stores["state"].append(state.cpu())
            stores["write_adjoint"].append(write_adjoint.cpu())
            stores["effective_adjoint"].append(effective.cpu()); stores["target"].append(reference.cpu())
        print(f"  rows {start + len(ids)}/{len(source)}", flush=True)
    return {name: torch.cat(rows) for name, rows in stores.items()}, maximum_forward_error, maximum_contraction_error


@torch.inference_mode()
def covariance_basis(rows):
    values = rows.float().cuda(); covariance = values.T @ values
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0).double().cpu()
    basis = eigenvectors[:, order].cpu()
    shares = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    rank90 = int(torch.searchsorted(shares.cumsum(0), torch.tensor(.90, dtype=shares.dtype)).item() + 1)
    effective_rank = float(torch.exp(-(shares[shares > 0] * shares[shares > 0].log()).sum()))
    return basis, {"rank90": rank90, "effective_rank": effective_rank,
                   "top_energy_share": float(shares[0]),
                   "leading_energy_shares": shares[:64].tolist()}


@torch.inference_mode()
def behavior_fidelity(data, basis, rank):
    vectors = basis[:, :rank].float().cuda()
    effective = data["effective_adjoint"].float().cuda()
    projected = (effective @ vectors) @ vectors.T
    prediction = torch.einsum("bnd,bd->bn", data["forward"].float().cuda(), projected).cpu()
    return score(data["target"], prediction)


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    axes = build_input_axes(model, config)
    source = load_rows(config["atlas_discovery"])
    ids = torch.as_tensor(np.asarray(source[:1, :128], dtype=np.int64), device="cuda")
    upstream_error, autograd_error = autograd_check(model, ids, axes, config)
    print("atlas discovery", flush=True)
    discovery, forward_error, contraction_error = collect_factors(
        model, config["atlas_discovery"], axes, config)
    bases = {}; spectra = {}
    basis_rows = {
        "forward": discovery["forward"].flatten(0, 1),
        "state": discovery["state"],
        "write_adjoint": discovery["write_adjoint"],
        "effective_adjoint": discovery["effective_adjoint"],
    }
    for name, values in basis_rows.items():
        print("basis", name, flush=True); bases[name], spectra[name] = covariance_basis(values)
    torch.save({name: value[:, :max(config["subspace_ranks"])] for name, value in bases.items()}, BASIS_OUTPUT)
    generator = torch.Generator().manual_seed(config["seed"])
    random_bases = []
    for _ in range(config["random_replicates"]):
        matrix = torch.randn(axes.shape[0], 64, generator=generator)
        random_bases.append(torch.linalg.qr(matrix, mode="reduced").Q)
    results = {}; maximum_forward_error = max(upstream_error, forward_error)
    maximum_contraction_error = contraction_error
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data, error, contraction = collect_factors(model, spec, axes, config)
        maximum_forward_error = max(maximum_forward_error, error)
        maximum_contraction_error = max(maximum_contraction_error, contraction)
        methods = {}
        for name, basis in bases.items():
            methods[name] = {str(rank): behavior_fidelity(data, basis, rank)
                             for rank in config["subspace_ranks"]}
        random_values = [behavior_fidelity(data, basis, 64) for basis in random_bases]
        results[corpus] = {"sites": len(data["target"]), "methods": methods,
                           "random_rank64": {"median": float(np.median(random_values)),
                                             "p95": float(np.quantile(random_values, .95)),
                                             "maximum": max(random_values)}}
    exact_error = max(maximum_forward_error, maximum_contraction_error)
    hypotheses = {
        "A0_forward_adjoint_contraction_is_exact": exact_error <= 1e-5,
        "A1_rank64_effective_adjoint_subspace_is_sufficient": all(row["methods"]["effective_adjoint"]["64"] >= .90 for row in results.values()),
        "A2_rank64_forward_subspace_is_sufficient": all(row["methods"]["forward"]["64"] >= .90 for row in results.values()),
        "A3_rank128_forward_or_adjoint_is_high_fidelity": all(max(row["methods"]["forward"]["128"], row["methods"]["effective_adjoint"]["128"]) >= .95 for row in results.values()),
        "A4_bilinear_interaction_broadens_the_adjoint": spectra["effective_adjoint"]["rank90"] >= spectra["write_adjoint"]["rank90"] + 32,
        "A5_natural_subspaces_beat_random": all(min(row["methods"]["forward"]["64"], row["methods"]["effective_adjoint"]["64"]) >= row["random_rank64"]["p95"] + .10 for row in results.values()),
        "A6_factor_subspaces_transfer_ood": all(
            results[name]["methods"][method]["64"] >= results["fineweb"]["methods"][method]["64"] - .10
            for name in ("wikitext", "tinystories") for method in ("forward", "effective_adjoint")),
        "A7_state_or_write_adjoint_alone_is_inferior": all(
            max(row["methods"]["state"]["64"], row["methods"]["write_adjoint"]["64"])
            <= max(row["methods"]["forward"]["64"], row["methods"]["effective_adjoint"]["64"]) - .05
            for row in results.values()),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["A0_forward_adjoint_contraction_is_exact"].update({
        "autograd_relative_error": autograd_error,
        "maximum_forward_error": maximum_forward_error,
        "maximum_contraction_error": maximum_contraction_error})
    payload = {"config": config, "hypotheses": hypothesis_rows,
               "discovery_spectra": spectra, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Exact forward-tangent and backward-adjoint atlas for the Layer-16 loss path", "",
             "Each 32D behavioral vector is exactly factorized as `y_s = J_s h_s`, where `J_s` transports the fixed Layer-12 basis into the Layer-16 state and `h_s = B(x_s)^T g_s` is the effective state adjoint induced by the bilinear MLP and downstream loss adjoint. Frozen PCA spans are learned separately from forward tangent rows, states, raw write adjoints, and effective adjoints.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Autograd error: `{autograd_error:.2e}`; maximum contraction error: `{maximum_contraction_error:.2e}`.", "",
              "| Discovery family | Energy R90 | Effective rank | Top share |", "|---|---:|---:|---:|"]
    for name, row in spectra.items(): lines.append(f"| {name} | {row['rank90']} | {row['effective_rank']:.1f} | {row['top_energy_share']:.3f} |")
    lines += ["", "| Corpus | Basis | Behavior R32 / R64 / R128 / R256 |", "|---|---|---:|"]
    for corpus, row in results.items():
        for name in ("forward", "effective_adjoint", "state", "write_adjoint"):
            values = row["methods"][name]
            lines.append(f"| {corpus} | {name} | {values['32']:.3f} / {values['64']:.3f} / {values['128']:.3f} / {values['256']:.3f} |")
        lines.append(f"| {corpus} | random rank64 p95 | {row['random_rank64']['p95']:.3f} |")
    lines += ["", "A basis derived from one factor is scored only after inserting it into the exact contraction. This is more behaviorally relevant than its raw explained variance. The factor bases are shared coordinates; the site-specific `J`, `x`, `g`, and `h` remain part of the executable explanation."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
