#!/usr/bin/env python3
"""Predict finite transport from an analytic chain of bilinear MLP Jacobians."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin12_router_finite_transport import edge_values, fidelity
from run_bilin12_router_transport_operator_decomposition import batch_worlds, central
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_analytic_mlp_jacobian.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_analytic_mlp_jacobian.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_ANALYTIC_MLP_JACOBIAN.md"


def norm(x): return F.rms_norm(x, (x.shape[-1],))


def rms_jvp(x, dx):
    # x [B,D], dx [B,N,D]
    r = x.square().mean(-1, keepdim=True).sqrt()
    cross = (x[:, None] * dx).mean(-1, keepdim=True)
    return dx / r[:, None] - x[:, None] * cross / r[:, None].pow(3)


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


@torch.inference_mode()
def boundary(model, ids, layer):
    x0 = norm(model.embedding(ids)); x, bus = x0, None
    for index, block in enumerate(model.blocks):
        x, bus = block(x, bus, x0)
        if index == layer: return x, x0, bus
    raise RuntimeError


@torch.inference_mode()
def analytic_position(model, ids, position, config, axes, project_each_layer):
    layer = config["layer"]; x, x0, bus = boundary(model, ids, layer)
    bsz = len(ids); current = x; dx = config["scale"] * x[:, position].square().mean(-1).sqrt()[:, None, None] * axes.T[None]
    for index in range(layer + 1, 12):
        block = model.blocks[index]
        mixed = block.lambdas[0] * current + block.lambdas[1] * x0
        dmixed = block.lambdas[0] * dx
        attention, bus = block.attn(norm(mixed), bus)
        pre = mixed + attention; dpre = dmixed  # attention response is frozen
        state = norm(pre); dstate = rms_jvp(pre[:, position], dpre)
        if index == 11:
            return dstate @ axes
        left = block.mlp.Left(state[:, position]); right = block.mlp.Right(state[:, position])
        dleft = dstate @ block.mlp.Left.weight.float().T
        dright = dstate @ block.mlp.Right.weight.float().T
        dwrite = (dleft * right[:, None] + left[:, None] * dright) @ block.mlp.Down.weight.float().T
        dx = dpre + dwrite
        if project_each_layer:
            dx = (dx @ axes) @ axes.T
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        current = pre + write
    raise RuntimeError


def score(target, prediction):
    return {"fidelity": fidelity(target, prediction),
            "cosine": float((target * prediction).sum() /
                            (target.square().sum().sqrt() * prediction.square().sum().sqrt()).clamp_min(1e-30))}


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); graph = json.loads(Path("artifacts/bilin18_tensor_decomposition/bilin12_router_graph_natural_causality.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": 32, "discovery": config["basis_discovery"]})
    block = model.blocks[11]; channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    forms = (raw + raw.transpose(-1, -2)) / 2
    natural, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural.float(), full_matrices=False); basis = vh[:32].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", basis, forms, basis).detach()
    rotation, _ = train_rotation(restricted, config); axes = (basis @ rotation.detach()).float()
    pairs = [tuple(pair) for pair in graph["selected_pairs"]]; results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        stores = {key: [] for key in ("native", "mlp", "analytic", "projected", "base_output")}
        source = load_rows(spec)
        for start in range(0, len(source), config["batch_size"]):
            raw_ids = np.asarray(source[start:start + config["batch_size"], :128], dtype=np.int64)
            ids = torch.as_tensor(raw_ids, device="cuda")
            finite = batch_worlds(model, config["layer"], ids, config["positions"], axes, config["scale"])
            stores["native"].append(central(finite["full"]))
            stores["mlp"].append(central(finite["mlp_only"]))
            # Recompute unperturbed final coordinates for edge-change baselines.
            x, x0, bus = boundary(model, ids, config["layer"])
            from run_bilin12_router_transport_operator_decomposition import base_suffix_cache
            final, _ = base_suffix_cache(model, config["layer"], x, x0, bus)
            for position in config["positions"]:
                stores["analytic"].append(analytic_position(model, ids, position, config, axes, False).cpu())
                stores["projected"].append(analytic_position(model, ids, position, config, axes, True).cpu())
                stores["base_output"].append((final[:, position] @ axes).cpu()[:, None].expand(-1, 32, -1))
        values = {key: torch.cat(rows).reshape(-1, 32) for key, rows in stores.items()}
        mlp_score = score(values["mlp"], values["analytic"])
        native_score = score(values["native"], values["analytic"])
        projected_score = score(values["mlp"], values["projected"])
        base = values["base_output"]
        edge_target = edge_values(base + values["mlp"], pairs) - edge_values(base, pairs)
        edge_prediction = edge_values(base + values["analytic"], pairs) - edge_values(base, pairs)
        results[corpus] = {"analytic_vs_mlp_path": mlp_score,
                           "analytic_vs_native": native_score,
                           "projected_vs_mlp_path": projected_score,
                           "analytic_edge_fidelity": fidelity(edge_target, edge_prediction)}
    a0 = all(row["analytic_vs_mlp_path"]["fidelity"] >= .999 for row in results.values())
    a1 = all(row["analytic_vs_native"]["fidelity"] >= .95 for row in results.values())
    a2 = all(row["analytic_edge_fidelity"] >= .99 for row in results.values())
    a3 = all(row["projected_vs_mlp_path"]["fidelity"] >= .80 for row in results.values())
    a4 = all(row["analytic_vs_mlp_path"]["fidelity"] >= row["projected_vs_mlp_path"]["fidelity"] + .10 for row in results.values())
    fw = results["fineweb"]["analytic_vs_native"]["fidelity"]
    a5 = all(results[c]["analytic_vs_native"]["fidelity"] >= fw - .03 for c in ("wikitext", "tinystories"))
    hypotheses = {name: {"supported": bool(value)} for name, value in {
        "A0_weight_jacobian_matches_finite_mlp_path": a0, "A1_weight_jacobian_explains_native_transport": a1,
        "A2_weight_jacobian_predicts_graph_changes": a2, "A3_fixed_late_subspace_is_sufficient": a3,
        "A4_off_subspace_routes_are_material": a4, "A5_weight_derived_account_transfers_ood": a5}.items()}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))
    lines = ["# Weight-derived analytic Jacobian chain", "",
             "For each natural site, the exact RMSNorm derivative and bilinear-MLP derivative are composed from Layers 7–10, with attention responses frozen as in the causal operator audit. Nothing is fit to perturbation outcomes. The resulting tangent prediction is compared with finite edits, the complete native path, and a version forcibly projected into the same fixed 32-dimensional late subspace after every MLP.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Corpus | Analytic→MLP path | Analytic→native | Repeated-32D→MLP path | Edge fidelity |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | {row['analytic_vs_mlp_path']['fidelity']:.4f} | "
                     f"{row['analytic_vs_native']['fidelity']:.4f} | {row['projected_vs_mlp_path']['fidelity']:.4f} | "
                     f"{row['analytic_edge_fidelity']:.4f} |")
    lines += ["", "A successful full-space Jacobian is a mechanistic local explanation: weights plus the current activation determine the causal transport. It is not yet a global semantic explanation, because it does not compress why natural states select those Jacobians. The repeated-projection control tests whether the 32D endpoint code is also an adequate internal carrier through the whole computation."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
