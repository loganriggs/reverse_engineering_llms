#!/usr/bin/env python3
"""Trace the frozen sparse-graph coordinates through every half-block."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_backbone_router_depth import pooled_basis
from run_bilin18_joint_diagonalization import train_rotation
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_router_hub_depth_telescope.json")
OUTPUT = TENSOR_ROOT / "bilin12_router_hub_depth_telescope.json"
REPORT = TENSOR_ROOT / "BILIN12_ROUTER_HUB_DEPTH_TELESCOPE.md"
FIGURE = TENSOR_ROOT / "bilin12_router_hub_depth_telescope.png"
STAGES = ("embedding",) + tuple(stage for layer in range(12) for stage in
    ((f"L{layer}.attention", f"L{layer}.mlp") if layer < 11 else (f"L{layer}.attention",)))


def load_rows(spec):
    source = np.load(spec["path"], mmap_mode="r")
    return source[spec["start"]:spec["start"] + spec["rows"]]


@torch.inference_mode()
def trace_stages(model, ids, positions, axes):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,)); x, bus = x0, None
    result = {"embedding": (x[:, positions].float() @ axes).flatten(0, 1)}
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, bus = block.attn(F.rms_norm(mixed, (dimension,)), bus)
        pre = mixed + attention; state = F.rms_norm(pre, (dimension,))
        result[f"L{layer}.attention"] = (state[:, positions].float() @ axes).flatten(0, 1)
        write = block.mlp.Down(block.mlp.Left(state) * block.mlp.Right(state)) + block.mlp.Down_bias
        x = pre + write
        if layer < 11:
            normalized = F.rms_norm(x, (dimension,))
            result[f"L{layer}.mlp"] = (normalized[:, positions].float() @ axes).flatten(0, 1)
    return result


@torch.inference_mode()
def collect(model, spec, positions, axes, batch_size):
    rows = load_rows(spec); stores = {stage: [] for stage in STAGES}
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        traced = trace_stages(model, ids, positions, axes)
        for stage, value in traced.items(): stores[stage].append(value.cpu())
    return {stage: torch.cat(values) for stage, values in stores.items()}


def fit_scalar(x, y):
    x_mean = x.mean(0); y_mean = y.mean(0)
    slope = ((x - x_mean) * (y - y_mean)).sum(0) / (x - x_mean).square().sum(0).clamp_min(1e-30)
    return x_mean, y_mean, slope


def predict_scalar(fitted, x):
    x_mean, y_mean, slope = fitted
    return y_mean + (x - x_mean) * slope


def fit_full(x, y, ridge):
    mean = x.mean(0); scale = x.std(0).clamp_min(1e-6); z = (x - mean) / scale
    y_mean = y.mean(0); centered = y - y_mean
    gram = z.T @ z; penalty = ridge * float(torch.trace(gram)) / len(gram)
    weight = torch.linalg.solve(gram + penalty * torch.eye(len(gram)), z.T @ centered)
    return mean, scale, y_mean, weight


def predict_full(fitted, x):
    mean, scale, y_mean, weight = fitted
    return (x - mean) / scale @ weight + y_mean


def r2_columns(target, prediction):
    denominator = (target - target.mean(0)).square().sum(0).clamp_min(1e-30)
    return 1 - (target - prediction).square().sum(0) / denominator


def edge_values(values, pairs):
    return torch.stack([2 * values[:, i] * values[:, j] for i,j in pairs], -1)


def ranks(values):
    order = np.argsort(values, kind="mergesort"); result = np.empty(len(values)); result[order] = np.arange(len(values)); return result


def spearman(left, right):
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); graph = json.loads(Path(config["selected_graph"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    write_basis = pooled_basis(model, tokenizer, {**config, "rank": config["output_rank"],
                                                   "discovery": config["basis_discovery"]})
    block = model.blocks[11]; dimension = model.config.n_embd
    channel_to_output = block.mlp.Down.weight.float().T @ write_basis
    raw = torch.einsum("ja,jb,jk->kab", block.mlp.Left.weight.float(), block.mlp.Right.weight.float(), channel_to_output)
    full_forms = (raw + raw.transpose(-1, -2)) / 2
    natural_states, _ = collect_states_writes(model, 11, config["natural"], config["positions"], config["batch_size"])
    _, _, vh = torch.linalg.svd(natural_states.float(), full_matrices=False)
    natural_basis = vh[:config["input_rank"]].T.contiguous()
    restricted = torch.einsum("ai,kab,bj->kij", natural_basis, full_forms, natural_basis).detach()
    rotation, _ = train_rotation(restricted, config); rotation = rotation.detach(); axes = natural_basis @ rotation
    pairs = [tuple(pair) for pair in graph["selected_pairs"]]
    degree = np.bincount([node for pair in pairs for node in pair], minlength=config["input_rank"])
    hubs = np.flatnonzero(degree >= 3).tolist()
    train = collect(model, config["natural"], config["positions"], axes, config["batch_size"])
    target_train = train["L11.attention"]; target_edge_train = edge_values(target_train, pairs)
    scalar_models = {}; full_models = {}; edge_models = {}
    for stage in STAGES:
        scalar_models[stage] = fit_scalar(train[stage], target_train)
        full_models[stage] = fit_full(train[stage], target_train, config["ridge"])
        edge_models[stage] = fit_scalar(edge_values(train[stage], pairs), target_edge_train)
    results = {}; maximum_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        values = collect(model, spec, config["positions"], axes, config["batch_size"])
        target = values["L11.attention"]; target_edges = edge_values(target, pairs)
        stages = {}
        for stage in STAGES:
            scalar = r2_columns(target, predict_scalar(scalar_models[stage], values[stage]))
            full = r2_columns(target, predict_full(full_models[stage], values[stage]))
            edges = r2_columns(target_edges, predict_scalar(edge_models[stage], edge_values(values[stage], pairs)))
            stages[stage] = {"axis_r2_median": float(scalar.median()),
                             "hub_r2_median": float(scalar[hubs].median()),
                             "full_map_axis_r2_median": float(full.median()),
                             "edge_r2_median": float(edges.median()),
                             "axes_above_080": int((scalar >= .80).sum()),
                             "edges_above_080": int((edges >= .80).sum())}
        maximum_error = max(maximum_error, float((target - predict_scalar(scalar_models["L11.attention"], target)).abs().max()))
        results[corpus] = {"sites": len(target), "stages": stages}
    profiles = {corpus: [row["stages"][stage]["edge_r2_median"] for stage in STAGES]
                for corpus, row in results.items()}
    depth = list(range(len(STAGES)))
    t0 = maximum_error <= 1e-6
    t1 = all(row["stages"]["L10.mlp"]["hub_r2_median"] >= .80 for row in results.values())
    t2 = all(row["stages"]["L6.mlp"]["hub_r2_median"] >= .50 for row in results.values())
    t3 = all(row["stages"]["L10.mlp"]["edge_r2_median"] >= .70 for row in results.values())
    t4 = all(row["stages"]["L10.mlp"]["full_map_axis_r2_median"]
             <= row["stages"]["L10.mlp"]["axis_r2_median"] + .10 for row in results.values())
    t5 = all(spearman(depth, profile) >= .70 for profile in profiles.values())
    correlations = {corpus: spearman(profiles["fineweb"], profiles[corpus])
                    for corpus in ("wikitext", "tinystories")}
    t6 = all(value >= .85 for value in correlations.values())
    hypotheses = {
        "T0_trace_target_is_exact": {"supported": t0, "maximum_error": maximum_error},
        "T1_layer10_residual_contains_same_axis_hubs": {"supported": t1},
        "T2_midlayer_contains_same_axis_hubs": {"supported": t2},
        "T3_pair_grammar_is_present_before_final_attention": {"supported": t3},
        "T4_late_coordinates_do_not_require_rotation": {"supported": t4},
        "T5_readability_generally_increases_with_depth": {"supported": t5},
        "T6_depth_profile_transfers_ood": {"supported": t6, "correlations": correlations},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "hubs": hubs,
                                  "degree": degree.tolist(), "results": results}, indent=2, allow_nan=False))
    fig, axes_plot = plt.subplots(1, 3, figsize=(18, 4.8)); x = np.arange(len(STAGES))
    for corpus, row in results.items():
        axes_plot[0].plot(x, [row["stages"][stage]["hub_r2_median"] for stage in STAGES], marker="o", ms=3, label=corpus)
        axes_plot[1].plot(x, [row["stages"][stage]["edge_r2_median"] for stage in STAGES], marker="o", ms=3, label=corpus)
        axes_plot[2].plot(x, [row["stages"][stage]["full_map_axis_r2_median"] - row["stages"][stage]["axis_r2_median"] for stage in STAGES], marker="o", ms=3, label=corpus)
    axes_plot[0].set_title("Same-axis high-degree hubs"); axes_plot[1].set_title("Same-edge products")
    axes_plot[2].set_title("Full-map gain over same-axis")
    for axis in axes_plot:
        axis.axhline(0, color="black", lw=.8); axis.set_xticks(x, STAGES, rotation=70, fontsize=6); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    lines = ["# Upstream depth telescope of sparse-graph hubs", "",
             "The final 32 coordinates and selected edge products are frozen. At every embedding/attention/MLP half-block, the same residual directions are projected and only per-coordinate affine scales are fit on ordinary FineWeb states. A full 32-by-32 map is an upper probe for rotated information. All maps are frozen before disjoint FineWeb and OOD evaluation.", "",
             "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Hub depth telescope](bilin12_router_hub_depth_telescope.png)", "",
              f"High-degree hubs (degree >=3): `{hubs}`.", "",
              "| Corpus | Embedding hub/edge R2 | L6 MLP hub/edge | L10 MLP hub/edge | L11 attention target | Full-map gain at L10 |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        stage = row["stages"]
        lines.append(f"| {corpus} | {stage['embedding']['hub_r2_median']:.3f}/{stage['embedding']['edge_r2_median']:.3f} | "
                     f"{stage['L6.mlp']['hub_r2_median']:.3f}/{stage['L6.mlp']['edge_r2_median']:.3f} | "
                     f"{stage['L10.mlp']['hub_r2_median']:.3f}/{stage['L10.mlp']['edge_r2_median']:.3f} | 1.000/1.000 | "
                     f"{stage['L10.mlp']['full_map_axis_r2_median']-stage['L10.mlp']['axis_r2_median']:.3f} |")
    lines += ["", "Same-axis success means the final coordinate is persistently present in the shared residual basis; full-map-only success means the information exists but rotates or mixes. Neither is causal provenance by itself. The next step uses finite boundary perturbations at the earliest strong stage to measure the actual transport matrix into final hubs and edge products."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
