#!/usr/bin/env python3
"""Rotation-invariant Fisher geometry of the transported output subspace."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_router_jacobian_sparse_basis_redteam import greedy_overlap
from run_bilin18_layer17_tangent_output_semantics import evaluate_individual, select_individual, vocabulary_table
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT
from run_bilin18_upstream_objective_basis_discovery import collect_programs


CONFIG = Path("configs/bilin18_output_subspace_fisher_geometry.json")
OUTPUT = TENSOR_ROOT / "bilin18_output_subspace_fisher_geometry.json"
REPORT = TENSOR_ROOT / "BILIN18_OUTPUT_SUBSPACE_FISHER_GEOMETRY.md"
FIGURE = TENSOR_ROOT / "bilin18_output_subspace_fisher_geometry.png"


def cell_grams(sites, layers):
    result = {family: {layer: [] for layer in layers} for family in ("gradient", "contribution")}
    for site in sites:
        gradient = site["gradient"].double(); contribution = gradient * site["product"].double()[None]
        for family, value in (("gradient", gradient), ("contribution", contribution)):
            gram = value @ value.T; gram /= torch.trace(gram).clamp_min(1e-30)
            result[family][site["layer"]].append(gram)
    return result


def eigensystem(matrices):
    matrix = torch.stack(matrices).mean(0); values, vectors = torch.linalg.eigh(matrix)
    order = values.argsort(descending=True); values = values[order].clamp_min(0); vectors = vectors[:, order]
    probability = values / values.sum().clamp_min(1e-30)
    effective_rank = float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum()))
    participation = float(values.sum().square() / values.square().sum().clamp_min(1e-30))
    return matrix, values, vectors, {"top1_share": float(probability[0]), "top4_share": float(probability[:4].sum()), "effective_rank": effective_rank, "participation_rank": participation, "lambda1_to_mean": float(values[0] / values.mean().clamp_min(1e-30)), "eigenvalues": values.tolist()}


def pooled(grams, family, layers, row_selector=None):
    values = []
    for layer in layers:
        layer_values = grams[family][layer]
        if row_selector is not None: layer_values = [value for index, value in enumerate(layer_values) if row_selector(index)]
        values.extend(layer_values)
    return values


def canonical_overlap(first, second, rank):
    singular = torch.linalg.svdvals(first[:, :rank].T @ second[:, :rank])
    return float(singular.square().mean())


def rotation_null(matrices, replicates, generator):
    summaries = []
    dimension = matrices[0].shape[0]
    for _ in range(replicates):
        rotated = []
        for matrix in matrices:
            q = torch.linalg.qr(torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)).Q
            rotated.append(q @ matrix @ q.T)
        summaries.append(eigensystem(rotated)[3])
    return {key + "_p95": float(np.quantile([row[key] for row in summaries], .95)) for key in ("top1_share", "top4_share", "lambda1_to_mean")}


def semantic_auc(scores, labels, names, bases, minimum):
    train = np.arange(len(scores)) % 2 == 0; test = ~train; result = {}
    for name, q in bases.items():
        selected = select_individual((scores[train] @ q.float()).numpy(), labels[train], names, minimum)
        result[name] = evaluate_individual((scores[test] @ q.float()).numpy(), labels[test], selected)["median_auc"]
    return result


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False); axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    vocabulary_scores = (model.lm_head.weight.float()[:tokenizer.vocab_size] @ axes).detach().cpu()
    contrasts = torch.zeros(model.lm_head.weight.shape[0], config["rank"], device="cuda"); contrasts[:tokenizer.vocab_size] = vocabulary_scores.cuda()
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery_rows = corpus[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"]]
    evaluation_rows = corpus[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"]]
    print("collect Fisher discovery", flush=True); discovery_sites, _, _ = collect_programs(model, discovery_rows, config["positions"], config["source_layers"], contrasts)
    print("collect Fisher evaluation", flush=True); evaluation_sites, _, _ = collect_programs(model, evaluation_rows, config["positions"], config["source_layers"], contrasts)
    discovery = cell_grams(discovery_sites, config["source_layers"]); evaluation = cell_grams(evaluation_sites, config["source_layers"])
    results, vectors = {}, {}
    generator = torch.Generator().manual_seed(config["seed"])
    rows_per_layer = config["rows_per_split"] * len(config["positions"])
    for family in ("gradient", "contribution"):
        results[family], vectors[family] = {}, {}
        for name, grams in (("discovery", discovery), ("evaluation", evaluation)):
            matrices = pooled(grams, family, config["source_layers"]); _, _, vec, summary = eigensystem(matrices)
            results[family][name] = summary; vectors[family][name] = vec
        half_vectors = []
        for half in (0, 1):
            matrices = pooled(discovery, family, config["source_layers"], lambda index, half=half: (index // (rows_per_layer // 2)) == half)
            half_vectors.append(eigensystem(matrices)[2])
        results[family]["split_half_greedy_overlap"] = greedy_overlap(half_vectors[0].float(), half_vectors[1].float())
        results[family]["discovery_evaluation_greedy_overlap"] = greedy_overlap(vectors[family]["discovery"].float(), vectors[family]["evaluation"].float())
        results[family]["top4_discovery_evaluation_overlap"] = canonical_overlap(vectors[family]["discovery"], vectors[family]["evaluation"], 4)
        results[family]["evaluation_rotation_null"] = rotation_null(pooled(evaluation, family, config["source_layers"]), config["rotation_null_replicates"], generator)
        results[family]["layers"] = {}
        for layer in config["source_layers"]:
            _, _, dvec, dsummary = eigensystem(discovery[family][layer]); _, _, evec, esummary = eigensystem(evaluation[family][layer])
            results[family]["layers"][str(layer)] = {"discovery": dsummary, "evaluation": esummary, "top4_overlap": canonical_overlap(dvec, evec, 4), "greedy_overlap": greedy_overlap(dvec.float(), evec.float())}
    gradient_q = vectors["gradient"]["discovery"].float(); contribution_q = vectors["contribution"]["discovery"].float()
    random_qs = [torch.linalg.qr(torch.randn(config["rank"], config["rank"], generator=generator)).Q for _ in range(32)]
    bases = {"gradient_fisher": gradient_q, "contribution_fisher": contribution_q, **{f"random_{i}": q for i, q in enumerate(random_qs)}}
    _, labels, names = vocabulary_table(tokenizer); semantics = semantic_auc(vocabulary_scores, labels, names, bases, config["minimum_positive"]); semantic_random_p95 = float(np.quantile([semantics[f"random_{i}"] for i in range(32)], .95))
    gradient_eval = results["gradient"]["evaluation"]; gradient_null = results["gradient"]["evaluation_rotation_null"]
    contribution_eval = results["contribution"]["evaluation"]; contribution_null = results["contribution"]["evaluation_rotation_null"]
    hypotheses = {
        "F0_gradient_metric_has_shared_outlier": {"supported": gradient_eval["top1_share"] >= gradient_null["top1_share_p95"] + .02},
        "F1_gradient_metric_has_low_rank_core": {"supported": gradient_eval["top4_share"] >= gradient_null["top4_share_p95"] + .05},
        "F2_gradient_axes_are_reproducible": {"supported": results["gradient"]["split_half_greedy_overlap"] >= .80 and results["gradient"]["discovery_evaluation_greedy_overlap"] >= .80},
        "F3_contribution_metric_has_shared_core": {"supported": contribution_eval["top4_share"] >= contribution_null["top4_share_p95"] + .05},
        "F4_leading_subspaces_transfer": {"supported": min(results[family]["top4_discovery_evaluation_overlap"] for family in ("gradient", "contribution")) >= .80},
        "F5_fisher_axes_are_semantically_privileged": {"supported": max(semantics["gradient_fisher"], semantics["contribution_fisher"]) >= semantic_random_p95 + .03},
    }
    output = {"config": config, "hypotheses": hypotheses, "results": results, "semantic_auc": semantics, "semantic_random_p95": semantic_random_p95}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False)); torch.save({"gradient": gradient_q, "contribution": contribution_q}, TENSOR_ROOT / "bilin18_output_subspace_fisher_axes.pt")
    lines = ["# Rotation-invariant Fisher geometry of the output subspace", "", "Every FineWeb site supplies an exact 16-by-channel tensor map from the stable late output subspace to a Layer-0--3 MLP product. Its normalized Gram matrix is invariant to the reported output basis. Averaging these metrics asks whether the data distribution selects shared important objective directions. A per-context Haar rotation null preserves every local spectrum while destroying shared orientation.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Fisher geometry](bilin18_output_subspace_fisher_geometry.png)", "", "| Family | Split/eval axis overlap | Top-4 subspace overlap | Eval top1/top4 | Rotation-null p95 top1/top4 | Effective rank | Semantic AUROC/random p95 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family, key in (("gradient", "gradient_fisher"), ("contribution", "contribution_fisher")):
        row = results[family]; ev = row["evaluation"]; null = row["evaluation_rotation_null"]
        lines.append(f"| {family} | {row['split_half_greedy_overlap']:.3f}/{row['discovery_evaluation_greedy_overlap']:.3f} | {row['top4_discovery_evaluation_overlap']:.3f} | {ev['top1_share']:.3f}/{ev['top4_share']:.3f} | {null['top1_share_p95']:.3f}/{null['top4_share_p95']:.3f} | {ev['effective_rank']:.2f} | {semantics[key]:.3f}/{semantic_random_p95:.3f} |")
    lines += ["", "## Per-layer geometry", "", "| Family | Layer | Eval top1/top4 | Effective rank | Discovery/evaluation top4 overlap |", "|---|---:|---:|---:|---:|"]
    for family in ("gradient", "contribution"):
        for layer in config["source_layers"]:
            row = results[family]["layers"][str(layer)]; ev = row["evaluation"]
            lines.append(f"| {family} | {layer} | {ev['top1_share']:.3f}/{ev['top4_share']:.3f} | {ev['effective_rank']:.2f} | {row['top4_overlap']:.3f} |")
    lines += ["", "## Guardrails", "", "Fisher eigenvectors are canonical only relative to the declared FineWeb distribution, source layers, positions, and output subspace. Nearly equal eigenvalues make individual axes unstable even when a leading subspace transfers. The rotation null asks whether alignment is shared across contexts; it does not supply semantic meaning. Semantic labels are evaluated only after the metric is frozen."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, plots = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for family in ("gradient", "contribution"):
        values = results[family]["evaluation"]["eigenvalues"]; plots[0].plot(np.arange(1, len(values) + 1), np.asarray(values) / np.sum(values), marker="o", label=family)
        plots[1].plot(config["source_layers"], [results[family]["layers"][str(layer)]["evaluation"]["top4_share"] for layer in config["source_layers"]], marker="o", label=family)
    plots[0].axhline(1 / config["rank"], color="gray", linestyle=":"); plots[0].set(xlabel="Eigenvalue rank", ylabel="Normalized eigenvalue", title="Pooled output-subspace metric")
    plots[1].set(xlabel="Source layer", ylabel="Top-four share", title="Layerwise anisotropy")
    for plot in plots: plot.grid(alpha=.2); plot.legend()
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "results": results, "semantic_auc": {key: semantics[key] for key in ("gradient_fisher", "contribution_fisher")}, "semantic_random_p95": semantic_random_p95}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
