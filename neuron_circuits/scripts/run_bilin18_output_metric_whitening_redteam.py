#!/usr/bin/env python3
"""Red-team output Fisher outliers against unembedding-induced metric anisotropy."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_output_basis_upstream_compactness import (
    batched_final_adjoint, propagate_source_adjoints,
)
from run_bilin18_output_subspace_fisher_geometry import (
    canonical_overlap, eigensystem, rotation_null,
)
from run_bilin18_layer17_tangent_output_semantics import (
    evaluate_individual, select_individual, vocabulary_table,
)
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_output_metric_whitening_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_output_metric_whitening_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_OUTPUT_METRIC_WHITENING_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_output_metric_whitening_redteam.png"


def residual_adjoints(final_state, position, axes):
    local = axes.T
    adjoint = torch.zeros(axes.shape[1], *final_state.shape, device=final_state.device)
    adjoint[:, 0, position] = local
    # The residual-dual score is axis dot final RMS-normalized state.
    root = final_state.float().square().mean(-1, keepdim=True).sqrt()
    coupling = (final_state.float()[None] * adjoint).mean(-1, keepdim=True)
    return adjoint / root[None] - final_state.float()[None] * coupling / root[None].pow(3)


def collect_metric_grams(model, rows, config, axes, contrasts):
    grams = {name: [] for name in (*contrasts.keys(), "residual_dual")}
    for row_index, raw_row in enumerate(rows):
        ids = torch.as_tensor(np.asarray(raw_row[:128], dtype=np.int64), device="cuda")[None]
        with torch.no_grad(): trace = forward_layers_native(model, ids, len(model.blocks) - 1)
        for position in config["positions"]:
            final_state = trace[-1]["output"]
            adjoints = {name: batched_final_adjoint(model, final_state, position, contrast) for name, contrast in contrasts.items()}
            adjoints["residual_dual"] = residual_adjoints(final_state, position, axes)
            for name, final in adjoints.items():
                gradients = propagate_source_adjoints(model, trace, position, final, config["source_layers"])
                matrix = torch.cat([gradients[layer] for layer in config["source_layers"]], dim=1).double()
                gram = matrix @ matrix.T; gram /= torch.trace(gram).clamp_min(1e-30); grams[name].append(gram.cpu())
        print(f"metric redteam rows {row_index + 1}/{len(rows)}", flush=True)
    return grams


def summarize_metric(discovery, evaluation, null_replicates, generator):
    _, _, dvec, dsummary = eigensystem(discovery); _, _, evec, esummary = eigensystem(evaluation)
    null = rotation_null(evaluation, null_replicates, generator)
    return {"discovery": dsummary, "evaluation": esummary, "top1_squared_cosine": float((dvec[:, 0] @ evec[:, 0]).square()), "top4_overlap": canonical_overlap(dvec, evec, 4), "rotation_null": null}, dvec, evec


def main():
    config = json.loads(CONFIG.read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    core = torch.load(config["core_path"], map_location="cpu", weights_only=False); axes = core["output"][:, :config["rank"]].float().cuda()
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    weight = model.lm_head.weight.float(); vocabulary = tokenizer.vocab_size
    raw_valid = weight[:vocabulary] @ axes
    raw = torch.zeros(weight.shape[0], config["rank"], device="cuda"); raw[:vocabulary] = raw_valid
    gram = raw_valid.T @ raw_valid
    values, vectors = torch.linalg.eigh(gram.double()); order = values.argsort(descending=True); values = values[order]; vectors = vectors[:, order]
    inverse = vectors @ torch.diag(values.clamp_min(1e-12).reciprocal()) @ vectors.T
    inverse_sqrt = vectors @ torch.diag(values.clamp_min(1e-12).rsqrt()) @ vectors.T
    # `logit_orthonormal` gives unit Euclidean logit contrasts. `logit_dual`
    # makes their response to residual axes the identity: C^T D = I.
    contrasts = {"raw_logit": raw, "logit_orthonormal": raw @ inverse_sqrt.float(), "logit_dual": raw @ inverse.float()}
    corpus = np.load(config["corpus"], mmap_mode="r")
    discovery_rows = corpus[config["discovery_start"]:config["discovery_start"] + config["rows_per_split"]]
    evaluation_rows = corpus[config["evaluation_start"]:config["evaluation_start"] + config["rows_per_split"]]
    print("collect metric discovery", flush=True); discovery = collect_metric_grams(model, discovery_rows, config, axes, contrasts)
    print("collect metric evaluation", flush=True); evaluation = collect_metric_grams(model, evaluation_rows, config, axes, contrasts)
    generator = torch.Generator().manual_seed(config["seed"]); results, discovery_vectors, evaluation_vectors = {}, {}, {}
    for name in (*contrasts.keys(), "residual_dual"):
        results[name], discovery_vectors[name], evaluation_vectors[name] = summarize_metric(discovery[name], evaluation[name], config["rotation_null_replicates"], generator)
    unembedding_axis = vectors[:, 0].float().cpu(); raw_alignment = float((evaluation_vectors["raw_logit"][:, 0].float() @ unembedding_axis).square())
    cross_metric = {name: float((evaluation_vectors["raw_logit"][:, 0] @ evaluation_vectors[name][:, 0]).square()) for name in ("logit_orthonormal", "logit_dual", "residual_dual")}
    cross_top4 = {name: canonical_overlap(evaluation_vectors["raw_logit"], evaluation_vectors[name], 4) for name in ("logit_orthonormal", "logit_dual", "residual_dual")}
    vocabulary_scores = raw_valid.detach().cpu(); _, vocabulary_labels, vocabulary_names = vocabulary_table(tokenizer)
    even = np.arange(tokenizer.vocab_size) % 2 == 0; odd = ~even
    semantic, selections = {}, {}
    semantic_bases = {name: discovery_vectors[name].float() for name in (*contrasts.keys(), "residual_dual")}
    random_qs = [torch.linalg.qr(torch.randn(config["rank"], config["rank"], generator=generator)).Q for _ in range(32)]
    semantic_bases.update({f"random_{index}": q for index, q in enumerate(random_qs)})
    for name, q in semantic_bases.items():
        selected = select_individual((vocabulary_scores[even] @ q).numpy(), vocabulary_labels[even], vocabulary_names, 20)
        semantic[name] = evaluate_individual((vocabulary_scores[odd] @ q).numpy(), vocabulary_labels[odd], selected)["median_auc"]
        if name in (*contrasts.keys(), "residual_dual"): selections[name] = selected
    semantic_random_p95 = float(np.quantile([semantic[f"random_{index}"] for index in range(32)], .95))
    token_examples = {}
    texts = [tokenizer.decode([index]) for index in range(tokenizer.vocab_size)]
    for name in (*contrasts.keys(), "residual_dual"):
        rotated = vocabulary_scores @ discovery_vectors[name].float()
        token_examples[name] = []
        for axis_index in range(4):
            values_axis = rotated[:, axis_index]; high = values_axis.topk(4).indices.tolist(); low = (-values_axis).topk(4).indices.tolist()
            token_examples[name].append({"promoted": [texts[index] for index in high], "suppressed": [texts[index] for index in low], "label": selections[name][axis_index]["label"]})
    raw_top1 = results["raw_logit"]["evaluation"]["top1_share"]
    hypotheses = {
        "W0_raw_fisher_is_unembedding_gain": {"supported": raw_alignment >= .90},
        "W1_residual_metric_removes_raw_outlier": {"supported": results["residual_dual"]["evaluation"]["top1_share"] <= raw_top1 - .20},
        "W2_logit_dual_removes_raw_outlier": {"supported": results["logit_dual"]["evaluation"]["top1_share"] <= raw_top1 - .20},
        "W3_residual_metric_has_shared_structure": {"supported": results["residual_dual"]["evaluation"]["top4_share"] >= results["residual_dual"]["rotation_null"]["top4_share_p95"] + .05},
        "W4_metric_specific_subspaces_transfer": {"supported": all(results[name]["top4_overlap"] >= .80 for name in results)},
        "W5_metric_choice_changes_the_leading_axis": {"supported": max(cross_metric["logit_dual"], cross_metric["residual_dual"]) < .80},
    }
    output = {"config": config, "hypotheses": hypotheses, "unembedding_gram": {"top1_share": float(values[0] / values.sum()), "top4_share": float(values[:4].sum() / values.sum()), "top_eigenvalues": values[:8].tolist()}, "raw_fisher_to_unembedding_top1_squared_cosine": raw_alignment, "raw_top1_cross_metric_squared_cosine": cross_metric, "raw_top4_cross_metric_overlap": cross_top4, "semantic_auc": semantic, "semantic_random_p95": semantic_random_p95, "top4_token_examples": token_examples, "results": results}
    OUTPUT.write_text(json.dumps(output, indent=2, allow_nan=False)); torch.save({"discovery": discovery_vectors, "evaluation": evaluation_vectors}, TENSOR_ROOT / "bilin18_output_metric_fisher_axes.pt")
    lines = ["# Metric red team of the output-subspace Fisher outlier", "", "The same rank-16 residual output subspace is transported under four dual metrics: raw logit contrasts `W u`, Euclidean-orthonormal logit contrasts, logit duals satisfying `C^T D = I`, and direct residual-space dual axes after final RMSNorm. This distinguishes dynamic network preference from anisotropy injected by the unembedding metric.", "", f"Raw-Fisher leading-axis squared cosine with the weight-only unembedding-Gram leading axis: **{raw_alignment:.3f}**.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Metric red team](bilin18_output_metric_whitening_redteam.png)", "", "| Metric | Eval top1/top4 | Effective rank | Rotation-null top1/top4 p95 | Top1 transfer | Top4 transfer | Raw-leading-axis overlap |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in (*contrasts.keys(), "residual_dual"):
        row = results[name]; ev = row["evaluation"]; null = row["rotation_null"]; overlap = 1.0 if name == "raw_logit" else cross_metric[name]
        lines.append(f"| {name} | {ev['top1_share']:.3f}/{ev['top4_share']:.3f} | {ev['effective_rank']:.2f} | {null['top1_share_p95']:.3f}/{null['top4_share_p95']:.3f} | {row['top1_squared_cosine']:.3f} | {row['top4_overlap']:.3f} | {overlap:.3f} |")
    lines += ["", "## Cross-metric and semantic audit", "", f"Raw-logit versus residual-dual top-four subspace overlap: **{cross_top4['residual_dual']:.3f}**. Residual-dual axis semantic AUROC: **{semantic['residual_dual']:.3f}**, versus matched-random p95 **{semantic_random_p95:.3f}**.", "", "| Residual axis | Selected vocabulary label | Promoted examples | Suppressed examples |", "|---:|---|---|---|"]
    for axis_index, example in enumerate(token_examples["residual_dual"]):
        lines.append(f"| {axis_index} | `{example['label']}` | `{example['promoted']}` | `{example['suppressed']}` |")
    lines += ["", "## Guardrails", "", "A Fisher axis is canonical only after choosing a dual metric. Raw logits weight directions by unembedding gain; residual duals weight internal residual coordinates equally; logit duals weight equal coordinates within the reachable logit subspace. None is universally correct. Stable low-dimensional subspace geometry is stronger evidence than naming individual axes, especially when matched rotations explain semantic AUROC equally well."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, plots = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for name in (*contrasts.keys(), "residual_dual"):
        eigen = np.asarray(results[name]["evaluation"]["eigenvalues"]); plots[0].plot(np.arange(1, len(eigen) + 1), eigen / eigen.sum(), marker="o", label=name)
    plots[1].bar(np.arange(3), [cross_metric[name] for name in ("logit_orthonormal", "logit_dual", "residual_dual")]); plots[1].set_xticks(np.arange(3), ["orthonormal", "logit dual", "residual dual"], rotation=20)
    plots[0].set(xlabel="Eigenvalue rank", ylabel="Normalized eigenvalue", title="Metric-dependent Fisher spectrum"); plots[1].set(ylabel="Squared cosine", title="Overlap with raw-logit leading axis")
    for plot in plots: plot.grid(alpha=.2)
    plots[0].legend(); fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "raw_alignment": raw_alignment, "cross_metric": cross_metric, "cross_top4": cross_top4, "semantic_auc": {name: semantic[name] for name in (*contrasts.keys(), "residual_dual")}, "semantic_random_p95": semantic_random_p95, "results": results}, indent=2)); print(REPORT)


if __name__ == "__main__": main()
