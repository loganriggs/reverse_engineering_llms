#!/usr/bin/env python3
"""Split the semantic suffix adjoint by MLP and attention response counts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, cosine, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_bidegree_pilot.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_bidegree_pilot.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_BIDEGREE_PILOT.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_bidegree_pilot.png"


def add(store, key, value): store[key] = store[key] + value if key in store else value


def bidegree_program(model, trace, position, contrast, maximum_total):
    current = {(0, 0): final_adjoint(model, trace[-1]["output"], position, contrast)}
    for index in range(len(model.blocks) - 1, 1, -1):
        block = model.blocks[index]; coordinates = sorted(current); rows = torch.stack([current[key] for key in coordinates]); mlp_rows = mlp_vjp(block, trace[index]["pre"].float(), rows)
        attention_inputs, attention_map = [], []
        for row_index, (m_degree, a_degree) in enumerate(coordinates):
            if m_degree + a_degree + 1 <= maximum_total:
                attention_inputs.append(rows[row_index]); attention_map.append(("a", m_degree, a_degree))
            if m_degree + a_degree + 2 <= maximum_total:
                attention_inputs.append(mlp_rows[row_index]); attention_map.append(("am", m_degree, a_degree))
        previous = trace[index - 1]["output"].float(); x0 = trace[index]["x0"].float(); mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0; bus = trace[index - 1]["value_bus"]
        attention_rows = attention_vjp(block, mixed, bus, torch.stack(attention_inputs)) if attention_inputs else []
        updated = {}
        for row_index, key in enumerate(coordinates):
            m_degree, a_degree = key; add(updated, key, block.lambdas[0].float() * rows[row_index])
            if m_degree + a_degree + 1 <= maximum_total: add(updated, (m_degree + 1, a_degree), block.lambdas[0].float() * mlp_rows[row_index])
        for value, (kind, m_degree, a_degree) in zip(attention_rows, attention_map):
            target = (m_degree, a_degree + 1) if kind == "a" else (m_degree + 1, a_degree + 1); add(updated, target, block.lambdas[0].float() * value)
        current = updated
    down = model.blocks[1].mlp.Down.weight.float(); return {key: value[0, position] @ down for key, value in current.items()}


def metrics(prediction, exact, delta, rank):
    exact_contribution = exact * delta; contribution = prediction * delta; exact_support = exact_contribution.topk(rank).indices; support = contribution.topk(rank).indices
    return {"gradient_cosine": cosine(prediction, exact), "gradient_relative_error": float((prediction - exact).norm() / exact.norm().clamp_min(1e-30)), "contribution_cosine": cosine(contribution, exact_contribution), "true_leverage_ratio": float(exact_contribution[support].sum() / exact_contribution[exact_support].sum().clamp_min(1e-30))}


def main():
    config = json.loads(CONFIG.read_text()); causal = json.loads(Path(config["source_config"]).read_text()); old = json.loads(Path(config["source_results"]).read_text()); source = json.loads(Path(causal["source_config"]).read_text()); tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True); flags = np.stack([vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]); positive = flags.sum(0).astype(np.float32); negative = len(flags) - positive; contrast = torch.as_tensor(flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None], device="cuda", dtype=torch.float32)
    model = load_full_model("cuda"); model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    corpus = np.load(source["corpus"], mmap_mode="r"); heldout_start = source["start_row"] + source["discovery_rows"]
    selected = []
    for feature in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature]; selected.extend((next(row for row in rows if row["selection"] == "extreme"), next(row for row in rows if row["selection"] == "uniform")))
    records = []; heatmaps = []
    for site, row in enumerate(selected):
        feature = FEATURES.index(row["feature"]); vector = contrast[:, feature]; donor_raw = np.asarray(corpus[heldout_start + row["donor_row"], :source["sequence_length"]], dtype=np.int64); recipient_raw = np.asarray(corpus[heldout_start + row["recipient_row"], :source["sequence_length"]], dtype=np.int64); donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]; recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad(): donor_trace = forward_layers_native(model, donor_ids, 1); recipient_trace = forward_layers_native(model, recipient_ids, 17)
        delta = product_at(model, donor_trace, 1, row["donor_position"]) - product_at(model, recipient_trace, 1, row["recipient_position"]); exact = semantic_gradient(model, recipient_trace, 1, row["recipient_position"], vector); grid = bidegree_program(model, recipient_trace, row["recipient_position"], vector, config["maximum_total_degree"])
        select = lambda predicate: sum((value for key, value in grid.items() if predicate(*key)), torch.zeros_like(exact)); predictions = {"full": select(lambda m,a: True), "mlp_only": select(lambda m,a: a == 0), "attention_only": select(lambda m,a: m == 0), "axes_union": select(lambda m,a: m == 0 or a == 0), "rectangle4": select(lambda m,a: m <= 4 and a <= 4), "rectangle8": select(lambda m,a: m <= 8 and a <= 8), "rectangle12": select(lambda m,a: m <= 12 and a <= 12), "total12": select(lambda m,a: m + a <= 12), "total16": select(lambda m,a: m + a <= 16)}
        records.append({"feature": row["feature"], "selection": row["selection"], "metrics": {name: metrics(value, exact, delta, config["support_rank"]) for name, value in predictions.items()}})
        heatmap = np.zeros((17,17));
        for (m_degree,a_degree), value in grid.items():
            if m_degree <= 16 and a_degree <= 16: heatmap[m_degree,a_degree] = float(value.norm())
        heatmaps.append(heatmap); print("bidegree site", site + 1, "/", len(selected), row["feature"], flush=True)
    methods = list(records[0]["metrics"]); summaries = {method: {metric: float(np.median([row["metrics"][method][metric] for row in records])) for metric in ("gradient_cosine", "gradient_relative_error", "contribution_cosine", "true_leverage_ratio")} for method in methods}; full, mlp, attention, axes, rectangle4, rectangle8 = (summaries[name] for name in ("full","mlp_only","attention_only","axes_union","rectangle4","rectangle8"))
    hypotheses = {"B0_bidegree_recomposition_is_exact": {"supported": all(row["metrics"]["full"]["gradient_cosine"] >= .999 and row["metrics"]["full"]["gradient_relative_error"] <= .02 for row in records)}, "B1_both_operator_families_are_material": {"supported": mlp["gradient_cosine"] < .90 and attention["gradient_cosine"] < .90}, "B2_cross_family_paths_are_necessary": {"supported": axes["gradient_cosine"] < .80}, "B3_eight_by_eight_rectangle_is_sufficient": {"supported": rectangle8["gradient_cosine"] >= .90}, "B4_four_by_four_rectangle_is_insufficient": {"supported": rectangle4["gradient_cosine"] < .80}, "B5_eight_by_eight_preserves_sparse_leverage": {"supported": rectangle8["true_leverage_ratio"] >= .80}}
    median_heatmap = np.median(np.stack(heatmaps), 0); OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# MLP×attention bidegree pilot for semantic suffix gradients", "", "Every exact backward path is indexed by how many MLP responses and attention responses it contains; the within-block attention-after-MLP term increments both counts. Sixteen frozen sites test pure-family axes, cross-family rectangles, and total-degree controls.", "", "## Locked hypotheses", ""]
    for name, value in hypotheses.items(): lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "![Bidegree pilot](bilin18_layer1_suffix_bidegree_pilot.png)", "", "| Program | Gradient cosine | Relative error | Contribution cosine | Leverage ratio |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        value = summaries[method]; lines.append(f"| {method} | {value['gradient_cosine']:.3f} | {value['gradient_relative_error']:.3f} | {value['contribution_cosine']:.3f} | {value['true_leverage_ratio']:.3f} |")
    lines += ["", "## Guardrail", "", "Bidegree is an invariant interaction-count description, not a localization to particular layers or heads. A rectangle can still contain many exact paths executed through the native state-conditioned operators."]
    REPORT.write_text("\n".join(lines) + "\n")
    fig, axes_plot = plt.subplots(1,2,figsize=(12,4.8),layout="constrained"); image=axes_plot[0].imshow(np.log10(median_heatmap.T + 1e-12),origin="lower",aspect="auto",cmap="magma"); axes_plot[0].set(xlabel="MLP response count",ylabel="Attention response count",title="Median log10 path-family norm"); fig.colorbar(image,ax=axes_plot[0]); names=("mlp_only","attention_only","axes_union","rectangle4","rectangle8","total12","total16","full"); axes_plot[1].bar(range(len(names)),[summaries[name]["gradient_cosine"] for name in names]); axes_plot[1].set_xticks(range(len(names)),names,rotation=35,ha="right"); axes_plot[1].set(ylabel="Median gradient cosine",title="Which operator compositions matter?"); axes_plot[1].grid(axis="y",alpha=.2); fig.savefig(FIGURE,dpi=180);plt.close(fig)
    print(json.dumps({"hypotheses":hypotheses,"summaries":summaries},indent=2));print(REPORT)


if __name__ == "__main__": main()
