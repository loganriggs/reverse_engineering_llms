#!/usr/bin/env python3
"""Locate whether L17H2's generic edit axis arises before or inside c_v."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import select_records
from run_bilin18_boundary_style_H2_value_echo import layer17_details, project
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_H2_edit_mode_origin.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_H2_edit_mode_origin.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_H2_EDIT_MODE_ORIGIN.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_H2_edit_mode_origin.png"


def leading_axis(values):
    covariance = values.double().T @ values.double()
    _, vectors = torch.linalg.eigh(covariance)
    return vectors[:, -1].float()


def explained(values, axis):
    prediction = project(values, axis[None])
    denominator = values.double().square().sum().clamp_min(1e-30)
    return float(1 - (values.double() - prediction.double()).square().sum() / denominator)


@torch.inference_mode()
def collect(model, raw, records, head, batch_size):
    dimension = model.config.n_embd
    residual_chunks, value_chunks = [], []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        size = len(batch)
        original = np.stack([raw[row[0], :128].copy() for row in batch])
        swapped = original.copy()
        for index, row in enumerate(batch):
            swapped[index, row[3]] = row[5]
        cue = torch.as_tensor([row[3] for row in batch], device="cuda")
        boundary = torch.as_tensor([row[1] for row in batch], device="cuda")
        _, mixed, details = layer17_details(model, original, swapped, cue, boundary)
        normalized = F.rms_norm(mixed, (dimension,))
        residual_delta = normalized[size:] - normalized[:size]
        value_delta = details["current_value"][size:, :, head] - details["current_value"][:size, :, head]
        position = torch.arange(original.shape[1], device="cuda")[None]
        mask = (position > boundary[:, None]) & (position < 127)
        residual_chunks.append(residual_delta[mask].float().cpu())
        value_chunks.append(value_delta[mask].float().cpu())
    return torch.cat(residual_chunks), torch.cat(value_chunks)


def normalized_cosine(first, second):
    return float(abs(F.cosine_similarity(first.double(), second.double(), dim=0)))


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    controls = json.loads(Path(config["generic_swap_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack(
        [vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]
    )
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[
        :, FEATURES.index("period")
    ].astype(bool)
    spelling_pairs = token_pairs(tokenizer, base["spelling_pairs"])[4:]
    control_pairs = token_pairs(tokenizer, controls["control_pairs"])

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    panels = {}
    for family, pairs, maximum in (
        ("spelling", spelling_pairs, 64),
        ("control", control_pairs, config["examples_per_label_per_split"]),
    ):
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            records = select_records(
                raw,
                boundary_flags,
                pairs,
                counts,
                127,
                maximum,
                config["seed"] + split + (0 if family == "spelling" else 100),
            )
            print("edit origin", family, split, len(records), flush=True)
            panels[(family, split)] = collect(
                model, raw, records, config["head"], config["batch_size"]
            )

    axes = {}
    metrics = {}
    for family in ("spelling", "control"):
        residual0, value0 = panels[(family, 0)]
        residual1, value1 = panels[(family, 1)]
        axes[(family, "residual")] = leading_axis(residual0)
        axes[(family, "value")] = leading_axis(value0)
        metrics[f"{family}_residual_r1"] = explained(residual1, axes[(family, "residual")])
        metrics[f"{family}_value_r1"] = explained(value1, axes[(family, "value")])

    metrics["residual_axis_cosine"] = normalized_cosine(
        axes[("spelling", "residual")], axes[("control", "residual")]
    )
    metrics["value_axis_cosine"] = normalized_cosine(
        axes[("spelling", "value")], axes[("control", "value")]
    )

    block = model.blocks[17].attn
    head_dim = model.config.n_embd // model.config.n_head
    start = config["head"] * head_dim
    weight = block.c_v.weight.detach().float().cpu()[start : start + head_dim]
    for family in ("spelling", "control"):
        mapped = weight @ axes[(family, "residual")]
        metrics[f"{family}_mapped_residual_to_value_cosine"] = normalized_cosine(
            mapped, axes[(family, "value")]
        )
    left, _, right = torch.linalg.svd(weight.double(), full_matrices=False)
    metrics["weight_top_output_spelling_value_cosine"] = normalized_cosine(
        left[:, 0].float(), axes[("spelling", "value")]
    )
    metrics["weight_top_output_control_value_cosine"] = normalized_cosine(
        left[:, 0].float(), axes[("control", "value")]
    )
    metrics["weight_top_input_spelling_residual_cosine"] = normalized_cosine(
        right[0].float(), axes[("spelling", "residual")]
    )

    r0 = all(metrics[f"{family}_residual_r1"] >= 0.50 for family in ("spelling", "control"))
    r1 = metrics["residual_axis_cosine"] >= 0.8
    r2 = all(
        metrics[f"{family}_mapped_residual_to_value_cosine"] >= 0.9
        for family in ("spelling", "control")
    )
    r3 = all(
        metrics[f"{family}_value_r1"] >= metrics[f"{family}_residual_r1"] + 0.10
        for family in ("spelling", "control")
    )
    r4 = all(
        metrics[f"weight_top_output_{family}_value_cosine"] >= 0.5
        for family in ("spelling", "control")
    )
    hypotheses = {
        "R0_edit_mode_preexists_value_projection": {"supported": r0},
        "R1_prevalue_mode_is_generic": {"supported": r1},
        "R2_weights_map_residual_axis_to_value_axis": {"supported": r2},
        "R3_value_projection_canonicalizes_the_mode": {"supported": r3},
        "R4_global_weight_outlier_is_the_data_mode": {"supported": r4},
        "R5_weight_predictable_generic_edit_bus": {"supported": all((r0, r1, r2, r3, r4))},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "metrics": metrics}, indent=2))

    lines = [
        "# Origin of the generic L17H2 edit mode",
        "",
        "Post-boundary normalized residual deltas and native H2 current-value deltas are compared for spelling and sixteen unrelated token swaps. Split-0 axes are frozen and evaluated on split 1; the native H2 c_v slice then maps the residual axes into value space without fitting.",
        "",
        "## Locked results",
        "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += ["", "| Metric | Value |", "|---|---:|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value:.4f} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A shared residual axis would locate the edit bus before H2. Strong native weight mapping but weak global singular-vector alignment would mean c_v faithfully transforms a data-conditioned mode rather than selecting its largest isotropic weight mode. Canonicalization requires a genuine increase in held-out rank-1 energy, not merely high axis cosine.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = ["spelling residual", "spelling value", "control residual", "control value"]
    values = [metrics["spelling_residual_r1"], metrics["spelling_value_r1"], metrics["control_residual_r1"], metrics["control_value_r1"]]
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    axis.bar(np.arange(4), values)
    axis.set(xticks=np.arange(4), xticklabels=labels, ylabel="Held-out rank-1 energy", title="Where the generic edit mode becomes low rank")
    axis.grid(alpha=0.2)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "metrics": metrics}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
