#!/usr/bin/env python3
"""Localize attention modulation in the Layer-1-to-logit backward program."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin12_layer01_dynamic_alignment_replication import forward_layers_native, selected_delta
from run_bilin12_layer01_frozen_taylor_behavior import logits_for_deltas
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_layer1_output_aware_coalitions import semantic_gradient
from run_bilin18_layer1_suffix_bidegree_pilot import metrics
from run_bilin18_layer1_suffix_degree_pilot import attention_vjp, final_adjoint, mlp_vjp
from run_bilin18_layer1_supervised_coalition_causality import class_score, product_at
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer1_suffix_layer_localization_pilot.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer1_suffix_layer_localization_pilot.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER1_SUFFIX_LAYER_LOCALIZATION_PILOT.md"
FIGURE = TENSOR_ROOT / "bilin18_layer1_suffix_layer_localization_pilot.png"


def masked_attention_program(model, trace, position, contrast, masks):
    """Apply exact local VJPs, omitting only attention response in masked blocks."""
    names = list(masks)
    final = final_adjoint(model, trace[-1]["output"], position, contrast)
    current = final.repeat(len(names), 1, 1, 1)
    for index in range(len(model.blocks) - 1, 1, -1):
        block = model.blocks[index]
        mlp_rows = mlp_vjp(block, trace[index]["pre"].float(), current)
        updated = current + mlp_rows
        enabled = [row for row, name in enumerate(names) if index in masks[name]]
        if enabled:
            selected = updated[enabled]
            previous = trace[index - 1]["output"].float()
            x0 = trace[index]["x0"].float()
            mixed = block.lambdas[0].float() * previous + block.lambdas[1].float() * x0
            bus = trace[index - 1]["value_bus"]
            routed = attention_vjp(block, mixed, bus, selected)
            updated[enabled] += routed
        current = block.lambdas[0].float() * updated
    down = model.blocks[1].mlp.Down.weight.float()
    return {
        name: current[row, 0, position] @ down
        for row, name in enumerate(names)
    }


def make_masks(window_sizes):
    layers = list(range(2, 18))
    masks = {"mlp_only": set(), "full": set(layers)}
    for width in window_sizes:
        masks[f"source_{width}"] = set(layers[:width])
        masks[f"logit_{width}"] = set(layers[-width:])
    masks["both_ends_4"] = set(layers[:4] + layers[-4:])
    for layer in layers:
        masks[f"single_{layer}"] = {layer}
        masks[f"leaveout_{layer}"] = set(layers) - {layer}
    return masks


def summarize(records, method):
    rows = [row["metrics"][method] for row in records]
    shifts = np.asarray([row["finite_shifts"][method] for row in records])
    return {
        **{
            key + "_median": float(np.median([value[key] for value in rows]))
            for key in (
                "gradient_cosine", "gradient_relative_error",
                "contribution_cosine", "true_leverage_ratio",
            )
        },
        "gradient_cosine_q10": float(np.quantile(
            [value["gradient_cosine"] for value in rows], .1,
        )),
        "finite_directional_fraction": float(np.mean(shifts > 0)),
        "finite_mean_shift": float(shifts.mean()),
    }


def main():
    config = json.loads(CONFIG.read_text())
    causal = json.loads(Path(config["source_config"]).read_text())
    old = json.loads(Path(config["source_results"]).read_text())
    source = json.loads(Path(causal["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    flags = np.stack([
        vocabulary_features(tokenizer.decode([index]))
        for index in range(tokenizer.vocab_size)
    ])
    positive = flags.sum(0).astype(np.float32)
    negative = len(flags) - positive
    contrast = torch.as_tensor(
        flags / positive.clip(1)[None] - (1 - flags) / negative.clip(1)[None],
        device="cuda", dtype=torch.float32,
    )
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    corpus = np.load(source["corpus"], mmap_mode="r")
    heldout_start = source["start_row"] + source["discovery_rows"]
    selected = []
    for feature in causal["features"]:
        rows = [row for row in old["records"] if row["feature"] == feature]
        selected.extend((
            next(row for row in rows if row["selection"] == "extreme"),
            next(row for row in rows if row["selection"] == "uniform"),
        ))

    masks = make_masks(config["window_sizes"])
    support_rank = config["support_rank"]
    records = []
    for site, row in enumerate(selected):
        feature = FEATURES.index(row["feature"])
        vector = contrast[:, feature]
        donor_raw = np.asarray(
            corpus[heldout_start + row["donor_row"], :source["sequence_length"]],
            dtype=np.int64,
        )
        recipient_raw = np.asarray(
            corpus[heldout_start + row["recipient_row"], :source["sequence_length"]],
            dtype=np.int64,
        )
        donor_ids = torch.as_tensor(donor_raw, device="cuda")[None]
        recipient_ids = torch.as_tensor(recipient_raw, device="cuda")[None]
        with torch.no_grad():
            donor_trace = forward_layers_native(model, donor_ids, 1)
            recipient_trace = forward_layers_native(model, recipient_ids, 17)
        delta = (
            product_at(model, donor_trace, 1, row["donor_position"])
            - product_at(model, recipient_trace, 1, row["recipient_position"])
        )
        exact = semantic_gradient(
            model, recipient_trace, 1, row["recipient_position"], vector,
        )
        predictions = masked_attention_program(
            model, recipient_trace, row["recipient_position"], vector, masks,
        )
        predictions["exact"] = exact
        supports = {
            name: (prediction * delta).topk(support_rank).indices
            for name, prediction in predictions.items()
        }
        deltas = {
            "base": torch.zeros_like(delta),
            **{
                name: selected_delta(delta, support)
                for name, support in supports.items()
            },
        }
        with torch.no_grad():
            logits = logits_for_deltas(
                model, 1, recipient_trace[1], deltas, row["recipient_position"],
            )
        base = class_score(logits["base"], vector)
        finite = {
            name: class_score(value, vector) - base
            for name, value in logits.items() if name != "base"
        }
        records.append({
            "feature": row["feature"],
            "selection": row["selection"],
            "metrics": {
                name: metrics(prediction, exact, delta, support_rank)
                for name, prediction in predictions.items()
            },
            "finite_shifts": finite,
        })
        print(
            "layer-localization site", site + 1, "/", len(selected),
            row["feature"], row["selection"], flush=True,
        )

    methods = list(records[0]["metrics"])
    summaries = {method: summarize(records, method) for method in methods}
    exact_shift = summaries["exact"]["finite_mean_shift"]
    best4 = max(summaries["source_4"], summaries["logit_4"], key=lambda x: x["gradient_cosine_median"])
    best8 = max(summaries["source_8"], summaries["logit_8"], key=lambda x: x["gradient_cosine_median"])
    best_single = max(
        (summaries[f"single_{layer}"] for layer in range(2, 18)),
        key=lambda x: x["gradient_cosine_median"],
    )
    worst_leaveout = min(
        (summaries[f"leaveout_{layer}"] for layer in range(2, 18)),
        key=lambda x: x["gradient_cosine_median"],
    )
    hypotheses = {
        "L0_full_recurrence_is_exact": {
            "supported": all(
                row["metrics"]["full"]["gradient_cosine"] >= .999
                and row["metrics"]["full"]["gradient_relative_error"] <= .02
                for row in records
            ),
        },
        "L1_four_block_attention_window_is_sufficient": {
            "supported": best4["gradient_cosine_median"] >= .90
            and best4["finite_mean_shift"] >= .80 * exact_shift,
        },
        "L2_eight_block_attention_window_is_sufficient": {
            "supported": best8["gradient_cosine_median"] >= .90
            and best8["finite_mean_shift"] >= .80 * exact_shift,
        },
        "L3_one_attention_block_is_sufficient": {
            "supported": best_single["gradient_cosine_median"] >= .80,
        },
        "L4_one_attention_block_is_decisive": {
            "supported": 1 - worst_leaveout["gradient_cosine_median"] >= .10,
        },
        "L5_both_ends_beat_either_four_block_end": {
            "supported": summaries["both_ends_4"]["gradient_cosine_median"]
            - best4["gradient_cosine_median"] >= .10,
        },
        "L6_mlp_only_retains_causal_direction": {
            "supported": summaries["mlp_only"]["finite_directional_fraction"] >= .90,
        },
    }
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "summaries": summaries,
        "records": records,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    display = [
        "mlp_only", "source_1", "source_2", "source_4", "source_8",
        "source_12", "logit_1", "logit_2", "logit_4", "logit_8",
        "logit_12", "both_ends_4", "full",
    ]
    lines = [
        "# Suffix attention-layer localization pilot", "",
        "Exact local backward operators are retained, but the attention response is enabled only in specified suffix blocks. MLP and residual responses remain native in every block. Sixteen frozen sites are evaluated with exact finite top-64 channel patches.",
        "", "## Locked hypotheses", "",
    ]
    for name, value in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if value['supported'] else 'not supported'}.**")
    lines += [
        "", "![Layer localization](bilin18_layer1_suffix_layer_localization_pilot.png)",
        "", "| Attention-enabled blocks | Gradient cosine | Q10 | Contribution cosine | Finite direction | Shift / exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in display:
        value = summaries[method]
        lines.append(
            f"| {method} | {value['gradient_cosine_median']:.3f} | "
            f"{value['gradient_cosine_q10']:.3f} | "
            f"{value['contribution_cosine_median']:.3f} | "
            f"{value['finite_directional_fraction']:.3f} | "
            f"{value['finite_mean_shift'] / exact_shift:.3f} |"
        )
    lines += [
        "", "## Guardrail", "",
        "This is a local backward-mechanism intervention: disabling a block's attention response removes its Jacobian contribution while preserving the native forward state and all MLP responses. It localizes sensitivity routing, not a standalone forward circuit.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    sizes = config["window_sizes"]
    axes[0].plot(
        sizes, [summaries[f"source_{width}"]["gradient_cosine_median"] for width in sizes],
        marker="o", label="nearest Layer 1",
    )
    axes[0].plot(
        sizes, [summaries[f"logit_{width}"]["gradient_cosine_median"] for width in sizes],
        marker="o", label="nearest logits",
    )
    axes[0].axhline(summaries["mlp_only"]["gradient_cosine_median"], color="gray", linestyle=":", label="MLP-only")
    axes[0].set(xlabel="Attention-enabled suffix blocks", ylabel="Median gradient cosine", title="Where does attention modulation accumulate?")
    axes[0].legend(); axes[0].grid(alpha=.2)
    layers = list(range(2, 18))
    single = [summaries[f"single_{layer}"]["gradient_cosine_median"] for layer in layers]
    leaveout_drop = [1 - summaries[f"leaveout_{layer}"]["gradient_cosine_median"] for layer in layers]
    axes[1].plot(layers, single, marker="o", label="only this block enabled")
    axes[1].plot(layers, leaveout_drop, marker="o", label="cosine loss when omitted")
    axes[1].set(xlabel="Suffix block", ylabel="Median cosine / cosine loss", title="Single-block sufficiency and necessity")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(FIGURE, dpi=180); plt.close(fig)
    print(json.dumps({"hypotheses": hypotheses, "summaries": summaries}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
