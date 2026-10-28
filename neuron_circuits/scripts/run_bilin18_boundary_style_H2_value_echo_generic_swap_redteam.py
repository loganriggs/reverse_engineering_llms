#!/usr/bin/env python3
"""Distinguish the spelling echo from generic one-token substitution geometry."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from run_bilin18_boundary_dialect_counterfactual import select_records
from run_bilin18_boundary_style_H2_value_echo import collect_value_deltas, fit_bases, project
from run_bilin18_boundary_style_lexical_generalization import token_pairs
from run_bilin18_layer01_supervised_feature_census import FEATURES, vocabulary_features
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_boundary_style_H2_value_echo_generic_swap_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin18_boundary_style_H2_value_echo_generic_swap_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_BOUNDARY_STYLE_H2_VALUE_ECHO_GENERIC_SWAP_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin18_boundary_style_H2_value_echo_generic_swap_redteam.png"


def explained_energy(values, basis):
    prediction = project(values, basis)
    denominator = values.double().square().sum().clamp_min(1e-30)
    return float(1 - (values.double() - prediction.double()).square().sum() / denominator)


def main():
    config = json.loads(CONFIG.read_text())
    base = json.loads(Path(config["source_config"]).read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    corpus = np.load(base["corpus"], mmap_mode="r")
    counts = np.bincount(np.asarray(corpus).reshape(-1), minlength=tokenizer.vocab_size)
    flags = np.stack(
        [vocabulary_features(tokenizer.decode([index])) for index in range(tokenizer.vocab_size)]
    )
    boundary_flags = flags[:, FEATURES.index("newline")].astype(bool) | flags[
        :, FEATURES.index("period")
    ].astype(bool)
    spelling_pairs = token_pairs(tokenizer, base["spelling_pairs"])
    control_pairs = token_pairs(tokenizer, config["control_pairs"])

    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    spelling_cues = spelling_pairs[4:]
    panels = {}
    counts_by_panel = {}
    for family, pairs in (("spelling", spelling_cues), ("control", control_pairs)):
        for split, (begin, end) in enumerate(base["splits"]):
            raw = np.asarray(corpus[begin:end, :128], dtype=np.int64)
            maximum = 64 if family == "spelling" else config["examples_per_label_per_split"]
            records = select_records(
                raw,
                boundary_flags,
                pairs,
                counts,
                127,
                maximum,
                config["seed"] + split + (0 if family == "spelling" else 100),
            )
            print("generic swap panel", family, split, len(records), flush=True)
            panels[(family, split)] = collect_value_deltas(model, raw, records, config)
            counts_by_panel[f"{family}_{split}"] = len(records)

    spelling_basis, _ = fit_bases(panels[("spelling", 0)], config)
    control_basis, _ = fit_bases(panels[("control", 0)], config)
    metrics = {
        "spelling_self_heldout_r1": explained_energy(panels[("spelling", 1)], spelling_basis[:1]),
        "control_self_heldout_r1": explained_energy(panels[("control", 1)], control_basis[:1]),
        "spelling_on_control_split0_r1": explained_energy(panels[("control", 0)], spelling_basis[:1]),
        "spelling_on_control_split1_r1": explained_energy(panels[("control", 1)], spelling_basis[:1]),
        "control_on_spelling_heldout_r1": explained_energy(panels[("spelling", 1)], control_basis[:1]),
        "axis_abs_cosine": float(abs(spelling_basis[0].double() @ control_basis[0].double())),
    }
    c0 = counts_by_panel["control_0"] == 80 and counts_by_panel["control_1"] == 80
    c1 = metrics["spelling_on_control_split0_r1"] < 0.25 and metrics["spelling_on_control_split1_r1"] < 0.25
    c2 = metrics["control_on_spelling_heldout_r1"] < 0.25
    c3 = metrics["axis_abs_cosine"] < 0.5
    cross_max = max(
        metrics["spelling_on_control_split0_r1"],
        metrics["spelling_on_control_split1_r1"],
        metrics["control_on_spelling_heldout_r1"],
    )
    c4 = metrics["spelling_self_heldout_r1"] >= 2 * max(cross_max, 1e-30)
    hypotheses = {
        "C0_control_panel_is_balanced": {"supported": c0},
        "C1_unrelated_swaps_do_not_use_spelling_axis": {"supported": c1},
        "C2_unrelated_axis_does_not_explain_spelling": {"supported": c2},
        "C3_rank1_axes_are_distinct": {"supported": c3},
        "C4_spelling_axis_has_twofold_specificity": {"supported": c4},
        "C5_echo_is_not_generic_token_swap_geometry": {"supported": all((c0, c1, c2, c3, c4))},
    }
    for value in hypotheses.values():
        value["supported"] = bool(value["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "counts": counts_by_panel, "metrics": metrics}, indent=2))

    lines = [
        "# Generic-token-swap red team of the L17H2 spelling echo",
        "",
        "The spelling rank-1 basis is compared with a basis fit to sixteen unrelated single-token substitution pairs. Both use the same boundary selection and Layer-17 H2 post-boundary value deltas; bases are fit on split 0 and evaluated on split 1.",
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
        "Distinct low-rank axes reject a generic one-token-perturbation bottleneck for this control panel. The controls are semantically heterogeneous, so failure to share one axis does not prove that the spelling direction is uniquely sociolinguistic; matched coherent control families are a stricter future comparison.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    labels = ["spell→spell", "control→control", "spell→control s0", "spell→control s1", "control→spell"]
    values = [
        metrics["spelling_self_heldout_r1"],
        metrics["control_self_heldout_r1"],
        metrics["spelling_on_control_split0_r1"],
        metrics["spelling_on_control_split1_r1"],
        metrics["control_on_spelling_heldout_r1"],
    ]
    figure, axis = plt.subplots(figsize=(9, 4.5), layout="constrained")
    axis.bar(np.arange(len(values)), values)
    axis.set(xticks=np.arange(len(values)), xticklabels=labels, ylabel="Rank-1 explained energy", title="Within-family versus cross-family H2 value axes")
    axis.grid(alpha=0.2)
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print(json.dumps({"hypotheses": hypotheses, "metrics": metrics}, indent=2))
    print(REPORT)


if __name__ == "__main__":
    main()
