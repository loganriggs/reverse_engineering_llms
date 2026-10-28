#!/usr/bin/env python3
"""Exploratory scale and corpus red-team for decomposition causal scores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from evaluate_bilin18_decomposition_metrics import (
    DISPLAY,
    ROOT,
    causal_evaluation,
    corpus_rows,
    load_candidates,
)
from run_bilin18_qkov_causal_edges import load_full_model


CONFIG = Path("configs/bilin18_decomposition_evaluation.json")
OUTPUT = ROOT / "causal_metric_redteam.json"
FIGURE = ROOT / "causal_scale_sweep.png"
REPORT = ROOT / "CAUSAL_METRIC_REDTEAM.md"


def main():
    config = json.loads(CONFIG.read_text())
    config["bootstrap_resamples"] = 1000
    device = "cuda"
    all_candidates = load_candidates(device)
    candidates = {name: all_candidates[name] for name in config["primary_models"]}
    candidates["native_control"] = None
    model = load_full_model(device)
    corpora = corpus_rows(config)

    scale_specs = {
        **{f"gaussian_{scale:.2f}": {"kind": "gaussian", "scale": scale}
           for scale in (0.02, 0.05, 0.10, 0.20)},
        **{f"natural_mix_{scale:.2f}": {"kind": "natural_mix", "scale": scale}
           for scale in (0.05, 0.10, 0.25, 0.50)},
    }
    fineweb = {name: {} for name in candidates}
    # Evaluate one scale per call so Gaussian directions are paired across scale.
    for name, spec in scale_specs.items():
        print("scale", name, flush=True)
        rows = causal_evaluation(
            model, candidates, corpora["fineweb"][:24], {name: spec}, config, device)
        for candidate in candidates:
            fineweb[candidate][name] = rows[candidate][name]

    ood = {}
    primary_specs = {
        "gaussian_0.10": {"kind": "gaussian", "scale": 0.10},
        "natural_mix_0.25": {"kind": "natural_mix", "scale": 0.25},
    }
    for corpus in ("wikitext103", "tinystories"):
        print("ood", corpus, flush=True)
        ood[corpus] = causal_evaluation(
            model, candidates, corpora[corpus][:24], primary_specs, config, device)

    result = {
        "status": "exploratory_red_team_after_confirmatory_run",
        "rows_per_condition": 24,
        "fineweb_scale_sweep": fineweb,
        "ood_primary_scales": ood,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False))

    display = DISPLAY | {"native_control": "Native control"}
    colors = {"cp2048_weight": "#4C78A8", "cp2048_activation": "#F58518",
              "routed_mixed": "#54A24B", "native_control": "#333333"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, kind, scales in ((axes[0], "gaussian", (0.02, 0.05, 0.10, 0.20)),
                               (axes[1], "natural_mix", (0.05, 0.10, 0.25, 0.50))):
        for candidate in candidates:
            values = [fineweb[candidate][f"{kind}_{scale:.2f}"]["fisher_effect_fidelity"]
                      for scale in scales]
            axis.plot(scales, values, marker="o", label=display[candidate], color=colors[candidate])
        axis.axhline(0.9, color="black", linestyle="--", linewidth=1)
        axis.axhline(0, color="#999999", linewidth=0.8)
        axis.set_xlabel("Intervention scale")
        axis.set_title("Gaussian directions" if kind == "gaussian" else "Natural-activation mixture")
    axes[0].set_ylabel("Softmax-Fisher effect fidelity")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = [
        "# Causal metric red-team", "",
        "This is an exploratory analysis run after the confirmatory scorecard. It tests whether the causal ranking is an artifact of one intervention magnitude or one corpus.", "",
        "![Causal scale sweep](causal_scale_sweep.png)", "",
        "## FineWeb scale sweep", "",
        "| Intervention | Direct CP | Activation CP | Routed CP | Native control |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in scale_specs:
        lines.append(
            f"| {condition} | "
            + " | ".join(f"{fineweb[name][condition]['fisher_effect_fidelity']:.3f}"
                         for name in candidates) + " |"
        )
    lines += ["", "## OOD primary scales", "",
              "| Corpus/intervention | Direct CP | Activation CP | Routed CP | Native control |",
              "|---|---:|---:|---:|---:|"]
    for corpus, values in ood.items():
        for condition in primary_specs:
            lines.append(
                f"| {corpus} / {condition} | "
                + " | ".join(f"{values[name][condition]['fisher_effect_fidelity']:.3f}"
                             for name in candidates) + " |"
            )
    lines += ["", "The native-control path must equal 1.0 up to numerical precision. A stable ordering across scales and corpora argues against a threshold artifact; it does not establish fidelity to other intervention families.", ""]
    REPORT.write_text("\n".join(lines))
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
