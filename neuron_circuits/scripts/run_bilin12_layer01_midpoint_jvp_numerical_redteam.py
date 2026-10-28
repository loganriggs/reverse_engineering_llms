#!/usr/bin/env python3
"""Finite-difference convergence audit for the midpoint product JVP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_layer01_adaptive_chart_router import (
    cosine,
    finite_difference,
    midpoint_product_jvp,
)
from run_bilin18_layer8_semantic_counterfactuals_fixed import select_windows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer01_midpoint_jvp_numerical_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_layer01_midpoint_jvp_numerical_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_LAYER01_MIDPOINT_JVP_NUMERICAL_REDTEAM.md"


def main():
    red = json.loads(CONFIG.read_text())
    router = json.loads(Path(red["source_config"]).read_text())
    parent = json.loads(Path(router["source_config"]).read_text())
    config = json.loads(Path(parent["source_config"]).read_text())
    pairs = json.loads(Path(config["evaluation_pair_config"]).read_text())
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    arrays = {
        corpus: np.load(spec["path"], mmap_mode="r")[spec["start"]:spec["start"] + spec["rows"]]
        for corpus, spec in config["evaluation"].items()
    }
    records = []
    edit_position, position = config["edit_position"], config["prediction_position"]
    for corpus_index, (corpus, rows) in enumerate(arrays.items()):
        rng = np.random.default_rng(config["window_seed"] + 1000 + corpus_index)
        for family, family_pairs in pairs["new_families"].items():
            for source_token, target_token in family_pairs:
                row_index, start = select_windows(
                    rows, source_token, edit_position, position, red["contexts_per_pair"], rng,
                )[0]
                raw = np.asarray(rows[row_index, start:start + config["sequence_length"]], dtype=np.int64).copy()
                changed = raw.copy(); changed[edit_position] = target_token
                base_ids = torch.as_tensor(raw, device="cuda")[None]
                changed_ids = torch.as_tensor(changed, device="cuda")[None]
                for layer in config["layers"]:
                    tangent, point, direction = midpoint_product_jvp(
                        model, base_ids, changed_ids, layer, position, router["path_point"],
                    )
                    metrics = {}
                    for epsilon in red["epsilons"]:
                        finite = finite_difference(model, point, direction, layer, position, epsilon)
                        metrics[str(epsilon)] = {
                            "cosine": cosine(tangent, finite),
                            "relative_error": float((tangent - finite).norm() / finite.norm().clamp_min(1e-30)),
                        }
                    records.append({"corpus": corpus, "family": family, "pair": f"{source_token}->{target_token}", "layer": layer, "metrics": metrics})
                print("JVP numeric", corpus, family, source_token, target_token, flush=True)

    summaries = {}
    best_epsilons = []
    for layer in config["layers"]:
        summaries[str(layer)] = {}
        for corpus in config["evaluation"]:
            selected = [row for row in records if row["layer"] == layer and row["corpus"] == corpus]
            epsilon_rows = {}
            for epsilon in red["epsilons"]:
                metrics = [row["metrics"][str(epsilon)] for row in selected]
                epsilon_rows[str(epsilon)] = {
                    "minimum_cosine": min(row["cosine"] for row in metrics),
                    "median_relative_error": float(np.median([row["relative_error"] for row in metrics])),
                    "maximum_relative_error": max(row["relative_error"] for row in metrics),
                }
            best = min(red["epsilons"], key=lambda epsilon: epsilon_rows[str(epsilon)]["maximum_relative_error"])
            best_epsilons.append(best)
            summaries[str(layer)][corpus] = {"epsilons": epsilon_rows, "best_epsilon": best}

    shared_valid = []
    for epsilon in red["epsilons"]:
        if all(
            summaries[str(layer)][corpus]["epsilons"][str(epsilon)]["minimum_cosine"] >= 0.9999
            and summaries[str(layer)][corpus]["epsilons"][str(epsilon)]["maximum_relative_error"] <= 0.01
            for layer in config["layers"] for corpus in config["evaluation"]
        ):
            shared_valid.append(epsilon)
    best_shared_error = min(
        max(summaries[str(layer)][corpus]["epsilons"][str(epsilon)]["maximum_relative_error"] for layer in config["layers"] for corpus in config["evaluation"])
        for epsilon in red["epsilons"]
    )
    failed_error = max(
        summaries[str(layer)][corpus]["epsilons"]["0.001"]["maximum_relative_error"]
        for layer in config["layers"] for corpus in config["evaluation"]
    )
    hypotheses = {
        "N0_one_shared_epsilon_validates_the_jvp": {"supported": bool(shared_valid), "passing_epsilons": shared_valid},
        "N1_the_failed_epsilon_is_roundoff_limited": {"supported": failed_error - best_shared_error >= 0.01, "failed_error": failed_error, "best_shared_error": best_shared_error},
        "N2_direction_is_stable_across_the_sweep": {"supported": all(row["minimum_cosine"] >= 0.999 for layer_rows in summaries.values() for cell in layer_rows.values() for row in cell["epsilons"].values())},
        "N3_no_cell_requires_a_different_epsilon_regime": {"supported": len(set(best_epsilons)) == 1, "best_epsilons": best_epsilons},
    }
    OUTPUT.write_text(json.dumps({"config": red, "hypotheses": hypotheses, "summaries": summaries, "records": records}, indent=2, allow_nan=False))
    lines = ["# Numerical red-team of the midpoint product JVP", "", "A centered finite-difference sweep separates truncation error at large epsilon from float32 cancellation at small epsilon. The JVP implementation and examples are unchanged from the adaptive-chart experiment; only the numerical comparator step is varied.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Layer | Corpus | epsilon: min cosine / median error / max error | Best epsilon |", "|---:|---|---|---:|"]
    for layer in config["layers"]:
        for corpus in config["evaluation"]:
            row = summaries[str(layer)][corpus]
            values = "; ".join(f"{epsilon:g}: {row['epsilons'][str(epsilon)]['minimum_cosine']:.6f}/{row['epsilons'][str(epsilon)]['median_relative_error']:.4f}/{row['epsilons'][str(epsilon)]['maximum_relative_error']:.4f}" for epsilon in red["epsilons"])
            lines.append(f"| {layer} | {corpus} | {values} | {row['best_epsilon']:g} |")
    lines += ["", "A failed small-epsilon amplitude gate does not invalidate the analytic JVP if the finite difference converges toward it at a larger shared epsilon while direction remains stable. The original preregistered gate remains recorded as failed."]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(hypotheses, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
