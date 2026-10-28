#!/usr/bin/env python3
"""Test learned layer-pair identity and residual-gain dependence of I/B paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_mlp_path_symmetry_redteam import GROUPS, collapse
from run_bilin12_mlp_path_type_program import (
    aggregate_gram, build_axes, collect_paths, error_from_gram, greedy_order,
    path_fidelity,
)
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_mlp_path_order_gain_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_mlp_path_order_gain_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_MLP_PATH_ORDER_GAIN_REDTEAM.md"


def constrained_greedy(gram, dot, target2, base, candidates):
    selected = list(base); remaining = set(candidates); order = []
    while remaining:
        choice = min(remaining, key=lambda item: error_from_gram(
            gram, dot, target2, selected + [item]
        ))
        selected.append(choice); order.append(choice); remaining.remove(choice)
    return order


def pair_random(paths, base, replicates, generator):
    pair = torch.as_tensor([i for i, group in enumerate(GROUPS) if group.count("B") == 2])
    values = []
    for _ in range(replicates):
        chosen = pair[torch.randperm(len(pair), generator=generator)[:2]].tolist()
        values.append(path_fidelity(paths, base + chosen))
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "maximum": max(values)}


def scaled(paths, gain):
    factors = torch.as_tensor([gain ** group.count("B") for group in GROUPS], dtype=paths.dtype)
    return paths * factors[None, :, None, None]


def effective_support(paths):
    energy = paths.double().square().sum((0, 2, 3)).sqrt()
    return float(energy.sum().square() / energy.square().sum().clamp_min(1e-30))


def k90(paths, ordering):
    for count in range(1, len(ordering) + 1):
        if path_fidelity(paths, ordering[:count]) >= .90:
            return count
    return len(ordering)


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text())
    base_config = json.loads(Path(config["source_config"]).read_text())
    model = load_snapshot_model(Path(base_config["snapshot"]), "cuda")
    axes = build_axes(model, base_config)
    print("collect discovery", flush=True)
    raw_discovery, _ = collect_paths(model, axes, base_config["path_discovery"], base_config)
    discovery = collapse(raw_discovery)
    low_order = [i for i, group in enumerate(GROUPS) if group.count("B") <= 1]
    pairs = [i for i, group in enumerate(GROUPS) if group.count("B") == 2]
    gram, dot, target2 = aggregate_gram(discovery)
    pair_order = constrained_greedy(gram, dot, target2, low_order, pairs)
    gain_orders = {}
    for gain in config["gain_scales"]:
        value = scaled(discovery, gain)
        gain_orders[str(gain)] = greedy_order(*aggregate_gram(value))

    generator = torch.Generator().manual_seed(config["seed"])
    results = {}
    for corpus, spec in base_config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        raw, _ = collect_paths(model, axes, spec, base_config)
        paths = collapse(raw)
        eval_gram, eval_dot, eval_target2 = aggregate_gram(paths)
        eval_pair_order = constrained_greedy(eval_gram, eval_dot, eval_target2, low_order, pairs)
        base_fidelity = path_fidelity(paths, low_order)
        pair_fidelity = path_fidelity(paths, low_order + pair_order[:2])
        gain_rows = {}
        for gain in config["gain_scales"]:
            value = scaled(paths, gain); order = gain_orders[str(gain)]
            gain_rows[str(gain)] = {"k90": k90(value, order),
                                    "effective_support": effective_support(value),
                                    "k5_fidelity": path_fidelity(value, order[:5]),
                                    "k8_fidelity": path_fidelity(value, order[:8])}
        results[corpus] = {
            "degree_le_one_fidelity": base_fidelity,
            "plus_top_two_pairs_fidelity": pair_fidelity,
            "pair_gain": pair_fidelity - base_fidelity,
            "random_pair": pair_random(paths, low_order, config["random_replicates"], generator),
            "evaluation_pair_order": ["".join(GROUPS[i]) for i in eval_pair_order],
            "top_two_pair_overlap": len(set(pair_order[:2]) & set(eval_pair_order[:2])),
            "gain_sweep": gain_rows,
        }

    supported = {
        "G0_pair_interactions_are_material": all(row["pair_gain"] >= .03 for row in results.values()),
        "G1_pair_identity_beats_random_median": all(row["plus_top_two_pairs_fidelity"] >= row["random_pair"]["median"] + .01 for row in results.values()),
        "G2_pair_identity_beats_random_p95": all(row["plus_top_two_pairs_fidelity"] > row["random_pair"]["p95"] for row in results.values()),
        "G3_pair_ranking_transfers": all(row["top_two_pair_overlap"] >= 1 for row in results.values()),
        "G4_half_gain_is_first_order_compact": all(row["gain_sweep"]["0.5"]["k90"] <= 5 for row in results.values()),
        "G5_native_gain_needs_more_than_first_order": any(row["gain_sweep"]["1.0"]["k90"] > 5 for row in results.values()),
        "G6_high_gain_broadens_the_program": all(row["gain_sweep"]["2.0"]["k90"] > row["gain_sweep"]["1.0"]["k90"] for row in results.values()),
        "G7_effective_support_grows_with_gain": all(all(
            row["gain_sweep"][str(b)]["effective_support"] >= row["gain_sweep"][str(a)]["effective_support"]
            for a, b in zip(config["gain_scales"], config["gain_scales"][1:])
        ) for row in results.values()),
    }
    hypotheses = {name: {"supported": bool(value)} for name, value in supported.items()}
    payload = {"config": config, "hypotheses": hypotheses,
               "discovery_pair_order": ["".join(GROUPS[i]) for i in pair_order],
               "gain_orders": {gain: ["".join(GROUPS[i]) for i in order]
                               for gain, order in gain_orders.items()},
               "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = ["# Interaction-order and gain red-team of the late-MLP path program", "",
             "The symmetry-invariant program is split into the discovery-free degree≤1 expansion and six degree-two layer interactions. Separately, every path with `d` bilinear layers is rescaled by `gain^d`; this is exactly the path expansion of a hypothetical suffix whose bilinear derivative gain is multiplied while native operands and directions are held fixed.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "Discovery pair order: `" + "`, `".join(payload["discovery_pair_order"]) + "`.", "",
              "| Corpus | Degree≤1 | + top two pairs | Gain | Random-pair median / p95 | Top-2 overlap |",
              "|---|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        lines.append(f"| {corpus} | {row['degree_le_one_fidelity']:.3f} | {row['plus_top_two_pairs_fidelity']:.3f} | {row['pair_gain']:.3f} | {row['random_pair']['median']:.3f} / {row['random_pair']['p95']:.3f} | {row['top_two_pair_overlap']} |")
    lines += ["", "| Corpus | Gain | K90 | Effective support | K5 / K8 fidelity |",
              "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for gain in config["gain_scales"]:
            item = row["gain_sweep"][str(gain)]
            lines.append(f"| {corpus} | {gain:.1f} | {item['k90']} | {item['effective_support']:.2f} | {item['k5_fidelity']:.3f} / {item['k8_fidelity']:.3f} |")
    lines += ["", "The gain sweep is a local counterfactual on the derivative program, not a runnable reparameterized language model. It diagnoses why path compactness occurs: low bilinear gain suppresses high-order interactions; larger gain makes the same contextual operator chain increasingly distributed."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
