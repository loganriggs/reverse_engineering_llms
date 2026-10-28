#!/usr/bin/env python3
"""Quotient the late I/L/R path program by exact bilinear-factor symmetry."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_mlp_path_type_program import (
    PATHS, aggregate_gram, build_axes, collect_paths, error_from_gram,
    greedy_order, oracle_fidelity, path_fidelity,
)
from run_bilin12_router_finite_transport import fidelity
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_mlp_path_symmetry_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_mlp_path_symmetry_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_MLP_PATH_SYMMETRY_REDTEAM.md"
GROUPS = tuple(itertools.product(("I", "B"), repeat=4))


def group_index(path):
    return GROUPS.index(tuple("I" if term == "I" else "B" for term in path))


def collapse(paths):
    result = torch.zeros(paths.shape[0], len(GROUPS), *paths.shape[2:], dtype=paths.dtype)
    for index, path in enumerate(PATHS):
        result[:, group_index(path)] += paths[:, index]
    return result


def random_summary(paths, budget, replicates, generator, degree_profile=None):
    values = []
    degrees = [group.count("B") for group in GROUPS]
    for _ in range(replicates):
        if degree_profile is None:
            selected = torch.randperm(len(GROUPS), generator=generator)[:budget].tolist()
        else:
            selected = []
            for degree, count in degree_profile.items():
                pool = torch.as_tensor([i for i, value in enumerate(degrees) if value == degree])
                choice = pool[torch.randperm(len(pool), generator=generator)[:count]]
                selected.extend(choice.tolist())
        values.append(path_fidelity(paths, selected))
    return {"median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "maximum": max(values)}


@torch.inference_mode(False)
def main():
    redteam = json.loads(CONFIG.read_text())
    base = json.loads(Path(redteam["source_config"]).read_text())
    model = load_snapshot_model(Path(base["snapshot"]), "cuda")
    axes = build_axes(model, base)
    print("collect discovery", flush=True)
    raw_discovery, _ = collect_paths(model, axes, base["path_discovery"], base)
    discovery = collapse(raw_discovery)
    gram, dot, target2 = aggregate_gram(discovery)
    ordering = greedy_order(gram, dot, target2)
    generator = torch.Generator().manual_seed(redteam["seed"])
    degree_one = [i for i, group in enumerate(GROUPS) if group.count("B") <= 1]
    results = {}; maximum_error = 0.0
    for corpus, spec in base["evaluation"].items():
        print("evaluate", corpus, flush=True)
        raw, finite = collect_paths(model, axes, spec, base, include_finite=True)
        paths = collapse(raw)
        original = raw.sum(1); quotient = paths.sum(1)
        relative = ((original - quotient).flatten(1).norm(dim=1)
                    / original.flatten(1).norm(dim=1).clamp_min(1e-30))
        maximum_error = max(maximum_error, float(relative.max()))
        rows = {}
        for budget in redteam["budgets"]:
            selected = ordering[:budget]
            profile = {}
            for index in selected:
                degree = GROUPS[index].count("B")
                profile[degree] = profile.get(degree, 0) + 1
            tangent = paths[:, selected].sum(1)
            rows[str(budget)] = {
                "frozen_fidelity": path_fidelity(paths, selected),
                "oracle_fidelity": oracle_fidelity(paths, budget),
                "uniform_random": random_summary(paths, budget, redteam["random_replicates"], generator),
                "degree_matched_random": random_summary(paths, budget, redteam["random_replicates"], generator, profile),
                "degree_profile": profile,
                "finite_mlp_path_fidelity": fidelity(finite.reshape_as(tangent), tangent),
            }
        results[corpus] = {"sites": len(paths),
                           "degree_le_one_fidelity": path_fidelity(paths, degree_one),
                           "budgets": rows}

    fineweb4 = results["fineweb"]["budgets"]["4"]["frozen_fidelity"]
    supported = {
        "R0_symmetry_quotient_is_exact": maximum_error <= 1e-6,
        "R1_frozen_four_group_program_is_compact": all(row["budgets"]["4"]["frozen_fidelity"] >= .85 for row in results.values()),
        "R2_frozen_eight_group_program_is_high_fidelity": all(row["budgets"]["8"]["frozen_fidelity"] >= .95 for row in results.values()),
        "R3_group_program_beats_uniform_random": all(row["budgets"]["4"]["frozen_fidelity"] >= row["budgets"]["4"]["uniform_random"]["p95"] + .05 for row in results.values()),
        "R4_group_program_beats_degree_matched_random": all(row["budgets"]["4"]["frozen_fidelity"] >= row["budgets"]["4"]["degree_matched_random"]["p95"] + .03 for row in results.values()),
        "R5_degree_one_truncation_explains_compactness": all(row["degree_le_one_fidelity"] >= .85 for row in results.values()),
        "R6_context_specific_group_selection_is_small": all(row["budgets"]["4"]["oracle_fidelity"] < row["budgets"]["4"]["frozen_fidelity"] + .05 for row in results.values()),
        "R7_group_program_transfers_ood": all(results[name]["budgets"]["4"]["frozen_fidelity"] >= fineweb4 - .05 for name in ("wikitext", "tinystories")),
        "R8_group_program_predicts_finite_effects": all(row["budgets"]["8"]["finite_mlp_path_fidelity"] >= .90 for row in results.values()),
    }
    hypotheses = {name: {"supported": bool(value)} for name, value in supported.items()}
    hypotheses["R0_symmetry_quotient_is_exact"]["maximum_relative_error"] = maximum_error
    payload = {"config": redteam, "hypotheses": hypotheses,
               "group_order": ["".join(GROUPS[index]) for index in ordering],
               "degree_le_one_groups": ["".join(GROUPS[index]) for index in degree_one],
               "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))

    lines = ["# Symmetry red-team of the late-MLP path program", "",
             "`Left` and `Right` are exchangeable factors of each bilinear neuron, so an ontology that treats them as distinct is not canonical. Summing `L+R` within every active layer quotients that exact symmetry and reduces 81 paths to 16 invariant layer-subset paths: residual `I` or full bilinear derivative `B` at each of Layers 7–10.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum quotient recomposition error: `{maximum_error:.2e}`.", "",
              "Frozen group order: `" + "`, `".join(payload["group_order"]) + "`.", "",
              "| Corpus | K | Frozen | Oracle | Uniform p95 | Degree-matched p95 | Finite effect |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for budget in redteam["budgets"]:
            item = row["budgets"][str(budget)]
            lines.append(f"| {corpus} | {budget} | {item['frozen_fidelity']:.3f} | {item['oracle_fidelity']:.3f} | {item['uniform_random']['p95']:.3f} | {item['degree_matched_random']['p95']:.3f} | {item['finite_mlp_path_fidelity']:.3f} |")
        lines.append(f"| {corpus} | degree≤1 (5) | {row['degree_le_one_fidelity']:.3f} | — | — | — | — |")
    lines += ["", "If the discovery-free degree≤1 program matches the frozen program, compactness mostly reflects a low-order residual expansion. If particular layer subsets beat degree-matched controls, the ordering contains additional learned structure. Either way, I/B groups are more canonical than individual L/R arms."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
