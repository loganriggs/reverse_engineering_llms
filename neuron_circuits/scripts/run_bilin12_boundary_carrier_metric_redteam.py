#!/usr/bin/env python3
"""Red-team carrier attribution across metrics and all hybrid insertion contexts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from bilin_checkpoint_utils import load_snapshot_model
from run_bilin12_boundary_carrier_depth_audit import (
    FACTORS,
    exact_kl,
    fisher_terms,
    shapley_effects,
    suffix_worlds,
    trace,
)
from run_bilin12_late_address_program_audit import load_rows
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_boundary_carrier_metric_redteam.json")
OUTPUT = TENSOR_ROOT / "bilin12_boundary_carrier_metric_redteam.json"
REPORT = TENSOR_ROOT / "BILIN12_BOUNDARY_CARRIER_METRIC_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_boundary_carrier_metric_redteam.png"
METRICS = ("baseline_fisher", "changed_fisher", "forward_kl", "target_error_kl")


def scalar_shapley(world_values):
    result = {}
    count = len(FACTORS)
    for factor_index, factor in enumerate(FACTORS):
        bit = 1 << factor_index
        value = 0.0
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            value += weight * (world_values[mask | bit] - world_values[mask])
        result[factor] = float(value)
    return result


def empty_accumulator():
    return {
        "examples": 0,
        "baseline_energy": 0.0,
        "changed_energy": 0.0,
        "baseline_shapley_cross": {factor: 0.0 for factor in FACTORS},
        "changed_shapley_cross": {factor: 0.0 for factor in FACTORS},
        "marginal_cross": {
            factor: {str(mask): 0.0 for mask in range(8) if not mask & (1 << index)}
            for index, factor in enumerate(FACTORS)
        },
        "forward_world_kl": [0.0] * 8,
        "target_world_kl": [0.0] * 8,
    }


def update(acc, worlds):
    base, changed = worlds[0], worlds[7]
    target = changed - base
    baseline_energy, _, _, _ = fisher_terms(base, target, target)
    changed_energy, _, _, _ = fisher_terms(changed, target, target)
    acc["examples"] += len(base)
    acc["baseline_energy"] += float(baseline_energy.sum())
    acc["changed_energy"] += float(changed_energy.sum())
    effects = shapley_effects(worlds)
    for factor, effect in effects.items():
        _, _, baseline_cross, _ = fisher_terms(base, target, effect)
        _, _, changed_cross, _ = fisher_terms(changed, target, effect)
        acc["baseline_shapley_cross"][factor] += float(baseline_cross.sum())
        acc["changed_shapley_cross"][factor] += float(changed_cross.sum())
    for factor_index, factor in enumerate(FACTORS):
        bit = 1 << factor_index
        for mask in range(8):
            if mask & bit:
                continue
            marginal = worlds[mask | bit] - worlds[mask]
            _, _, cross, _ = fisher_terms(base, target, marginal)
            acc["marginal_cross"][factor][str(mask)] += float(cross.sum())
    for mask in range(8):
        acc["forward_world_kl"][mask] += float(exact_kl(base, worlds[mask]).sum())
        acc["target_world_kl"][mask] += float(exact_kl(changed, worlds[mask]).sum())


def finalize(acc):
    baseline_energy = max(acc["baseline_energy"], 1e-30)
    changed_energy = max(acc["changed_energy"], 1e-30)
    baseline = {factor: acc["baseline_shapley_cross"][factor] / baseline_energy for factor in FACTORS}
    changed = {factor: acc["changed_shapley_cross"][factor] / changed_energy for factor in FACTORS}
    forward_denominator = max(acc["forward_world_kl"][7], 1e-30)
    target_denominator = max(acc["target_world_kl"][0], 1e-30)
    forward_game = [value / forward_denominator for value in acc["forward_world_kl"]]
    target_game = [1 - value / target_denominator for value in acc["target_world_kl"]]
    forward = scalar_shapley(forward_game)
    target_error = scalar_shapley(target_game)
    marginal = {
        factor: {
            mask: value / baseline_energy for mask, value in rows.items()
        } for factor, rows in acc["marginal_cross"].items()
    }
    ranges = {
        factor: max(rows.values()) - min(rows.values()) for factor, rows in marginal.items()
    }
    signs = {
        factor: all(value >= 0 for value in rows.values()) for factor, rows in marginal.items()
    }
    return {
        "examples": acc["examples"],
        "allocations": {
            "baseline_fisher": baseline,
            "changed_fisher": changed,
            "forward_kl": forward,
            "target_error_kl": target_error,
        },
        "allocation_sums": {
            "baseline_fisher": sum(baseline.values()),
            "changed_fisher": sum(changed.values()),
            "forward_kl": sum(forward.values()),
            "target_error_kl": sum(target_error.values()),
        },
        "baseline_fisher_marginals": marginal,
        "marginal_ranges": ranges,
        "marginal_nonnegative": signs,
    }


@torch.inference_mode()
def evaluate(model, tokenizer, spec, config):
    rows = load_rows(spec)
    newline = tokenizer.encode("\n", add_special_tokens=False)[0]
    accumulators = [empty_accumulator() for _ in model.blocks]
    for start in range(0, len(rows), config["batch_size"]):
        raw = np.asarray(rows[start:start + config["batch_size"], :config["sequence_length"]], dtype=np.int64)
        base_ids = torch.as_tensor(raw, device="cuda")
        keep = base_ids[:, config["position"]] != newline
        if not bool(keep.any()):
            continue
        base_ids = base_ids[keep]
        changed_ids = base_ids.clone(); changed_ids[:, config["position"]] = newline
        base_boundaries, _ = trace(model, base_ids)
        changed_boundaries, _ = trace(model, changed_ids)
        for layer in range(len(model.blocks)):
            update(accumulators[layer], suffix_worlds(model, layer, base_boundaries[layer], changed_boundaries[layer]))
    return {str(layer): finalize(acc) for layer, acc in enumerate(accumulators)}


def main():
    config = json.loads(CONFIG.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("evaluating", corpus, flush=True)
        results[corpus] = evaluate(model, tokenizer, spec, config)

    rows = [results[c][str(layer)] for c in results for layer in range(12)]
    hypotheses = {
        "R1_marginal_signs_are_context_stable": {"supported": all(
            all(row["marginal_nonnegative"].values()) for row in rows
        )},
        "R2_fisher_reference_is_not_decisive": {"supported": all(
            abs(row["allocations"]["baseline_fisher"][factor] - row["allocations"]["changed_fisher"][factor]) <= .10
            for row in rows for factor in FACTORS
        )},
        "R3_embedding_is_material_under_kl_games": {"supported": all(
            results[c]["0"]["allocations"][metric]["embedding_stream"] >= .25
            for c in results for metric in ("forward_kl", "target_error_kl")
        )},
        "R4_value_bus_remains_a_minor_route": {"supported": all(
            abs(row["allocations"][metric]["value_bus"]) <= .15
            for row in rows for metric in METRICS
        )},
        "R5_layer10_residual_dominates_all_metrics": {"supported": all(
            results[c]["10"]["allocations"][metric]["residual_sequence"] >= .90
            for c in results for metric in METRICS
        )},
        "R6_kl_direction_is_not_decisive": {"supported": all(
            abs(row["allocations"]["forward_kl"][factor] - row["allocations"]["target_error_kl"][factor]) <= .15
            for row in rows for factor in FACTORS
        )},
        "R7_hybrid_order_dependence_is_small": {"supported": all(
            value <= .25 for row in rows for value in row["marginal_ranges"].values()
        )},
        "R8_fresh_data_replicates_progressive_compilation": {"supported": all(
            all(
                results[c][str(layer + 1)]["allocations"]["baseline_fisher"]["residual_sequence"]
                >= results[c][str(layer)]["allocations"]["baseline_fisher"]["residual_sequence"] - .03
                for layer in range(11)
            ) for c in results
        )},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "results": results}, indent=2, allow_nan=False))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.7))
    layers = np.arange(12)
    colors = {"residual_sequence": "#4c78a8", "embedding_stream": "#f58518", "value_bus": "#54a24b"}
    for metric, linestyle in (("baseline_fisher", "-"), ("changed_fisher", "--")):
        for factor in FACTORS:
            values = np.asarray([[results[c][str(l)]["allocations"][metric][factor] for l in layers] for c in results])
            axes[0].plot(layers, values.mean(0), color=colors[factor], ls=linestyle, marker="o" if metric == "baseline_fisher" else None, label=f"{factor}/{metric}")
    axes[0].set_title("Fisher-reference red team")
    for metric, linestyle in (("forward_kl", "-"), ("target_error_kl", "--")):
        for factor in FACTORS:
            values = np.asarray([[results[c][str(l)]["allocations"][metric][factor] for l in layers] for c in results])
            axes[1].plot(layers, values.mean(0), color=colors[factor], ls=linestyle, marker="o" if metric == "forward_kl" else None, label=f"{factor}/{metric}")
    axes[1].set_title("Exact-KL-game red team")
    for factor in FACTORS:
        values = np.asarray([[results[c][str(l)]["marginal_ranges"][factor] for l in layers] for c in results])
        axes[2].plot(layers, values.mean(0), color=colors[factor], marker="o", label=factor)
    axes[2].axhline(.25, color="black", ls="--", lw=.8); axes[2].set_title("Hybrid-context marginal range")
    for axis in axes:
        axis.set_xlabel("boundary after layer"); axis.grid(alpha=.2); axis.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)

    lines = [
        "# Carrier metric and hybrid-order red team",
        "",
        "This fresh 48-example-per-corpus panel repeats the exact eight-world carrier decomposition under four definitions of behavioral credit: baseline Fisher projection, changed-distribution Fisher projection, forward KL accumulated from the base model, and reduction of reverse KL error to the changed model. It also reports every carrier's marginal projection in all four contexts formed by the other two carriers.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Metric red team](bilin12_boundary_carrier_metric_redteam.png)",
        "",
        "| Corpus | Layer | Carrier | Base-Fisher | Changed-Fisher | Forward KL | Target-error KL | Marginal range | All signs + |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus in results:
        for layer in (0, 4, 7, 10, 11):
            row = results[corpus][str(layer)]
            for factor in FACTORS:
                lines.append(
                    f"| {corpus} | {layer} | {factor} | "
                    f"{row['allocations']['baseline_fisher'][factor]:.3f} | {row['allocations']['changed_fisher'][factor]:.3f} | "
                    f"{row['allocations']['forward_kl'][factor]:.3f} | {row['allocations']['target_error_kl'][factor]:.3f} | "
                    f"{row['marginal_ranges'][factor]:.3f} | {row['marginal_nonnegative'][factor]} |"
                )
    lines += [
        "",
        "Fisher projection attributes a logit-vector direction, whereas the two KL games attribute nonlinear distributional change. Agreement supports a robust carrier story; disagreement identifies a use-case distinction rather than one metric being universally correct. Hybrid marginals diagnose whether the result depends on inserting a carrier into an inconsistent base/changed context.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
