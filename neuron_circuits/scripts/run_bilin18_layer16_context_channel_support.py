#!/usr/bin/env python3
"""Distinguish per-context channel density from a broad union of sparse coalitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from run_bilin18_layer16_ce_aware_mode_selection import score
from run_bilin18_layer16_upstream_tangent_channels import build_input_axes, collect
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer16_context_channel_support.json")
OUTPUT = TENSOR_ROOT / "bilin18_layer16_context_channel_support.json"
REPORT = TENSOR_ROOT / "BILIN18_LAYER16_CONTEXT_CHANNEL_SUPPORT.md"


def quantiles(values):
    array = np.asarray(values, dtype=float)
    return {"p10": float(np.quantile(array, .10)), "median": float(np.median(array)),
            "p90": float(np.quantile(array, .90)), "mean": float(np.mean(array))}


@torch.inference_mode()
def fixed_subset_metrics(contribution, target, budgets):
    dot = (contribution.double() * target[:, None].double()).sum(-1)
    energy = contribution.double().square().sum(-1)
    ordering = torch.argsort(2 * dot - energy, dim=1, descending=True)
    target_energy = target.double().square().sum(-1).clamp_min(1e-30)
    rows = {}; site_fidelities = {}
    for budget in budgets:
        index = ordering[:, :budget, None].expand(-1, -1, target.shape[1])
        prediction = contribution.gather(1, index).sum(1)
        site = 1 - (prediction.double() - target.double()).square().sum(-1) / target_energy
        rows[str(budget)] = {"aggregate_fidelity": score(target, prediction),
                             "site_fidelity": quantiles(site.cpu().tolist())}
        site_fidelities[budget] = site
    k90 = []
    for site in range(len(target)):
        reached = next((budget for budget in budgets if site_fidelities[budget][site] >= .90), budgets[-1])
        k90.append(reached)
    return rows, k90


@torch.inference_mode(False)
def omp_support(contribution, target, maximum=32):
    k90 = []; k99 = []
    for site in range(len(target)):
        dictionary = contribution[site].float().cuda()
        desired = target[site].float().cuda()
        norms = dictionary.norm(dim=1).clamp_min(1e-12)
        selected = []; available = torch.ones(len(dictionary), dtype=torch.bool, device="cuda")
        prediction = torch.zeros_like(desired); reached90 = reached99 = maximum + 1
        denominator = desired.square().sum().clamp_min(1e-30)
        for count in range(1, maximum + 1):
            residual = desired - prediction
            correlation = (dictionary @ residual).abs() / norms
            correlation[~available] = -torch.inf
            index = int(torch.argmax(correlation)); selected.append(index); available[index] = False
            design = dictionary[selected].T
            coefficients = torch.linalg.lstsq(design, desired).solution
            prediction = design @ coefficients
            fidelity = 1 - float((desired - prediction).square().sum() / denominator)
            if fidelity >= .90 and reached90 > maximum: reached90 = count
            if fidelity >= .99 and reached99 > maximum: reached99 = count
        k90.append(reached90); k99.append(reached99)
    return k90, k99


@torch.inference_mode(False)
def main():
    config = json.loads(CONFIG.read_text()); model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    axes = build_input_axes(model, config); results = {}; maximum_error = 0.0
    for corpus, spec in config["evaluation"].items():
        print("evaluate", corpus, flush=True)
        data = collect(model, spec, axes, config)
        contribution = data["contribution"].double(); target = data["target"].double()
        maximum_error = max(maximum_error, data["maximum_sum_error"])
        magnitudes = contribution.square().sum(-1).sqrt()
        norm_support = magnitudes.sum(1).square() / magnitudes.square().sum(1).clamp_min(1e-30)
        target_energy = target.square().sum(-1).clamp_min(1e-30)
        projection = (contribution * target[:, None]).sum(-1) / target_energy[:, None]
        absolute_mass = projection.abs().sum(1)
        projection_support = absolute_mass.square() / projection.square().sum(1).clamp_min(1e-30)
        fixed, fixed_k90 = fixed_subset_metrics(data["contribution"], data["target"], config["budgets"])
        omp_k90, omp_k99 = omp_support(data["contribution"], data["target"])
        results[corpus] = {
            "sites": len(target),
            "norm_effective_support": quantiles(norm_support.cpu().tolist()),
            "target_projection_effective_support": quantiles(projection_support.cpu().tolist()),
            "absolute_projection_mass": quantiles(absolute_mass.cpu().tolist()),
            "fixed_subset": fixed,
            "fixed_k90": quantiles(fixed_k90),
            "fraction_fixed_k90_at_most_1024": float(np.mean(np.asarray(fixed_k90) <= 1024)),
            "reweighted_omp_k90": quantiles(omp_k90),
            "reweighted_omp_k99": quantiles(omp_k99),
        }
    fineweb_support = results["fineweb"]["norm_effective_support"]["median"]
    hypotheses = {
        "S0_context_channel_sum_is_exact": maximum_error <= 1e-5,
        "S1_contexts_have_sparse_norm_participation": all(row["norm_effective_support"]["median"] <= 512 for row in results.values()),
        "S2_contexts_have_sparse_target_projection": all(row["target_projection_effective_support"]["median"] <= 512 for row in results.values()),
        "S3_fixed_coefficient_context512_is_sufficient": all(row["fixed_subset"]["512"]["aggregate_fidelity"] >= .90 for row in results.values()),
        "S4_most_contexts_have_small_fixed_subsets": all(row["fraction_fixed_k90_at_most_1024"] >= .90 for row in results.values()),
        "S5_contexts_use_substantial_cancellation": all(row["absolute_projection_mass"]["median"] >= 2.0 for row in results.values()),
        "S6_reweighted_span_is_compact": all(row["reweighted_omp_k90"]["median"] <= 16 for row in results.values()),
        "S7_participation_support_is_ood_stable": all(abs(results[name]["norm_effective_support"]["median"] / fineweb_support - 1) <= .20 for name in ("wikitext", "tinystories")),
    }
    hypothesis_rows = {name: {"supported": bool(value)} for name, value in hypotheses.items()}
    hypothesis_rows["S0_context_channel_sum_is_exact"]["maximum_relative_error"] = maximum_error
    payload = {"config": config, "hypotheses": hypothesis_rows, "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = ["# Per-context native support and cancellation on natural upstream tangents", "",
             "This asks whether the broad fixed channel bank is merely the union of sparse context-specific coalitions. Effective support is computed from contribution-vector norms and from absolute target projections. Fixed subsets keep the model's native coefficient of one. OMP is reported separately and may freely reweight selected channels, so it measures descriptive span rather than native execution.", "", "## Locked hypotheses", ""]
    for name, row in hypothesis_rows.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Maximum channel-sum error: `{maximum_error:.2e}`.", "",
              "| Corpus | Norm support p10 / med / p90 | Projection support med | Abs projection mass med | Fixed512 agg | Fixed K90 med | OMP K90 / K99 med |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        support = row["norm_effective_support"]
        lines.append(f"| {corpus} | {support['p10']:.0f} / {support['median']:.0f} / {support['p90']:.0f} | {row['target_projection_effective_support']['median']:.0f} | {row['absolute_projection_mass']['median']:.2f} | {row['fixed_subset']['512']['aggregate_fidelity']:.3f} | {row['fixed_k90']['median']:.0f} | {row['reweighted_omp_k90']['median']:.0f} / {row['reweighted_omp_k99']['median']:.0f} |")
    lines += ["", "A small reweighted OMP span does not imply sparse native execution: in a 32D target space, arbitrary rescaling can synthesize a target from at most 32 independent vectors. The fixed-coefficient and participation statistics answer whether the model's actual channel contributions are sparse within a context."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__": main()
