#!/usr/bin/env python3
"""Test a union of variance- and Fisher-preserving Layer-17 tail subspaces."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_continuous_fisher import evaluate
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_dual_metric_union.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
FISHER_ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_fisher.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_dual_metric_union.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_dual_metric_union.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_DUAL_METRIC_UNION.md"
FIGURE = TENSOR_ROOT / "layer17_tail_dual_metric_union.png"


def union(*subspaces: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(torch.cat(subspaces, dim=1), mode="reduced").Q


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE_ARRAYS)
    fisher = np.load(FISHER_ARRAYS)
    output_basis = torch.from_numpy(source["tail_output_basis"][:, :4]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float().detach()
    left = block.mlp.Left.weight.float().detach()
    right = block.mlp.Right.weight.float().detach()

    discovery_x, _ = collect(model, config["layer"], config["discovery"], config)
    square, inverse, covariance = covariance_transforms(discovery_x, config["covariance_floor_fraction"])
    cleaned_forms, traces = [], []
    for component in range(4):
        coefficients = down.T @ output_basis[:, component]
        raw = left.T @ (coefficients[:, None] * right)
        raw = 0.5 * (raw + raw.T)
        clean, trace = trace_clean(square @ raw @ square)
        cleaned_forms.append(clean); traces.append(trace)
    traces_tensor = torch.stack(traces)
    _, common_basis = common_input(cleaned_forms)
    candidate_basis = common_basis[:, :config["candidate_pool_rank"]]
    forms = torch.einsum("di,kde,ej->kij", candidate_basis, torch.stack(cleaned_forms), candidate_basis).detach()

    eye = torch.eye(config["candidate_pool_rank"], device="cuda")
    fisher8 = torch.from_numpy(fisher["continuous_a"]).cuda().float()
    frobenius8, frobenius16, frobenius32 = eye[:, :8], eye[:, :16], eye[:, :32]
    quotients = {
        "fisher8": fisher8,
        "frobenius8": frobenius8,
        "frobenius16": frobenius16,
        "frobenius32": frobenius32,
        "union16": union(frobenius8, fisher8),
        "union24": union(frobenius16, fisher8),
        "union40": union(frobenius32, fisher8),
    }
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("external evaluation", corpus, flush=True)
        results[corpus] = evaluate(
            model, spec, config, output_basis, inverse, candidate_basis,
            forms, traces_tensor, quotients,
        )

    u1 = all(
        row["behavioral"]["union16"]["mean_exact_kl"]
        <= 0.80 * row["behavioral"]["fisher8"]["mean_exact_kl"]
        for row in results.values()
    )
    u2 = all(row["behavioral"]["union16"]["median_exact_kl"] <= 0.005 for row in results.values())
    u3 = all(row["scores"]["union16"]["centered_vector_r2"] >= 0.90 for row in results.values())
    u4 = all(row["behavioral"]["union16"]["ce_delta"] <= 0.01 for row in results.values())
    u5 = all(row["behavioral"]["union16"]["top1_agreement"] >= 0.95 for row in results.values())
    u6 = all(
        row["behavioral"]["union40"]["mean_exact_kl"]
        >= 0.80 * row["behavioral"]["union16"]["mean_exact_kl"]
        for row in results.values()
    )
    hypotheses = {
        "U1_union16_improves_fisher8_mean_kl": {"supported": u1},
        "U2_union16_reaches_median_kl_gate": {"supported": u2},
        "U3_union16_preserves_native_scores": {"supported": u3},
        "U4_union16_preserves_task_ce": {"supported": u4},
        "U5_union16_reaches_top1_gate": {"supported": u5},
        "U6_larger_union_has_diminishing_returns": {"supported": u6},
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses, "covariance": covariance,
        "results": results,
    }, indent=2, allow_nan=False))
    np.savez_compressed(ARRAYS, candidate_basis=candidate_basis.cpu().numpy(), **{
        name: quotient.cpu().numpy() for name, quotient in quotients.items()
    })

    lines = [
        "# Dual-metric union for the Layer-17 tail program", "",
        "This test asks whether the high-variance Frobenius core and the downstream-sensitive Fisher core are complementary. Their span is evaluated without refitting by exact final-logit interventions on three corpora.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Dual-metric union](layer17_tail_dual_metric_union.png)", "",
              "| Corpus | Program | Rank | Score R² | Mean KL | Median KL | CE Δ | Top1 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        for name, quotient in quotients.items():
            score, behavior = row["scores"][name], row["behavioral"][name]
            lines.append(
                f"| {corpus} | {name} | {quotient.shape[1]} | {score['centered_vector_r2']:.3f} | "
                f"{behavior['mean_exact_kl']:.5f} | {behavior['median_exact_kl']:.5f} | "
                f"{behavior['ce_delta']:.5f} | {behavior['top1_agreement']:.3f} |"
            )
    lines += ["", "If the union wins, variance reconstruction and behavioral preservation are distinct but compatible explanatory objects. If it does not, the remaining error is not merely the union of those two fixed global subspaces."]
    REPORT.write_text("\n".join(lines) + "\n")

    programs = list(quotients)
    figure, plots = plt.subplots(1, 3, figsize=(15, 4.5))
    for corpus, row in results.items():
        plots[0].plot(programs, [row["scores"][n]["centered_vector_r2"] for n in programs], marker="o", label=corpus)
        plots[1].plot(programs, [row["behavioral"][n]["mean_exact_kl"] for n in programs], marker="o", label=corpus)
        plots[2].plot(programs, [row["behavioral"][n]["ce_delta"] for n in programs], marker="o", label=corpus)
    for plot, title, ylabel in zip(plots, ("Native score", "Distribution", "Observed token"), ("centered R²", "mean exact KL", "mean CE delta")):
        plot.set_title(title); plot.set_ylabel(ylabel); plot.tick_params(axis="x", rotation=35); plot.grid(alpha=0.2); plot.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
