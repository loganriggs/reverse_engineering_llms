#!/usr/bin/env python3
"""Measure behavior versus continuous Layer-17 tail-program rank."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_continuous_fisher import (
    continuous_prediction,
    evaluate,
    optimize_subspace,
)
from run_bilin18_layer17_tail_fisher_basis import collect_fisher, fisher_quadratic
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_continuous_rank_frontier.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_continuous_rank_frontier.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_rank_frontier.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_CONTINUOUS_RANK_FRONTIER.md"
FIGURE = TENSOR_ROOT / "layer17_tail_continuous_rank_frontier.png"


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE_ARRAYS)
    output_basis = torch.from_numpy(source["tail_output_basis"][:, :4]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval(); model.blocks.float().eval(); model.lm_head.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float().detach()
    left = block.mlp.Left.weight.float().detach()
    right = block.mlp.Right.weight.float().detach()

    print("constructing frozen rank-sweep problem", flush=True)
    discovery_x, discovery_write = collect(model, config["layer"], config["discovery"], config)
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
    z = discovery_x @ inverse
    coordinates = (z @ candidate_basis).detach()
    isotropic = (z.square().sum(1, keepdim=True) * traces_tensor[None]).detach()
    target = (discovery_write @ output_basis - isotropic).detach()
    metric = collect_fisher(model, config["discovery"], config, output_basis).detach()
    row_ids = torch.arange(len(discovery_x), device="cuda") // len(config["positions"])
    even = torch.nonzero((row_ids % 2) == 0, as_tuple=False).flatten()
    odd = torch.nonzero((row_ids % 2) == 1, as_tuple=False).flatten()

    quotients, histories, opposite_half_fisher = {}, {}, {}
    for rank in config["ranks"]:
        print("optimizing rank", rank, flush=True)
        rank_config = {**config, "rank": rank}
        quotient, info = optimize_subspace(
            coordinates, forms, target, metric, even, odd, rank_config,
            config["seed"] + rank, initialization="frobenius_prefix",
        )
        name = f"rank_{rank}"
        quotients[name] = quotient
        histories[name] = info
        prediction = continuous_prediction(coordinates[odd], forms, quotient)
        opposite_half_fisher[name] = float(fisher_quadratic(
            target[odd] - prediction, metric[odd]
        ).mean())

    results = {}
    for corpus, spec in config["evaluation"].items():
        print("external evaluation", corpus, flush=True)
        results[corpus] = evaluate(
            model, spec, config, output_basis, inverse, candidate_basis,
            forms, traces_tensor, quotients,
        )

    smallest_universal_gate = None
    for rank in config["ranks"]:
        name = f"rank_{rank}"
        if all(row["behavioral"][name]["median_exact_kl"] <= 0.005 for row in results.values()):
            smallest_universal_gate = rank
            break
    fisher_values = [opposite_half_fisher[f"rank_{rank}"] for rank in config["ranks"]]
    r1 = all(right <= 1.02 * left for left, right in zip(fisher_values, fisher_values[1:]))
    r2 = all(row["behavioral"]["rank_16"]["median_exact_kl"] <= 0.005 for row in results.values())
    r3 = all(
        row["behavioral"]["rank_16"]["mean_exact_kl"]
        <= 0.80 * row["behavioral"]["rank_8"]["mean_exact_kl"]
        for row in results.values()
    )
    r4 = all(row["behavioral"]["rank_16"]["ce_delta"] <= 0.01 for row in results.values())
    r5 = all(
        row["behavioral"]["rank_32"]["median_exact_kl"] <= 0.002
        and row["behavioral"]["rank_32"]["top1_agreement"] >= 0.97
        for row in results.values()
    )
    r6 = smallest_universal_gate is not None and smallest_universal_gate <= 16
    hypotheses = {
        "R1_fisher_error_decreases_with_rank": {"supported": r1},
        "R2_rank16_reaches_median_kl_gate": {"supported": r2},
        "R3_rank16_improves_rank8_mean_kl": {"supported": r3},
        "R4_rank16_preserves_task_ce": {"supported": r4},
        "R5_rank32_is_high_fidelity": {"supported": r5},
        "R6_smallest_universal_gate_is_at_most_16": {"supported": r6, "smallest_universal_gate": smallest_universal_gate},
    }
    OUTPUT.write_text(json.dumps({
        "config": config, "hypotheses": hypotheses, "covariance": covariance,
        "opposite_half_fisher": opposite_half_fisher, "histories": histories,
        "smallest_universal_gate": smallest_universal_gate, "results": results,
    }, indent=2, allow_nan=False))
    np.savez_compressed(ARRAYS, candidate_basis=candidate_basis.cpu().numpy(), **{
        name: quotient.cpu().numpy() for name, quotient in quotients.items()
    })

    lines = [
        "# Continuous Layer-17 behavioral rank frontier", "",
        "A fixed 128D tensor-derived input pool is compressed to continuously optimized subspaces from rank 2 through 32. Fits use even FineWeb discovery rows, validate on odd rows, and are evaluated with exact final-logit interventions on disjoint FineWeb, WikiText, and TinyStories.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", f"Smallest tested universal median-KL gate: **{smallest_universal_gate}**.", "",
              "![Rank frontier](layer17_tail_continuous_rank_frontier.png)", "",
              "| Rank | Opposite-half Fisher | " + " | ".join(f"{c} median KL / mean KL / CE Δ / top1" for c in results) + " |",
              "|---:|---:|" + "---:|" * len(results)]
    for rank in config["ranks"]:
        name = f"rank_{rank}"
        cells = []
        for corpus, row in results.items():
            behavior = row["behavioral"][name]
            cells.append(
                f"{behavior['median_exact_kl']:.5f} / {behavior['mean_exact_kl']:.5f} / "
                f"{behavior['ce_delta']:.5f} / {behavior['top1_agreement']:.3f}"
            )
        lines.append(f"| {rank} | {opposite_half_fisher[name]:.4e} | " + " | ".join(cells) + " |")
    lines += ["", "KL measures imitation of the native model; CE delta measures task loss against the observed next token. They answer different questions and are reported together."]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, plots = plt.subplots(1, 3, figsize=(15, 4.5))
    ranks = config["ranks"]
    plots[0].plot(ranks, fisher_values, marker="o")
    plots[0].set_title("Held-out local behavior"); plots[0].set_ylabel("Fisher error")
    for corpus, row in results.items():
        plots[1].plot(ranks, [row["behavioral"][f"rank_{rank}"]["median_exact_kl"] for rank in ranks], marker="o", label=corpus)
        plots[2].plot(ranks, [row["behavioral"][f"rank_{rank}"]["ce_delta"] for rank in ranks], marker="o", label=corpus)
    plots[1].axhline(0.005, color="black", linestyle="--", linewidth=1)
    plots[1].set_title("Exact distribution fidelity"); plots[1].set_ylabel("median KL")
    plots[2].axhline(0.01, color="black", linestyle="--", linewidth=1)
    plots[2].set_title("Observed-token utility"); plots[2].set_ylabel("mean CE delta")
    for plot in plots:
        plot.set_xlabel("continuous subspace rank"); plot.grid(alpha=0.2)
        if plot.get_legend_handles_labels()[0]:
            plot.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(FIGURE, dpi=180); plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
