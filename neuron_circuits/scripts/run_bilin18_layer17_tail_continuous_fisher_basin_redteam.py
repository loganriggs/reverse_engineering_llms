#!/usr/bin/env python3
"""Red-team continuous Layer-17 Fisher subspaces with non-prefix initializations."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bilin18_depth_query_map_replication import collect
from run_bilin18_layer17_tail_cleaning_sweep import covariance_transforms, trace_clean
from run_bilin18_layer17_tail_continuous_fisher import (
    canonical_overlap,
    continuous_prediction,
    evaluate,
    optimize_subspace,
)
from run_bilin18_layer17_tail_fisher_basis import collect_fisher, fisher_quadratic
from run_bilin18_layer17_tail_tensor_canonicalization import common_input
from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_layer17_tail_continuous_fisher_basin_redteam.json")
SOURCE_ARRAYS = TENSOR_ROOT / "layer17_bottleneck_tail_ontology.npz"
PREFIX_ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_fisher.npz"
OUTPUT = TENSOR_ROOT / "layer17_tail_continuous_fisher_basin_redteam.json"
ARRAYS = TENSOR_ROOT / "layer17_tail_continuous_fisher_basin_redteam.npz"
REPORT = TENSOR_ROOT / "LAYER17_TAIL_CONTINUOUS_FISHER_BASIN_REDTEAM.md"
FIGURE = TENSOR_ROOT / "layer17_tail_continuous_fisher_basin_redteam.png"


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE_ARRAYS)
    prefix = np.load(PREFIX_ARRAYS)
    output_basis = torch.from_numpy(source["tail_output_basis"][:, :4]).cuda().float()
    model = load_full_model("cuda")
    model.embedding.float().eval()
    model.blocks.float().eval()
    model.lm_head.float().eval()
    block = model.blocks[config["layer"]]
    down = block.mlp.Down.weight.float().detach()
    left = block.mlp.Left.weight.float().detach()
    right = block.mlp.Right.weight.float().detach()

    print("reconstructing frozen discovery problem", flush=True)
    discovery_x, discovery_write = collect(model, config["layer"], config["discovery"], config)
    square, inverse, covariance = covariance_transforms(
        discovery_x, config["covariance_floor_fraction"]
    )
    cleaned_forms, traces = [], []
    for component in range(4):
        coefficients = down.T @ output_basis[:, component]
        raw = left.T @ (coefficients[:, None] * right)
        raw = 0.5 * (raw + raw.T)
        clean, trace = trace_clean(square @ raw @ square)
        cleaned_forms.append(clean)
        traces.append(trace)
    traces_tensor = torch.stack(traces)
    _, common_basis = common_input(cleaned_forms)
    candidate_basis = common_basis[:, : config["candidate_pool_rank"]]
    forms = torch.einsum(
        "di,kde,ej->kij", candidate_basis, torch.stack(cleaned_forms), candidate_basis
    ).detach()
    z = discovery_x @ inverse
    coordinates = (z @ candidate_basis).detach()
    isotropic = (z.square().sum(1, keepdim=True) * traces_tensor[None]).detach()
    target = (discovery_write @ output_basis - isotropic).detach()
    metric = collect_fisher(model, config["discovery"], config, output_basis).detach()
    row_ids = torch.arange(len(discovery_x), device="cuda") // len(config["positions"])
    halves = {
        "even": torch.nonzero((row_ids % 2) == 0, as_tuple=False).flatten(),
        "odd": torch.nonzero((row_ids % 2) == 1, as_tuple=False).flatten(),
    }

    quotients = {
        "prefix_a": torch.from_numpy(prefix["continuous_a"]).cuda().float(),
        "frobenius": torch.eye(config["candidate_pool_rank"], device="cuda")[:, : config["rank"]],
    }
    fit_info = {}
    fit_specs = {row["name"]: row for row in config["fits"]}
    for fit_spec in config["fits"]:
        name = fit_spec["name"]
        train_half = fit_spec["train_half"]
        validation_half = "odd" if train_half == "even" else "even"
        print("optimizing", name, fit_spec["initialization"], flush=True)
        quotient, info = optimize_subspace(
            coordinates, forms, target, metric,
            halves[train_half], halves[validation_half], config,
            config["seed"] + fit_spec["seed_offset"],
            initialization=fit_spec["initialization"],
        )
        quotients[name] = quotient
        fit_info[name] = info

    objective = {}
    for name, quotient in quotients.items():
        prediction = continuous_prediction(coordinates, forms, quotient)
        objective[name] = {
            half: float(fisher_quadratic(
                target[indices] - prediction[indices], metric[indices]
            ).mean())
            for half, indices in halves.items()
        }
    overlaps = {}
    for first, second in combinations(quotients, 2):
        overlaps[f"{first}__{second}"] = canonical_overlap(quotients[first], quotients[second])

    results = {}
    for corpus, spec in config["evaluation"].items():
        print("external evaluation", corpus, flush=True)
        results[corpus] = evaluate(
            model, spec, config, output_basis, inverse, candidate_basis,
            forms, traces_tensor, quotients,
        )

    random_names = [row["name"] for row in config["fits"] if row["initialization"] == "random"]
    def opposite_half(name: str) -> str:
        return "odd" if fit_specs[name]["train_half"] == "even" else "even"

    b1 = all(
        objective[name][opposite_half(name)]
        <= 0.85 * objective["frobenius"][opposite_half(name)]
        for name in random_names
    )
    random_pairs = list(combinations(random_names, 2))
    b2 = all(overlaps[f"{a}__{b}"]["mean"] >= 0.80 for a, b in random_pairs)
    b3 = all(overlaps[f"prefix_a__{name}"]["mean"] >= 0.80 for name in random_names)
    b4 = all(
        row["behavioral"][name]["mean_exact_kl"]
        < row["behavioral"]["frobenius"]["mean_exact_kl"]
        for row in results.values() for name in random_names
    )
    orthogonal = "orthogonal_even"
    b5 = (
        objective[orthogonal]["odd"] <= 0.85 * objective["frobenius"]["odd"]
        and overlaps[f"prefix_a__{orthogonal}"]["mean"] >= 0.80
    )
    b6 = any(
        all(row["behavioral"][name]["median_exact_kl"] <= 0.005 for row in results.values())
        for name in random_names
    )
    hypotheses = {
        "B1_random_starts_improve_heldout_fisher": {"supported": b1},
        "B2_random_fits_recover_same_subspace": {"supported": b2},
        "B3_random_fits_match_prefix_solution": {"supported": b3},
        "B4_random_improvement_transfers": {"supported": b4},
        "B5_adversarial_start_escapes": {"supported": b5},
        "B6_random_solution_reaches_gate": {"supported": b6},
    }
    payload = {
        "config": config,
        "hypotheses": hypotheses,
        "covariance": covariance,
        "objectives": objective,
        "overlaps": overlaps,
        "fit_info": fit_info,
        "results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    np.savez_compressed(ARRAYS, **{
        "candidate_basis": candidate_basis.cpu().numpy(),
        **{name: quotient.cpu().numpy() for name, quotient in quotients.items()},
    })

    lines = [
        "# Random-start red team of the continuous Layer-17 Fisher subspace",
        "",
        "The earlier two split fits shared a near-Frobenius initialization. This preregistered red team repeats the frozen rank-eight problem from three random starts and one start deliberately orthogonal to the Frobenius prefix.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "",
        "![Basin red team](layer17_tail_continuous_fisher_basin_redteam.png)",
        "",
        "| Program | Even Fisher | Odd Fisher | Cosine to prefix |",
        "|---|---:|---:|---:|",
    ]
    for name, row in objective.items():
        cosine = 1.0 if name == "prefix_a" else overlaps.get(f"prefix_a__{name}", {"mean": float("nan")})["mean"]
        lines.append(f"| {name} | {row['even']:.4e} | {row['odd']:.4e} | {cosine:.3f} |")
    lines += [
        "",
        "| Corpus | Program | Mean exact KL | Median exact KL | Top-1 agreement |",
        "|---|---|---:|---:|---:|",
    ]
    for corpus, row in results.items():
        for name in quotients:
            behavior = row["behavioral"][name]
            lines.append(
                f"| {corpus} | {name} | {behavior['mean_exact_kl']:.5f} | "
                f"{behavior['median_exact_kl']:.5f} | {behavior['top1_agreement']:.3f} |"
            )
    lines += [
        "",
        "Agreement after random and adversarial initialization is evidence for a recoverable behavioral subspace. Failure would demote the earlier stable-looking basis to one useful local quotient rather than a canonical circuit.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    figure, plots = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(quotients)
    plots[0].bar(np.arange(len(names)) - 0.18, [objective[n]["even"] for n in names], 0.36, label="even")
    plots[0].bar(np.arange(len(names)) + 0.18, [objective[n]["odd"] for n in names], 0.36, label="odd")
    plots[0].set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    plots[0].set_ylabel("Fisher score error")
    plots[0].set_title("Discovery split objective")
    plots[0].legend()
    random_matrix = np.asarray([
        [canonical_overlap(quotients[a], quotients[b])["mean"] for b in names]
        for a in names
    ])
    image = plots[1].imshow(random_matrix, vmin=0, vmax=1, cmap="viridis")
    plots[1].set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    plots[1].set_yticks(np.arange(len(names)), names)
    plots[1].set_title("Mean canonical cosine")
    figure.colorbar(image, ax=plots[1], fraction=0.046)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
