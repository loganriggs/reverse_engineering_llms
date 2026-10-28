#!/usr/bin/env python3
"""Red-team exact-KL subspace fitting with a local learning-rate sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from bilin_checkpoint_utils import collect_states_writes, load_snapshot_model
from run_bilin12_layer11_behavior_subspace import collect_final_sites, optimize_basis
from run_bilin12_layer11_bottleneck_tail_replication import behavior_metrics
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin12_layer11_behavior_optimizer_redteam.json")
SOURCE = TENSOR_ROOT / "bilin12_layer11_bottleneck_tail_replication.npz"
OUTPUT = TENSOR_ROOT / "bilin12_layer11_behavior_optimizer_redteam.json"
ARRAYS = TENSOR_ROOT / "bilin12_layer11_behavior_optimizer_redteam.npz"
REPORT = TENSOR_ROOT / "BILIN12_LAYER11_BEHAVIOR_OPTIMIZER_REDTEAM.md"
FIGURE = TENSOR_ROOT / "bilin12_layer11_behavior_optimizer_redteam.png"


def baseline_name(family: str, rank: int) -> str:
    return f"natural{rank}" if family == "global" else f"raw4_tail{rank - 4}"


def main() -> None:
    config = json.loads(CONFIG.read_text())
    source = np.load(SOURCE)
    raw4 = torch.from_numpy(source["raw_outlier_basis"]).cuda().float()
    tail = torch.from_numpy(source["tail_basis"]).cuda().float()
    natural = torch.from_numpy(source["natural_basis"]).cuda().float()
    post_bias_mean = torch.from_numpy(source["post_bias_write_mean"]).cuda().float()
    model = load_snapshot_model(Path(config["snapshot"]), "cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    discovery = collect_final_sites(
        model, config["layer"], config["discovery"], config["batch_size"]
    )

    sweep = {}
    selected = {}
    arrays = {}
    for rank in config["ranks"]:
        for family in config["families"]:
            key = f"{family}{rank}"
            rows = []
            candidate_bases = []
            if family == "global":
                initialization = natural[:, :rank]
                anchor = None
            else:
                initialization = tail[:, : rank - raw4.shape[1]]
                anchor = raw4
            for index, learning_rate in enumerate(config["learning_rates"]):
                print("fit", key, "lr", learning_rate, flush=True)
                fit_config = {**config, "learning_rate": learning_rate}
                basis, metadata = optimize_basis(
                    model, discovery, post_bias_mean, initialization, anchor,
                    fit_config, config["seed"] + 1000 * rank + 100 * (family == "anchored") + index,
                )
                initial = metadata["history"][0]["validation_kl"]
                row = {
                    "learning_rate": learning_rate,
                    "initial_validation_kl": initial,
                    "best_validation_kl": metadata["best_validation_kl"],
                    "relative_validation_improvement": (initial - metadata["best_validation_kl"]) / initial,
                    "best_step": min(metadata["history"], key=lambda item: item["validation_kl"])["step"],
                    "history": metadata["history"],
                    "candidate_index": index,
                }
                rows.append(row)
                candidate_bases.append(basis.cpu().numpy())
            winner = min(rows, key=lambda item: item["best_validation_kl"])
            selected[key] = candidate_bases[winner["candidate_index"]]
            arrays[f"selected_{key}"] = selected[key].astype(np.float32)
            sweep[key] = {"candidates": rows, "selected": winner}

    baselines = {
        "natural20": natural[:, :20].cpu().numpy(),
        "raw4_tail16": torch.cat((raw4, tail[:, :16]), 1).cpu().numpy(),
        "natural68": natural[:, :68].cpu().numpy(),
        "raw4_tail64": torch.cat((raw4, tail[:, :64]), 1).cpu().numpy(),
    }
    bases = {**baselines, **selected}
    results = {}
    for corpus, spec in config["evaluation"].items():
        print("external", corpus, flush=True)
        behavior = behavior_metrics(
            model, config["layer"], spec, config["batch_size"],
            post_bias_mean.cpu().numpy(), bases,
        )
        _, write = collect_states_writes(model, config["layer"], spec, [-1], config["batch_size"])
        centered = write.cpu().numpy() - source["pre_bias_write_mean"]
        variance_fraction = {
            name: float(np.square(centered @ basis).sum() / np.square(centered).sum())
            for name, basis in bases.items()
        }
        results[corpus] = {"behavior": behavior, "variance_fraction": variance_fraction}

    hypotheses = {
        "R1_local_optimization_improves_validation": {
            "supported": all(
                max(row["relative_validation_improvement"] for row in sweep[key]["candidates"]) >= 0.02
                for key in sweep
            ),
        },
        "R2_rank20_gain_survives_external_data": {
            "supported": all(
                results[corpus]["behavior"][f"{family}20"]["median_kl"]
                <= 0.95 * results[corpus]["behavior"][baseline_name(family, 20)]["median_kl"]
                for corpus in results for family in config["families"]
            ),
        },
        "R3_rank68_reaches_near_sufficiency": {
            "supported": any(
                all(results[corpus]["behavior"][f"{family}68"]["median_kl"] <= 0.02 for corpus in results)
                for family in config["families"]
            ),
        },
        "R4_behavior_fit_trades_away_variance": {
            "supported": any(
                all(
                    results[corpus]["variance_fraction"][f"{family}{rank}"]
                    < results[corpus]["variance_fraction"][baseline_name(family, rank)]
                    and results[corpus]["behavior"][f"{family}{rank}"]["median_kl"]
                    < results[corpus]["behavior"][baseline_name(family, rank)]["median_kl"]
                    for corpus in results
                )
                for rank in config["ranks"] for family in config["families"]
            ),
        },
        "R5_anchor_is_not_costly": {
            "supported": all(
                results[corpus]["behavior"][f"anchored{rank}"]["median_kl"]
                <= 1.1 * results[corpus]["behavior"][f"global{rank}"]["median_kl"]
                for corpus in results for rank in config["ranks"]
            ),
        },
    }
    np.savez_compressed(ARRAYS, **arrays)
    OUTPUT.write_text(json.dumps({
        "config": config,
        "hypotheses": hypotheses,
        "sweep": sweep,
        "results": results,
    }, indent=2, allow_nan=False))

    lines = [
        "# Layer-11 behavior-subspace optimizer red team",
        "",
        "The failed aggressive exact-KL optimization is red-teamed with zero-noise local starts and three learning rates. Selection uses only interleaved FineWeb validation KL; selected bases are frozen before disjoint FineWeb, WikiText, and TinyStories evaluation.",
        "",
        "## Locked hypotheses",
        "",
    ]
    for name, row in hypotheses.items():
        lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "![Optimizer red team](bilin12_layer11_behavior_optimizer_redteam.png)", "", "| Fit | Selected LR | Step | Validation improvement |", "|---|---:|---:|---:|"]
    for key, row in sweep.items():
        winner = row["selected"]
        lines.append(f"| {key} | {winner['learning_rate']:.0e} | {winner['best_step']} | {winner['relative_validation_improvement']:.1%} |")
    lines += ["", "| Corpus | PCA20 / global20 | Raw+tail16 / anchored20 | PCA68 / global68 | Raw+tail64 / anchored68 |", "|---|---:|---:|---:|---:|"]
    for corpus, row in results.items():
        b = row["behavior"]
        lines.append(
            f"| {corpus} | {b['natural20']['median_kl']:.5f} / {b['global20']['median_kl']:.5f} | "
            f"{b['raw4_tail16']['median_kl']:.5f} / {b['anchored20']['median_kl']:.5f} | "
            f"{b['natural68']['median_kl']:.5f} / {b['global68']['median_kl']:.5f} | "
            f"{b['raw4_tail64']['median_kl']:.5f} / {b['anchored68']['median_kl']:.5f} |"
        )
    lines += [
        "",
        "Small local improvements would validate a metric mismatch without establishing a compact circuit. The near-sufficiency gate remains the decisive test. If a fixed rank-68 space stays far above it, the remaining candidates are context routing, nonlinear reconstruction, or genuinely broad support—not merely an unlucky PCA ordering.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")

    keys = list(sweep)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for key in keys:
        rows = sweep[key]["candidates"]
        axes[0].plot(
            [row["learning_rate"] for row in rows],
            [row["relative_validation_improvement"] for row in rows],
            marker="o", label=key,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("learning rate")
    axes[0].set_ylabel("validation KL improvement")
    axes[0].set_title("Local optimizer sweep")
    axes[0].legend(fontsize=8)
    labels = ["global20", "anchored20", "global68", "anchored68"]
    x = np.arange(len(labels))
    width = 0.25
    for offset, corpus in enumerate(results):
        ratios = [
            results[corpus]["behavior"][name]["median_kl"]
            / results[corpus]["behavior"][baseline_name("global" if name.startswith("global") else "anchored", int(name[-2:]))]["median_kl"]
            for name in labels
        ]
        axes[1].bar(x + (offset - 1) * width, ratios, width, label=corpus)
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("held-out KL / PCA baseline")
    axes[1].set_title("External transfer")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=180)
    plt.close(figure)
    print("wrote", OUTPUT, ARRAYS, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
