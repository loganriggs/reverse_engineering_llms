#!/usr/bin/env python3
"""Many-seed statistics for the maximum-load residual/bias experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from run_self_bilinear_compositional_memory import (
    circuit_metrics,
    build_bank,
    train,
)


CONFIGURATIONS = {
    "full": (True, True),
    "no_bias": (True, False),
    "no_residual": (False, True),
    "neither": (False, False),
}
FAMILIES = ("lookup", "parity_2", "parity_4", "parity_8")
EVALUATION_KEYS = ("lookup", "parity_2_heldout", "parity_4_heldout", "parity_8_heldout")
THRESHOLDS = ("0.8", "0.9", "0.99")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--computation-tasks", type=int, default=18432)
    parser.add_argument("--computation-fraction", type=float, default=.75)
    parser.add_argument("--test-tasks-per-arity", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=.005)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--configuration", choices=["all", *CONFIGURATIONS], default="all")
    parser.add_argument("--circuit-examples", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("artifacts/self_bilinear_seed_statistics"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def mean_sd_ci(values):
    total = len(values)
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "n_total": total, "missing": total, "mean": None, "sd": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    rng = random.Random(1729)
    bootstrap = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return {
        "n": len(values), "n_total": total, "missing": total-len(values), "mean": mean, "sd": sd,
        "ci95_low": bootstrap[249], "ci95_high": bootstrap[9750],
    }


def distribution_stats(values):
    base = mean_sd_ci(values)
    valid = sorted(value for value in values if value is not None)
    if not valid:
        return {**base, "median": None, "q05": None, "q95": None}
    def quantile(fraction):
        position = fraction * (len(valid)-1)
        low = math.floor(position); high = math.ceil(position)
        return valid[low] if low == high else valid[low] * (high-position) + valid[high] * (position-low)
    return {**base, "median": statistics.median(valid), "q05": quantile(.05), "q95": quantile(.95)}


def wilson(successes, total, z=1.96):
    proportion = successes / total
    denominator = 1 + z*z/total
    center = (proportion + z*z/(2*total)) / denominator
    half = z * math.sqrt(proportion*(1-proportion)/total + z*z/(4*total*total)) / denominator
    return {"successes": successes, "n": total, "fraction": proportion, "wilson95_low": max(0.0, center-half), "wilson95_high": min(1.0, center+half)}


def summarize(rows):
    summary = {"architectures": {}, "full_model_circuits": {}}
    for name in CONFIGURATIONS:
        selected = [row for row in rows if row["configuration"] == name]
        if not selected:
            continue
        summary["architectures"][name] = {
            "convergence": wilson(sum(row["converged"] for row in selected), len(selected)),
            "final_loss": mean_sd_ci([row["final_loss"] for row in selected]),
            "steps": mean_sd_ci([row["steps"] for row in selected]),
            "accuracy": {key: mean_sd_ci([row["evaluations"][key] for row in selected]) for key in EVALUATION_KEYS},
            "cross_entropy": {key: distribution_stats([row.get("evaluation_cross_entropy", {}).get(key) for row in selected]) for key in EVALUATION_KEYS},
        }
    full = [row for row in rows if row["configuration"] == "full" and "circuits" in row]
    if full:
        for family in FAMILIES:
            circuit_summary = {
                "intact_cross_entropy": mean_sd_ci([row["circuits"][family]["full_cross_entropy"] for row in full]),
                "ce_margin_rank_correlation": mean_sd_ci([row["circuits"][family]["ce_margin_spearman_rank_correlation"] for row in full]),
                "accuracy_shared_neurons": {},
                "ce_recovery_shared_neurons": {},
            }
            for threshold in THRESHOLDS:
                accuracy_counts = [row["circuits"][family]["accuracy_thresholds"][threshold]["shared"] for row in full]
                ce_counts = [row["circuits"][family]["ce_recovery_thresholds"][threshold]["shared"] for row in full]
                circuit_summary["accuracy_shared_neurons"][threshold] = mean_sd_ci(accuracy_counts)
                circuit_summary["ce_recovery_shared_neurons"][threshold] = mean_sd_ci(ce_counts)
            summary["full_model_circuits"][family] = circuit_summary
    return summary


def plot_architectures(summary, path):
    names = [name for name in CONFIGURATIONS if name in summary["architectures"]]
    positions = list(range(len(names)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    width = .19
    for offset, key in enumerate(EVALUATION_KEYS):
        stats = [summary["architectures"][name]["accuracy"][key] for name in names]
        means = [x["mean"] for x in stats]
        axes[0].bar(
            [x + (offset-1.5)*width for x in positions], means, width,
            yerr=[[mean-x["ci95_low"] for mean,x in zip(means,stats)], [x["ci95_high"]-mean for mean,x in zip(means,stats)]], capsize=3,
            label=key.replace("_heldout", "").replace("_", " "),
        )
    axes[0].set_xticks(positions, [name.replace("_", "\n") for name in names])
    axes[0].set_ylim(0, 1.06); axes[0].set_ylabel("accuracy, mean ± 95% CI"); axes[0].legend(fontsize=8)
    convergence = [summary["architectures"][name]["convergence"] for name in names]
    axes[1].bar(positions, [x["fraction"] for x in convergence], color="#6A5ACD")
    axes[1].errorbar(
        positions, [x["fraction"] for x in convergence],
        yerr=[[x["fraction"]-x["wilson95_low"] for x in convergence], [x["wilson95_high"]-x["fraction"] for x in convergence]],
        fmt="none", color="black", capsize=4,
    )
    axes[1].set_xticks(positions, [name.replace("_", "\n") for name in names]); axes[1].set_ylim(0, 1.06)
    axes[1].set_ylabel("convergence fraction, Wilson 95% CI")
    for axis in axes: axis.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_circuits(rows, path):
    full = [row for row in rows if row["configuration"] == "full" and "circuits" in row]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for axis, metric, title in (
        (axes[0], "accuracy_thresholds", "90% accuracy"),
        (axes[1], "ce_recovery_thresholds", "90% CE recovery"),
    ):
        values = [[value for row in full if (value := row["circuits"][family][metric]["0.9"]["shared"]) is not None] for family in FAMILIES]
        axis.boxplot(values, tick_labels=[x.replace("_", " ") for x in FAMILIES], showmeans=True)
        axis.set_title(title); axis.grid(axis="y", alpha=.25); axis.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("retained layer-neuron activations")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_cross_entropy(rows, path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    names = list(CONFIGURATIONS)
    for axis, key in zip(axes, EVALUATION_KEYS):
        values = [[row["evaluation_cross_entropy"][key] for row in rows if row["configuration"] == name and "evaluation_cross_entropy" in row] for name in names]
        axis.boxplot(values, tick_labels=[name.replace("_", "\n") for name in names], showmeans=True, showfliers=True)
        axis.set_yscale("symlog", linthresh=1e-4); axis.set_title(key.replace("_heldout", "").replace("_", " ")); axis.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("held-out cross-entropy (symlog)")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def write_summary(root, output):
    summary = output["summary"]
    lines = [
        "# Self-bilinear seed statistics", "",
        f"This study uses {len(output['seeds'])} independent seeds at {output['config']['computation_tasks']} computation tasks. Each seed changes fact values, sampled train/test subsets, and parameter initialization. JSON mean intervals are 10,000-resample bootstrap 95% confidence intervals across seeds; convergence intervals are Wilson binomial intervals. Tables show mean ± sample SD.", "",
        "## Architecture outcomes", "",
        "| architecture | converged | lookup mean ± SD | parity 2 | parity 4 | parity 8 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CONFIGURATIONS:
        if name not in summary["architectures"]: continue
        row = summary["architectures"][name]; conv = row["convergence"]
        cells = []
        for key in EVALUATION_KEYS:
            stat = row["accuracy"][key]; cells.append(f"{stat['mean']:.3f} ± {stat['sd']:.3f}")
        lines.append(f"| {name.replace('_', ' ')} | {conv['successes']}/{conv['n']} ({conv['fraction']:.1%}) | " + " | ".join(cells) + " |")
    lines += ["", "## Held-out cross-entropy variance", "",
        "CE is heavy-tailed, so both mean ± SD and median are shown. Unlike accuracy, CE penalizes rare confident errors without bound.", "",
        "| architecture | lookup | parity 2 | parity 4 | parity 8 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CONFIGURATIONS:
        if name not in summary["architectures"]: continue
        cells = []
        for key in EVALUATION_KEYS:
            stat = summary["architectures"][name]["cross_entropy"][key]
            cells.append("pending" if stat["mean"] is None else f"{stat['mean']:.3g} ± {stat['sd']:.3g} (med {stat['median']:.3g})")
        lines.append(f"| {name.replace('_', ' ')} | " + " | ".join(cells) + " |")
    if summary["full_model_circuits"]:
        lines += ["", "## Full-model circuit-size variance", "",
            "Counts are shared task-family masks. Values are mean ± SD across independently trained models.", "",
            "| family | neurons for 90% accuracy | neurons for 90% CE recovery | CE/margin rank correlation |",
            "|---|---:|---:|---:|",
        ]
        def formatted(stat, digits=1):
            if stat["mean"] is None:
                return f"not attained (0/{stat['n_total']})"
            suffix = f"; {stat['n']}/{stat['n_total']} attained" if stat["missing"] else ""
            return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}{suffix}"
        for family in FAMILIES:
            row = summary["full_model_circuits"][family]
            acc = row["accuracy_shared_neurons"]["0.9"]; ce = row["ce_recovery_shared_neurons"]["0.9"]; corr = row["ce_margin_rank_correlation"]
            lines.append(f"| {family} | {formatted(acc)} | {formatted(ce)} | {formatted(corr, 3)} |")
    lines += ["", "The complete JSON contains 80%, 90%, and 99% thresholds, confidence intervals, per-seed curves, losses, steps, and accuracies. Failed seeds are retained in unconditional architecture means rather than silently discarded.", ""]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    existing = json.loads(path.read_text()) if path.exists() else {"runs": []}
    rows = existing["runs"]
    requested = CONFIGURATIONS if args.configuration == "all" else {args.configuration: CONFIGURATIONS[args.configuration]}
    for configuration, (residual, bias) in requested.items():
        for seed in args.seeds:
            previous = next((row for row in rows if row["configuration"] == configuration and row["seed"] == seed), None)
            if previous is not None and "evaluation_cross_entropy" in previous:
                continue
            bank_args = SimpleNamespace(**vars(args))
            bank_args.computation_tasks = [args.computation_tasks]
            bank = build_bank(bank_args, seed)
            row, model, _, lookup, data = train(
                args, seed, args.depth, args.computation_tasks // 3, (2, 4, 8), bank,
                residual_stream=residual, decoder_bias=bias,
            )
            row["configuration"] = configuration
            with torch.inference_mode():
                row["evaluation_cross_entropy"] = {
                    "lookup": float(F.cross_entropy(model(lookup[0]), lookup[1])),
                    **{f"parity_{arity}_heldout": float(F.cross_entropy(model(data["test"][arity][0]), data["test"][arity][1])) for arity in (2,4,8)},
                }
            if configuration == "full" and row["converged"]:
                families = {"lookup": lookup, **{f"parity_{arity}": (data["test"][arity][0][:args.circuit_examples], data["test"][arity][1][:args.circuit_examples]) for arity in (2,4,8)}}
                row["circuits"] = {name: circuit_metrics(model, dataset) for name, dataset in families.items()}
            if previous is not None:
                rows[rows.index(previous)] = row
            else:
                rows.append(row)
            partial = {
                "config": {"facts":args.facts,"neurons":args.neurons,"depth":args.depth,"computation_tasks":args.computation_tasks},
                "seeds": args.seeds, "runs": rows,
            }
            partial["summary"] = summarize(rows)
            path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            print(configuration, seed, "converged", row["converged"], "loss", f"{row['final_loss']:.3e}", row["evaluations"], flush=True)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    output = json.loads(path.read_text()); output["summary"] = summarize(output["runs"])
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_summary(args.output, output)
    plot_architectures(output["summary"], args.output / "architecture_seed_variance.png")
    plot_cross_entropy(output["runs"], args.output / "cross_entropy_seed_variance.png")
    if output["summary"]["full_model_circuits"]:
        plot_circuits(output["runs"], args.output / "circuit_seed_variance.png")


if __name__ == "__main__":
    main()
