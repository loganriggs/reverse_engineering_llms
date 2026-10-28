#!/usr/bin/env python3
"""Compute the preregistered endpoints for locked multiclass confirmation runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def interval(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def value(row: dict, method: str) -> float:
    if method == "raw":
        return row["raw_confirmation"]["kl_recovery"]
    if method == "rotation":
        return row["rotation_confirmation"]["kl_recovery"]["mean"]
    if method == "pca":
        return row["pca_confirmation"]["kl_recovery"]
    raise ValueError(method)


def threshold_rank(rows: list[dict], method: str, threshold: float) -> int | None:
    return next((row["rank"] for row in rows if value(row, method) >= threshold), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input.glob("seed_*/results.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise SystemExit("No locked runs found")
    summary: dict = {
        "runs": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "converged": [run["training"]["converged"] for run in runs],
        "confirmation": {
            metric: interval([run["evaluations"]["confirmation"][metric] for run in runs])
            for metric in ("accuracy", "top5_accuracy", "cross_entropy", "mean_cyclic_error")
        },
        "families": {},
    }
    for family in runs[0]["analysis"]["families"]:
        summary["families"][family] = {"layers": {}}
        for layer in runs[0]["analysis"]["families"][family]["layers"]:
            all_rows = [run["analysis"]["families"][family]["layers"][layer] for run in runs]
            sparse_rows = [[row for row in rows if row["rank"] <= 64] for rows in all_rows]
            axis_privilege = [statistics.mean(value(row, "raw") - value(row, "rotation") for row in rows) for rows in sparse_rows]
            subspace_advantage = [statistics.mean(value(row, "pca") - value(row, "raw") for row in rows) for rows in sparse_rows]
            layer_output: dict = {
                "axis_privilege": interval(axis_privilege),
                "distributed_subspace_advantage": interval(subspace_advantage),
                "threshold_ranks": {},
                "curve": [],
            }
            for threshold in (0.8, 0.9):
                threshold_output = {}
                for method in ("raw", "rotation", "pca"):
                    ranks = [threshold_rank(rows, method, threshold) for rows in all_rows]
                    reached = [rank for rank in ranks if rank is not None]
                    threshold_output[method] = {
                        "per_seed": ranks,
                        "reached_count": len(reached),
                        "median_if_reached": statistics.median(reached) if reached else None,
                    }
                layer_output["threshold_ranks"][str(threshold)] = threshold_output
            for index, row in enumerate(all_rows[0]):
                layer_output["curve"].append(
                    {"rank": row["rank"]}
                    | {
                        method: interval([value(rows[index], method) for rows in all_rows])
                        for method in ("raw", "rotation", "pca")
                    }
                )
            summary["families"][family]["layers"][layer] = layer_output
    destination = args.output or args.input / "locked_summary.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
