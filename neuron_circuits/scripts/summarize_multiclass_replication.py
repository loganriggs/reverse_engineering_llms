#!/usr/bin/env python3
"""Aggregate sealed multiclass runs without refitting or selecting endpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def mean_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.input.glob("seed_*/results.json"))
    if not paths:
        raise SystemExit(f"No seed_*/results.json files found under {args.input}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary: dict = {
        "runs": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "confirmation": {},
        "layers": {},
    }
    for metric in ("accuracy", "top5_accuracy", "cross_entropy", "mean_cyclic_error"):
        summary["confirmation"][metric] = mean_ci(
            [run["evaluations"]["confirmation"][metric] for run in runs]
        )
    for layer in runs[0]["analysis"]["layers"]:
        summary["layers"][layer] = []
        for row_index, row in enumerate(runs[0]["analysis"]["layers"][layer]):
            raw = [run["analysis"]["layers"][layer][row_index]["raw_confirmation"]["kl_recovery"] for run in runs]
            rotation = [run["analysis"]["layers"][layer][row_index]["rotation_confirmation"]["kl_recovery"]["mean"] for run in runs]
            pca = [run["analysis"]["layers"][layer][row_index]["pca_confirmation"]["kl_recovery"] for run in runs]
            summary["layers"][layer].append({
                "rank": row["rank"],
                "raw": mean_ci(raw),
                "rotation": mean_ci(rotation),
                "pca": mean_ci(pca),
                "raw_minus_rotation": mean_ci([a - b for a, b in zip(raw, rotation)]),
                "pca_minus_raw": mean_ci([a - b for a, b in zip(pca, raw)]),
            })
    destination = args.output or args.input / "replication_summary.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
