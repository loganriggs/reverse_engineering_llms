#!/usr/bin/env python3
"""Run the preregistered one-factor clean-axis scaling study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


BASE = {"classes": 30, "operations": 4, "depth": 2, "neurons": 128}
GRID = {
    "classes": [10, 30, 100],
    "operations": [2, 4, 8],
    "depth": [1, 2, 4],
    "neurons": [64, 128, 256],
}


def configurations():
    seen = set()
    for factor, values in GRID.items():
        for value in values:
            config = BASE | {factor: value}
            key = tuple(config.items())
            if key not in seen:
                seen.add(key)
                yield factor, value, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/clean_axis_scaling"))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"base": BASE, "grid": GRID, "seeds": list(range(args.seeds)), "runs": []}
    for factor, value, config in configurations():
        slug = f"c{config['classes']}_o{config['operations']}_d{config['depth']}_w{config['neurons']}"
        for seed in range(args.seeds):
            destination = args.output / slug / f"seed_{seed}"
            result = destination / "results.json"
            manifest["runs"].append({"factor": factor, "value": value, "seed": seed, "slug": slug, **config})
            if result.exists() and not args.force:
                print(f"SKIP {slug} seed {seed}", flush=True)
                continue
            command = [
                sys.executable, "scripts/run_multiclass_subspace_pilot.py",
                "--input-mode", "prototypes", "--classes", str(config["classes"]),
                "--content-dim", "128", "--operations", str(config["operations"]),
                "--code-dim", "16", "--neurons", str(config["neurons"]),
                "--depth", str(config["depth"]), "--train-samples-per-cell", "32",
                "--confirmation-samples-per-cell", "8", "--max-steps", "5000",
                "--rotations", "5", "--seed", str(seed), "--device", args.device,
                "--output", str(destination),
            ]
            print(f"START {slug} seed {seed}", flush=True)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            print(f"DONE  {slug} seed {seed}", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
