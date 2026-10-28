#!/usr/bin/env python3
"""Train fixed self-bilinear lookup tasks across depths 1--4 for synergy audits."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


def load_experiment():
    path = Path(__file__).with_name("run_self_bilinear_compositional_memory.py")
    spec = importlib.util.spec_from_file_location("self_bilinear_experiment", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/causal_synergy/memory_models"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(7, 19, 31))
    parser.add_argument("--depths", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--facts", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment()
    run_args = SimpleNamespace(
        facts=args.facts, neurons=args.neurons, computation_tasks=[18432],
        computation_fraction=.75, test_tasks_per_arity=1024, batch_size=1024,
        max_steps=args.max_steps, learning_rate=.005, device=args.device,
    )
    rows = []
    for seed in args.seeds:
        bank = experiment.build_bank(run_args, seed)
        for task, tasks_each, supported in (
            ("lookup", 0, ()),
            ("lookup_parity2", 6144, (2,)),
        ):
            for depth in args.depths:
                destination = args.output / task / f"depth_{depth}" / f"seed_{seed}"
                checkpoint_path = destination / "checkpoint.pt"
                if checkpoint_path.exists() and not args.force:
                    metadata = json.loads((destination / "results.json").read_text())
                    print(f"SKIP {destination}", flush=True)
                else:
                    row, model, values, _, _ = experiment.train(
                        run_args, seed, depth, tasks_each, supported, bank,
                        residual_stream=True, decoder_bias=True,
                    )
                    destination.mkdir(parents=True, exist_ok=True)
                    checkpoint = {
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "values": values.detach().cpu(),
                        "config": {
                            "facts": args.facts, "neurons": args.neurons, "depth": depth,
                            "seed": seed, "task": task, "tasks_per_arity": tasks_each,
                            "supported_arities": list(supported), "residual_stream": True,
                            "decoder_bias": True,
                        },
                    }
                    torch.save(checkpoint, checkpoint_path)
                    metadata = {**row, "task": task}
                    (destination / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                    print(f"DONE {destination} converged={row['converged']} loss={row['final_loss']:.3e}", flush=True)
                rows.append(metadata)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
