#!/usr/bin/env python3
"""Compare learned raw-neuron cleanliness with rotation and untrained nulls."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch


def load_experiment():
    path = Path(__file__).with_name("run_self_bilinear_compositional_memory.py")
    spec = importlib.util.spec_from_file_location("self_bilinear_experiment", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def cleanliness(features, activations):
    centered = activations - activations.mean(0)
    standardized = centered / centered.square().mean(0).sqrt().clamp_min(1e-6)
    explained = (features.T @ standardized / len(activations)).square()
    strength = explained.max(0).values
    return strength * (strength / explained.sum(0).clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/self_bilinear_memory/full_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/self_bilinear_redteam"))
    parser.add_argument("--rotations", type=int, default=100)
    parser.add_argument("--untrained", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    experiment = load_experiment()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    run = SimpleNamespace(
        facts=config["facts"], neurons=config["neurons"], depth=config["depth"],
        computation_tasks=[18432], test_tasks_per_arity=1024, device=args.device,
    )
    seed = 7; bank = experiment.build_bank(run, seed); values = checkpoint["values"].to(args.device)
    lookup = experiment.lookup_data(run, values)
    data = experiment.encoded_bank(run, bank, values, 6144)
    inputs, targets = experiment.analysis_dataset(lookup, data)
    candidate = torch.cat(((inputs[:, :run.facts] != 0).float(), inputs[:, run.facts:].gt(0).float(), targets[:, None].float()), 1)
    candidate -= candidate.mean(0); scale = candidate.square().mean(0).sqrt(); candidate = candidate[:, scale > 1e-6] / scale[scale > 1e-6]
    model = experiment.DeepSelfBilinear(run.facts + len(experiment.ARITIES), run.neurons, run.depth, residual_stream=True, decoder_bias=True).to(args.device)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    with torch.inference_mode(): learned = model(inputs, return_activations=True)[1]
    generator = torch.Generator(device=args.device).manual_seed(123)
    result = {}
    for layer_index, activations in enumerate(learned):
        raw = float((cleanliness(candidate, activations) > args.threshold).float().mean())
        rotations = []
        for _ in range(args.rotations):
            rotation = torch.linalg.qr(torch.randn(run.neurons, run.neurons, generator=generator, device=args.device)).Q
            rotations.append(float((cleanliness(candidate, activations @ rotation) > args.threshold).float().mean()))
        untrained = []
        for untrained_seed in range(args.untrained):
            torch.manual_seed(1000 + untrained_seed)
            null = experiment.DeepSelfBilinear(run.facts + len(experiment.ARITIES), run.neurons, run.depth, residual_stream=True, decoder_bias=True).to(args.device).eval()
            with torch.inference_mode(): null_activations = null(inputs, return_activations=True)[1][layer_index]
            untrained.append(float((cleanliness(candidate, null_activations) > args.threshold).float().mean()))
        result[str(layer_index + 1)] = {
            "raw_fraction_above_threshold": raw,
            "rotation_mean": statistics.mean(rotations), "rotation_sd": statistics.stdev(rotations),
            "rotation_min": min(rotations), "rotation_max": max(rotations),
            "untrained_mean": statistics.mean(untrained), "untrained_sd": statistics.stdev(untrained),
            "rotation_values": rotations, "untrained_values": untrained,
        }
    payload = {"threshold":args.threshold,"rotations":args.rotations,"untrained_models":args.untrained,"layers":result}
    (args.output / "cleanliness_nulls.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    layers = list(result); positions = list(range(len(layers))); width = .25
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar([x-width for x in positions], [result[x]["raw_fraction_above_threshold"] for x in layers], width, label="raw learned")
    axis.bar(positions, [result[x]["rotation_mean"] for x in layers], width, yerr=[result[x]["rotation_sd"] for x in layers], capsize=4, label="random rotations")
    axis.bar([x+width for x in positions], [result[x]["untrained_mean"] for x in layers], width, yerr=[result[x]["untrained_sd"] for x in layers], capsize=4, label="untrained")
    axis.set_xticks(positions, [f"layer {x}" for x in layers]); axis.set_ylabel(f"fraction cleanliness > {args.threshold}")
    axis.set_ylim(0, .8); axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(args.output / "cleanliness_nulls.png", dpi=180); plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
