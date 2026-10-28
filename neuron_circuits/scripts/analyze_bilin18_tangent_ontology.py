#!/usr/bin/env python3
"""Separate bilinear value, tangent, curvature, and finite-effect fidelity."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from evaluate_bilin18_causal_cp import load_cp
from run_bilin18_tensor_decomposition import ROOT, teacher_bilinear


DATA = Path("artifacts/bilin18_sparse_codes/layer1_inputs_large.npz")
OUTPUT = ROOT / "tangent_ontology_results.json"
REPORT = ROOT / "TANGENT_ONTOLOGY_REPORT.md"
FIGURE = ROOT / "tangent_ontology_summary.png"
SPLITS = ("fineweb_test", "wikitext103", "tinystories")
MODELS = {
    "direct_cp": ROOT / "cp_r2048.pt",
    "activation_cp": ROOT / "cp_r2048_activation.pt",
    "kl_r2": ROOT / "cp2048_kl_r2.pt",
    "causal_r2": ROOT / "cp2048_causal_r2.pt",
    "causal_tensor_r2": ROOT / "cp2048_causal_tensor_r2.pt",
}
DISPLAY = {
    "direct_cp": "Direct CP",
    "activation_cp": "Activation CP",
    "kl_r2": "KL R2",
    "causal_r2": "Causal R2",
    "causal_tensor_r2": "Causal+tensor R2",
}
SCALES = (0.02, 0.05, 0.10, 0.20, 0.25, 0.50)


class Agreement:
    def __init__(self):
        self.error = 0.0
        self.target = 0.0
        self.prediction = 0.0
        self.dot = 0.0
        self.per_fidelity = []
        self.per_cosine = []

    def add(self, target, prediction):
        target, prediction = target.float(), prediction.float()
        error = (prediction - target).square().sum(1)
        target_energy = target.square().sum(1).clamp_min(1e-20)
        prediction_energy = prediction.square().sum(1).clamp_min(1e-20)
        dot = (target * prediction).sum(1)
        self.error += float(error.sum())
        self.target += float(target_energy.sum())
        self.prediction += float(prediction_energy.sum())
        self.dot += float(dot.sum())
        self.per_fidelity.extend((1 - error / target_energy).cpu().tolist())
        self.per_cosine.extend((dot / torch.sqrt(target_energy * prediction_energy)).cpu().tolist())

    def result(self):
        return {
            "aggregate_fidelity": 1 - self.error / max(self.target, 1e-30),
            "aggregate_cosine": self.dot / math.sqrt(max(self.target * self.prediction, 1e-30)),
            "median_per_example_fidelity": float(np.median(self.per_fidelity)),
            "mean_per_example_fidelity": float(np.mean(self.per_fidelity)),
            "median_per_example_cosine": float(np.median(self.per_cosine)),
            "target_energy": self.target,
            "error_energy": self.error,
            "examples": len(self.per_fidelity),
        }


@torch.inference_mode()
def evaluate_split(models, x_cpu, teacher, bias, seed, batch_size=256):
    accumulators = {
        name: {
            direction: {metric: Agreement() for metric in
                        ("value", "tangent", "curvature", *[f"finite_{scale:.2f}" for scale in SCALES])}
            for direction in ("gaussian", "natural")}
        for name in models
    }
    identity_error = 0.0
    generator = torch.Generator(device="cuda").manual_seed(seed)
    left, right, down = teacher
    for start in range(0, len(x_cpu), batch_size):
        x = x_cpu[start:start + batch_size].cuda()
        gaussian = torch.randn(x.shape, generator=generator, device="cuda")
        gaussian = gaussian / gaussian.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        natural = x.roll(shifts=1, dims=0) - x
        teacher_value = teacher_bilinear(x, x, left, right, down) + bias
        for direction, v in (("gaussian", gaussian), ("natural", natural)):
            teacher_tangent = 2 * teacher_bilinear(x, v, left, right, down)
            teacher_curvature = teacher_bilinear(v, v, left, right, down)
            # Verify the exact quadratic identity on one representative scale.
            exact = (teacher_bilinear(x + 0.1 * v, x + 0.1 * v, left, right, down)
                     - teacher_bilinear(x, x, left, right, down))
            analytic = 0.1 * teacher_tangent + 0.01 * teacher_curvature
            identity_error = max(identity_error, float((exact - analytic).abs().max()))
            for name, model in models.items():
                prediction_value = model(x)
                prediction_tangent = 2 * model.bilinear(x, v)
                prediction_curvature = model.bilinear(v, v)
                rows = accumulators[name][direction]
                rows["value"].add(teacher_value, prediction_value)
                rows["tangent"].add(teacher_tangent, prediction_tangent)
                rows["curvature"].add(teacher_curvature, prediction_curvature)
                for scale in SCALES:
                    target = scale * teacher_tangent + scale * scale * teacher_curvature
                    prediction = scale * prediction_tangent + scale * scale * prediction_curvature
                    rows[f"finite_{scale:.2f}"].add(target, prediction)
    result = {
        name: {direction: {metric: accumulator.result()
                           for metric, accumulator in metrics.items()}
               for direction, metrics in directions.items()}
        for name, directions in accumulators.items()
    }
    return result, identity_error


def make_figure(result):
    names = tuple(MODELS)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for column, direction in enumerate(("gaussian", "natural")):
        ax = axes[0, column]
        x = np.arange(3); width = 0.15
        for index, name in enumerate(names):
            metrics = result["splits"]["fineweb_test"][name][direction]
            ax.bar(x + (index - 2) * width,
                   [metrics[key]["aggregate_fidelity"] for key in
                    ("value", "tangent", "curvature")], width,
                   color=colors[index], label=DISPLAY[name])
        ax.set_xticks(x, ("Value f(x)", "Tangent J(x)v", "Curvature B(v,v)"))
        ax.set_ylabel("Layer-write fidelity")
        ax.set_title(f"{direction.title()} directions")
        if column == 0:
            ax.legend(fontsize=7)
    for column, direction in enumerate(("gaussian", "natural")):
        ax = axes[1, column]
        for index, name in enumerate(names):
            metrics = result["splits"]["fineweb_test"][name][direction]
            ax.plot(SCALES,
                    [metrics[f"finite_{scale:.2f}"]["aggregate_fidelity"] for scale in SCALES],
                    marker="o", color=colors[index], label=DISPLAY[name])
        ax.set_xlabel("Intervention scale α")
        ax.set_ylabel("Layer-write effect fidelity")
        ax.set_title(f"Exact finite effect: αJv + α²B(v,v)")
    fig.tight_layout(); fig.savefig(FIGURE, dpi=180); plt.close(fig)


def write_report(result):
    fw = result["splits"]["fineweb_test"]
    lines = [
        "# Bilinear value–tangent–curvature ontology", "",
        "A bilinear layer has an exact local decomposition:", "",
        "`f(x + αv) - f(x) = α J_f(x)v + α² B_f(v,v)`.", "",
        "This separates three meanings of a faithful replacement: matching values on the language manifold, matching local derivatives, and matching global/off-manifold curvature.", "",
        "![Tangent ontology summary](tangent_ontology_summary.png)", "",
        "## FineWeb layer-write fidelity", "",
        "| Model | Value | Gaussian tangent | Gaussian curvature | Natural tangent | Natural curvature |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        lines.append(
            f"| {DISPLAY[name]} | {fw[name]['gaussian']['value']['aggregate_fidelity']:.3f} "
            f"| {fw[name]['gaussian']['tangent']['aggregate_fidelity']:.3f} "
            f"| {fw[name]['gaussian']['curvature']['aggregate_fidelity']:.3f} "
            f"| {fw[name]['natural']['tangent']['aggregate_fidelity']:.3f} "
            f"| {fw[name]['natural']['curvature']['aggregate_fidelity']:.3f} |"
        )
    lines += ["", "## Interpretation", "",
              "- **Value fidelity** asks whether the replacement behaves correctly at states language actually visits.",
              "- **Tangent fidelity** asks whether small causal changes are mapped in the same direction. It is the relevant layer-local object as intervention size tends to zero.",
              "- **Curvature fidelity** asks whether the quadratic response away from the base state is preserved. Isotropic tensor probes primarily measure this global operator.",
              "- **Finite-effect fidelity** mixes tangent and curvature with scale-dependent weights. Therefore a replacement can improve at larger edits while remaining locally wrong.",
              "- Downstream Fisher fidelity can still differ because later layers amplify, suppress, or identify different write directions.",
              "", "## Red-team limits", "",
              "- Gaussian and within-corpus donor directions are only two tangent distributions.",
              "- Layer-write agreement does not prove downstream causal agreement; the existing scorecard measures that separately.",
              "- All CP refinements inherit the teacher-selected initialization, so these metrics compare functions rather than establish a canonical factor basis.",
              f"- Maximum absolute error in the checked quadratic identity was `{result['maximum_quadratic_identity_error']:.3e}`.",
              "", "Raw results: `tangent_ontology_results.json`."]
    REPORT.write_text("\n".join(lines) + "\n")


def main():
    device = "cuda"
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    teacher = tuple(state[key].float().to(device) for key in (
        "transformer.h.1.mlp.Left.weight", "transformer.h.1.mlp.Right.weight",
        "transformer.h.1.mlp.Down.weight"))
    bias = state["transformer.h.1.mlp.Down_bias"].float().to(device)
    del state
    models = {name: load_cp(path, device) for name, path in MODELS.items()}
    archive = np.load(DATA)
    result = {"formula": "f(x + alpha v) - f(x) = alpha J_f(x)v + alpha^2 B_f(v,v)",
              "splits": {}, "maximum_quadratic_identity_error": 0.0,
              "examples_per_split": 8192}
    for index, split in enumerate(SPLITS):
        print("tangent ontology", split, flush=True)
        x = torch.from_numpy(archive[f"{split}__input1"][:8192].astype(np.float32))
        rows, identity_error = evaluate_split(models, x, teacher, bias, 91000 + index)
        result["splits"][split] = rows
        result["maximum_quadratic_identity_error"] = max(
            result["maximum_quadratic_identity_error"], identity_error)
    OUTPUT.write_text(json.dumps(result, indent=2))
    make_figure(result); write_report(result)
    print("wrote", OUTPUT, REPORT, FIGURE, flush=True)


if __name__ == "__main__":
    main()
