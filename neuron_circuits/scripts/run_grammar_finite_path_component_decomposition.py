#!/usr/bin/env python3
"""Integrate exact QK/value/bilinear tangent paths over finite activation edits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuron_circuits.grammar_data import VOCAB, encode_prompts, labels_for_rho, load_manifest
from neuron_circuits.grammar_model import ActivationPatch
from run_grammar_exact_factor_chain import local_block_components, rms_jacobian
from run_grammar_final_interface_redteam import load_model
from run_grammar_jacobian_regime_sweep import correct


CONFIG = Path("configs/grammar_finite_path_component_decomposition.json")
OUTPUT = Path("artifacts/grammar_regimes/grammar_finite_path_component_decomposition.json")
REPORT = Path("artifacts/grammar_regimes/GRAMMAR_FINITE_PATH_COMPONENT_DECOMPOSITION.md")
TERMS = ("I", "Q", "V", "ML", "MR", "MLQ", "MLV", "MRQ", "MRV")


@torch.no_grad()
def centered_answers(model, tokens, patch=None):
    logits = model(tokens, patch=patch)[:, -1, list(VOCAB.answers)].float()
    return logits - logits.mean(1, keepdim=True)


@torch.no_grad()
def path_derivative_terms(model, prompt, delta, position, device):
    tokens = torch.as_tensor(prompt.astype(np.int64), device=device)[None]
    hidden = model.embedding(tokens)[0] + model.position[:tokens.shape[1]]
    hidden, _ = model.blocks[0](hidden[None]); hidden = hidden[0].clone()
    hidden[-1] += model.blocks[0].down(delta * position)
    hidden, first = local_block_components(model.blocks[1], hidden)
    hidden, second = local_block_components(model.blocks[2], hidden)
    hidden, third = local_block_components(model.blocks[3], hidden)
    answer_weight = model.unembedding.weight[list(VOCAB.answers)]
    center = torch.eye(8, device=device) - torch.ones(8, 8, device=device) / 8
    readout = center @ answer_weight @ rms_jacobian(hidden[-1], model.final_norm.weight, model.final_norm.eps)
    tangent = second["A"] @ first["A"] @ model.blocks[0].down.weight @ delta
    matrices = {
        "I": torch.eye(model.width, device=device), "Q": third["Q"], "V": third["V"],
        "ML": third["ML"], "MR": third["MR"],
        "MLQ": third["ML"] @ third["Q"], "MLV": third["ML"] @ third["V"],
        "MRQ": third["MR"] @ third["Q"], "MRV": third["MR"] @ third["V"],
    }
    return {name: readout @ matrix @ tangent for name, matrix in matrices.items()}


def fidelity(actual, predicted):
    return 1 - float((actual - predicted).square().sum() / actual.square().sum().clamp_min(1e-30))


def audit(spec, rho, prompts, config, device):
    root = Path(spec["root"]); model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
    tokens = torch.as_tensor(prompts.astype(np.int64), device=device)
    with torch.no_grad():
        base, activations = model(tokens, return_activations=True)
        base = base[:, -1, list(VOCAB.answers)].float(); base -= base.mean(1, keepdim=True)
    generator = torch.Generator(device="cpu").manual_seed(config["seed"] + spec["seed"] * 100 + round(rho * 10))
    direction = torch.randn(activations[0].shape, generator=generator)
    direction /= direction.norm(dim=1, keepdim=True).clamp_min(1e-30)
    base_size = activations[0].cpu().norm(dim=1, keepdim=True)
    nodes, weights = np.polynomial.legendre.leggauss(config["quadrature_points"])
    nodes, weights = (nodes + 1) / 2, weights / 2
    scales = {}
    for scale in config["relative_activation_norms"]:
        delta = direction * base_size * scale; delta_device = delta.to(device)
        changed = centered_answers(model, tokens, ActivationPatch(0, delta_device, additive=True))
        actual = changed - base
        integrated = {name: torch.zeros_like(actual) for name in TERMS}
        for row, prompt in enumerate(prompts):
            for position, weight in zip(nodes, weights):
                terms = path_derivative_terms(model, prompt, delta_device[row], float(position), device)
                for name in TERMS: integrated[name][row] += float(weight) * terms[name]
        predicted = sum(integrated.values())
        denominator = actual.square().sum().clamp_min(1e-30)
        projection = {name: float((actual * value).sum() / denominator) for name, value in integrated.items()}
        norm_ratio = {name: float(value.square().sum().sqrt() / denominator.sqrt()) for name, value in integrated.items()}
        grouped_values = {
            "identity": integrated["I"],
            "qk_direct": integrated["Q"], "value_direct": integrated["V"],
            "bilinear_direct": integrated["ML"] + integrated["MR"],
            "interactions": integrated["MLQ"] + integrated["MLV"] + integrated["MRQ"] + integrated["MRV"],
        }
        grouped_projection = {name: float((actual * value).sum() / denominator)
                              for name, value in grouped_values.items()}
        grouped_norm = {name: float(value.square().sum().sqrt() / denominator.sqrt())
                        for name, value in grouped_values.items()}
        scales[str(scale)] = {"fidelity": fidelity(actual, predicted),
                              "relative_error": float((actual - predicted).norm() / actual.norm().clamp_min(1e-30)),
                              "projection": projection, "norm_ratio": norm_ratio,
                              "grouped_projection": grouped_projection, "grouped_norm_ratio": grouped_norm,
                              "projection_sum": float(sum(projection.values()))}
    return {"seed": spec["seed"], "rho": rho, "sites": len(prompts), "scales": scales}


def main():
    config = json.loads(CONFIG.read_text()); device = "cuda" if torch.cuda.is_available() else "cpu"; runs = []
    for spec in config["roots"]:
        root = Path(spec["root"]); manifest = load_manifest(root / "manifest.npz")
        facts = np.asarray(manifest["train_facts"]); rule = labels_for_rho(manifest, "train", 0.0)[0]
        prompt = encode_prompts(manifest, facts, config["template"]); eligible = []
        for rho in config["rhos"]:
            model = load_model(root / f"rho_{rho:g}" / "depth_4" / "checkpoint.pt", device)
            labels = labels_for_rho(manifest, "train", rho)[0]; eligible.append(correct(model, prompt, labels, device))
        common = np.logical_and.reduce(eligible); rng = np.random.default_rng(config["seed"] + spec["seed"])
        selected = []
        for relation in range(8):
            for answer in range(8):
                candidates = np.flatnonzero(common & (facts[:, 2] == relation) & (rule == answer)); rng.shuffle(candidates)
                selected.extend(candidates[:config["examples_per_relation_answer"]])
        prompts = prompt[np.asarray(selected)]
        for rho in config["rhos"]:
            print("audit", spec["seed"], rho, flush=True); runs.append(audit(spec, rho, prompts, config, device))
    mixed = [row for row in runs if row["rho"] == .5]
    scale_keys = [str(value) for value in config["relative_activation_norms"]]
    i0 = all(row["scales"][scale]["fidelity"] >= .9999 for row in runs for scale in scale_keys)
    i1 = all(row["scales"][scale]["grouped_norm_ratio"]["bilinear_direct"]
             > row["scales"][scale]["grouped_norm_ratio"]["qk_direct"] for row in mixed for scale in scale_keys)
    i2 = all(row["scales"][scale]["grouped_norm_ratio"]["qk_direct"] >= .05 for row in mixed for scale in scale_keys)
    i3 = all(row["scales"][scale]["grouped_norm_ratio"]["value_direct"] <= .02 for row in runs for scale in scale_keys)
    i4 = all(row["scales"][scale]["grouped_norm_ratio"]["interactions"] >= .05 for row in mixed for scale in scale_keys)
    groups = ("identity", "qk_direct", "value_direct", "bilinear_direct", "interactions")
    i5 = all(max(abs(row["scales"][scale_keys[0]]["grouped_projection"][group]
                         - row["scales"][scale_keys[1]]["grouped_projection"][group]) for group in groups) <= .15
             for row in runs)
    hypotheses = {name: {"supported": bool(value)} for name, value in zip(config["hypotheses"], (i0, i1, i2, i3, i4, i5))}
    OUTPUT.write_text(json.dumps({"config": config, "hypotheses": hypotheses, "runs": runs}, indent=2, allow_nan=False))
    lines = ["# Finite path decomposition of QK/value/bilinear effects", "",
             "A real additive Layer-0 activation edit is followed from zero to full dose. Sixteen-point Gauss–Legendre quadrature integrates the exact analytic derivative. The final block expands into identity, Q, V, ML, MR, and four bilinear-by-attention interaction paths; their sum is the complete finite answer-logit change.", "", "## Locked hypotheses", ""]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += ["", "| Seed | rho | Scale | Fidelity | Projection I/Q/V/M/cross | Norm I/Q/V/M/cross |", "|---:|---:|---:|---:|---:|---:|"]
    for row in runs:
        for scale in scale_keys:
            result = row["scales"][scale]; p, n = result["grouped_projection"], result["grouped_norm_ratio"]
            lines.append(f"| {row['seed']} | {row['rho']} | {scale} | {result['fidelity']:.6f} | "
                         f"{p['identity']:.3f}/{p['qk_direct']:.3f}/{p['value_direct']:.3f}/{p['bilinear_direct']:.3f}/{p['interactions']:.3f} | "
                         f"{n['identity']:.3f}/{n['qk_direct']:.3f}/{n['value_direct']:.3f}/{n['bilinear_direct']:.3f}/{n['interactions']:.3f} |")
    lines += ["", "Signed projection measures alignment with the realized finite effect and sums to one when quadrature is exact. Norm ratios expose large cancelling or orthogonal paths. This is a path-integrated causal attribution of an actual intervention, not a claim that each algebraic term can be independently executed as a model module."]
    REPORT.write_text("\n".join(lines) + "\n"); print("wrote", REPORT, flush=True)


if __name__ == "__main__":
    main()
