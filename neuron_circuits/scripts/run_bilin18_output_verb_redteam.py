#!/usr/bin/env python3
"""Matched-null red teams for recurring bus-aware output verbs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import load_full_model
from run_bilin18_spectral_spike_pilot import TENSOR_ROOT


CONFIG = Path("configs/bilin18_output_verb_redteam.json")
SOURCE = TENSOR_ROOT / "bilin18_bus_aware_function_templates.json"
OUTPUT = TENSOR_ROOT / "bilin18_output_verb_redteam.json"
REPORT = TENSOR_ROOT / "BILIN18_OUTPUT_VERB_REDTEAM.md"


def median_nearest(vectors):
    vectors = F.normalize(vectors.float(), dim=1)
    cosine = (vectors @ vectors.T).abs()
    cosine.fill_diagonal_(-1)
    return float(cosine.max(1).values.median()), cosine


def leading_from_qr(qo, ro, rb):
    left, _, _ = torch.linalg.svd(ro @ rb.T, full_matrices=False)
    return qo @ left[:, 0]


def main():
    config = json.loads(CONFIG.read_text())
    source = json.loads(SOURCE.read_text())
    model = load_full_model("cuda")
    model.blocks.float().eval()
    for parameter in model.blocks.parameters(): parameter.requires_grad_(False)
    heads = model.config.n_head; dimension = model.config.n_embd; head_dim = dimension // heads
    q_outputs = []; r_outputs = []; r_inputs = []; combined_modes = []; cproj_modes = []; labels = []
    covariance = torch.zeros(dimension, dimension, device="cuda")
    for layer in config["layers"]:
        block = model.blocks[layer].attn; lamb = float(block.lamb.detach())
        for head in range(heads):
            sl = slice(head * head_dim, (head + 1) * head_dim)
            output = block.c_proj.weight[:, sl].float()
            value = block.c_v.weight[sl].float()
            selector = torch.zeros(head_dim, dimension, device="cuda"); selector[:, sl] = torch.eye(head_dim, device="cuda")
            combined = torch.cat(((1 - lamb) * value, lamb * selector), 1)
            qo, ro = torch.linalg.qr(output, mode="reduced")
            _, rb = torch.linalg.qr(combined.T, mode="reduced")
            combined_modes.append(leading_from_qr(qo, ro, rb))
            cproj_left, _, _ = torch.linalg.svd(ro, full_matrices=False)
            cproj_modes.append(qo @ cproj_left[:, 0])
            q_outputs.append(qo); r_outputs.append(ro); r_inputs.append(rb)
            covariance += output @ (combined @ combined.T) @ output.T
            labels.append(f"L{layer}H{head}")
    combined_modes = torch.stack(combined_modes)
    cproj_modes = torch.stack(cproj_modes)
    observed, observed_cosine = median_nearest(combined_modes)
    cproj_alignment = (F.normalize(combined_modes, dim=1) * F.normalize(cproj_modes, dim=1)).sum(1).abs()
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    subspace_null = []
    for _ in range(config["null_draws"]):
        vectors = []
        for qo in q_outputs:
            coefficient = torch.randn(head_dim, generator=generator, device="cuda")
            vectors.append(qo @ coefficient)
        subspace_null.append(median_nearest(torch.stack(vectors))[0])
    rng = np.random.default_rng(config["seed"])
    pairing_null = []
    for _ in range(config["pairing_permutations"]):
        permutation = rng.permutation(len(labels))
        modes = [leading_from_qr(q_outputs[index], r_outputs[index], r_inputs[int(permutation[index])]) for index in range(len(labels))]
        pairing_null.append(median_nearest(torch.stack(modes))[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = eigenvalues.median().clamp_min(1e-12) * 1e-4
    inverse_root = eigenvectors @ torch.diag(eigenvalues.clamp_min(floor).rsqrt()) @ eigenvectors.T
    whitened = combined_modes @ inverse_root
    whitened_nearest, whitened_cosine = median_nearest(whitened)
    haar_null = []
    for _ in range(config["null_draws"]):
        vectors = torch.randn(len(labels), dimension, generator=generator, device="cuda")
        haar_null.append(median_nearest(vectors)[0])
    subspace_p95 = float(np.quantile(subspace_null, .95)); pairing_p95 = float(np.quantile(pairing_null, .95)); haar_p95 = float(np.quantile(haar_null, .95))
    median_alignment = float(cproj_alignment.median())
    hypotheses = {
        "R1_verbs_exceed_output_subspace_null": {"supported": observed >= subspace_p95 + .03, "observed": observed, "null_p95": subspace_p95, "difference": observed - subspace_p95},
        "R2_verbs_depend_on_learned_input_output_pairing": {"supported": observed >= pairing_p95 + .03, "observed": observed, "null_p95": pairing_p95, "difference": observed - pairing_p95},
        "R3_verbs_survive_global_output_whitening": {"supported": whitened_nearest >= haar_p95 + .05, "whitened": whitened_nearest, "haar_p95": haar_p95, "difference": whitened_nearest - haar_p95},
        "R4_verbs_are_not_just_top_cproj_directions": {"supported": median_alignment < .80, "median_alignment": median_alignment},
    }
    for row in hypotheses.values(): row["supported"] = bool(row["supported"])
    pairs = []
    for index in range(len(labels)):
        values = observed_cosine[index].clone(); values[index] = -1
        neighbor = int(values.argmax())
        if index < neighbor:
            overlap = float(torch.square(q_outputs[index].T @ q_outputs[neighbor]).sum() / head_dim)
            pairs.append({"first": labels[index], "second": labels[neighbor], "verb_cosine": float(values[neighbor]), "whitened_cosine": float(whitened_cosine[index, neighbor]), "output_subspace_overlap": overlap})
    pairs.sort(key=lambda row: row["verb_cosine"], reverse=True)
    payload = {
        "config": config, "hypotheses": hypotheses,
        "observed_median_nearest": observed,
        "subspace_null": {"median": float(np.median(subspace_null)), "p95": subspace_p95},
        "pairing_null": {"median": float(np.median(pairing_null)), "p95": pairing_p95},
        "whitened": {"observed": whitened_nearest, "haar_median": float(np.median(haar_null)), "haar_p95": haar_p95},
        "cproj_alignment": {"median": median_alignment, "values": dict(zip(labels, cproj_alignment.cpu().tolist()))},
        "nearest_pairs": pairs,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False))
    lines = [
        "# Matched-null red team for recurring output verbs", "",
        "The bus-aware atlas found leading output directions much closer than Haar-random vectors. This post-result red team asks whether that recurrence survives three stronger controls: random directions inside each native c_proj output subspace, permutation of learned input maps across output projections, and whitening by the aggregate output-operator covariance.", "",
        "## Locked hypotheses", "",
    ]
    for name, row in hypotheses.items(): lines.append(f"- **{name}: {'supported' if row['supported'] else 'not supported'}.**")
    lines += [
        "", "## Results", "",
        f"- Observed median nearest verb cosine: `{observed:.3f}`.",
        f"- Native-output-subspace null median/p95: `{np.median(subspace_null):.3f}/{subspace_p95:.3f}`.",
        f"- Input/output-pairing null median/p95: `{np.median(pairing_null):.3f}/{pairing_p95:.3f}`.",
        f"- Whitened observed versus Haar p95: `{whitened_nearest:.3f}/{haar_p95:.3f}`.",
        f"- Combined-operator versus c_proj-only leading-direction alignment: median `{median_alignment:.3f}`.",
        "", "Strongest unique pairs:", "",
    ]
    for row in pairs[:12]: lines.append(f"- {row['first']} ↔ {row['second']}: verb `{row['verb_cosine']:.3f}`, whitened `{row['whitened_cosine']:.3f}`, normalized output-subspace overlap `{row['output_subspace_overlap']:.3f}`")
    lines += [
        "", "## Guardrails", "",
        "These are post-result controls and therefore diagnose, rather than independently confirm, the original recurrence claim. Output-subspace overlap is not itself a semantic function. Whitening can erase a genuinely reused direction if global reuse is exactly what created the anisotropy; its purpose is to distinguish a discrete family from one broad shared residual geometry. Only a pair that survives matched controls should advance to finite causal replacement.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"hypotheses": hypotheses}, indent=2)); print(REPORT)


if __name__ == "__main__":
    main()
