#!/usr/bin/env python3
"""Audit cancellation and redundancy between paired square replacement units."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_final_mlp_compression import final_inputs
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from train_bilin18_final_mlp_architectures import NarrowSquare


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = Path("artifacts/bilin18_final_mlp_architectures/square_k128.pt")
    output = Path("artifacts/bilin18_final_mlp_architectures/square_k128_pairs.json")
    corpus = np.load(DEFAULT_CORPUS, mmap_mode="r")
    model = load_full_model(device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    student = NarrowSquare(model.config.n_embd, 128).to(device)
    student.load_state_dict(saved["state_dict"])
    reader = F.normalize(student.reader, dim=1)
    values = []
    for start in range(14000, 14400, 4):
        ids = torch.as_tensor(np.asarray(corpus[start:start + 4, :128], dtype=np.int64), device=device)
        pre, _ = final_inputs(model, ids)
        x = F.rms_norm(pre, (model.config.n_embd,)).float()
        values.append(F.linear(x, reader).square().flatten(0, 1).cpu())
    activation = torch.cat(values).float()
    first, second = activation[:, :64], activation[:, 64:]
    first_centered = first - first.mean(0)
    second_centered = second - second.mean(0)
    correlation = (first_centered * second_centered).mean(0) / (
        first_centered.std(0, correction=0) * second_centered.std(0, correction=0)).clamp_min(1e-8)
    response = (model.lm_head.weight.float() @ student.down).cpu()
    a, b = response[:, :64], response[:, 64:]
    output_cosine = (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp_min(1e-8)
    aa, bb, ab = a.square().sum(0), b.square().sum(0), (a * b).sum(0)
    pair_norm = (first.square() * aa + second.square() * bb + 2 * first * second * ab).clamp_min(0).sqrt()
    separate_norm = first * aa.sqrt() + second * bb.sqrt()
    cancellation = 1 - pair_norm / separate_norm.clamp_min(1e-8)
    rows = []
    for pair in range(64):
        rows.append({
            "pair": pair, "units": [pair, pair + 64],
            "activation_correlation": float(correlation[pair]),
            "output_direction_cosine": float(output_cosine[pair]),
            "mean_cancellation_fraction": float(cancellation[:, pair].mean()),
        })
    result = {
        "examples": len(activation),
        "pair_activation_correlation_median": float(correlation.median()),
        "pair_activation_correlation_q10": float(correlation.quantile(.1)),
        "pair_activation_correlation_q90": float(correlation.quantile(.9)),
        "pair_output_cosine_median": float(output_cosine.median()),
        "pair_output_cosine_q10": float(output_cosine.quantile(.1)),
        "pair_output_cosine_q90": float(output_cosine.quantile(.9)),
        "mean_pair_cancellation_fraction": float(cancellation.mean()),
        "pairs": rows,
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))
    print("highest cancellation", sorted(rows, key=lambda x: x["mean_cancellation_fraction"], reverse=True)[:8])


if __name__ == "__main__":
    main()
