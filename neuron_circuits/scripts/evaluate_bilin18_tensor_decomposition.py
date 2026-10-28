#!/usr/bin/env python3
"""Evaluate CP, LL1, and routed tensor decompositions inside the full model."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from collect_bilin18_sparse_code_data import OOD
from refine_bilin18_tensor_decomposition import load_model
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import CPBilinear, ROOT


def load_activation(name):
    payload = torch.load(ROOT / f"{name}_activation.pt", map_location="cpu",
                         weights_only=True)
    model = load_model(name); model.load_state_dict(payload["state_dict"])
    return model


def load_routed(name):
    payload = torch.load(ROOT / f"{name}.pt", map_location="cpu", weights_only=True)
    model = CPBilinear(1152, payload["configuration"]["rank"], 0)
    model.load_state_dict(payload["state_dict"])
    return model


@torch.no_grad()
def forward(model, ids, replacement=None, routed=False):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,)); x0, value_bus = x, None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention; normalized = F.rms_norm(pre, (dimension,))
        if layer == 1 and replacement is not None:
            flat = normalized.float().flatten(0, 1)
            if routed:
                write, _ = replacement.forward_sparse(flat, 128)
            else:
                write = replacement(flat)
            write = write.view_as(normalized).to(normalized.dtype)
        else:
            product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
            write = block.mlp.Down(product) + block.mlp.Down_bias
        x = pre + write
    logits = model.lm_head(F.rms_norm(x, (dimension,))).float()
    return 30 * torch.tanh(logits / 30)


@torch.no_grad()
def evaluate(model, replacements, rows, batch_size=2):
    totals = {name: {"nll": 0., "kl": 0., "same": 0, "n": 0}
              for name in replacements}
    baseline_nll, baseline_n = 0., 0
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        targets = torch.as_tensor(raw[:, 1:129], device="cuda")
        base = forward(model, ids); base_logp = F.log_softmax(base, -1); base_p = base_logp.exp()
        baseline_nll += float(F.nll_loss(base_logp.flatten(0, 1), targets.flatten(),
                                         reduction="sum"))
        baseline_n += targets.numel(); base_top = base.argmax(-1)
        for name, (replacement, routed) in replacements.items():
            logits = forward(model, ids, replacement, routed)
            logp = F.log_softmax(logits, -1); row = totals[name]
            row["nll"] += float(F.nll_loss(logp.flatten(0, 1), targets.flatten(),
                                           reduction="sum"))
            row["kl"] += float((base_p * (base_logp - logp)).sum())
            row["same"] += int((base_top == logits.argmax(-1)).sum())
            row["n"] += targets.numel()
    base_ce = baseline_nll / baseline_n
    return {"baseline_cross_entropy": base_ce, "replacements": {
        name: {"cross_entropy_delta": row["nll"] / row["n"] - base_ce,
               "mean_kl_from_full": row["kl"] / row["n"],
               "top1_agreement": row["same"] / row["n"],
               "perplexity_ratio": math.exp(min(20, row["nll"] / row["n"] - base_ce))}
        for name, row in totals.items()}}


def main():
    started = time.time()
    replacements = {
        "cp512_weight": (load_model("cp_r512").cuda(), False),
        "cp2048_weight": (load_model("cp_r2048").cuda(), False),
        "ll1_weight": (load_model("ll1_r256_l8").cuda(), False),
        "cp512_activation": (load_activation("cp_r512").cuda(), False),
        "cp2048_activation": (load_activation("cp_r2048").cuda(), False),
        "ll1_activation": (load_activation("ll1_r256_l8").cuda(), False),
        "routed_behavior": (load_routed("routed_behavior").cuda(), True),
        "routed_mixed": (load_routed("routed_mixed").cuda(), True),
    }
    model = load_full_model("cuda")
    fineweb = np.load(DEFAULT_CORPUS, mmap_mode="r")
    corpora = {
        "fineweb": fineweb[18000:18064],
        "wikitext103": np.load(Path("artifacts/bilin18_ood_corpora") /
                                "wikitext103_1200x129.npy", mmap_mode="r")[1024:1088],
        "tinystories": np.load(Path("artifacts/bilin18_ood_corpora") /
                                "tinystories_1200x129.npy", mmap_mode="r")[1024:1088],
    }
    result = {"corpora": {}}
    for name, rows in corpora.items():
        print("evaluating", name, flush=True)
        result["corpora"][name] = evaluate(model, replacements, rows)
    result["tokens_per_corpus"] = 8192
    result["elapsed_seconds"] = time.time() - started
    (ROOT / "full_model_results.json").write_text(json.dumps(result, indent=2))
    print(f"wrote full-model tensor results in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
