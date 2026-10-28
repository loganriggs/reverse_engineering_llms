#!/usr/bin/env python3
"""Iteration 1: activation-weighted refinement of direct CP/LL1 decompositions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from run_bilin18_tensor_decomposition import (CPBilinear, DATA, LL1Bilinear, ROOT,
                                               natural_metrics,
                                               tensor_probe_metrics)
from train_bilin18_sparse_codes import SPLITS


def load_model(name):
    payload = torch.load(ROOT / f"{name}.pt", map_location="cpu", weights_only=True)
    config = payload["configuration"]
    if config["type"] == "CPBilinear":
        model = CPBilinear(1152, config["rank"], 0)
    else:
        model = LL1Bilinear(1152, config["blocks"], config["block_rank"], 0)
    model.load_state_dict(payload["state_dict"])
    return model


def train_activation(model, x, h, down, bias, steps, seed):
    model.cuda().train(); down, bias = down.cuda(), bias.cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=5e-4, betas=(.9, .95), weight_decay=1e-6)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(1, steps + 1):
        index = torch.randint(len(x), (256,), generator=generator)
        bx, bh = x[index].cuda(), h[index].cuda()
        target = F.linear(bh, down, bias)
        prediction = model(bx)
        cosine = (1 - F.cosine_similarity(prediction, target, dim=1)).mean()
        lognorm = (prediction.norm(dim=1).clamp_min(1e-6).log() -
                   target.norm(dim=1).clamp_min(1e-6).log()).square().mean()
        relative = ((prediction - target).square().sum(1) /
                    target.square().sum(1).clamp_min(1e-8)).mean()
        loss = cosine + .2 * lognorm + .05 * relative
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach()))
        if step % 200 == 0:
            print(type(model).__name__, step, np.mean(losses[-100:]), flush=True)
    return losses


def save(model, name):
    if isinstance(model, CPBilinear):
        config = {"type": "CPBilinear", "rank": model.rank}
    else:
        config = {"type": "LL1Bilinear", "blocks": model.blocks,
                  "block_rank": model.block_rank}
    torch.save({"state_dict": model.cpu().state_dict(), "configuration": config},
               ROOT / f"{name}_activation.pt")


def main():
    started = time.time()
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    left = state["transformer.h.1.mlp.Left.weight"].float()
    right = state["transformer.h.1.mlp.Right.weight"].float()
    down = state["transformer.h.1.mlp.Down.weight"].float()
    bias = state["transformer.h.1.mlp.Down_bias"].float(); del state
    inputs, products = np.load(DATA / "layer1_inputs_large.npz"), \
        np.load(DATA / "natural_large.npz")
    x = {split: torch.from_numpy(inputs[f"{split}__input1"].astype(np.float32))
         for split in ("fineweb_train",) + SPLITS}
    h = {split: torch.from_numpy(products[f"{split}__h1"].astype(np.float32))
         for split in ("fineweb_train",) + SPLITS}
    result = {"models": {}}
    for index, name in enumerate(("cp_r512", "cp_r2048", "ll1_r256_l8")):
        model = load_model(name)
        before_tensor = tensor_probe_metrics(model, left, right, down, 30000 + index)
        losses = train_activation(model, x["fineweb_train"], h["fineweb_train"],
                                  down, bias, 1600, 40000 + index)
        row = {"before_tensor": before_tensor,
               "after_tensor": tensor_probe_metrics(model, left, right, down,
                                                     50000 + index),
               "final_loss": float(np.mean(losses[-100:])), "natural": {}}
        for split in SPLITS:
            row["natural"][split] = natural_metrics(model, x[split], h[split], down, bias)
        result["models"][name] = row
        save(model, name)
    result["elapsed_seconds"] = time.time() - started
    (ROOT / "activation_refinement_results.json").write_text(json.dumps(result, indent=2))
    print(f"wrote activation refinement in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
