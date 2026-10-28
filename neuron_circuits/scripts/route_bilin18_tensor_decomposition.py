#!/usr/bin/env python3
"""Iteration 2: sparse routing over explicit CP tensor factors."""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from refine_bilin18_tensor_decomposition import load_model
from run_bilin18_tensor_decomposition import DATA, ROOT, teacher_bilinear
from train_bilin18_sparse_codes import SPLITS, fidelity


K = 128


def natural_loss(prediction, target):
    cosine = (1 - F.cosine_similarity(prediction, target, dim=1)).mean()
    lognorm = (prediction.norm(dim=1).clamp_min(1e-6).log() -
               target.norm(dim=1).clamp_min(1e-6).log()).square().mean()
    relative = ((prediction - target).square().sum(1) /
                target.square().sum(1).clamp_min(1e-8)).mean()
    return cosine + .2 * lognorm + .05 * relative


def train(model, x, h, left, right, down, bias, steps, seed, tensor_weight):
    model.cuda().train(); left, right, down, bias = [v.cuda() for v in
                                                    (left, right, down, bias)]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=4e-4, betas=(.9, .95), weight_decay=1e-6)
    cpu_generator = torch.Generator().manual_seed(seed)
    gpu_generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    losses = []
    for step in range(1, steps + 1):
        index = torch.randint(len(x), (256,), generator=cpu_generator)
        bx, bh = x[index].cuda(), h[index].cuda()
        target = F.linear(bh, down, bias)
        prediction, _ = model.forward_sparse(bx, K)
        loss = natural_loss(prediction, target)
        if tensor_weight:
            gx = torch.randn(128, 1152, generator=gpu_generator, device="cuda")
            gy = torch.randn(128, 1152, generator=gpu_generator, device="cuda")
            with torch.no_grad():
                tensor_target = teacher_bilinear(gx, gy, left, right, down)
            tensor_prediction, _ = model.bilinear_sparse(gx, gy, K)
            tensor_loss = ((tensor_prediction - tensor_target).square().sum() /
                           tensor_target.square().sum().clamp_min(1e-8))
            loss = loss + tensor_weight * tensor_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach()))
        if step % 200 == 0:
            print("routed", tensor_weight, step, np.mean(losses[-100:]), flush=True)
    return losses


@torch.no_grad()
def evaluate_natural(model, x, h, down, bias, batch_size=256):
    model.cuda().eval(); down, bias = down.cuda(), bias.cuda()
    targets, predictions = [], []
    counts = torch.zeros(model.rank, dtype=torch.float64)
    for start in range(0, len(x), batch_size):
        bx, bh = x[start:start + batch_size].cuda(), h[start:start + batch_size].cuda()
        target = F.linear(bh, down, bias)
        prediction, code = model.forward_sparse(bx, K)
        targets.append(target.cpu()); predictions.append(prediction.cpu())
        counts += (code != 0).sum(0).cpu()
    total = counts.sum()
    return fidelity(torch.cat(targets), torch.cat(predictions)), {
        "features_ever_used": int((counts > 0).sum()),
        "effective_support": float(total.square() / counts.square().sum().clamp_min(1e-20)),
        "top_32_usage_fraction": float(counts.topk(32).values.sum() / total),
    }


@torch.no_grad()
def evaluate_tensor(model, left, right, down, seed, batches=16):
    model.cuda().eval(); left, right, down = left.cuda(), right.cuda(), down.cuda()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    targets, predictions = [], []
    for _ in range(batches):
        x = torch.randn(128, 1152, generator=generator, device="cuda")
        y = torch.randn(128, 1152, generator=generator, device="cuda")
        targets.append(teacher_bilinear(x, y, left, right, down).cpu())
        prediction, _ = model.bilinear_sparse(x, y, K)
        predictions.append(prediction.cpu())
    return fidelity(torch.cat(targets), torch.cat(predictions))


def load_activation_cp():
    payload = torch.load(ROOT / "cp_r2048_activation.pt", map_location="cpu",
                         weights_only=True)
    model = load_model("cp_r2048")
    model.load_state_dict(payload["state_dict"])
    return model


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
    configurations = {
        "routed_behavior": (load_activation_cp(), 0.0),
        "routed_mixed": (load_model("cp_r2048"), .25),
    }
    result = {"models": {}, "active_features": K}
    for index, (name, (model, tensor_weight)) in enumerate(configurations.items()):
        before_tensor = evaluate_tensor(model, left, right, down, 60000 + index)
        losses = train(model, x["fineweb_train"], h["fineweb_train"], left, right,
                       down, bias, 1600, 61000 + index, tensor_weight)
        row = {"tensor_weight": tensor_weight, "before_tensor": before_tensor,
               "after_tensor": evaluate_tensor(model, left, right, down,
                                                62000 + index),
               "final_loss": float(np.mean(losses[-100:])), "natural": {},
               "usage": {}}
        for split in SPLITS:
            row["natural"][split], usage = evaluate_natural(
                model, x[split], h[split], down, bias)
            row["usage"][split] = usage
        result["models"][name] = row
        torch.save({"state_dict": model.cpu().state_dict(),
                    "configuration": {"type": "CPBilinear", "rank": model.rank,
                                      "k": K}}, ROOT / f"{name}.pt")
    result["elapsed_seconds"] = time.time() - started
    (ROOT / "sparse_routing_results.json").write_text(json.dumps(result, indent=2))
    print(f"wrote sparse routing iteration in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
