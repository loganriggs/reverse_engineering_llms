#!/usr/bin/env python3
"""Train a bilinear SAE and sparse bilinear replacements for Bilin18 layer 1."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from train_bilin18_sparse_codes import SPLITS, fidelity


ROOT = Path("artifacts/bilin18_bilinear_codes")
DATA = Path("artifacts/bilin18_sparse_codes")


class SparseBilinearDictionary(nn.Module):
    """Normalized bilinear features with an explicit, identifiable activation gain."""

    def __init__(self, input_dim, features, output_dim, k, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.left = nn.Parameter(torch.randn(features, input_dim) / math.sqrt(input_dim))
        self.right = nn.Parameter(torch.randn(features, input_dim) / math.sqrt(input_dim))
        self.log_gain = nn.Parameter(torch.zeros(features))
        self.decoder = nn.Parameter(torch.randn(output_dim, features) / math.sqrt(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.k = k

    def balanced(self):
        left = F.normalize(self.left, dim=1)
        right = F.normalize(self.right, dim=1)
        decoder = F.normalize(self.decoder, dim=0)
        gain = self.log_gain.clamp(-8, 8).exp()
        return left, right, gain, decoder

    def encode(self, x, k=None):
        left, right, gain, _ = self.balanced()
        values = (F.linear(x, left) * F.linear(x, right)) * gain
        keep = min(self.k if k is None else k, values.shape[-1])
        if keep < values.shape[-1]:
            top = torch.topk(values.abs(), keep, dim=-1)
            code = torch.zeros_like(values)
            code.scatter_(1, top.indices, values.gather(1, top.indices))
            return code
        return values

    def forward(self, x, k=None):
        code = self.encode(x, k)
        decoder = F.normalize(self.decoder, dim=0)
        return F.linear(code, decoder, self.bias), code


def init_from_teacher(module, teacher_left, teacher_right, teacher_down, units,
                      output_std=None):
    """Initialize selected features as exact native bilinear channels."""
    with torch.no_grad():
        left = teacher_left[units]
        right = teacher_right[units]
        left_norm = left.norm(dim=1).clamp_min(1e-8)
        right_norm = right.norm(dim=1).clamp_min(1e-8)
        module.left.copy_(left / left_norm[:, None])
        module.right.copy_(right / right_norm[:, None])
        if output_std is None:
            decoder = teacher_down[:, units]
        else:
            decoder = torch.zeros_like(module.decoder)
            decoder[units, torch.arange(len(units), device=units.device)] = 1 / output_std[units]
        decoder_norm = decoder.norm(dim=0).clamp_min(1e-8)
        module.decoder.copy_(decoder / decoder_norm)
        module.log_gain.copy_((left_norm * right_norm * decoder_norm).log())


def batch_indices(size, batch_size, generator):
    return torch.randint(size, (batch_size,), generator=generator)


def train_sae(model, train_x, train_target, steps, seed):
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, betas=(.9, .95),
                                  weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(1, steps + 1):
        index = batch_indices(len(train_x), 256, generator)
        x = train_x[index].cuda(); target = train_target[index].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _ = model(x)
            loss = F.mse_loss(prediction.float(), target)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach()))
        if step % 200 == 0:
            print("bilinear_sae", seed, step, np.mean(losses[-100:]), flush=True)
    return losses


def train_replacement(model, train_x, train_h, teacher_down, teacher_bias, steps, seed):
    model.cuda().train(); teacher_down = teacher_down.cuda(); teacher_bias = teacher_bias.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, betas=(.9, .95),
                                  weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(1, steps + 1):
        index = batch_indices(len(train_x), 256, generator)
        x = train_x[index].cuda(); h = train_h[index].cuda()
        target = F.linear(h, teacher_down, teacher_bias)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _ = model(x)
        prediction = prediction.float()
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
            print("replacement", len(model.left), model.k, seed, step,
                  np.mean(losses[-100:]), flush=True)
    return losses


@torch.inference_mode()
def evaluate_sae(model, x, target, batch_size=256):
    predictions, codes = [], []
    model.cuda().eval()
    for start in range(0, len(x), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, code = model(x[start:start + batch_size].cuda())
        predictions.append(prediction.float().cpu())
        codes.append(code.to(torch.float16).cpu())
    prediction = torch.cat(predictions)
    return fidelity(target, prediction), torch.cat(codes)


@torch.inference_mode()
def evaluate_replacement(model, x, h, teacher_down, teacher_bias, batch_size=256):
    target_rows, prediction_rows = [], []
    model.cuda().eval(); teacher_down = teacher_down.cuda(); teacher_bias = teacher_bias.cuda()
    for start in range(0, len(x), batch_size):
        bx = x[start:start + batch_size].cuda(); bh = h[start:start + batch_size].cuda()
        target = F.linear(bh, teacher_down, teacher_bias)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _ = model(bx)
        target_rows.append(target.cpu()); prediction_rows.append(prediction.float().cpu())
    return fidelity(torch.cat(target_rows), torch.cat(prediction_rows))


def bilinear_feature_stability(first, second, chunk=128):
    """Best symmetric rank-one-form cosine, invariant to swap and sign."""
    a = F.normalize(first.left.detach().float().cpu(), dim=1)
    b = F.normalize(first.right.detach().float().cpu(), dim=1)
    c = F.normalize(second.left.detach().float().cpu(), dim=1)
    d = F.normalize(second.right.detach().float().cpu(), dim=1)
    norm1 = (.5 * (1 + (a * b).sum(1).square())).sqrt()
    norm2 = (.5 * (1 + (c * d).sum(1).square())).sqrt()
    maxima = []
    for start in range(0, len(a), chunk):
        similarity = .5 * ((a[start:start + chunk] @ c.T) *
                           (b[start:start + chunk] @ d.T) +
                           (a[start:start + chunk] @ d.T) *
                           (b[start:start + chunk] @ c.T))
        similarity = similarity.abs() / (norm1[start:start + chunk, None] * norm2[None])
        maxima.append(similarity.max(1).values)
    values = torch.cat(maxima)
    return {"median_best_form_cosine": float(values.median()),
            "mean_best_form_cosine": float(values.mean()),
            "fraction_above_0_8": float((values > .8).float().mean()),
            "fraction_above_0_95": float((values > .95).float().mean())}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    input_archive = np.load(DATA / "layer1_inputs_large.npz")
    product_archive = np.load(DATA / "natural_large.npz")
    x = {split: torch.from_numpy(input_archive[f"{split}__input1"].astype(np.float32))
         for split in ("fineweb_train",) + SPLITS}
    h_raw = {split: torch.from_numpy(product_archive[f"{split}__h1"].astype(np.float32))
             for split in ("fineweb_train",) + SPLITS}
    h_mean = h_raw["fineweb_train"].mean(0)
    h_std = h_raw["fineweb_train"].std(0).clamp_min(1e-4)
    h = {split: (value - h_mean) / h_std for split, value in h_raw.items()}
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    teacher_left = state["transformer.h.1.mlp.Left.weight"].float()
    teacher_right = state["transformer.h.1.mlp.Right.weight"].float()
    teacher_down = state["transformer.h.1.mlp.Down.weight"].float()
    teacher_bias = state["transformer.h.1.mlp.Down_bias"].float()
    del state
    score = h_raw["fineweb_train"].var(0) * teacher_down.norm(dim=0).square()
    ranking = score.argsort(descending=True)

    results = {"bilinear_sae": {}, "replacements": {},
               "ranking": [int(i) for i in ranking.tolist()]}
    sae_models = []
    for seed in (201, 202):
        model = SparseBilinearDictionary(1152, 2048, 4608, 128, seed)
        losses = train_sae(model, x["fineweb_train"], h["fineweb_train"], 2000, seed)
        row = {"final_loss": float(np.mean(losses[-100:])), "evaluation": {}}
        for split in SPLITS:
            row["evaluation"][split], _ = evaluate_sae(model, x[split], h[split])
        torch.save(model.cpu().state_dict(), ROOT / f"bilinear_sae_seed{seed}.pt")
        results["bilinear_sae"][str(seed)] = row
        sae_models.append(model)
    results["bilinear_sae_stability"] = bilinear_feature_stability(*sae_models)

    replacement_models = {}
    for features, k in ((128, 128), (512, 128), (2048, 128)):
        seed = 300 + features
        model = SparseBilinearDictionary(1152, features, 1152, k, seed).cuda()
        units = ranking[:features].cuda()
        init_from_teacher(model, teacher_left.cuda(), teacher_right.cuda(),
                          teacher_down.cuda(), units)
        with torch.no_grad():
            model.bias.copy_(teacher_bias.cuda())
        losses = train_replacement(model, x["fineweb_train"], h_raw["fineweb_train"],
                                   teacher_down, teacher_bias, 1600, seed)
        row = {"features": features, "active": k,
               "parameters": sum(p.numel() for p in model.parameters()),
               "final_loss": float(np.mean(losses[-100:])), "evaluation": {}}
        for split in SPLITS:
            row["evaluation"][split] = evaluate_replacement(
                model, x[split], h_raw[split], teacher_down, teacher_bias)
        name = f"m{features}_k{k}"
        results["replacements"][name] = row
        torch.save(model.cpu().state_dict(), ROOT / f"replacement_{name}.pt")
        replacement_models[name] = model

    results["elapsed_seconds"] = time.time() - started
    (ROOT / "local_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote bilinear-code results in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
