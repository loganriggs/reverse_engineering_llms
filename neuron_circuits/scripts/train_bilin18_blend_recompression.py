#!/usr/bin/env python3
"""Recompress the validated rank-4096 value–tangent blend into CP-2048."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from evaluate_bilin18_causal_cp import load_cp
from explore_bilin18_value_tangent_blend import BilinearBlend
from run_bilin18_tensor_decomposition import ROOT


DATA = "artifacts/bilin18_sparse_codes/layer1_inputs_large.npz"
OUTPUT = ROOT / "blend_recompression_training.json"


def relative_mse(target, prediction):
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1e-8)


@torch.no_grad()
def validation(student, target, x, seed, tangent_weight, curvature_weight):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = []
    for start in range(0, len(x), 256):
        bx = x[start:start + 256].cuda()
        v = torch.randn(bx.shape, generator=generator, device="cuda")
        v = v / v.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        target_value, prediction_value = target(bx), student(bx)
        loss = relative_mse(target_value, prediction_value)
        if tangent_weight:
            target_tangent = 2 * target.bilinear(bx, v)
            prediction_tangent = 2 * student.bilinear(bx, v)
            loss = loss + tangent_weight * relative_mse(target_tangent, prediction_tangent)
        if curvature_weight:
            target_curvature = target.bilinear(v, v)
            prediction_curvature = student.bilinear(v, v)
            loss = loss + curvature_weight * relative_mse(target_curvature, prediction_curvature)
        values.append(float(loss))
    return float(np.mean(values))


def train_variant(name, student, target, train_x, validation_x, tangent_weight, curvature_weight):
    student.cuda().train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-5, betas=(0.9, 0.95),
                                  weight_decay=1e-7)
    cpu_generator = torch.Generator().manual_seed(20261021)
    gpu_generator = torch.Generator(device="cuda").manual_seed(20261022)
    initial = validation(student, target, validation_x, 20261023,
                         tangent_weight, curvature_weight)
    best, best_step = initial, 0
    best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    validations = [{"step": 0, "score": initial}]
    losses = []
    started = time.time()
    for step in range(1, 1001):
        indices = torch.randint(len(train_x), (128,), generator=cpu_generator)
        x = train_x[indices].cuda()
        if step % 2:
            v = torch.randn(x.shape, generator=gpu_generator, device="cuda")
            v = v / v.square().mean(1, keepdim=True).sqrt().clamp_min(1e-8)
        else:
            v = x.roll(1, 0) - x
        with torch.no_grad():
            target_value = target(x)
            target_tangent = 2 * target.bilinear(x, v)
            target_curvature = target.bilinear(v, v)
        prediction_value = student(x)
        loss = relative_mse(target_value, prediction_value)
        if tangent_weight:
            loss = loss + tangent_weight * relative_mse(
                target_tangent, 2 * student.bilinear(x, v))
        if curvature_weight:
            loss = loss + curvature_weight * relative_mse(
                target_curvature, student.bilinear(v, v))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            student.eval()
            score = validation(student, target, validation_x, 20261023,
                               tangent_weight, curvature_weight)
            validations.append({"step": step, "score": score})
            if score < best:
                best, best_step = score, step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in student.state_dict().items()}
            student.train()
            print(name, step, f"train={np.mean(losses[-100:]):.5f}",
                  f"validation={score:.5f}", f"best={best:.5f}", flush=True)
    student.load_state_dict(best_state)
    torch.save({"state_dict": student.cpu().state_dict(),
                "configuration": {"type": "CPBilinear", "rank": 2048},
                "target": "0.8 * cp2048_causal_r2 + 0.2 * direct_cp",
                "tangent_weight": tangent_weight, "curvature_weight": curvature_weight,
                "best_step": best_step, "best_validation": best}, ROOT / f"{name}.pt")
    return {"initial_validation": initial, "best_validation": best,
            "best_step": best_step, "validations": validations,
            "elapsed_seconds": time.time() - started}


def main():
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    global_model = load_cp(ROOT / "cp_r2048.pt", "cuda")
    target = BilinearBlend(language, global_model, 0.2).cuda().eval()
    archive = np.load(DATA)
    train_x = torch.from_numpy(archive["fineweb_train__input1"].astype(np.float32))
    validation_x = torch.from_numpy(archive["fineweb_test__input1"][:4096].astype(np.float32))
    result = {"status": "exploratory_recompression", "variants": {}}
    for name, tangent, curvature in (
            ("blend_recompress_value", 0.0, 0.0),
            ("blend_recompress_tangent", 0.25, 0.05)):
        student = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
        result["variants"][name] = train_variant(
            name, student, target, train_x, validation_x, tangent, curvature)
        del student; torch.cuda.empty_cache()
    OUTPUT.write_text(json.dumps(result, indent=2))
    print("wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
