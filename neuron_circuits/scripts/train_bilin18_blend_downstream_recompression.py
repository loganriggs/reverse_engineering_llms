#!/usr/bin/env python3
"""Fisher-weight CP-2048 recompression against the exact validated blend."""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from evaluate_bilin18_causal_cp import load_cp
from explore_bilin18_value_tangent_blend import BilinearBlend
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT
from train_bilin18_causal_cp import (
    apply_intervention,
    distribution_kl,
    fisher_effect_loss,
    prefix_to_layer1,
    student_outputs,
    suffix_from_layer1,
)


OUTPUT = ROOT / "blend_downstream_recompression_training.json"


@torch.no_grad()
def target_outputs(model, target, x0, value_bus, pre, base_input, changed_input):
    base = suffix_from_layer1(model, x0, value_bus, pre + target(base_input.float()).to(pre.dtype))
    changed = suffix_from_layer1(
        model, x0, value_bus, pre + target(changed_input.float()).to(pre.dtype))
    return base, changed


@torch.no_grad()
def validate(model, target, student, rows, config):
    cpu_generator = torch.Generator().manual_seed(20261025)
    gpu_generator = torch.Generator(device="cuda").manual_seed(20261026)
    values = []
    for offset, index in enumerate(range(17000, 17008)):
        raw = np.asarray(rows[index:index + 1, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
        if offset % 2 == 0:
            direction = torch.randn(normalized.shape, generator=gpu_generator, device="cuda")
            direction = direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            changed = normalized + 0.1 * direction.to(normalized.dtype)
        else:
            changed = normalized + 0.25 * (normalized.roll(1, 1) - normalized)
        target_base, target_changed = target_outputs(
            model, target, x0, value_bus, pre, normalized, changed)
        student_base, student_changed = student_outputs(
            model, student, x0, value_bus, pre, normalized, changed)
        score = (distribution_kl(target_base, student_base)
                 + distribution_kl(target_changed, student_changed)
                 + fisher_effect_loss(target_base, target_changed,
                                      student_base, student_changed))
        values.append(float(score))
    return float(np.mean(values))


def main():
    config = json.load(open("configs/bilin18_causal_distillation_r2.json"))
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.requires_grad_(False)
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    direct = load_cp(ROOT / "cp_r2048.pt", "cuda")
    target = BilinearBlend(language, direct, 0.2).cuda().eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    student = load_cp(ROOT / "blend_recompress_tangent.pt", "cuda").train()
    rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-6,
                                  betas=(0.9, 0.95), weight_decay=1e-7)
    cpu_generator = torch.Generator().manual_seed(20261027)
    gpu_generator = torch.Generator(device="cuda").manual_seed(20261028)
    initial = validate(model, target, student, rows, config)
    best, best_step = initial, 0
    best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    validations = [{"step": 0, "score": initial}]
    history = []
    started = time.time()
    for step in range(1, 501):
        index = int(torch.randint(0, 16000, (1,), generator=cpu_generator))
        raw = np.asarray(rows[index:index + 1, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
        changed, kind, scale = apply_intervention(
            normalized, config, cpu_generator, gpu_generator)
        target_base, target_changed = target_outputs(
            model, target, x0, value_bus, pre, normalized, changed)
        student_base, student_changed = student_outputs(
            model, student, x0, value_bus, pre, normalized, changed)
        base_kl = distribution_kl(target_base, student_base)
        changed_kl = distribution_kl(target_changed, student_changed)
        effect = fisher_effect_loss(
            target_base, target_changed, student_base, student_changed)
        loss = base_kl + changed_kl + effect
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); optimizer.step()
        history.append({"step": step, "loss": float(loss.detach()),
                        "base_kl": float(base_kl.detach()),
                        "changed_kl": float(changed_kl.detach()),
                        "effect_loss": float(effect.detach()),
                        "intervention": kind, "scale": scale})
        if step % 25 == 0:
            student.eval(); score = validate(model, target, student, rows, config); student.train()
            validations.append({"step": step, "score": score})
            if score < best:
                best, best_step = score, step
                best_state = {key: value.detach().cpu().clone()
                              for key, value in student.state_dict().items()}
            print("downstream recompress", step, f"loss={np.mean([r['loss'] for r in history[-25:]]):.5f}",
                  f"validation={score:.5f}", f"best={best:.5f}", flush=True)
    student.load_state_dict(best_state)
    torch.save({"state_dict": student.cpu().state_dict(),
                "configuration": {"type": "CPBilinear", "rank": 2048},
                "target": "downstream behavior of exact lambda=.2 blend",
                "best_step": best_step, "best_validation": best},
               ROOT / "blend_recompress_downstream.pt")
    OUTPUT.write_text(json.dumps({"initial_validation": initial,
                                  "best_validation": best, "best_step": best_step,
                                  "validations": validations, "history": history,
                                  "elapsed_seconds": time.time() - started}, indent=2))
    print("wrote downstream recompression", flush=True)


if __name__ == "__main__":
    main()
