#!/usr/bin/env python3
"""Fine-tune rank-256 correction quotients for distinct intervention geometries."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_bilin18_causal_cp import load_cp
from explore_bilin18_value_tangent_blend import BilinearBlend
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT
from train_bilin18_causal_cp import (
    distribution_kl, fisher_effect_loss, prefix_to_layer1, suffix_from_layer1)
from train_bilin18_downstream_correction_quotient import (
    LowRankCorrection, layer_relative_loss, outputs, trainable_outputs)


CONFIG = Path("configs/bilin18_correction_quotient_specialists.json")
OUTPUT = ROOT / "correction_quotient_specialist_training.json"


def intervene(normalized, spec, generator):
    if spec["kind"] == "gaussian":
        direction = torch.randn(normalized.shape, generator=generator, device="cuda")
        direction = direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        return normalized + spec["scale"] * direction.to(normalized.dtype)
    return normalized + spec["scale"] * (normalized.roll(1, 1) - normalized)


@torch.no_grad()
def validate(model, target, student, rows, spec, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = []
    for index in range(17100, 17108):
        raw = np.asarray(rows[index:index + 1, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
        changed = intervene(normalized, spec, generator)
        target_base, target_changed = outputs(
            model, target, x0, value_bus, pre, normalized, changed)
        student_base, student_changed = outputs(
            model, student, x0, value_bus, pre, normalized, changed)
        values.append(float(distribution_kl(target_base, student_base)
                            + distribution_kl(target_changed, student_changed)
                            + fisher_effect_loss(target_base, target_changed,
                                                 student_base, student_changed)))
    return float(np.mean(values))


def main():
    config = json.loads(CONFIG.read_text())
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    direct = load_cp(ROOT / "cp_r2048.pt", "cuda")
    for candidate in (language, direct): candidate.requires_grad_(False)
    target = BilinearBlend(language, direct, .2).cuda().eval()
    target.requires_grad_(False)
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head): module.requires_grad_(False)
    rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    initial = torch.load(ROOT / "downstream_correction_rank256.pt",
                         map_location="cpu", weights_only=True)
    result = {"config": config, "specialists": {}}
    for offset, (name, spec) in enumerate(config["specialists"].items()):
        dummy = torch.zeros(1152, config["rank"], device="cuda")
        student = LowRankCorrection(language, direct, dummy, config["rank"], .2).cuda().train()
        with torch.no_grad():
            student.encoder.copy_(initial["encoder"].cuda())
            student.decoder.copy_(initial["decoder"].cuda())
        optimizer = torch.optim.AdamW((student.encoder, student.decoder),
                                      lr=config["learning_rate"], betas=(.9, .95),
                                      weight_decay=1e-7)
        cpu_generator = torch.Generator().manual_seed(config["seed"] + offset)
        gpu_generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 10 + offset)
        initial_score = validate(model, target, student, rows, spec, config["seed"] + 20 + offset)
        best, best_step = initial_score, 0
        best_state = {"encoder": student.encoder.detach().cpu().clone(),
                      "decoder": student.decoder.detach().cpu().clone()}
        validations = [{"step": 0, "score": initial_score}]
        history = []; started = time.time()
        for step in range(1, config["steps"] + 1):
            index = int(torch.randint(0, 16000, (1,), generator=cpu_generator))
            raw = np.asarray(rows[index:index + 1, :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
            changed = intervene(normalized, spec, gpu_generator)
            with torch.no_grad():
                target_base, target_changed = outputs(
                    model, target, x0, value_bus, pre, normalized, changed)
                target_write, target_changed_write = target(normalized.float()), target(changed.float())
            student_base, student_changed = trainable_outputs(
                model, student, x0, value_bus, pre, normalized, changed)
            student_write, student_changed_write = student(normalized.float()), student(changed.float())
            base_kl = distribution_kl(target_base, student_base)
            changed_kl = distribution_kl(target_changed, student_changed)
            effect = fisher_effect_loss(target_base, target_changed,
                                        student_base, student_changed)
            layer = .5 * (layer_relative_loss(target_write, student_write)
                          + layer_relative_loss(target_changed_write, student_changed_write))
            loss = base_kl + changed_kl + effect + config["layer_loss_weight"] * layer
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_((student.encoder, student.decoder), 1.0)
            optimizer.step()
            history.append({"step": step, "loss": float(loss.detach()),
                            "base_kl": float(base_kl.detach()),
                            "changed_kl": float(changed_kl.detach()),
                            "effect": float(effect.detach()), "layer": float(layer.detach())})
            if step % config["validation_every"] == 0:
                student.eval(); score = validate(
                    model, target, student, rows, spec, config["seed"] + 20 + offset); student.train()
                validations.append({"step": step, "score": score})
                if score < best:
                    best, best_step = score, step
                    best_state = {"encoder": student.encoder.detach().cpu().clone(),
                                  "decoder": student.decoder.detach().cpu().clone()}
                print("specialist", name, step, f"validation={score:.6f}",
                      f"best={best:.6f}", flush=True)
        torch.save({**best_state, "rank": config["rank"], "weight": .2,
                    "best_step": best_step, "best_validation": best,
                    "specialist": name}, ROOT / f"correction_quotient_{name}.pt")
        result["specialists"][name] = {
            "initial_validation": initial_score, "best_validation": best,
            "best_step": best_step, "validations": validations, "history": history,
            "elapsed_seconds": time.time() - started}
        del student; torch.cuda.empty_cache()
    OUTPUT.write_text(json.dumps(result, indent=2))
    print("wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
