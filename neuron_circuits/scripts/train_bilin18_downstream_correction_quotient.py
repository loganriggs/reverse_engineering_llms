#!/usr/bin/env python3
"""Fit low-rank linear correction maps with a downstream causal objective."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate_bilin18_causal_cp import load_cp
from explore_bilin18_correction_subspace import basis_from_covariance, fit_covariances
from explore_bilin18_value_tangent_blend import BilinearBlend
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT
from train_bilin18_causal_cp import (
    apply_intervention,
    distribution_kl,
    fisher_effect_loss,
    prefix_to_layer1,
    suffix_from_layer1,
)


CONFIG = Path("configs/bilin18_downstream_correction_quotient.json")
DATA = Path("artifacts/bilin18_sparse_codes/layer1_inputs_large.npz")
OUTPUT = ROOT / "downstream_correction_quotient_training.json"


class LowRankCorrection(nn.Module):
    def __init__(self, language, direct, basis, rank, weight):
        super().__init__()
        self.language = language
        self.direct = direct
        self.encoder = nn.Parameter(basis[:, :rank].T.contiguous())
        self.decoder = nn.Parameter(basis[:, :rank].contiguous())
        self.weight = weight
        self.rank = rank

    def forward(self, x):
        base = self.language(x)
        correction = self.direct(x) - base
        compressed = F.linear(correction, self.encoder)
        projected = F.linear(compressed, self.decoder)
        return base + self.weight * projected


@torch.no_grad()
def outputs(model, replacement, x0, value_bus, pre, base_input, changed):
    base_write = replacement(base_input.float()).to(pre.dtype)
    changed_write = replacement(changed.float()).to(pre.dtype)
    base = suffix_from_layer1(model, x0, value_bus, pre + base_write)
    altered = suffix_from_layer1(model, x0, value_bus, pre + changed_write)
    return base, altered


def trainable_outputs(model, replacement, x0, value_bus, pre, base_input, changed):
    base_write = replacement(base_input.float()).to(pre.dtype)
    changed_write = replacement(changed.float()).to(pre.dtype)
    base = suffix_from_layer1(model, x0, value_bus, pre + base_write)
    altered = suffix_from_layer1(model, x0, value_bus, pre + changed_write)
    return base, altered


def layer_relative_loss(target, prediction):
    return ((prediction - target).square().sum() /
            target.square().sum().clamp_min(1e-8))


@torch.no_grad()
def validation(model, target, student, rows, config, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = []
    spec = config["validation_rows"]
    for offset, index in enumerate(range(spec["start"], spec["stop"])):
        raw = np.asarray(rows[index:index + 1, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
        if offset % 2 == 0:
            direction = torch.randn(normalized.shape, generator=generator, device="cuda")
            direction = direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            changed = normalized + .1 * direction.to(normalized.dtype)
        else:
            changed = normalized + .25 * (normalized.roll(1, 1) - normalized)
        target_base, target_changed = outputs(
            model, target, x0, value_bus, pre, normalized, changed)
        student_base, student_changed = outputs(
            model, student, x0, value_bus, pre, normalized, changed)
        values.append(float(
            distribution_kl(target_base, student_base)
            + distribution_kl(target_changed, student_changed)
            + fisher_effect_loss(target_base, target_changed,
                                 student_base, student_changed)))
    return float(np.mean(values))


def main():
    config = json.loads(CONFIG.read_text())
    language = load_cp(ROOT / "cp2048_causal_r2.pt", "cuda")
    direct = load_cp(ROOT / "cp_r2048.pt", "cuda")
    for candidate in (language, direct):
        candidate.requires_grad_(False)
    archive = np.load(DATA)
    x = torch.from_numpy(archive["fineweb_train__input1"].astype(np.float32))
    _, normalized, _ = fit_covariances(language, direct, x, 16384, config["seed"])
    hybrid_basis, _ = basis_from_covariance(sum(normalized.values()))
    target = BilinearBlend(language, direct, config["blend_lambda"]).cuda().eval()
    target.requires_grad_(False)
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        module.requires_grad_(False)
    rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    all_results = {"config": config, "ranks": {}}
    for rank in config["ranks"]:
        student = LowRankCorrection(
            language, direct, hybrid_basis, rank, config["blend_lambda"]).cuda().train()
        optimizer = torch.optim.AdamW(
            (student.encoder, student.decoder), lr=config["learning_rate"],
            betas=(.9, .95), weight_decay=config["weight_decay"])
        cpu_generator = torch.Generator().manual_seed(config["seed"] + rank)
        gpu_generator = torch.Generator(device="cuda").manual_seed(config["seed"] + 2 * rank)
        initial = validation(model, target, student, rows, config, config["seed"] + 3 * rank)
        best, best_step = initial, 0
        best_state = {"encoder": student.encoder.detach().cpu().clone(),
                      "decoder": student.decoder.detach().cpu().clone()}
        validations = [{"step": 0, "score": initial}]
        history = []
        started = time.time()
        for step in range(1, config["steps"] + 1):
            index = int(torch.randint(config["train_rows"]["start"],
                                      config["train_rows"]["stop"], (1,),
                                      generator=cpu_generator))
            raw = np.asarray(rows[index:index + 1, :129], dtype=np.int64)
            ids = torch.as_tensor(raw[:, :128], device="cuda")
            x0, value_bus, pre, normalized_input = prefix_to_layer1(model, ids)
            changed, kind, scale = apply_intervention(
                normalized_input, config, cpu_generator, gpu_generator)
            with torch.no_grad():
                target_base, target_changed = outputs(
                    model, target, x0, value_bus, pre, normalized_input, changed)
                target_write = target(normalized_input.float())
                target_changed_write = target(changed.float())
            student_base, student_changed = trainable_outputs(
                model, student, x0, value_bus, pre, normalized_input, changed)
            student_write = student(normalized_input.float())
            student_changed_write = student(changed.float())
            base_kl = distribution_kl(target_base, student_base)
            changed_kl = distribution_kl(target_changed, student_changed)
            effect = fisher_effect_loss(
                target_base, target_changed, student_base, student_changed)
            layer_loss = .5 * (layer_relative_loss(target_write, student_write)
                               + layer_relative_loss(target_changed_write,
                                                     student_changed_write))
            loss = base_kl + changed_kl + effect + config["layer_loss_weight"] * layer_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_((student.encoder, student.decoder), 1.0)
            optimizer.step()
            history.append({"step": step, "loss": float(loss.detach()),
                            "base_kl": float(base_kl.detach()),
                            "changed_kl": float(changed_kl.detach()),
                            "effect": float(effect.detach()),
                            "layer_loss": float(layer_loss.detach()),
                            "intervention": kind, "scale": scale})
            if step % config["validation_every"] == 0:
                student.eval()
                score = validation(model, target, student, rows, config,
                                   config["seed"] + 3 * rank)
                student.train()
                validations.append({"step": step, "score": score})
                if score < best:
                    best, best_step = score, step
                    best_state = {"encoder": student.encoder.detach().cpu().clone(),
                                  "decoder": student.decoder.detach().cpu().clone()}
                print("quotient", rank, step, f"validation={score:.6f}",
                      f"best={best:.6f}", flush=True)
        with torch.no_grad():
            student.encoder.copy_(best_state["encoder"].cuda())
            student.decoder.copy_(best_state["decoder"].cuda())
        torch.save({"encoder": best_state["encoder"], "decoder": best_state["decoder"],
                    "rank": rank, "weight": config["blend_lambda"],
                    "best_step": best_step, "best_validation": best},
                   ROOT / f"downstream_correction_rank{rank}.pt")
        all_results["ranks"][str(rank)] = {
            "initial_validation": initial, "best_validation": best,
            "best_step": best_step, "validations": validations,
            "history": history, "elapsed_seconds": time.time() - started}
        del student
        torch.cuda.empty_cache()
    OUTPUT.write_text(json.dumps(all_results, indent=2))
    print("wrote", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
