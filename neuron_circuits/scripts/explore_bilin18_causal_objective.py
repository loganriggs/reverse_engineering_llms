#!/usr/bin/env python3
"""Sweep stable learning rates and causal-effect weights under one validation metric."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch

from audit_bilin18_mlp_discriminators import SNAPSHOT
from refine_bilin18_tensor_decomposition import load_model
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT
from train_bilin18_causal_cp import train_variant, validation_score


CONFIG = "configs/bilin18_causal_distillation.json"


def fresh_student(config, device):
    student = load_model("cp_r2048")
    payload = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=True)
    student.load_state_dict(payload["state_dict"])
    return student.to(device)


def main():
    base = json.load(open(CONFIG))
    device = "cuda"
    model = load_full_model(device)
    for module in (model.embedding, model.blocks, model.lm_head):
        module.requires_grad_(False)
    rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    teacher_weights = tuple(state[key].float().to(device) for key in (
        "transformer.h.1.mlp.Left.weight", "transformer.h.1.mlp.Right.weight",
        "transformer.h.1.mlp.Down.weight"))
    del state

    common_variant = copy.deepcopy(base["variants"]["cp2048_causal"])
    result = {"status": "exploratory_objective_sweep", "common_validation_variant": common_variant,
              "trials": {}}
    for learning_rate in (1e-6, 3e-6, 1e-5):
        for effect_weight in (0.0, 0.25, 1.0):
            config = copy.deepcopy(base)
            config.update({"optimizer": "adamw", "learning_rate": learning_rate,
                           "steps": 160, "validation_every": 10,
                           "save_checkpoint": False})
            variant = copy.deepcopy(common_variant)
            variant["effect_weight"] = effect_weight
            name = f"explore_causal_lr{learning_rate:.0e}_effect{effect_weight:g}"
            print("TRIAL", name, flush=True)
            student = fresh_student(config, device)
            training = train_variant(
                model, rows, student, name, variant, config, teacher_weights, device)
            common_score = validation_score(
                model, rows, student, common_variant, config, teacher_weights, device)
            result["trials"][name] = {
                "learning_rate": learning_rate, "effect_weight": effect_weight,
                "own_initial_validation": training["initial_validation"],
                "own_best_validation": training["best_validation"],
                "best_step": min(training["validations"], key=lambda row: row["score"])["step"],
                "common_validation": common_score,
            }
            del student; torch.cuda.empty_cache()
    ordered = sorted(result["trials"], key=lambda name: result["trials"][name]["common_validation"])
    result["ranking"] = ordered
    (ROOT / "causal_objective_sweep.json").write_text(json.dumps(result, indent=2))
    print("BEST COMMON VALIDATION")
    for name in ordered:
        print(name, result["trials"][name])


if __name__ == "__main__":
    main()
