#!/usr/bin/env python3
"""Red-team optimizer sensitivity for downstream CP distillation."""

from __future__ import annotations

import copy
import json

import numpy as np
import torch

from audit_bilin18_mlp_discriminators import SNAPSHOT
from refine_bilin18_tensor_decomposition import load_model
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT
from train_bilin18_causal_cp import train_variant


CONFIG = "configs/bilin18_causal_distillation.json"


def fresh_student(config, device, subset):
    student = load_model("cp_r2048")
    payload = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=True)
    student.load_state_dict(payload["state_dict"])
    if subset == "output_gain":
        student.left.requires_grad_(False)
        student.right.requires_grad_(False)
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

    trials = []
    for subset in ("all", "output_gain"):
        for learning_rate in (3e-7, 1e-6, 3e-6, 1e-5):
            trials.append(("adamw", learning_rate, subset))
    for subset in ("all", "output_gain"):
        for learning_rate in (1e-4, 3e-4, 1e-3):
            trials.append(("sgd", learning_rate, subset))

    result = {"status": "exploratory_after_failed_preregistered_optimization",
              "trials": {}}
    variant = base["variants"]["cp2048_kl"]
    for optimizer, learning_rate, subset in trials:
        config = copy.deepcopy(base)
        config.update({"optimizer": optimizer, "learning_rate": learning_rate,
                       "steps": 120, "validation_every": 10,
                       "save_checkpoint": False})
        name = f"explore_kl_{optimizer}_{learning_rate:.0e}_{subset}"
        print("TRIAL", name, flush=True)
        student = fresh_student(config, device, subset)
        training = train_variant(
            model, rows, student, name, variant, config, teacher_weights, device)
        result["trials"][name] = {
            "optimizer": optimizer, "learning_rate": learning_rate, "subset": subset,
            "initial_validation": training["initial_validation"],
            "best_validation": training["best_validation"],
            "best_step": min(training["validations"], key=lambda row: row["score"])["step"],
            "improvement": training["initial_validation"] - training["best_validation"],
            "validations": training["validations"],
        }
        del student; torch.cuda.empty_cache()
    ordered = sorted(result["trials"], key=lambda name: result["trials"][name]["best_validation"])
    result["ranking"] = ordered
    (ROOT / "causal_optimization_sweep.json").write_text(json.dumps(result, indent=2))
    print("BEST")
    for name in ordered[:8]:
        print(name, result["trials"][name])


if __name__ == "__main__":
    main()
