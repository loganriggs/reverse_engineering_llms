#!/usr/bin/env python3
"""Random-initialization control for the M2048/K128 layer-1 replacement."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from audit_bilin18_mlp_discriminators import SNAPSHOT
from train_bilin18_bilinear_codes import (ROOT, SparseBilinearDictionary,
                                           bilinear_feature_stability,
                                           evaluate_replacement,
                                           train_replacement)
from train_bilin18_sparse_codes import SPLITS


DATA = Path("artifacts/bilin18_sparse_codes")


def main():
    started = time.time()
    inputs = np.load(DATA / "layer1_inputs_large.npz")
    products = np.load(DATA / "natural_large.npz")
    x = {split: torch.from_numpy(inputs[f"{split}__input1"].astype(np.float32))
         for split in ("fineweb_train",) + SPLITS}
    h = {split: torch.from_numpy(products[f"{split}__h1"].astype(np.float32))
         for split in ("fineweb_train",) + SPLITS}
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    down = state["transformer.h.1.mlp.Down.weight"].float()
    bias = state["transformer.h.1.mlp.Down_bias"].float()
    del state
    models, result = [], {"seeds": {}}
    for seed in (2401, 2402):
        model = SparseBilinearDictionary(1152, 2048, 1152, 128, seed).cuda()
        with torch.no_grad():
            model.bias.copy_(bias.cuda())
        losses = train_replacement(model, x["fineweb_train"], h["fineweb_train"],
                                   down, bias, 2000, seed)
        row = {"final_loss": float(np.mean(losses[-100:])), "evaluation": {}}
        for split in SPLITS:
            row["evaluation"][split] = evaluate_replacement(
                model, x[split], h[split], down, bias)
        result["seeds"][str(seed)] = row
        torch.save(model.cpu().state_dict(), ROOT / f"replacement_m2048_k128_scratch{seed}.pt")
        models.append(model)
    result["stability"] = bilinear_feature_stability(*models)
    result["elapsed_seconds"] = time.time() - started
    (ROOT / "scratch_results.json").write_text(json.dumps(result, indent=2))
    print(f"wrote scratch control in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
