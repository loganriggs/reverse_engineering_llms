#!/usr/bin/env python3
"""Large-data matched SAE-vs-native replication for Bilin18 layer 1."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from train_bilin18_sparse_codes import (SPLITS, TopKAutoencoder, category_analysis,
                                         decoder_stability, train)


ROOT = Path("artifacts/bilin18_sparse_codes")


def summarize(target_energy, residual_energy, cosine):
    score = 1 - torch.cat(residual_energy) / torch.cat(target_energy).clamp_min(1e-20)
    cos = torch.cat(cosine)
    return {"median_fidelity": float(score.median()), "mean_fidelity": float(score.mean()),
            "median_cosine": float(cos.median())}


@torch.inference_mode()
def evaluate_sae(model, values, std, down, batch_size=128):
    model.eval(); down = down.cuda(); std = std.cuda()
    act_t, act_r, act_c = [], [], []
    write_t, write_r, write_c, codes = [], [], [], []
    for start in range(0, len(values), batch_size):
        x = values[start:start + batch_size].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstruction, code = model(x, 128)
        reconstruction = reconstruction.float()
        target_write = (x * std) @ down.T
        predicted_write = (reconstruction * std) @ down.T
        act_t.append(x.square().sum(1).cpu())
        act_r.append((x - reconstruction).square().sum(1).cpu())
        act_c.append(F.cosine_similarity(x, reconstruction, dim=1).cpu())
        write_t.append(target_write.square().sum(1).cpu())
        write_r.append((target_write - predicted_write).square().sum(1).cpu())
        write_c.append(F.cosine_similarity(target_write, predicted_write, dim=1).cpu())
        codes.append(code.to(torch.float16).cpu())
    return {"activation": summarize(act_t, act_r, act_c),
            "write": summarize(write_t, write_r, write_c)}, torch.cat(codes)


@torch.inference_mode()
def evaluate_native(values, std, down, batch_size=128):
    down = down.cuda(); std = std.cuda()
    act_t, act_r, act_c, write_t, write_r, write_c = [], [], [], [], [], []
    for start in range(0, len(values), batch_size):
        x = values[start:start + batch_size].cuda()
        selected = torch.topk(x.abs(), 128, dim=1)
        reconstruction = torch.zeros_like(x).scatter(
            1, selected.indices, x.gather(1, selected.indices))
        target_write = (x * std) @ down.T
        predicted_write = (reconstruction * std) @ down.T
        act_t.append(x.square().sum(1).cpu())
        act_r.append((x - reconstruction).square().sum(1).cpu())
        act_c.append(F.cosine_similarity(x, reconstruction, dim=1).cpu())
        write_t.append(target_write.square().sum(1).cpu())
        write_r.append((target_write - predicted_write).square().sum(1).cpu())
        write_c.append(F.cosine_similarity(target_write, predicted_write, dim=1).cpu())
    return {"activation": summarize(act_t, act_r, act_c),
            "write": summarize(write_t, write_r, write_c)}


def main():
    started = time.time()
    archive = np.load(ROOT / "natural_large.npz")
    raw = {split: torch.from_numpy(archive[f"{split}__h1"].astype(np.float32))
           for split in ("fineweb_train",) + SPLITS}
    mean = raw["fineweb_train"].mean(0)
    std = raw["fineweb_train"].std(0).clamp_min(1e-4)
    values = {split: (value - mean) / std for split, value in raw.items()}
    del raw
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    down = state["transformer.h.1.mlp.Down.weight"].float()
    del state
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    ids = {split: archive[f"{split}__token_ids"] for split in ("fineweb_train",) + SPLITS}

    results = {"counts": {split: len(value) for split, value in values.items()},
               "seeds": {}, "native": {}}
    models = []
    for seed in (101, 102):
        model = TopKAutoencoder(4608, 6144, 128, seed)
        losses = train(model, [values["fineweb_train"]], 2000, 256, 3e-4,
                       label=f"large_activation_sae_seed{seed}")
        model.cuda()
        row = {"final_loss": float(np.mean(losses[-100:])), "evaluation": {}}
        codes = {}
        for split in SPLITS:
            row["evaluation"][split], codes[split] = evaluate_sae(
                model, values[split], std, down)
        _, train_code = evaluate_sae(model, values["fineweb_train"][:32768], std, down)
        row["categories"] = category_analysis(
            train_code, codes, ids["fineweb_train"][:32768],
            {split: ids[split] for split in SPLITS}, tokenizer)
        torch.save(model.cpu().state_dict(), ROOT / f"activation_sae_large_seed{seed}.pt")
        models.append(model)
        results["seeds"][str(seed)] = row
    results["stability"] = decoder_stability(*models)

    for split in SPLITS:
        results["native"][split] = evaluate_native(values[split], std, down)
    results["native_categories"] = category_analysis(
        values["fineweb_train"][:32768], {split: values[split] for split in SPLITS},
        ids["fineweb_train"][:32768], {split: ids[split] for split in SPLITS}, tokenizer)
    results["elapsed_seconds"] = time.time() - started
    (ROOT / "large_sae_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote large SAE results in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
