#!/usr/bin/env python3
"""Collect matched natural and causal data for sparse-code comparisons."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


ROOT = Path("artifacts/bilin18_sparse_codes")
OOD = Path("artifacts/bilin18_ood_corpora")
POSITIONS = tuple(range(7, 128, 8))
VARIANTS = ("top1_remove", "top4_remove", "top16_remove", "quiet8_add_2z")


@torch.inference_mode()
def front(model, ids):
    d = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (d,))
    x0, value_bus = x, None
    captured = {}
    for layer in range(3):
        block = model.blocks[layer]
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (d,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (d,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1:
            captured.update({"pre1": pre, "product1": product, "x0": x0,
                             "value_bus": value_bus})
        if layer == 2:
            captured.update({"input2": normalized, "product2": product})
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    captured["x1"] = (captured["pre1"] + model.blocks[1].mlp.Down(captured["product1"]) +
                      model.blocks[1].mlp.Down_bias)
    return captured


@torch.inference_mode()
def calibration(model, rows, batch_size=8):
    width = model.blocks[1].mlp.Left.weight.shape[0]
    total = torch.zeros(width, device="cuda")
    square = torch.zeros_like(total)
    count = 0
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        product = front(model, torch.as_tensor(raw, device="cuda"))["product1"][:, -1].float()
        total += product.sum(0)
        square += product.square().sum(0)
        count += len(raw)
    mean = total / count
    std = (square / count - mean.square()).clamp_min(1e-6).sqrt()
    return mean, std


@torch.inference_mode()
def natural_panel(model, rows, batch_size=8):
    products, inputs, tokens = [], [], []
    position_index = torch.as_tensor(POSITIONS, device="cuda")
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        state = front(model, torch.as_tensor(raw, device="cuda"))
        products.append(state["product1"][:, position_index].cpu().to(torch.float16).numpy())
        inputs.append(state["input2"][:, position_index].cpu().to(torch.float16).numpy())
        tokens.append(raw[:, POSITIONS])
    return {"h1": np.concatenate(products).reshape(-1, products[0].shape[-1]),
            "input2": np.concatenate(inputs).reshape(-1, inputs[0].shape[-1]),
            "token_ids": np.concatenate(tokens).reshape(-1)}


@torch.inference_mode()
def causal_panel(model, rows, mean, std, seed, batch_size=8):
    down_norm = model.blocks[1].mlp.Down.weight.detach().float().norm(dim=0)
    saved = {key: [] for key in ("source_delta", "context", "input_delta", "target_delta",
                                  "variant", "token_id")}
    rng = np.random.default_rng(seed)
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        ids = torch.as_tensor(raw, device="cuda")
        state = front(model, ids)
        batch = len(raw)
        native = state["product1"][:, -1].float()
        z = (native - mean) / std
        ranking = torch.argsort(z.abs() * down_norm, dim=1, descending=True)
        modified = state["product1"].repeat_interleave(len(VARIANTS), dim=0).clone()
        variant_ids = []
        for row_index in range(batch):
            for variant_index, name in enumerate(VARIANTS):
                output_row = row_index * len(VARIANTS) + variant_index
                if name.startswith("top"):
                    k = int(name.split("top")[1].split("_")[0])
                    selected = ranking[row_index, :k]
                    modified[output_row, -1, selected] = mean[selected].to(modified.dtype)
                else:
                    quiet = torch.where(z[row_index].abs() < .25)[0].cpu().numpy()
                    selected_np = rng.choice(quiet, 8, replace=False)
                    selected = torch.as_tensor(selected_np, device="cuda")
                    signs = torch.as_tensor(rng.choice([-1.0, 1.0], 8), device="cuda")
                    modified[output_row, -1, selected] = (
                        mean[selected] + signs * 2 * std[selected]).to(modified.dtype)
                variant_ids.append(variant_index)

        repeat = len(VARIANTS)
        pre1 = state["pre1"].repeat_interleave(repeat, dim=0)
        x0 = state["x0"].repeat_interleave(repeat, dim=0)
        value_bus = state["value_bus"].repeat_interleave(repeat, dim=0)
        x1_modified = pre1 + model.blocks[1].mlp.Down(modified) + model.blocks[1].mlp.Down_bias
        block2 = model.blocks[2]
        mixed = block2.lambdas[0] * x1_modified + block2.lambdas[1] * x0
        attention, _ = block2.attn(F.rms_norm(mixed, (model.config.n_embd,)), value_bus)
        input_modified = F.rms_norm(mixed + attention, (model.config.n_embd,))
        product_modified = (block2.mlp.Left(input_modified) * block2.mlp.Right(input_modified))

        base_product1 = state["product1"][:, -1].repeat_interleave(repeat, dim=0).float()
        base_input2 = state["input2"][:, -1].repeat_interleave(repeat, dim=0).float()
        base_product2 = state["product2"][:, -1].repeat_interleave(repeat, dim=0).float()
        saved["source_delta"].append((base_product1 - modified[:, -1].float()).cpu().to(torch.float16).numpy())
        saved["context"].append(base_input2.cpu().to(torch.float16).numpy())
        saved["input_delta"].append((base_input2 - input_modified[:, -1].float()).cpu().to(torch.float16).numpy())
        saved["target_delta"].append((base_product2 - product_modified[:, -1].float()).cpu().to(torch.float16).numpy())
        saved["variant"].append(np.asarray(variant_ids, dtype=np.uint8))
        saved["token_id"].append(np.repeat(raw[:, -1], repeat))
    return {key: np.concatenate(value) for key, value in saved.items()}


def main():
    started = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    model = load_full_model("cuda")
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    fineweb = np.load(DEFAULT_CORPUS, mmap_mode="r")
    wiki = np.load(OOD / "wikitext103_1200x129.npy", mmap_mode="r")
    tiny = np.load(OOD / "tinystories_1200x129.npy", mmap_mode="r")
    mean, std = calibration(model, fineweb[10000:10400])
    splits = {
        "fineweb_train": fineweb[12000:12512],
        "fineweb_test": fineweb[16000:16128],
        "wikitext103": wiki[900:1028],
        "tinystories": tiny[900:1028],
    }
    natural, causal = {}, {}
    for index, (name, rows) in enumerate(splits.items()):
        natural[name] = natural_panel(model, rows)
        causal_rows = rows if name != "fineweb_train" else fineweb[12512:13536]
        causal[name] = causal_panel(model, causal_rows, mean, std, 20261019 + index)
        print(name, "natural", len(natural[name]["h1"]),
              "causal", len(causal[name]["source_delta"]), flush=True)
    arrays = {}
    for family_name, family in (("natural", natural), ("causal", causal)):
        for split_name, values in family.items():
            for key, value in values.items():
                arrays[f"{family_name}__{split_name}__{key}"] = value
    arrays["layer1_mean"] = mean.cpu().numpy()
    arrays["layer1_std"] = std.cpu().numpy()
    np.savez_compressed(ROOT / "data.npz", **arrays)
    manifest = {
        "positions": list(POSITIONS), "variants": list(VARIANTS),
        "natural_counts": {name: len(values["h1"]) for name, values in natural.items()},
        "causal_counts": {name: len(values["source_delta"]) for name, values in causal.items()},
        "elapsed_seconds": time.time() - started,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote data in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
