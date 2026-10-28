#!/usr/bin/env python3
"""Collect layer-1 normalized MLP inputs aligned with the large activation panel."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from collect_bilin18_sparse_code_data import OOD, POSITIONS, front
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


ROOT = Path("artifacts/bilin18_sparse_codes")


@torch.inference_mode()
def collect(model, rows, batch_size=16):
    inputs = []
    positions = torch.as_tensor(POSITIONS, device="cuda")
    dimension = model.config.n_embd
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        state = front(model, torch.as_tensor(raw, device="cuda"))
        normalized = F.rms_norm(state["pre1"], (dimension,))
        inputs.append(normalized[:, positions].cpu().to(torch.float16).numpy())
        if start and start % 2048 == 0:
            print(start, "/", len(rows), flush=True)
    return np.concatenate(inputs).reshape(-1, dimension)


def main():
    started = time.time()
    model = load_full_model("cuda")
    fineweb = np.load(DEFAULT_CORPUS, mmap_mode="r")
    wiki = np.load(OOD / "wikitext103_1200x129.npy", mmap_mode="r")
    tiny = np.load(OOD / "tinystories_1200x129.npy", mmap_mode="r")
    rows = {
        "fineweb_train": fineweb[:8192],
        "fineweb_test": fineweb[8192:9216],
        "wikitext103": wiki[:1024],
        "tinystories": tiny[:1024],
    }
    arrays = {}
    for split, panel in rows.items():
        arrays[f"{split}__input1"] = collect(model, panel)
        print(split, arrays[f"{split}__input1"].shape, flush=True)
    np.savez_compressed(ROOT / "layer1_inputs_large.npz", **arrays)
    (ROOT / "layer1_inputs_large_manifest.json").write_text(json.dumps({
        "positions": list(POSITIONS),
        "counts": {split: len(value) for split, value in
                   ((name, arrays[f"{name}__input1"]) for name in rows)},
        "elapsed_seconds": time.time() - started,
    }, indent=2))
    print(f"wrote aligned layer-1 inputs in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
