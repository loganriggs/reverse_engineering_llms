#!/usr/bin/env python3
"""Collect a substantially larger natural-activation panel for the SAE red team."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from collect_bilin18_sparse_code_data import OOD, POSITIONS, front
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


ROOT = Path("artifacts/bilin18_sparse_codes")


@torch.inference_mode()
def collect(model, rows, batch_size=16):
    products, tokens = [], []
    positions = torch.as_tensor(POSITIONS, device="cuda")
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :128], dtype=np.int64)
        product = front(model, torch.as_tensor(raw, device="cuda"))["product1"]
        products.append(product[:, positions].cpu().to(torch.float16).numpy())
        tokens.append(raw[:, POSITIONS])
        if start and start % 2048 == 0:
            print(start, "/", len(rows), flush=True)
    return (np.concatenate(products).reshape(-1, products[0].shape[-1]),
            np.concatenate(tokens).reshape(-1))


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
    arrays, counts = {}, {}
    for split, split_rows in rows.items():
        h1, ids = collect(model, split_rows)
        arrays[f"{split}__h1"] = h1
        arrays[f"{split}__token_ids"] = ids
        counts[split] = len(h1)
        print(split, h1.shape, flush=True)
    np.savez_compressed(ROOT / "natural_large.npz", **arrays)
    manifest = {"positions": list(POSITIONS), "counts": counts,
                "elapsed_seconds": time.time() - started}
    (ROOT / "natural_large_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote large SAE panel in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
