#!/usr/bin/env python3
"""Finite downstream validation of five layer1->2 canonicalizations."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from collect_bilin18_canonicalization import positional_source_mean
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


ROOT = Path("artifacts/bilin18_canonicalization")
OOD = Path("artifacts/bilin18_ood_corpora")


@torch.inference_mode()
def run(model, ids, units, replacement=None, restoration=None,
        state_target=None, return_state2=False):
    d = model.config.n_embd
    position = ids.shape[1] - 1
    x = F.rms_norm(model.embedding(ids), (d,))
    x0, value_bus = x, None
    captured_state2 = None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (d,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (d,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        if layer == 1 and replacement is not None:
            product = product.clone()
            for unit, value in replacement.items():
                product[0, position, unit] = value
        state_restoration = restoration is not None and restoration.numel() == d
        if layer == 2 and restoration is not None and not state_restoration:
            product = product.clone()
            product[0, position] += restoration.to(product.dtype)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
        if layer == 2 and state_restoration:
            x = x.clone()
            x[0, position] = (x[0, position].float() + restoration.float()).to(x.dtype)
        if layer == 2 and state_target is not None:
            x = x.clone()
            x[0, position] = state_target.to(x.dtype)
        if layer == 2:
            captured_state2 = x[0, position].float().clone()
    raw = model.lm_head(F.rms_norm(x, (d,))[:, -1]).float()
    logits = (30 * torch.tanh(raw / 30))[0]
    return (logits, captured_state2) if return_state2 else logits


def kl(reference, candidate):
    logp = F.log_softmax(reference, -1)
    return float((logp.exp() * (logp - F.log_softmax(candidate, -1))).sum())


def main():
    started = time.time()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    vectors = np.load(ROOT / "restoration_vectors.npz")
    selected = vectors["indices"].astype(int)
    methods = [key for key in vectors.files if key != "indices"]
    model = load_full_model("cuda" if torch.cuda.is_available() else "cpu")
    units = manifest["units"]
    fineweb = np.load(DEFAULT_CORPUS, mmap_mode="r")
    wiki = np.load(OOD / "wikitext103_1200x129.npy", mmap_mode="r")
    tiny = np.load(OOD / "tinystories_1200x129.npy", mmap_mode="r")
    class Args:
        sequence_length = 128
        batch_size = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
    positional_mean = positional_source_mean(model, fineweb[10000:10400], units, Args)
    rows_by_split = {
        "fineweb_test": fineweb[15000:18000],
        "wikitext103": wiki,
        "tinystories": tiny,
    }
    records = []
    for selected_column, global_index in enumerate(selected):
        site = manifest["sites"][int(global_index)]
        rows = rows_by_split[site["split"]]
        position = site["position"]
        full_row = np.asarray(rows[site["local_row"]], dtype=np.int64)
        ids = torch.as_tensor(full_row[:position + 1], device=Args.device)[None]
        target = int(full_row[position + 1])
        replacement = {unit: positional_mean[position, column]
                       for column, unit in enumerate(units)}
        base, base_state2 = run(model, ids, units, return_state2=True)
        changed = run(model, ids, units, replacement=replacement)
        full_delta = base - changed
        base_kl = max(kl(base, changed), 1e-12)
        base_ce = float(F.cross_entropy(base[None], torch.tensor([target], device=base.device)))
        changed_ce = float(F.cross_entropy(changed[None], torch.tensor([target], device=base.device)))
        for method in methods:
            restoration = torch.as_tensor(vectors[method][selected_column].astype(np.float32),
                                          device=Args.device)
            if method.startswith("exact_layer2_state_overwrite"):
                restored = run(model, ids, units, replacement=replacement,
                               state_target=base_state2)
            else:
                restored = run(model, ids, units, replacement=replacement,
                               restoration=restoration)
            residual = base - restored
            restored_ce = float(F.cross_entropy(
                restored[None], torch.tensor([target], device=base.device)))
            records.append({
                "split": site["split"], "site_index": int(global_index), "method": method,
                "logit_recovery": float(1 - residual.norm() / full_delta.norm().clamp_min(1e-12)),
                "kl_recovery": float(1 - kl(base, restored) / base_kl),
                "base_ce": base_ce, "changed_ce": changed_ce, "restored_ce": restored_ce,
            })
        if (selected_column + 1) % 12 == 0:
            print(f"finite validation {selected_column + 1}/{len(selected)}", flush=True)

    result = {"methods": {}, "sites": len(selected), "records": records,
              "elapsed_seconds": time.time() - started}
    for method in methods:
        result["methods"][method] = {}
        for split in rows_by_split:
            subset = [row for row in records if row["method"] == method and row["split"] == split]
            logit = np.asarray([row["logit_recovery"] for row in subset])
            kl_values = np.asarray([row["kl_recovery"] for row in subset])
            base_gap = np.asarray([abs(row["changed_ce"] - row["base_ce"]) for row in subset])
            residual_gap = np.asarray([abs(row["restored_ce"] - row["base_ce"]) for row in subset])
            result["methods"][method][split] = {
                "logit_recovery_median": float(np.median(logit)),
                "logit_recovery_mean": float(logit.mean()),
                "positive_logit_recovery_fraction": float((logit > 0).mean()),
                "kl_recovery_median": float(np.median(kl_values)),
                "absolute_ce_recovery": float(1 - residual_gap.mean() / base_gap.mean().clip(1e-12)),
            }
    (ROOT / "finite_validation.json").write_text(json.dumps(result, indent=2))
    print(f"wrote finite validation in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
