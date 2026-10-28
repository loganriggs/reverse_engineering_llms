#!/usr/bin/env python3
"""Collect a frozen causal tensor for canonicalizing the layer1 -> layer2 path.

The source is the four-neuron ``suffix_ing`` code discovered without natural
text. Sites span ordinary score quantiles rather than only activation tails.
FineWeb train is the only fitting split; three disjoint panels are transfer-only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from discover_bilin18_weight_only_circuits import token_flags
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from validate_bilin18_weight_only_circuits import layer1_selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites-per-split", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fineweb", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ood-dir", type=Path, default=Path("artifacts/bilin18_ood_corpora"))
    parser.add_argument("--discovery", type=Path,
                        default=Path("artifacts/bilin18_weight_only_circuits/discovery.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/bilin18_canonicalization"))
    return parser.parse_args()


def score_rows(model, rows, units, decoder, ing_flags, args):
    unit_index = torch.as_tensor(units, device=args.device)
    candidates = []
    for start in range(0, len(rows), args.batch_size):
        raw = np.asarray(rows[start:start + args.batch_size, :args.sequence_length], dtype=np.int64)
        ids = torch.as_tensor(raw, device=args.device)
        values = layer1_selected(model, ids, unit_index).cpu().numpy()
        scores = ((values - decoder["mean"]) / decoder["std"]) @ decoder["weight"]
        mask = ing_flags[raw]
        mask[:, :8] = False
        for batch_row, position in zip(*np.where(mask)):
            candidates.append((float(scores[batch_row, position]), start + int(batch_row),
                               int(position), values[batch_row, position].tolist()))
    # Uniform score quantiles prevent the causal tensor from describing tails only.
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < args.sites_per_split:
        raise RuntimeError(f"only {len(candidates)} -ing sites")
    indices = np.linspace(0, len(candidates) - 1, args.sites_per_split).round().astype(int)
    return [candidates[index] for index in indices]


@torch.inference_mode()
def positional_source_mean(model, rows, units, args):
    unit_index = torch.as_tensor(units, device=args.device)
    total = torch.zeros(args.sequence_length, len(units), device=args.device)
    count = 0
    for start in range(0, len(rows), args.batch_size):
        raw = np.asarray(rows[start:start + args.batch_size, :args.sequence_length], dtype=np.int64)
        values = layer1_selected(model, torch.as_tensor(raw, device=args.device), unit_index)
        total += values.sum(0)
        count += len(raw)
    return total / count


@torch.inference_mode()
def trace(model, ids, units, replacement=None, stop_at_layer2=False):
    d = model.config.n_embd
    position = ids.shape[1] - 1
    x = F.rms_norm(model.embedding(ids), (d,))
    x0, value_bus = x, None
    captured = {}
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (d,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (d,))
        left = block.mlp.Left(normalized)
        right = block.mlp.Right(normalized)
        product = left * right
        if layer == 1:
            captured["source"] = product[0, position, units].float().clone()
            if replacement is not None:
                product = product.clone()
                for unit, value in replacement.items():
                    product[0, position, unit] = value
        if layer == 2:
            captured.update({
                "mixed2": mixed[0, position].float().clone(),
                "attention2": attention_write[0, position].float().clone(),
                "input2": normalized[0, position].float().clone(),
                "left2": left[0, position].float().clone(),
                "right2": right[0, position].float().clone(),
                "product2": product[0, position].float().clone(),
            })
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
        if layer == 2:
            captured["state2"] = x[0, position].float().clone()
            if stop_at_layer2:
                return captured
    raw = model.lm_head(F.rms_norm(x, (d,))[:, -1]).float()
    captured["logits"] = (30 * torch.tanh(raw / 30))[0].clone()
    return captured


def main():
    args = parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    discovery = json.loads(args.discovery.read_text())
    row = next(item for item in discovery["categories"] if item["name"] == "suffix_ing")
    units = row["units"]
    decoder = {"mean": np.asarray(row["decoder_mean"], dtype=np.float32),
               "std": np.asarray(row["decoder_std"], dtype=np.float32),
               "weight": np.asarray(row["decoder_weight"], dtype=np.float32)}
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    ing_index = [item["name"] for item in discovery["categories"]].index("suffix_ing")
    ing_flags = np.asarray([token_flags(tokenizer.decode([i]))[ing_index]
                            for i in range(tokenizer.vocab_size)], dtype=bool)
    model = load_full_model(args.device)
    for module in (model.embedding, model.blocks, model.lm_head):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    fineweb = np.load(args.fineweb, mmap_mode="r")
    corpora = {
        "fineweb_train": (fineweb[11000:14000], 11000),
        "fineweb_test": (fineweb[15000:18000], 15000),
        "wikitext103": (np.load(args.ood_dir / "wikitext103_1200x129.npy", mmap_mode="r"), 0),
        "tinystories": (np.load(args.ood_dir / "tinystories_1200x129.npy", mmap_mode="r"), 0),
    }
    positional_mean = positional_source_mean(model, fineweb[10000:10400], units, args)
    site_manifest = []
    arrays = {key: [] for key in (
        "source_delta", "target_delta", "individual_target_delta", "mixed2_delta",
        "attention2_delta", "input2_delta", "state2_delta", "input2_base", "left2_base", "right2_base",
        "base_logits", "changed_logits",
    )}
    for split, (rows, row_offset) in corpora.items():
        sites = score_rows(model, rows, units, decoder, ing_flags, args)
        print(f"selected {len(sites)} quantile sites for {split}", flush=True)
        for number, (score, row_index, position, _) in enumerate(sites):
            raw = np.asarray(rows[row_index, :position + 1], dtype=np.int64)
            ids = torch.as_tensor(raw, device=args.device)[None]
            base = trace(model, ids, units)
            replacement = {unit: positional_mean[position, column]
                           for column, unit in enumerate(units)}
            changed = trace(model, ids, units, replacement=replacement)
            individual = []
            for unit in units:
                one = trace(model, ids, units, replacement={unit: replacement[unit]},
                            stop_at_layer2=True)
                individual.append((base["product2"] - one["product2"]).cpu().to(torch.float16).numpy())
            arrays["source_delta"].append((base["source"] - positional_mean[position]).cpu().numpy())
            arrays["target_delta"].append((base["product2"] - changed["product2"]).cpu().to(torch.float16).numpy())
            arrays["individual_target_delta"].append(np.stack(individual))
            arrays["mixed2_delta"].append((base["mixed2"] - changed["mixed2"]).cpu().to(torch.float16).numpy())
            arrays["attention2_delta"].append((base["attention2"] - changed["attention2"]).cpu().to(torch.float16).numpy())
            arrays["input2_delta"].append((base["input2"] - changed["input2"]).cpu().to(torch.float16).numpy())
            arrays["state2_delta"].append((base["state2"] - changed["state2"]).cpu().numpy())
            arrays["input2_base"].append(base["input2"].cpu().to(torch.float16).numpy())
            arrays["left2_base"].append(base["left2"].cpu().to(torch.float16).numpy())
            arrays["right2_base"].append(base["right2"].cpu().to(torch.float16).numpy())
            arrays["base_logits"].append(base["logits"].cpu().to(torch.float16).numpy())
            arrays["changed_logits"].append(changed["logits"].cpu().to(torch.float16).numpy())
            site_manifest.append({
                "split": split, "row": row_offset + row_index, "local_row": row_index,
                "position": position, "score": score, "token_id": int(raw[-1]),
                "token": tokenizer.decode([int(raw[-1])]),
                "context": tokenizer.decode(raw[max(0, len(raw) - 32):].tolist()),
            })
            if (number + 1) % 16 == 0:
                print(f"{split}: traced {number + 1}/{len(sites)}", flush=True)

    np.savez_compressed(args.output / "causal_tensor.npz",
                        **{key: np.stack(value) for key, value in arrays.items()})
    manifest = {
        "source": "frozen suffix_ing circuit",
        "units": units,
        "sites": site_manifest,
        "split_counts": {split: args.sites_per_split for split in corpora},
        "selection": "uniform quantiles of frozen suffix_ing score among true -ing tokens",
        "fineweb_fit_rows": [11000, 14000],
        "fineweb_test_rows": [15000, 18000],
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(site_manifest)} sites in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
