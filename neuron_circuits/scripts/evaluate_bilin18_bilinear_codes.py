#!/usr/bin/env python3
"""End-to-end and semantic evaluation of Bilin18 bilinear sparse codes."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import TOKENIZER
from collect_bilin18_sparse_code_data import OOD
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from train_bilin18_bilinear_codes import ROOT, SparseBilinearDictionary
from train_bilin18_sparse_codes import (SPLITS, TopKAutoencoder, category_analysis)


SPARSE_ROOT = Path("artifacts/bilin18_sparse_codes")


def load_dictionary(path, input_dim, features, output_dim, k):
    model = SparseBilinearDictionary(input_dim, features, output_dim, k, 0)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


@torch.inference_mode()
def forward(model, ids, replacement_name, replacement, h_mean, h_std):
    dimension = model.config.n_embd
    x = F.rms_norm(model.embedding(ids), (dimension,))
    x0, value_bus = x, None
    for layer, block in enumerate(model.blocks):
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention_write
        normalized = F.rms_norm(pre, (dimension,))
        if layer != 1 or replacement_name == "baseline":
            product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
            write = block.mlp.Down(product) + block.mlp.Down_bias
        elif replacement_name == "linear_sae":
            product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
            standardized = (product.float() - h_mean) / h_std
            reconstructed, _ = replacement(standardized.flatten(0, 1), 128)
            reconstructed = reconstructed.view_as(product).float() * h_std + h_mean
            write = block.mlp.Down(reconstructed.to(product.dtype)) + block.mlp.Down_bias
        elif replacement_name == "bilinear_sae":
            reconstructed, _ = replacement(normalized.float().flatten(0, 1))
            reconstructed = reconstructed.view(*normalized.shape[:-1], 4608) * h_std + h_mean
            write = block.mlp.Down(reconstructed.to(normalized.dtype)) + block.mlp.Down_bias
        else:
            write, _ = replacement(normalized.float().flatten(0, 1))
            write = write.view_as(normalized).to(normalized.dtype)
        x = pre + write
    final = F.rms_norm(x, (dimension,))
    raw = model.lm_head(final).float()
    return 30 * torch.tanh(raw / 30)


@torch.inference_mode()
def evaluate_corpus(model, replacements, rows, h_mean, h_std, batch_size=2):
    totals = {name: {"nll": 0., "kl": 0., "same": 0, "n": 0}
              for name in replacements}
    baseline_nll = 0.; baseline_n = 0
    for start in range(0, len(rows), batch_size):
        raw = np.asarray(rows[start:start + batch_size, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :128], device="cuda")
        targets = torch.as_tensor(raw[:, 1:129], device="cuda")
        base_logits = forward(model, ids, "baseline", None, h_mean, h_std)
        base_logp = F.log_softmax(base_logits, dim=-1)
        base_p = base_logp.exp()
        baseline_nll += float(F.nll_loss(
            base_logp.flatten(0, 1), targets.flatten(), reduction="sum"))
        baseline_n += targets.numel()
        base_top = base_logits.argmax(-1)
        for name, replacement in replacements.items():
            logits = forward(model, ids, name, replacement, h_mean, h_std)
            logp = F.log_softmax(logits, dim=-1)
            row = totals[name]
            row["nll"] += float(F.nll_loss(
                logp.flatten(0, 1), targets.flatten(), reduction="sum"))
            row["kl"] += float((base_p * (base_logp - logp)).sum())
            row["same"] += int((base_top == logits.argmax(-1)).sum())
            row["n"] += targets.numel()
        if start and start % 16 == 0:
            print("end_to_end", start, "/", len(rows), flush=True)
    base_ce = baseline_nll / baseline_n
    return {"baseline_cross_entropy": base_ce, "replacements": {
        name: {
            "cross_entropy": row["nll"] / row["n"],
            "cross_entropy_delta": row["nll"] / row["n"] - base_ce,
            "perplexity_ratio": math.exp(min(20, row["nll"] / row["n"] - base_ce)),
            "mean_kl_from_full": row["kl"] / row["n"],
            "top1_agreement": row["same"] / row["n"],
        } for name, row in totals.items()}}


@torch.inference_mode()
def encode_dictionary(model, values, batch_size=256):
    model.cuda(); codes = []
    for start in range(0, len(values), batch_size):
        code = model.encode(values[start:start + batch_size].cuda())
        codes.append(code.to(torch.float16).cpu())
    return torch.cat(codes)


def semantic_evaluation(models):
    input_archive = np.load(SPARSE_ROOT / "layer1_inputs_large.npz")
    product_archive = np.load(SPARSE_ROOT / "natural_large.npz")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    train_x = torch.from_numpy(
        input_archive["fineweb_train__input1"][:32768].astype(np.float32))
    test_x = {split: torch.from_numpy(
        input_archive[f"{split}__input1"].astype(np.float32)) for split in SPLITS}
    train_ids = product_archive["fineweb_train__token_ids"][:32768]
    test_ids = {split: product_archive[f"{split}__token_ids"] for split in SPLITS}
    result = {}
    for name, model in models.items():
        train_code = encode_dictionary(model, train_x)
        test_codes = {split: encode_dictionary(model, value)
                      for split, value in test_x.items()}
        result[name] = category_analysis(
            train_code, test_codes, train_ids, test_ids, tokenizer)
        model.cpu()
    return result


def main():
    started = time.time(); ROOT.mkdir(parents=True, exist_ok=True)
    product_archive = np.load(SPARSE_ROOT / "natural_large.npz")
    train_h = torch.from_numpy(
        product_archive["fineweb_train__h1"].astype(np.float32))
    h_mean = train_h.mean(0).cuda(); h_std = train_h.std(0).clamp_min(1e-4).cuda()
    del train_h
    linear = TopKAutoencoder(4608, 6144, 128, 101)
    linear.load_state_dict(torch.load(
        SPARSE_ROOT / "activation_sae_large_seed101.pt", map_location="cpu",
        weights_only=True))
    bilinear = load_dictionary(ROOT / "bilinear_sae_seed201.pt", 1152, 2048, 4608, 128)
    replacements = {"linear_sae": linear.cuda(), "bilinear_sae": bilinear.cuda()}
    replacement_models = {}
    for features in (128, 512, 2048):
        name = f"m{features}_k128"
        item = load_dictionary(ROOT / f"replacement_{name}.pt",
                               1152, features, 1152, 128)
        replacements[name] = item.cuda(); replacement_models[name] = item
    # Inference-time sparsity red team for the largest replacement. These clones
    # share identical weights; only the number of retained product features changes.
    for active in (32, 64, 256, 512, 2048):
        name = f"m2048_evalk{active}"
        item = load_dictionary(ROOT / "replacement_m2048_k128.pt",
                               1152, 2048, 1152, active)
        replacements[name] = item.cuda()
    scratch = load_dictionary(ROOT / "replacement_m2048_k128_scratch2401.pt",
                              1152, 2048, 1152, 128)
    replacements["m2048_k128_scratch"] = scratch.cuda()

    model = load_full_model("cuda")
    fineweb = np.load(DEFAULT_CORPUS, mmap_mode="r")
    corpora = {
        "fineweb": fineweb[18000:18064],
        "wikitext103": np.load(OOD / "wikitext103_1200x129.npy", mmap_mode="r")[1024:1088],
        "tinystories": np.load(OOD / "tinystories_1200x129.npy", mmap_mode="r")[1024:1088],
    }
    end_to_end = {}
    for name, rows in corpora.items():
        print("evaluating", name, flush=True)
        end_to_end[name] = evaluate_corpus(
            model, replacements, rows, h_mean, h_std)

    # Semantic meanings of the new bilinear coordinates. Include the SAE and the
    # largest replacement; smaller students inherit subsets of the same native init.
    semantics = semantic_evaluation({
        "bilinear_sae": bilinear.cpu(),
        "replacement_m2048_k128": replacement_models["m2048_k128"].cpu(),
        "replacement_m2048_scratch": scratch.cpu(),
    })
    result = {"end_to_end": end_to_end, "semantics": semantics,
              "eval_rows_per_corpus": 64, "tokens_per_corpus": 8192,
              "elapsed_seconds": time.time() - started}
    (ROOT / "evaluation_results.json").write_text(json.dumps(result, indent=2))
    print(f"wrote end-to-end results in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
