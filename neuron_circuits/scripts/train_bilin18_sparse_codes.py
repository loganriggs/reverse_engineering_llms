#!/usr/bin/env python3
"""Train matched sparse representations and transcoders for the Bilin18 path."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from audit_bilin18_mlp_discriminators import SNAPSHOT, TOKENIZER
from discover_bilin18_weight_only_circuits import auc, token_flags


ROOT = Path("artifacts/bilin18_sparse_codes")
SPLITS = ("fineweb_test", "wikitext103", "tinystories")


class TopKAutoencoder(nn.Module):
    def __init__(self, dimension, features, k, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Linear(dimension, features)
        self.decoder = nn.Linear(features, dimension, bias=False)
        self.k = k
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.encoder.weight.T, dim=0))

    def encode(self, x, k=None):
        values = F.relu(self.encoder(x))
        keep = min(k or self.k, values.shape[-1])
        top = torch.topk(values, keep, dim=-1)
        sparse = torch.zeros_like(values)
        sparse.scatter_(1, top.indices, top.values)
        return sparse

    def forward(self, x, k=None):
        code = self.encode(x, k)
        return self.decoder(code), code

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))


class SparseTranscoder(nn.Module):
    def __init__(self, source_dim, context_dim, target_dim, features, k, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.source = nn.Linear(source_dim, features)
        self.context = nn.Linear(context_dim, features, bias=False) if context_dim else None
        self.decoder = nn.Linear(features, target_dim, bias=False)
        self.k = k
        nn.init.kaiming_uniform_(self.source.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    def forward(self, source, context=None):
        pre = self.source(source)
        if self.context is not None:
            pre = pre + self.context(context)
        values = F.relu(pre)
        top = torch.topk(values, self.k, dim=-1)
        code = torch.zeros_like(values)
        code.scatter_(1, top.indices, top.values)
        return self.decoder(code), code

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))


class BilinearSparseTranscoder(nn.Module):
    """Sparse dictionary whose feature gates explicitly multiply delta and context."""

    def __init__(self, input_dim, target_dim, features, k, seed):
        super().__init__()
        torch.manual_seed(seed)
        # The three terms mirror the exact finite change of a bilinear product:
        # delta-left * base-right + base-left * delta-right - delta-left * delta-right.
        self.delta_left = nn.Linear(input_dim, features)
        self.context_right = nn.Linear(input_dim, features)
        self.context_left = nn.Linear(input_dim, features)
        self.delta_right = nn.Linear(input_dim, features)
        self.quad_left = nn.Linear(input_dim, features, bias=False)
        self.quad_right = nn.Linear(input_dim, features, bias=False)
        self.context = self.context_left  # Signals contextual input to the shared trainer.
        self.decoder = nn.Linear(features, target_dim, bias=False)
        self.k = k
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))

    def forward(self, delta, context):
        values = (self.delta_left(delta) * self.context_right(context) +
                  self.context_left(context) * self.delta_right(delta) -
                  self.quad_left(delta) * self.quad_right(delta))
        top = torch.topk(values.abs(), self.k, dim=-1)
        code = torch.zeros_like(values)
        code.scatter_(1, top.indices, values.gather(1, top.indices))
        return self.decoder(code), code

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))


def train(model, arrays, steps, batch_size, lr, context=None, label="model"):
    model.cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .95))
    generator = torch.Generator().manual_seed(20261019)
    losses = []
    for step in range(steps):
        index = torch.randint(len(arrays[0]), (batch_size,), generator=generator)
        batch = [array[index].cuda(non_blocking=True) for array in arrays]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if context is None:
                reconstruction, _ = model(batch[0])
                loss = F.mse_loss(reconstruction.float(), batch[0])
            else:
                reconstruction, _ = model(batch[0], batch[1] if model.context is not None else None)
                loss = F.mse_loss(reconstruction.float(), batch[-1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.normalize_decoder()
        losses.append(float(loss.detach()))
        if (step + 1) % 100 == 0:
            print(label, step + 1, f"loss={np.mean(losses[-100:]):.5f}", flush=True)
    return losses


@torch.inference_mode()
def reconstruct_sae(model, values, k, batch_size=128):
    output, codes = [], []
    model.eval()
    for start in range(0, len(values), batch_size):
        x = values[start:start + batch_size].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reconstruction, code = model(x, k)
        output.append(reconstruction.float().cpu())
        codes.append(code.to(torch.float16).cpu())
    return torch.cat(output), torch.cat(codes)


@torch.inference_mode()
def reconstruct_transcoder(model, source, context, batch_size=128):
    output = []
    model.eval()
    for start in range(0, len(source), batch_size):
        x = source[start:start + batch_size].cuda()
        c = context[start:start + batch_size].cuda() if model.context is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, _ = model(x, c)
        output.append(prediction.float().cpu())
    return torch.cat(output)


def fidelity(target, prediction):
    target, prediction = target.float(), prediction.float()
    per = 1 - (target - prediction).square().sum(1) / target.square().sum(1).clamp_min(1e-20)
    cosine = F.cosine_similarity(target, prediction, dim=1)
    return {"median_fidelity": float(per.median()), "mean_fidelity": float(per.mean()),
            "median_cosine": float(cosine.median())}


def normalize(train, values, centered=True):
    mean = train.mean(0) if centered else torch.zeros(train.shape[1])
    std = train.std(0).clamp_min(1e-4)
    return [(value - mean) / std for value in values], mean, std


def category_analysis(train_code, test_codes, train_ids, test_ids, tokenizer):
    flags = np.stack([token_flags(tokenizer.decode([i])) for i in range(tokenizer.vocab_size)])
    category_names = (
        "newline", "opening_delimiter", "closing_delimiter", "quote", "code_operator",
        "url_web", "non_ascii", "word_continuation", "suffix_ing", "suffix_ed",
        "suffix_ly", "all_caps", "comma_period", "colon_semicolon",
    )
    train_labels = flags[np.asarray(train_ids)]
    code = train_code.float().numpy()
    code_mean, code_std = code.mean(0), code.std(0).clip(1e-5)
    z = (code - code_mean) / code_std
    result = {}
    for ci, name in enumerate(category_names):
        labels = train_labels[:, ci].astype(bool)
        if labels.sum() < 8:
            continue
        gaps = z[labels].mean(0) - z[~labels].mean(0)
        feature = int(np.argmax(np.abs(gaps)))
        sign = float(np.sign(gaps[feature]) or 1)
        row = {"feature": feature, "train_gap": float(gaps[feature]), "auc": {}}
        for split, test_code in test_codes.items():
            labels_test = flags[np.asarray(test_ids[split]), ci]
            score = sign * ((test_code[:, feature].float().numpy() - code_mean[feature]) /
                            code_std[feature])
            row["auc"][split] = auc(labels_test, score)
        result[name] = row
    return result


def decoder_stability(first, second, chunk=128):
    first = F.normalize(first.decoder.weight.detach().float().T.cpu(), dim=1)
    second = F.normalize(second.decoder.weight.detach().float().T.cpu(), dim=1)
    maxima = []
    for start in range(0, len(first), chunk):
        maxima.append((first[start:start + chunk] @ second.T).abs().max(1).values)
    values = torch.cat(maxima)
    return {"median_best_cosine": float(values.median()),
            "mean_best_cosine": float(values.mean()),
            "fraction_above_0_8": float((values > .8).float().mean())}


def main():
    started = time.time()
    data = np.load(ROOT / "data.npz")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True)
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    down1 = state["transformer.h.1.mlp.Down.weight"].float()
    left2 = state["transformer.h.2.mlp.Left.weight"].float()
    right2 = state["transformer.h.2.mlp.Right.weight"].float()
    del state

    # Natural activation SAE.
    natural_raw = {split: torch.from_numpy(data[f"natural__{split}__h1"].astype(np.float32))
                   for split in ("fineweb_train",) + SPLITS}
    natural_values, natural_mean, natural_std = normalize(
        natural_raw["fineweb_train"], [natural_raw["fineweb_train"]] +
        [natural_raw[split] for split in SPLITS])
    natural = {"fineweb_train": natural_values[0]} | {
        split: natural_values[index + 1] for index, split in enumerate(SPLITS)}
    activation_models, activation_losses = [], []
    for seed in (1, 2):
        model = TopKAutoencoder(4608, 6144, 128, seed)
        losses = train(model, [natural["fineweb_train"]], 400, 64, 3e-4,
                       label=f"activation_sae_seed{seed}")
        activation_models.append(model.cpu())
        activation_losses.append(losses)

    results = {"activation_sae": {"seeds": {}, "native": {}},
               "causal_delta_sae": {}, "transcoders": {}, "elapsed_seconds": None}
    test_ids = {split: data[f"natural__{split}__token_ids"] for split in SPLITS}
    decoder_write = down1
    for seed_index, model in enumerate(activation_models):
        model.cuda()
        seed_result = {"reconstruction": {}, "write_reconstruction": {}}
        codes_by_split = {}
        for k in (32, 128, 512):
            for split in SPLITS:
                reconstruction, code = reconstruct_sae(model, natural[split], k)
                if k == 128:
                    codes_by_split[split] = code
                seed_result["reconstruction"].setdefault(str(k), {})[split] = fidelity(
                    natural[split], reconstruction)
                raw_error = (natural[split] - reconstruction) * natural_std
                target_write = (natural[split] * natural_std) @ decoder_write.T
                predicted_write = target_write - raw_error @ decoder_write.T
                seed_result["write_reconstruction"].setdefault(str(k), {})[split] = fidelity(
                    target_write, predicted_write)
        train_reconstruction, train_code = reconstruct_sae(model, natural["fineweb_train"], 128)
        seed_result["categories"] = category_analysis(
            train_code, codes_by_split, data["natural__fineweb_train__token_ids"],
            test_ids, tokenizer)
        seed_result["final_loss"] = float(np.mean(activation_losses[seed_index][-50:]))
        results["activation_sae"]["seeds"][str(seed_index + 1)] = seed_result
        torch.save(model.state_dict(), ROOT / f"activation_sae_seed{seed_index + 1}.pt")
        model.cpu()
    results["activation_sae"]["stability"] = decoder_stability(*activation_models)
    results["activation_sae"]["native_categories"] = category_analysis(
        natural["fineweb_train"], {split: natural[split] for split in SPLITS},
        data["natural__fineweb_train__token_ids"], test_ids, tokenizer)

    for k in (32, 128, 512):
        results["activation_sae"]["native"][str(k)] = {}
        for split in SPLITS:
            x = natural[split]
            selected = torch.topk(x.abs(), k, dim=1)
            reconstruction = torch.zeros_like(x).scatter(1, selected.indices,
                                                          x.gather(1, selected.indices))
            raw_error = (x - reconstruction) * natural_std
            target_write = (x * natural_std) @ decoder_write.T
            predicted_write = target_write - raw_error @ decoder_write.T
            results["activation_sae"]["native"][str(k)][split] = {
                "activation": fidelity(x, reconstruction),
                "write": fidelity(target_write, predicted_write),
            }

    # Causal post-attention input-delta SAE.
    causal_input_raw = {split: torch.from_numpy(
        data[f"causal__{split}__input_delta"].astype(np.float32))
        for split in ("fineweb_train",) + SPLITS}
    input_values, input_mean, input_std = normalize(
        causal_input_raw["fineweb_train"], [causal_input_raw["fineweb_train"]] +
        [causal_input_raw[split] for split in SPLITS], centered=False)
    causal_input = {"fineweb_train": input_values[0]} | {
        split: input_values[index + 1] for index, split in enumerate(SPLITS)}
    delta_sae = TopKAutoencoder(1152, 2048, 32, 3)
    delta_losses = train(delta_sae, [causal_input["fineweb_train"]], 700, 128, 5e-4,
                         label="causal_delta_sae")
    delta_sae.cuda()
    results["causal_delta_sae"]["final_loss"] = float(np.mean(delta_losses[-50:]))

    def causal_fidelity(split, reconstruction):
        input_reconstruction = reconstruction * input_std
        input_target = causal_input_raw[split]
        context = torch.from_numpy(data[f"causal__{split}__context"].astype(np.float32))
        base_left = context @ left2.T
        base_right = context @ right2.T
        delta_left = input_reconstruction @ left2.T
        delta_right = input_reconstruction @ right2.T
        product_prediction = (delta_left * base_right + base_left * delta_right -
                              delta_left * delta_right)
        product_target = torch.from_numpy(
            data[f"causal__{split}__target_delta"].astype(np.float32))
        return {"input": fidelity(input_target, input_reconstruction),
                "product": fidelity(product_target, product_prediction)}

    for k in (4, 16, 32, 128):
        results["causal_delta_sae"].setdefault("ranks", {})[str(k)] = {}
        for split in SPLITS:
            reconstruction, _ = reconstruct_sae(delta_sae, causal_input[split], k)
            results["causal_delta_sae"]["ranks"][str(k)][split] = causal_fidelity(
                split, reconstruction)

    # Matched controls: a dense PCA subspace and sparse native coordinates.
    # Both operate on the same standardized intervention delta as the SAE.
    _, _, pca_vectors = torch.pca_lowrank(
        causal_input["fineweb_train"].cuda(), q=128, center=False, niter=4)
    pca_vectors = pca_vectors.cpu()
    results["causal_delta_sae"]["baselines"] = {"pca": {}, "native": {}}
    for k in (4, 16, 32, 128):
        results["causal_delta_sae"]["baselines"]["pca"][str(k)] = {}
        results["causal_delta_sae"]["baselines"]["native"][str(k)] = {}
        basis = pca_vectors[:, :k]
        for split in SPLITS:
            x = causal_input[split]
            pca_reconstruction = (x @ basis) @ basis.T
            selected = torch.topk(x.abs(), k, dim=1)
            native_reconstruction = torch.zeros_like(x).scatter(
                1, selected.indices, x.gather(1, selected.indices))
            results["causal_delta_sae"]["baselines"]["pca"][str(k)][split] = \
                causal_fidelity(split, pca_reconstruction)
            results["causal_delta_sae"]["baselines"]["native"][str(k)][split] = \
                causal_fidelity(split, native_reconstruction)
    torch.save(delta_sae.cpu().state_dict(), ROOT / "causal_delta_sae.pt")

    # Source-only and context-conditioned sparse transcoders.
    source_raw = {split: torch.from_numpy(data[f"causal__{split}__source_delta"].astype(np.float32))
                  for split in ("fineweb_train",) + SPLITS}
    context_raw = {split: torch.from_numpy(data[f"causal__{split}__context"].astype(np.float32))
                   for split in ("fineweb_train",) + SPLITS}
    target_raw = {split: torch.from_numpy(data[f"causal__{split}__target_delta"].astype(np.float32))
                  for split in ("fineweb_train",) + SPLITS}
    source_values, source_mean, source_std = normalize(
        source_raw["fineweb_train"], [source_raw["fineweb_train"]] +
        [source_raw[split] for split in SPLITS], centered=False)
    context_values, context_mean, context_std = normalize(
        context_raw["fineweb_train"], [context_raw["fineweb_train"]] +
        [context_raw[split] for split in SPLITS], centered=True)
    target_values, target_mean, target_std = normalize(
        target_raw["fineweb_train"], [target_raw["fineweb_train"]] +
        [target_raw[split] for split in SPLITS], centered=False)
    source = {"fineweb_train": source_values[0]} | {
        split: source_values[index + 1] for index, split in enumerate(SPLITS)}
    context = {"fineweb_train": context_values[0]} | {
        split: context_values[index + 1] for index, split in enumerate(SPLITS)}
    target = {"fineweb_train": target_values[0]} | {
        split: target_values[index + 1] for index, split in enumerate(SPLITS)}
    for name, context_dim in (("source_only", 0), ("context_conditioned", 1152)):
        transcoder = SparseTranscoder(4608, context_dim, 4608, 2048, 32, 10 + context_dim)
        losses = train(transcoder, [source["fineweb_train"], context["fineweb_train"],
                                    target["fineweb_train"]], 700, 64, 3e-4,
                       context=True, label=name)
        row = {"final_loss": float(np.mean(losses[-50:])), "transfer": {}}
        for split in SPLITS:
            prediction = reconstruct_transcoder(transcoder, source[split], context[split])
            row["transfer"][split] = fidelity(target[split], prediction)
        results["transcoders"][name] = row
        torch.save(transcoder.cpu().state_dict(), ROOT / f"transcoder_{name}.pt")

    # Fair architecture comparison on the post-attention delta: additive gates versus
    # gates with the multiplication required by the model's actual layer-2 MLP.
    for name, transcoder in (
            ("input_context_additive",
             SparseTranscoder(1152, 1152, 4608, 2048, 32, 30)),
            ("input_context_bilinear",
             BilinearSparseTranscoder(1152, 4608, 2048, 32, 31))):
        losses = train(transcoder, [causal_input["fineweb_train"],
                                    context["fineweb_train"], target["fineweb_train"]],
                       1000, 64, 3e-4, context=True, label=name)
        row = {"final_loss": float(np.mean(losses[-50:])), "transfer": {}}
        for split in SPLITS:
            prediction = reconstruct_transcoder(
                transcoder, causal_input[split], context[split])
            row["transfer"][split] = fidelity(target[split], prediction)
        results["transcoders"][name] = row
        torch.save(transcoder.cpu().state_dict(), ROOT / f"transcoder_{name}.pt")

    results["elapsed_seconds"] = time.time() - started
    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote sparse-code results in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
