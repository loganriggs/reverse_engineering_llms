#!/usr/bin/env python3
"""Direct CP and LL1 decompositions of Bilin18 layer 1's implicit weight tensor."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from train_bilin18_sparse_codes import SPLITS, fidelity


ROOT = Path("artifacts/bilin18_tensor_decomposition")
DATA = Path("artifacts/bilin18_sparse_codes")


class CPBilinear(nn.Module):
    def __init__(self, dimension, rank, seed):
        super().__init__(); torch.manual_seed(seed)
        self.left = nn.Parameter(torch.randn(rank, dimension) / math.sqrt(dimension))
        self.right = nn.Parameter(torch.randn(rank, dimension) / math.sqrt(dimension))
        self.output = nn.Parameter(torch.randn(dimension, rank) / math.sqrt(dimension))
        self.gain = nn.Parameter(torch.ones(rank))
        self.bias = nn.Parameter(torch.zeros(dimension), requires_grad=False)
        self.rank = rank

    def factors(self):
        return (F.normalize(self.left, dim=1), F.normalize(self.right, dim=1),
                F.normalize(self.output, dim=0), self.gain)

    def bilinear(self, x, y):
        left, right, output, gain = self.factors()
        feature = self.bilinear_features(x, y, left, right, gain)
        return F.linear(feature, output)

    def bilinear_features(self, x, y, left=None, right=None, gain=None):
        if left is None:
            left, right, _, gain = self.factors()
        return .5 * (F.linear(x, left) * F.linear(y, right) +
                     F.linear(y, left) * F.linear(x, right)) * gain

    def bilinear_sparse(self, x, y, k):
        left, right, output, gain = self.factors()
        feature = self.bilinear_features(x, y, left, right, gain)
        selected = torch.topk(feature.abs(), min(k, self.rank), dim=1)
        sparse = torch.zeros_like(feature)
        sparse.scatter_(1, selected.indices, feature.gather(1, selected.indices))
        return F.linear(sparse, output), sparse

    def forward_sparse(self, x, k):
        output, code = self.bilinear_sparse(x, x, k)
        return output + self.bias, code

    def forward(self, x):
        return self.bilinear(x, x) + self.bias


class LL1Bilinear(nn.Module):
    """Sum of R multilinear-rank-(L,L,1) blocks, symmetric in the input modes."""

    def __init__(self, dimension, blocks, block_rank, seed):
        super().__init__(); torch.manual_seed(seed)
        self.left = nn.Parameter(torch.randn(blocks, block_rank, dimension) /
                                 math.sqrt(dimension))
        self.right = nn.Parameter(torch.randn(blocks, block_rank, dimension) /
                                  math.sqrt(dimension))
        self.output = nn.Parameter(torch.randn(dimension, blocks) / math.sqrt(dimension))
        self.gain = nn.Parameter(torch.ones(blocks, block_rank))
        self.bias = nn.Parameter(torch.zeros(dimension), requires_grad=False)
        self.blocks, self.block_rank = blocks, block_rank

    def factors(self):
        left = F.normalize(self.left, dim=2)
        right = F.normalize(self.right, dim=2)
        output = F.normalize(self.output, dim=0)
        return left, right, output, self.gain

    def bilinear(self, x, y):
        left, right, output, gain = self.factors()
        flat_left, flat_right = left.flatten(0, 1), right.flatten(0, 1)
        feature = .5 * (F.linear(x, flat_left) * F.linear(y, flat_right) +
                        F.linear(y, flat_left) * F.linear(x, flat_right))
        blocks = (feature.view(len(x), self.blocks, self.block_rank) * gain).sum(2)
        return F.linear(blocks, output)

    def forward(self, x):
        return self.bilinear(x, x) + self.bias


@torch.no_grad()
def initialize_cp(model, left, right, down, ranking):
    units = ranking[:model.rank]
    l, r, d = left[units], right[units], down[:, units]
    ln, rn, dn = l.norm(dim=1).clamp_min(1e-8), r.norm(dim=1).clamp_min(1e-8), \
        d.norm(dim=0).clamp_min(1e-8)
    model.left.copy_(l / ln[:, None]); model.right.copy_(r / rn[:, None])
    model.output.copy_(d / dn[None]); model.gain.copy_(ln * rn * dn)


@torch.inference_mode()
def output_basis(left, right, down, dimension, blocks, samples=8192, batch_size=128):
    outputs = []
    generator = torch.Generator(device="cuda").manual_seed(20261024)
    for _ in range(0, samples, batch_size):
        x = torch.randn(batch_size, dimension, generator=generator, device="cuda")
        y = torch.randn(batch_size, dimension, generator=generator, device="cuda")
        feature = .5 * ((x @ left.T) * (y @ right.T) +
                        (y @ left.T) * (x @ right.T))
        outputs.append((feature @ down.T).cpu())
    outputs = torch.cat(outputs)
    _, _, vectors = torch.pca_lowrank(outputs.cuda(), q=blocks, center=False, niter=4)
    return vectors.cpu()


@torch.no_grad()
def initialize_ll1(model, left, right, down, basis):
    """Project onto output directions, then eigentruncate each symmetric input matrix."""
    model.output.copy_(basis)
    left, right, down, basis = [value.cuda() for value in (left, right, down, basis)]
    for block in range(model.blocks):
        coefficients = basis[:, block] @ down
        matrix = .5 * (left.T @ (coefficients[:, None] * right) +
                       right.T @ (coefficients[:, None] * left))
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        chosen = eigenvalues.abs().topk(model.block_rank).indices
        values = eigenvalues[chosen]
        vectors = eigenvectors[:, chosen].T
        model.left[block].copy_(vectors)
        model.right[block].copy_(vectors)
        model.gain[block].copy_(values)
        if (block + 1) % 16 == 0:
            print("LL1 initialize", block + 1, "/", model.blocks, flush=True)


def teacher_bilinear(x, y, left, right, down):
    feature = .5 * ((x @ left.T) * (y @ right.T) +
                    (y @ left.T) * (x @ right.T))
    return feature @ down.T


def train_tensor(model, left, right, down, steps, seed):
    model.cuda().train(); left, right, down = left.cuda(), right.cuda(), down.cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-4, betas=(.9, .95), weight_decay=1e-7)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    validation_generator = torch.Generator(device="cuda").manual_seed(seed + 100000)
    validation_x = torch.randn(512, left.shape[1], generator=validation_generator,
                               device="cuda")
    validation_y = torch.randn(512, left.shape[1], generator=validation_generator,
                               device="cuda")
    with torch.no_grad():
        validation_target = teacher_bilinear(
            validation_x, validation_y, left, right, down)
        initial_prediction = model.bilinear(validation_x, validation_y)
        best_loss = float((initial_prediction - validation_target).square().sum() /
                          validation_target.square().sum().clamp_min(1e-8))
    losses = []
    best_state = {key: value.detach().cpu().clone()
                  for key, value in model.state_dict().items()}
    for step in range(1, steps + 1):
        x = torch.randn(128, left.shape[1], generator=generator, device="cuda")
        y = torch.randn_like(x, generator=generator)
        with torch.no_grad():
            target = teacher_bilinear(x, y, left, right, down)
        prediction = model.bilinear(x, y)
        loss = ((prediction - target).square().sum() /
                target.square().sum().clamp_min(1e-8))
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            with torch.no_grad():
                validation_prediction = model.bilinear(validation_x, validation_y)
                validation_loss = float((validation_prediction - validation_target).square().sum() /
                                        validation_target.square().sum().clamp_min(1e-8))
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
        if step % 250 == 0:
            print(type(model).__name__, step, np.mean(losses[-100:]),
                  "best", best_loss, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return losses, best_loss


@torch.no_grad()
def tensor_probe_metrics(model, left, right, down, seed, batches=16):
    model.cuda().eval(); left, right, down = left.cuda(), right.cuda(), down.cuda()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    targets, predictions = [], []
    for _ in range(batches):
        x = torch.randn(128, left.shape[1], generator=generator, device="cuda")
        y = torch.randn_like(x, generator=generator)
        targets.append(teacher_bilinear(x, y, left, right, down).cpu())
        predictions.append(model.bilinear(x, y).cpu())
    return fidelity(torch.cat(targets), torch.cat(predictions))


@torch.no_grad()
def natural_metrics(model, x, h, down, bias, batch_size=256):
    model.cuda().eval(); down, bias = down.cuda(), bias.cuda()
    targets, predictions = [], []
    for start in range(0, len(x), batch_size):
        bx, bh = x[start:start + batch_size].cuda(), h[start:start + batch_size].cuda()
        targets.append(F.linear(bh, down, bias).cpu())
        predictions.append(model(bx).cpu())
    return fidelity(torch.cat(targets), torch.cat(predictions))


def save_model(model, name):
    configuration = {"type": type(model).__name__}
    if isinstance(model, CPBilinear):
        configuration["rank"] = model.rank
    else:
        configuration.update({"blocks": model.blocks, "block_rank": model.block_rank})
    torch.save({"state_dict": model.cpu().state_dict(), "configuration": configuration},
               ROOT / f"{name}.pt")


def main():
    ROOT.mkdir(parents=True, exist_ok=True); started = time.time()
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    left = state["transformer.h.1.mlp.Left.weight"].float()
    right = state["transformer.h.1.mlp.Right.weight"].float()
    down = state["transformer.h.1.mlp.Down.weight"].float()
    bias = state["transformer.h.1.mlp.Down_bias"].float()
    del state
    form_norm = .5 * (left.square().sum(1) * right.square().sum(1) +
                      (left * right).sum(1).square())
    ranking = (down.square().sum(0) * form_norm).argsort(descending=True)

    models = {
        "cp_r128": CPBilinear(1152, 128, 101),
        "cp_r512": CPBilinear(1152, 512, 102),
        "cp_r2048": CPBilinear(1152, 2048, 103),
    }
    for model in models.values():
        initialize_cp(model, left, right, down, ranking)
        model.bias.copy_(bias)

    basis512 = output_basis(left.cuda(), right.cuda(), down.cuda(), 1152, 512)
    for name, blocks, block_rank in (("ll1_r64_l8", 64, 8),
                                     ("ll1_r128_l4", 128, 4),
                                     ("ll1_r256_l8", 256, 8),
                                     ("ll1_r512_l4", 512, 4)):
        model = LL1Bilinear(1152, blocks, block_rank, 200 + blocks).cuda()
        initialize_ll1(model, left, right, down, basis512[:, :blocks])
        model.bias.copy_(bias.cuda())
        models[name] = model.cpu()

    inputs = np.load(DATA / "layer1_inputs_large.npz")
    products = np.load(DATA / "natural_large.npz")
    natural_x = {split: torch.from_numpy(inputs[f"{split}__input1"].astype(np.float32))
                 for split in SPLITS}
    natural_h = {split: torch.from_numpy(products[f"{split}__h1"].astype(np.float32))
                 for split in SPLITS}
    results = {"models": {}, "objective": "isotropic Gaussian bilinear tensor probes"}
    for index, (name, model) in enumerate(models.items()):
        before = tensor_probe_metrics(model, left, right, down, 9000 + index)
        losses, best_loss = train_tensor(model, left, right, down, 2000, 5000 + index)
        row = {"parameters": sum(p.numel() for p in model.parameters()),
               "before_tensor_probe": before,
               "after_tensor_probe": tensor_probe_metrics(
                   model, left, right, down, 10000 + index),
               "final_loss": float(np.mean(losses[-100:])),
               "best_validation_relative_mse": best_loss, "natural": {}}
        for split in SPLITS:
            row["natural"][split] = natural_metrics(
                model, natural_x[split], natural_h[split], down, bias)
        results["models"][name] = row
        save_model(model, name)
    results["elapsed_seconds"] = time.time() - started
    (ROOT / "weight_factorization_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote direct tensor decompositions in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
