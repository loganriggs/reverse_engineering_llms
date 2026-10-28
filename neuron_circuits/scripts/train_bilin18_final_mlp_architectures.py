#!/usr/bin/env python3
"""Test linear and symmetric-square alternatives to the final bilinear MLP."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from train_bilin18_final_mlp_variants import extract, evaluate_corpus


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ranking", type=Path, default=Path(
        "artifacts/bilin18_final_mlp_compression_replicate/results.json"))
    parser.add_argument("--train-start", type=int, default=9000)
    parser.add_argument("--train-rows", type=int, default=600)
    parser.add_argument("--validation-start", type=int, default=9700)
    parser.add_argument("--validation-rows", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--state-batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/bilin18_final_mlp_architectures"))
    return parser.parse_args()


class LowRankLinear(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.up = nn.Linear(dimension, rank, bias=False)
        self.down = nn.Linear(rank, dimension)

    def forward(self, x):
        return self.down(self.up(x))


class AffineLinear(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)

    def forward(self, x):
        return self.linear(x)


class NarrowSquare(nn.Module):
    def __init__(self, dimension, width):
        super().__init__()
        self.reader = nn.Parameter(torch.empty(width, dimension))
        self.down = nn.Parameter(torch.empty(dimension, width))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, x):
        reader = F.normalize(self.reader, dim=1)
        return F.linear(F.linear(x, reader).square(), self.down, self.bias)


@torch.no_grad()
def set_mean_bias(module, train_x, train_y, device):
    count = min(4096, len(train_x))
    x = train_x[:count].to(device).float()
    y = train_y[:count].to(device).float()
    if isinstance(module, NarrowSquare):
        module.bias.copy_((y - module(x) + module.bias).mean(0))
    else:
        module.down.bias.copy_((y - module(x) + module.down.bias).mean(0))


@torch.no_grad()
def init_square(module, teacher, units, train_x, train_y, device):
    left = teacher.Left.weight[units].float().to(device)
    right = teacher.Right.weight[units].float().to(device)
    down = teacher.Down.weight[:, units].float().to(device)
    plus, minus = left + right, left - right
    plus_norm = plus.norm(dim=1).clamp_min(1e-8)
    minus_norm = minus.norm(dim=1).clamp_min(1e-8)
    module.reader[:len(units)].copy_(plus / plus_norm[:, None])
    module.reader[len(units):].copy_(minus / minus_norm[:, None])
    module.down[:, :len(units)].copy_(down * plus_norm.square()[None, :] / 4)
    module.down[:, len(units):].copy_(-down * minus_norm.square()[None, :] / 4)
    set_mean_bias(module, train_x, train_y, device)


def train(name, module, train_data, validation, model, args):
    tx, tp, tw, _ = train_data
    vx, vp, vw, _ = validation
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + sum(map(ord, name)))
    curve, started = [], time.perf_counter()
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(tx), (args.state_batch_size,), generator=generator)
        x, pre, teacher = [value[ids].to(args.device).float() for value in (tx, tp, tw)]
        predicted = module(x)
        teacher_state = F.rms_norm(pre + teacher, (model.config.n_embd,))
        student_state = F.rms_norm(pre + predicted, (model.config.n_embd,))
        loss = (1 - F.cosine_similarity(student_state, teacher_state, -1)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            with torch.no_grad():
                count = min(1024, len(vx))
                x = vx[:count].to(args.device).float()
                pre = vp[:count].to(args.device).float()
                teacher = vw[:count].to(args.device).float()
                predicted = module(x)
                state_cos = F.cosine_similarity(
                    F.rms_norm(pre + predicted, (model.config.n_embd,)),
                    F.rms_norm(pre + teacher, (model.config.n_embd,)), -1).mean()
                write_cos = F.cosine_similarity(predicted, teacher, -1).mean()
            point = {"step": step, "training_loss": float(loss.detach()),
                     "validation_state_cosine": float(state_cos),
                     "validation_write_cosine": float(write_cos)}
            curve.append(point)
            print(name, point, flush=True)
    return curve, time.perf_counter() - started


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = np.load(args.corpus, mmap_mode="r")
    model = load_full_model(args.device)
    for parameter in list(model.embedding.parameters()) + list(model.blocks.parameters()) + list(model.lm_head.parameters()):
        parameter.requires_grad_(False)
    train_data = extract(model, corpus, args.train_start, args.train_rows, args)
    validation = extract(model, corpus, args.validation_start, args.validation_rows, args)
    ranking = json.loads(args.ranking.read_text())["ranking_order"]
    dimension = model.config.n_embd
    torch.manual_seed(args.seed)
    models = {
        "lowrank_linear_k32": LowRankLinear(dimension, 32).to(args.device),
        "lowrank_linear_k64": LowRankLinear(dimension, 64).to(args.device),
        "affine_linear": AffineLinear(dimension).to(args.device),
        "square_k64": NarrowSquare(dimension, 64).to(args.device),
        "square_k128": NarrowSquare(dimension, 128).to(args.device),
    }
    tx, _, tw, _ = train_data
    for name in ("lowrank_linear_k32", "lowrank_linear_k64"):
        module = models[name]
        nn.init.normal_(module.up.weight, std=.02)
        nn.init.zeros_(module.down.weight)
        nn.init.zeros_(module.down.bias)
        set_mean_bias(module, tx, tw, args.device)
    nn.init.zeros_(models["affine_linear"].linear.weight)
    models["affine_linear"].linear.bias.data.copy_(tw[:4096].float().mean(0).to(args.device))
    init_square(models["square_k64"], model.blocks[-1].mlp,
                torch.tensor(ranking[:32], device=args.device), tx, tw, args.device)
    init_square(models["square_k128"], model.blocks[-1].mlp,
                torch.tensor(ranking[:64], device=args.device), tx, tw, args.device)

    training = {}
    for name, module in models.items():
        curve, seconds = train(name, module, train_data, validation, model, args)
        training[name] = {"parameters": sum(p.numel() for p in module.parameters()),
                          "seconds": seconds, "curve": curve}
        torch.save({"state_dict": module.state_dict(), "architecture": type(module).__name__,
                    "name": name}, args.output / f"{name}.pt")
    panels = {"fineweb": corpus[11000:11000 + args.eval_rows]}
    for name in ("wikitext103", "tinystories"):
        panels[name] = np.load(Path("artifacts/bilin18_ood_corpora") /
                               f"{name}_1200x129.npy", mmap_mode="r")
    evaluations = {name: evaluate_corpus(model, models, panel, args.eval_rows, args)
                   for name, panel in panels.items()}
    result = {"training": training, "evaluations": evaluations}
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(evaluations, indent=2))


if __name__ == "__main__":
    main()
