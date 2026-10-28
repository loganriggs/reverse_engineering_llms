#!/usr/bin/env python3
"""Distill Bilin18's final MLP into a much narrower bilinear layer.

The student represents

    f(x) = bias + sum_j d_j (l_j^T x) (r_j^T x).

Rows of ``l`` and ``r`` are normalized in the forward pass.  This quotients out
the reciprocal scaling gauge of a rank-one bilinear factor; the remaining
factor magnitude lives in ``d``.  Swapping l/r and moving two signs between the
three factors leave the function unchanged, so evaluation compares functions,
not arbitrarily chosen factor coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_bilin18_final_mlp_compression import final_inputs, logits_from_products
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--ranking", type=Path, default=Path(
        "artifacts/bilin18_final_mlp_compression_replicate/results.json"))
    parser.add_argument("--widths", default="32,64,128,256")
    parser.add_argument("--train-start", type=int, default=5000)
    parser.add_argument("--train-rows", type=int, default=400)
    parser.add_argument("--validation-start", type=int, default=5500)
    parser.add_argument("--validation-rows", type=int, default=100)
    parser.add_argument("--eval-start", type=int, default=5800)
    parser.add_argument("--eval-rows", type=int, default=400)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--norm-loss-weight", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/bilin18_final_mlp_student"))
    return parser.parse_args()


class NarrowBilinear(nn.Module):
    def __init__(self, dimension: int, width: int):
        super().__init__()
        self.left = nn.Parameter(torch.empty(width, dimension))
        self.right = nn.Parameter(torch.empty(width, dimension))
        self.down = nn.Parameter(torch.empty(dimension, width))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def balanced_factors(self):
        left = F.normalize(self.left, dim=1)
        right = F.normalize(self.right, dim=1)
        return left, right, self.down

    def forward(self, x):
        left, right, down = self.balanced_factors()
        product = F.linear(x, left) * F.linear(x, right)
        return F.linear(product, down, self.bias)


@torch.inference_mode()
def extract_examples(model, corpus, start_row, rows, args):
    xs, ys = [], []
    block = model.blocks[-1]
    started = time.perf_counter()
    panel = corpus[start_row:start_row + rows]
    for start in range(0, len(panel), args.extract_batch_size):
        ids = torch.as_tensor(
            np.asarray(panel[start:start + args.extract_batch_size, :args.sequence_length], dtype=np.int64),
            device=args.device,
        )
        pre, products = final_inputs(model, ids)
        normalized = F.rms_norm(pre, (model.config.n_embd,))
        target = block.mlp.Down(products) + block.mlp.Down_bias
        # Teacher writes occasionally exceed float16's 65,504 maximum.  CPU
        # bfloat16 preserves their range while keeping the cached panel small.
        xs.append(normalized.flatten(0, 1).to(device="cpu", dtype=torch.bfloat16))
        ys.append(target.flatten(0, 1).to(device="cpu", dtype=torch.bfloat16))
        if (start // args.extract_batch_size + 1) % 50 == 0:
            print(f"extract {start_row}: {start + len(ids)}/{rows} rows", flush=True)
    return torch.cat(xs), torch.cat(ys), time.perf_counter() - started


def initialize_from_teacher(student, teacher_mlp, units, train_x, train_y, device):
    with torch.no_grad():
        left = teacher_mlp.Left.weight[units].float().to(device)
        right = teacher_mlp.Right.weight[units].float().to(device)
        down = teacher_mlp.Down.weight[:, units].float().to(device)
        left_norm = left.norm(dim=1).clamp_min(1e-8)
        right_norm = right.norm(dim=1).clamp_min(1e-8)
        student.left.copy_(left / left_norm[:, None])
        student.right.copy_(right / right_norm[:, None])
        student.down.copy_(down * (left_norm * right_norm)[None, :])
        # Fold the mean omitted teacher write into the student bias.  Estimate
        # on a bounded sample so initialization remains cheap.
        count = min(4096, len(train_x))
        sample_x = train_x[:count].to(device=device, dtype=torch.float32)
        sample_y = train_y[:count].to(device=device, dtype=torch.float32)
        student.bias.copy_((sample_y - student(sample_x) + student.bias).mean(0))


def agreement(pred, target):
    cosine = F.cosine_similarity(pred, target, dim=-1)
    target_norm = target.norm(dim=-1).clamp_min(1e-8)
    pred_norm = pred.norm(dim=-1)
    return {
        "mean_cosine": float(cosine.mean()),
        "median_cosine": float(cosine.median()),
        "cosine_q10": float(cosine.quantile(.1)),
        "mean_norm_ratio": float((pred_norm / target_norm).mean()),
        "relative_l2": float((pred - target).norm() / target.norm().clamp_min(1e-8)),
    }


@torch.inference_mode()
def validate_student(student, x, y, device, batch_size=1024):
    sums = {"count": 0, "cosine": 0., "relative_sq": 0., "target_sq": 0.,
            "norm_ratio": 0.}
    cosines = []
    for start in range(0, len(x), batch_size):
        bx = x[start:start + batch_size].to(device=device, dtype=torch.float32)
        by = y[start:start + batch_size].to(device=device, dtype=torch.float32)
        pred = student(bx)
        cos = F.cosine_similarity(pred, by, dim=-1)
        n = len(bx)
        sums["count"] += n
        sums["cosine"] += float(cos.sum())
        sums["relative_sq"] += float((pred - by).square().sum())
        sums["target_sq"] += float(by.square().sum())
        sums["norm_ratio"] += float((pred.norm(dim=-1) / by.norm(dim=-1).clamp_min(1e-8)).sum())
        cosines.append(cos.cpu())
    all_cos = torch.cat(cosines)
    return {
        "mean_cosine": sums["cosine"] / sums["count"],
        "median_cosine": float(all_cos.median()),
        "cosine_q10": float(all_cos.quantile(.1)),
        "mean_norm_ratio": sums["norm_ratio"] / sums["count"],
        "relative_l2": math.sqrt(sums["relative_sq"] / max(sums["target_sq"], 1e-12)),
    }


def train_one(width, model, ranking, train_x, train_y, val_x, val_y, args):
    torch.manual_seed(args.seed + width)
    student = NarrowBilinear(model.config.n_embd, width).to(args.device)
    units = torch.tensor(ranking[:width], device=args.device)
    initialize_from_teacher(student, model.blocks[-1].mlp, units, train_x, train_y, args.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 10 * width)
    started = time.perf_counter()
    curve = []
    for step in range(1, args.steps + 1):
        ids = torch.randint(len(train_x), (args.train_batch_size,), generator=generator)
        x = train_x[ids].to(device=args.device, dtype=torch.float32)
        target = train_y[ids].to(device=args.device, dtype=torch.float32)
        pred = student(x)
        cosine_loss = (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
        log_norm_loss = (
            torch.log(pred.norm(dim=-1).clamp_min(1e-6))
            - torch.log(target.norm(dim=-1).clamp_min(1e-6))
        ).square().mean()
        loss = cosine_loss + args.norm_loss_weight * log_norm_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            metrics = validate_student(student, val_x, val_y, args.device)
            point = {"step": step, "loss": float(loss.detach()),
                     "cosine_loss": float(cosine_loss.detach()),
                     "log_norm_loss": float(log_norm_loss.detach()), **metrics}
            curve.append(point)
            print(json.dumps({"width": width, **point}), flush=True)
    elapsed = time.perf_counter() - started
    validation = validate_student(student, val_x, val_y, args.device)
    return student.eval(), curve, validation, elapsed


@torch.inference_mode()
def end_to_end(model, students, corpus, args):
    totals = {
        width: {"n": 0, "nll": 0., "kl": 0., "same": 0, "cos": 0.,
                "relative_sq": 0., "target_sq": 0.}
        for width in students
    }
    baseline_nll = 0.
    baseline_n = 0
    panel = corpus[args.eval_start:args.eval_start + args.eval_rows]
    started = time.perf_counter()
    for start in range(0, len(panel), args.extract_batch_size):
        raw = np.asarray(panel[start:start + args.extract_batch_size, :args.sequence_length + 1],
                         dtype=np.int64)
        ids = torch.as_tensor(raw[:, :args.sequence_length], device=args.device)
        target_ids = torch.as_tensor(raw[:, args.sequence_length], device=args.device)
        pre, products = final_inputs(model, ids)
        pre = pre[:, -1:]
        products = products[:, -1:]
        base_logits, teacher_write, _ = logits_from_products(model, pre, products)
        base_logits = base_logits[:, 0]
        base_logp = F.log_softmax(base_logits, -1)
        base_p = base_logp.exp()
        baseline_nll += float(F.nll_loss(base_logp, target_ids, reduction="sum"))
        baseline_n += len(ids)
        normalized = F.rms_norm(pre, (model.config.n_embd,)).float()
        for width, student in students.items():
            write = student(normalized).to(pre.dtype)
            post = pre + write
            final = F.rms_norm(post, (model.config.n_embd,))
            raw_logits = model.lm_head(final).float()[:, 0]
            logits = 30 * torch.tanh(raw_logits / 30)
            logp = F.log_softmax(logits, -1)
            item = totals[width]
            item["n"] += len(ids)
            item["nll"] += float(F.nll_loss(logp, target_ids, reduction="sum"))
            item["kl"] += float((base_p * (base_logp - logp)).sum())
            item["same"] += int((base_logits.argmax(-1) == logits.argmax(-1)).sum())
            predicted = write.float()
            teacher = teacher_write.float()
            item["cos"] += float(F.cosine_similarity(predicted, teacher, dim=-1).sum())
            item["relative_sq"] += float((predicted - teacher).square().sum())
            item["target_sq"] += float(teacher.square().sum())
        if (start // args.extract_batch_size + 1) % 50 == 0:
            print(f"end-to-end {start + len(ids)}/{len(panel)} rows", flush=True)
    base_ce = baseline_nll / baseline_n
    results = {}
    for width, item in totals.items():
        ce = item["nll"] / item["n"]
        results[str(width)] = {
            "cross_entropy": ce,
            "cross_entropy_delta": ce - base_ce,
            "perplexity_ratio": math.exp(ce - base_ce),
            "mean_kl_from_full": item["kl"] / item["n"],
            "top1_agreement": item["same"] / item["n"],
            "write_mean_cosine": item["cos"] / item["n"],
            "write_relative_l2": math.sqrt(item["relative_sq"] / max(item["target_sq"], 1e-12)),
        }
    return {"baseline_cross_entropy": base_ce, "students": results,
            "seconds": time.perf_counter() - started}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    corpus = np.load(args.corpus, mmap_mode="r")
    ranking_data = json.loads(args.ranking.read_text())
    ranking = ranking_data["ranking_order"]
    widths = [int(value) for value in args.widths.split(",")]
    model = load_full_model(args.device)

    train_x, train_y, train_seconds = extract_examples(
        model, corpus, args.train_start, args.train_rows, args)
    val_x, val_y, validation_seconds = extract_examples(
        model, corpus, args.validation_start, args.validation_rows, args)
    print({"train_examples": len(train_x), "validation_examples": len(val_x),
           "extract_seconds": train_seconds + validation_seconds}, flush=True)

    students, training = {}, {}
    for width in widths:
        student, curve, validation, seconds = train_one(
            width, model, ranking, train_x, train_y, val_x, val_y, args)
        students[width] = student
        training[str(width)] = {"seconds": seconds, "curve": curve,
                                "validation": validation}
        torch.save({"state_dict": student.state_dict(), "width": width,
                    "dimension": model.config.n_embd},
                   args.output / f"student_k{width}.pt")

    e2e = end_to_end(model, students, corpus, args)
    original_parameters = sum(p.numel() for p in model.blocks[-1].mlp.parameters())
    result = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "layer": 17,
        "objective": "mean(1-cosine(student_write, teacher_write)) + norm_loss_weight * mean(log_norm_ratio^2)",
        "symmetry_handling": {
            "reciprocal_scaling": "left/right rows normalized; scale carried by down",
            "left_right_swap": "function-space objective is invariant",
            "sign": "function-space objective is invariant to paired factor sign moves",
        },
        "config": {**vars(args), "corpus": str(args.corpus), "ranking": str(args.ranking),
                   "output": str(args.output)},
        "original_mlp_parameters": original_parameters,
        "student_parameters": {
            str(width): sum(p.numel() for p in student.parameters())
            for width, student in students.items()
        },
        "timing": {"train_extract_seconds": train_seconds,
                   "validation_extract_seconds": validation_seconds},
        "training": training,
        "end_to_end": e2e,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(e2e, indent=2))
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
