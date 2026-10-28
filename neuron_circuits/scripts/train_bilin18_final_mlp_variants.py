#!/usr/bin/env python3
"""Compare alternative learned replacements for Bilin18's final MLP."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_bilin18_final_mlp_compression import final_inputs, logits_from_products
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from train_bilin18_final_mlp_student import NarrowBilinear, initialize_from_teacher


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
    parser.add_argument("--logit-batch-size", type=int, default=32)
    parser.add_argument("--state-steps", type=int, default=1500)
    parser.add_argument("--logit-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path(
        "artifacts/bilin18_final_mlp_variants"))
    parser.add_argument("--eval-rows", type=int, default=1200)
    parser.add_argument(
        "--specs",
        default=("random_write_k32,32,random,write;ranked_state_k32,32,ranked,state;"
                 "random_state_k32,32,random,state;ranked_kl_k32,32,ranked,kl;"
                 "ranked_hybrid_k32,32,ranked,hybrid;ranked_ce_k32,32,ranked,ce;"
                 "ranked_kl_k64,64,ranked,kl"),
        help="Semicolon-separated name,width,initialization,objective entries.",
    )
    return parser.parse_args()


@torch.inference_mode()
def extract(model, corpus, start_row, rows, args):
    xs, pres, writes, labels = [], [], [], []
    block = model.blocks[-1]
    panel = corpus[start_row:start_row + rows]
    for start in range(0, len(panel), args.extract_batch_size):
        raw = np.asarray(panel[start:start + args.extract_batch_size,
                               :args.sequence_length + 1], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :args.sequence_length], device=args.device)
        pre, products = final_inputs(model, ids)
        write = block.mlp.Down(products) + block.mlp.Down_bias
        xs.append(F.rms_norm(pre, (model.config.n_embd,)).flatten(0, 1).cpu().to(torch.bfloat16))
        pres.append(pre.flatten(0, 1).cpu().to(torch.bfloat16))
        writes.append(write.flatten(0, 1).cpu().to(torch.bfloat16))
        labels.append(torch.as_tensor(raw[:, 1:args.sequence_length + 1]).flatten())
    return torch.cat(xs), torch.cat(pres), torch.cat(writes), torch.cat(labels)


def initialize_random(student, train_x, train_y, device):
    with torch.no_grad():
        torch.nn.init.normal_(student.left)
        torch.nn.init.normal_(student.right)
        count = min(4096, len(train_y))
        y = train_y[:count].to(device=device, dtype=torch.float32)
        bias = y.mean(0)
        student.bias.copy_(bias)
        component_rms = (y - bias).square().mean().sqrt()
        torch.nn.init.normal_(student.down, std=float(component_rms / math.sqrt(student.down.shape[1])))


def student_logits(model, pre, write):
    final = F.rms_norm(pre + write, (model.config.n_embd,))
    raw = F.linear(final, model.lm_head.weight.float())
    return 30 * torch.tanh(raw / 30)


def objective_value(kind, model, student, x, pre, teacher_write, labels):
    predicted = student(x)
    if kind == "write":
        cosine = (1 - F.cosine_similarity(predicted, teacher_write, dim=-1)).mean()
        norm = (torch.log(predicted.norm(dim=-1).clamp_min(1e-6))
                - torch.log(teacher_write.norm(dim=-1).clamp_min(1e-6))).square().mean()
        return cosine + .25 * norm, {"write_cosine_loss": cosine, "norm_loss": norm}

    teacher_state = F.rms_norm(pre + teacher_write, (model.config.n_embd,))
    student_state = F.rms_norm(pre + predicted, (model.config.n_embd,))
    state_loss = (1 - F.cosine_similarity(student_state, teacher_state, dim=-1)).mean()
    if kind == "state":
        return state_loss, {"state_cosine_loss": state_loss}

    teacher_logits = 30 * torch.tanh(F.linear(teacher_state, model.lm_head.weight.float()) / 30)
    logits = 30 * torch.tanh(F.linear(student_state, model.lm_head.weight.float()) / 30)
    if kind == "kl":
        teacher_logp = F.log_softmax(teacher_logits, -1)
        kl = (teacher_logp.exp() * (teacher_logp - F.log_softmax(logits, -1))).sum(-1).mean()
        return kl + .1 * state_loss, {"kl": kl, "state_cosine_loss": state_loss}
    if kind == "ce":
        ce = F.cross_entropy(logits, labels)
        return ce, {"ce": ce}
    if kind == "hybrid":
        teacher_logp = F.log_softmax(teacher_logits, -1)
        kl = (teacher_logp.exp() * (teacher_logp - F.log_softmax(logits, -1))).sum(-1).mean()
        ce = F.cross_entropy(logits, labels)
        return kl + .1 * ce + .1 * state_loss, {"kl": kl, "ce": ce,
                                                "state_cosine_loss": state_loss}
    raise ValueError(kind)


def train_variant(name, width, init, kind, model, ranking, train, validation, args):
    train_x, train_pre, train_write, train_labels = train
    val_x, val_pre, val_write, val_labels = validation
    torch.manual_seed(args.seed + width + sum(map(ord, name)))
    student = NarrowBilinear(model.config.n_embd, width).to(args.device)
    if init == "ranked":
        units = torch.tensor(ranking[:width], device=args.device)
        initialize_from_teacher(student, model.blocks[-1].mlp, units,
                                train_x, train_write, args.device)
    elif init == "random":
        initialize_random(student, train_x, train_write, args.device)
    else:
        raise ValueError(init)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    logit_kind = kind in {"kl", "ce", "hybrid"}
    steps = args.logit_steps if logit_kind else args.state_steps
    batch_size = args.logit_batch_size if logit_kind else args.state_batch_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 100 * width + len(name))
    curve, started = [], time.perf_counter()
    for step in range(1, steps + 1):
        ids = torch.randint(len(train_x), (batch_size,), generator=generator)
        values = [item[ids].to(args.device) for item in train]
        x, pre, teacher_write = [value.float() for value in values[:3]]
        labels = values[3].long()
        loss, parts = objective_value(kind, model, student, x, pre, teacher_write, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            with torch.no_grad():
                count = min(512, len(val_x))
                vx = val_x[:count].to(args.device).float()
                vp = val_pre[:count].to(args.device).float()
                vw = val_write[:count].to(args.device).float()
                vl = val_labels[:count].to(args.device).long()
                vloss, vparts = objective_value(kind, model, student, vx, vp, vw, vl)
                write_cos = F.cosine_similarity(student(vx), vw, -1).mean()
                state_cos = F.cosine_similarity(
                    F.rms_norm(vp + student(vx), (model.config.n_embd,)),
                    F.rms_norm(vp + vw, (model.config.n_embd,)), -1).mean()
            point = {"step": step, "train_loss": float(loss.detach()),
                     "validation_loss": float(vloss), "validation_write_cosine": float(write_cos),
                     "validation_state_cosine": float(state_cos),
                     **{f"train_{key}": float(value.detach()) for key, value in parts.items()},
                     **{f"validation_{key}": float(value) for key, value in vparts.items()}}
            curve.append(point)
            print(json.dumps({"variant": name, **point}), flush=True)
    return student.eval(), curve, time.perf_counter() - started


@torch.inference_mode()
def evaluate_corpus(model, students, corpus, rows, args):
    totals = {name: {"n": 0, "nll": 0., "kl": 0., "same": 0,
                     "write_cos": 0., "state_cos": 0.} for name in students}
    baseline_nll, baseline_n = 0., 0
    for start in range(0, rows, args.extract_batch_size):
        raw = np.asarray(corpus[start:start + args.extract_batch_size,
                                :args.sequence_length + 1], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :args.sequence_length], device=args.device)
        labels = torch.as_tensor(raw[:, args.sequence_length], device=args.device)
        pre, products = final_inputs(model, ids)
        pre = pre[:, -1:].float()
        products = products[:, -1:]
        base_logits, teacher_write, _ = logits_from_products(model, pre.to(products.dtype), products)
        base_logits = base_logits[:, 0]
        teacher_write = teacher_write.float()
        base_logp = F.log_softmax(base_logits, -1)
        base_p = base_logp.exp()
        baseline_nll += float(F.nll_loss(base_logp, labels, reduction="sum"))
        baseline_n += len(ids)
        x = F.rms_norm(pre, (model.config.n_embd,))
        teacher_state = F.rms_norm(pre + teacher_write, (model.config.n_embd,))
        for name, student in students.items():
            write = student(x)
            logits = student_logits(model, pre, write)[:, 0]
            logp = F.log_softmax(logits, -1)
            item = totals[name]
            item["n"] += len(ids)
            item["nll"] += float(F.nll_loss(logp, labels, reduction="sum"))
            item["kl"] += float((base_p * (base_logp - logp)).sum())
            item["same"] += int((base_logits.argmax(-1) == logits.argmax(-1)).sum())
            item["write_cos"] += float(F.cosine_similarity(write, teacher_write, -1).sum())
            item["state_cos"] += float(F.cosine_similarity(
                F.rms_norm(pre + write, (model.config.n_embd,)), teacher_state, -1).sum())
    base_ce = baseline_nll / baseline_n
    output = {"baseline_cross_entropy": base_ce, "students": {}}
    for name, item in totals.items():
        ce = item["nll"] / item["n"]
        output["students"][name] = {
            "cross_entropy": ce, "cross_entropy_delta": ce - base_ce,
            "perplexity_ratio": math.exp(ce - base_ce),
            "kl_from_full": item["kl"] / item["n"],
            "top1_agreement": item["same"] / item["n"],
            "write_cosine": item["write_cos"] / item["n"],
            "final_state_cosine": item["state_cos"] / item["n"],
        }
    return output


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = load_full_model(args.device)
    for parameter in list(model.embedding.parameters()) + list(model.blocks.parameters()) + list(model.lm_head.parameters()):
        parameter.requires_grad_(False)
    corpus = np.load(args.corpus, mmap_mode="r")
    ranking = json.loads(args.ranking.read_text())["ranking_order"]
    train = extract(model, corpus, args.train_start, args.train_rows, args)
    validation = extract(model, corpus, args.validation_start, args.validation_rows, args)

    specs = []
    for entry in args.specs.split(";"):
        name, width, init, kind = entry.split(",")
        specs.append((name, int(width), init, kind))
    students, training = {}, {}
    for name, width, init, kind in specs:
        student, curve, seconds = train_variant(
            name, width, init, kind, model, ranking, train, validation, args)
        students[name] = student
        training[name] = {"width": width, "initialization": init, "objective": kind,
                          "seconds": seconds, "curve": curve}
        torch.save({"state_dict": student.state_dict(), "width": width,
                    "dimension": model.config.n_embd, "name": name},
                   args.output / f"{name}.pt")

    panels = {
        "fineweb": (corpus[11000:11000 + args.eval_rows], args.eval_rows),
    }
    for name in ("wikitext103", "tinystories"):
        path = Path("artifacts/bilin18_ood_corpora") / f"{name}_1200x129.npy"
        array = np.load(path, mmap_mode="r")
        panels[name] = (array, min(args.eval_rows, len(array)))
    evaluations = {}
    for name, (panel, rows) in panels.items():
        evaluations[name] = evaluate_corpus(model, students, panel, rows, args)
        print(name, json.dumps(evaluations[name], indent=2), flush=True)
    result = {
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "config": {**vars(args), "corpus": str(args.corpus), "ranking": str(args.ranking),
                   "output": str(args.output)},
        "training": training, "evaluations": evaluations,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
