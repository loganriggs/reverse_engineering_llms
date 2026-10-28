#!/usr/bin/env python3
"""Distill Bilin18 layer-1 CP students with downstream causal objectives."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_bilin18_mlp_discriminators import SNAPSHOT
from refine_bilin18_tensor_decomposition import load_model
from run_bilin18_qkov_causal_edges import DEFAULT_CORPUS, load_full_model
from run_bilin18_tensor_decomposition import ROOT, teacher_bilinear


CONFIG = Path("configs/bilin18_causal_distillation.json")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variants", nargs="*", default=None)
    return parser.parse_args()


@torch.no_grad()
def prefix_to_layer1(model, ids):
    dimension = model.config.n_embd
    x0 = F.rms_norm(model.embedding(ids), (dimension,))
    x, value_bus = x0, None
    block0 = model.blocks[0]
    mixed = block0.lambdas[0] * x + block0.lambdas[1] * x0
    attention, value_bus = block0.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    pre = mixed + attention
    normalized = F.rms_norm(pre, (dimension,))
    product = block0.mlp.Left(normalized) * block0.mlp.Right(normalized)
    x = pre + block0.mlp.Down(product) + block0.mlp.Down_bias

    block1 = model.blocks[1]
    mixed = block1.lambdas[0] * x + block1.lambdas[1] * x0
    attention, value_bus = block1.attn(F.rms_norm(mixed, (dimension,)), value_bus)
    pre = mixed + attention
    normalized = F.rms_norm(pre, (dimension,))
    return x0, value_bus, pre, normalized


def suffix_from_layer1(model, x0, value_bus, layer1_output):
    dimension = model.config.n_embd
    x = layer1_output
    for block in model.blocks[2:]:
        mixed = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, value_bus = block.attn(F.rms_norm(mixed, (dimension,)), value_bus)
        pre = mixed + attention
        normalized = F.rms_norm(pre, (dimension,))
        product = block.mlp.Left(normalized) * block.mlp.Right(normalized)
        x = pre + block.mlp.Down(product) + block.mlp.Down_bias
    logits = model.lm_head(F.rms_norm(x, (dimension,))).float()
    return 30 * torch.tanh(logits / 30)


def teacher_layer1_write(model, mlp_input):
    block = model.blocks[1]
    product = block.mlp.Left(mlp_input) * block.mlp.Right(mlp_input)
    return block.mlp.Down(product) + block.mlp.Down_bias


def student_layer1_write(student, mlp_input):
    flat = mlp_input.float().flatten(0, 1)
    return student(flat).view_as(mlp_input).to(mlp_input.dtype)


def apply_intervention(normalized, config, cpu_generator, gpu_generator):
    gaussian = bool(torch.randint(2, (1,), generator=cpu_generator))
    if gaussian:
        choices = config["interventions"]["gaussian_scales"]
        scale = choices[int(torch.randint(len(choices), (1,), generator=cpu_generator))]
        direction = torch.randn(
            normalized.shape, generator=gpu_generator, device=normalized.device,
            dtype=torch.float32)
        direction = direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
        return normalized + scale * direction.to(normalized.dtype), "gaussian", scale
    choices = config["interventions"]["natural_mix_scales"]
    scale = choices[int(torch.randint(len(choices), (1,), generator=cpu_generator))]
    donor = normalized.roll(shifts=1, dims=1)
    return normalized + scale * (donor - normalized), "natural_mix", scale


def distribution_kl(teacher_logits, student_logits):
    teacher_logp = F.log_softmax(teacher_logits.float(), -1)
    teacher_p = teacher_logp.exp()
    student_logp = F.log_softmax(student_logits.float(), -1)
    return (teacher_p * (teacher_logp - student_logp)).sum(-1).mean()


def fisher_effect_loss(teacher_base, teacher_changed, student_base, student_changed):
    weights = F.softmax(teacher_base.float(), -1)
    teacher = teacher_changed.float() - teacher_base.float()
    student = student_changed.float() - student_base.float()
    teacher = teacher - (weights * teacher).sum(-1, keepdim=True)
    student = student - (weights * student).sum(-1, keepdim=True)
    return (weights * (student - teacher).square()).sum(-1).mean()


@torch.no_grad()
def teacher_outputs(model, x0, value_bus, pre, base_input, changed_input=None):
    base_write = teacher_layer1_write(model, base_input)
    base = suffix_from_layer1(model, x0, value_bus, pre + base_write)
    changed = None
    if changed_input is not None:
        changed_write = teacher_layer1_write(model, changed_input)
        changed = suffix_from_layer1(model, x0, value_bus, pre + changed_write)
    return base, changed


def student_outputs(model, student, x0, value_bus, pre, base_input, changed_input=None):
    base_write = student_layer1_write(student, base_input)
    base = suffix_from_layer1(model, x0, value_bus, pre + base_write)
    changed = None
    if changed_input is not None:
        changed_write = student_layer1_write(student, changed_input)
        changed = suffix_from_layer1(model, x0, value_bus, pre + changed_write)
    return base, changed


def tensor_loss(student, teacher_weights, gpu_generator):
    left, right, down = teacher_weights
    x = torch.randn(16, left.shape[1], generator=gpu_generator, device=left.device)
    y = torch.randn(16, left.shape[1], generator=gpu_generator, device=left.device)
    with torch.no_grad():
        target = teacher_bilinear(x, y, left, right, down)
    prediction = student.bilinear(x, y)
    return (prediction - target).square().sum() / target.square().sum().clamp_min(1e-8)


@torch.no_grad()
def validation_score(model, rows, student, variant, config, teacher_weights, device):
    spec = config["validation_rows"]
    gpu_generator = torch.Generator(device=device).manual_seed(config["seed"] + 7000)
    values = []
    for offset, index in enumerate(range(spec["start"], spec["stop"])):
        raw = np.asarray(rows[index:index + 1, :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :config["sequence_length"]], device=device)
        x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
        changed_input = None
        if variant["intervention_kl_weight"]:
            if offset % 2 == 0:
                direction = torch.randn(
                    normalized.shape, generator=gpu_generator, device=device,
                    dtype=torch.float32)
                direction = direction / direction.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
                changed_input = normalized + 0.1 * direction.to(normalized.dtype)
            else:
                changed_input = normalized + 0.25 * (normalized.roll(1, 1) - normalized)
        teacher_base, teacher_changed = teacher_outputs(
            model, x0, value_bus, pre, normalized, changed_input)
        student_base, student_changed = student_outputs(
            model, student, x0, value_bus, pre, normalized, changed_input)
        score = distribution_kl(teacher_base, student_base)
        if changed_input is not None:
            score = (score + variant["intervention_kl_weight"] *
                     distribution_kl(teacher_changed, student_changed)
                     + variant["effect_weight"] * fisher_effect_loss(
                         teacher_base, teacher_changed, student_base, student_changed))
        values.append(float(score))
    score = float(np.mean(values))
    if variant["tensor_weight"]:
        score += variant["tensor_weight"] * float(
            tensor_loss(student, teacher_weights, gpu_generator))
    return score


def save_student(student, name, config, training):
    torch.save({
        "state_dict": student.cpu().state_dict(),
        "configuration": {"type": "CPBilinear", "rank": student.rank},
        "distillation_config": config,
        "training": training,
    }, ROOT / f"{name}.pt")


def train_variant(model, rows, student, name, variant, config, teacher_weights, device, smoke=False):
    student.to(device).train()
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer_name = config.get("optimizer", "adamw")
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters, lr=config["learning_rate"], betas=(0.9, 0.95),
            weight_decay=config["weight_decay"])
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            parameters, lr=config["learning_rate"], momentum=config.get("momentum", 0.9),
            weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"unknown optimizer {optimizer_name}")
    cpu_generator = torch.Generator().manual_seed(config["seed"])
    gpu_generator = torch.Generator(device=device).manual_seed(config["seed"] + 1)
    steps = 3 if smoke else config["steps"]
    history = []
    student.eval()
    initial_validation = validation_score(
        model, rows, student, variant, config, teacher_weights, device)
    student.train()
    validations = [{"step": 0, "score": initial_validation}]
    best_validation = initial_validation
    best_state = {key: value.detach().cpu().clone()
                  for key, value in student.state_dict().items()}
    started = time.time()
    train_spec = config["train_rows"]
    for step in range(1, steps + 1):
        index = int(torch.randint(
            train_spec["start"], train_spec["stop"], (1,), generator=cpu_generator))
        raw = np.asarray(rows[index:index + config["batch_size"], :129], dtype=np.int64)
        ids = torch.as_tensor(raw[:, :config["sequence_length"]], device=device)
        x0, value_bus, pre, normalized = prefix_to_layer1(model, ids)
        use_causal = variant["intervention_kl_weight"] > 0
        changed_input, kind, scale = (apply_intervention(
            normalized, config, cpu_generator, gpu_generator) if use_causal
            else (None, "none", 0.0))
        teacher_base, teacher_changed = teacher_outputs(
            model, x0, value_bus, pre, normalized, changed_input)
        student_base, student_changed = student_outputs(
            model, student, x0, value_bus, pre, normalized, changed_input)
        base_kl = distribution_kl(teacher_base, student_base)
        changed_kl = torch.zeros((), device=device)
        effect = torch.zeros((), device=device)
        if use_causal:
            changed_kl = distribution_kl(teacher_changed, student_changed)
            effect = fisher_effect_loss(
                teacher_base, teacher_changed, student_base, student_changed)
        isotropic = torch.zeros((), device=device)
        if variant["tensor_weight"]:
            isotropic = tensor_loss(student, teacher_weights, gpu_generator)
        loss = (base_kl + variant["intervention_kl_weight"] * changed_kl
                + variant["effect_weight"] * effect
                + variant["tensor_weight"] * isotropic)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        row = {"step": step, "loss": float(loss.detach()), "base_kl": float(base_kl.detach()),
               "changed_kl": float(changed_kl.detach()), "effect_loss": float(effect.detach()),
               "tensor_loss": float(isotropic.detach()), "intervention": kind, "scale": scale}
        history.append(row)
        if step == 1 or step % 25 == 0:
            elapsed = time.time() - started
            print(name, step, f"loss={row['loss']:.5f}", f"base={row['base_kl']:.5f}",
                  f"changed={row['changed_kl']:.5f}", f"effect={row['effect_loss']:.5f}",
                  f"tensor={row['tensor_loss']:.5f}", f"{elapsed:.1f}s", flush=True)
        if not smoke and step % config["validation_every"] == 0:
            student.eval()
            score = validation_score(
                model, rows, student, variant, config, teacher_weights, device)
            validations.append({"step": step, "score": score})
            if score < best_validation:
                best_validation = score
                best_state = {key: value.detach().cpu().clone()
                              for key, value in student.state_dict().items()}
            student.train()
            print(name, step, f"validation={score:.5f}",
                  f"best={best_validation:.5f}", flush=True)
    if best_state is not None:
        student.load_state_dict(best_state)
    training = {"history": history, "elapsed_seconds": time.time() - started,
                "validations": validations, "best_validation": best_validation,
                "initial_validation": initial_validation,
                "steps": steps, "final_25": {key: float(np.mean([row[key] for row in history[-25:]]))
                                               for key in ("loss", "base_kl", "changed_kl",
                                                           "effect_loss", "tensor_loss")}}
    if not smoke and config.get("save_checkpoint", True):
        save_student(student, name, config, training)
    return training


def main():
    args = parse_args()
    config = json.loads(args.config.read_text())
    device = args.device
    model = load_full_model(device)
    for module in (model.embedding, model.blocks, model.lm_head):
        module.requires_grad_(False)
    rows = np.load(DEFAULT_CORPUS, mmap_mode="r")
    state = torch.load(SNAPSHOT / "pytorch_model.bin", map_location="cpu",
                       weights_only=True, mmap=True)
    teacher_weights = tuple(state[key].float().to(device) for key in (
        "transformer.h.1.mlp.Left.weight", "transformer.h.1.mlp.Right.weight",
        "transformer.h.1.mlp.Down.weight"))
    del state
    selected = args.variants or list(config["variants"])
    results = {"config": config, "variants": {}}
    for name in selected:
        if name not in config["variants"]:
            raise ValueError(f"unknown variant {name}")
        student = load_model("cp_r2048")
        payload = torch.load(config["base_checkpoint"], map_location="cpu", weights_only=True)
        student.load_state_dict(payload["state_dict"])
        results["variants"][name] = train_variant(
            model, rows, student, name, config["variants"][name], config,
            teacher_weights, device, args.smoke)
        del student
        torch.cuda.empty_cache()
        if args.smoke:
            break
    if not args.smoke:
        training_log = Path(config.get(
            "training_log", ROOT / "causal_distillation_training.json"))
        training_log.write_text(json.dumps(results, indent=2))
        print("wrote causal distillation checkpoints and training log", flush=True)


if __name__ == "__main__":
    main()
