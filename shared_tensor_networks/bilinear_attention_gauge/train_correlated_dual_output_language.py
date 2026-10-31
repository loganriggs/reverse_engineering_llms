"""Train the necessary-two-head retrieval toy on a correlated key-bit law."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_dual_output_language import DualOutputAttention, evaluate, exhaustive


HERE = Path(__file__).parent
OUTPUT = HERE / "correlated_dual_models"


def batch(n, generator, same_probability=.9, swap_probability=.2,
          holdout=True):
    a = torch.randint(0, 2, (n,), generator=generator)
    same = torch.rand(n, generator=generator) < same_probability
    b = torch.where(same, a, 1 - a)
    filler = torch.randint(0, 2, (n,), generator=generator)
    swap = torch.rand(n, generator=generator) < swap_probability
    if holdout:
        # Withhold one entire rare (A,B,query) combination, keeping all
        # individual tokens and the other unequal-bit combinations in train.
        excluded = (a == 0) & (b == 1) & swap
        swap[excluded] = False
    keys = torch.stack((a, 2 + b, 4 + filler), dim=1)
    perm = torch.rand(n, 3, generator=generator).argsort(dim=1)
    seq = torch.cat((keys.gather(1, perm), (6 + swap.long())[:, None]), dim=1)
    target = torch.stack((torch.where(swap, b, a),
                          torch.where(swap, a, b)), dim=1).float()
    return seq, target


def head_ablation(model):
    seq, target, attrs = exhaustive()
    with torch.no_grad():
        _, heads, _, _ = model(seq, return_heads=True)
        results = []
        for removed in range(2):
            altered = heads.clone()
            altered[:, removed] = 0
            predicted = model.read(altered) > 0
            joint = (predicted == target).all(dim=1)
            mask = torch.tensor([a["heldout"] for a in attrs])
            results.append(dict(removed=removed,
                full_joint_accuracy=float(joint.float().mean()),
                rare_heldout_joint_accuracy=float(joint[mask].float().mean())))
    return results


def train(seed, steps=2500, batch_size=512, lr=.003):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(1000 + seed)
    model = DualOutputAttention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for step in range(1, steps + 1):
        model.train()
        seq, target = batch(batch_size, generator)
        loss = F.binary_cross_entropy_with_logits(model(seq), target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 1 or step % 250 == 0 or step == steps:
            row = dict(step=step, train_bce=float(loss.detach()),
                       exhaustive=evaluate(model))
            history.append(row)
            print(seed, step, row["train_bce"],
                  row["exhaustive"]["all"]["joint_accuracy"],
                  row["exhaustive"]["rare_heldout"]["joint_accuracy"],
                  flush=True)
    return model, dict(seed=seed, steps=steps, batch_size=batch_size,
        same_probability=.9, swap_probability=.2,
        holdout="A0/B1 with swap query",
        task="two retrieved key bits, optionally swap output roles",
        history=history, head_ablations=head_ablation(model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    for seed in range(args.seeds):
        model, result = train(seed, args.steps, args.batch_size)
        torch.save(model.state_dict(), OUTPUT / f"seed{seed}.pt")
        (OUTPUT / f"seed{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
