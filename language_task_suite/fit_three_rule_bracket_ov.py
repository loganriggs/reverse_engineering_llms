"""Fit low-rank OV factors to preserve bracket-site teacher predictions."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_ov_canonical_truncation import (
    activation_grams, model_from_checkpoint, replace_ov)
from eval_three_rule_ov_canonical import teacher_kl
from train_sports_alias_bracket_caps import draw, evaluate, stack


def trainable_ov(model, rank):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for layer in model.layers:
        if not hasattr(layer, 'q1'):
            continue
        keep = torch.zeros(layer.n_head*layer.d_head,
                           device=layer.o.weight.device)
        for head in range(layer.n_head):
            keep[head*layer.d_head:head*layer.d_head+rank] = 1
        for parameter, mask in ((layer.o.weight, keep[None, :]),
                                (layer.v.weight, keep[:, None])):
            parameter.requires_grad_(True)
            parameter.register_hook(lambda grad, mask=mask: grad*mask)
            trainable.append(parameter)
    return trainable


@torch.no_grad()
def unused_channel_max(model, rank):
    largest = 0.
    for layer in model.layers:
        if not hasattr(layer, 'q1'):
            continue
        if rank == layer.d_head:
            continue
        for head in range(layer.n_head):
            sl = slice(head*layer.d_head+rank,
                       (head+1)*layer.d_head)
            largest = max(largest,
                          layer.o.weight[:, sl].abs().max().item(),
                          layer.v.weight[sl, :].abs().max().item())
    return largest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fit-worlds', type=int, default=80)
    p.add_argument('--test-worlds', type=int, default=40)
    p.add_argument('--rank', type=int, default=2)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--background-weight', type=float, default=0.,
                   help='Weight on KL at sampled non-rule positions')
    p.add_argument('--seed', type=int, default=55501)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    teacher, cfg = model_from_checkpoint(a.checkpoint, a.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(a.seed)
    fit, fit_attempts = draw(rng, a.fit_worlds, 'train', tokenizer,
                             cfg['n_ctx'], unique=True)
    test, test_attempts = draw(rng, a.test_worlds, 'heldout', tokenizer,
                               cfg['n_ctx'], unique=True)
    assert not {key for key, _ in fit} & {key for key, _ in test}
    fit, test = stack(fit), stack(test)
    grams, _ = activation_grams(teacher, fit, a.device)
    candidate = deepcopy(teacher)
    replace_ov(candidate, grams, a.rank, 'weight', .001)
    initial = evaluate(candidate, test, a.device)
    initial_kl = teacher_kl(teacher, candidate, test, a.device)
    x = torch.cat([arm['x'] for arm in fit])
    y = torch.cat([arm['y'] for arm in fit])
    sites = torch.cat([arm['site'] for arm in fit])
    trainable = trainable_ov(candidate, a.rank)
    optimizer = torch.optim.Adam(trainable, lr=a.lr)
    torch.manual_seed(a.seed)
    losses = []
    for step in range(a.steps):
        ix = torch.randint(len(x), (a.batch_size,))
        xb, yb, sb = (tensor[ix].to(a.device)
                      for tensor in (x, y, sites))
        positions = torch.arange(len(xb), device=a.device)
        with torch.no_grad():
            teacher_logits = teacher(xb).float()
        candidate_logits = candidate(xb).float()
        target = teacher_logits[positions, sb[:, 1]].softmax(-1)
        predicted = candidate_logits[positions, sb[:, 1]]
        bracket_kl = F.kl_div(predicted.log_softmax(-1), target,
                        reduction='batchmean')
        loss = bracket_kl
        if a.background_weight:
            bg = torch.arange(0, xb.shape[1], 32, device=a.device)
            valid = yb[:, bg] != -100
            for kind in range(3):
                valid &= bg[None, :] != sb[:, kind:kind+1]
            bg_target = teacher_logits[:, bg].softmax(-1)
            bg_predicted = candidate_logits[:, bg].log_softmax(-1)
            bg_kl = F.kl_div(bg_predicted, bg_target,
                             reduction='none').sum(-1)[valid].mean()
            loss = bracket_kl+a.background_weight*bg_kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 100 == 0:
            print('step', step+1, 'KL', round(sum(losses[-50:])/50, 4),
                  flush=True)
    unused_max = unused_channel_max(candidate, a.rank)
    assert unused_max == 0., f'Rank mask leaked: {unused_max}'
    report = {
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(a).items()},
        'fit_attempts': fit_attempts, 'test_attempts': test_attempts,
        'native': evaluate(teacher, test, a.device),
        'initial': initial, 'initial_teacher_kl': initial_kl,
        'final': evaluate(candidate, test, a.device),
        'final_teacher_kl': teacher_kl(teacher, candidate, test, a.device),
        'last50_fit_site_kl': sum(losses[-50:])/min(len(losses), 50),
        'unused_ov_channel_max_abs': unused_max,
        'scope': 'Only first rank OV coordinates per head in both attention '
                 'layers trained. All other parameters frozen. Distillation '
                 'loss uses teacher bracket-site KL across eight arms and '
                 'optionally sampled non-rule background KL. Heldout games '
                 'are evaluation only.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('saved', a.output)


if __name__ == '__main__':
    main()
