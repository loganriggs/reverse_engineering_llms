"""Rule-site OV distillation on separate tasks, evaluated on their composition."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_score_bracket_ov import teacher_kl
from eval_sports_ov_canonical_truncation import (
    activation_grams, model_from_checkpoint, replace_ov)
from fit_three_rule_bracket_ov import trainable_ov, unused_channel_max
from train_sports_score_bracket import draw, evaluate, stack
from train_sports_score_bracket_mixture import separate_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fit-plans-per-rule', type=int, default=80)
    p.add_argument('--test-worlds', type=int, default=28)
    p.add_argument('--carrier-words', type=int, default=100)
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--scope', choices=('both', 'bracket', 'score'),
                   default='both')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--background-weight', type=float, default=.1)
    p.add_argument('--seed', type=int, default=55901)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    teacher, cfg = model_from_checkpoint(a.checkpoint, a.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(a.seed)
    score_rows, score_games, _ = separate_data(
        rng, a.fit_plans_per_rule, 'score', tokenizer, cfg['n_ctx'],
        a.carrier_words)
    bracket_rows, bracket_games, _ = separate_data(
        rng, a.fit_plans_per_rule, 'bracket', tokenizer, cfg['n_ctx'],
        a.carrier_words)
    all_rows = score_rows+bracket_rows
    fit = ({key: torch.stack([row[key] for row in all_rows])
            for key in ('x', 'y', 'site')},)
    train = (all_rows if a.scope == 'both' else
             bracket_rows if a.scope == 'bracket' else score_rows)
    x = torch.stack([row['x'] for row in train])
    y = torch.stack([row['y'] for row in train])
    sites = torch.stack([row['site'] for row in train])
    records, attempts = draw(random.Random(a.seed), a.test_worlds,
                              'heldout', tokenizer, cfg['n_ctx'],
                              a.carrier_words, 'winning_score',
                              unique=True)
    test = stack(records)
    grams, _ = activation_grams(teacher, fit, a.device)
    student = deepcopy(teacher)
    replace_ov(student, grams, a.rank, 'weight', .001)
    initial = evaluate(student, test, a.device)
    initial_kl = teacher_kl(teacher, student, test, a.device)
    parameters = trainable_ov(student, a.rank)
    optimizer = torch.optim.Adam(parameters, lr=a.lr)
    torch.manual_seed(a.seed)
    losses = []
    for step in range(a.steps):
        ix = torch.randint(len(x), (a.batch_size,))
        xb, yb, sb = (tensor[ix].to(a.device)
                      for tensor in (x, y, sites))
        positions = torch.arange(len(xb), device=a.device)
        target_pos = sb.max(-1).values
        assert (target_pos >= 0).all()
        with torch.no_grad():
            teacher_logits = teacher(xb).float()
        student_logits = student(xb).float()
        target = teacher_logits[positions, target_pos].softmax(-1)
        predicted = student_logits[positions, target_pos].log_softmax(-1)
        site_kl = F.kl_div(predicted, target, reduction='batchmean')
        loss = site_kl
        if a.background_weight:
            bg = torch.arange(0, xb.shape[1], 32, device=a.device)
            valid = yb[:, bg] != -100
            valid &= bg[None, :] != target_pos[:, None]
            bg_target = teacher_logits[:, bg].softmax(-1)
            bg_predicted = student_logits[:, bg].log_softmax(-1)
            bg_kl = F.kl_div(bg_predicted, bg_target,
                             reduction='none').sum(-1)[valid].mean()
            loss = site_kl+a.background_weight*bg_kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 100 == 0:
            print('step', step+1, 'loss',
                  round(sum(losses[-50:])/50, 4), flush=True)
    unused = unused_channel_max(student, a.rank)
    assert unused == 0.
    report = {
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(a).items()},
        'fit_games': {'score': score_games, 'bracket': bracket_games},
        'fit_rows': len(train), 'test_attempts': attempts,
        'test_game_sha256': [hashlib.sha256(repr(key).encode()).hexdigest()
                              for key, _ in records],
        'native': evaluate(teacher, test, a.device),
        'initial': initial, 'initial_teacher_kl': initial_kl,
        'final': evaluate(student, test, a.device),
        'final_teacher_kl': teacher_kl(teacher, student, test, a.device),
        'last50_fit_loss': sum(losses[-50:])/min(len(losses), 50),
        'unused_ov_channel_max_abs': unused,
        'scope': 'Only low-rank OV factors trained. Site and optional '
                 'background KL use separate-rule reports, never joint '
                 'edited examples. Joint four-arm reports are test only.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('saved', a.output)


if __name__ == '__main__':
    main()
