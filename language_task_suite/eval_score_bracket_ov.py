"""OV reductions fit on separate-rule reports, evaluated on joint reports."""

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
from eval_sports_ov_canonical_truncation import (
    activation_grams, max_probe_difference, model_from_checkpoint, replace_ov)
from train_sports_score_bracket import draw, evaluate, stack
from train_sports_score_bracket_mixture import separate_data


@torch.no_grad()
def teacher_kl(teacher, student, panel, device):
    sums = {key: 0. for key in ('bracket', 'score', 'background')}
    counts = {key: 0 for key in sums}
    for arm in panel:
        for start in range(0, len(arm['x']), 2):
            row = {key: value[start:start+2].to(device)
                   for key, value in arm.items()}
            base = teacher(row['x']).float()
            test = student(row['x']).float()
            ix = torch.arange(len(row['x']), device=device)
            for k, name in enumerate(('bracket', 'score')):
                pos = row['site'][:, k]
                p = base[ix, pos].softmax(-1)
                logq = test[ix, pos].log_softmax(-1)
                sums[name] += F.kl_div(logq, p,
                                       reduction='none').sum(-1).sum().item()
                counts[name] += len(row['x'])
            pos = torch.arange(0, row['x'].shape[1], 32,
                               device=device)
            valid = row['y'][:, pos] != -100
            for k in range(2):
                valid &= pos[None, :] != row['site'][:, k:k+1]
            p = base[:, pos].softmax(-1)
            logq = test[:, pos].log_softmax(-1)
            divergence = F.kl_div(logq, p, reduction='none').sum(-1)
            sums['background'] += divergence[valid].sum().item()
            counts['background'] += valid.sum().item()
    return {name: sums[name]/counts[name] for name in sums}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fit-plans-per-rule', type=int, default=40)
    p.add_argument('--test-worlds', type=int, default=28)
    p.add_argument('--carrier-words', type=int, default=100)
    p.add_argument('--ranks', default='2,4,8,16')
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
    rows = score_rows+bracket_rows
    fit = ({key: torch.stack([row[key] for row in rows])
            for key in ('x', 'y', 'site')},)
    records, attempts = draw(random.Random(a.seed), a.test_worlds,
                              'heldout', tokenizer, cfg['n_ctx'],
                              a.carrier_words, 'winning_score',
                              unique=True)
    test = stack(records)
    grams, counts = activation_grams(teacher, fit, a.device)
    report = {
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(a).items()},
        'fit_games': {'score': score_games, 'bracket': bracket_games},
        'fit_activation_counts': counts,
        'test_attempts': attempts,
        'test_game_sha256': [hashlib.sha256(repr(key).encode()).hexdigest()
                              for key, _ in records],
        'native': evaluate(teacher, test, a.device),
        'rows': [],
        'scope': 'OV products only. Activation covariance fitted on '
                 'separate score-only and bracket-only training reports; '
                 'joint four-arm reports used only for evaluation.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for method in ('weight', 'data'):
        for rank in (int(v) for v in a.ranks.split(',')):
            student = deepcopy(teacher)
            errors = replace_ov(student, grams, rank, method, .001)
            probe = max_probe_difference(teacher, student, test, a.device)
            if rank == 16 and probe > .001:
                raise AssertionError(f'Full-rank gauge changed logits: {probe}')
            score = evaluate(student, test, a.device)
            report['rows'].append({
                'method': method, 'rank_per_head': rank,
                'max_probe_logit_change': probe,
                'mean_weight_rel_error': sum(
                    row['weight_rel_error'] for row in errors)/len(errors),
                'mean_input_weighted_rel_error': sum(
                    row['input_weighted_rel_error'] for row in errors)/
                    len(errors),
                'heldout': score,
                'teacher_kl': teacher_kl(teacher, student, test, a.device)})
            print(method, rank, 'bracket',
                  round(score['bracket']['paired_top1'], 3),
                  'score', round(score['numeric']['paired_top1'], 3),
                  flush=True)
            a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('saved', a.output)


if __name__ == '__main__':
    main()
