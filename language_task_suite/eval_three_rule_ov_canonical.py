"""Compare weight-only and activation-aware OV gauges on three-rule reports."""

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
    activation_grams, max_probe_difference, model_from_checkpoint, replace_ov)
from train_sports_alias_bracket_caps import KINDS, draw, evaluate, stack


@torch.no_grad()
def teacher_kl(teacher, candidate, panel, device, batch_size=2):
    sums = {key: 0. for key in (*KINDS, 'background')}
    counts = {key: 0 for key in sums}
    for arm in panel:
        for start in range(0, len(arm['x']), batch_size):
            row = {k: v[start:start+batch_size].to(device)
                   for k, v in arm.items()}
            base = teacher(row['x']).float()
            test = candidate(row['x']).float()
            ix = torch.arange(len(row['x']), device=device)
            for kind, name in enumerate(KINDS):
                positions = row['site'][:, kind]
                p = F.softmax(base[ix, positions], -1)
                logq = F.log_softmax(test[ix, positions], -1)
                sums[name] += F.kl_div(logq, p,
                                       reduction='none').sum(-1).sum().item()
                counts[name] += len(row['x'])
            positions = torch.arange(0, row['x'].shape[1], 32,
                                     device=device)
            valid = row['y'][:, positions] != -100
            for kind in range(3):
                valid &= positions[None, :] != row['site'][:, kind:kind+1]
            p = F.softmax(base[:, positions], -1)
            logq = F.log_softmax(test[:, positions], -1)
            divergence = F.kl_div(logq, p, reduction='none').sum(-1)
            sums['background'] += divergence[valid].sum().item()
            counts['background'] += valid.sum().item()
    return {key: sums[key]/counts[key] for key in sums} | \
        {'background_positions': counts['background']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fit-worlds', type=int, default=80)
    p.add_argument('--test-worlds', type=int, default=40)
    p.add_argument('--ranks', default='4,8,16')
    p.add_argument('--ridge-fraction', type=float, default=.001)
    p.add_argument('--seed', type=int, default=55401)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    model, cfg = model_from_checkpoint(args.checkpoint, args.device)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    fit, fit_attempts = draw(rng, args.fit_worlds, 'train', tokenizer,
                             cfg['n_ctx'], unique=True)
    test, test_attempts = draw(rng, args.test_worlds, 'heldout', tokenizer,
                               cfg['n_ctx'], unique=True)
    assert not {key for key, _ in fit} & {key for key, _ in test}
    fit, test = stack(fit), stack(test)
    grams, counts = activation_grams(model, fit, args.device)
    result = {
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(args).items()},
        'fit_attempts': fit_attempts, 'test_attempts': test_attempts,
        'fit_activation_counts': counts,
        'native': evaluate(model, test, args.device),
        'rows': [],
        'scope': 'All OV products in both attention layers; QK, MLP, '
                 'embedding and unembedding remain unchanged.',
        'caveat': 'Only bracket closure is behaviorally competent. '
                  'Capitalization has a dominant-word shortcut and alias '
                  'is weak. The report carrier has no complete LM oracle.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method in ('weight', 'data'):
        for rank in (int(value) for value in args.ranks.split(',')):
            candidate = deepcopy(model)
            errors = replace_ov(candidate, grams, rank, method,
                                args.ridge_fraction)
            probe = max_probe_difference(model, candidate, test,
                                         args.device)
            if rank == 16 and probe > .001:
                raise AssertionError(f'Full-rank gauge changed logits: {probe}')
            score = evaluate(candidate, test, args.device)
            kl = teacher_kl(model, candidate, test, args.device)
            result['rows'].append({
                'method': method, 'rank_per_head': rank,
                'max_probe_logit_change': probe,
                'mean_weight_rel_error': sum(
                    item['weight_rel_error'] for item in errors)/len(errors),
                'mean_input_weighted_rel_error': sum(
                    item['input_weighted_rel_error'] for item in errors)/
                    len(errors),
                'heldout': score, 'teacher_kl': kl})
            print(method, rank, 'bracket',
                  round(score['bracket']['paired_top1'], 3),
                  'bracket KL', round(kl['bracket'], 4), flush=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('saved', args.output)


if __name__ == '__main__':
    main()
