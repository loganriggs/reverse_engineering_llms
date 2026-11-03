"""Train separate score-copy and bracket reports; test their joint composition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from sports_alias_bracket_dgp import (
    realize as realize_bracket, sample_plan as sample_bracket)
from sports_report_carrier_dgp import (
    increment_winning_score, realize as realize_score,
    sample_plan as sample_score)
from train_sports_report_multirule import DeepModel, PAD, SPECS
from train_sports_score_bracket import draw as draw_joint, evaluate, stack


def one_site(text, start, end, kind, tokenizer, n_ctx):
    enc = tokenizer.encode(text, add_special_tokens=False)
    found = [i for i, (a, b) in enumerate(enc.offsets) if a <= start < b]
    if len(found) != 1 or found[0] < 1 or found[0] >= n_ctx or \
            enc.offsets[found[0]][1] != end:
        return None
    pos = found[0]-1
    length = min(len(enc.ids)-1, n_ctx, pos+41)
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:length] = torch.tensor(enc.ids[:length])
    y[:length] = torch.tensor(enc.ids[1:length+1])
    site = torch.tensor((-1, -1), dtype=torch.long)
    site[0 if kind == 'bracket' else 1] = pos
    return {'x': x, 'y': y, 'site': site}


def separate_data(rng, count, kind, tokenizer, n_ctx, words):
    rows, games, attempts = [], set(), 0
    while len(rows) < 2*count:
        attempts += 1
        if attempts > count*500:
            raise RuntimeError('Too few aligned single-rule reports')
        if kind == 'score':
            plan = sample_score(rng, source='rotowire', group='train',
                                carrier_words=words,
                                query_kind='winning_score')
            docs = (realize_score(plan),
                    realize_score(increment_winning_score(plan)))
            pair = [one_site(doc.text, doc.site.start, doc.site.end,
                             'score', tokenizer, n_ctx) for doc in docs]
            key = plan.source_game_key
        else:
            plan = sample_bracket(rng, group='train')
            docs = (realize_bracket(plan, 'a', 'round'),
                    realize_bracket(plan, 'a', 'square'))
            pair = [one_site(doc.text, doc.sites[1].start,
                             doc.sites[1].end, 'bracket', tokenizer,
                             n_ctx) for doc in docs]
            key = plan.alias.game_key
        if any(row is None for row in pair):
            continue
        rows.extend(pair)
        games.add(key)
    return rows, len(games), attempts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--single-plans', type=int, default=1500)
    p.add_argument('--joint-plans', type=int, default=0,
                   help='Zero gives a whole-combination training holdout')
    p.add_argument('--test-worlds', type=int, default=20)
    p.add_argument('--carrier-words', type=int, default=100)
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--bracket-weight', type=float, default=100.)
    p.add_argument('--score-weight', type=float, default=500.)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=55801)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(a.seed)
    score_rows, score_games, score_attempts = separate_data(
        rng, a.single_plans, 'score', tokenizer, a.n_ctx,
        a.carrier_words)
    bracket_rows, bracket_games, bracket_attempts = separate_data(
        rng, a.single_plans, 'bracket', tokenizer, a.n_ctx,
        a.carrier_words)
    joint_rows = []
    if a.joint_plans:
        joint, _ = draw_joint(rng, a.joint_plans, 'train', tokenizer,
                              a.n_ctx, a.carrier_words, 'winning_score')
        joint_rows = [row for _, quad in joint for row in quad]
    test, test_attempts = draw_joint(rng, a.test_worlds, 'heldout',
                                      tokenizer, a.n_ctx, a.carrier_words,
                                      'winning_score', unique=True)
    testing = stack(test)
    train = score_rows+bracket_rows+joint_rows
    xtrain = torch.stack([row['x'] for row in train])
    ytrain = torch.stack([row['y'] for row in train])
    strain = torch.stack([row['site'] for row in train])
    torch.manual_seed(a.seed+a.depth)
    model = DeepModel(50257, a.d_model, a.heads,
                      SPECS[f'block{a.depth}'], a.n_ctx,
                      norm='rms', residual='lerp',
                      attention='bilinear').to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    starting = evaluate(model, testing, a.device)
    losses = []
    for step in range(a.steps):
        model.train()
        indices = torch.randint(len(xtrain), (a.batch_size,))
        x, y = xtrain[indices].to(a.device), ytrain[indices].to(a.device)
        site = strain[indices].to(a.device)
        logits = model(x).float()
        per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                              ignore_index=-100, reduction='none').view_as(y)
        weights = (y != -100).float()
        ix = torch.arange(len(x), device=a.device)
        for k, weight in ((0, a.bracket_weight), (1, a.score_weight)):
            valid = site[:, k] >= 0
            weights[ix[valid], site[valid, k]] = weight
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 500 == 0:
            print('step', step+1, 'loss', round(losses[-1], 3), flush=True)
    report = {
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in vars(a).items()},
        'data': {'score_plans': a.single_plans,
                 'score_unique_games': score_games,
                 'score_attempts': score_attempts,
                 'bracket_plans': a.single_plans,
                 'bracket_unique_games': bracket_games,
                 'bracket_attempts': bracket_attempts,
                 'joint_plans': a.joint_plans,
                 'train_rows': len(train),
                 'test_games': len(test), 'test_attempts': test_attempts,
                 'test_game_sha256': [hashlib.sha256(repr(key).encode())
                                      .hexdigest() for key, _ in test]},
        'starting_test': starting,
        'training': {'last50_loss': sum(losses[-50:])/min(len(losses), 50)},
        'test': evaluate(model, testing, a.device),
        'caveat': 'Score and bracket are trained in separate reports when '
                  'joint_plans=0. Paired heldout report format combines '
                  'them, so failures may reflect form transfer as well as '
                  'composition. No full natural-prose next-token oracle.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    torch.save({'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                'config': report['config']}, a.checkpoint)
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
