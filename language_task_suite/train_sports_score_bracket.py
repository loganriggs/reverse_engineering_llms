"""Bilinear LM pilot for margin arithmetic and bracket closure in reports."""

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
from sports_score_bracket_dgp import realize, sample_plan
from train_sports_alias_bracket import _tv
from train_sports_report_multirule import DeepModel, PAD, SPECS


ARMS = (('base', 'round'), ('base', 'square'),
        ('increment', 'round'), ('increment', 'square'))
KINDS = ('bracket', 'numeric')
EDGES = {'bracket': ((0, 1), (2, 3)),
         'numeric': ((0, 2), (1, 3))}
OTHER = {'bracket': 'numeric', 'numeric': 'bracket'}


def aligned(doc, tokenizer, n_ctx):
    enc = tokenizer.encode(doc.text, add_special_tokens=False)
    positions, targets = [], []
    for site in doc.sites:
        found = [i for i, (start, end) in enumerate(enc.offsets)
                 if start <= site.start < end]
        if len(found) != 1 or found[0] < 1 or found[0] >= n_ctx or \
                enc.offsets[found[0]][1] != site.end:
            return None
        positions.append(found[0]-1)
        targets.append(enc.ids[found[0]])
    length = min(len(enc.ids)-1, n_ctx, max(positions)+41)
    x = torch.full((n_ctx,), PAD, dtype=torch.long)
    y = torch.full((n_ctx,), -100, dtype=torch.long)
    x[:length] = torch.tensor(enc.ids[:length])
    y[:length] = torch.tensor(enc.ids[1:length+1])
    return {'x': x, 'y': y,
            'site': torch.tensor(positions, dtype=torch.long),
            'target': torch.tensor(targets, dtype=torch.long)}


def quad(plan, tokenizer, n_ctx):
    rows = tuple(aligned(realize(plan, *arm), tokenizer, n_ctx)
                 for arm in ARMS)
    if any(row is None for row in rows):
        return None
    for k, name in enumerate(KINDS):
        for left, right in EDGES[name]:
            if rows[left]['target'][k] == rows[right]['target'][k]:
                return None
    assert rows[0]['target'][0] == rows[2]['target'][0]
    assert rows[1]['target'][0] == rows[3]['target'][0]
    assert rows[0]['target'][1] == rows[1]['target'][1]
    assert rows[2]['target'][1] == rows[3]['target'][1]
    return rows


def draw(rng, count, group, tokenizer, n_ctx, words=100,
         query_kind='margin',
         unique=False, excluded=()):
    out, seen, attempts = [], set(excluded), 0
    while len(out) < count:
        attempts += 1
        if attempts > count*1000:
            raise RuntimeError('Too few aligned score×bracket games')
        plan = sample_plan(rng, group, carrier_words=words,
                           query_kind=query_kind)
        key = plan.score.source_game_key
        if unique and key in seen:
            continue
        rows = quad(plan, tokenizer, n_ctx)
        if rows is None:
            continue
        out.append((key, rows))
        seen.add(key)
    return out, attempts


def stack(records):
    return tuple({key: torch.stack([rows[i][key] for _, rows in records])
                  for key in ('x', 'y', 'site', 'target')}
                 for i in range(4))


@torch.no_grad()
def evaluate(model, panel, device, batch_size=4):
    model.eval()
    n = len(panel[0]['x'])
    counts = {name: {'correct': 0, 'paired': 0, 'candidate': 0,
                     'direction': 0, 'nll': 0., 'own_tv': 0.,
                     'other_tv': 0.} for name in KINDS}
    token_loss = token_count = 0.
    for start in range(0, n, batch_size):
        rows = [{key: value[start:start+batch_size].to(device)
                 for key, value in arm.items()} for arm in panel]
        logits = [model(row['x']).float() for row in rows]
        ix = torch.arange(len(rows[0]['x']), device=device)
        z = [[out[ix, row['site'][:, k]] for k in range(2)]
             for out, row in zip(logits, rows)]
        correct = [[z[i][k].argmax(-1) == rows[i]['target'][:, k]
                    for k in range(2)] for i in range(4)]
        for i in range(4):
            for k, name in enumerate(KINDS):
                counts[name]['correct'] += correct[i][k].sum().item()
                counts[name]['nll'] += F.cross_entropy(
                    z[i][k], rows[i]['target'][:, k],
                    reduction='sum').item()
            token_loss += F.cross_entropy(
                logits[i].flatten(0, 1), rows[i]['y'].flatten(),
                ignore_index=-100, reduction='sum').item()
            token_count += (rows[i]['y'] != -100).sum().item()
        for k, name in enumerate(KINDS):
            for left, right in EDGES[name]:
                ta, tb = rows[left]['target'][:, k], \
                         rows[right]['target'][:, k]
                la = z[left][k][ix, ta]-z[left][k][ix, tb]
                lb = z[right][k][ix, ta]-z[right][k][ix, tb]
                c = counts[name]
                c['paired'] += (correct[left][k] & correct[right][k]).sum().item()
                c['candidate'] += ((la > 0) & (lb < 0)).sum().item()
                c['direction'] += (la > lb).sum().item()
                c['own_tv'] += _tv(z[left][k], z[right][k]).sum().item()
            for left, right in EDGES[OTHER[name]]:
                counts[name]['other_tv'] += _tv(
                    z[left][k], z[right][k]).sum().item()
    result = {'examples': n, 'token_nll': token_loss/token_count}
    for name in KINDS:
        c = counts[name]
        result[name] = {
            'top1': c['correct']/(4*n),
            'paired_top1': c['paired']/(2*n),
            'candidate_paired': c['candidate']/(2*n),
            'edit_direction': c['direction']/(2*n),
            'site_nll': c['nll']/(4*n),
            'own_edit_tv': c['own_tv']/(2*n),
            'other_rule_edit_tv': c['other_tv']/(2*n)}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--train-worlds', type=int, default=800)
    p.add_argument('--val-worlds', type=int, default=8)
    p.add_argument('--test-worlds', type=int, default=16)
    p.add_argument('--carrier-words', type=int, default=100)
    p.add_argument('--query-kind', choices=('margin', 'winning_score'),
                   default='margin')
    p.add_argument('--n-ctx', type=int, default=256)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--bracket-weight', type=float, default=100.)
    p.add_argument('--numeric-weight', type=float, default=500.)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--seed', type=int, default=55701)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    a = p.parse_args()
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(a.seed)
    train, train_attempts = draw(rng, a.train_worlds, 'train', tokenizer,
                                 a.n_ctx, a.carrier_words, a.query_kind)
    val, val_attempts = draw(rng, a.val_worlds, 'heldout', tokenizer,
                             a.n_ctx, a.carrier_words, a.query_kind,
                             unique=True)
    test, test_attempts = draw(rng, a.test_worlds, 'heldout', tokenizer,
                               a.n_ctx, a.carrier_words, a.query_kind,
                               unique=True,
                               excluded={key for key, _ in val})
    assert not {key for key, _ in train} & {key for key, _ in val+test}
    training, validation, testing = map(stack, (train, val, test))
    xtrain = torch.cat([arm['x'] for arm in training])
    ytrain = torch.cat([arm['y'] for arm in training])
    strain = torch.cat([arm['site'] for arm in training])
    torch.manual_seed(a.seed+a.depth)
    model = DeepModel(50257, a.d_model, a.heads,
                      SPECS[f'block{a.depth}'], a.n_ctx,
                      norm='rms', residual='lerp',
                      attention='bilinear').to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    starting = evaluate(model, validation, a.device)
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
        weights[ix, site[:, 0]] = a.bracket_weight
        weights[ix, site[:, 1]] = a.numeric_weight
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
        'data': {'train_plans': len(train),
                 'unique_train_games': len({key for key, _ in train}),
                 'val_games': len(val), 'test_games': len(test),
                 'train_attempts': train_attempts,
                 'val_attempts': val_attempts,
                 'test_attempts': test_attempts,
                 'test_game_sha256': [hashlib.sha256(repr(key).encode())
                                      .hexdigest() for key, _ in test]},
        'starting_validation': starting,
        'training': {'last50_loss': sum(losses[-50:])/min(len(losses), 50)},
        'validation': evaluate(model, validation, a.device),
        'test': evaluate(model, testing, a.device),
        'caveat': 'Natural carrier with inserted score and numeric-query sentences; '
                  'no full next-token oracle. All four combinations are '
                  'trained, so no whole-combination transfer claim.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    torch.save({'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                'config': report['config']}, a.checkpoint)
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
