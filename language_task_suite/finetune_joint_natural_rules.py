"""Continue a sports bilinear LM on three controlled natural-text rule sites.

Bridge experiment only: empirical prose has no full next-token oracle. The
three scored targets are alias reuse, bracket closure, and spelling style.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from train_sports_alias_bracket import draw, evaluate, stack
from train_sports_report_multirule import DeepModel, PAD, SPECS
from train_corpus_spelling_style_bilinear import _pair_metrics


def pad_style(row, n_ctx):
    out = dict(row)
    width = row['x'].shape[1]
    if width > n_ctx:
        raise ValueError('Style context longer than sports model context')
    out['x'] = F.pad(row['x'], (0, n_ctx-width), value=PAD)
    out['y'] = F.pad(row['y'], (0, n_ctx-width), value=-100)
    assert (out['site_position'] < n_ctx).all()
    return out


def weighted_loss(model, rows, site_weights):
    logits = model(rows['x']).float()
    y = rows['y']
    per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                          ignore_index=-100, reduction='none').view_as(y)
    weights = (y != -100).float()
    ix = torch.arange(len(y), device=y.device)
    if len(site_weights) == 1:
        weights[ix, rows['site_position']] = site_weights[0]
    else:
        for kind, weight in enumerate(site_weights):
            weights[ix, rows['site'][:, kind]] = weight
    return ((per*weights).sum(1)/weights.sum(1)).mean()


def indexed(rows, indices, device):
    return {key: value[indices].to(device) for key, value in rows.items()}


def evaluate_all(model, sports, style, device):
    return {'sports': evaluate(model, sports, device, batch_size=4),
            'style': _pair_metrics(model, style, device, batch_size=2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-checkpoint', type=Path, required=True)
    p.add_argument('--style-cache', type=Path, default=Path(
        '/home/loganriggs/Coding/neuron_circuits/artifacts/'
        'corpus_spelling_style_pilot/dataset.pt'))
    p.add_argument('--train-worlds', type=int, default=1000)
    p.add_argument('--test-worlds', type=int, default=40)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--style-weight', type=float, default=100.)
    p.add_argument('--seed', type=int, default=41501)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.batch_size % 2:
        raise ValueError('Even batch size required for 50/50 mixture')
    torch.set_num_threads(4)
    saved = torch.load(args.source_checkpoint, map_location='cpu',
                       weights_only=True)
    cfg = saved['config']
    if cfg['depth'] != 2 or cfg['n_ctx'] != 320:
        raise ValueError('Expected saved depth-2, context-320 sports model')
    style_cache = torch.load(args.style_cache, map_location='cpu',
                             weights_only=True)
    if style_cache['vocab_size'] != 50257:
        raise ValueError('Vocabulary mismatch')
    style_train = pad_style(style_cache['train'], cfg['n_ctx'])
    style_test = {arm: pad_style(style_cache['test'][arm], cfg['n_ctx'])
                  for arm in ('us', 'uk')}
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    rng = random.Random(args.seed)
    sports_train, train_attempts = draw(
        rng, args.train_worlds, 'train', tokenizer, cfg['n_ctx'])
    sports_test, test_attempts = draw(
        rng, args.test_worlds, 'heldout', tokenizer, cfg['n_ctx'],
        unique=True)
    assert not {key for key, _ in sports_train} & \
        {key for key, _ in sports_test}
    sports_train = stack(sports_train)
    sports_test = stack(sports_test)
    flat_sports = {key: torch.cat([arm[key] for arm in sports_train])
                   for key in ('x', 'y', 'site', 'target')}
    initial = DeepModel(50257, cfg['d_model'], cfg['heads'],
                        SPECS['block2'], cfg['n_ctx'], norm='rms',
                        residual='lerp', attention='bilinear').to(args.device)
    initial.load_state_dict(saved['state_dict'])
    starting = evaluate_all(initial, sports_test, style_test, args.device)
    result = {
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        'data': {'sports_train_worlds': args.train_worlds,
                 'sports_test_worlds': args.test_worlds,
                 'sports_train_attempts': train_attempts,
                 'sports_test_attempts': test_attempts,
                 'style_train_examples': len(style_train['x']),
                 'style_test_pairs': len(style_test['us']['x']),
                 'sports_game_overlap': 0},
        'objective': 'Each source example gets a mean weighted token CE. '
                     'Sports sites: alias 500, bracket 100. Style site: '
                     f'{args.style_weight:g}. Mixed arm takes half its '
                     'minibatch from each source; control is sports only. '
                     'Both arms have equal updates and tokens may differ.',
        'limitation': 'Natural text carriers; only edited sites have known '
                      'rules. Source domains differ. Style test is a '
                      'separate document split, sports test a disjoint-game '
                      'split. This does not establish a fully sampled '
                      'multitask next-token distribution.',
        'starting': starting, 'arms': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for arm in ('mixed', 'sports_only'):
        torch.manual_seed(args.seed)
        model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                          SPECS['block2'], cfg['n_ctx'], norm='rms',
                          residual='lerp', attention='bilinear').to(args.device)
        model.load_state_dict(saved['state_dict'])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        sampler = random.Random(args.seed+100)
        start = time.time()
        losses = []
        for step in range(args.steps):
            model.train()
            half = args.batch_size//2
            n_sports = half if arm == 'mixed' else args.batch_size
            sport_ix = torch.tensor([sampler.randrange(len(flat_sports['x']))
                                     for _ in range(n_sports)])
            sport_rows = indexed(flat_sports, sport_ix, args.device)
            opt.zero_grad(set_to_none=True)
            sport_loss = weighted_loss(model, sport_rows, (500., 100.))
            if arm == 'mixed':
                style_ix = torch.tensor([
                    sampler.randrange(len(style_train['x']))
                    for _ in range(half)])
                style_rows = indexed(style_train, style_ix, args.device)
                loss = (sport_loss+weighted_loss(model, style_rows,
                                                 (args.style_weight,)))/2
            else:
                loss = sport_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite {arm} step {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(loss.item())
            if (step+1) % 250 == 0:
                print(arm, step+1, round(loss.item(), 4), flush=True)
        result['arms'][arm] = {
            'last50_weighted_loss': sum(losses[-50:])/len(losses[-50:]),
            'seconds': round(time.time()-start, 1),
            'test': evaluate_all(model, sports_test, style_test,
                                 args.device)}
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        torch.save({'state_dict': {key: value.detach().cpu()
                                   for key, value in model.state_dict().items()},
                    'config': cfg, 'arm': arm},
                   args.output.with_name(args.output.stem+f'_{arm}.pt'))
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
