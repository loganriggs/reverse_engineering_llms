"""Train a small bilinear LM on page-disjoint ToTTo table descriptions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model


def select(rows, limit, seed):
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(rows), generator=rng)[:limit].tolist()
    return [rows[i] for i in indices]


@torch.no_grad()
def evaluate(model, rows, device, batch_size=4):
    model.eval()
    sums = Counter()
    per_family = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        x = torch.stack([row['x'].long() for row in batch]).to(device)
        y = torch.stack([row['y'].long() for row in batch]).to(device)
        logits = model(x).float()
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                               ignore_index=-100, reduction='sum')
        sums['token_nll_sum'] += loss.item()
        sums['tokens'] += (y != -100).sum().item()
        predictions = logits.argmax(-1)
        for i, row in enumerate(batch):
            all_first = True
            for positions, family in zip(row['sites'], row['families']):
                p = torch.tensor(positions, dtype=torch.long,
                                 device=device)
                target = y[i, p]
                site_logits = logits[i, p]
                site_loss = F.cross_entropy(site_logits, target,
                                            reduction='sum')
                first = (predictions[i, p[0]] == target[0]).item()
                phrase = bool((predictions[i, p] == target).all().item())
                sums['site_token_nll_sum'] += site_loss.item()
                sums['site_tokens'] += len(positions)
                sums['sites'] += 1
                sums['first_exact'] += first
                sums['phrase_teacher_forced_exact'] += phrase
                all_first &= first
                family_sum = per_family.setdefault(family, Counter())
                family_sum['sites'] += 1
                family_sum['first_exact'] += first
                family_sum['phrase_exact'] += phrase
            sums['examples'] += 1
            sums['all_sites_first_exact'] += all_first
    if not sums['examples']:
        return {'examples': 0}
    return {
        'examples': sums['examples'], 'sites': sums['sites'],
        'token_nll': sums['token_nll_sum']/sums['tokens'],
        'site_token_nll': sums['site_token_nll_sum']/sums['site_tokens'],
        'first_bpe_top1': sums['first_exact']/sums['sites'],
        'phrase_teacher_forced_top1':
            sums['phrase_teacher_forced_exact']/sums['sites'],
        'all_sites_first_top1':
            sums['all_sites_first_exact']/sums['examples'],
        'top_families': {name: {'sites': counter['sites'],
                                'first_bpe_top1':
                                    counter['first_exact']/counter['sites'],
                                'phrase_teacher_forced_top1':
                                    counter['phrase_exact']/counter['sites']}
                         for name, counter in sorted(
                             per_family.items(),
                             key=lambda item: -item[1]['sites'])[:15]},
    }


def save_state(path, model, optimizer, rng, step, history, identity):
    state = {'identity': identity, 'step': step,
             'model': {k: v.detach().cpu()
                       for k, v in model.state_dict().items()},
             'optimizer': optimizer.state_dict(),
             'rng': rng.get_state(), 'history': history,
             'cuda_rng': torch.cuda.get_rng_state_all()
                         if next(model.parameters()).is_cuda else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=2)
    parser.add_argument('--eval-limit', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=300)
    parser.add_argument('--site-weight', type=float, default=10.)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--seed', type=int, default=74901)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    splits = cache['splits']
    train = splits['train']
    if not train or not splits['val'] or not splits['test']:
        raise ValueError('Missing page-disjoint training/evaluation split')
    eval_rows = {name: select(splits[name], args.eval_limit,
                              args.seed+1000+i)
                 for i, name in enumerate(('val','test','combo_val',
                                          'combo_test'))}
    torch.manual_seed(args.seed)
    model = _model(cache['vocab_size'], cache['n_ctx'], args.depth,
                   args.d_model, args.heads, args.device, 'rms')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed+1)
    identity = (str(args.cache.resolve()), args.depth, args.d_model,
                args.heads, args.batch_size, args.site_weight,
                args.lr, args.seed, cache['n_ctx'])
    if args.state.exists():
        state = torch.load(args.state, map_location='cpu', weights_only=True)
        if tuple(state['identity']) != identity or state['step'] > args.steps:
            raise ValueError('Resume identity mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['rng'])
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        start, history = state['step'], state['history']
        print('resumed', start, flush=True)
    else:
        start = 0
        history = {'0': {name: evaluate(model, rows, args.device,
                                        args.eval_batch_size)
                         for name, rows in eval_rows.items()}}
        save_state(args.state, model, optimizer, rng, 0, history, identity)
    report = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
              'parameters': model.n_params,
              'split_sizes': {name: len(rows) for name, rows in splits.items()},
              'holdout_pair': cache['holdout_pair'],
              'history': history,
              'scope': 'Full-token CE with extra loss at exact human-corrected '
                       'table-value mentions. Page-disjoint fit/test and '
                       'visible Title+Year whole-combination training holdout. '
                       'First-BPE and teacher-forced phrase prediction are '
                       'behavioral gates, not causal fidelity or circuit '
                       'recovery.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(start, args.steps):
        model.train()
        indices = torch.randint(len(train), (args.batch_size,),
                                generator=rng).tolist()
        batch = [train[i] for i in indices]
        x = torch.stack([row['x'].long() for row in batch]).to(args.device)
        y = torch.stack([row['y'].long() for row in batch]).to(args.device)
        logits = model(x).float()
        per = F.cross_entropy(logits.flatten(0,1), y.flatten(),
                              ignore_index=-100,
                              reduction='none').view_as(y)
        weights = (y != -100).float()
        for i, row in enumerate(batch):
            for positions in row['sites']:
                weights[i, list(positions)] = args.site_weight
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step+1}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        completed = step+1
        if completed % args.eval_every == 0 or completed == args.steps:
            history[str(completed)] = {
                'last_train_loss': float(loss.detach()),
                **{name: evaluate(model, rows, args.device,
                                   args.eval_batch_size)
                   for name, rows in eval_rows.items()}}
            report['history'] = history
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            save_state(args.state, model, optimizer, rng, completed,
                       history, identity)
            print(completed, 'val first',
                  round(history[str(completed)]['val']['first_bpe_top1'],3),
                  'combo val first',
                  round(history[str(completed)]['combo_val'].get(
                      'first_bpe_top1', 0),3), flush=True)


if __name__ == '__main__':
    main()
