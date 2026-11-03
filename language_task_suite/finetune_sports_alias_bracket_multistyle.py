"""Controlled style-mix continuation from a competent RotoWire checkpoint.

Both arms start from the same frozen model. The control trains on RotoWire;
the mixed arm uses half RotoWire and half SBNation examples per update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_sports_alias_bracket_position_control import groups
from eval_sports_alias_bracket_style_transfer import draw
from train_sports_alias_bracket import evaluate, stack
from train_sports_report_multirule import DeepModel, SPECS


SOURCE_DIR = Path('/home/loganriggs/Coding/neuron_circuits/'
                  'artifacts/rotowire')
DEFAULT_CACHE = Path('/home/loganriggs/Coding/neuron_circuits/artifacts/'
                     'multistyle_alias_bracket_panel_98371_98351_320.pt')


def flat(records):
    data = stack(records)
    return {name: torch.cat([arm[name] for arm in data])
            for name in ('x', 'y', 'site')}


def step(model, optimizer, batches, arm, batch_size, alias_weight,
         bracket_weight, device):
    chosen = (('rotowire', batch_size) if arm == 'rotowire' else
              ('rotowire', batch_size//2))
    sources = [chosen]
    if arm == 'mixed':
        sources.append(('sbnation', batch_size//2))
    pieces = []
    for source, count in sources:
        data = batches[source]
        indices = torch.randint(len(data['x']), (count,))
        pieces.append({key: value[indices] for key, value in data.items()})
    x = torch.cat([row['x'] for row in pieces]).to(device)
    y = torch.cat([row['y'] for row in pieces]).to(device)
    site = torch.cat([row['site'] for row in pieces]).to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x).float()
    per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                          ignore_index=-100, reduction='none').view_as(y)
    weights = (y != -100).float()
    ix = torch.arange(len(x), device=device)
    weights[ix, site[:, 0]] = alias_weight
    weights[ix, site[:, 1]] = bracket_weight
    loss = (per*weights).sum()/weights.sum()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return float(loss.detach())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-output', type=Path, required=True)
    p.add_argument('--arm', choices=('rotowire', 'mixed'), required=True)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=98401)
    p.add_argument('--train-seed', type=int, default=98371)
    p.add_argument('--eval-seed', type=int, default=98351)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu')
    p.add_argument('--data-cache', type=Path, default=DEFAULT_CACHE)
    a = p.parse_args()
    if a.batch_size % 2:
        raise ValueError('Batch size must be even')
    torch.set_num_threads(4)
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cfg = state['config']
    if cfg['alias_prior'] != 'roster' or cfg['alias_exponent'] != 0:
        raise ValueError('Only uniform roster alias prior is supported')
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    cache_config = (a.train_seed, a.eval_seed, cfg['n_ctx'])
    if a.data_cache.exists():
        cached = torch.load(a.data_cache, map_location='cpu',
                            weights_only=True)
        if tuple(cached['config']) != cache_config:
            raise ValueError('Data cache configuration mismatch')
        train_records = cached['train_records']
        heldout_records = cached['heldout_records']
        counts = cached['counts']
    else:
        train_records, heldout_records, counts = {}, {}, {}
        for source in ('rotowire', 'sbnation'):
            archive = SOURCE_DIR/f'{source}.tar.bz2'
            train_records[source], train_count = draw(
                None, a.train_seed, tokenizer, cfg['n_ctx'], archive,
                group='train')
            heldout_records[source], heldout_count = draw(
                None, a.eval_seed, tokenizer, cfg['n_ctx'], archive,
                group='heldout')
            counts[source] = {'train': train_count, 'heldout': heldout_count}
        a.data_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'config': cache_config,
                    'train_records': train_records,
                    'heldout_records': heldout_records,
                    'counts': counts}, a.data_cache)
    for source in ('rotowire', 'sbnation'):
        assert {key for key, _ in train_records[source]}.isdisjoint(
            {key for key, _ in heldout_records[source]})
    assert {key for rows in train_records.values() for key, _ in rows}.isdisjoint(
        {key for rows in heldout_records.values() for key, _ in rows})
    batches = {source: flat(rows) for source, rows in train_records.items()}
    divided = {source: groups(rows) for source, rows in
               heldout_records.items()}
    matching = {i: min(len(divided[source][i]) for source in divided)
                for i in range(5)}
    matched = {source: [row for i in range(5)
                        for row in divided[source][i][:matching[i]]]
               for source in divided}
    eval_batches = {source: stack(rows) for source, rows in matched.items()}
    torch.manual_seed(a.seed)
    model = DeepModel(50257, cfg['d_model'], cfg['heads'],
                      SPECS[f"block{cfg['depth']}"], cfg['n_ctx'],
                      norm='rms', residual='lerp',
                      attention='bilinear').to(a.device)
    model.load_state_dict(state['state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    trajectory = []

    def measure(completed, last_loss):
        result = {source: evaluate(model, batch, a.device)
                  for source, batch in eval_batches.items()}
        trajectory.append({'step': completed, 'last_loss': last_loss,
                           'matched': result})
        print('step', completed, 'loss',
              round(last_loss, 4) if last_loss is not None else None,
              'bracket', {source: round(value['bracket']['paired_top1'], 3)
                          for source, value in result.items()}, flush=True)

    measure(0, None)
    loss = 0.
    for i in range(1, a.steps+1):
        loss = step(model, optimizer, batches, a.arm, a.batch_size,
                    500., 100., a.device)
        if i % 200 == 0 or i == a.steps:
            measure(i, loss)
    a.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                'config': cfg, 'fine_tune': {'arm': a.arm, 'steps': a.steps,
                                            'lr': a.lr, 'seed': a.seed}},
               a.model_output)
    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'input_checkpoint_sha256': hashlib.sha256(
                  a.checkpoint.read_bytes()).hexdigest(),
              'training_counts': counts,
              'matched_per_position_bin': matching,
              'heldout_game_hashes': {source: [hashlib.sha256(
                  repr(key).encode()).hexdigest() for key, _ in rows]
                  for source, rows in matched.items()},
              'trajectory': trajectory,
              'caveat': ('Fine-tuning reinitializes AdamW but shares the '
                         'same starting model and update count across arms. '
                         'Only one alias pair and bracket span per unique '
                         'training game are cached, with repetitions via '
                         'minibatches. Match by coarse target position, '
                         'not lexical content or exact position. Human '
                         'prose lacks a full oracle.')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Saved', a.output, flush=True)


if __name__ == '__main__':
    main()
