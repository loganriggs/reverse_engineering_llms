"""Diagnostic: can the same small bilinear LM learn nonce copying alone?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model
from train_corpus_nonce_copy_mixture import score
from train_corpus_list_joint_supervised import sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial-checkpoint', type=Path)
    p.add_argument('--nonce-cache', type=Path, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--joint-test-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--norm', choices=('none', 'rms'), default='none')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--site-weight', type=float, default=1000.)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--seed', type=int, default=72103)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    nonce = torch.load(a.nonce_cache, map_location='cpu', weights_only=True)
    lists = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(a.bracket_cache, map_location='cpu', weights_only=True)
    joint = torch.load(a.joint_test_cache, map_location='cpu', weights_only=True)
    torch.manual_seed(a.seed)
    model = _model(50257, 256, a.depth, a.d_model, a.heads,
                   a.device, a.norm)
    if a.initial_checkpoint:
        saved = torch.load(a.initial_checkpoint, map_location='cpu',
                           weights_only=True)
        assert (saved['depth'], saved['d_model'], saved['heads'],
                saved['norm']) == (a.depth, a.d_model, a.heads, a.norm)
        model.load_state_dict(saved['state_dict'])
    model.eval()
    initial = score(model, nonce, lists, brackets, joint, a.device)
    train = {key: torch.cat([nonce['paired']['train'][arm][key]
                             for arm in ('left', 'right')])
             for key in ('x', 'y', 'site')}
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    torch.manual_seed(a.seed)
    losses = []
    for step in range(a.steps):
        row = sample(train, a.batch_size, a.device)
        out = model(row['x']).float()
        per = F.cross_entropy(out.flatten(0, 1), row['y'].flatten(),
                              ignore_index=-100, reduction='none').view_as(row['y'])
        weights = (row['y'] != -100).float()
        ix = torch.arange(len(row['x']), device=a.device)
        weights[ix[:, None], row['site']] = a.site_weight
        loss = (per * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 1000 == 0:
            print('step', step + 1, 'loss',
                  round(sum(losses[-50:]) / 50, 3), flush=True)
    model.eval()
    final = score(model, nonce, lists, brackets, joint, a.device)
    report = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'initial': initial, 'final': final,
              'loss_last50': sum(losses[-50:]) / 50,
              'scope': 'Focused diagnostic on nonce-only paired training '
                       'windows with all-token CE and strong marked-site '
                       'weight. This is not a realistic mixture; list and '
                       'bracket retention are measured for interference.'}
    a.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                'depth': a.depth, 'd_model': a.d_model, 'heads': a.heads,
                'n_ctx': 256, 'vocab_size': 50257, 'norm': a.norm}, a.checkpoint)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
