"""Add nonce copying to a list+bracket bilinear LM, with behavioral gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model, evaluate as eval_bracket
from train_corpus_list_delimiter_mixture import eval_joint, eval_list
from train_corpus_list_joint_supervised import list_rows, sample
from eval_corpus_nonce_copy import evaluate as eval_nonce


def score(model, nonce, lists, brackets, joint, device):
    train_subset = {arm: {key: value[:min(256, len(value))]
                          for key, value in nonce['paired']['train'][arm].items()}
                    for arm in ('left', 'right')}
    return {
        'nonce_train_first256': eval_nonce(model, train_subset, device),
        'nonce_val': eval_nonce(model, nonce['paired']['val'], device),
        'nonce_test_seen_codes': eval_nonce(model, nonce['paired']['test'], device),
        'nonce_test_unseen_codes': eval_nonce(model, nonce['test_unseen_codes'], device),
        'list_test': eval_list(model, lists['list']['test'], device),
        'bracket_test': eval_bracket(model, brackets['test'], device, batch_size=4),
        'joint_test': eval_joint(model, joint, device),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--nonce-cache', type=Path, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--joint-test-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--nonce-site-weight', type=float, default=100.)
    p.add_argument('--list-site-weight', type=float, default=100.)
    p.add_argument('--bracket-site-weight', type=float, default=100.)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--seed', type=int, default=72003)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    assert a.batch_size % 4 == 0
    torch.set_num_threads(4)
    nonce = torch.load(a.nonce_cache, map_location='cpu', weights_only=True)
    lists = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(a.bracket_cache, map_location='cpu', weights_only=True)
    joint = torch.load(a.joint_test_cache, map_location='cpu', weights_only=True)
    assert nonce['n_ctx'] == lists['n_ctx'] == brackets['n_ctx'] == 256
    saved = torch.load(a.initial_checkpoint, map_location='cpu',
                       weights_only=True)
    model = _model(50257, 256, 2, 64, 4, a.device, 'none')
    model.load_state_dict(saved['state_dict'])
    model.eval()
    initial = score(model, nonce, lists, brackets, joint, a.device)
    nonce_train = {key: torch.cat([nonce['paired']['train'][arm][key]
                                    for arm in ('left', 'right')])
                   for key in ('x', 'y', 'site')}
    list_train = list_rows(lists)
    bracket_train = brackets['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    torch.manual_seed(a.seed)
    losses = []
    for step in range(a.steps):
        n_nonce = a.batch_size // 2
        n_list = a.batch_size // 4
        n_bracket = a.batch_size - n_nonce - n_list
        chunks = [sample(nonce_train, n_nonce, a.device),
                  sample(list_train, n_list, a.device),
                  sample(bracket_train, n_bracket, a.device)]
        x = torch.cat([row['x'] for row in chunks])
        y = torch.cat([row['y'] for row in chunks])
        out = model(x).float()
        per = F.cross_entropy(out.flatten(0, 1), y.flatten(),
                              ignore_index=-100, reduction='none').view_as(y)
        weights = (y != -100).float()
        cursor = 0
        for kind, row, weight in (
                ('nonce', chunks[0], a.nonce_site_weight),
                ('list', chunks[1], a.list_site_weight),
                ('bracket', chunks[2], a.bracket_site_weight)):
            ix = torch.arange(cursor, cursor + len(row['x']), device=a.device)
            if kind == 'nonce':
                weights[ix[:, None], row['site']] = weight
            elif kind == 'list':
                weights[ix, row['site'][:, 0]] = weight
            else:
                weights[ix, row['site_position']] = weight
            cursor += len(row['x'])
        loss = (per * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 500 == 0:
            print('step', step + 1, 'loss',
                  round(sum(losses[-50:]) / 50, 3), flush=True)
    model.eval()
    final = score(model, nonce, lists, brackets, joint, a.device)
    report = {
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(a).items()},
        'train': {'nonce_source_documents': len(nonce['paired']['train']['left']['x']),
                  'nonce_paired_arms': len(nonce_train['x']),
                  'list_arms': len(list_train['x']),
                  'bracket_sources': len(bracket_train['x']),
                  'loss_last50': sum(losses[-50:]) / 50},
        'initial': initial, 'final': final,
        'scope': 'Continue a joint-trained two-rule model with balanced '
                 'nonce-copy, list, and bracket examples; all-token CE plus '
                 'marked-site weights. Test source documents and heldout '
                 'nonce codes are never used in training. The nonce rule '
                 'is a fictional alternate-world alias, not a whole-text '
                 'oracle or verified natural-world entity reference.'}
    a.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                'depth': 2, 'd_model': 64, 'heads': 4, 'n_ctx': 256,
                'vocab_size': 50257, 'norm': 'none'}, a.checkpoint)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
