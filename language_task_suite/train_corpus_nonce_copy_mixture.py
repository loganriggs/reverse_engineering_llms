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
    p.add_argument('--initial-checkpoint', type=Path)
    p.add_argument('--nonce-cache', type=Path, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--joint-test-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--norm', choices=('none', 'rms'), default='none')
    p.add_argument('--resume-state', type=Path)
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
    resume = (torch.load(a.resume_state, map_location='cpu',
                         weights_only=True)
              if a.resume_state and a.resume_state.exists() else None)
    if resume is not None:
        assert resume['config'] == (a.depth, a.d_model, a.heads, a.norm,
                                    a.seed, a.steps)
        model.load_state_dict(resume['state_dict'])
        initial = resume['initial']
    else:
        initial = score(model, nonce, lists, brackets, joint, a.device)
    nonce_train = {key: torch.cat([nonce['paired']['train'][arm][key]
                                    for arm in ('left', 'right')])
                   for key in ('x', 'y', 'site')}
    list_train = list_rows(lists)
    bracket_train = brackets['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['cpu_rng'])
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        losses = resume['losses']
        start_step = resume['next_step']
        print('resuming at step', start_step, flush=True)
    else:
        torch.manual_seed(a.seed)
        losses = []
        start_step = 0
    for step in range(start_step, a.steps):
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
            if a.resume_state:
                state = {'config': (a.depth, a.d_model, a.heads, a.norm,
                                    a.seed, a.steps),
                         'next_step': step + 1,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'cpu_rng': torch.get_rng_state(),
                         'cuda_rng': torch.cuda.get_rng_state_all()
                                     if a.device.startswith('cuda') else [],
                         'initial': initial, 'losses': losses}
                a.resume_state.parent.mkdir(parents=True, exist_ok=True)
                temp = a.resume_state.with_suffix('.tmp')
                torch.save(state, temp)
                temp.replace(a.resume_state)
    model.eval()
    final = score(model, nonce, lists, brackets, joint, a.device)
    report = {
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(a).items()},
        'train': {'nonce_paired_realizations': len(nonce['paired']['train']['left']['x']),
                  'nonce_paired_arms': len(nonce_train['x']),
                  'list_arms': len(list_train['x']),
                  'bracket_sources': len(bracket_train['x']),
                  'loss_last50': sum(losses[-50:]) / 50},
        'initial': initial, 'final': final,
        'scope': 'Train on mixed nonce-copy, list, and bracket examples; all-token CE plus '
                 'marked-site weights. Test source documents and heldout '
                 'nonce codes are never used in training. The nonce rule '
                 'is a fictional alternate-world alias, not a whole-text '
                 'oracle or verified natural-world entity reference.'}
    a.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {key: value.detach().cpu()
                               for key, value in model.state_dict().items()},
                'depth': a.depth, 'd_model': a.d_model, 'heads': a.heads,
                'n_ctx': 256, 'vocab_size': 50257, 'norm': a.norm}, a.checkpoint)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    if a.resume_state:
        a.resume_state.unlink(missing_ok=True)
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
