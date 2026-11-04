"""Joint code/list/bracket continuation with ordinary-order heldout gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_transfer import one_kind
from train_corpus_delimiter_bilinear import _model
from train_corpus_list_joint_supervised import list_rows


def code_rows(records, arm):
    key = 'natural' if arm == 'natural' else 'edit'
    return {'x': torch.stack([row['x_'+key] for row in records]),
            'y': torch.stack([row['y_'+key] for row in records]),
            'site': torch.tensor([row['pos_'+key] for row in records])}


def sample(data, count, generator, device):
    indices = torch.randint(len(data['x']), (count,), generator=generator)
    return {key: value[indices].to(device) for key, value in data.items()}


@torch.no_grad()
def evaluate(model, code_panels, other_panels, train_codes, train_pairs,
             device, tokenizer):
    report = {'code': {}, 'list_bracket': {}}
    for label, records in code_panels.items():
        new_pair = [row for row in records
                    if (row['source_code'], row['edit_code']) not in train_pairs]
        unseen_code = [row for row in records
                       if row['source_code'] not in train_codes]
        report['code'][label] = {
            'all': one_kind(model, records, device, 4, tokenizer),
            'new_pair': one_kind(model, new_pair, device, 4, tokenizer),
            'unseen_source_code': one_kind(model, unseen_code, device,
                                           4, tokenizer)}
    for label, panel in other_panels.items():
        report['list_bracket'][label] = {
            kind: one_kind(model, [row for row in panel if row['kind'] == kind],
                           device, 4, tokenizer)
            for kind in ('list', 'bracket')}
    return report


def save_model(path, model, a):
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = {'state_dict': {key: value.detach().cpu()
                            for key, value in model.state_dict().items()},
             'depth': a.depth, 'd_model': a.d_model, 'heads': a.heads,
             'norm': a.norm, 'n_ctx': 256, 'vocab_size': 50257}
    temp = path.with_suffix('.tmp')
    torch.save(saved, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--code-train-caches', type=Path, nargs='+', required=True)
    p.add_argument('--code-test-caches', type=Path, nargs=2, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--other-test-caches', type=Path, nargs=2, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--resume-state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--eval-steps', default='2000,4000')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--code-site-weight', type=float, default=200.)
    p.add_argument('--list-site-weight', type=float, default=500.)
    p.add_argument('--bracket-site-weight', type=float, default=500.)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--sample-seed', type=int, default=83601)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.batch_size != 8:
        raise ValueError('This pilot uses a fixed 2+2+2+2 batch allocation')
    eval_steps = {int(x) for x in a.eval_steps.split(',')}
    torch.set_num_threads(4)
    code_train_parts = [torch.load(path, map_location='cpu', weights_only=True)
                        for path in a.code_train_caches]
    code_train = [row for part in code_train_parts for row in part['records']]
    code_panels = {f'panel_{i+1}': torch.load(path, map_location='cpu',
                                             weights_only=True)['records']
                   for i, path in enumerate(a.code_test_caches)}
    other_panels = {f'panel_{i+1}': torch.load(path, map_location='cpu',
                                              weights_only=True)['records']
                    for i, path in enumerate(a.other_test_caches)}
    ranges = [(part['start'], part['start']+part['n'])
              for part in code_train_parts]
    ranges += [(torch.load(path, map_location='cpu', weights_only=True)['start'],
                torch.load(path, map_location='cpu', weights_only=True)['start']+
                torch.load(path, map_location='cpu', weights_only=True)['n'])
               for path in a.code_test_caches]
    assert not any(max(left, other_left) < min(right, other_right)
                   for i, (left, right) in enumerate(ranges)
                   for j, (other_left, other_right) in enumerate(ranges)
                   if i != j)
    train_codes = {row['source_code'] for row in code_train}
    train_pairs = {(row['source_code'], row['edit_code'])
                   for row in code_train}
    lists = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(a.bracket_cache, map_location='cpu', weights_only=True)
    l = list_rows(lists)
    b = {'x': brackets['train']['x'], 'y': brackets['train']['y'],
         'site': brackets['train']['site_position']}
    data = {'code_natural': code_rows(code_train, 'natural'),
            'code_edit': code_rows(code_train, 'edit'),
            'list': {'x': l['x'], 'y': l['y'],
                     'site': l['site'][:, 0]},
            'bracket': b}
    saved = torch.load(a.initial_checkpoint, map_location='cpu',
                       weights_only=True)
    a.depth, a.d_model, a.heads, a.norm = (saved['depth'], saved['d_model'],
                                          saved['heads'], saved['norm'])
    torch.manual_seed(a.sample_seed)
    model = _model(saved['vocab_size'], saved['n_ctx'], a.depth, a.d_model,
                   a.heads, a.device, a.norm)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    generator = torch.Generator(device='cpu').manual_seed(a.sample_seed)
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    identity = (a.run_name, a.steps, a.lr, a.batch_size,
                a.code_site_weight, a.list_site_weight,
                a.bracket_site_weight, a.sample_seed, len(code_train))
    resume = torch.load(a.resume_state, map_location='cpu', weights_only=True) \
        if a.resume_state.exists() else None
    if resume:
        if tuple(resume['identity']) != identity:
            raise ValueError('Resume file does not match this run')
        model.load_state_dict(resume['state_dict'])
        optimizer.load_state_dict(resume['optimizer'])
        generator.set_state(resume['sample_rng'])
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        initial, milestones, losses = (resume['initial'], resume['milestones'],
                                      resume['losses'])
        start = resume['next_step']
        print('resuming at', start, flush=True)
    else:
        initial = evaluate(model, code_panels, other_panels, train_codes,
                           train_pairs, a.device, tokenizer)
        milestones, losses, start = {}, [], 0
    report = {'config': {key: [str(x) for x in value] if isinstance(value, list)
                         else str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'train_sources': sum(part['n'] for part in code_train_parts),
              'train_code_documents': len(code_train),
              'train_code_pairs': len(train_pairs),
              'train_rows': {key: len(row['x']) for key,row in data.items()},
              'initial': initial, 'milestones': milestones,
              'scope': 'Balanced 2 natural-code + 2 edited-code + 2 list + '
                       '2 bracket rows/update. Code edits are explicit '
                       'alternate-world training augmentation; all rows '
                       'use full-token CE plus marked-site weights. '
                       'Code train and fresh code test source ranges disjoint; '
                       'fresh ordinary-order code/list/bracket panels used '
                       'only for evaluation. Not a full natural-text oracle.'}
    weights_by_kind = (a.code_site_weight, a.code_site_weight,
                       a.list_site_weight, a.bracket_site_weight)
    kinds = ('code_natural', 'code_edit', 'list', 'bracket')
    for step in range(start, a.steps):
        sampled = [sample(data[kind], 2, generator, a.device) for kind in kinds]
        x = torch.cat([row['x'] for row in sampled])
        y = torch.cat([row['y'] for row in sampled])
        out = model(x).float()
        per = F.cross_entropy(out.flatten(0, 1), y.flatten(),
                              ignore_index=-100,
                              reduction='none').view_as(y)
        weights = (y != -100).float()
        for index, (row, weight) in enumerate(zip(sampled, weights_by_kind)):
            ix = torch.arange(index*2, index*2+2, device=a.device)
            weights[ix, row['site']] = weight
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        completed = step+1
        if completed in eval_steps:
            milestones[str(completed)] = {
                'scores': evaluate(model, code_panels, other_panels,
                                   train_codes, train_pairs, a.device, tokenizer),
                'loss_last50': sum(losses[-50:])/min(50,len(losses))}
            save_model(a.checkpoint_dir/f'{a.run_name}_step{completed}.pt',
                       model, a)
            report['milestones'] = milestones
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2)+'\n')
            print('evaluated', completed, flush=True)
        if completed % a.save_every == 0:
            print('step', completed, 'loss',
                  round(sum(losses[-50:])/min(50,len(losses)), 3), flush=True)
            state = {'identity': identity, 'next_step': completed,
                     'state_dict': {key: value.detach().cpu()
                                    for key,value in model.state_dict().items()},
                     'optimizer': optimizer.state_dict(),
                     'sample_rng': generator.get_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all()
                                  if a.device.startswith('cuda') else [],
                     'initial': initial, 'milestones': milestones,
                     'losses': losses}
            a.resume_state.parent.mkdir(parents=True, exist_ok=True)
            temp = a.resume_state.with_suffix('.tmp')
            torch.save(state, temp)
            temp.replace(a.resume_state)
    a.resume_state.unlink(missing_ok=True)
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
