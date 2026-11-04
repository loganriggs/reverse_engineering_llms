"""Matched natural-code versus source-scrambled bilinear continuations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from audit_fineweb_surfaces import DEFAULT_TOKENIZER
from eval_corpus_rule_reference_transfer import one_kind
from train_corpus_delimiter_bilinear import _model


def load_records(paths):
    parts = [torch.load(path, map_location='cpu', weights_only=True)
             for path in paths]
    ranges = [(part['start'], part['start']+part['n']) for part in parts]
    if len(set(ranges)) != len(ranges):
        raise ValueError('Repeated source range')
    for i, (left, right) in enumerate(ranges):
        if any(max(left, other_left) < min(right, other_right)
               for j, (other_left, other_right) in enumerate(ranges) if i != j):
            raise ValueError('Overlapping source ranges')
    return [row for part in parts for row in part['records']], ranges


def training_tensors(records, condition):
    rows = {'x': [], 'y': [], 'site': []}
    for record in records:
        site = record['pos_natural']
        arms = ('natural', 'edit') if condition == 'paired' else (condition,)
        for arm in arms:
            x = (record['x_natural'] if arm == 'natural'
                 else record['x_edit']).clone()
            y = (record['y_natural'] if arm == 'natural'
                 else record['y_edit']).clone()
            if arm == 'source_scrambled':
                y[site] = record['token_natural']
                assert (x[:site+1] != record['x_natural'][:site+1]).any()
            expected = (record['token_edit'] if arm == 'edit'
                        else record['token_natural'])
            assert y[site].item() == expected
            y[site+1:] = -100  # Prefix-through-answer loss for all conditions.
            rows['x'].append(x)
            rows['y'].append(y)
            rows['site'].append(site)
    return {key: (torch.stack(value) if key != 'site' else
                  torch.tensor(value, dtype=torch.long))
            for key, value in rows.items()}


@torch.no_grad()
def scores(model, panels, train_codes, train_pairs, device, tokenizer):
    result = {}
    for label, records in panels.items():
        seen = [row for row in records
                if row['source_code'] in train_codes]
        unseen = [row for row in records
                  if row['source_code'] not in train_codes]
        new_pair = [row for row in records
                    if (row['source_code'], row['edit_code']) not in train_pairs]
        result[label] = {
            'all': one_kind(model, records, device, 4, tokenizer),
            'seen_source_code': one_kind(model, seen, device, 4, tokenizer)
            if seen else None,
            'unseen_source_code': one_kind(model, unseen, device, 4, tokenizer)
            if unseen else None,
            'new_code_pair': one_kind(model, new_pair, device, 4, tokenizer)
            if new_pair else None,
        }
    return result


def train_condition(model, data, a, condition, panels, train_codes,
                    train_pairs, tokenizer):
    path = a.checkpoint_dir / f'{a.run_name}_{condition}.resume.pt'
    final_path = a.checkpoint_dir / f'{a.run_name}_{condition}.pt'
    resume = torch.load(path, map_location='cpu', weights_only=True) \
        if path.exists() else None
    identity = (condition, a.steps, a.sample_seed, a.lr, a.site_weight,
                a.batch_size, len(data['x']))
    if resume and tuple(resume['identity']) != identity:
        raise ValueError('Resume file does not match this training run')
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    generator = torch.Generator(device='cpu').manual_seed(a.sample_seed)
    if resume:
        model.load_state_dict(resume['state_dict'])
        optimizer.load_state_dict(resume['optimizer'])
        generator.set_state(resume['sample_rng'])
        if a.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(resume['cuda_rng'])
        losses = resume['losses']
        start = resume['next_step']
        initial = resume['initial']
        print(condition, 'resuming', start, flush=True)
    else:
        start = 0
        losses = []
        initial = scores(model, panels, train_codes, train_pairs,
                         a.device, tokenizer)
        print(condition, 'initial panel one', initial['panel_1']['all']
              ['paired_two_way_fraction'], flush=True)
    model.eval()
    for step in range(start, a.steps):
        indices = torch.randint(len(data['x']), (a.batch_size,),
                                generator=generator)
        x = data['x'][indices].to(a.device)
        y = data['y'][indices].to(a.device)
        site = data['site'][indices].to(a.device)
        out = model(x).float()
        per = F.cross_entropy(out.flatten(0, 1), y.flatten(),
                              ignore_index=-100,
                              reduction='none').view_as(y)
        weights = (y != -100).float()
        weights[torch.arange(len(x), device=a.device), site] = a.site_weight
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite {condition} step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % a.save_every == 0:
            print(condition, 'step', step+1, 'loss',
                  round(sum(losses[-50:])/min(50,len(losses)), 3), flush=True)
            state = {'identity': identity, 'next_step': step+1,
                     'state_dict': {key: value.detach().cpu()
                                    for key, value in model.state_dict().items()},
                     'optimizer': optimizer.state_dict(),
                     'sample_rng': generator.get_state(),
                     'cuda_rng': torch.cuda.get_rng_state_all()
                                  if a.device.startswith('cuda') else [],
                     'losses': losses, 'initial': initial}
            a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            torch.save(state, temp)
            temp.replace(path)
    final = scores(model, panels, train_codes, train_pairs,
                   a.device, tokenizer)
    saved = {'state_dict': {key: value.detach().cpu()
                            for key, value in model.state_dict().items()},
             'depth': a.depth, 'd_model': a.d_model, 'heads': a.heads,
             'n_ctx': 256, 'vocab_size': 50257, 'norm': a.norm}
    temp = final_path.with_suffix('.tmp')
    torch.save(saved, temp)
    temp.replace(final_path)
    path.unlink(missing_ok=True)
    return {'initial': initial, 'final': final,
            'loss_last50': sum(losses[-50:])/min(50,len(losses)),
            'checkpoint': str(final_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--train-caches', type=Path, nargs='+', required=True)
    p.add_argument('--test-caches', type=Path, nargs=2, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--conditions', default='natural,source_scrambled')
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--site-weight', type=float, default=200.)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--sample-seed', type=int, default=83501)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    train, train_ranges = load_records(a.train_caches)
    test_1, r1 = load_records(a.test_caches[:1])
    test_2, r2 = load_records(a.test_caches[1:])
    all_ranges = train_ranges+r1+r2
    assert len(set(all_ranges)) == len(all_ranges)
    assert not any(max(left, other_left) < min(right, other_right)
                   for i, (left, right) in enumerate(all_ranges)
                   for j, (other_left, other_right) in enumerate(all_ranges)
                   if i != j)
    panels = {'panel_1': test_1, 'panel_2': test_2}
    train_codes = {row['source_code'] for row in train}
    train_pairs = {(row['source_code'], row['edit_code']) for row in train}
    saved = torch.load(a.initial_checkpoint, map_location='cpu',
                       weights_only=True)
    a.depth, a.d_model, a.heads, a.norm = (saved['depth'], saved['d_model'],
                                          saved['heads'], saved['norm'])
    torch.manual_seed(a.sample_seed)
    base = _model(saved['vocab_size'], saved['n_ctx'], a.depth, a.d_model,
                  a.heads, a.device, a.norm)
    base.load_state_dict(saved['state_dict'])
    tokenizer = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    report = {'config': {key: [str(x) for x in value] if isinstance(value, list)
                         else str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'train': {'sources': sum(right-left for left,right in train_ranges),
                        'selected_documents': len(train),
                        'distinct_source_codes': len(train_codes),
                        'distinct_source_edit_pairs': len(train_pairs)},
              'test': {label: {'selected_documents': len(records),
                               'new_code_pair': sum((row['source_code'],
                                                     row['edit_code'])
                                                    not in train_pairs
                                                    for row in records),
                               'unseen_source_code': sum(row['source_code']
                                                         not in train_codes
                                                         for row in records)}
                       for label, records in panels.items()},
              'scope': 'Matched continuations from identical bilinear checkpoint '
                       'and source-document rows. Natural condition sees '
                       'unmodified real-code prefixes. Source-scrambled '
                       'control substitutes all prior selected-code tokens '
                       'but retains original target identity/frequency. '
                       'Paired condition trains on natural plus one '
                       'consistent acronym-substitution arm per source, '
                       'an explicitly enriched alternate-world distribution. '
                       'All conditions use next-token CE through target '
                       'only and upweight the target. No model weight '
                       'claim until heldout paired causal response passes.',
              'conditions': {}}
    for condition in a.conditions.split(','):
        if condition not in ('natural', 'source_scrambled', 'paired'):
            raise ValueError(condition)
        data = training_tensors(train, condition)
        model = deepcopy(base)
        result = train_condition(model, data, a, condition, panels,
                                 train_codes, train_pairs, tokenizer)
        report['conditions'][condition] = result
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2)+'\n')
        del model, data
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
