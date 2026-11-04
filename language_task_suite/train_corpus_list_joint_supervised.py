"""Fine-tune list+bracket LMs with joint edits, testing on disjoint sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model, evaluate as eval_bracket
from train_corpus_list_delimiter_mixture import eval_joint, eval_list


def list_rows(data):
    arms = [data['list_by_ordinal']['train'][ordinal][shift]
            for ordinal in (3, 4, 5) for shift in (0, 1)]
    return {key: torch.cat([arm[key] for arm in arms])
            for key in ('x', 'y', 'site')}


def sample(rows, size, device):
    indices = torch.randint(len(rows['x']), (size,))
    return {key: value[indices].to(device) for key, value in rows.items()}


def train_one(model, list_data, bracket_data, joint_data, a):
    list_train = list_rows(list_data)
    bracket_train = bracket_data['train']
    joint_arms = list(joint_data.values())
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    torch.manual_seed(a.seed)
    losses = []
    for step in range(a.steps):
        joint = [sample(joint_arms[(step + i) % 4], a.batch_size // 8,
                        a.device) for i in range(4)]
        individual_list = sample(list_train, a.batch_size // 4, a.device)
        individual_bracket = sample(bracket_train, a.batch_size // 4, a.device)
        batches = joint + [individual_list, individual_bracket]
        x = torch.cat([batch['x'] for batch in batches])
        y = torch.cat([batch['y'] for batch in batches])
        output = model(x).float()
        per = F.cross_entropy(output.flatten(0, 1), y.flatten(),
                              ignore_index=-100, reduction='none').view_as(y)
        weights = (y != -100).float()
        cursor = 0
        for index, batch in enumerate(batches):
            count = len(batch['x'])
            ix = torch.arange(cursor, cursor + count, device=a.device)
            if index < 4:
                weights[ix, batch['site'][:, 0]] = a.bracket_site_weight
                weights[ix, batch['site'][:, 1]] = a.list_site_weight
            elif index == 4:
                weights[ix, batch['site'][:, 0]] = a.list_site_weight
            else:
                weights[ix, batch['site_position']] = a.bracket_site_weight
            cursor += count
        loss = (per * weights).sum() / weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 250 == 0:
            print('step', step + 1, 'loss', round(sum(losses[-50:]) / 50, 3),
                  flush=True)
    return {'last50_weighted_loss': sum(losses[-50:]) / min(50, len(losses)),
            'joint_train_sources': len(joint_arms[0]['x']),
            'list_train_arms': len(list_train['x']),
            'bracket_train_sources': len(bracket_train['x'])}


def evaluate(model, list_data, bracket_data, joint_test, device):
    return {
        'list_test': eval_list(model, list_data['list']['test'], device),
        'bracket_test': eval_bracket(model, bracket_data['test'], device,
                                     batch_size=4),
        'joint_original_test': eval_joint(model, list_data['joint']['test'],
                                          device),
        'joint_fresh_test': eval_joint(model, joint_test, device),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--joint-train-cache', type=Path, required=True)
    p.add_argument('--joint-test-cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--list-site-weight', type=float, default=100.)
    p.add_argument('--bracket-site-weight', type=float, default=100.)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--seed', type=int, default=71701)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    assert a.batch_size % 8 == 0
    torch.set_num_threads(4)
    list_data = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    bracket_data = torch.load(a.bracket_cache, map_location='cpu',
                              weights_only=True)
    joint_train = torch.load(a.joint_train_cache, map_location='cpu',
                             weights_only=True)
    joint_test = torch.load(a.joint_test_cache, map_location='cpu',
                            weights_only=True)
    saved = torch.load(a.initial_checkpoint, map_location='cpu',
                       weights_only=True)
    model = _model(50257, 256, 2, 64, 4, a.device, 'none')
    model.load_state_dict(saved['state_dict'])
    model.eval()
    initial = evaluate(model, list_data, bracket_data, joint_test, a.device)
    train = train_one(model, list_data, bracket_data, joint_train, a)
    model.eval()
    final = evaluate(model, list_data, bracket_data, joint_test, a.device)
    report = {
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(a).items()},
        'train': train, 'initial': initial, 'final': final,
        'scope': 'Joint-supervised capability condition. Four joint edit arms '
                 'from source offsets 115000..214999 mixed with independent '
                 'list/bracket windows; fresh joint test sources '
                 '235000..334999 never used in training. This tests '
                 'source-disjoint transfer, not zero-shot composition.'}
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
