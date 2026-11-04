"""Distill low-rank OV factors at separate FineWeb list/bracket rule sites."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model, evaluate as eval_bracket
from train_corpus_list_delimiter_mixture import eval_joint, eval_list
from eval_corpus_list_ov_reduction import site_kl
from joint_list_bracket_causal import evaluate as eval_causal_joint
from ov_linear_reduction import activation_grams, attentions, replace_ov


def trainable_ov(model, rank):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for layer in attentions(model):
        keep = torch.zeros(layer.n_head * layer.d_head,
                           device=layer.o.weight.device)
        for head in range(layer.n_head):
            keep[head * layer.d_head:head * layer.d_head + rank] = 1
        for parameter, mask in ((layer.o.weight, keep[None, :]),
                                (layer.v.weight, keep[:, None])):
            parameter.requires_grad_(True)
            parameter.register_hook(lambda grad, mask=mask: grad * mask)
            trainable.append(parameter)
    return trainable


@torch.no_grad()
def unused_channel_max(model, rank):
    return max((max(layer.o.weight[:, head * layer.d_head + rank:
                                  (head + 1) * layer.d_head].abs().max().item(),
                    layer.v.weight[head * layer.d_head + rank:
                                   (head + 1) * layer.d_head].abs().max().item())
                for layer in attentions(model)
                for head in range(layer.n_head)), default=0.)


def training_rows(data, brackets, limit):
    list_arms = [data['list_by_ordinal']['train'][ordinal][arm]
                 for ordinal in (3, 4, 5) for arm in (0, 1)]
    list_rows = {key: torch.cat([row[key][:limit] for row in list_arms])
                 for key in ('x', 'y', 'site')}
    bracket_rows = {key: brackets['train'][key][:limit]
                    for key in ('x', 'y')}
    bracket_rows['site'] = brackets['train']['site_position'][:limit, None]
    return list_rows, bracket_rows


@torch.no_grad()
def background_kl(teacher, student, arms, device, site_key, batch_size=4):
    total, count = 0., 0
    for row in arms:
        for start in range(0, len(row['x']), batch_size):
            x = row['x'][start:start + batch_size].to(device)
            y = row['y'][start:start + batch_size].to(device)
            site = row[site_key][start:start + batch_size].to(device)
            site = site[:, 0] if site.ndim == 2 else site
            positions = torch.arange(0, x.shape[1], 32, device=device)
            mask = (y[:, positions] != -100) & (positions[None, :] != site[:, None])
            if not mask.any():
                continue
            reference = teacher(x).float()[:, positions][mask]
            candidate = student(x).float()[:, positions][mask]
            total += F.kl_div(candidate.log_softmax(-1), reference.softmax(-1),
                              reduction='none').sum(-1).sum().item()
            count += int(mask.sum())
    return total / count


def scores(teacher, student, data, brackets, device, joint_panel=None):
    list_test = data['list']['test']
    bracket_test = brackets['test']
    return {
        'list': eval_list(student, list_test, device),
        'bracket': eval_bracket(student, bracket_test, device, batch_size=4),
        'joint': eval_joint(student, joint_panel if joint_panel is not None
                            else data['joint']['test'], device),
        'joint_causal_effects': eval_causal_joint(
            teacher, student, joint_panel if joint_panel is not None
            else data['joint']['test'], device),
        'list_site_teacher_kl': site_kl(teacher, student,
                                       [list_test[0], list_test[1]], device, 'site'),
        'bracket_site_teacher_kl': site_kl(teacher, student,
                                          [bracket_test['round'], bracket_test['square']],
                                          device, 'site_position'),
        'list_background_teacher_kl': background_kl(teacher, student,
                                                    [list_test[0], list_test[1]],
                                                    device, 'site'),
        'bracket_background_teacher_kl': background_kl(teacher, student,
                                                       [bracket_test['round'],
                                                        bracket_test['square']],
                                                       device, 'site_position'),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--joint-cache', type=Path,
                   help='Independent four-arm test panel; never used in fitting')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--fit-rows-per-arm', type=int, default=64)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--background-weight', type=float, default=.1)
    p.add_argument('--scopes', default='list,bracket,both')
    p.add_argument('--seed', type=int, default=71601)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    data = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(a.bracket_cache, map_location='cpu', weights_only=True)
    joint_panel = (torch.load(a.joint_cache, map_location='cpu',
                              weights_only=True) if a.joint_cache else None)
    assert data['n_ctx'] == brackets['n_ctx'] == 256
    list_rows, bracket_rows = training_rows(data, brackets, a.fit_rows_per_arm)
    gram_rows = {key: torch.cat([list_rows[key], bracket_rows[key]])
                 for key in ('x', 'y')}
    rows = {'list': list_rows, 'bracket': bracket_rows}
    report = {'config': {key: ([str(v) for v in value] if isinstance(value, list)
                               else str(value) if isinstance(value, Path) else value)
                         for key, value in vars(a).items()},
              'fit_rows': {key: len(value['x']) for key, value in rows.items()},
              'scope': 'Only retained per-head OV coordinates optimized on separate-rule '
                       'train windows. QK, MLP, embedding, and unembedding frozen. '
                       'Joint four-arm documents are test only; teacher KL is '
                       'functional fidelity, not independent task discovery.',
              'models': {}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        teacher = _model(50257, 256, 2, 64, 4, a.device, 'none')
        teacher.load_state_dict(saved['state_dict'])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        grams, counts = activation_grams(teacher, gram_rows, a.device)
        model_results = {'gram_counts': counts, 'rows': []}
        report['models'][path.stem] = model_results
        for method in ('weight', 'data'):
            student = deepcopy(teacher)
            replace_ov(student, grams, a.rank, method)
            model_results['rows'].append({'method': method,
                                          'scores': scores(teacher, student, data,
                                                           brackets, a.device,
                                                           joint_panel)})
            del student
        for scope in a.scopes.split(','):
            assert scope in ('list', 'bracket', 'both')
            student = deepcopy(teacher)
            replace_ov(student, grams, a.rank, 'weight')
            trainable = trainable_ov(student, a.rank)
            optimizer = torch.optim.Adam(trainable, lr=a.lr)
            torch.manual_seed(a.seed)
            losses = []
            for step in range(a.steps):
                kinds = ([scope] * a.batch_size if scope != 'both'
                         else ['list' if i % 2 == 0 else 'bracket'
                               for i in range(a.batch_size)])
                chunks = []
                for kind in ('list', 'bracket'):
                    size = kinds.count(kind)
                    if size:
                        indices = torch.randint(len(rows[kind]['x']), (size,))
                        chunks.append({key: value[indices].to(a.device)
                                       for key, value in rows[kind].items()})
                x = torch.cat([chunk['x'] for chunk in chunks])
                y = torch.cat([chunk['y'] for chunk in chunks])
                site = torch.cat([chunk['site'] for chunk in chunks])[:, 0]
                ix = torch.arange(len(x), device=a.device)
                with torch.no_grad():
                    reference = teacher(x).float()
                output = student(x).float()
                loss = F.kl_div(output[ix, site].log_softmax(-1),
                                reference[ix, site].softmax(-1),
                                reduction='batchmean')
                if a.background_weight:
                    positions = torch.arange(0, x.shape[1], 32, device=a.device)
                    valid = ((y[:, positions] != -100) &
                             (positions[None, :] != site[:, None]))
                    if valid.any():
                        bg = F.kl_div(output[:, positions][valid].log_softmax(-1),
                                      reference[:, positions][valid].softmax(-1),
                                      reduction='batchmean')
                        loss = loss + a.background_weight * bg
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss {path} {scope} {step}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
                if (step + 1) % 100 == 0:
                    print(path.stem, scope, 'step', step + 1,
                          'loss', round(sum(losses[-50:]) / 50, 4), flush=True)
            unused = unused_channel_max(student, a.rank)
            assert unused == 0., unused
            model_results['rows'].append({'method': 'rule_aware', 'scope': scope,
                                          'loss_last50': sum(losses[-50:]) /
                                                         min(50, len(losses)),
                                          'unused_channel_max': unused,
                                          'scores': scores(teacher, student, data,
                                                           brackets, a.device,
                                                           joint_panel)})
            a.output.write_text(json.dumps(report, indent=2) + '\n')
            del student
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
