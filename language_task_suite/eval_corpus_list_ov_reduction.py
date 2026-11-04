"""Weight-only versus train-data-aware OV truncation on learned list rules."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ov_linear_reduction import activation_grams, replace_ov
from train_corpus_delimiter_bilinear import _model, evaluate as eval_bracket
from train_corpus_list_delimiter_mixture import eval_joint, eval_list


@torch.no_grad()
def site_kl(teacher, student, arms, device, site_key, batch_size=4):
    total = 0.
    count = 0
    for row in arms:
        for start in range(0, len(row['x']), batch_size):
            batch = {k: v[start:start+batch_size].to(device)
                     for k, v in row.items()}
            ix = torch.arange(len(batch['x']), device=device)
            position = batch[site_key]
            if position.ndim == 2:
                position = position[:, 0]
            reference = teacher(batch['x']).float()[ix, position]
            result = student(batch['x']).float()[ix, position]
            total += F.kl_div(result.log_softmax(-1),
                              reference.softmax(-1),
                              reduction='none').sum(-1).sum().item()
            count += len(batch['x'])
    return total/count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--ranks', default='4,8,16')
    p.add_argument('--fit-list-rows', type=int, default=192)
    p.add_argument('--fit-bracket-rows', type=int, default=192)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    data = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(a.bracket_cache, map_location='cpu',
                          weights_only=True)
    assert data['n_ctx'] == brackets['n_ctx'] == 256
    fitting = {key: torch.cat([
        data['list']['train'][0][key][:a.fit_list_rows],
        data['list']['train'][1][key][:a.fit_list_rows],
        brackets['train'][key][:a.fit_bracket_rows]])
        for key in ('x', 'y')}
    report = {
        'config': {key: ([str(v) for v in value] if isinstance(value, list)
                         else str(value) if isinstance(value, Path) else value)
                   for key, value in vars(a).items()},
        'fit_rows': len(fitting['x']), 'models': {},
        'scope': 'Activation covariance fitted only on separate-rule '
                 'training windows. Rank truncates OV products per head; '
                 'QK, MLP, embedding, and unembedding remain frozen. '
                 'Heldout list/bracket behavior and joint four-arm '
                 'candidate fidelity are separate from reconstruction.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        teacher = _model(50257, 256, 2, 64, 4, a.device, 'none')
        teacher.load_state_dict(saved['state_dict'])
        teacher.eval()
        grams, counts = activation_grams(teacher, fitting, a.device)
        test = data['list']['test']
        baseline = {
            'list': eval_list(teacher, test, a.device),
            'bracket': eval_bracket(teacher, brackets['test'], a.device,
                                    batch_size=4),
            'joint': eval_joint(teacher, data['joint']['test'], a.device)}
        rows = []
        for method in ('weight', 'data'):
            for rank in (int(value) for value in a.ranks.split(',')):
                candidate = deepcopy(teacher)
                errors = replace_ov(candidate, grams, rank, method)
                with torch.no_grad():
                    probe = float((teacher(test[0]['x'][:2].to(a.device)) -
                                   candidate(test[0]['x'][:2].to(a.device)))
                                  .abs().max())
                if rank == 16 and probe > .001:
                    raise AssertionError(f'Full rank changed logits: {probe}')
                score = {
                    'method': method, 'rank_per_head': rank,
                    'probe_max_logit_change': probe,
                    'mean_weight_rel_error': sum(e['weight_rel_error']
                         for e in errors)/len(errors),
                    'mean_input_weighted_rel_error': sum(
                         e['input_weighted_rel_error'] for e in errors)/
                         len(errors),
                    'list': eval_list(candidate, test, a.device),
                    'bracket': eval_bracket(candidate, brackets['test'],
                                            a.device, batch_size=4),
                    'joint': eval_joint(candidate, data['joint']['test'],
                                        a.device),
                    'list_site_teacher_kl': site_kl(
                        teacher, candidate, [test[0], test[1]],
                        a.device, 'site'),
                    'bracket_site_teacher_kl': site_kl(
                        teacher, candidate,
                        [brackets['test']['round'],
                         brackets['test']['square']],
                        a.device, 'site_position')}
                rows.append(score)
                print(path.stem, method, rank,
                      'list', round(score['list']['both_top1'], 3),
                      'bracket2way', round(
                          score['bracket']['both_two_way'], 3),
                      'joint2way', round(score['joint'][
                          'all_arms_both_rules_two_way'], 3), flush=True)
                report['models'][path.stem] = {
                    'fit_activation_counts': counts,
                    'baseline': baseline, 'rows': rows}
                a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
