"""Weight-only versus train-activation OV truncation on three learned rules."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ov_linear_reduction import activation_grams, replace_ov
from train_corpus_delimiter_bilinear import _model, evaluate as eval_bracket
from train_corpus_list_delimiter_mixture import eval_joint, eval_list
from eval_corpus_nonce_copy import evaluate as eval_nonce
from probe_corpus_three_rule_heads import weight_scores


def fit_rows(nonce, lists, brackets, per_rule, draws_per_source, seed):
    generator = torch.Generator().manual_seed(seed)
    arms = []
    for arm in ('left', 'right'):
        row = nonce['paired']['train'][arm]
        n_sources = len(row['x']) // draws_per_source
        indices = torch.randperm(n_sources, generator=generator)[:per_rule]
        indices = indices * draws_per_source
        arms.append({key: row[key][indices] for key in ('x', 'y')})
    for arm in (0, 1):
        row = lists['list']['train'][arm]
        indices = torch.randperm(len(row['x']), generator=generator)[:per_rule]
        arms.append({key: row[key][indices] for key in ('x', 'y')})
    row = brackets['train']
    indices = torch.randperm(len(row['x']), generator=generator)[:2 * per_rule]
    arms.append({key: row[key][indices] for key in ('x', 'y')})
    return {key: torch.cat([row[key] for row in arms]) for key in ('x', 'y')}


@torch.no_grad()
def teacher_kl(teacher, student, row, site_key, device, background=False,
               max_rows=64, batch_size=4):
    total, count = 0., 0
    for start in range(0, min(len(row['x']), max_rows), batch_size):
        x = row['x'][start:start + batch_size].to(device)
        y = row['y'][start:start + batch_size].to(device)
        site = row[site_key][start:start + batch_size].to(device)
        if site.ndim == 1:
            site = site[:, None]
        if background:
            positions = torch.arange(0, x.shape[1], 32, device=device)
            mask = ((y[:, positions] != -100) &
                    (positions[None, :, None] != site[:, None, :]).all(-1))
            if not mask.any():
                continue
            ref = teacher(x).float()[:, positions][mask]
            out = student(x).float()[:, positions][mask]
        else:
            ix = torch.arange(len(x), device=device)
            ref = teacher(x).float()[ix[:, None], site].flatten(0, 1)
            out = student(x).float()[ix[:, None], site].flatten(0, 1)
        total += F.kl_div(out.log_softmax(-1), ref.softmax(-1),
                          reduction='none').sum(-1).sum().item()
        count += len(out)
    return total / count


def behavior(model, nonce, lists, brackets, joint, device):
    return {'nonce_seen': eval_nonce(model, nonce['paired']['test'], device),
            'nonce_unseen': eval_nonce(model, nonce['test_unseen_codes'], device),
            'list': eval_list(model, lists['list']['test'], device),
            'bracket': eval_bracket(model, brackets['test'], device,
                                    batch_size=4),
            'list_bracket_joint': eval_joint(model, joint, device)}


def fidelity(teacher, student, nonce, lists, brackets, device):
    panels = {'nonce': (nonce['paired']['test']['left'], 'site'),
              'list': (lists['list']['test'][0], 'site'),
              'bracket': (brackets['test']['round'], 'site_position')}
    return {kind: {'site_teacher_kl': teacher_kl(teacher, student, row, key,
                                                device),
                   'background_teacher_kl': teacher_kl(
                       teacher, student, row, key, device, background=True)}
            for kind, (row, key) in panels.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--nonce-cache', type=Path, required=True)
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--joint-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ranks', default='8,16,24,32')
    p.add_argument('--kept-heads', default='8,12')
    p.add_argument('--fit-sources-per-arm', type=int, default=64)
    p.add_argument('--nonce-draws-per-source', type=int, default=8)
    p.add_argument('--seed', type=int, default=72301)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    nonce = torch.load(a.nonce_cache, map_location='cpu', weights_only=True)
    lists = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(a.bracket_cache, map_location='cpu', weights_only=True)
    joint = torch.load(a.joint_cache, map_location='cpu', weights_only=True)
    fitting = fit_rows(nonce, lists, brackets, a.fit_sources_per_arm,
                       a.nonce_draws_per_source, a.seed)
    report = {'config': {key: [str(x) for x in value] if isinstance(value, list)
                         else str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'fit_rows': len(fitting['x']), 'models': {},
              'scope': 'Per-head OV product SVD; all QK, MLP, embeddings, '
                       'unembedding and other functions stay frozen. Data '
                       'metric fitted on disjoint training-source windows, '
                       'balanced across rule families with one alias draw '
                       'per nonce source. No heldout document or joint '
                       'list-bracket combination enters the fit. This is '
                       'a local OV canonical reduction, not a canonical '
                       'form of the full attention computation.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    ranks = [int(value) for value in a.ranks.split(',')]
    for path in a.checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        teacher = _model(saved['vocab_size'], saved['n_ctx'], saved['depth'],
                         saved['d_model'], saved['heads'], a.device,
                         saved['norm'])
        teacher.load_state_dict(saved['state_dict'])
        teacher.eval()
        grams, counts = activation_grams(teacher, fitting, a.device)
        key = path.stem
        rows = []
        report['models'][key] = {
            'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'fit_activation_counts': counts,
            'native': behavior(teacher, nonce, lists, brackets, joint,
                               a.device), 'rows': rows}
        head_scores = weight_scores(teacher)
        n_heads = len(head_scores)
        for method in ('weight', 'data'):
            for rank in ranks:
                student = deepcopy(teacher)
                errors = replace_ov(student, grams, rank, method)
                with torch.no_grad():
                    x = nonce['paired']['test']['left']['x'][:2].to(a.device)
                    max_logit_change = float((teacher(x) - student(x)).abs().max())
                if rank == saved['d_model'] // saved['heads']:
                    assert max_logit_change < .001, max_logit_change
                row = {'method': method, 'rank_per_head': rank,
                       'retained_ov_rank_budget': rank * n_heads,
                       'max_logit_change_probe': max_logit_change,
                       'mean_weight_rel_error': sum(e['weight_rel_error']
                           for e in errors) / len(errors),
                       'mean_input_weighted_rel_error': sum(
                           e['input_weighted_rel_error'] for e in errors) /
                           len(errors),
                       'behavior': behavior(student, nonce, lists, brackets,
                                            joint, a.device),
                       'fidelity': fidelity(teacher, student, nonce, lists,
                                            brackets, a.device)}
                rows.append(row)
                a.output.write_text(json.dumps(report, indent=2) + '\n')
                print(key, method, rank,
                      'nonce', round(row['behavior']['nonce_seen']['greedy_both'], 3),
                      'list', round(row['behavior']['list']['both_top1'], 3),
                      'bracket', round(row['behavior']['bracket']['both_two_way'], 3),
                      flush=True)
        for kept in (int(value) for value in a.kept_heads.split(',')):
            assert 1 <= kept <= n_heads
            student = deepcopy(teacher)
            ranked = sorted(head_scores, key=lambda row: -row['product_norm'])
            pruned = ranked[kept:]
            for head in pruned:
                layer = student.layers[head['layer']]
                sl = slice(head['head'] * layer.d_head,
                           (head['head'] + 1) * layer.d_head)
                with torch.no_grad():
                    layer.o.weight[:, sl].zero_()
                    layer.v.weight[sl, :].zero_()
            row = {'method': 'weight_head_prune', 'heads_kept': kept,
                   'retained_ov_rank_budget': kept * student.layers[
                       head_scores[0]['layer']].d_head,
                   'pruned_heads': [{'layer': head['layer'],
                                    'head': head['head'],
                                    'weight_rank': rank + kept + 1}
                                   for rank, head in enumerate(pruned)],
                   'behavior': behavior(student, nonce, lists, brackets,
                                        joint, a.device),
                   'fidelity': fidelity(teacher, student, nonce, lists,
                                        brackets, a.device)}
            rows.append(row)
            a.output.write_text(json.dumps(report, indent=2) + '\n')
            print(key, 'weight_head_prune', kept,
                  'nonce', round(row['behavior']['nonce_seen']['greedy_both'], 3),
                  'list', round(row['behavior']['list']['both_top1'], 3),
                  'bracket', round(row['behavior']['bracket']['both_two_way'], 3),
                  flush=True)
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
