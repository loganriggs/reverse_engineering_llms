"""Continue a natural-delimiter bilinear LM on separate list and bracket tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from train_corpus_delimiter_bilinear import _model, evaluate as eval_bracket


@torch.no_grad()
def eval_list(model, pair, device, batch_size=8):
    model.eval()
    n = len(pair[0]['x'])
    counts = dict(top1_0=0, top1_1=0, both_top1=0,
                  two_way_0=0, two_way_1=0, paired_two_way=0,
                  positive_edit=0, nll=0.)
    for start in range(0, n, batch_size):
        rows = [{k: v[start:start+batch_size].to(device)
                 for k, v in pair[arm].items()} for arm in (0, 1)]
        logits = [model(row['x']).float() for row in rows]
        ix = torch.arange(len(rows[0]['x']), device=device)
        z = [out[ix, row['site'][:, 0]] for out, row in zip(logits, rows)]
        a, b = rows[0]['target'][:, 0], rows[1]['target'][:, 0]
        assert (a != b).all()
        c0, c1 = z[0].argmax(-1) == a, z[1].argmax(-1) == b
        p0 = z[0][ix, a] > z[0][ix, b]
        p1 = z[1][ix, b] > z[1][ix, a]
        delta = (z[1][ix, b]-z[1][ix, a]) - \
                (z[0][ix, b]-z[0][ix, a])
        counts['top1_0'] += c0.sum().item()
        counts['top1_1'] += c1.sum().item()
        counts['both_top1'] += (c0 & c1).sum().item()
        counts['two_way_0'] += p0.sum().item()
        counts['two_way_1'] += p1.sum().item()
        counts['paired_two_way'] += (p0 & p1).sum().item()
        counts['positive_edit'] += (delta > 0).sum().item()
        counts['nll'] += sum(F.cross_entropy(out, row['target'][:, 0],
                                            reduction='sum').item()
                             for out, row in zip(z, rows))
    return {'n': n, **{key: (value/(2*n) if key == 'nll' else value/n)
                       for key, value in counts.items()}}


@torch.no_grad()
def eval_joint(model, panel, device, batch_size=4):
    model.eval()
    n = len(panel[(0, 'round')]['x'])
    totals = {kind: dict(paired_top1=0, paired_two_way=0,
                         edit_direction=0) for kind in ('bracket', 'list')}
    all_exact = all_two_way = 0
    for start in range(0, n, batch_size):
        rows = {arm: {k: v[start:start+batch_size].to(device)
                      for k, v in panel[arm].items()} for arm in panel}
        out = {arm: model(row['x']).float() for arm, row in rows.items()}
        ix = torch.arange(len(rows[(0, 'round')]['x']), device=device)
        z = {arm: [out[arm][ix, rows[arm]['site'][:, k]] for k in (0, 1)]
             for arm in panel}
        doc_exact = torch.ones(len(ix), dtype=torch.bool, device=device)
        doc_two_way = torch.ones(len(ix), dtype=torch.bool, device=device)
        for arm in panel:
            shift, style = arm
            for k, other in ((0, (shift, 'square' if style == 'round'
                                   else 'round')),
                             (1, (1-shift, style))):
                target = rows[arm]['target'][:, k]
                alternative = rows[other]['target'][:, k]
                doc_exact &= z[arm][k].argmax(-1) == target
                doc_two_way &= z[arm][k][ix, target] > \
                               z[arm][k][ix, alternative]
        all_exact += doc_exact.sum().item()
        all_two_way += doc_two_way.sum().item()
        for kind, k, edges in (
                ('bracket', 0,
                 (((0, 'round'), (0, 'square')),
                  ((1, 'round'), (1, 'square')))),
                ('list', 1,
                 (((0, 'round'), (1, 'round')),
                  ((0, 'square'), (1, 'square'))))):
            for left, right in edges:
                a, b = rows[left]['target'][:, k], rows[right]['target'][:, k]
                assert (a != b).all()
                la, lb = z[left][k], z[right][k]
                exact = ((la.argmax(-1) == a) & (lb.argmax(-1) == b))
                pair = ((la[ix, a] > la[ix, b]) &
                        (lb[ix, b] > lb[ix, a]))
                direction = ((lb[ix, b]-lb[ix, a]) >
                             (la[ix, b]-la[ix, a]))
                totals[kind]['paired_top1'] += exact.sum().item()
                totals[kind]['paired_two_way'] += pair.sum().item()
                totals[kind]['edit_direction'] += direction.sum().item()
    return {'n': n, 'all_arms_both_rules_top1': all_exact/n,
            'all_arms_both_rules_two_way': all_two_way/n,
            **{kind: {k: v/(2*n) for k, v in values.items()}
               for kind, values in totals.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--list-cache', type=Path, required=True)
    p.add_argument('--bracket-cache', type=Path, required=True)
    p.add_argument('--initial-checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--list-site-weight', type=float, default=100.)
    p.add_argument('--bracket-site-weight', type=float, default=50.)
    p.add_argument('--list-ordinals', default='3',
                   help='Comma-separated item positions given site weights')
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--seed', type=int, default=71501)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(4)
    data = torch.load(a.list_cache, map_location='cpu', weights_only=True)
    brackets = torch.load(a.bracket_cache, map_location='cpu',
                          weights_only=True)
    assert data['n_ctx'] == brackets['n_ctx'] == 256
    assert data['vocab_size'] == brackets['vocab_size'] == 50257
    saved = torch.load(a.initial_checkpoint, map_location='cpu',
                       weights_only=True)
    assert (saved['depth'], saved['d_model'], saved['heads'],
            saved['n_ctx'], saved['norm']) == (2, 64, 4, 256, 'none')
    model = _model(50257, 256, 2, 64, 4, a.device, 'none')
    model.load_state_dict(saved['state_dict'])
    torch.manual_seed(a.seed)
    ordinals = tuple(int(value) for value in a.list_ordinals.split(','))
    available = data.get('list_by_ordinal', {})
    assert all(ordinal in available.get('train', {3: data['list']['train']})
               for ordinal in ordinals)

    def ordinal_panel(split, ordinal):
        if ordinal == 3 and 'list_by_ordinal' not in data:
            return data['list'][split]
        return data['list_by_ordinal'][split][ordinal]

    def scores():
        result = {
            'list_val': eval_list(model, data['list']['val'], a.device),
            'list_test': eval_list(model, data['list']['test'], a.device),
            'bracket_val': eval_bracket(model, brackets['val'], a.device,
                                      batch_size=4),
            'bracket_test': eval_bracket(model, brackets['test'], a.device,
                                       batch_size=4),
            'joint_val': eval_joint(model, data['joint']['val'], a.device),
            'joint_test': eval_joint(model, data['joint']['test'], a.device)}
        if 'list_by_ordinal' in data:
            for split in ('val', 'test'):
                result[f'list_{split}_by_ordinal'] = {
                    str(ordinal): eval_list(model,
                        ordinal_panel(split, ordinal), a.device)
                    for ordinal in sorted(data['list_by_ordinal'][split])}
        return result

    initial = scores()
    list_train = {k: torch.cat([
                    ordinal_panel('train', ordinal)[arm][k]
                    for ordinal in ordinals for arm in (0, 1)])
                  for k in ('x', 'y', 'site')}
    bracket_train = brackets['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    losses = []
    for step in range(a.steps):
        model.train()
        n_list = a.batch_size//2
        n_bracket = a.batch_size-n_list
        list_ix = torch.randint(len(list_train['x']), (n_list,))
        bracket_ix = torch.randint(len(bracket_train['x']), (n_bracket,))
        x = torch.cat([list_train['x'][list_ix],
                       bracket_train['x'][bracket_ix]]).to(a.device)
        y = torch.cat([list_train['y'][list_ix],
                       bracket_train['y'][bracket_ix]]).to(a.device)
        sites = torch.cat([list_train['site'][list_ix, 0],
                           bracket_train['site_position'][bracket_ix]]
                          ).to(a.device)
        logits = model(x).float()
        per = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                              ignore_index=-100, reduction='none').view_as(y)
        weights = (y != -100).float()
        ix = torch.arange(len(x), device=a.device)
        weights[ix, sites] = torch.tensor(
            [a.list_site_weight]*n_list +
            [a.bracket_site_weight]*n_bracket, device=a.device)
        loss = (per*weights).sum()/weights.sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1) % 250 == 0:
            print('step', step+1, 'loss', round(losses[-1], 3), flush=True)
            snapshot = a.checkpoint.with_suffix('.resume.pt')
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in
                                       model.state_dict().items()},
                        'optimizer': optimizer.state_dict(),
                        'step': step+1}, snapshot)
    final = scores()
    report = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()},
              'train': {'list_source_documents': len(data['list']['train'][0]['x']),
                        'list_arms': len(list_train['x']),
                        'list_weighted_ordinals': list(ordinals),
                        'bracket_source_documents': len(bracket_train['x']),
                        'joint_edit_documents': 0,
                        'loss_last50': sum(losses[-50:])/50},
              'initial': initial, 'final': final,
              'scope': 'Continue frozen-corpus delimiter model on a 50/50 '
                       'batch mixture of independent list-only and '
                       'bracket-only examples. No joint edit documents '
                       'in training. Report list and bracket behavior '
                       'separately before weight analysis.'}
    a.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in
                               model.state_dict().items()},
                'depth': 2, 'd_model': 64, 'heads': 4, 'n_ctx': 256,
                'vocab_size': 50257, 'norm': 'none',
                'list_cache': str(a.list_cache),
                'bracket_cache': str(a.bracket_cache)}, a.checkpoint)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print('saved', a.output, flush=True)


if __name__ == '__main__':
    main()
