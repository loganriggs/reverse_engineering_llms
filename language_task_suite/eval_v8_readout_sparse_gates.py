"""Check whether a faithful v8 readout basis has sparse reusable coordinates.

Uses a prefit orthonormal dictionary and held-out exact-support section edits.
The donor-delta TopK is an edit-aware ceiling, not an activation featurizer.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from eval_v3_two_path_holdout import prepare, readout
from eval_v8_rule_holdout_readout import generator, scope_panel


def topk(values, k):
    if k == values.shape[-1]:
        return values
    chosen = values.abs().topk(k, dim=-1).indices
    mask = torch.zeros_like(values).scatter_(-1, chosen, 1.)
    return values * mask


def summarize(model, data, delta, before_mask=None, after_mask=None):
    with torch.no_grad():
        sufficient = readout(model, data['r0'] + delta) - data['p0']
        remaining = readout(model, data['r0'] + data['dr'] - delta) - data['p0']
    rows = {}
    for kind in sorted(set(data['kinds'])):
        mask = torch.tensor([item == kind for item in data['kinds']])
        mean_l1 = lambda x: float(x[mask].abs().sum(-1).mean())
        row = dict(n=int(mask.sum()),
                   native_effect_l1=mean_l1(data['native']),
                   sufficient_to_teacher_l1=mean_l1(sufficient-data['native']),
                   sufficient_to_oracle_l1=mean_l1(sufficient-data['oracle']),
                   remaining_effect_l1=mean_l1(remaining),
                   residual_delta_l2=float((data['dr'][mask]-delta[mask])
                                           .square().sum(-1).mean().sqrt()))
        if before_mask is not None and after_mask is not None:
            both = (before_mask & after_mask).sum(-1)
            either = (before_mask | after_mask).sum(-1).clamp_min(1)
            row['state_gate_jaccard'] = float((both/either)[mask].mean())
        rows[kind] = row
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/home/loganriggs/Coding/toy_models_of_tensor_networks/attn_circuits')
    p.add_argument('--model-dir', default='language_task_suite/results/v8_scope_balanced_continue_seed101')
    p.add_argument('--basis', default='language_task_suite/results/v8_scope_keyfit_colorholdout_rank16_seed46001.pt')
    p.add_argument('--basis-name', default='fit_delta_pca')
    p.add_argument('--fit-seed', type=int, default=46001)
    p.add_argument('--test-seed', type=int, default=48001)
    p.add_argument('--fit-streams', type=int, default=500)
    p.add_argument('--test-streams', type=int, default=500)
    p.add_argument('--cap', type=int, default=120)
    p.add_argument('--ks', default='2,4,8,16')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, a.source)
    from model import BilinearGPT, Config
    folder = Path(a.model_dir)
    saved = json.loads((folder/'results.json').read_text())
    record = saved['records'][0]
    model = BilinearGPT(Config(**record['model_config']))
    model.load_state_dict(torch.load(folder/f"model_depth{record['depth']}.pt",
                                     map_location='cpu', weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vocab = generator(0, saved['generator_kwargs']).token2id
    basis = torch.load(a.basis, map_location='cpu', weights_only=True)[a.basis_name]
    assert basis[0] == 'combined'
    u = basis[1]
    assert torch.allclose(u.T @ u, torch.eye(u.shape[1]), atol=1e-5)
    ks = [int(value) for value in a.ks.split(',')]
    assert all(0 < k <= u.shape[1] for k in ks)
    raw_fit, _ = scope_panel(a.fit_seed, saved['generator_kwargs'],
                             a.fit_streams, a.cap)
    fit_pairs = [pair for pair in raw_fit if pair['kind'] == 'section_key']
    raw_test, attempted = scope_panel(a.test_seed, saved['generator_kwargs'],
                                      a.test_streams, a.cap)
    assert fit_pairs and raw_test
    assert set(pair['kind'] for pair in raw_test) == {
        'section_key', 'relevant_color', 'irrelevant_color'}
    fit = prepare(model, fit_pairs, vocab)
    test = prepare(model, raw_test, vocab)
    center = torch.cat((fit['r0'], fit['r0']+fit['dr'])).mean(0)
    fixed_order = (fit['dr'] @ u).square().mean(0).argsort(descending=True)
    coords_delta = test['dr'] @ u
    rows = {}
    for k in ks:
        fixed_mask = torch.zeros(u.shape[1])
        fixed_mask[fixed_order[:k]] = 1.
        fixed = ((coords_delta * fixed_mask) @ u.T)
        donor = topk(coords_delta, k) @ u.T
        before = (test['r0']-center) @ u
        after = (test['r0']+test['dr']-center) @ u
        before_sparse, after_sparse = topk(before,k), topk(after,k)
        state_delta = (after_sparse-before_sparse) @ u.T
        before_mask = before_sparse != 0
        after_mask = after_sparse != 0
        rows[str(k)] = dict(
            fixed_axis=summarize(model,test,fixed),
            edit_delta_topk=summarize(model,test,donor),
            state_topk=summarize(model,test,state_delta,before_mask,after_mask))
    full_delta = ((test['dr'] @ u) @ u.T)
    result = dict(config=vars(a), fit_counts=dict(Counter(x['kind'] for x in fit_pairs)),
                  test_counts=dict(Counter(x['kind'] for x in raw_test)),
                  attempted=dict(attempted),
                  max_endpoint_replay_error=max(fit['replay_error'],test['replay_error']),
                  full_dictionary=summarize(model,test,full_delta),rows=rows,
                  caveat='State TopK uses each endpoint separately and a fit-panel '
                         'center. Edit-delta TopK observes both endpoints and is '
                         'only a ceiling. This is a local final-readout test, not '
                         'whole-model decomposition or new-task discovery.')
    path=Path(a.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    for k, methods in rows.items():
        print(k,{name:round(scores['relevant_color']['sufficient_to_teacher_l1'],4)
                 for name,scores in methods.items()},flush=True)


if __name__ == '__main__':
    main()
