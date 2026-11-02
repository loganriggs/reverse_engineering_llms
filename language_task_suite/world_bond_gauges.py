"""Exact attention-bond gauges and sparse gates on competent document-world models.

Fit every basis on owner/color worlds from the train composition fold only.
The trained model saw all three tasks; *featurizer fitting* withholds place.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from document_world import causal_pair, nuisance_pair, sample_world
from train import DeepModel, SPECS, encode


def varimax(matrix, max_iter=200, tolerance=1e-9):
    """Orthomax gamma=1, following the existing gauge experiments."""
    n, k = matrix.shape
    r = np.eye(k)
    previous = -np.inf
    for _ in range(max_iter):
        z = matrix @ r
        gradient = matrix.T @ (z**3-z @ np.diag(np.sum(z*z, axis=0)/n))
        u, s, vh = np.linalg.svd(gradient, full_matrices=False)
        r = u @ vh
        value = float(np.sum(s))
        if value-previous < max(tolerance, abs(value)*tolerance):
            break
        previous = value
    return r


def model_from_checkpoint(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    k = ck['model_kwargs']
    model = DeepModel(len(ck['vocabulary']), k['d_model'], k['n_head'],
                      SPECS[f"block{ck['depth']}"], ck['n_ctx'],
                      norm=k['norm'], residual=k['residual'], attention=k['attention']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    attention = model.layers[-2]
    assert attention.__class__.__name__ == 'Attention'
    return model, attention, ck


@torch.no_grad()
def collect_bond(model, attention, examples, vocab, device, batch=128):
    rows = []
    for start in range(0, len(examples), batch):
        cases = examples[start:start+batch]
        x, last = encode(cases, vocab, device)[:2]
        def hook(_module, inputs):
            z = inputs[0]
            rows.append(z[torch.arange(len(cases), device=device), last].detach().cpu())
        h = attention.o.register_forward_pre_hook(hook)
        try:
            model(x)
        finally:
            h.remove()
    return torch.cat(rows).numpy().astype(np.float64)


def orthogonal_random(rng, d):
    q, r = np.linalg.qr(rng.normal(size=(d,d)))
    return q*np.sign(np.diag(r))[None,:]


def fit_bases(model, attention, vocab, device, n_fit, seed):
    examples = [ex for ex in sample_world(n_fit*3, seed, genres=(0,1))
                if ex.family in ('world_owner', 'world_color')][:n_fit]
    assert len(examples) == n_fit
    z = collect_bond(model, attention, examples, vocab, device)
    edits = [causal_pair(ex)[1] for ex in examples[:min(1000,len(examples))]]
    delta = collect_bond(model, attention, edits, vocab, device)-z[:len(edits)]
    weight = attention.o.weight.detach().cpu().numpy().astype(np.float64)
    _,_,vh = np.linalg.svd(z, full_matrices=False)
    activation_pca = vh.T
    _,_,vh = np.linalg.svd(weight, full_matrices=False)
    weight_pca = vh.T
    rng = np.random.default_rng(seed+500)
    bases = dict(physical=np.eye(z.shape[1]),
                 activation_pca=activation_pca,
                 weight_pca=weight_pca,
                 activation_varimax=activation_pca@varimax(z@activation_pca),
                 weight_varimax=weight_pca@varimax(weight@weight_pca),
                 edit_varimax=activation_pca@varimax(delta@activation_pca))
    for i in range(3):
        bases[f'random_{i}'] = orthogonal_random(rng, z.shape[1])
    for name, q in bases.items():
        assert np.max(np.abs(q.T@q-np.eye(len(q)))) < 1e-8, name
    return bases, dict(n_fit=len(examples), n_edits=len(edits),
                       fit_families=['world_owner','world_color'],
                       heldout_family='world_place')


def axis_signatures(model, attention, vocab, device, bases, seed, n=200):
    """Post-fit semantic readout; these task labels never select a basis."""
    examples = sample_world(n*5, seed, split='composition', genres=(0,1))
    groups = {family: [ex for ex in examples if ex.family == family][:n]
              for family in ('world_owner','world_color','world_place')}
    deltas = {}
    nuisance = {}
    for family, group in groups.items():
        base = collect_bond(model, attention, group, vocab, device)
        causal = collect_bond(model, attention, [causal_pair(ex)[1] for ex in group],
                              vocab, device)
        other = collect_bond(model, attention, [nuisance_pair(ex)[1] for ex in group],
                             vocab, device)
        deltas[family] = causal-base
        nuisance[family] = other-base
    output = {}
    for name,q in bases.items():
        energy = np.stack([np.mean((deltas[f]@q)**2,axis=0) for f in groups])
        nuisance_energy = np.stack([np.mean((nuisance[f]@q)**2,axis=0)
                                    for f in groups])
        balanced = energy/np.maximum(energy.sum(axis=1,keepdims=True),1e-30)
        purity = float(balanced.max(axis=0).sum()/balanced.sum())
        output[name] = dict(equal_family_energy_axis_purity=purity,
                            best_axis={f:int(np.argmax(balanced[i]/
                                np.maximum(np.delete(balanced,i,axis=0).mean(axis=0),1e-30)))
                                for i,f in enumerate(groups)},
                            causal_total_energy={f:float(energy[i].sum())
                                                 for i,f in enumerate(groups)},
                            nuisance_total_energy={f:float(nuisance_energy[i].sum())
                                                   for i,f in enumerate(groups)})
    return output


@torch.no_grad()
def evaluate(model, attention, examples, vocab, device, basis, keep, batch=128):
    q = torch.from_numpy(basis.astype(np.float32)).to(device)
    totals = defaultdict(lambda: dict(n=0, native_correct=0, gated_correct=0,
                                      agreement=0, logit_error=0., logit_energy=0.,
                                      kl_sum=0., bond_error=0., bond_energy=0.))
    for start in range(0, len(examples), batch):
        cases = examples[start:start+batch]
        x, last, target = encode(cases, vocab, device)[:3]
        index = torch.arange(len(cases), device=device)
        native = model(x)[index,last]
        bond_stats = []
        def hook(_module, inputs):
            z = inputs[0]
            old = z[index,last]
            rotated = old@q
            if keep < q.shape[0]:
                selected = rotated.abs().topk(keep,dim=1).indices
                mask = torch.zeros_like(rotated).scatter_(1,selected,1.)
                rotated = rotated*mask
            new = rotated@q.T
            bond_stats.append(((new-old).square().sum(dim=1), old.square().sum(dim=1)))
            z = z.clone()
            z[index,last] = new
            return (z,)
        h = attention.o.register_forward_pre_hook(hook)
        try:
            gated = model(x)[index,last]
        finally:
            h.remove()
        if keep == q.shape[0]:
            assert (gated-native).abs().max().item() < 2e-4
        kl = F.kl_div(F.log_softmax(gated,dim=-1), F.softmax(native,dim=-1),
                      reduction='none').sum(dim=-1)
        be, bn = bond_stats[0]
        for i, ex in enumerate(cases):
            row = totals[ex.family]
            row['n'] += 1
            row['native_correct'] += int(native[i].argmax() == target[i])
            row['gated_correct'] += int(gated[i].argmax() == target[i])
            row['agreement'] += int(gated[i].argmax() == native[i].argmax())
            row['logit_error'] += (gated[i]-native[i]).square().sum().item()
            row['logit_energy'] += native[i].square().sum().item()
            row['kl_sum'] += kl[i].item()
            row['bond_error'] += be[i].item()
            row['bond_energy'] += bn[i].item()
    return {family: dict(n=row['n'], native_accuracy=row['native_correct']/row['n'],
                         gated_accuracy=row['gated_correct']/row['n'],
                         prediction_agreement=row['agreement']/row['n'],
                         relative_logit_mse=row['logit_error']/max(row['logit_energy'],1e-30),
                         mean_kl=row['kl_sum']/row['n'],
                         relative_bond_mse=row['bond_error']/max(row['bond_energy'],1e-30))
            for family,row in totals.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--n-fit', type=int, default=3000)
    p.add_argument('--n-test', type=int, default=500)
    p.add_argument('--keeps', default='4,8,12,24,48')
    p.add_argument('--seed', type=int, default=700)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='language_task_suite/world_bond_gauges.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(a.device)
    model, attention, ck = model_from_checkpoint(a.checkpoint, device)
    bases, fit = fit_bases(model, attention, ck['vocabulary'], device, a.n_fit, a.seed)
    signatures = axis_signatures(model, attention, ck['vocabulary'], device,
                                 bases, a.seed+10, n=min(200,a.n_test))
    splits = dict(iid=sample_world(a.n_test, a.seed+1, genres=(0,1)),
                  composition=sample_world(a.n_test, a.seed+2, split='composition', genres=(0,1)),
                  held_genre=sample_world(a.n_test, a.seed+3, genres=(2,)))
    rows = {}
    for split, examples in splits.items():
        rows[split] = {}
        for name, basis in bases.items():
            rows[split][name] = {}
            for keep in [int(v) for v in a.keeps.split(',')]:
                rows[split][name][str(keep)] = evaluate(model,attention,examples,
                    ck['vocabulary'],device,basis,keep)
            print(split,name,flush=True)
    output = dict(config=vars(a), fit=fit, axis_signatures=signatures,
                  comparison='exact orthogonal full-rank gauges; top-|coordinate| gate on final attention bond at answer position',
                  results=rows)
    path=Path(a.output)
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(f'Saved {path}')


if __name__ == '__main__':
    main()
