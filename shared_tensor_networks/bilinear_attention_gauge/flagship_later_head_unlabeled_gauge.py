"""Unlabeled exact rotations of fixed later-head rank-4 subspaces."""

import json

import numpy as np
import scipy.optimize as so
import torch

from flagship_later_head_exact_gauge import effective_axes, rotation, normalized_rows
from flagship_later_head_gauge_capture import TARGETS
from flagship_transfer import ART

SEED=20261102
STARTS=8


def sparse_axis(z,r):
    y=z@r
    center=y-y.mean(axis=0)
    rms=np.sqrt(np.mean(center*center,axis=0)).clip(1e-12)
    ratio=np.mean(np.abs(center),axis=0)/rms
    return int(np.argmin(ratio)),ratio.tolist()


def unlabeled_objective(theta,kind,z):
    t,r=rotation(theta)
    value=z@r
    if kind=='activation_l1':
        cost=torch.sqrt(value.square()+1e-4).sum(-1).mean()
    elif kind=='varimax':
        cost=-(value**4).sum(-1).mean()
    else:raise ValueError(kind)
    grad=torch.autograd.grad(cost,t)[0]
    return float(cost.detach()),grad.detach().numpy()


def fit(z,kind,rng):
    normalized=torch.tensor(normalized_rows(z),dtype=torch.float64)
    runs=[]
    for i in range(STARTS):
        initial=np.zeros(6) if i==0 else rng.normal(0,.7,size=6)
        result=so.minimize(unlabeled_objective,initial,args=(kind,normalized),
                           jac=True,method='BFGS',
                           options=dict(maxiter=400,gtol=1e-7))
        _,r=rotation(result.x)
        runs.append((float(result.fun),r.detach().numpy(),bool(result.success)))
    return sorted(runs,key=lambda v:v[0])


def pca(z):
    _,_,v=np.linalg.svd(z,full_matrices=False)
    r=v.T
    if np.linalg.det(r)<0:r[:,-1]*=-1
    return r


def run():
    rng=np.random.default_rng(SEED)
    activations=np.load(ART/'later_head_unlabeled_activations.npz')
    edits=np.load(ART/'later_head_gauge_capture.npz')
    edit_meta=json.loads((ART/'later_head_gauge_capture.json').read_text())
    existing=np.load(ART/'later_head_exact_gauge_rotations.npz')
    rotations={};records=[]
    for target in TARGETS:
        d=edits[f'{target}__holdout__d'].astype(np.float64)
        g=edits[f'{target}__holdout__g'].astype(np.float64)
        fam=np.array([r['family'] for r in edit_meta['rows'][f'{target}__holdout']])
        own=fam==target
        for length in (5,32):
            z=activations[f'{target}__fineweb_train__l{length}'].astype(np.float64)
            for method in ('pca_raw','pca_centered','activation_l1_raw',
                           'activation_l1_centered','varimax_centered'):
                train=z-z.mean(axis=0) if method.endswith('centered') else z
                if method.startswith('pca'):
                    r=pca(train);starts=None
                else:
                    kind='varimax' if method.startswith('varimax') else 'activation_l1'
                    starts=fit(train,kind,rng)
                    r=starts[0][1]
                axis,ratios=sparse_axis(z,r)
                key=f'{target}__l{length}__{method}'
                rotations[key]=r
                responses=(d@r)*(g@r)
                full=np.sum(d*g,axis=-1)
                selected=responses[:,axis]
                row=dict(target=target,length=length,method=method,
                         selected_axis=axis,train_sparsity_ratios=ratios,
                         holdout_target_effective_axes=float(np.mean(effective_axes(responses[own]))),
                         holdout_target_axis_linear_relative_mae=float(
                             np.mean(np.abs(selected[own]-full[own]))/np.mean(np.abs(full[own]))),
                         holdout_target_oracle_best_axis_relative_mae=float(min(
                             np.mean(np.abs(responses[own,j]-full[own]))/np.mean(np.abs(full[own]))
                             for j in range(4))),
                         train_cost=None if starts is None else starts[0][0],
                         optimization_success_count=None if starts is None else sum(x[2] for x in starts))
                for panel in ('fineweb_test','wikitext103','tinystories'):
                    y=activations[f'{target}__{panel}__l{length}'].astype(np.float64)
                    row[f'{panel}_activation_effective_axes']=float(np.mean(effective_axes(y@r)))
                    rare,_=sparse_axis(y,r)
                    row[f'{panel}_same_rare_axis']=rare==axis
                records.append(row)
                print(target,length,method,'axis',axis,'target linear rel',
                      round(row['holdout_target_axis_linear_relative_mae'],3),flush=True)
            # Preexisting SVD, weight L1, and random rotations use the same
            # unlabeled selection rule, so target labels never choose an axis.
            controls={'svd':np.eye(4),'weight_l1':existing[f'{target}__weight__all']}
            controls.update({f'random_{i}':existing[f'{target}__random_{i}'] for i in range(64)})
            for method,r in controls.items():
                axis,ratios=sparse_axis(z,r)
                rotations[f'{target}__l{length}__{method}']=r
                response=(d@r)*(g@r);full=np.sum(d*g,axis=-1)
                records.append(dict(target=target,length=length,method=method,
                    selected_axis=axis,train_sparsity_ratios=ratios,
                    holdout_target_effective_axes=float(np.mean(effective_axes(response[own]))),
                    holdout_target_axis_linear_relative_mae=float(
                        np.mean(np.abs(response[own,axis]-full[own]))/np.mean(np.abs(full[own])))))
    result=dict(description='Unlabeled rotations and rare-axis selection from natural head activations; target-task edit gradients used only for evaluation',
                seed=SEED,starts=STARTS,
                axis_rule='smallest centered mean absolute / RMS on FineWeb training activations',
                records=records)
    (ART/'later_head_unlabeled_gauge.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(ART/'later_head_unlabeled_gauge_rotations.npz',**rotations)
    print('Saved unlabeled gauge',flush=True)


if __name__=='__main__':run()
