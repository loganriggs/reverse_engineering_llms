"""Exact rotations of fixed rank-4 later-head cuts under three legibility priors."""

import json

import numpy as np
import scipy.optimize as so
from scipy.optimize import linear_sum_assignment
import torch

from flagship_later_head_gauge_capture import TARGETS
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

SEED=20261101
STARTS=8
RANDOM_CONTROLS=64


def effective_axes(x):
    x=np.abs(x)
    return (x.sum(axis=-1)**2/(x*x).sum(axis=-1).clip(1e-30)).tolist()


def normalized_rows(x):
    return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)


SKEW=[]
for i in range(4):
    for j in range(i+1,4):
        k=np.zeros((4,4));k[i,j]=1.;k[j,i]=-1.
        SKEW.append(k)
SKEW=torch.tensor(np.stack(SKEW),dtype=torch.float64)


def rotation(theta):
    t=torch.tensor(theta,dtype=torch.float64,requires_grad=True)
    matrix=torch.matrix_exp(torch.einsum('a,aij->ij',t,SKEW))
    return t,matrix


def objective(theta,kind,w,d,g):
    t,r=rotation(theta)
    if kind=='weight':
        value=w@r
        value=value/torch.sqrt((w*w).mean()).clamp_min(1e-12)
        cost=torch.sqrt(value.square()+1e-4).mean()
    elif kind=='edit':
        value=d@r
        cost=torch.sqrt(value.square()+1e-4).sum(-1).mean()
    elif kind=='effect':
        value=(d@r)*(g@r)
        # For fixed total signed linear response, smooth |c| is convex and
        # can prefer spreading equal-sign contributions. p=1/2 is a genuine
        # concentration prior; the multistart search handles nonconvexity.
        cost=torch.pow(value.square()+1e-8,.25).sum(-1).mean()
    else:raise ValueError(kind)
    grad=torch.autograd.grad(cost,t)[0]
    return float(cost.detach()),grad.detach().numpy()


def fit(kind,w,d,g,rng):
    starts=[]
    for start in range(STARTS):
        initial=np.zeros(6) if start==0 else rng.normal(0,.7,size=6)
        result=so.minimize(objective,initial,args=(kind,w,d,g),jac=True,
                           method='BFGS',options=dict(maxiter=400,gtol=1e-7))
        _,r=rotation(result.x)
        starts.append(dict(cost=float(result.fun),rotation=r.detach().numpy(),
                           success=bool(result.success),iterations=int(result.nit)))
    starts.sort(key=lambda s:s['cost'])
    return starts


def alignment(a,b):
    overlap=np.abs(a.T@b)
    i,j=linear_sum_assignment(-overlap)
    return float(overlap[i,j].mean())


def panel_metrics(d,g,r):
    coords=d@r
    effects=coords*(g@r)
    return dict(edit_effective_axes_mean=float(np.mean(effective_axes(coords))),
                effect_effective_axes_mean=float(np.mean(effective_axes(effects))),
                effect_cancel_ratio=float(np.mean(np.abs(effects).sum(-1)/np.abs(effects.sum(-1)).clip(1e-12))))


def run():
    unit=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
    zero=np.zeros(6)
    split=np.zeros(6);split[0]=np.pi/4
    assert objective(zero,'effect',None,unit,unit)[0]<objective(split,'effect',None,unit,unit)[0]
    rng=np.random.default_rng(SEED)
    data=np.load(ART/'later_head_gauge_capture.npz')
    meta=json.loads((ART/'later_head_gauge_capture.json').read_text())
    bases=np.load(ART/'later_head_rank_pilot_bases.npz')
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    records=[];rotations={}
    for target,(layer,head) in TARGETS.items():
        b=bases[f'{target}__weight'][:,:4]
        w=state[f'transformer.h.{layer}.attn.c_proj.weight'].numpy()[:,head*128:(head+1)*128]@b
        wt=torch.tensor(w,dtype=torch.float64)
        train_d=data[f'{target}__train__d'].astype(np.float64)
        train_g=data[f'{target}__train__g'].astype(np.float64)
        hold_d=data[f'{target}__holdout__d'].astype(np.float64)
        hold_g=data[f'{target}__holdout__g'].astype(np.float64)
        train_family=np.array([x['family'] for x in meta['rows'][f'{target}__train']])
        hold_family=np.array([x['family'] for x in meta['rows'][f'{target}__holdout']])
        base={'svd':np.eye(4)}
        for kind in ('weight','edit','effect'):
            fit_panels=('all',) if kind=='weight' else ('all','exclude_target','target_only')
            for fit_panel in fit_panels:
                mask=(np.ones(len(train_d),dtype=bool) if fit_panel=='all' else
                      train_family!=target if fit_panel=='exclude_target' else
                      train_family==target)
                d=torch.tensor(normalized_rows(train_d[mask]),dtype=torch.float64)
                g=torch.tensor(normalized_rows(train_g[mask]),dtype=torch.float64)
                starts=fit(kind,wt,d,g,rng)
                key=f'{kind}__{fit_panel}'
                base[key]=starts[0]['rotation']
                rotations[f'{target}__{key}']=starts[0]['rotation']
                records.append(dict(target=target,variant=key,fit_examples=int(mask.sum()),
                    training_cost=starts[0]['cost'],
                    worst_start_cost=starts[-1]['cost'],
                    start_alignment_mean=float(np.mean([alignment(starts[0]['rotation'],s['rotation']) for s in starts[1:]])),
                    optimization_success_count=sum(s['success'] for s in starts)))
        random_rotations=[np.linalg.qr(rng.standard_normal((4,4)))[0] for _ in range(RANDOM_CONTROLS)]
        for i,r in enumerate(random_rotations):
            rotations[f'{target}__random_{i}']=r
        for key,r in {**base,**{f'random_{i}':z for i,z in enumerate(random_rotations)}}.items():
            record=next((x for x in records if x['target']==target and x['variant']==key),
                        dict(target=target,variant=key))
            record['train_all']=panel_metrics(train_d,train_g,r)
            record['holdout_all']=panel_metrics(hold_d,hold_g,r)
            record['train_target']=panel_metrics(train_d[train_family==target],train_g[train_family==target],r)
            record['holdout_target']=panel_metrics(hold_d[hold_family==target],hold_g[hold_family==target],r)
            if key in base and key!='svd':
                # Axis selection uses only train target labels when permitted;
                # the exclude-target variants do not receive a target axis.
                if key.endswith('__all') or key.endswith('__target_only'):
                    train_effect=(train_d[train_family==target]@r)*(train_g[train_family==target]@r)
                    axis=int(np.argmax(np.mean(np.abs(train_effect),axis=0)))
                    hold_effect=(hold_d[hold_family==target]@r)*(hold_g[hold_family==target]@r)
                    record['selected_axis']=axis
                    record['selected_axis_holdout_effect_share']=float(
                        np.mean(np.abs(hold_effect[:,axis])/np.abs(hold_effect).sum(-1).clip(1e-12)))
            if key.startswith('random_'):
                continue
            print(target,key,'target holdout effect axes',
                  round(record['holdout_target']['effect_effective_axes_mean'],3),flush=True)
        # Append SVD and random records not already added by the fit loop.
        records.append(dict(target=target,variant='svd',
                            train_all=panel_metrics(train_d,train_g,np.eye(4)),
                            holdout_all=panel_metrics(hold_d,hold_g,np.eye(4)),
                            train_target=panel_metrics(train_d[train_family==target],train_g[train_family==target],np.eye(4)),
                            holdout_target=panel_metrics(hold_d[hold_family==target],hold_g[hold_family==target],np.eye(4))))
        for i,r in enumerate(random_rotations):
            records.append(dict(target=target,variant=f'random_{i}',
                                train_all=panel_metrics(train_d,train_g,r),
                                holdout_all=panel_metrics(hold_d,hold_g,r),
                                train_target=panel_metrics(train_d[train_family==target],train_g[train_family==target],r),
                                holdout_target=panel_metrics(hold_d[hold_family==target],hold_g[hold_family==target],r)))
    result=dict(description='Exact SO(4) rotations of fixed weight-rank-4 head cuts; optimize native output weight L1, edit-vector L1, or local edit-effect L1',
                seed=SEED,starts=STARTS,random_controls=RANDOM_CONTROLS,
                note='Exclude-target variants fit only other task families. Local effect is (gradient dot axis)*(edit delta dot axis); finite validation remains required.',
                records=records)
    (ART/'later_head_exact_gauge.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez(ART/'later_head_exact_gauge_rotations.npz',**rotations)
    print('Saved exact gauge',flush=True)


if __name__=='__main__':run()
