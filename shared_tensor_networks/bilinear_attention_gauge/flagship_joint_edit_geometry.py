"""Joint-diagonal intervention geometry across 36 native 546M heads.

For each head, fix a train-PCA rank-four response subspace. Within it,
compare PCA coordinates with orthogonal joint diagonalization of ten
input-edit-type covariance operators. Input edit types define the prior;
no answer labels or downstream gradients enter fitting.
"""

import json
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from flagship_transfer import ART


LAYERS=(9,11,13,15)
HEADS=9
RANK=4
STARTS=8
PAIRS=tuple(combinations(range(RANK),2))


def rotation(angles):
    result=np.eye(RANK)
    for angle,(p,q) in zip(angles,PAIRS):
        c,s=np.cos(angle),np.sin(angle)
        turn=np.eye(RANK)
        turn[p,p]=turn[q,q]=c
        turn[p,q]=-s
        turn[q,p]=s
        result=turn@result
    return result


def covariances(z,types,names):
    return np.stack([z[types==name].T@z[types==name]/np.sum(types==name)
                     for name in names])


def normalize(cov,mode):
    if mode=='equal_type':
        return cov/np.trace(cov,axis1=1,axis2=2)[:,None,None].clip(1e-30)
    if mode=='raw_energy':
        return cov/np.mean(np.trace(cov,axis1=1,axis2=2)).clip(1e-30)
    raise ValueError(mode)


def cost(cov):
    off=cov.copy()
    index=np.arange(RANK)
    off[:,index,index]=0
    return float(np.sum(off*off))


def fit(cov,seed):
    rng=np.random.default_rng(seed)
    def objective(angles):
        matrix=rotation(angles)
        return cost(np.einsum('ia,tij,jb->tab',matrix,cov,matrix))
    starts=[np.zeros(6)]+[rng.uniform(-np.pi,np.pi,6) for _ in range(STARTS-1)]
    results=[minimize(objective,start,method='BFGS',
                      options=dict(maxiter=300,gtol=1e-6)) for start in starts]
    selected=min(results,key=lambda result:result.fun)
    return rotation(selected.x),dict(start_costs=[float(r.fun) for r in results],
        iterations=int(selected.nit),optimizer_success=bool(selected.success))


def check():
    rng=np.random.default_rng(20261116)
    q,_=np.linalg.qr(rng.normal(size=(RANK,RANK)))
    diagonal=rng.uniform(.2,2,size=(10,RANK))
    cov=np.einsum('ia,ta,ja->tij',q,diagonal,q)
    fitted,_=fit(cov,20261117)
    error=cost(np.einsum('ia,tij,jb->tab',fitted,cov,fitted))
    assert error<1e-8,error
    assert axis_match(fitted,q)>.999999


def axis_match(a,b):
    overlap=np.abs(a.T@b)
    row,column=linear_sum_assignment(-overlap)
    return float(overlap[row,column].mean())


def concentration(cov,rotation):
    diagonal=np.diagonal(np.einsum('ia,tij,jb->tab',rotation,cov,rotation),
                         axis1=1,axis2=2)
    shares=diagonal/np.sum(diagonal,axis=0,keepdims=True).clip(1e-30)
    return float(np.mean(np.max(shares,axis=0)))


def run():
    check()
    data=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    train_types=np.array([r['type'] for r in meta['panels']['train']])
    val_types=np.array([r['type'] for r in meta['panels']['validation']])
    names=sorted(set(train_types))
    assert len(names)==10 and set(names)==set(val_types)
    rows=[];factors={}
    for layer in LAYERS:
        x=data[f'layer{layer}__train'].astype(np.float64)
        y=data[f'layer{layer}__validation'].astype(np.float64)
        for head in range(HEADS):
            train=x[:,head,:]
            val=y[:,head,:]
            _,_,vh=np.linalg.svd(train,full_matrices=False)
            basis=vh[:RANK].T
            z=train@basis;v=val@basis
            full_train=float(np.sum(train*train))
            full_val=float(np.sum(val*val))
            response_coverage=dict(train=float(np.sum(z*z)/full_train),
                                   validation=float(np.sum(v*v)/full_val))
            cov_train=covariances(z,train_types,names)
            cov_val=covariances(v,val_types,names)
            for mode_index,mode in enumerate(('equal_type','raw_energy')):
                ct=normalize(cov_train,mode)
                cv=normalize(cov_val,mode)
                fitted,diagnostic=fit(ct,20261113+layer*1000+head*10+mode_index)
                independent,_=fit(cv,20261114+layer*1000+head*10+mode_index)
                test_cov=np.einsum('ia,tij,jb->tab',fitted,cv,fitted)
                identity_cost=cost(cv)
                rng=np.random.default_rng(20261115+layer*1000+head*10+mode_index)
                random_cost=[]
                for _ in range(32):
                    q,_=np.linalg.qr(rng.standard_normal((RANK,RANK)))
                    random_cost.append(cost(np.einsum('ia,tij,jb->tab',q,cv,q)))
                selected_concentration=concentration(cv,fitted)
                record=dict(layer=layer,head=head,mode=mode,
                    rank4_response_coverage=response_coverage,
                    train_cost=float(diagnostic['start_costs'][np.argmin(
                        diagnostic['start_costs'])]),
                    validation_cost=cost(test_cov),
                    validation_pca_cost=identity_cost,
                    validation_random_cost_mean=float(np.mean(random_cost)),
                    validation_offdiag_ratio_vs_pca=float(cost(test_cov)/max(
                        identity_cost,1e-30)),
                    validation_offdiag_ratio_vs_random=float(cost(test_cov)/max(
                        np.mean(random_cost),1e-30)),
                    train_validation_axis_alignment=axis_match(fitted,independent),
                    validation_type_concentration=selected_concentration,
                    validation_pca_type_concentration=concentration(cv,np.eye(RANK)),
                    fit_iterations=diagnostic['iterations'],
                    fit_optimizer_success=diagnostic['optimizer_success'],
                    fit_start_costs=diagnostic['start_costs'])
                rows.append(record)
                factors[f'l{layer}_h{head}_{mode}_rank4_basis']=basis
                factors[f'l{layer}_h{head}_{mode}_rotation']=fitted
            print('layer',layer,'head',head,'complete',flush=True)
    assert len(rows)==72
    result=dict(description='36-head, two-panel edit-operator joint diagonalization; fit uses input-edit type membership but no answer labels or downstream gradients. Rank-four subspace is selected by training-response PCA.',
        layers=LAYERS,heads=HEADS,n_types=len(names),types=names,
        n_train=len(train_types),n_validation=len(val_types),
        methods=['equal_type','raw_energy'],starts=STARTS,
        rows=rows)
    (ART/'all_head_joint_edit_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ART/'all_head_joint_edit_geometry_factors.npz',**factors)
    for mode in ('equal_type','raw_energy'):
        selected=[r for r in rows if r['mode']==mode]
        print(mode,'coverage',np.mean([r['rank4_response_coverage']['validation']
                                      for r in selected]),
              'ratio_pca',np.mean([r['validation_offdiag_ratio_vs_pca']
                                   for r in selected]),
              'axis_alignment',np.mean([r['train_validation_axis_alignment']
                                        for r in selected]))


if __name__=='__main__':run()
