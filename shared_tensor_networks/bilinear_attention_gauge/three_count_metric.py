"""Uniform versus frequency-weighted rank budget on three-bit attention."""

from __future__ import annotations

import json

import numpy as np
import scipy.optimize as so
import torch

from analyze_three_count_language import head_tables,flatten,unflatten,logits,score
from dual_joint_function_canonical import HERE
from train_three_count_language import ThreeCountAttention,PAD,examples


def weighted_fit(matrix,weight,rank):
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    left=u[:,:rank]*np.sqrt(s[:rank])
    right=np.sqrt(s[:rank,None])*v[:rank]
    start=np.r_[left.ravel(),right.ravel()]
    root=np.sqrt(weight)
    def residual(z):
        l=z[:left.size].reshape(left.shape)
        r=z[left.size:].reshape(rank,matrix.shape[1])
        return ((l@r-matrix)*root).ravel()
    optimized=so.least_squares(residual,start,max_nfev=10000,
        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    z=optimized.x
    l=z[:left.size].reshape(left.shape)
    r=z[left.size:].reshape(rank,matrix.shape[1])
    return l@r,dict(nfev=optimized.nfev,
        weighted_error=float(np.sum(residual(z)**2)),
        optimality=float(optimized.optimality))


def pair_weights(tokens):
    q=tokens[:,-1]-8
    keys=tokens[:,:-1]
    valid=keys!=PAD
    counts=np.zeros((2,8),dtype=float)
    np.add.at(counts,(np.broadcast_to(q[:,None],keys.shape)[valid],keys[valid]),1.)
    assert np.all(counts>0)
    repeated=np.repeat(counts,3,axis=0)
    return repeated/repeated.sum()


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=ThreeCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"three_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        tables=head_tables(model)
        matrix=flatten(sum(tables))
        bias=model.read.bias.detach().numpy()
        train_t,_,_=examples(100000,10000+seed,True)
        empirical=pair_weights(train_t.numpy())
        uniform=np.ones_like(empirical)/empirical.size
        metrics={"uniform":uniform,"empirical":empirical,
                 "half_uniform_empirical":.5*(uniform+empirical)}
        test_t,target_t,attrs=examples(30000,30000+seed,False)
        test=test_t.numpy();target=target_t.numpy().astype(float)
        reference=logits(tables,test,bias)
        for rank in (2,3):
            for name,weight in metrics.items():
                reduced,fit=weighted_fit(matrix,weight,rank)
                scored=score([unflatten(reduced)],test,target,attrs,bias,reference)
                rows.append(dict(seed=seed,rank=rank,measure=name,fit=fit,
                    relative_uniform_table_mse=float(np.sum((reduced-matrix)**2)/
                                                     np.sum(matrix**2)),
                    scores=scored))
    output=dict(n_seeds=10,n_frequency_samples=100000,n_test=30000,
        fit="joint 6x8 token function weighted low-rank least squares, SVD initialization; no task labels",
        measures="uniform pair cells, training-frequency pair cells, half mixture",
        rows=rows)
    (HERE/"three_count_metric.json").write_text(json.dumps(output,indent=2)+"\n")
    for rank in (2,3):
        for name in metrics:
            selected=[r for r in rows if r["rank"]==rank and r["measure"]==name]
            print(rank,name,*[round(float(np.mean([r["scores"][part]
                ["joint_accuracy"] for r in selected])),4)
                for part in ("all","rare_heldout")],flush=True)
    return output


if __name__=="__main__":
    run()
