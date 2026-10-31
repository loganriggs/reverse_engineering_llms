"""Uniform versus training-frequency metric for the joint toy token function."""

from __future__ import annotations

import json

import numpy as np
import scipy.optimize as so
import torch

from analyze_dual_output_language import factors,score
from analyze_toy_language import pair_measure,pair_table
from dual_joint_function_canonical import HERE,flatten,unflatten,truncate
from train_dual_output_language import DualOutputAttention,batch,exhaustive


def weighted_fit(matrix,weight,rank):
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    left=u[:,:rank]*np.sqrt(s[:rank])
    right=np.sqrt(s[:rank,None])*v[:rank]
    shape=left.shape
    initial=np.concatenate((left.ravel(),right.ravel()))
    root=np.sqrt(weight)
    def residual(parameters):
        l=parameters[:left.size].reshape(shape)
        r=parameters[left.size:].reshape(rank,6)
        return ((l@r-matrix)*root).ravel()
    result=so.least_squares(residual,initial,max_nfev=3000,ftol=1e-12,
                            xtol=1e-12,gtol=1e-12)
    l=result.x[:left.size].reshape(shape)
    r=result.x[left.size:].reshape(rank,6)
    return l@r,dict(nfev=result.nfev,weighted_error=float(np.sum(residual(result.x)**2)),
                    optimality=float(result.optimality))


def run():
    torch.set_num_threads(1)
    seq_t,target_t,attrs=exhaustive()
    seq=seq_t.numpy();target=target_t.numpy().astype(float)
    rows=[]
    for seed in range(10):
        model=DualOutputAttention().double()
        model.load_state_dict(torch.load(HERE/f"dual_output_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        heads=[factors(model,h) for h in range(2)]
        total=sum(pair_table(h,np.eye(h["rank"])) for h in heads)
        matrix=flatten(total)
        bias=model.read.bias.detach().numpy()
        fit_seq,_=batch(100000,torch.Generator().manual_seed(100000+seed))
        pair=pair_measure(fit_seq.numpy())
        assert np.all(pair>0)
        empirical=np.repeat(pair,2,axis=0)
        empirical=empirical/np.sum(empirical)
        uniform=np.ones_like(empirical)/empirical.size
        measures={"uniform":uniform,"empirical":empirical,
                  "half_uniform_empirical":.5*(uniform+empirical)}
        for rank in (1,2):
            for name,weight in measures.items():
                approx,fit=weighted_fit(matrix,weight,rank)
                all_score=score([unflatten(approx)],seq,target,bias,attrs)[0]
                rows.append(dict(seed=seed,rank=rank,measure=name,
                    fit=fit,relative_uniform_table_mse=float(
                        np.sum((matrix-approx)**2)/np.sum(matrix**2)),
                    scores=all_score))
    result=dict(n_fit_sequences=100000,
                fit_uses="query-key frequencies only, no labels or held-out rare combination",
                metric="uniform, empirical training-frequency, or half mixture over 4x6 function table",
                rows=rows)
    (HERE/"dual_joint_metric.json").write_text(json.dumps(result,indent=2)+"\n")
    for rank in (1,2):
        for measure in measures:
            sub=[r for r in rows if r["rank"]==rank and r["measure"]==measure]
            print(rank,measure,
                  np.mean([r["scores"]["joint_accuracy"] for r in sub]),
                  np.mean([r["scores"]["rare_heldout"]["joint_accuracy"] for r in sub]),
                  np.mean([r["relative_uniform_table_mse"] for r in sub]),flush=True)
    return result


if __name__=="__main__":
    run()
