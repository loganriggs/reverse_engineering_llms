"""Direct rank-64 pair-objective control for sequence-response optimization."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_matrix_free import CORPUS,HD,SNAPSHOT,rows
from flagship_transfer import ART


def tensor_part(part):
    u,v,w,_,_=part
    target=np.sum(u*v,axis=1)
    energy=np.mean(w*target**2)
    return tuple(torch.from_numpy(np.asarray(z)).to(torch.float64)
                 for z in (u,v,w,target)),float(energy)


def error(part,energy,a,b,ix=None):
    u,v,w,target=part if ix is None else tuple(z[ix] for z in part)
    score=torch.sum((u@a)*(v@b),dim=1)
    return torch.mean(w*(score-target).square())/energy


def run(n_pairs=65536,steps=600,batch_size=2048,seed=41,lr=.001):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    train={};validation={};energy={};val_energy={}
    for law in ("uniform","corpus"):
        fit,val,_=rows(state,0,seed,n_pairs,n_pairs,128,
            distribution=law,corpus=corpus if law=="corpus" else None,
            val_docs=(15000,17500))
        train[law],energy[law]=tensor_part(fit)
        validation[law],val_energy[law]=tensor_part(val)
    initial=np.load(ART/"head0_l128_uniform_corpus_half.npz")["rank64"]
    left,s,right=np.linalg.svd(initial,full_matrices=False)
    a=torch.nn.Parameter(torch.from_numpy(left[:,:64]*np.sqrt(s[:64])).to(torch.float64))
    b=torch.nn.Parameter(torch.from_numpy(right[:64].T*np.sqrt(s[:64])).to(torch.float64))
    opt=torch.optim.Adam((a,b),lr=lr)
    rng=np.random.default_rng(seed+2000)
    def score(parts,energies):
        with torch.no_grad():
            return {law:float(error(parts[law],energies[law],a,b))
                    for law in ("uniform","corpus")}
    original=score(validation,val_energy)
    best=(original["uniform"]+original["corpus"])/2
    best_cut=initial.copy();best_step=0;trace=[]
    for step in range(steps):
        opt.zero_grad()
        components=[]
        for law in ("uniform","corpus"):
            ix=torch.from_numpy(rng.choice(n_pairs,batch_size,replace=False))
            components.append(error(train[law],energy[law],a,b,ix))
        loss=sum(components)/2
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((a,b),10.)
        opt.step()
        if step==0 or (step+1)%50==0 or step+1==steps:
            scores=score(validation,val_energy)
            objective=(scores["uniform"]+scores["corpus"])/2
            if objective<best:
                best=objective
                best_step=step+1
                best_cut=(a@b.T).detach().numpy().copy()
            record=dict(step=step+1,validation=scores,objective=objective)
            trace.append(record)
            print(record,flush=True)
    name="head0_l128_pair_refit_rank64"
    np.savez_compressed(ART/f"{name}.npz",cut=best_cut)
    result=dict(name=name,rank=64,head=0,seed=seed,n_pairs_per_law=n_pairs,
        steps=steps,batch_size_per_law=batch_size,learning_rate=lr,
        fit_corpus_documents=[0,15000],selection_corpus_documents=[15000,17500],
        initial_validation=original,best_step=best_step,best_objective=best,
        trace=trace,scope="QK1 pair-response objective; no sequence aggregation or CE")
    (ART/f"{name}.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-pairs",type=int,default=65536)
    parser.add_argument("--steps",type=int,default=600)
    parser.add_argument("--batch-size",type=int,default=2048)
    parser.add_argument("--seed",type=int,default=41)
    parser.add_argument("--lr",type=float,default=.001)
    args=parser.parse_args()
    run(args.n_pairs,args.steps,args.batch_size,args.seed,args.lr)
