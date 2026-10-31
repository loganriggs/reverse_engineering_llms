"""Uniform, pair-frequency, and sequence-aware cuts of the nonlinear toy."""

from __future__ import annotations

import json

import numpy as np
import scipy.optimize as so
import torch

from analyze_xor_count_language import (attention_table,attention_from_table,
    downstream,flatten,unflatten,score)
from dual_joint_function_canonical import HERE
from three_count_metric import weighted_fit
from train_xor_count_language import XorCountAttention,examples


def counts_and_query(tokens):
    count=np.stack([np.sum(tokens[:,:-1]==key,axis=1)
                    for key in range(6)],axis=1).astype(float)
    return count,tokens[:,-1]-6


def data_measure(tokens):
    count,query=counts_and_query(tokens)
    pair=np.zeros((2,6),dtype=float)
    covariance=[]
    for q in range(2):
        selected=count[query==q]
        pair[q]=selected.sum(0)
        covariance.append(selected.T@selected/len(tokens))
    pair=np.repeat(pair,2,axis=0)
    pair/=pair.sum()
    return pair,covariance


def sequence_fit(matrix,covariance,rank):
    u,s,v=np.linalg.svd(matrix,full_matrices=False)
    left=u[:,:rank]*np.sqrt(s[:rank])
    right=np.sqrt(s[:rank,None])*v[:rank]
    start=np.r_[left.ravel(),right.ravel()]
    roots=[np.linalg.cholesky(c+np.eye(6)*1e-12) for c in covariance]
    # C=L L.T, so squared row error is ||row@L||^2.
    def residual(z):
        l=z[:left.size].reshape(left.shape)
        r=z[left.size:].reshape(rank,6)
        delta=l@r-matrix
        return np.concatenate([(delta[2*q:2*q+2]@roots[q]).ravel()
                               for q in range(2)])
    fit=so.least_squares(residual,start,max_nfev=10000,
        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    z=fit.x
    reduced=z[:left.size].reshape(left.shape)@z[left.size:].reshape(rank,6)
    return reduced,dict(nfev=fit.nfev,loss=float(np.sum(residual(z)**2)),
                        optimality=float(fit.optimality))


def run():
    torch.set_num_threads(1)
    rows=[]
    for seed in range(10):
        model=XorCountAttention().double()
        model.load_state_dict(torch.load(HERE/f"xor_count_seed{seed}.pt",
                                         map_location="cpu",weights_only=True))
        model.eval()
        matrix=flatten(attention_table(model))
        train_t,_,_=examples(50000,1000+seed,True)
        pair,covariance=data_measure(train_t.numpy())
        uniform=np.ones_like(pair)/pair.size
        test_t,target_t,attrs=examples(20000,3000+seed,False)
        tokens=test_t.numpy();targets=target_t.numpy().astype(float)
        original_heads=attention_from_table(unflatten(matrix),tokens)
        original_logits=downstream(model,original_heads)
        for rank in (1,2):
            measures={
                "uniform_pair":weighted_fit(matrix,uniform,rank),
                "training_pair":weighted_fit(matrix,pair,rank),
                "half_pair":weighted_fit(matrix,.5*(uniform+pair),rank),
                "training_sequence":sequence_fit(matrix,covariance,rank)}
            for name,(reduced,fit) in measures.items():
                scored=score(model,unflatten(reduced),tokens,targets,attrs,
                             original_logits,original_heads)
                rows.append(dict(seed=seed,rank=rank,measure=name,fit=fit,
                    relative_uniform_table_mse=float(np.sum((reduced-matrix)**2)/
                                                     np.sum(matrix**2)),
                    scores=scored))
        print("seed",seed,flush=True)
    output=dict(n_seeds=10,n_train_unlabeled=50000,n_test=20000,
        fit="rank-constrained 4x6 first-layer head table from weights; data variants use only unlabeled training sequences",
        measures="uniform token cells; train pair counts; half mixture; train sequence head-output MSE",
        evaluation="independent variable-count A/B/XOR sequences and rare whole-combination holdout",
        rows=rows)
    (HERE/"xor_count_metric.json").write_text(json.dumps(output,indent=2)+"\n")
    for rank in (1,2):
        for name in ("uniform_pair","training_pair","half_pair","training_sequence"):
            selected=[r for r in rows if r["rank"]==rank and r["measure"]==name]
            print(rank,name,
                "joint",np.mean([r["scores"]["all"]["joint_accuracy"] for r in selected]),
                "rare",np.mean([r["scores"]["rare_heldout"]["joint_accuracy"] for r in selected]),
                "head",np.mean([r["scores"]["relative_head_mse"] for r in selected]),
                "logit",np.mean([r["scores"]["relative_logit_mse"] for r in selected]),
                flush=True)
    return output


if __name__=="__main__":run()
