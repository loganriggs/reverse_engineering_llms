"""Rank-constrained QK1 fit to actual first-layer head sequence outputs.

This is a direct sequence-response optimization, not WTG canonicalization.
It starts from a predeclared mixed-metric WTG cut and tests whether optimizing
the correct aggregation target improves held-out sequence and intervention
fidelity. No later layers or CE are included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary
from flagship_transfer import ART


def features(state,tokens):
    length=len(tokens)
    ids,inverse=np.unique(tokens,return_inverse=True)
    z=project(state,ids,0,("c_q","c_k","c_q2","c_k2","c_v"))
    cos,sin=rotary(length)
    pos=np.arange(length)
    u=rotate(z["c_q"][inverse],pos,cos,sin)
    v=rotate(z["c_k"][inverse],pos,cos,sin)
    u2=rotate(z["c_q2"][inverse],pos,cos,sin)
    v2=rotate(z["c_k2"][inverse],pos,cos,sin)
    b=((u2@v2.T)/(HD**2))*np.tril(np.ones((length,length)))
    c=z["c_v"][inverse]
    return tuple(torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64)
                 for x in (u,v,b,c))


def sample_features(state,corpus,begin,end,n,seed,length):
    rng=np.random.default_rng(seed)
    docs=rng.choice(np.arange(begin,end),size=n,replace=False)
    output=[]
    for doc in docs:
        start=int(rng.integers(0,corpus.shape[1]-length+1))
        tokens=np.asarray(corpus[doc,start:start+length],dtype=np.int64)
        output.append(features(state,tokens))
    return tuple(torch.stack([part[i] for part in output]) for i in range(4)),docs


def head_values(parts,x,mode):
    u,v,b,c=parts
    if mode=="constant":
        b=torch.tril(torch.ones_like(b))
    elif mode!="natural":
        raise ValueError(mode)
    score=torch.bmm(torch.matmul(u,x),v.transpose(1,2))*b
    return torch.bmm(score,c)


def squared_gram_error(delta,gram):
    return torch.einsum("nlo,op,nlp->",delta,gram,delta)


def loss_for(parts,x,targets,gram,mode):
    prediction=head_values(parts,x,mode)
    delta=prediction-targets[mode]
    return squared_gram_error(delta,gram)/squared_gram_error(targets[mode],gram)


def run(n_train=256,n_validation=128,steps=400,batch_size=8,rank=64,
        constant_weight=0.,seed=41,lr=.003):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64)
    gram=output.T@output
    train,train_docs=sample_features(state,corpus,0,15000,n_train,seed,128)
    validation,val_docs=sample_features(state,corpus,15000,17500,
                                        n_validation,seed+1,128)
    modes=("natural","constant")
    eye=torch.eye(HD,dtype=torch.float64)
    train_targets={mode:head_values(train,eye,mode).detach() for mode in modes}
    val_targets={mode:head_values(validation,eye,mode).detach() for mode in modes}
    initial=np.load(ART/"head0_l128_uniform_corpus_half.npz")[f"rank{rank}"]
    left,s,right=np.linalg.svd(initial,full_matrices=False)
    a=torch.nn.Parameter(torch.from_numpy(left[:,:rank]*np.sqrt(s[:rank])).to(torch.float64))
    b=torch.nn.Parameter(torch.from_numpy(right[:rank].T*np.sqrt(s[:rank])).to(torch.float64))
    optimizer=torch.optim.Adam((a,b),lr=lr)
    rng=np.random.default_rng(seed+1000)
    def score(parts,targets,x):
        return {mode:float(loss_for(parts,x,targets,gram,mode)) for mode in modes}
    initial_scores=score(validation,val_targets,torch.from_numpy(initial))
    trace=[]
    best_objective=(1-constant_weight)*initial_scores["natural"]+\
                   constant_weight*initial_scores["constant"]
    best_cut=initial.copy()
    best_step=0
    for step in range(steps):
        ix=torch.from_numpy(rng.choice(n_train,batch_size,replace=False))
        parts=tuple(t[ix] for t in train)
        targets={mode:t[ix] for mode,t in train_targets.items()}
        optimizer.zero_grad()
        x=a@b.T
        natural=loss_for(parts,x,targets,gram,"natural")
        if constant_weight:
            constant=loss_for(parts,x,targets,gram,"constant")
            loss=(1-constant_weight)*natural+constant_weight*constant
        else:
            loss=natural
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((a,b),10.)
        optimizer.step()
        if step==0 or (step+1)%50==0 or step+1==steps:
            with torch.no_grad():
                x=a@b.T
                train_score=score(train,train_targets,x)
                val_score=score(validation,val_targets,x)
            record=dict(step=step+1,minibatch_loss=float(loss.detach()),
                        train=train_score,validation=val_score)
            trace.append(record)
            objective=(1-constant_weight)*val_score["natural"]+\
                      constant_weight*val_score["constant"]
            if objective<best_objective:
                best_objective=objective
                best_cut=x.detach().numpy().copy()
                best_step=step+1
            print(record,flush=True)
    x=best_cut
    name=f"head0_l128_sequence_fit_rank{rank}_cw{constant_weight:g}"
    np.savez_compressed(ART/f"{name}.npz",cut=x)
    result=dict(name=name,head=0,rank=rank,seed=seed,n_train=n_train,
        n_validation=n_validation,steps=steps,batch_size=batch_size,
        learning_rate=lr,constant_weight=constant_weight,
        fit_documents=[0,15000],validation_documents=[15000,17500],
        fit_doc_ids=train_docs.tolist(),validation_doc_ids=val_docs.tolist(),
        initial_validation=initial_scores,trace=trace,
        selection="lowest validation mixture error including initial cut",
        best_step=best_step,best_validation_objective=best_objective,
        output_gram="c_proj_head0.T @ c_proj_head0; values kept in width-128 coordinates",
        scope="first-layer head output after causal sum and OV; no later layers or CE")
    (ART/f"{name}.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-train",type=int,default=256)
    parser.add_argument("--n-validation",type=int,default=128)
    parser.add_argument("--steps",type=int,default=400)
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--rank",type=int,default=64)
    parser.add_argument("--constant-weight",type=float,default=0.)
    parser.add_argument("--seed",type=int,default=41)
    parser.add_argument("--lr",type=float,default=.003)
    args=parser.parse_args()
    run(args.n_train,args.n_validation,args.steps,args.batch_size,
        args.rank,args.constant_weight,args.seed,args.lr)
