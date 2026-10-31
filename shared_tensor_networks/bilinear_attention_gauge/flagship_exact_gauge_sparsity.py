"""Exact-gauge sparsity priors on a fixed rank-64 first-layer QK1 cut.

Compare SVD factors, random orthogonal gauges, factor-weight L1, and sparse
pair contributions. Orthogonal rotations change component coordinates but
leave the bond matrix and complete model function exactly unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.optimize as so
import torch

from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART
from flagship_full_model_eval import get_cuts


def pair_features(part,a,b,constant=False):
    u,v,natural_weight,_,constant_weight=part
    w=constant_weight if constant else natural_weight
    f=u@a;g=v@b
    score=np.sum(f*g,axis=1)
    energy=w*score**2
    return tuple(torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64)
                 for x in (f,g,w,score,energy))


def contribution_stats(part,rotation):
    f,g,w,score,energy=part
    z=(f@rotation)*(g@rotation)
    absz=z.abs()
    sq=z.square()
    support=absz.sum(1).square()/sq.sum(1).clamp_min(1e-30)
    weighted_support=(energy*support).sum()/energy.sum()
    cancellation=(w*absz.sum(1)).sum()/(w*score.abs()).sum()
    top1=(energy*sq.max(1).values/sq.sum(1).clamp_min(1e-30)).sum()/energy.sum()
    return dict(effective_active=float(weighted_support),
                cancellation_factor=float(cancellation),
                top_component_energy_fraction=float(top1))


def activation_loss(parts,rotation):
    values=[]
    for f,g,w,score,energy in parts:
        z=(f@rotation)*(g@rotation)
        support=z.abs().sum(1).square()/z.square().sum(1).clamp_min(1e-30)
        values.append((energy*support).sum()/energy.sum())
    return sum(values)/len(values)


def weight_loss(a,b,rotation):
    return ((a@rotation).abs().mean()+(b@rotation).abs().mean())


def fit(kind,train,validation,a,b,seed,steps=400,batch_size=2048,lr=.01):
    generator=np.random.default_rng(seed)
    n=a.shape[1]
    k=torch.nn.Parameter(torch.zeros((n,n),dtype=torch.float64))
    optimizer=torch.optim.Adam((k,),lr=lr)
    best_rotation=torch.eye(n,dtype=torch.float64)
    if kind in ("activation","intervention_activation",
                "rare_balanced_activation"):
        names=(("uniform","corpus") if kind=="activation" else
               ("uniform","corpus","rare") if kind=="rare_balanced_activation"
               else ("uniform","corpus","uniform_constant","corpus_constant"))
        best=float(activation_loss([validation[name] for name in names],
                                   best_rotation))
    else:
        best=float(weight_loss(a,b,best_rotation))
    best_step=0;trace=[]
    for step in range(steps):
        optimizer.zero_grad()
        rotation=torch.matrix_exp(k-k.T)
        if kind in ("activation","intervention_activation",
                    "rare_balanced_activation"):
            minibatches=[]
            for law in names:
                part=train[law]
                ix=torch.from_numpy(generator.choice(len(part[0]),batch_size,
                                                     replace=False))
                minibatches.append(tuple(z[ix] for z in part))
            loss=activation_loss(minibatches,rotation)
        elif kind=="factor_weight":
            loss=weight_loss(a,b,rotation)
        else:
            raise ValueError(kind)
        if not torch.isfinite(loss):
            raise ValueError((kind,step))
        loss.backward()
        torch.nn.utils.clip_grad_norm_((k,),10.)
        optimizer.step()
        if step==0 or (step+1)%50==0 or step+1==steps:
            with torch.no_grad():
                rotation=torch.matrix_exp(k-k.T)
                measured=(activation_loss([validation[name] for name in names],
                                          rotation)
                          if kind in ("activation","intervention_activation",
                                      "rare_balanced_activation")
                          else weight_loss(a,b,rotation))
                value=float(measured)
                if value<best:
                    best=value;best_step=step+1
                    best_rotation=rotation.detach().clone()
                row=dict(step=step+1,minibatch_loss=float(loss.detach()),
                         selection_objective=value)
                trace.append(row)
                print(kind,seed,row,flush=True)
    return best_rotation.numpy(),dict(best_step=best_step,best_objective=best,
                                      trace=trace)


def signature(part,rotation):
    f,g,*_=part
    z=((f@rotation)*(g@rotation)).numpy()
    z=z-z.mean(0,keepdims=True)
    norm=np.linalg.norm(z,axis=0)
    return z/np.maximum(norm,1e-20)


def match_signature(x,y):
    correlation=x.T@y
    row,col=so.linear_sum_assignment(-np.abs(correlation))
    return float(np.mean(np.abs(correlation[row,col])))


def check_gradient():
    rng=np.random.default_rng(127)
    f=torch.from_numpy(rng.normal(size=(32,4)))
    g=torch.from_numpy(rng.normal(size=(32,4)))
    score=(f*g).sum(1)
    w=torch.from_numpy(np.exp(rng.normal(size=32)))
    energy=w*score.square()
    part=(f,g,w,score,energy)
    k=torch.tensor(rng.normal(scale=.03,size=(4,4)),requires_grad=True)
    def objective(matrix):
        rotation=torch.matrix_exp(matrix-matrix.T)
        return activation_loss([part],rotation)
    gradient=torch.autograd.grad(objective(k),k)[0]
    eps=1e-6
    direction=torch.zeros_like(k);direction[0,1]=1
    finite=float((objective(k.detach()+eps*direction)-
                  objective(k.detach()-eps*direction))/(2*eps))
    autodiff=float(gradient[0,1])
    relative=abs(autodiff-finite)/max(abs(finite),abs(autodiff),1e-12)
    assert relative<1e-6,(autodiff,finite,relative)
    return dict(autodiff=autodiff,central_difference=finite,
                relative_error=relative)


def run(n_pairs=32768,n_ood=8192,steps=400,batch_size=2048,
        seeds=(0,1,2),lr=.01,cut_name="sequence_fit_rank64_cw0"):
    torch.set_num_threads(1)
    gradient_check=check_gradient()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    x=get_cuts()[cut_name]
    left,s,right=np.linalg.svd(x,full_matrices=False)
    rank=int(np.linalg.matrix_rank(x))
    a=left[:,:rank]*np.sqrt(s[:rank]);b=right[:rank].T*np.sqrt(s[:rank])
    ta=torch.from_numpy(a);tb=torch.from_numpy(b)
    fit_parts={};val_parts={};test_parts={}
    for law in ("uniform","corpus"):
        train,val,_=rows(state,0,41,n_pairs,n_pairs,128,
            distribution=law,corpus=corpus if law=="corpus" else None,
            val_docs=(15000,17500))
        fit_parts[law]=pair_features(train,a,b)
        val_parts[law]=pair_features(val,a,b)
        fit_parts[law+"_constant"]=pair_features(train,a,b,True)
        val_parts[law+"_constant"]=pair_features(val,a,b,True)
        _,test,_=rows(state,0,23,1,n_ood,128,
            distribution=law,corpus=corpus if law=="corpus" else None,
            val_docs=(17500,20000))
        test_parts[law]=pair_features(test,a,b)
        test_parts[law+"_constant"]=pair_features(test,a,b,True)
    vocab=state["transformer.wte.weight"].shape[0]
    frequency=np.bincount(corpus[:15000,:512].ravel(),minlength=vocab)
    pools={"rare_1_to_10":np.flatnonzero((frequency>=1)&(frequency<=10)),
           "common_1000_plus":np.flatnonzero(frequency>=1000),
           "unseen_zero":np.flatnonzero(frequency==0)}
    rare_fit,rare_val,_=rows(state,0,41,n_pairs,n_pairs,128,
        distribution="uniform",token_pool=pools["rare_1_to_10"])
    fit_parts["rare"]=pair_features(rare_fit,a,b)
    val_parts["rare"]=pair_features(rare_val,a,b)
    for name,pool in pools.items():
        _,test,_=rows(state,0,23,1,n_ood,128,
            distribution="uniform",token_pool=pool)
        test_parts[name]=pair_features(test,a,b)
        test_parts[name+"_constant"]=pair_features(test,a,b,True)
    for name,path in DATASETS.items():
        other=np.load(path,mmap_mode="r")
        _,test,_=rows(state,0,23,1,n_ood,128,
            distribution="corpus",corpus=other,
            train_docs=(0,600),val_docs=(600,1200))
        test_parts[name]=pair_features(test,a,b)
        test_parts[name+"_constant"]=pair_features(test,a,b,True)
    del state
    rotations={"svd":np.eye(rank)}
    fit_info={}
    for seed in seeds:
        rng=np.random.default_rng(100+seed)
        q,r=np.linalg.qr(rng.normal(size=(rank,rank)))
        rotations[f"random_{seed}"]=q@np.diag(np.sign(np.diag(r)))
        for kind in ("activation","intervention_activation",
                     "rare_balanced_activation","factor_weight"):
            rotation,info=fit(kind,fit_parts,val_parts,ta,tb,seed,
                              steps,batch_size,lr)
            rotations[f"{kind}_{seed}"]=rotation
            fit_info[f"{kind}_{seed}"]=info
    identity=np.eye(rank)
    records=[]
    for name,rotation in rotations.items():
        orthogonal=float(np.linalg.norm(rotation.T@rotation-identity))
        bond_error=float(np.linalg.norm((a@rotation)@(b@rotation).T-x)/
                         np.linalg.norm(x))
        assert orthogonal<1e-9 and bond_error<1e-9,(name,orthogonal,bond_error)
        rt=torch.from_numpy(rotation)
        for split,parts in (("validation",val_parts),("test",test_parts)):
            for law,part in parts.items():
                records.append(dict(gauge=name,split=split,law=law,
                    orthogonality_error=orthogonal,bond_relative_error=bond_error,
                    factor_weight_l1=float(weight_loss(ta,tb,rt)),
                    **contribution_stats(part,rt)))
    signatures={name:signature(test_parts["corpus"],torch.from_numpy(rotation))
                for name,rotation in rotations.items()}
    matches=[]
    for kind in ("activation","intervention_activation",
                 "rare_balanced_activation","factor_weight","random"):
        for i,seed in enumerate(seeds):
            for other_seed in seeds[i+1:]:
                a_name=f"{kind}_{seed}";b_name=f"{kind}_{other_seed}"
                matches.append(dict(kind=kind,seeds=[seed,other_seed],
                    mean_matched_abs_correlation=match_signature(
                        signatures[a_name],signatures[b_name])))
    output=dict(head=0,rank=rank,cut=cut_name,
        n_fit_pairs_per_law=n_pairs,n_validation_pairs_per_law=n_pairs,
        n_test_pairs_per_law=n_ood,steps=steps,batch_size=batch_size,
        seeds=list(seeds),learning_rate=lr,
        fit_fineweb_documents=[0,15000],
        validation_fineweb_documents=[15000,17500],
        test_fineweb_documents=[17500,20000],
        token_pool_sizes={name:int(len(pool)) for name,pool in pools.items()},
        scope="exact orthogonal gauges of fixed bond; sparsity proxies, not semantic labels",
        gradient_check=gradient_check,fit_info=fit_info,
        records=records,matches=matches)
    suffix="" if cut_name=="sequence_fit_rank64_cw0" else f"_{cut_name}"
    (ART/f"head0_exact_gauge_sparsity{suffix}.json").write_text(
        json.dumps(output,indent=2)+"\n")
    np.savez_compressed(ART/f"head0_exact_gauge_rotations{suffix}.npz",**rotations)
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-pairs",type=int,default=32768)
    parser.add_argument("--n-ood",type=int,default=8192)
    parser.add_argument("--steps",type=int,default=400)
    parser.add_argument("--batch-size",type=int,default=2048)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--lr",type=float,default=.01)
    parser.add_argument("--cut-name",default="sequence_fit_rank64_cw0",
                        choices=tuple(get_cuts()))
    args=parser.parse_args()
    run(args.n_pairs,args.n_ood,args.steps,args.batch_size,tuple(args.seeds),
        args.lr,args.cut_name)
