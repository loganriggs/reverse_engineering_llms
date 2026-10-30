"""Direct rank-constrained sampled-environment fit after full-width WTG.

This minimizes the same weighted pair-function MSE as the environment, rather
than assuming the WTG singular cut is an optimal truncation. Selection uses
training loss only; held-out pairs are scored after optimization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize
import torch

from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from matrix_free_environment import SampledBondEnvironment


def factors(x,rank):
    u,s,vh=sla.svd(x,full_matrices=False)
    root=np.sqrt(s[:rank])
    return u[:,:rank]*root,vh[:rank].T*root


def loss_grad(z,u,v,w,target,denom,rank):
    d=u.shape[1]
    aa=z[:d*rank].reshape(d,rank)
    bb=z[d*rank:].reshape(d,rank)
    ua=u@aa
    vb=v@bb
    delta=np.sum(ua*vb,axis=1)-target
    weighted=w*delta
    loss=float(np.sum(weighted*delta)/denom)
    ga=2*u.T@(weighted[:,None]*vb)/denom
    gb=2*v.T@(weighted[:,None]*ua)/denom
    return loss,np.r_[ga.ravel(),gb.ravel()]


def check_gradient():
    rng=np.random.default_rng(101)
    u=rng.normal(size=(25,4));v=rng.normal(size=(25,4))
    w=np.exp(rng.normal(size=25));w/=w.sum()
    target=np.sum(u*v,axis=1)
    denom=np.sum(w*target**2)
    z=rng.normal(size=16)
    f=lambda zz:loss_grad(zz,u,v,w,target,denom,2)
    analytic=f(z)[1]
    numeric=np.array([(f(z+np.eye(len(z))[j]*1e-6)[0]-
                       f(z-np.eye(len(z))[j]*1e-6)[0])/(2e-6)
                      for j in range(len(z))])
    error=float(np.max(np.abs(analytic-numeric)))
    assert error<1e-8,error
    return error


def optimize(env,initial,rank,max_iter):
    u,v,w=env.u,env.v,env.w
    target=np.sum(u*v,axis=1)
    denom=float(np.sum(w*target**2))
    d=env.d
    a,b=factors(initial,rank)
    objective=lambda z:loss_grad(z,u,v,w,target,denom,rank)
    z0=np.r_[a.ravel(),b.ravel()]
    result=minimize(objective,z0,method="L-BFGS-B",jac=True,
                    options=dict(maxiter=max_iter,ftol=1e-13,gtol=1e-8,
                                 maxls=30))
    a=result.x[:d*rank].reshape(d,rank)
    b=result.x[d*rank:].reshape(d,rank)
    x=a@b.T
    return x,dict(success=bool(result.success),message=str(result.message),
                  iterations=int(result.nit),gradient_norm=float(np.linalg.norm(result.jac)),
                  initial_train_error=env.relative_error(initial),
                  final_train_error=env.relative_error(x))


def main(args):
    torch.set_num_threads(1)
    check_gradient()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r") if args.distribution=="corpus" else None
    train,val,coverage=rows(state,args.head,args.seed,args.n_train,args.n_val,4,
                            distribution=args.distribution,corpus=corpus)
    def make_env(part):
        u,v,w,_,_=part
        return SampledBondEnvironment(u,v,w/w.sum())
    train_env,val_env=make_env(train),make_env(val)
    stem=f"head{args.head}_seed{args.seed}_n{args.n_train}"
    if args.distribution=="corpus":stem+="_corpus"
    source=Path(__file__).with_name("full_width_results")/(stem+".npz")
    saved=np.load(source)
    outcomes=[];arrays={}
    for rank in args.ranks:
        for start in ("wtg","local"):
            initial=saved[f"{start}_rank{rank}"]
            x,info=optimize(train_env,initial,rank,args.max_iter)
            row=dict(rank=rank,start=start,**info,
                     validation_error=val_env.relative_error(x))
            outcomes.append(row)
            arrays[f"{start}_rank{rank}"]=x
        print(args.head,args.distribution,rank,outcomes[-2:])
    output=args.output or Path(__file__).with_name("full_width_results")/(stem+"_fet.json")
    np.savez_compressed(output.with_suffix(".npz"),**arrays)
    output.write_text(json.dumps(dict(head=args.head,seed=args.seed,
        distribution=args.distribution,n_train=args.n_train,n_val=args.n_val,
        max_iter=args.max_iter,selection="training loss only; no validation selection",
        source=str(source),coverage=coverage,outcomes=outcomes),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--head",type=int,default=0)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--distribution",choices=["uniform","corpus"],default="corpus")
    p.add_argument("--n-train",type=int,default=16384)
    p.add_argument("--n-val",type=int,default=16384)
    p.add_argument("--ranks",type=int,nargs="+",default=[16])
    p.add_argument("--max-iter",type=int,default=300)
    p.add_argument("--output",type=Path)
    main(p.parse_args())
