"""Full-width, corpus-free WTG pilot on a flagship first-layer attention head.

Token IDs are sampled uniformly from the *entire* vocabulary. The bond stays
at physical width 128; there is no selected-domain SVD quotient. Monte Carlo
samples approximate the uniform first-layer token/position metric. CPU-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import torch
from torch.nn import functional as F

from flagship_pilot import SNAPSHOT,selected_factors
from matrix_free_environment import SampledBondEnvironment


HD=128
WIDTH=1152
CORPUS=Path("/home/loganriggs/Coding/neuron_circuits/artifacts/"
            "bilin18_layers_0_2/corpus/fineweb_viewer_20k.npy")


def sample_pairs(rng,vocab,n,length):
    positions=np.array([(i,j) for i in range(length) for j in range(i+1)])
    p=positions[rng.integers(0,len(positions),size=n)]
    q=rng.integers(0,vocab,size=n,dtype=np.int64)
    k=rng.integers(0,vocab,size=n,dtype=np.int64)
    same=p[:,0]==p[:,1]
    k[same]=q[same]
    return q,k,p[:,0],p[:,1]


def sample_corpus_pairs(rng,corpus,n,length,doc_start,doc_end):
    assert corpus.ndim==2 and corpus.shape[1]>=length
    assert 0<=doc_start<doc_end<=len(corpus)
    positions=np.array([(i,j) for i in range(length) for j in range(i+1)])
    p=positions[rng.integers(0,len(positions),size=n)]
    docs=rng.integers(doc_start,doc_end,size=n)
    starts=rng.integers(0,corpus.shape[1]-length+1,size=n)
    q=np.asarray(corpus[docs,starts+p[:,0]],dtype=np.int64)
    k=np.asarray(corpus[docs,starts+p[:,1]],dtype=np.int64)
    return q,k,p[:,0],p[:,1]


def project(state,ids,head,names):
    sl=slice(head*HD,(head+1)*HD)
    emb=state["transformer.wte.weight"][torch.from_numpy(ids)].to(torch.float64)
    x0=F.rms_norm(emb,(WIDTH,))
    lambdas=state["transformer.h.0.lambdas"].to(torch.float64)
    h=F.rms_norm((lambdas[0]+lambdas[1])*x0,(WIDTH,))
    prefix="transformer.h.0.attn."
    out={}
    for name in names:
        weight=state[prefix+name+".weight"][sl].to(torch.float64)
        z=h@weight.T
        if name!="c_v":
            z=F.rms_norm(z,(HD,))
        out[name]=z.numpy()
    return out


def rotary(length):
    inv=1.0/(10000**(torch.arange(0,HD,2,dtype=torch.float32)/HD))
    freq=torch.outer(torch.arange(length,dtype=torch.float32),inv)
    return freq.cos().bfloat16().to(torch.float64).numpy(),\
           freq.sin().bfloat16().to(torch.float64).numpy()


def rotate(z,pos,cos,sin):
    left,right=np.split(z,2,axis=-1)
    c,s=cos[pos],sin[pos]
    return np.concatenate((left*c+right*s,-left*s+right*c),axis=-1)


def rows(state,head,seed,n_train,n_val,length,swap=False,
         distribution="uniform",corpus=None,train_docs=(0,15000),val_docs=(15000,20000)):
    vocab=state["transformer.wte.weight"].shape[0]
    rng=np.random.default_rng(seed)
    if distribution=="uniform":
        train=sample_pairs(rng,vocab,n_train,length)
        val=sample_pairs(rng,vocab,n_val,length)
    elif distribution=="corpus":
        assert corpus is not None
        train=sample_corpus_pairs(rng,corpus,n_train,length,*train_docs)
        val=sample_corpus_pairs(rng,corpus,n_val,length,*val_docs)
    else:
        raise ValueError(distribution)
    q_ids=np.concatenate((train[0],val[0]))
    k_ids=np.concatenate((train[1],val[1]))
    q_unique,q_inverse=np.unique(q_ids,return_inverse=True)
    k_unique,k_inverse=np.unique(k_ids,return_inverse=True)
    query=project(state,q_unique,head,("c_q","c_q2"))
    key=project(state,k_unique,head,("c_k","c_k2","c_v"))
    sl=slice(head*HD,(head+1)*HD)
    output=state["transformer.h.0.attn.c_proj.weight"][:,sl].to(torch.float64).numpy()
    output_gram=output.T@output
    c_norm=np.einsum("nr,rs,ns->n",key["c_v"],output_gram,key["c_v"],optimize=True)
    assert np.min(c_norm)>=0
    cos,sin=rotary(length)

    def make(part,start,end):
        q,k,i,j=part
        qi=q_inverse[start:end]
        ki=k_inverse[start:end]
        u=rotate(query["c_q"][qi],i,cos,sin)
        v=rotate(key["c_k"][ki],j,cos,sin)
        u2=rotate(query["c_q2"][qi],i,cos,sin)
        v2=rotate(key["c_k2"][ki],j,cos,sin)
        if swap:
            u,v,u2,v2=u2,v2,u,v
        b=np.sum(u2*v2,axis=1)/(HD**2)
        w=b*b*c_norm[ki]
        assert np.all(np.isfinite(w)) and np.sum(w)>0
        return u,v,w,b,c_norm[ki]

    train_rows=make(train,0,n_train)
    val_rows=make(val,n_train,n_train+n_val)
    return train_rows,val_rows,dict(vocab=vocab,distribution=distribution,
        unique_query_tokens=int(len(q_unique)),unique_key_tokens=int(len(k_unique)),
        sampled_causal_positions=length*(length+1)//2)


def local_svd(u,v,rank):
    # SVD of the complete sampled QK1 matrix via its d-by-d QR core.
    _,ru=sla.qr(u,mode="economic")
    _,rv=sla.qr(v,mode="economic")
    core=ru@rv.T
    left,s,right=sla.svd(core)
    approximation=(left[:,:rank]*s[:rank])@right[:rank]
    return sla.solve(ru,approximation)@sla.inv(rv.T)


def gate(state,head,length,seed):
    rng=np.random.default_rng(seed+991)
    vocab=state["transformer.wte.weight"].shape[0]
    q,k,i,j=sample_pairs(rng,vocab,4,length)
    ids=np.unique(np.r_[q,k])
    query=project(state,ids,head,("c_q","c_q2"))
    key=project(state,ids,head,("c_k","c_k2","c_v"))
    cos,sin=rotary(length)
    u=rotate(query["c_q"][np.searchsorted(ids,q)],i,cos,sin)
    v=rotate(key["c_k"][np.searchsorted(ids,k)],j,cos,sin)
    u2=rotate(query["c_q2"][np.searchsorted(ids,q)],i,cos,sin)
    v2=rotate(key["c_k2"][np.searchsorted(ids,k)],j,cos,sin)
    b=np.sum(u2*v2,axis=1)/(HD**2)
    output=state["transformer.h.0.attn.c_proj.weight"][:,head*HD:(head+1)*HD].to(torch.float64).numpy()
    c=key["c_v"][np.searchsorted(ids,k)]@output.T
    old_u,old_v,old_b,old_c=selected_factors(state,ids,head,length)
    qi=np.searchsorted(ids,q);ki=np.searchsorted(ids,k)
    errors=[np.max(np.abs(u-old_u[i,qi])),np.max(np.abs(v-old_v[j,ki])),
            np.max(np.abs(b-old_b[i,j,qi,ki])),np.max(np.abs(c-old_c[ki]))]
    gate_error=float(max(errors))
    assert gate_error<1e-10,errors
    return gate_error


def main(args):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    direct_gate=gate(state,args.head,args.length,args.seed)
    corpus=np.load(args.corpus,mmap_mode="r") if args.distribution=="corpus" else None
    train,val,coverage=rows(state,args.head,args.seed,args.n_train,args.n_val,args.length,
                            swap=args.swap,distribution=args.distribution,corpus=corpus,
                            train_docs=tuple(args.train_docs),val_docs=tuple(args.val_docs))
    u,v,w,_,_=train
    uv,vv,wv,_,_=val
    env=SampledBondEnvironment(u,v,w/np.sum(w))
    validation=SampledBondEnvironment(uv,vv,wv/np.sum(wv))
    _,info=env.weighted_trace_gauge(max(args.ranks),max_iter=args.max_iter,tol=args.tol)
    results=[]
    arrays={}
    for rank in args.ranks:
        x,details=env.weighted_trace_gauge(rank,max_iter=args.max_iter,tol=args.tol)
        local=local_svd(u,v,rank)
        arrays[f"wtg_rank{rank}"]=x
        arrays[f"local_rank{rank}"]=local
        results.append(dict(rank=rank,wtg_train_error=env.relative_error(x),
                            wtg_validation_error=validation.relative_error(x),
                            local_train_error=env.relative_error(local),
                            local_validation_error=validation.relative_error(local)))
    output=args.output or Path(__file__).with_name(
        f"flagship_matrix_free_h{args.head}_n{args.n_train}_seed{args.seed}.json")
    array_path=output.with_suffix(".npz")
    np.savez_compressed(array_path,**arrays)
    data=dict(scope="full physical width-128 head; Monte Carlo over declared token and causal-position measure",
              checkpoint=str(SNAPSHOT/"pytorch_model.bin"),head=args.head,
              swap_qk_branches=args.swap,
              distribution=args.distribution,
              corpus=str(args.corpus) if corpus is not None else None,
              train_docs=args.train_docs if corpus is not None else None,
              val_docs=args.val_docs if corpus is not None else None,
              seed=args.seed,length=args.length,n_train=args.n_train,n_val=args.n_val,
              direct_factor_gate_max_abs=direct_gate,coverage=coverage,
              boundary=info,results=results,arrays=str(array_path))
    output.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(dict(output=str(output),head=args.head,seed=args.seed,swap=args.swap,
                          distribution=args.distribution,
                          n_train=args.n_train,n_val=args.n_val,
                          gate_max_abs=direct_gate,results=results),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--head",type=int,default=0)
    p.add_argument("--swap",action="store_true")
    p.add_argument("--distribution",choices=["uniform","corpus"],default="uniform")
    p.add_argument("--corpus",type=Path,default=CORPUS)
    p.add_argument("--train-docs",type=int,nargs=2,default=[0,15000])
    p.add_argument("--val-docs",type=int,nargs=2,default=[15000,20000])
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--length",type=int,default=4)
    p.add_argument("--n-train",type=int,default=4096)
    p.add_argument("--n-val",type=int,default=4096)
    p.add_argument("--ranks",type=int,nargs="+",default=[8,16,32])
    p.add_argument("--max-iter",type=int,default=100)
    p.add_argument("--tol",type=float,default=1e-8)
    p.add_argument("--output",type=Path)
    main(p.parse_args())
