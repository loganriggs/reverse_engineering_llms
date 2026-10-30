"""Sequence-aggregated layer-0 head output fidelity for saved QK1 cuts.

This sums causal key contributions through the actual OV projection on token
sequences. It does not propagate the edit through layers 1-17 or measure CE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary
from flagship_pilot import selected_factors
from flagship_transfer import ART,load_cuts
from flagship_ood_corpora import DATASETS


def sequence_head(state,tokens,head,cuts):
    length=len(tokens)
    ids,inverse=np.unique(tokens,return_inverse=True)
    projection=project(state,ids,head,("c_q","c_k","c_q2","c_k2","c_v"))
    cos,sin=rotary(length)
    positions=np.arange(length)
    u=rotate(projection["c_q"][inverse],positions,cos,sin)
    v=rotate(projection["c_k"][inverse],positions,cos,sin)
    u2=rotate(projection["c_q2"][inverse],positions,cos,sin)
    v2=rotate(projection["c_k2"][inverse],positions,cos,sin)
    b=(u2@v2.T)/(HD**2)
    b*=np.tril(np.ones((length,length)))
    sl=slice(head*HD,(head+1)*HD)
    output=state["transformer.h.0.attn.c_proj.weight"][:,sl].to(torch.float64).numpy()
    value=projection["c_v"][inverse]@output.T
    all_cuts={"identity":np.eye(HD),**cuts}
    return {name:((u@x@v.T)*b)@value for name,x in all_cuts.items()}


def check_small(state):
    seq=np.array([3,8,11,3],dtype=np.int64)
    computed=sequence_head(state,seq,0,{})["identity"]
    ids=np.unique(seq);inverse=np.searchsorted(ids,seq)
    u,v,b,c=selected_factors(state,ids,0,len(seq))
    direct=np.stack([sum((u[i,inverse[i]]@v[j,inverse[j]])*
                         b[i,j,inverse[i],inverse[j]]*c[inverse[j]]
                         for j in range(i+1)) for i in range(len(seq))])
    error=float(np.max(np.abs(direct-computed)))
    assert error<1e-10,error
    return error


def check_direct_long(state,length=128):
    rng=np.random.default_rng(811)
    tokens=rng.integers(0,state["transformer.wte.weight"].shape[0],size=length)
    computed=sequence_head(state,tokens,0,{})["identity"]
    input_ids=torch.from_numpy(tokens)
    emb=state["transformer.wte.weight"][input_ids].to(torch.float64)
    x0=F.rms_norm(emb,(1152,))
    lambdas=state["transformer.h.0.lambdas"].to(torch.float64)
    h=F.rms_norm((lambdas[0]+lambdas[1])*x0,(1152,))
    sl=slice(0,HD)
    prefix="transformer.h.0.attn."
    inverse=1.0/(10000**(torch.arange(0,HD,2,dtype=torch.float32)/HD))
    phase=torch.outer(torch.arange(length,dtype=torch.float32),inverse)
    cosine=phase.cos().bfloat16().to(torch.float64)
    sine=phase.sin().bfloat16().to(torch.float64)
    def qk(name):
        z=h@state[prefix+name+".weight"][sl].to(torch.float64).T
        z=F.rms_norm(z,(HD,))
        left,right=z.chunk(2,dim=-1)
        return torch.cat((left*cosine+right*sine,-left*sine+right*cosine),dim=-1)
    q1,k1,q2,k2=[qk(name) for name in ("c_q","c_k","c_q2","c_k2")]
    score=(q1@k1.T)*(q2@k2.T)/(HD**2)
    score=score.tril()
    value=h@state[prefix+"c_v.weight"][sl].to(torch.float64).T
    direct=(score@value)@state[prefix+"c_proj.weight"][:,sl].to(torch.float64).T
    error=float(np.max(np.abs(direct.numpy()-computed)))
    assert error<1e-10,error
    return error


def run(n_docs=32,seeds=(23,29)):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    gate=check_small(state)
    long_gate=check_direct_long(state)
    all_cuts=load_cuts()
    names=("uniform_fit_l4_wtg16","uniform_fit_l4_direct_wtg_start16",
           "corpus_fit_l4_direct_wtg_start16","uniform_fit_l128_wtg16",
           "corpus_fit_l128_wtg16","corpus_fit_l128_direct_wtg_start16",
           "uniform_fit_l128_wtg32","uniform_fit_l128_wtg64",
           "uniform_fit_l128_wtg96","corpus_fit_l128_wtg32",
           "corpus_fit_l128_wtg64","corpus_fit_l128_wtg96")
    cuts={name:all_cuts[name] for name in names}
    corpora={"fineweb_heldout":(CORPUS,15000,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[]
    for corpus_name,(path,begin,end) in corpora.items():
        corpus=np.load(path,mmap_mode="r")
        assert end<=len(corpus)
        for length in (4,128):
            assert length<=corpus.shape[1]
            for seed in seeds:
                rng=np.random.default_rng(seed+length*1000)
                docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
                totals={name:dict(sse=0.,target_energy=0.,final_sse=0.,
                                  final_energy=0.) for name in names}
                for doc in docs:
                    start=int(rng.integers(0,corpus.shape[1]-length+1))
                    tokens=np.asarray(corpus[doc,start:start+length],dtype=np.int64)
                    outputs=sequence_head(state,tokens,0,cuts)
                    original=outputs["identity"]
                    energy=float(np.sum(original**2))
                    final_energy=float(np.sum(original[-1]**2))
                    for name in names:
                        delta=outputs[name]-original
                        totals[name]["sse"]+=float(np.sum(delta**2))
                        totals[name]["target_energy"]+=energy
                        totals[name]["final_sse"]+=float(np.sum(delta[-1]**2))
                        totals[name]["final_energy"]+=final_energy
                for name,stat in totals.items():
                    records.append(dict(corpus=corpus_name,length=length,
                        seed=seed,n_docs=n_docs,cut=name,
                        sequence_head_error=stat["sse"]/stat["target_energy"],
                        final_position_head_error=stat["final_sse"]/stat["final_energy"],
                        **stat))
                print(corpus_name,length,seed,flush=True)
    result=dict(head=0,ranks=[16,32,64,96],small_exact_gate_max_abs=gate,
        direct_length128_gate_max_abs=long_gate,
        seeds=list(seeds),n_docs_per_seed=n_docs,
        scope="layer-0 head output after causal sum and OV; no later layers or CE",
        corpus_paths={k:str(p) for k,(p,_,_) in corpora.items()},records=records)
    output=ART/"head0_sequence_transfer.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-docs",type=int,default=32)
    parser.add_argument("--seeds",type=int,nargs="+",default=[23,29])
    args=parser.parse_args()
    run(args.n_docs,tuple(args.seeds))
