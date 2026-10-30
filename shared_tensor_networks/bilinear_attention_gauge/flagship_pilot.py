"""CPU-only selected-domain WTG pilot on the cached 546M bilinear checkpoint.

This is a *restricted token-domain quotient*, not a full-vocabulary or full-model
canonicalization. It never loads the 18-layer module or uses the running GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import scipy.linalg as sla

from run import (cut_wtg, env_error, fet_als, gram_environment, local_svd,
                 pair_measure, token_table, weighted_trace_gauge, wtg_boundary_check)


SNAPSHOT = Path("/home/loganriggs/.cache/huggingface/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240")


def selected_factors(state, ids, head, length, swap=False):
    hd = 128
    sl = slice(head*hd,(head+1)*hd)
    emb = state["transformer.wte.weight"][ids].to(torch.float64)
    x0 = F.rms_norm(emb,(1152,))
    lambdas = state["transformer.h.0.lambdas"].to(torch.float64)
    h = F.rms_norm((lambdas[0]+lambdas[1])*x0,(1152,))
    prefix = "transformer.h.0.attn."

    def projection(name, normalize):
        weight = state[prefix+name+".weight"]
        z = h @ weight[sl].to(torch.float64).T
        return F.rms_norm(z,(hd,)) if normalize else z

    q1,k1 = projection("c_q",True), projection("c_k",True)
    q2,k2 = projection("c_q2",True), projection("c_k2",True)
    v = projection("c_v",False)
    c = v @ state[prefix+"c_proj.weight"][:,sl].to(torch.float64).T
    inv = 1.0/(10000**(torch.arange(0,hd,2,dtype=torch.float32)/hd))
    freqs = torch.outer(torch.arange(length,dtype=torch.float32),inv)
    cos = freqs.cos().bfloat16().to(torch.float64)
    sin = freqs.sin().bfloat16().to(torch.float64)

    def rotate(z,pos):
        left,right = z.chunk(2,dim=-1)
        return torch.cat((left*cos[pos]+right*sin[pos],-left*sin[pos]+right*cos[pos]),dim=-1)

    q1p = torch.stack([rotate(q1,pos) for pos in range(length)]).numpy()
    k1p = torch.stack([rotate(k1,pos) for pos in range(length)]).numpy()
    q2p = torch.stack([rotate(q2,pos) for pos in range(length)]).numpy()
    k2p = torch.stack([rotate(k2,pos) for pos in range(length)]).numpy()
    if swap:
        u,vv,other_u,other_v = q2p,k2p,q1p,k1p
    else:
        u,vv,other_u,other_v = q1p,k1p,q2p,k2p
    b = np.einsum("iar,jbr->ijab",other_u,other_v,optimize=True)/(hd**2)
    b *= np.tril(np.ones((length,length)))[:,:,None,None]
    return u,vv,b,c.numpy()


def quotient(u,v):
    """Exact QK on selected token/position domain, refactored at visible rank."""
    length,vocab,raw_width = u.shape
    qk = u.reshape(-1,raw_width) @ v.reshape(-1,raw_width).T
    left,s,right = sla.svd(qk,full_matrices=False)
    threshold = s[0]*1e-11
    rank = int(np.sum(s>threshold))
    root = np.sqrt(s[:rank])
    uq = (left[:,:rank]*root).reshape(length,vocab,rank)
    vq = (right[:rank].T*root).reshape(length,vocab,rank)
    max_abs = float(np.max(np.abs(qk-uq.reshape(-1,rank)@vq.reshape(-1,rank).T)))
    return uq,vq,rank,max_abs


def full_head_sum_gate(state,ids,length):
    """Independent direct attention calculation against all nine token tables."""
    rng=np.random.default_rng(197)
    local=rng.integers(0,len(ids),size=(16,length))
    seq=ids[local]
    emb=state["transformer.wte.weight"][seq].to(torch.float64)
    x0=F.rms_norm(emb,(1152,))
    lambdas=state["transformer.h.0.lambdas"].to(torch.float64)
    h=F.rms_norm((lambdas[0]+lambdas[1])*x0,(1152,))
    prefix="transformer.h.0.attn."
    inv=1.0/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
    freqs=torch.outer(torch.arange(length,dtype=torch.float32),inv)
    cos=freqs.cos().bfloat16().to(torch.float64)[None,:,None,:]
    sin=freqs.sin().bfloat16().to(torch.float64)[None,:,None,:]

    def qk(name):
        z=(h@state[prefix+name+".weight"].to(torch.float64).T).reshape(16,length,9,128)
        z=F.rms_norm(z,(128,))
        left,right=z.chunk(2,dim=-1)
        return torch.cat((left*cos+right*sin,-left*sin+right*cos),dim=-1)

    q1,k1=qk("c_q"),qk("c_k")
    q2,k2=qk("c_q2"),qk("c_k2")
    s1=torch.einsum("bqhr,bkhr->bhqk",q1,k1)/128
    s2=torch.einsum("bqhr,bkhr->bhqk",q2,k2)/128
    mask=torch.tril(torch.ones(length,length,dtype=torch.bool))
    pattern=(s1*s2).masked_fill(~mask,0)
    value=(h@state[prefix+"c_v.weight"].to(torch.float64).T).reshape(16,length,9,128)
    heads=torch.einsum("bhqk,bkhr->bqhr",pattern,value).reshape(16,length,1152)
    direct=(heads@state[prefix+"c_proj.weight"].to(torch.float64).T).numpy()

    all_tables=np.zeros((length,length,len(ids),len(ids),1152))
    for head in range(9):
        u,v,b,c=selected_factors(state,ids,head,length)
        all_tables+=token_table(u,v,b,c,np.eye(128))
    reconstructed=np.zeros_like(direct)
    for n in range(len(local)):
        for i in range(length):
            for j in range(i+1):
                reconstructed[n,i]+=all_tables[i,j,local[n,i],local[n,j]]
    return float(np.max(np.abs(direct-reconstructed)))


def one_case(state, ids, head, length, ranks):
    u0,v0,b,c = selected_factors(state,ids,head,length)
    u,v,visible_rank,quotient_error = quotient(u0,v0)
    table = token_table(u,v,b,c,np.eye(visible_rank))
    original = token_table(u0,v0,b,c,np.eye(128))
    exact_table_error = float(np.max(np.abs(table-original)))
    assert exact_table_error < 1e-7,exact_table_error
    measure = pair_measure(length,np.full(len(ids),1/len(ids)))
    env = gram_environment(u,v,b,c,measure)
    _,singular,cycle,x,y = weighted_trace_gauge(env,max(ranks))
    reconstruction, boundary = wtg_boundary_check(u,v,b,c,measure,x,y,singular)
    assert reconstruction < 1e-7 and boundary < 1e-7

    su0,sv0,sb,sc = selected_factors(state,ids,head,length,swap=True)
    su,sv,swap_rank,_ = quotient(su0,sv0)
    swapped = token_table(su,sv,sb,sc,np.eye(swap_rank))
    swap_table_error = float(np.max(np.abs(table-swapped)))
    assert swap_table_error < 1e-7
    swap_env = gram_environment(su,sv,sb,sc,measure)
    _,_,swap_cycle,_,_ = weighted_trace_gauge(swap_env,max(ranks))

    rng = np.random.default_rng(2878)
    gauge = sla.expm(0.12*(rng.normal(size=(visible_rank,visible_rank))-
                            rng.normal(size=(visible_rank,visible_rank)).T))
    # Any invertible gauge should leave the function and WTG truncation fixed.
    ug = u@gauge
    vg = v@sla.inv(gauge).T
    envg = gram_environment(ug,vg,b,c,measure)
    _,sg,cyg,xg,yg = weighted_trace_gauge(envg,max(ranks))
    records = []
    for rank in ranks:
        local = local_svd(u,v,rank)
        wtg = cut_wtg(x,y,singular,rank)
        fet,trace = fet_als(env,wtg,rank,iterations=600)
        wtg_g = cut_wtg(xg,yg,sg,rank)
        gauge_diff = float(np.max(np.abs(token_table(u,v,b,c,wtg)-
                                          token_table(ug,vg,b,c,wtg_g))))
        assert gauge_diff < 1e-7
        records.append({"rank":rank,"local_relative_mse":env_error(env,local),
                        "wtg_relative_mse":env_error(env,wtg),
                        "fet_relative_mse":env_error(env,fet),
                        "wtg_gauge_max_abs":gauge_diff,
                        "fet_iterations":len(trace),
                        "fet_step_converged":len(trace)>1 and abs(trace[-1]-trace[-2])<1e-12})
    return {"head":head,"selected_tokens":ids.tolist(),"length":length,
            "visible_bond_rank":visible_rank,"quotient_qk_max_abs":quotient_error,
            "quotient_table_max_abs":exact_table_error,
            "branch_swap_table_max_abs":swap_table_error,
            "wtg_full_reconstruction_max_abs":reconstruction,
            "wtg_boundary_defect":boundary,
            "cycle_entropy_bits":cycle,"branch_swap_cycle_entropy_bits":swap_cycle,
            "cycle_entropy_gauge_abs":abs(cycle-cyg),"records":records}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--heads",type=int,nargs="+",default=[0,4])
    p.add_argument("--seeds",type=int,nargs="+",default=[17,23])
    p.add_argument("--token-count",type=int,default=8)
    p.add_argument("--length",type=int,default=4)
    p.add_argument("--ranks",type=int,nargs="+",default=[4,8,16])
    p.add_argument("--output",type=Path,default=Path(__file__).with_name("flagship_results.json"))
    args=p.parse_args()
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",weights_only=True,mmap=True)
    vocab=state["transformer.wte.weight"].shape[0]
    cases=[]
    for seed in args.seeds:
        ids=np.random.default_rng(seed).choice(vocab,args.token_count,replace=False)
        direct_gate=full_head_sum_gate(state,ids,args.length)
        assert direct_gate<1e-9,direct_gate
        for head in args.heads:
            result=one_case(state,ids,head,args.length,args.ranks)
            result["seed"]=seed
            result["full_head_sum_direct_max_abs"]=direct_gate
            cases.append(result)
            print("case",seed,head,result["visible_bond_rank"],result["cycle_entropy_bits"],
                  result["branch_swap_cycle_entropy_bits"],result["records"],flush=True)
    digest=hashlib.sha256()
    with (SNAPSHOT/"pytorch_model.bin").open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            digest.update(chunk)
    data={"checkpoint":str(SNAPSHOT/"pytorch_model.bin"),
          "checkpoint_sha256":digest.hexdigest(),
          "scope":"selected-token, first-layer, selected-head quotient; CPU-only", "cases":cases}
    args.output.write_text(json.dumps(data,indent=2)+"\n")


if __name__=="__main__":
    main()
