"""Exact pair extraction and rank-one canonical reduction for dual-output toy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from analyze_toy_language import (KEY_IDS,QUERY_IDS,env_pair,env_sequence,
                                  pair_measure,pair_table,rms)
from run import fet_als
from toy_factorial import weighted_trace_gauge
from train_dual_output_language import DualOutputAttention,batch,exhaustive


def factors(model,head):
    e=model.embed.weight.detach().numpy()
    hd=model.qk_width
    sl=slice(head*hd,(head+1)*hd)
    def qk(name,ids):
        return rms(e[ids]@getattr(model,name).weight.detach().numpy()[sl].T)
    u=qk("q1",QUERY_IDS)/hd
    v=qk("k1",KEY_IDS)
    b=qk("q2",QUERY_IDS)@qk("k2",KEY_IDS).T/hd
    value=e[KEY_IDS]@model.v.weight.detach().numpy()[head]
    c=value[:,None]*model.read.weight.detach().numpy()[:,head][None,:]
    score=u@v.T
    left,s,right=np.linalg.svd(score,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10))
    root=np.sqrt(s[:rank])
    uq,vq=left[:,:rank]*root,right[:rank].T*root
    assert np.max(np.abs(score-uq@vq.T))<1e-12
    return dict(u=uq,v=vq,b=b,c=c,rank=rank,physical_width=hd)


def logits(tables,seq,bias):
    q=seq[:,-1]-6
    k=seq[:,:-1]
    return bias+sum(np.sum(t[q[:,None],k],axis=1) for t in tables)


def score(tables,seq,targets,bias,attrs=None):
    z=logits(tables,seq,bias)
    p=z>0
    y=targets.astype(bool)
    ce=F.binary_cross_entropy_with_logits(torch.from_numpy(z),
                                          torch.from_numpy(targets),
                                          reduction="none").numpy().mean(axis=1)
    joint=np.all(p==y,axis=1)
    result=dict(joint_accuracy=float(np.mean(joint)),
                bit_accuracy=float(np.mean(p==y)),bce=float(np.mean(ce)))
    if attrs is not None:
        for name,mask in (
            ("common",np.array([not a["swap"] for a in attrs])),
            ("rare_seen",np.array([bool(a["swap"]) and not a["heldout"]
                                    for a in attrs],dtype=bool)),
            ("rare_heldout",np.array([a["heldout"] for a in attrs]))):
            assert mask.dtype==np.bool_,name
            result[name]=dict(n=int(mask.sum()),joint_accuracy=float(np.mean(joint[mask])),
                              bit_accuracy=float(np.mean((p==y)[mask])),
                              bce=float(np.mean(ce[mask])))
    return result,z


def one_seed(seed,n_fit=100000,n_val=50000):
    torch.set_num_threads(1)
    model=DualOutputAttention().double()
    path=Path(__file__).with_name(f"dual_output_seed{seed}.pt")
    model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    model.eval()
    heads=[factors(model,h) for h in range(2)]
    assert [h["rank"] for h in heads]==[2,2]
    original=[pair_table(h,np.eye(h["rank"])) for h in heads]
    bias=model.read.bias.detach().numpy()
    seq_t,targets_t,attrs=exhaustive()
    seq,targets=seq_t.numpy(),targets_t.numpy().astype(np.float64)
    direct=model(seq_t).detach().numpy()
    gate=float(np.max(np.abs(direct-logits(original,seq,bias))))
    assert gate<1e-10,gate
    fit_seq,_=batch(n_fit,torch.Generator().manual_seed(100000+seed))
    check_seq,_=batch(n_fit,torch.Generator().manual_seed(300000+seed))
    val_seq,val_targets=batch(n_val,torch.Generator().manual_seed(200000+seed))
    fit_seq,check_seq=fit_seq.numpy(),check_seq.numpy()
    val_seq,val_targets=val_seq.numpy(),val_targets.numpy().astype(np.float64)
    measures={"uniform_pair":np.ones((2,6))/12,
              "empirical_pair":pair_measure(fit_seq)}
    envs={"uniform_pair":[env_pair(h,measures["uniform_pair"]) for h in heads],
          "empirical_pair":[env_pair(h,measures["empirical_pair"]) for h in heads],
          "empirical_sequence":[env_sequence(h,fit_seq) for h in heads]}
    def normalized(e,rank):
        ident=np.eye(rank).ravel()
        return e/(ident@e@ident)
    envs["hybrid_sequence_uniform"]=[
        .5*normalized(envs["empirical_sequence"][i],h["rank"])+
        .5*normalized(envs["uniform_pair"][i],h["rank"])
        for i,h in enumerate(heads)]
    envs["sequence_plus_qk2_intervention"]=[
        .5*normalized(envs["empirical_sequence"][i],h["rank"])+
        .5*normalized(env_pair(dict(h,b=np.ones_like(h["b"])),measures["uniform_pair"]),h["rank"])
        for i,h in enumerate(heads)]
    env_comparison=[]
    for h in range(2):
        ident=np.eye(2).ravel()
        def normalized(e): return e/(ident@e@ident)
        pair_e=normalized(envs["empirical_pair"][h])
        seq_e=normalized(envs["empirical_sequence"][h])
        pair_check=normalized(env_pair(heads[h],pair_measure(check_seq)))
        seq_check=normalized(env_sequence(heads[h],check_seq))
        env_comparison.append(dict(head=h,
            sequence_vs_pair=float(np.linalg.norm(seq_e-pair_e)/np.linalg.norm(pair_e)),
            pair_sampling=float(np.linalg.norm(pair_check-pair_e)/np.linalg.norm(pair_e)),
            sequence_sampling=float(np.linalg.norm(seq_check-seq_e)/np.linalg.norm(seq_e))))
    baseline,original_logits=score(original,seq,targets,bias,attrs)
    baseline_val,original_val_logits=score(original,val_seq,val_targets,bias)
    ablations=[]
    for removed in range(2):
        modified=[t.copy() for t in original]
        modified[removed]=np.zeros_like(modified[removed])
        ablation_ex,_=score(modified,seq,targets,bias,attrs)
        ablation_val,_=score(modified,val_seq,val_targets,bias)
        ablations.append(dict(removed_head=removed,exhaustive=ablation_ex,
                              validation=ablation_val))
    records=[]
    for name,envlist in envs.items():
        for method in ("wtg","fet"):
            xs=[]
            traces=[]
            for h,env in zip(heads,envlist):
                wtg,_,_,_=weighted_trace_gauge(env,1)
                if method=="wtg":
                    x=wtg;trace=[]
                else:
                    x,trace=fet_als(env,wtg,1,iterations=300)
                xs.append(x)
                traces.append(trace)
            tables=[pair_table(h,x) for h,x in zip(heads,xs)]
            exhaustive_score,ex_logits=score(tables,seq,targets,bias,attrs)
            val_score,val_logits=score(tables,val_seq,val_targets,bias)
            # Replacing QK2 by 1 probes transfer outside the fitted network.
            # This is a functional intervention, not a claim that the model
            # would retain its task accuracy after that replacement.
            cf_error=[]
            for h,x in zip(heads,xs):
                cf=dict(h,b=np.ones_like(h["b"]))
                target=pair_table(cf,np.eye(h["rank"]))
                approx=pair_table(cf,x)
                cf_error.append(float(np.mean((target-approx)**2)/np.mean(target**2)))
            records.append(dict(measure=name,method=method,
                exhaustive=exhaustive_score,validation=val_score,
                exhaustive_relative_logit_mse=float(np.mean((ex_logits-original_logits)**2)/
                                                    np.mean(original_logits**2)),
                validation_relative_logit_mse=float(np.mean((val_logits-original_val_logits)**2)/
                                                    np.mean(original_val_logits**2)),
                qk2_constant_intervention_pair_mse=cf_error,
                fet_iterations=[len(t) for t in traces]))
    return dict(seed=seed,n_fit=n_fit,n_val=n_val,direct_table_max_abs=gate,
                head_quotient_ranks=[h["rank"] for h in heads],
                baseline_exhaustive=baseline,baseline_validation=baseline_val,
                head_ablations=ablations,environment_comparison=env_comparison,
                records=records)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--n-fit",type=int,default=100000)
    p.add_argument("--n-val",type=int,default=50000)
    args=p.parse_args()
    result=one_seed(args.seed,args.n_fit,args.n_val)
    path=Path(__file__).with_name(f"dual_output_analysis_seed{args.seed}.json")
    path.write_text(json.dumps(result,indent=2)+"\n")
    for row in result["records"]:
        print(args.seed,row["measure"],row["method"],
              row["exhaustive"]["joint_accuracy"],
              row["exhaustive"]["rare_heldout"]["joint_accuracy"],
              row["validation"]["bce"],flush=True)
