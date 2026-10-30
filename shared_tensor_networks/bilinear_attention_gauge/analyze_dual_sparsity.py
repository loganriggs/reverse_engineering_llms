"""Metric x legibility-prior comparison on learned dual-output heads.

Penalty strength is selected by an unsupervised per-head function-error
allowance. Exhaustive task labels and held-out combinations are scored only
after selection. This is still a small toy, not model-scale evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_dual_output_language import factors,logits,score
from analyze_toy_language import pair_measure,pair_table
from toy_factorial import fit,environment,weighted_trace_gauge
from train_dual_output_language import DualOutputAttention,batch,exhaustive


OBJECTIVES=("native_l1","canonical_l1","pair_l1","intervention")
LAMBDAS=(0.,.005,.02,.08,.2,.5,1.)


def prior_value(head,x,measure,objective):
    if objective=="native_l1":
        return float(np.sum(np.abs(x))/2)
    if objective=="canonical_l1":
        _,_,gx,gy=weighted_trace_gauge(environment(head,measure),1)
        return float(np.sum(np.abs(gx@x@gy))/np.sum(np.abs(gx@gy)))
    if objective=="pair_l1":
        t=pair_table(head,x)
        original=pair_table(head,np.eye(2))
        normalizer=np.sum(measure[:,:,None]*original**2)
        return float(np.sum(measure*np.linalg.norm(t,axis=-1))/np.sqrt(normalizer))
    if objective=="intervention":
        cf=dict(head,b=np.ones_like(head["b"]))
        target=pair_table(cf,np.eye(2));approx=pair_table(cf,x)
        return float(np.sum(measure[:,:,None]*(target-approx)**2)/
                     np.sum(measure[:,:,None]*target**2))
    raise ValueError(objective)


def head_outcomes(head,x,measure):
    original=pair_table(head,np.eye(2))
    approx=pair_table(head,x)
    mse=float(np.sum(measure[:,:,None]*(original-approx)**2)/
              np.sum(measure[:,:,None]*original**2))
    cf=dict(head,b=np.ones_like(head["b"]))
    cf_target=pair_table(cf,np.eye(2));cf_approx=pair_table(cf,x)
    cf_mse=float(np.mean((cf_target-cf_approx)**2)/np.mean(cf_target**2))
    energy=np.sum(approx**2,axis=-1)
    support=float(np.sum(energy)**2/max(np.sum(energy**2),1e-20))
    return dict(fit_pair_mse=mse,qk2_constant_intervention_mse=cf_mse,
                effective_pair_support=support)


def one_seed(seed,n_fit=100000,n_val=50000,allowance=.02):
    torch.set_num_threads(1)
    model=DualOutputAttention().double()
    model.load_state_dict(torch.load(Path(__file__).with_name(f"dual_output_seed{seed}.pt"),
                                     map_location="cpu",weights_only=True))
    model.eval()
    heads=[factors(model,h) for h in range(2)]
    bias=model.read.bias.detach().numpy()
    seq_t,targets_t,attrs=exhaustive()
    seq,targets=seq_t.numpy(),targets_t.numpy().astype(np.float64)
    val_t,val_y=batch(n_val,torch.Generator().manual_seed(700000+seed))
    val,val_y=val_t.numpy(),val_y.numpy().astype(np.float64)
    fit_seq,_=batch(n_fit,torch.Generator().manual_seed(100000+seed))
    measures={"uniform_pair":np.ones((2,6))/12,
              "empirical_pair":pair_measure(fit_seq.numpy())}
    original=[pair_table(h,np.eye(2)) for h in heads]
    baseline_ex,_=score(original,seq,targets,bias,attrs)
    baseline_val,_=score(original,val,val_y,bias)
    records=[]
    selected=[]
    for measure_name,measure in measures.items():
        baselines=[]
        for h,head in enumerate(heads):
            bx,_,binfo=fit(head,measure,"fidelity",seed,lam=0.)
            baselines.append(dict(x=bx,mse=head_outcomes(head,bx,measure)["fit_pair_mse"],
                                  gradient_norm=binfo["gradient_norm"]))
        for objective in OBJECTIVES:
            xs=[]
            chosen=[]
            for h,head in enumerate(heads):
                candidates=[]
                for lam in LAMBDAS:
                    if lam==0.:
                        x=baselines[h]["x"]
                        grad=baselines[h]["gradient_norm"]
                    else:
                        x,_,info=fit(head,measure,objective,seed,lam=lam)
                        grad=info["gradient_norm"]
                    outcome=head_outcomes(head,x,measure)
                    prior=prior_value(head,x,measure,objective)
                    row=dict(seed=seed,measure=measure_name,objective=objective,
                             head=h,lambda_=lam,prior=prior,
                             gradient_norm=grad,baseline_fit_mse=baselines[h]["mse"],
                             **outcome)
                    records.append(row)
                    if grad<1e-4 and outcome["fit_pair_mse"]<=baselines[h]["mse"]+allowance:
                        candidates.append((prior,lam,x,row))
                assert candidates,(seed,measure_name,objective,h)
                _,lam,x,row=min(candidates,key=lambda item:item[0])
                xs.append(x)
                chosen.append(dict(head=h,lambda_=lam,**{
                    k:row[k] for k in ("fit_pair_mse","prior","qk2_constant_intervention_mse",
                                       "effective_pair_support")}))
            tables=[pair_table(head,x) for head,x in zip(heads,xs)]
            ex,_=score(tables,seq,targets,bias,attrs)
            validation,_=score(tables,val,val_y,bias)
            selected.append(dict(seed=seed,measure=measure_name,objective=objective,
                                 heads=chosen,exhaustive=ex,validation=validation))
    return dict(seed=seed,n_fit=n_fit,n_val=n_val,allowance=allowance,
                baseline_exhaustive=baseline_ex,baseline_validation=baseline_val,
                records=records,selected=selected)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--n-fit",type=int,default=100000)
    p.add_argument("--n-val",type=int,default=50000)
    p.add_argument("--allowance",type=float,default=.02)
    args=p.parse_args()
    result=one_seed(args.seed,args.n_fit,args.n_val,args.allowance)
    Path(__file__).with_name(f"dual_sparsity_seed{args.seed}.json").write_text(
        json.dumps(result,indent=2)+"\n")
    for row in result["selected"]:
        print(args.seed,row["measure"],row["objective"],
              [h["lambda_"] for h in row["heads"]],
              row["exhaustive"]["joint_accuracy"],
              row["exhaustive"]["rare_heldout"]["joint_accuracy"],flush=True)
