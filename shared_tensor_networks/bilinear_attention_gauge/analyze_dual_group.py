"""Weakly supervised group priors on trained dual-output heads.

The A/B/filler token-family partition is supplied by the toy generator.
No output-bit labels or held-out-combination labels enter fitting/selection.
Compare global-family and query-conditional-family groups against pair-response
sparsity using the same optimizer and reconstruction-error budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch

from analyze_dual_output_language import factors,score
from analyze_toy_language import pair_measure,pair_table
from toy_factorial import environment,weighted_trace_gauge
from train_dual_output_language import DualOutputAttention,batch,exhaustive


GROUPS=(np.array([0,1]),np.array([2,3]),np.array([4,5]))
LAMBDAS=(0.,.001,.005,.02,.08,.2,.5,1.)


def prior_and_gradient(head,x,objective):
    table=pair_table(head,x)
    original=pair_table(head,np.eye(2))
    scale=np.sqrt(np.sum(original**2))
    grad_t=np.zeros_like(table)
    if objective=="pair_l1":
        magnitude=np.sqrt(np.sum(table**2,axis=-1)+1e-8)
        value=float(np.sum(magnitude)/scale)
        grad_t=table/magnitude[:,:,None]/scale
    elif objective=="family_group":
        value=0.
        for ids in GROUPS:
            magnitude=np.sqrt(np.sum(table[:,ids,:]**2)+1e-8)
            value+=magnitude/scale
            grad_t[:,ids,:]=table[:,ids,:]/magnitude/scale
    elif objective=="query_family_group":
        value=0.
        for query in range(2):
            for ids in GROUPS:
                magnitude=np.sqrt(np.sum(table[query,ids,:]**2)+1e-8)
                value+=magnitude/scale
                grad_t[query,ids,:]=table[query,ids,:]/magnitude/scale
    else:
        raise ValueError(objective)
    response=head["b"][:,:,None]*head["c"][None,:,:]
    score_grad=np.sum(grad_t*response,axis=-1)
    grad_x=head["u"].T@score_grad@head["v"]
    return float(value),grad_x


def solve(head,measure,objective,lam,seed,max_iter=500):
    env=environment(head,measure)
    ident=np.eye(2)
    original=ident.ravel()
    denom=float(original@env@original)
    wtg,_,_,_=weighted_trace_gauge(env,1)
    rng=np.random.default_rng(seed)
    def factorize(x):
        u,s,vh=np.linalg.svd(x)
        return np.r_[u[:,0]*np.sqrt(s[0]),vh[0]*np.sqrt(s[0])]
    starts=[factorize(wtg),factorize(ident)]+[rng.normal(size=4) for _ in range(2)]
    def objective_fn(z):
        a,b=z[:2],z[2:]
        x=np.outer(a,b)
        delta=(x-ident).ravel()
        fidelity=float(delta@env@delta/denom)
        grad=(2*env@delta/denom).reshape(2,2)
        prior,pg=prior_and_gradient(head,x,objective)
        grad+=lam*pg
        return fidelity+lam*prior,np.r_[grad@b,grad.T@a]
    best=None
    for z0 in starts:
        result=minimize(objective_fn,z0,method="L-BFGS-B",jac=True,
                        options=dict(maxiter=max_iter,ftol=1e-13,gtol=1e-8,maxls=40))
        if best is None or result.fun<best.fun:best=result
    x=np.outer(best.x[:2],best.x[2:])
    delta=(x-ident).ravel()
    mse=float(delta@env@delta/denom)
    prior=prior_and_gradient(head,x,objective)[0]
    return x,dict(mse=mse,prior=prior,objective_loss=float(best.fun),
                  converged=bool(best.success),iterations=int(best.nit),
                  gradient_norm=float(np.linalg.norm(best.jac)))


def check_gradient():
    rng=np.random.default_rng(17)
    h=dict(u=rng.normal(size=(2,2)),v=rng.normal(size=(6,2)),
           b=rng.normal(size=(2,6)),c=rng.normal(size=(6,2)))
    x=rng.normal(size=(2,2))
    errs={}
    for name in ("pair_l1","family_group","query_family_group"):
        analytic=prior_and_gradient(h,x,name)[1]
        numeric=np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                d=np.zeros((2,2));d[i,j]=1e-6
                numeric[i,j]=(prior_and_gradient(h,x+d,name)[0]-
                              prior_and_gradient(h,x-d,name)[0])/(2e-6)
        errs[name]=float(np.max(np.abs(analytic-numeric)))
        assert errs[name]<1e-7,errs
    return errs


def head_diagnostics(head,x):
    original=pair_table(head,np.eye(2))
    reduced=pair_table(head,x)
    energy=np.array([np.sum(reduced[:,g,:]**2) for g in GROUPS])
    shares=energy/energy.sum()
    conditional=np.array([[np.sum(reduced[q,g,:]**2) for g in GROUPS]
                          for q in range(2)])
    conditional_support=np.sum(np.sqrt(conditional),axis=1)**2/np.sum(conditional,axis=1)
    effects=[]
    for ids in GROUPS:
        a,b=original[:,ids,:],reduced[:,ids,:]
        effects.append(float(np.sum((a-b)**2)/max(np.sum(a*a),1e-20)))
    counterfactual=dict(head,b=np.ones_like(head["b"]))
    cf_a=pair_table(counterfactual,np.eye(2))
    cf_b=pair_table(counterfactual,x)
    shifted=dict(head,b=np.roll(head["b"],1,axis=1))
    shift_a=pair_table(shifted,np.eye(2))
    shift_b=pair_table(shifted,x)
    flip_effects=[]
    flip_energies=[]
    flip_errors=[]
    for ids in GROUPS:
        a=original[:,ids[1],:]-original[:,ids[0],:]
        b=reduced[:,ids[1],:]-reduced[:,ids[0],:]
        effect_energy=float(np.sum(a*a));error=float(np.sum((a-b)**2))
        flip_energies.append(effect_energy);flip_errors.append(error)
        flip_effects.append(error/max(effect_energy,1e-20))
    query_a=original[1]-original[0]
    query_b=reduced[1]-reduced[0]
    query_energy=float(np.sum(query_a**2))
    query_error=float(np.sum((query_a-query_b)**2))
    return dict(family_energy_share=shares.tolist(),
                effective_family_support=float(np.sum(np.sqrt(energy))**2/energy.sum()),
                mean_query_conditional_family_support=float(np.mean(conditional_support)),
                family_removal_effect_mse=effects,
                bit_flip_effect_mse=flip_effects,
                bit_flip_effect_energy=flip_energies,
                bit_flip_error_energy=flip_errors,
                query_switch_effect_mse=query_error/max(query_energy,1e-20),
                query_switch_effect_energy=query_energy,
                query_switch_error_energy=query_error,
                qk2_constant_mse=float(np.sum((cf_a-cf_b)**2)/np.sum(cf_a**2)),
                qk2_key_shift_mse=float(np.sum((shift_a-shift_b)**2)/np.sum(shift_a**2)))


def run_seed(seed,allowance=.02,n_fit=100000,n_val=50000):
    torch.set_num_threads(1)
    model=DualOutputAttention().double()
    model.load_state_dict(torch.load(Path(__file__).with_name(f"dual_output_seed{seed}.pt"),
                                     map_location="cpu",weights_only=True))
    model.eval()
    heads=[factors(model,h) for h in range(2)]
    bias=model.read.bias.detach().numpy()
    seq_t,targets_t,attrs=exhaustive()
    seq,targets=seq_t.numpy(),targets_t.numpy().astype(np.float64)
    fit_seq,_=batch(n_fit,torch.Generator().manual_seed(100000+seed))
    val_t,val_y=batch(n_val,torch.Generator().manual_seed(700000+seed))
    val,val_y=val_t.numpy(),val_y.numpy().astype(np.float64)
    measures={"uniform_pair":np.ones((2,6))/12,
              "empirical_pair":pair_measure(fit_seq.numpy())}
    records=[];selected=[]
    for measure_name,measure in measures.items():
        for objective in ("pair_l1","family_group","query_family_group"):
            chosen=[];tables=[]
            for h,head in enumerate(heads):
                candidates=[]
                baseline=None
                for lam in LAMBDAS:
                    x,info=solve(head,measure,objective,lam,seed*1000+h)
                    row=dict(seed=seed,measure=measure_name,objective=objective,
                             head=h,lambda_=lam,**info,**head_diagnostics(head,x))
                    records.append(row)
                    if lam==0.:baseline=row["mse"]
                    if row["mse"]<=baseline+allowance and row["gradient_norm"]<1e-4:
                        candidates.append((row["prior"],x,row))
                assert candidates,(seed,measure_name,objective,h)
                _,x,row=min(candidates,key=lambda c:c[0])
                chosen.append(row)
                tables.append(pair_table(head,x))
            ex,_=score(tables,seq,targets,bias,attrs)
            validation,_=score(tables,val,val_y,bias)
            selected.append(dict(seed=seed,measure=measure_name,objective=objective,
                                 heads=chosen,exhaustive=ex,validation=validation))
    return dict(seed=seed,allowance=allowance,n_fit=n_fit,n_val=n_val,
                gradient_checks=check_gradient(),records=records,selected=selected)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--allowance",type=float,default=.02)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=run_seed(args.seed,args.allowance)
    out=args.output or Path(__file__).with_name(f"dual_group_seed{args.seed}.json")
    out.write_text(json.dumps(result,indent=2)+"\n")
    for row in result["selected"]:
        print(args.seed,row["measure"],row["objective"],
              [h["lambda_"] for h in row["heads"]],
              row["exhaustive"]["joint_accuracy"],
              row["exhaustive"]["rare_heldout"]["joint_accuracy"],flush=True)
