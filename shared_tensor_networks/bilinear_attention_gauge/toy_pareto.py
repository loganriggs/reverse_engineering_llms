"""Sweep legibility priors under a matched functional-error allowance."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from toy_factorial import (D, V, environment, fit, metric, relative_loss,
                           rule_recall, table, toy, weighted_trace_gauge)


OBJECTIVES=("fidelity","native_l1","canonical_l1","pair_l1","intervention")
MEASURES=("uniform","usage","hybrid")
LAMBDAS=(0.,.005,.02,.08,.2,.5,1.)


def outcomes(head,x,measure):
    _,_,gx,gy=weighted_trace_gauge(environment(head,measure),1)
    response=table(head,x)
    energy=np.sum(response**2,axis=-1)
    # Hill number of order two, uniform over all possible query-key pairs.
    support=float(np.sum(energy)**2/max(np.sum(energy**2),1e-20))
    mask=np.zeros((V,V),dtype=bool)
    for which in ("common","rare"):
        mask[np.ix_(head[which+"_q"],head[which+"_k"])]=True
    return dict(train_mse=relative_loss(head,x,measure),
                uniform_mse=relative_loss(head,x,metric("uniform")),
                usage_mse=relative_loss(head,x,metric("usage")),
                intervention_uniform_mse=relative_loss(head,x,metric("uniform"),True),
                common_rule_mse=rule_recall(head,x,"common"),
                rare_rule_mse=rule_recall(head,x,"rare"),
                effective_pair_support=support,
                rule_energy_fraction=float(np.sum(energy[mask])/max(np.sum(energy),1e-20)),
                native_bond_l1=float(np.sum(np.abs(x))/D),
                canonical_bond_l1=float(np.sum(np.abs(gx@x@gy))/
                                        max(np.sum(np.abs(gx@gy)),1e-20)))


def summarize(selected):
    summary=[]
    grouped=defaultdict(list)
    for r in selected:
        grouped[(r["measure"],r["objective"],r["target"])].append(r)
    for (measure_name,objective,target),rs in grouped.items():
        summary.append(dict(measure=measure_name,objective=objective,target=target,
                            mean_target_value=float(np.mean([r["target_value"] for r in rs])),
                            mean_train_mse=float(np.mean([r["train_mse"] for r in rs])),
                            mean_common_rule_mse=float(np.mean([r["common_rule_mse"] for r in rs])),
                            mean_rare_rule_mse=float(np.mean([r["rare_rule_mse"] for r in rs]))))
    return summary


def run(seeds=(0,1,2), allowance=.02):
    records=[]
    selected=[]
    for seed in seeds:
        heads=toy(seed)
        for measure_name in MEASURES:
            measure=metric(measure_name)
            for head_id,head in enumerate(heads):
                baseline_x,_,baseline_info=fit(head,measure,"fidelity",seed,lam=0.)
                baseline=outcomes(head,baseline_x,measure)["train_mse"]
                for objective in OBJECTIVES:
                    candidates=[]
                    for lam in ((0.,) if objective=="fidelity" else LAMBDAS):
                        if lam==0.:
                            x=baseline_x
                            fit_info=baseline_info
                        else:
                            x,_,fit_info=fit(head,measure,objective,seed,lam=lam)
                        rec=dict(seed=seed,measure=measure_name,head=head_id,
                                 objective=objective,lambda_=lam,
                                 baseline_train_mse=baseline,
                                 **outcomes(head,x,measure),
                                 fit_gradient_norm=fit_info["gradient_norm"],
                                 fit_converged=fit_info["success"])
                        records.append(rec)
                        candidates.append(rec)
                    feasible=[r for r in candidates if r["train_mse"]<=baseline+allowance
                              and r["fit_gradient_norm"]<1e-4]
                    assert feasible,(seed,measure_name,head_id,objective)
                    for target in ("effective_pair_support","intervention_uniform_mse",
                                   "canonical_bond_l1"):
                        best=min(feasible,key=lambda r:r[target])
                        selected.append(dict(seed=seed,measure=measure_name,head=head_id,
                                             objective=objective,target=target,
                                             lambda_=best["lambda_"],
                                             train_mse=best["train_mse"],
                                             target_value=best[target],
                                             common_rule_mse=best["common_rule_mse"],
                                             rare_rule_mse=best["rare_rule_mse"]))
    return dict(scope="known-rule toy; per-head rank one; constraint is baseline MSE + allowance",
                seeds=list(seeds),allowance=allowance,lambdas=LAMBDAS,
                records=records,selected=selected,summary=summarize(selected))


def merge(paths):
    parts=[json.loads(path.read_text()) for path in paths]
    assert len({p["allowance"] for p in parts})==1
    seeds=[s for p in parts for s in p["seeds"]]
    assert len(set(seeds))==len(seeds),"Duplicate seed in merge"
    records=[r for p in parts for r in p["records"]]
    selected=[r for p in parts for r in p["selected"]]
    return dict(scope=parts[0]["scope"],seeds=seeds,
                allowance=parts[0]["allowance"],lambdas=LAMBDAS,
                records=records,selected=selected,summary=summarize(selected))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--merge",type=Path,nargs="+")
    p.add_argument("--allowance",type=float,default=.02)
    p.add_argument("--output",type=Path,default=Path(__file__).with_name("toy_pareto.json"))
    args=p.parse_args()
    result=merge(args.merge) if args.merge else run(args.seeds,args.allowance)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    for row in result["summary"]:
        if row["target"]=="effective_pair_support":
            print(row,flush=True)
