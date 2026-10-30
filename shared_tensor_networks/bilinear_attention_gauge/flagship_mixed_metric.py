"""Predeclared mixtures of long-context uniform, corpus, and QK2 interventions.

Each component is normalized by its own original pair-response energy before
mixing, so coefficients weight relative squared errors rather than raw scale.
The rank-32/64 WTG cuts are evaluated on independent sampled fit-law pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from matrix_free_environment import SampledBondEnvironment


HERE=Path(__file__).parent
ART=HERE/"full_width_results"
RANKS=(32,64)
MIXTURES={
    "uniform_corpus_half":{"uniform":.5,"corpus":.5},
    "uniform_corpus_intervention_thirds":{
        "uniform":1/3,"corpus":1/3,"uniform_qk2_constant":1/3}}


def component(part,weight_kind="natural"):
    u,v,w,_,c_norm=part
    raw=w if weight_kind=="natural" else c_norm
    target=np.sum(u*v,axis=1)
    denominator=float(np.sum(raw*target**2))
    assert denominator>0
    return dict(u=u,v=v,raw=raw,denominator=denominator,
        environment=SampledBondEnvironment(u,v,raw/raw.sum()))


def mixture(components,coefficients):
    assert abs(sum(coefficients.values())-1)<1e-12
    u=np.concatenate([components[name]["u"] for name in coefficients])
    v=np.concatenate([components[name]["v"] for name in coefficients])
    w=np.concatenate([alpha*components[name]["raw"]/
                      components[name]["denominator"]
                      for name,alpha in coefficients.items()])
    result=SampledBondEnvironment(u,v,w)
    probe=np.eye(result.d)*.8+np.random.default_rng(991).normal(
        scale=.02,size=(result.d,result.d))
    explicit=sum(alpha*components[name]["environment"].relative_error(probe)
                 for name,alpha in coefficients.items())
    identity_check=float(abs(result.relative_error(probe)-explicit))
    assert identity_check<1e-12,identity_check
    return result,identity_check


def run(n_fit=65536,n_val=65536,seed=17):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    fit={};validation={};coverage={}
    for law in ("uniform","corpus"):
        train,val,info=rows(state,0,seed,n_fit,n_val,128,
            distribution=law,corpus=corpus if law=="corpus" else None)
        fit[law]=component(train)
        validation[law]=component(val)
        coverage[law]=info
        if law=="uniform":
            fit["uniform_qk2_constant"]=component(train,"constant")
            validation["uniform_qk2_constant"]=component(val,"constant")
    variants={}
    checks={}
    for name,coefficients in MIXTURES.items():
        train_env,fit_check=mixture(fit,coefficients)
        val_env,val_check=mixture(validation,coefficients)
        checks[name]=dict(fit=fit_check,validation=val_check)
        matrices={};details={}
        for rank in RANKS:
            cut,info=train_env.weighted_trace_gauge(rank,max_iter=100,tol=1e-8)
            matrices[rank]=cut
            details[rank]=dict(
                train_mixed_error=train_env.relative_error(cut),
                validation_mixed_error=val_env.relative_error(cut),
                train_components={key:part["environment"].relative_error(cut)
                                  for key,part in fit.items()},
                validation_components={key:part["environment"].relative_error(cut)
                                       for key,part in validation.items()},
                boundary=dict(left=info["left"],right=info["right"],
                              left_condition=info["left_condition"],
                              right_condition=info["right_condition"]))
        variants[name]=dict(coefficients=coefficients,details=details)
        np.savez_compressed(ART/f"head0_l128_{name}.npz",
                            **{f"rank{rank}":x for rank,x in matrices.items()})
    baselines={}
    for law in ("uniform","corpus"):
        suffix="_corpus" if law=="corpus" else ""
        path=ART/f"head0_seed17_n65536{suffix}_l128.npz"
        arrays=np.load(path)
        for rank in RANKS:
            x=arrays[f"wtg_rank{rank}"]
            baselines[f"{law}_rank{rank}"]={
                key:part["environment"].relative_error(x)
                for key,part in validation.items()}
    output=ART/"head0_l128_mixed_metric.json"
    result=dict(head=0,length=128,ranks=list(RANKS),fit_seed=seed,
        n_fit_per_law=n_fit,n_validation_per_law=n_val,
        corpus_fit_documents=[0,15000],corpus_validation_documents=[15000,20000],
        normalization="each component original response energy normalized to 1",
        identity_checks=checks,coverage=coverage,
        variants=variants,baseline_validation_components=baselines,
        scope="first-layer head pair contribution; no later-layer logits or CE")
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v["details"] for k,v in variants.items()},indent=2))
    return result


if __name__=="__main__":run()
