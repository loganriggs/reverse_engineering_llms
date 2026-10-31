"""Uniform, corpus, and mixed WTG cuts in pre-/post-RoPE coordinates."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_matched_rope_fit import LAWS,prepare
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART
from matrix_free_environment import SampledBondEnvironment


PRIORS={"uniform":0.0,"mixed":0.5,"corpus":1.0}


def environment(datasets,geometry,split,corpus_weight):
    arrays=[]
    for law,weight in (("uniform",1-corpus_weight),("corpus",corpus_weight)):
        if weight==0:
            continue
        (u,v,w,_),energy=datasets[geometry][split][law]
        arrays.append((u.numpy(),v.numpy(),
            w.numpy()*weight/(len(w)*energy)))
    return SampledBondEnvironment(
        np.concatenate([part[0] for part in arrays]),
        np.concatenate([part[1] for part in arrays]),
        np.concatenate([part[2] for part in arrays]))


def run(seed=41,n_train=32768,n_val=8192):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    datasets,coverage,gate=prepare(state,seed,n_train,n_val)
    cuts={};records=[]
    validation={geometry:{law:environment(datasets,geometry,"val",w)
                          for law,w in (("uniform",0.0),("corpus",1.0))}
                for geometry in ("post","pre")}
    for prior,corpus_weight in PRIORS.items():
        for geometry in ("post","pre"):
            train=environment(datasets,geometry,"train",corpus_weight)
            cut,info=train.weighted_trace_gauge(96,max_iter=100,tol=1e-7)
            key=f"{prior}__{geometry}"
            cuts[key]=cut
            singular=np.linalg.svd(cut,compute_uv=False)
            records.append(dict(prior=prior,geometry=geometry,rank=96,
                numerical_s96=float(singular[95]),
                numerical_s97=float(singular[96]),
                train_relative_pair_mse=train.relative_error(cut),
                validation_relative_pair_mse={
                    law:env.relative_error(cut)
                    for law,env in validation[geometry].items()},
                left_eigen_residual=info["left"]["eigen_residual"],
                right_eigen_residual=info["right"]["eigen_residual"],
                left_condition=info["left_condition"],
                right_condition=info["right_condition"]))
            print(seed,prior,geometry,records[-1]["validation_relative_pair_mse"],
                  flush=True)
    prefix=f"head0_rope_metric_factorial_seed{seed}"
    np.savez_compressed(ART/f"{prefix}.npz",**cuts)
    result=dict(layer=0,head=0,rank=96,seed=seed,
        priors=PRIORS,n_train_pairs_per_law=n_train,
        n_validation_pairs_per_law=n_val,
        heldout_relative_gaps=[32,48],
        fit_corpus_documents=[0,15000],
        selection_corpus_documents=[15000,17500],
        pre_post_identity_gate=gate,coverage=coverage,
        objective="weighted sum of per-law normalized QK1 pair score MSE, with QK2 and OV held exact",
        uniform_definition="uniform-vocabulary Monte Carlo over token IDs and causal cross positions; no natural-text data in the fit law",
        limitation="sampled uniform measure is not an exact full-vocabulary contraction",
        records=records,arrays=f"{prefix}.npz")
    (ART/f"{prefix}.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--n-train",type=int,default=32768)
    p.add_argument("--n-val",type=int,default=8192)
    a=p.parse_args()
    run(a.seed,a.n_train,a.n_val)
