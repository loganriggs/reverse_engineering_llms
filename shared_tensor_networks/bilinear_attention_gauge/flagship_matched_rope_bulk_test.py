"""Independent natural and QK2-constant pair tests for matched RoPE fits."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_matched_rope_fit import HELDOUT_GAPS,pre_features
from flagship_matrix_free import (
    CORPUS,SNAPSHOT,rows,sample_corpus_pairs,sample_pairs,
)
from flagship_transfer import ART
from matrix_free_environment import SampledBondEnvironment


def evaluate(state,law,n_pairs,seed,cuts):
    corpus=np.load(CORPUS,mmap_mode="r") if law=="corpus" else None
    train,_,_=rows(state,0,seed,n_pairs,1,128,distribution=law,
        corpus=corpus,train_docs=(17500,20000),val_docs=(17500,20000),
        position_mode="cross")
    rng=np.random.default_rng(seed)
    if law=="uniform":
        positions=sample_pairs(rng,50304,n_pairs,128,position_mode="cross")
    else:
        positions=sample_corpus_pairs(rng,corpus,n_pairs,128,17500,20000,
                                      position_mode="cross")
    u,v,natural_weight,_,value_norm=train
    pu,pv,gate=pre_features(u,v,positions[2])
    gap=positions[2]-positions[3]
    records=[]
    for branch,w in (("natural",natural_weight),
                     ("qk2_constant",value_norm)):
        for split,mask in (("all",np.ones(n_pairs,dtype=bool)),
                           ("allowed_gaps",~np.isin(gap,HELDOUT_GAPS)),
                           ("heldout_gaps",np.isin(gap,HELDOUT_GAPS))):
            for model_seed,geometry,matrix in cuts:
                features=(u,v) if geometry=="post" else (pu,pv)
                env=SampledBondEnvironment(
                    features[0][mask],features[1][mask],w[mask])
                records.append(dict(law=law,branch=branch,split=split,
                    model_seed=model_seed,geometry=geometry,
                    n_pairs=int(mask.sum()),relative_pair_mse=env.relative_error(matrix)))
    return records,gate


def run(n_pairs=65536,test_seed=73,model_seeds=(41,43,47)):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    cuts=[]
    for seed in model_seeds:
        p=ART/f"head0_matched_rope_fit_n32768_s0_q128_k8192_seed{seed}.npz"
        arrays=np.load(p)
        for geometry in ("post","pre"):
            cuts.append((seed,geometry,arrays[f"wtg_joint__{geometry}"]))
    records=[];gates={}
    for law in ("uniform","corpus"):
        result,gate=evaluate(state,law,n_pairs,test_seed,cuts)
        records.extend(result);gates[law]=gate
        print(law,gate,flush=True)
    output=dict(layer=0,head=0,rank=96,n_pairs_per_law=n_pairs,
        test_seed=test_seed,model_seeds=list(model_seeds),
        test_corpus_documents=[17500,20000],
        heldout_relative_gaps=list(HELDOUT_GAPS),identity_gates=gates,
        branch_definition="QK2 constant sets its scalar score to one, leaving QK1 and OV unchanged",
        scope="independent sampled QK1 pair-response metric; no sequence aggregation or downstream KL",
        records=records)
    (ART/f"head0_matched_rope_bulk_test_n{n_pairs}.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-pairs",type=int,default=65536)
    p.add_argument("--test-seed",type=int,default=73)
    a=p.parse_args()
    run(a.n_pairs,a.test_seed)
