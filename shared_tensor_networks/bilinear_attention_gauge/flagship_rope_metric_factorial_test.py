"""Held-out pair and fixed-position stress tests for the RoPE metric factorial."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_matched_rope_bulk_test import evaluate as evaluate_bulk_law
from flagship_matched_rope_fit import EVAL_PAIRS
from flagship_matrix_free import HD,SNAPSHOT
from flagship_prerope_cut_transfer import panel_response,row_rotation
from flagship_rank96_function_grouping import ids_and_projections
from flagship_rope_metric_factorial import PRIORS
from flagship_transfer import ART


def run(n_pairs=65536,n_queries=128,n_keys=8192,test_seed=73,
        model_seeds=(41,43,47)):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    cuts=[]
    for model_seed in model_seeds:
        arrays=np.load(ART/f"head0_rope_metric_factorial_seed{model_seed}.npz")
        for prior in PRIORS:
            for geometry in ("post","pre"):
                cuts.append((model_seed,prior,geometry,
                             arrays[f"{prior}__{geometry}"]))
    bulk=[];gates={}
    for law in ("uniform","corpus"):
        records,gate=evaluate_bulk_law(state,law,n_pairs,test_seed,cuts)
        bulk.extend(records);gates[law]=gate
        print("bulk",law,flush=True)
    query,key,value_sqrt,panel=ids_and_projections(state,n_queries,n_keys)
    stress=[]
    for law in ("uniform","corpus"):
        for name,(key_pos,query_pos) in EVAL_PAIRS.items():
            exact_score,exact_response=panel_response(
                query["test"],key["test"][law],value_sqrt,
                key_pos,query_pos,np.eye(HD))
            for model_seed,prior,geometry,matrix in cuts:
                x=(matrix if geometry=="post" else
                   np.linalg.solve(row_rotation(query_pos),
                                   matrix@row_rotation(query_pos)))
                score,response=panel_response(
                    query["test"],key["test"][law],value_sqrt,
                    key_pos,query_pos,x)
                stress.append(dict(law=law,pair=name,
                    positions=[key_pos,query_pos],model_seed=model_seed,
                    prior=prior,geometry=geometry,
                    score_relative_mse=float(np.sum((score-exact_score)**2)/
                                             np.sum(exact_score**2)),
                    response_relative_mse=float(np.sum((response-exact_response)**2)/
                                                np.sum(exact_response**2))))
        print("stress",law,flush=True)
    output=dict(layer=0,head=0,rank=96,
        n_bulk_pairs_per_law=n_pairs,
        n_query_anchors_per_law=n_queries,
        n_key_anchors_per_law=n_keys,
        test_seed=test_seed,model_seeds=list(model_seeds),
        priors=PRIORS,test_corpus_documents=[17500,20000],
        bulk_identity_gates=gates,
        stress_panel=panel,
        scope="independent pair-response and fixed-position tests; no downstream KL",
        bulk_records=bulk,stress_records=stress)
    name=f"head0_rope_metric_factorial_test_n{n_pairs}_q{n_queries}_k{n_keys}.json"
    (ART/name).write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-pairs",type=int,default=65536)
    p.add_argument("--n-queries",type=int,default=128)
    p.add_argument("--n-keys",type=int,default=8192)
    a=p.parse_args()
    run(a.n_pairs,a.n_queries,a.n_keys)
