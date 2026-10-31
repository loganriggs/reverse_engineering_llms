"""Matched multi-position rank-96 QK1 pair fits before and after RoPE."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_full_model_eval import get_cuts
from flagship_matrix_free import (
    CORPUS, HD, SNAPSHOT, rotate, rotary, rows, sample_corpus_pairs,
    sample_pairs,
)
from flagship_prerope_cut_transfer import panel_response, row_rotation
from flagship_rank96_function_grouping import ids_and_projections
from flagship_rank96_group_stability import CUTS
from flagship_transfer import ART
from matrix_free_environment import SampledBondEnvironment


HELDOUT_GAPS = (32, 48)
LAWS = ("uniform", "corpus")
EVAL_PAIRS = {"reference": (7, 8), "translated": (31, 32),
              "heldout_gap32": (32, 64), "heldout_gap48": (16, 64)}


def positions_for_rows(seed, n_train, n_val, law, corpus):
    rng = np.random.default_rng(seed)
    if law == "uniform":
        vocab = 50304
        train = sample_pairs(rng, vocab, n_train, 128, position_mode="cross")
        val = sample_pairs(rng, vocab, n_val, 128, position_mode="cross")
    else:
        train = sample_corpus_pairs(rng, corpus, n_train, 128, 0, 15000,
                                    position_mode="cross")
        val = sample_corpus_pairs(rng, corpus, n_val, 128, 15000, 17500,
                                  position_mode="cross")
    return ((train[2], train[3]), (val[2], val[3]))


def pre_features(post_u, post_v, query_positions):
    pre_u = np.empty_like(post_u)
    pre_v = np.empty_like(post_v)
    for p in np.unique(query_positions):
        mask = query_positions == p
        rotation = row_rotation(int(p))
        pre_u[mask] = post_u[mask] @ np.linalg.inv(rotation)
        pre_v[mask] = post_v[mask] @ rotation.T
    delta = np.sum(post_u * post_v, axis=1) - np.sum(pre_u * pre_v, axis=1)
    gate = float(np.max(np.abs(delta)) / max(np.max(np.abs(np.sum(post_u*post_v,axis=1))),1e-12))
    assert gate < 1e-10, gate
    return pre_u, pre_v, gate


def prepare(state, seed, n_train, n_val):
    corpus = np.load(CORPUS, mmap_mode="r")
    datasets = {"post": {"train": {}, "val": {}},
                "pre": {"train": {}, "val": {}}}
    coverage = {}; gates = []
    for law in LAWS:
        train, val, info = rows(state, 0, seed, n_train, n_val, 128,
            distribution=law, corpus=corpus if law=="corpus" else None,
            val_docs=(15000,17500), position_mode="cross")
        positions = positions_for_rows(seed,n_train,n_val,law,corpus)
        for split,part,pos in (("train",train,positions[0]),
                               ("val",val,positions[1])):
            u,v,w,_,_ = part
            gap = pos[0] - pos[1]
            keep = ~np.isin(gap, HELDOUT_GAPS)
            u,v,w = u[keep],v[keep],w[keep]
            p = pos[0][keep]
            pu,pv,gate = pre_features(u,v,p)
            gates.append(gate)
            for geometry,features in (("post",(u,v)),("pre",(pu,pv))):
                target = np.sum(features[0]*features[1],axis=1)
                energy = float(np.mean(w*target**2))
                datasets[geometry][split][law] = (
                    tuple(torch.from_numpy(np.asarray(z)).to(torch.float64)
                          for z in (*features,w,target)), energy)
            coverage[f"{law}/{split}"] = dict(
                sampled=len(gap),kept=int(keep.sum()),
                unique_query_positions=int(len(np.unique(p))),
                excluded_gaps=list(HELDOUT_GAPS))
        coverage[law] = info
    return datasets, coverage, max(gates)


def loss_on(part, a, b, indexes=None):
    (u,v,w,target),energy = part
    if indexes is not None:
        u,v,w,target = (z[indexes] for z in (u,v,w,target))
    estimate = torch.sum((u@a)*(v@b),dim=1)
    return torch.mean(w*(estimate-target).square())/energy


def factorize(x):
    u,s,v=np.linalg.svd(x,full_matrices=False)
    return (u[:,:96]*np.sqrt(s[:96]),v[:96].T*np.sqrt(s[:96]))


def wtg_initial(datasets, geometry):
    parts=[datasets[geometry]["train"][law] for law in LAWS]
    u=np.concatenate([part[0][0].numpy() for part in parts])
    v=np.concatenate([part[0][1].numpy() for part in parts])
    w=np.concatenate([part[0][2].numpy()/(len(part[0][2])*part[1]*len(LAWS))
                      for part in parts])
    environment=SampledBondEnvironment(u,v,w)
    cut,info=environment.weighted_trace_gauge(96,max_iter=100,tol=1e-7)
    return cut,dict(train_relative_error=environment.relative_error(cut),
                    left_boundary=info["left"],right_boundary=info["right"],
                    left_condition=info["left_condition"],
                    right_condition=info["right_condition"])


def fit(datasets, initial, geometry, steps, batch_size, seed, lr,
        initial_is_geometry=False):
    start=(initial if initial_is_geometry or geometry=="post" else
           row_rotation(8)@initial@np.linalg.inv(row_rotation(8)))
    fa,fb=factorize(start)
    a=torch.nn.Parameter(torch.from_numpy(fa.copy()))
    b=torch.nn.Parameter(torch.from_numpy(fb.copy()))
    opt=torch.optim.Adam((a,b),lr=lr)
    rng=np.random.default_rng(seed)
    def validate():
        with torch.no_grad():
            return {law:float(loss_on(datasets[geometry]["val"][law],a,b))
                    for law in LAWS}
    initial_val=validate();best_score=sum(initial_val.values())/len(LAWS)
    best=start.copy();best_step=0;trace=[]
    for step in range(steps):
        opt.zero_grad()
        pieces=[]
        for law in LAWS:
            n=len(datasets[geometry]["train"][law][0][0])
            ix=torch.from_numpy(rng.choice(n,size=min(batch_size,n),replace=False))
            pieces.append(loss_on(datasets[geometry]["train"][law],a,b,ix))
        objective=sum(pieces)/len(LAWS)
        assert torch.isfinite(objective), (geometry,step)
        objective.backward()
        torch.nn.utils.clip_grad_norm_((a,b),10.)
        opt.step()
        if step==0 or (step+1)%25==0 or step+1==steps:
            score=validate();mean=sum(score.values())/len(LAWS)
            if mean<best_score:
                best_score=mean;best_step=step+1
                best=(a@b.T).detach().numpy().copy()
            trace.append(dict(step=step+1,validation=score,mean=mean))
    return best,dict(initial_validation=initial_val,best_step=best_step,
                     best_validation_mean=best_score,trace=trace)


def evaluate(state, cuts, n_queries, n_keys):
    queries,keys,value_sqrt,panel=ids_and_projections(state,n_queries,n_keys)
    records=[]
    for split in ("fit","test"):
        for law in LAWS:
            for name,(kp,qp) in EVAL_PAIRS.items():
                exact_score, exact_response = panel_response(
                    queries[split],keys[split][law],value_sqrt,kp,qp,np.eye(HD))
                for run_name,(geometry,matrix) in cuts.items():
                    x=(matrix if geometry=="post" else
                       np.linalg.solve(row_rotation(qp),
                                       matrix@row_rotation(qp)))
                    score,response=panel_response(
                        queries[split],keys[split][law],value_sqrt,kp,qp,x)
                    records.append(dict(split=split,law=law,pair=name,
                        positions=[kp,qp],run=run_name,geometry=geometry,
                        score_relative_mse=float(np.sum((score-exact_score)**2)/
                                                 np.sum(exact_score**2)),
                        response_relative_mse=float(np.sum((response-exact_response)**2)/
                                                   np.sum(exact_response**2))))
    return records,panel


def run(n_train=32768,n_val=8192,steps=300,batch_size=2048,
        n_queries=128,n_keys=8192,seed=41,lr=.001,wtg_only=False):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    datasets,coverage,gate=prepare(state,seed,n_train,n_val)
    fits={};infos={};starts=get_cuts();wtg_infos={}
    for geometry in ("post","pre"):
        starts[f"wtg_joint/{geometry}"],wtg_infos[geometry]=wtg_initial(
            datasets,geometry)
    initializations=("wtg_joint",) if wtg_only else (*CUTS,"wtg_joint")
    for start_name in initializations:
        for geometry in ("post","pre"):
            name=f"{start_name}/{geometry}"
            initial=(starts[start_name] if start_name!="wtg_joint" else
                     starts[f"wtg_joint/{geometry}"])
            matrix,info=fit(datasets,initial,geometry,steps,
                            batch_size,seed+100,lr,
                            initial_is_geometry=start_name=="wtg_joint")
            fits[name]=(geometry,matrix);infos[name]=info
            print(name,info["best_step"],info["best_validation_mean"],flush=True)
    records,panel=evaluate(state,fits,n_queries,n_keys)
    arrays={name.replace("/","__"):matrix for name,(_,matrix) in fits.items()}
    suffix=f"_n{n_train}_s{steps}_q{n_queries}_k{n_keys}"
    if seed!=41 or wtg_only:
        suffix+=f"_seed{seed}"
    np.savez_compressed(ART/f"head0_matched_rope_fit{suffix}.npz",**arrays)
    result=dict(layer=0,head=0,rank=96,seed=seed,
        n_train_pairs_per_law=n_train,n_validation_pairs_per_law=n_val,
        steps=steps,batch_size_per_law=batch_size,learning_rate=lr,
        fit_corpus_documents=[0,15000],selection_corpus_documents=[15000,17500],
        test_corpus_documents=[17500,20000],
        heldout_relative_gaps=list(HELDOUT_GAPS),
        objective="equal mean of uniform and corpus relative weighted QK1 pair-score MSE; weights are QK2 squared times OV output norm squared",
        pre_post_identity_gate=gate,coverage=coverage,
        initializations=list(initializations),wtg_initial_info=wtg_infos,
        fit_info=infos,eval_panel=panel,eval_records=records,
        scope="matched pair-objective CPU fit, sampled tokens and positions; no sequence-level or full-model validation",
        arrays=f"head0_matched_rope_fit{suffix}.npz")
    (ART/f"head0_matched_rope_fit{suffix}.json").write_text(
        json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-train",type=int,default=32768)
    p.add_argument("--n-val",type=int,default=8192)
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--batch-size",type=int,default=2048)
    p.add_argument("--n-queries",type=int,default=128)
    p.add_argument("--n-keys",type=int,default=8192)
    p.add_argument("--seed",type=int,default=41)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--wtg-only",action="store_true")
    a=p.parse_args()
    run(a.n_train,a.n_val,a.steps,a.batch_size,a.n_queries,a.n_keys,
        a.seed,a.lr,a.wtg_only)
