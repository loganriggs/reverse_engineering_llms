"""Unlabeled function-response grouping on the fixed mixed pre-RoPE cut."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from dual_functional_grouping import connected_groups
from flagship_matrix_free import HD,SNAPSHOT,rotate,rotary
from flagship_prerope_cut_transfer import row_rotation
from flagship_rank96_function_grouping import (
    FIXED_THRESHOLDS,MODES,combine,ids_and_projections,
)
from flagship_transfer import ART


PAIRS={"fit":(7,8),"test_same":(7,8),
       "test_translated":(31,32),
       "test_gap32":(32,64),"test_gap48":(16,64)}


def component_signature(query,key,value_sqrt,a,b,rotation,key_pos,query_pos):
    cos,sin=rotary(max(key_pos,query_pos)+1)
    q1=query["c_q"]
    q2=rotate(query["c_q2"],np.full(len(query["c_q2"]),query_pos),cos,sin)
    k1=rotate(key["c_k"],np.full(len(key["c_k"]),key_pos),cos,sin)
    k2=rotate(key["c_k2"],np.full(len(key["c_k2"]),key_pos),cos,sin)
    pre_key=k1@row_rotation(query_pos).T
    value=key["c_v"]@value_sqrt
    other=q2@k2.T/HD
    left=q1@(a@rotation)
    right=pre_key@(b@rotation)
    result=np.empty((rotation.shape[1],len(q1),HD))
    for i in range(rotation.shape[1]):
        result[i]=(other*right[:,i][None,:])@value
        result[i]*=left[:,i,None]/(HD*len(k1))
    return result.reshape(rotation.shape[1],-1)


def run(n_queries=128,n_keys=8192,model_seed=41,steps=200,
        include_random_starts=False):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    qparts,kparts,value_sqrt,panel=ids_and_projections(
        state,n_queries,n_keys)
    del state
    y=np.load(ART/f"head0_rope_metric_factorial_seed{model_seed}.npz")[
        "mixed__pre"]
    left,s,right=np.linalg.svd(y,full_matrices=False)
    a=left[:,:96]*np.sqrt(s[:96]);b=right[:96].T*np.sqrt(s[:96])
    saved=np.load(ART/f"head0_prerope_gauge_sparsity_seed{model_seed}_s{steps}.npz")
    rotations={name:saved[name] for name in saved.files}
    if include_random_starts:
        extra=np.load(ART/f"head0_prerope_gauge_sparsity_seed{model_seed}_s{steps}_random_start.npz")
        rotations.update({f"randomstart_{name}":extra[name]
            for name in extra.files if not name.startswith("random_") and
            name!="svd"})
    gauges=list(rotations)
    matrices={};transport_controls=[]
    for gauge in gauges:
        rotation=rotations[gauge]
        sigs={}
        for split,(key_pos,query_pos) in PAIRS.items():
            panel_split="fit" if split=="fit" else "test"
            signatures={law:component_signature(qparts[panel_split],
                kparts[panel_split][law],value_sqrt,a,b,rotation,
                key_pos,query_pos) for law in ("uniform","corpus")}
            sigs[split]=signatures
            for mode in MODES:
                normalized=combine(signatures,mode)
                matrices[gauge,split,mode]=np.abs(normalized@normalized.T)
        for law in ("uniform","corpus"):
            original=sigs["fit"][law]
            moved=component_signature(qparts["fit"],kparts["fit"][law],
                value_sqrt,a,b,rotation,31,32)
            denom=float(np.sum(original**2))
            transport_controls.append(dict(gauge=gauge,law=law,
                same_ids_component_relative_mse=float(
                    np.sum((moved-original)**2)/denom),
                same_ids_total_relative_mse=float(
                    np.sum((moved.sum(0)-original.sum(0))**2)/
                    np.sum(original.sum(0)**2))))
        print(gauge,flush=True)
    records=[];thresholds={}
    for mode in MODES:
        null_edges=np.concatenate([
            matrices[f"random_{seed}","fit",mode][np.triu_indices(96,1)]
            for seed in (0,1,2)])
        null=float(np.quantile(null_edges,.999))
        thresholds[mode]=null
        for gauge in gauges:
            fit=matrices[gauge,"fit",mode]
            for threshold in (*FIXED_THRESHOLDS,null):
                edges=np.triu(fit>threshold,1)
                groups=connected_groups(fit,threshold)
                records.append(dict(mode=mode,gauge=gauge,
                    threshold=threshold,
                    threshold_source=("random_gauge_99.9_percentile" if
                        threshold==null else "fixed_sweep"),
                    n_edges=int(edges.sum()),n_groups=len(groups),
                    nontrivial_group_sizes=sorted((len(g) for g in groups
                        if len(g)>1),reverse=True),
                    edge_retention={split:float(np.mean(
                        matrices[gauge,split,mode][edges]>threshold))
                        if edges.any() else None
                        for split in ("test_same","test_translated",
                                      "test_gap32","test_gap48")}))
    output=dict(layer=0,head=0,rank=96,model_seed=model_seed,
        gauge_steps=steps,include_random_starts=include_random_starts,
        n_query_anchors_per_law=n_queries,
        n_key_anchors_per_law=n_keys,gauges=gauges,
        modes=list(MODES),null_thresholds=thresholds,
        fixed_thresholds=list(FIXED_THRESHOLDS),position_pairs=PAIRS,
        panel=panel,
        signature="mean complete-head contribution per query/output from one pre-RoPE QK1 component; QK2 and OV exact",
        graph_rule="absolute cosine of component signatures; connected components",
        scope="unlabeled sampled-token groups; no semantic labels or full-model group ablations",
        records=records,transport_controls=transport_controls)
    name=(f"head0_prerope_response_grouping_seed{model_seed}_s{steps}"
          f"_q{n_queries}_k{n_keys}.json")
    if include_random_starts:name=name[:-5]+"_both_starts.json"
    (ART/name).write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-queries",type=int,default=128)
    p.add_argument("--n-keys",type=int,default=8192)
    p.add_argument("--model-seed",type=int,default=41)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--include-random-starts",action="store_true")
    args=p.parse_args()
    run(args.n_queries,args.n_keys,args.model_seed,args.steps,
        args.include_random_starts)
