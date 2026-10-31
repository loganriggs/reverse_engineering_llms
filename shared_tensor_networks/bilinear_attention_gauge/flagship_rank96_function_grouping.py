"""Unlabeled ordinary-function response grouping on the 546M attention head."""

from __future__ import annotations

import json
import argparse

import numpy as np
import torch

from dual_functional_grouping import connected_groups
from flagship_full_model_eval import get_cuts
from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary
from flagship_rank96_group_stability import CUTS
from flagship_transfer import ART


MODES=("uniform","corpus","two_environment")
FIXED_THRESHOLDS=(.6,.7,.8,.9)
GAUGES=("svd","random_0","random_1","random_2",
        "intervention_activation_0","intervention_activation_1",
        "intervention_activation_2","randomstart_0","randomstart_1",
        "randomstart_2")


def ids_and_projections(state,n_queries=16,n_keys=512):
    vocab=state["transformer.wte.weight"].shape[0]
    panel=json.loads((ART/"head0_effective_score_metric.json").read_text())
    query_ids={"fit":np.r_[panel["query_ids"]["train_uniform"][:n_queries],
                             panel["query_ids"]["train_corpus"][:n_queries]],
               "test":np.r_[panel["query_ids"]["test_uniform"][:n_queries],
                              panel["query_ids"]["test_heldout_corpus"][:n_queries]]}
    assert len(np.intersect1d(query_ids["fit"],query_ids["test"]))==0
    corpus=np.load(CORPUS,mmap_mode="r")
    rng=np.random.default_rng(8421)
    uniform=rng.permutation(vocab)
    flat_fit=np.asarray(corpus[:15000,:512]).ravel()
    flat_test=np.asarray(corpus[17500:20000,:512]).ravel()
    key_ids={"fit":{"uniform":uniform[:n_keys],
                    "corpus":rng.choice(flat_fit,size=n_keys,replace=True)},
             "test":{"uniform":uniform[n_keys:2*n_keys],
                     "corpus":rng.choice(flat_test,size=n_keys,replace=True)}}
    all_query=np.unique(np.concatenate(list(query_ids.values())))
    all_key=np.unique(np.concatenate([z for law in key_ids.values()
                                      for z in law.values()]))
    q=project(state,all_query,0,("c_q","c_q2"))
    k=project(state,all_key,0,("c_k","c_k2","c_v"))
    qparts={split:{name:value[np.searchsorted(all_query,ids)]
                   for name,value in q.items()}
            for split,ids in query_ids.items()}
    kparts={split:{law:{name:value[np.searchsorted(all_key,ids)]
                        for name,value in k.items()}
                   for law,ids in laws.items()}
            for split,laws in key_ids.items()}
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64).numpy()
    gram=output.T@output
    eig,basis=np.linalg.eigh(gram)
    sqrt_gram=(basis*np.sqrt(np.maximum(eig,0)))@basis.T
    test_value=kparts["test"]["uniform"]["c_v"][:16]
    norm_gate=float(np.max(np.abs(np.sum((test_value@sqrt_gram)**2,axis=1)-
                                  np.sum((test_value@output.T)**2,axis=1))))
    assert norm_gate<1e-7,norm_gate
    return qparts,kparts,sqrt_gram,dict(
        fit_query_ids=query_ids["fit"].tolist(),
        test_query_ids=query_ids["test"].tolist(),
        fit_key_ids={name:ids.tolist() for name,ids in key_ids["fit"].items()},
        test_key_ids={name:ids.tolist() for name,ids in key_ids["test"].items()},
        value_norm_gate=norm_gate)


def signature(query,key,value_sqrt,a,b,rotation,key_pos,query_pos):
    cos,sin=rotary(max(key_pos,query_pos)+1)
    q1=rotate(query["c_q"],np.full(len(query["c_q"]),query_pos),cos,sin)
    q2=rotate(query["c_q2"],np.full(len(query["c_q2"]),query_pos),cos,sin)
    k1=rotate(key["c_k"],np.full(len(key["c_k"]),key_pos),cos,sin)
    k2=rotate(key["c_k2"],np.full(len(key["c_k2"]),key_pos),cos,sin)
    value=key["c_v"]@value_sqrt
    other=q2@k2.T/HD
    left=q1@(a@rotation)
    right=k1@(b@rotation)
    result=np.empty((rotation.shape[1],len(q1),HD))
    for i in range(rotation.shape[1]):
        result[i]=(other*right[:,i][None,:])@value
        result[i]*=left[:,i,None]/(HD*len(k1))
    return result.reshape(rotation.shape[1],-1)


def normalized(matrix):
    return matrix/np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-30)


def combine(signatures,mode):
    if mode=="two_environment":
        return normalized(np.concatenate((normalized(signatures["uniform"]),
                                          normalized(signatures["corpus"])),axis=1))
    return normalized(signatures[mode])


def run(n_queries=16,n_keys=512):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    qparts,kparts,value_sqrt,panel=ids_and_projections(state,n_queries,n_keys)
    del state
    records=[];null_thresholds={};transport_controls=[]
    for cut_name in CUTS:
        x=get_cuts()[cut_name]
        left,s,right=np.linalg.svd(x,full_matrices=False)
        a=left[:,:96]*np.sqrt(s[:96]);b=right[:96].T*np.sqrt(s[:96])
        identity=np.load(ART/f"head0_exact_gauge_rotations_{cut_name}.npz")
        randomstart=np.load(ART/f"head0_exact_gauge_rotations_{cut_name}_random_start.npz")
        rotations={name:identity[name] for name in GAUGES if name in identity.files}
        rotations.update({f"randomstart_{i}":randomstart[
            f"intervention_activation_{i}"] for i in range(3)})
        matrices={}
        for gauge,rotation in rotations.items():
            cos_shift,sin_shift=rotary(25)
            transported_a=rotate((a@rotation).T,
                np.full(rotation.shape[1],24),cos_shift,sin_shift).T
            transported_b=rotate((b@rotation).T,
                np.full(rotation.shape[1],24),cos_shift,sin_shift).T
            for split,pair in (("fit",(7,8)),("test_same",(7,8)),
                               ("test_relative",(31,32)),
                               ("test_transport",(31,32)),
                               ("test_shift",(32,64))):
                panel_split="fit" if split=="fit" else "test"
                if split=="test_transport":
                    factors=(transported_a,transported_b,np.eye(96))
                else:
                    factors=(a,b,rotation)
                signatures={law:signature(qparts[panel_split],
                    kparts[panel_split][law],value_sqrt,*factors,*pair)
                    for law in ("uniform","corpus")}
                for mode in MODES:
                    sig=combine(signatures,mode)
                    matrices[gauge,split,mode]=np.abs(sig@sig.T)
                if split=="fit":
                    fit_signatures=signatures
            for law in ("uniform","corpus"):
                original=fit_signatures[law]
                fixed=signature(qparts["fit"],kparts["fit"][law],
                    value_sqrt,a,b,rotation,31,32)
                moved=signature(qparts["fit"],kparts["fit"][law],
                    value_sqrt,transported_a,transported_b,np.eye(96),31,32)
                denominator=float(np.sum(original**2))
                total=original.sum(0)
                total_denom=float(np.sum(total**2))
                transport_controls.append(dict(cut=cut_name,gauge=gauge,law=law,
                    fixed_component_relative_mse=float(np.sum((fixed-original)**2)/denominator),
                    transported_component_relative_mse=float(np.sum((moved-original)**2)/denominator),
                    fixed_total_relative_mse=float(np.sum((fixed.sum(0)-total)**2)/total_denom),
                    transported_total_relative_mse=float(np.sum((moved.sum(0)-total)**2)/total_denom)))
        for mode in MODES:
            random_edges=np.concatenate([
                matrices[f"random_{seed}","fit",mode][np.triu_indices(96,1)]
                for seed in range(3)])
            null=float(np.quantile(random_edges,.999))
            null_thresholds[f"{cut_name}/{mode}"]=null
            for gauge in GAUGES:
                fit=matrices[gauge,"fit",mode]
                for threshold in (*FIXED_THRESHOLDS,null):
                    edges=np.triu(fit>threshold,1)
                    groups=connected_groups(fit,threshold)
                    sizes=sorted((len(group) for group in groups if len(group)>1),
                                 reverse=True)
                    records.append(dict(cut=cut_name,mode=mode,gauge=gauge,
                        threshold=threshold,
                        threshold_source=("random_gauge_99.9_percentile" if
                            threshold==null else "fixed_sweep"),
                        n_edges=int(edges.sum()),n_groups=len(groups),
                        nontrivial_group_sizes=sizes,
                        edge_retention={split:float(np.mean(
                            matrices[gauge,split,mode][edges]>threshold))
                            if edges.any() else None
                            for split in ("test_same","test_relative",
                                          "test_transport","test_shift")}))
        print(cut_name,flush=True)
    output=dict(layer=0,head=0,rank=96,cuts=list(CUTS),modes=list(MODES),
        gauges=list(GAUGES),fixed_thresholds=list(FIXED_THRESHOLDS),
        null_thresholds=null_thresholds,
        n_query_anchors_per_law=n_queries,n_key_anchors_per_law=n_keys,
        position_pairs={"fit":[7,8],"test_same":[7,8],
                        "test_relative":[31,32],
                        "test_transport":[31,32],"test_shift":[32,64]},
        panel=panel,
        signature="mean complete-head component response in exact 128-dimensional OV-output norm coordinates over declared key sample, per query and output direction",
        mode_definition="uniform and corpus use separate key samples; two_environment concatenates their separately normalized signatures",
        transport_definition="test_transport co-rotates both rank-96 factor bases by +24 positions; it represents a transported cut, not the fixed deployed cut",
        graph_rule="absolute cosine of component signatures; thresholded connected components",
        scope="exploratory sampled-token grouping panel; no semantic label or full-model group intervention",
        records=records,transport_controls=transport_controls)
    suffix="" if (n_queries,n_keys)==(16,512) else f"_q{n_queries}_k{n_keys}"
    (ART/f"head0_rank96_function_grouping{suffix}.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-queries",type=int,default=16)
    parser.add_argument("--n-keys",type=int,default=512)
    args=parser.parse_args()
    run(args.n_queries,args.n_keys)
