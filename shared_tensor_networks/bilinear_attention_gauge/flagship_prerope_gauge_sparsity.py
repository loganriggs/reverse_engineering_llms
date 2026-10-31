"""Exact-gauge legibility priors on the selected mixed pre-RoPE rank-96 cut."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_exact_gauge_sparsity import (
    check_gradient,contribution_stats,fit,match_signature,pair_features,
    signature,weight_loss,
)
from flagship_matched_rope_fit import HELDOUT_GAPS,positions_for_rows,pre_features
from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from flagship_rope_metric_factorial import PRIORS
from flagship_transfer import ART


KINDS=("factor_weight","activation","intervention_activation")
LAWS=("uniform","corpus")


def parts_for_law(state,law,seed,n_train,n_val,a,b,corpus):
    train,val,_=rows(state,0,seed,n_train,n_val,128,
        distribution=law,corpus=corpus if law=="corpus" else None,
        val_docs=(15000,17500),position_mode="cross")
    positions=positions_for_rows(seed,n_train,n_val,law,corpus)
    output={}
    for split,part,pos in (("train",train,positions[0]),
                           ("validation",val,positions[1])):
        u,v,w,b2,c_norm=part
        keep=~np.isin(pos[0]-pos[1],HELDOUT_GAPS)
        pu,pv,gate=pre_features(u[keep],v[keep],pos[0][keep])
        assert gate<1e-10
        raw=(pu,pv,w[keep],b2[keep],c_norm[keep])
        output[split]={law:pair_features(raw,a,b),
                       law+"_constant":pair_features(raw,a,b,True)}
    return output


def test_parts(state,law,seed,n_test,a,b,corpus):
    train,_,_=rows(state,0,seed,n_test,1,128,
        distribution=law,corpus=corpus if law=="corpus" else None,
        train_docs=(17500,20000),val_docs=(17500,20000),
        position_mode="cross")
    rng=np.random.default_rng(seed)
    if law=="uniform":
        from flagship_matrix_free import sample_pairs
        pos=sample_pairs(rng,50304,n_test,128,position_mode="cross")
    else:
        from flagship_matrix_free import sample_corpus_pairs
        pos=sample_corpus_pairs(rng,corpus,n_test,128,17500,20000,
                                position_mode="cross")
    u,v,w,b2,c_norm=train
    pu,pv,gate=pre_features(u,v,pos[2])
    assert gate<1e-10
    result={};gap=pos[2]-pos[3]
    for label,mask in (("",np.ones(n_test,dtype=bool)),
                       ("_heldout_gap",np.isin(gap,HELDOUT_GAPS))):
        raw=(pu[mask],pv[mask],w[mask],b2[mask],c_norm[mask])
        result[law+label]=pair_features(raw,a,b)
        result[law+"_constant"+label]=pair_features(raw,a,b,True)
    return result,dict(n_test=n_test,n_heldout_gap=int(np.isin(gap,HELDOUT_GAPS).sum()))


def run(n_train=32768,n_val=8192,n_test=32768,steps=200,
        batch_size=2048,seeds=(0,1,2),lr=.01,model_seed=41,
        random_start=False):
    torch.set_num_threads(4)
    gradient_gate=check_gradient()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    matrix=np.load(ART/f"head0_rope_metric_factorial_seed{model_seed}.npz")[
        "mixed__pre"]
    left,s,right=np.linalg.svd(matrix,full_matrices=False)
    rank=96
    a=left[:,:rank]*np.sqrt(s[:rank])
    b=right[:rank].T*np.sqrt(s[:rank])
    ta=torch.from_numpy(a);tb=torch.from_numpy(b)
    train={};validation={};test={};test_coverage={}
    for law in LAWS:
        p=parts_for_law(state,law,model_seed,n_train,n_val,a,b,corpus)
        train.update(p["train"]);validation.update(p["validation"])
        t,coverage=test_parts(state,law,73,n_test,a,b,corpus)
        test.update(t);test_coverage[law]=coverage
    rotations={"svd":np.eye(rank)};fit_info={}
    for seed in seeds:
        rng=np.random.default_rng(100+seed)
        q,r=np.linalg.qr(rng.normal(size=(rank,rank)))
        rotations[f"random_{seed}"]=q@np.diag(np.sign(np.diag(r)))
        for kind in KINDS:
            origin=None
            if random_start:
                q0,r0=np.linalg.qr(rng.normal(size=(rank,rank)))
                origin=q0@np.diag(np.sign(np.diag(r0)))
            rotation,info=fit(kind,train,validation,ta,tb,seed,
                              steps,batch_size,lr,initial_rotation=origin)
            rotations[f"{kind}_{seed}"]=rotation
            fit_info[f"{kind}_{seed}"]=info
    records=[]
    for name,rotation in rotations.items():
        orthogonal=float(np.linalg.norm(rotation.T@rotation-np.eye(rank)))
        bond_error=float(np.linalg.norm((a@rotation)@(b@rotation).T-matrix)/
                         np.linalg.norm(matrix))
        assert max(orthogonal,bond_error)<1e-9,(name,orthogonal,bond_error)
        rt=torch.from_numpy(rotation)
        for split,parts in (("validation",validation),("test",test)):
            for law,part in parts.items():
                records.append(dict(gauge=name,split=split,law=law,
                    orthogonality_error=orthogonal,
                    bond_relative_error=bond_error,
                    factor_weight_l1=float(weight_loss(ta,tb,rt)),
                    **contribution_stats(part,rt)))
    signatures={name:signature(test["corpus"],torch.from_numpy(rotation))
                for name,rotation in rotations.items()}
    matches=[]
    for kind in (*KINDS,"random"):
        for i,seed in enumerate(seeds):
            for other in seeds[i+1:]:
                matches.append(dict(kind=kind,seeds=[seed,other],
                    mean_matched_abs_correlation=match_signature(
                        signatures[f"{kind}_{seed}"],
                        signatures[f"{kind}_{other}"])))
    prefix=f"head0_prerope_gauge_sparsity_seed{model_seed}_s{steps}"
    if random_start:prefix+="_random_start"
    np.savez_compressed(ART/f"{prefix}.npz",**rotations)
    output=dict(layer=0,head=0,rank=rank,model_seed=model_seed,
        cut="mixed pre-RoPE WTG",n_train_per_law=n_train,
        n_validation_per_law=n_val,n_test_per_law=n_test,
        steps=steps,batch_size=batch_size,fit_seeds=list(seeds),
        kinds=list(KINDS),learning_rate=lr,random_start=random_start,
        heldout_relative_gaps=list(HELDOUT_GAPS),
        fit_corpus_documents=[0,15000],
        validation_corpus_documents=[15000,17500],
        test_corpus_documents=[17500,20000],
        exact_gauge_definition="orthogonal rotations of the SVD factors preserve the entire fixed rank-96 pre-RoPE cut",
        scope="sparsity and component reproducibility proxies, not semantic groups",
        gradient_gate=gradient_gate,test_coverage=test_coverage,
        fit_info=fit_info,records=records,matches=matches,
        rotations=f"{prefix}.npz")
    (ART/f"{prefix}.json").write_text(json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-train",type=int,default=32768)
    p.add_argument("--n-val",type=int,default=8192)
    p.add_argument("--n-test",type=int,default=32768)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--batch-size",type=int,default=2048)
    p.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--lr",type=float,default=.01)
    p.add_argument("--model-seed",type=int,default=41)
    p.add_argument("--random-start",action="store_true")
    args=p.parse_args()
    run(args.n_train,args.n_val,args.n_test,args.steps,args.batch_size,
        tuple(args.seeds),args.lr,args.model_seed,args.random_start)
