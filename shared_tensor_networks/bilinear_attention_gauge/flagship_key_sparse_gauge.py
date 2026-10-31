"""Key-side L1 gauges of a fixed 546M-model first-layer attention cut.

Uniform sampling is a Monte Carlo weight-only-domain approximation, not an
enumeration of the full token-by-position function. Corpus sampling changes
only the orientation prior; the rank-64 bond function stays fixed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_exact_gauge_sparsity import (contribution_stats,
    match_signature,pair_features,signature)
from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


def fit_key(train,validation,seed,steps=200,batch_size=4096,lr=.01):
    rng=np.random.default_rng(2000+seed)
    n=train.shape[1]
    k=torch.nn.Parameter(torch.zeros((n,n),dtype=torch.float64))
    optimizer=torch.optim.Adam((k,),lr=lr)
    best_rotation=torch.eye(n,dtype=torch.float64)
    best=float((validation@best_rotation).abs().mean())
    best_step=0;trace=[]
    for step in range(1,steps+1):
        idx=torch.from_numpy(rng.choice(len(train),batch_size,replace=False))
        rotation=torch.matrix_exp(k-k.T)
        loss=(train[idx]@rotation).abs().mean()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_((k,),10.)
        optimizer.step()
        if step==1 or step%25==0 or step==steps:
            with torch.no_grad():
                current=torch.matrix_exp(k-k.T)
                value=float((validation@current).abs().mean())
                if value<best:
                    best=value;best_step=step
                    best_rotation=current.detach().clone()
                trace.append(dict(step=step,train_loss=float(loss),
                                  validation_loss=value))
            print("key fit",seed,step,value,flush=True)
    return best_rotation.numpy(),dict(best_step=best_step,
                                      best_validation_l1=best,trace=trace)


def run(n_pairs=32768,n_test=8192,steps=200,seeds=(0,1,2)):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    x=np.load(ART/"head0_l128_sequence_fit_rank64_cw0.npz")["cut"]
    left,s,right=np.linalg.svd(x,full_matrices=False)
    a=left[:,:64]*np.sqrt(s[:64]);b=right[:64].T*np.sqrt(s[:64])
    laws={};test_parts={}
    for law in ("uniform","corpus"):
        train,val,_=rows(state,0,41,n_pairs,n_pairs,128,
            distribution=law,corpus=corpus if law=="corpus" else None,
            val_docs=(15000,17500))
        _,test,_=rows(state,0,23,1,n_test,128,
            distribution=law,corpus=corpus if law=="corpus" else None,
            val_docs=(17500,20000))
        laws[law]=(torch.from_numpy(np.ascontiguousarray(train[1]@b)),
                   torch.from_numpy(np.ascontiguousarray(val[1]@b)))
        test_parts[law]=pair_features(test,a,b)
    for name,path in DATASETS.items():
        other=np.load(path,mmap_mode="r")
        _,test,_=rows(state,0,23,1,n_test,128,
            distribution="corpus",corpus=other,
            train_docs=(0,600),val_docs=(600,1200))
        test_parts[name]=pair_features(test,a,b)
    del state
    old=np.load(ART/"head0_exact_gauge_rotations.npz")
    rotations={name:old[name] for name in ("svd","random_0",
        "factor_weight_0","activation_0","intervention_activation_0")}
    fit_info={}
    for law in laws:
        for seed in seeds:
            rotation,info=fit_key(*laws[law],seed+100*(law=="corpus"),steps)
            name=f"key_{law}_{seed}"
            rotations[name]=rotation;fit_info[name]=info
    records=[]
    for name,rotation in rotations.items():
        orthogonal=float(np.linalg.norm(rotation.T@rotation-np.eye(64)))
        bond_error=float(np.linalg.norm((a@rotation)@(b@rotation).T-x)/
                         np.linalg.norm(x))
        assert orthogonal<1e-9 and bond_error<1e-9,(name,orthogonal,bond_error)
        rt=torch.from_numpy(rotation)
        for law,part in test_parts.items():
            records.append(dict(gauge=name,law=law,
                key_l1=float((part[1]@rt).abs().mean()),
                orthogonality_error=orthogonal,bond_relative_error=bond_error,
                **contribution_stats(part,rt)))
    signatures={name:signature(test_parts["corpus"],torch.from_numpy(r))
                for name,r in rotations.items()}
    matches=[]
    for law in laws:
        for i,seed in enumerate(seeds):
            for other in seeds[i+1:]:
                names=[f"key_{law}_{seed}",f"key_{law}_{other}"]
                matches.append(dict(law=law,seeds=[seed,other],
                    component_pair_correlation=match_signature(
                        signatures[names[0]],signatures[names[1]])))
    output=dict(head=0,rank=64,cut="natural sequence-output fit",
        n_fit_pairs_per_law=n_pairs,n_validation_pairs_per_law=n_pairs,
        n_test_pairs_per_law=n_test,steps=steps,seeds=list(seeds),
        fit_fineweb_documents=[0,15000],
        validation_fineweb_documents=[15000,17500],
        test_fineweb_documents=[17500,20000],
        scope="exact bond-preserving orthogonal gauges; uniform sampled token/position domain versus corpus-aware key-side L1 prior; sparsity is not human legibility",
        fit_info=fit_info,records=records,matches=matches)
    (ART/"head0_key_sparse_gauge.json").write_text(json.dumps(output,indent=2)+"\n")
    np.savez_compressed(ART/"head0_key_sparse_rotations.npz",**rotations)
    for law in test_parts:
        print(law,{name:round(float(np.mean([r["effective_active"]
            for r in records if r["law"]==law and r["gauge"].startswith(name)])),2)
            for name in ("svd","activation_0","key_uniform_","key_corpus_")},
            flush=True)
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-pairs",type=int,default=32768)
    p.add_argument("--n-test",type=int,default=8192)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--seeds",type=int,default=3)
    args=p.parse_args()
    run(args.n_pairs,args.n_test,args.steps,tuple(range(args.seeds)))
