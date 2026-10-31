"""Calibrated full-model deletions of pre-RoPE function-response groups."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_component_ablation import kl_per_document
from flagship_matched_rope_fit import pre_features
from flagship_matched_rope_full_model import PositionAwareBondEdit
from flagship_matrix_free import (
    CORPUS,HD,SNAPSHOT,rows,sample_corpus_pairs,
)
from flagship_ood_corpora import DATASETS
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_transfer import ART


def candidates():
    base=np.load(ART/"head0_rope_metric_factorial_seed41.npz")["mixed__pre"]
    u,s,v=np.linalg.svd(base,full_matrices=False)
    a=u[:,:96]*np.sqrt(s[:96]);b=v[:96].T*np.sqrt(s[:96])
    info=json.load(open(ART/"head0_prerope_group_projector.json"))
    identity=np.load(ART/"head0_prerope_gauge_sparsity_seed41_s200.npz")
    randomstart=np.load(ART/"head0_prerope_gauge_sparsity_seed41_s200_random_start.npz")
    effects={};details={}
    for record in info["groups"]:
        name=record["gauge"]
        if name=="identity_0":
            rotation=identity["factor_weight_0"]
        elif name.startswith("randomstart_"):
            rotation=randomstart[f"factor_weight_{name.split('_')[1]}"]
        else:
            continue
        basis=rotation[:,record["indices"]]
        effects[name]=(a@basis)@(b@basis).T
        details[name]=dict(kind="response_group",size=len(record["indices"]),
            indices=record["indices"],
            fit_component_energy_fraction=record["component_energy_fraction"])
    rng=np.random.default_rng(5101)
    for seed in range(3):
        q,_=np.linalg.qr(rng.normal(size=(96,96)))
        basis=q[:,:10]
        name=f"haar_rank10_{seed}"
        effects[name]=(a@basis)@(b@basis).T
        details[name]=dict(kind="haar_rank10",size=10)
    return base,effects,details


def calibrate(state,base,effects,target_error,n_pairs=32768):
    corpus=np.load(CORPUS,mmap_mode="r")
    train,_,_=rows(state,0,811,n_pairs,1,128,
        distribution="corpus",corpus=corpus,
        train_docs=(15000,17500),val_docs=(15000,17500),
        position_mode="cross")
    pos=sample_corpus_pairs(np.random.default_rng(811),corpus,n_pairs,128,
                            15000,17500,position_mode="cross")
    u,v,w,_,_=train
    pu,pv,gate=pre_features(u,v,pos[2])
    assert gate<1e-10
    reference=np.sum((pu@base)*pv,axis=1)
    denominator=float(np.mean(w*reference**2))
    cuts={};records={}
    for name,effect in effects.items():
        removed=np.sum((pu@effect)*pv,axis=1)
        full_error=float(np.mean(w*removed**2)/denominator)
        fraction=float(np.sqrt(target_error/full_error))
        assert fraction<=1,(name,full_error,target_error)
        cuts[name]=base-fraction*effect
        records[name]=dict(full_removal_relative_pair_error=full_error,
                           deletion_fraction=fraction,
                           calibrated_relative_pair_error=target_error)
    return cuts,records,gate


def run(n_docs=8,sample_seeds=(23,29),batch_size=4,
        target_pair_error=.001,calibration_pairs=32768):
    torch.set_num_threads(4)
    config=json.loads((SNAPSHOT/"config.json").read_text())
    config.pop("step",None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    base,effects,details=candidates()
    cuts,calibration,gate=calibrate(state,base,effects,target_pair_error,
                                    calibration_pairs)
    del state
    editor=PositionAwareBondEdit(model)
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];identity_gate=None
    with torch.no_grad():
        for corpus_name,(path,begin,end) in corpora.items():
            corpus=np.load(path,mmap_mode="r")
            for sample_seed in sample_seeds:
                rng=np.random.default_rng(sample_seed+128000)
                docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
                for offset in range(0,n_docs,batch_size):
                    spans=[]
                    for doc in docs[offset:offset+batch_size]:
                        start=int(rng.integers(0,corpus.shape[1]-129+1))
                        spans.append(np.asarray(corpus[doc,start:start+129],
                                                dtype=np.int64))
                    batch=np.stack(spans)
                    idx=torch.from_numpy(batch[:,:-1].copy())
                    editor.cut=None;editor.pre_cut=None
                    original=logits(model,idx)
                    if identity_gate is None:
                        editor.set_cut("pre",np.eye(HD),128)
                        identity_gate=float((logits(model,idx)-original).abs().max())
                        assert identity_gate<1e-4
                    editor.set_cut("pre",base,128)
                    reference=logits(model,idx)
                    base_from_original=kl_per_document(original,reference)
                    for name,cut in cuts.items():
                        editor.set_cut("pre",cut,128)
                        candidate=logits(model,idx)
                        effect=kl_per_document(reference,candidate)
                        from_original=kl_per_document(original,candidate)
                        for i,doc in enumerate(docs[offset:offset+batch_size]):
                            records.append(dict(corpus=corpus_name,
                                sample_seed=sample_seed,document_id=int(doc),
                                document_position=offset+i,group=name,
                                base_from_original_kl=float(base_from_original[i]),
                                base_to_removed_kl=float(effect[i]),
                                removed_from_original_kl=float(from_original[i])))
                print(corpus_name,sample_seed,flush=True)
    output=dict(layer=0,head=0,rank=96,n_docs_per_seed=n_docs,
        sample_seeds=list(sample_seeds),target_relative_pair_error=target_pair_error,
        calibration_pairs=calibration_pairs,
        calibration_fineweb_documents=[15000,17500],
        test_fineweb_documents=[17500,20000],
        pre_post_identity_gate=gate,identity_cut_max_abs_logit=identity_gate,
        scope="matched pair-error partial deletion of largest response groups and Haar rank-10 controls; full-model per-document KL, no semantic labels",
        details=details,calibration=calibration,records=records)
    (ART/f"head0_prerope_group_ablation_n{n_docs}.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-docs",type=int,default=8)
    p.add_argument("--seeds",type=int,nargs="+",default=[23,29])
    p.add_argument("--target-pair-error",type=float,default=.001)
    args=p.parse_args()
    run(args.n_docs,tuple(args.seeds),target_pair_error=args.target_pair_error)
