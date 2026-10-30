"""Propagate saved layer-0 head-0 QK1 bond cuts through the 546M model.

CPU-first test on disjoint tokenized corpora. Measures original-to-edited KL,
next-token CE difference, and argmax flips. The exact native-model CE is a
correctness gate. No training occurs here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART,load_cuts

MODEL_DIR=Path("/home/loganriggs/Coding/reverse_engineer/tensor_language")
sys.path.insert(0,str(MODEL_DIR))
from jacclust.tt_model import GPT,GPTConfig


def logits(model,idx):
    x=model.transformer.wte(idx)
    x=F.rms_norm(x,(x.size(-1),))
    x0=x
    v1=None
    for block in model.transformer.h:
        x,v1=block(x,v1,x0)
    x=F.rms_norm(x,(x.size(-1),))
    return 30*torch.tanh(model.lm_head(x)/30)


class BondEdit:
    def __init__(self,model):
        self.attn=model.transformer.h[0].attn
        self.native=self.attn.squared_attention
        self.cut=None
        self.attn.squared_attention=self.forward

    def forward(self,q,k,v,q2,k2):
        native=self.native(q,k,v,q2,k2)
        if self.cut is None:
            return native
        d=q.shape[-1]
        qx=q[:,:,0,:]@self.cut.to(device=q.device,dtype=q.dtype)
        score1=(qx@k[:,:,0,:].transpose(-1,-2))/d
        score2=(q2[:,:,0,:]@k2[:,:,0,:].transpose(-1,-2))/d
        pattern=(score1*score2).tril()
        edited=pattern@v[:,:,0,:]
        result=native.clone()
        result[:,0]=edited
        return result


def get_cuts():
    saved=load_cuts()
    cuts={"uniform_corpus_half_wtg64":saved["uniform_corpus_half_wtg64"],
          "uniform_corpus_intervention_thirds_wtg64":
              saved["uniform_corpus_intervention_thirds_wtg64"]}
    for weight in ("0","0.25"):
        name=f"sequence_fit_rank64_cw{weight}"
        cuts[name]=np.load(ART/f"head0_l128_{name}.npz")["cut"]
    cuts["pair_refit_rank64"]=np.load(ART/"head0_l128_pair_refit_rank64.npz")["cut"]
    return cuts


def metrics(reference,candidate,target):
    p=F.log_softmax(reference.to(torch.float64),dim=-1)
    q=F.log_softmax(candidate.to(torch.float64),dim=-1)
    prob=p.exp()
    kl=(prob*(p-q)).sum(dim=-1)
    original_ce=-p.gather(-1,target[...,None]).squeeze(-1)
    edited_ce=-q.gather(-1,target[...,None]).squeeze(-1)
    return dict(kl_sum=float(kl.sum()),ce_original_sum=float(original_ce.sum()),
        ce_edited_sum=float(edited_ce.sum()),
        argmax_flips=int((reference.argmax(-1)!=candidate.argmax(-1)).sum()),
        token_count=target.numel(),
        max_abs_logit=float((reference-candidate).abs().max()))


def run(n_docs=8,length=128,seeds=(23,29),batch_size=4,threads=1,
        cut_names=None):
    torch.set_num_threads(threads)
    config=json.loads((SNAPSHOT/"config.json").read_text())
    config.pop("step",None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    editor=BondEdit(model)
    all_cuts={**get_cuts(),"head0_zero":np.zeros((HD,HD))}
    selected=tuple(all_cuts) if cut_names is None else cut_names
    cuts={name:torch.from_numpy(all_cuts[name]).to(torch.float32)
          for name in selected}
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[]
    identity_gate=None
    with torch.no_grad():
        for corpus_name,(path,begin,end) in corpora.items():
            corpus=np.load(path,mmap_mode="r")
            for seed in seeds:
                rng=np.random.default_rng(seed+length*1000)
                docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
                totals={name:dict(kl_sum=0.,ce_original_sum=0.,
                                  ce_edited_sum=0.,argmax_flips=0,
                                  token_count=0,max_abs_logit=0.)
                        for name in cuts}
                for offset in range(0,n_docs,batch_size):
                    spans=[]
                    for doc in docs[offset:offset+batch_size]:
                        start=int(rng.integers(0,corpus.shape[1]-length))
                        spans.append(np.asarray(corpus[doc,start:start+length+1],
                                                dtype=np.int64))
                    batch=np.stack(spans)
                    idx=torch.from_numpy(batch[:,:-1].copy())
                    target=torch.from_numpy(batch[:,1:].copy())
                    editor.cut=None
                    reference=logits(model,idx)
                    native_ce=model(idx,target)
                    check=float(abs(native_ce-F.cross_entropy(
                        reference.reshape(-1,reference.shape[-1]),target.ravel())))
                    assert check<1e-5,check
                    if identity_gate is None:
                        editor.cut=torch.eye(HD)
                        identity_gate=float((logits(model,idx)-reference).abs().max())
                        assert identity_gate<1e-4,identity_gate
                    for name,x in cuts.items():
                        editor.cut=x
                        candidate=logits(model,idx)
                        row=metrics(reference,candidate,target)
                        for key in ("kl_sum","ce_original_sum","ce_edited_sum",
                                    "argmax_flips","token_count"):
                            totals[name][key]+=row[key]
                        totals[name]["max_abs_logit"]=max(
                            totals[name]["max_abs_logit"],row["max_abs_logit"])
                for name,values in totals.items():
                    records.append(dict(corpus=corpus_name,seed=seed,n_docs=n_docs,
                        length=length,cut=name,**values,
                        kl_per_token=values["kl_sum"]/values["token_count"],
                        ce_delta_per_token=(values["ce_edited_sum"]-
                                            values["ce_original_sum"])/values["token_count"],
                        flip_fraction=values["argmax_flips"]/values["token_count"]))
                print(corpus_name,seed,flush=True)
    result=dict(head=0,layer=0,rank=64,length=length,
        n_docs_per_seed=n_docs,seeds=list(seeds),batch_size=batch_size,
        threads=threads,
        identity_cut_max_abs_logit=identity_gate,
        fineweb_fit_documents=[0,15000],
        fineweb_selection_documents=[15000,17500],
        fineweb_test_documents=[17500,20000],
        scope="full 18-layer model; QK1 cut on layer-0 head-0 only; original-to-edited KL and next-token CE",
        records=records)
    suffix="_zero" if selected==("head0_zero",) else ""
    (ART/f"head0_full_model_l{length}_n{n_docs}{suffix}.json").write_text(
        json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-docs",type=int,default=8)
    parser.add_argument("--length",type=int,default=128)
    parser.add_argument("--seeds",type=int,nargs="+",default=[23,29])
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--cuts",nargs="+",default=None)
    args=parser.parse_args()
    run(args.n_docs,args.length,tuple(args.seeds),args.batch_size,args.threads,
        None if args.cuts is None else tuple(args.cuts))
