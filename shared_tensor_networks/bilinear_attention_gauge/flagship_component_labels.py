"""Held-out token-form labels for exact-gauge attention components.

Labels are used only after basis fitting. A pure-JSON GPT-2 vocabulary decoder
avoids importing tokenizer native extensions in this environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary,sample_corpus_pairs
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART
from flagship_full_model_eval import get_cuts


VOCAB=Path("/home/loganriggs/.cache/huggingface/hub/models--gpt2/"
           "snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/vocab.json")
CATEGORIES=("space","digits","punctuation","space_word",
            "capitalized_word","word_fragment","mixed")
CAT_INDEX={name:i for i,name in enumerate(CATEGORIES)}


def byte_decoder():
    printable=list(range(ord("!"),ord("~")+1))+\
              list(range(ord("¡"),ord("¬")+1))+\
              list(range(ord("®"),ord("ÿ")+1))
    bs=printable[:];cs=printable[:];extra=0
    for value in range(256):
        if value not in printable:
            bs.append(value);cs.append(256+extra);extra+=1
    return {chr(symbol):byte for byte,symbol in zip(bs,cs)}


class Decoder:
    def __init__(self):
        self.inverse={i:token for token,i in json.loads(VOCAB.read_text()).items()}
        self.bytes=byte_decoder()

    def decode(self,token_id):
        token=self.inverse.get(int(token_id),"<extra-token>")
        if token.startswith("<|"):
            return token
        return bytes(self.bytes[c] for c in token).decode("utf-8","replace")

    def category(self,token_id):
        text=self.decode(token_id)
        core=text.strip()
        if not core:return CAT_INDEX["space"]
        if core.isdecimal():return CAT_INDEX["digits"]
        if all(not c.isalnum() and not c.isspace() for c in core):
            return CAT_INDEX["punctuation"]
        if core.isalpha():
            if text[0].isspace():return CAT_INDEX["space_word"]
            if core[0].isupper():return CAT_INDEX["capitalized_word"]
            return CAT_INDEX["word_fragment"]
        return CAT_INDEX["mixed"]


def check_decoder(decoder):
    assert decoder.decode(198)=="\n"
    assert decoder.decode(220)==" "
    assert decoder.decode(15496)=="Hello"
    assert decoder.category(11)==CAT_INDEX["punctuation"]
    assert decoder.category(15496)==CAT_INDEX["capitalized_word"]


def sample(state,corpus,begin,end,n,seed,a,b,decoder):
    rng=np.random.default_rng(seed)
    q,k,i,j=sample_corpus_pairs(rng,corpus,n,128,begin,end)
    q_ids,qinv=np.unique(q,return_inverse=True)
    k_ids,kinv=np.unique(k,return_inverse=True)
    query=project(state,q_ids,0,("c_q","c_q2"))
    key=project(state,k_ids,0,("c_k","c_k2","c_v"))
    cos,sin=rotary(128)
    u=rotate(query["c_q"][qinv],i,cos,sin)
    v=rotate(key["c_k"][kinv],j,cos,sin)
    u2=rotate(query["c_q2"][qinv],i,cos,sin)
    v2=rotate(key["c_k2"][kinv],j,cos,sin)
    score2=np.sum(u2*v2,axis=1)/(HD**2)
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64).numpy()
    c=key["c_v"][kinv]@output.T
    w=score2**2*np.sum(c*c,axis=1)
    fa=u@a;fb=v@b
    target=np.sum(fa*fb,axis=1)
    qcat=np.array([decoder.category(token) for token in q],dtype=np.int64)
    kcat=np.array([decoder.category(token) for token in k],dtype=np.int64)
    label=qcat*len(CATEGORIES)+kcat
    return dict(fa=fa,fb=fb,w=w,target=target,q=q,k=k,label=label)


def distribution(part,rotation):
    z=(part["fa"]@rotation)*(part["fb"]@rotation)
    energy=part["w"][:,None]*z**2
    ncat=len(CATEGORIES)**2
    component=np.zeros((z.shape[1],ncat))
    for i in range(z.shape[1]):
        component[i]=np.bincount(part["label"],weights=energy[:,i],minlength=ncat)
    component/=np.maximum(component.sum(1,keepdims=True),1e-30)
    base=np.bincount(part["label"],weights=part["w"]*part["target"]**2,
                     minlength=ncat)
    base/=base.sum()
    size=energy.sum(0)
    return component,base,size,z


def label_name(index):
    return CATEGORIES[index//len(CATEGORIES)]+" -> "+\
           CATEGORIES[index%len(CATEGORIES)]


def exemplars(part,z,component,decoder,n=6):
    score=part["w"]*z[:,component]**2
    order=np.argsort(score)[::-1]
    result=[];seen=set()
    for index in order:
        q=int(part["q"][index]);k=int(part["k"][index])
        if (q,k) in seen:continue
        seen.add((q,k))
        result.append(dict(query_id=q,key_id=k,query=decoder.decode(q),
                           key=decoder.decode(k),weighted_component_energy=float(score[index])))
        if len(result)==n:break
    return result


def run(n_pairs=32768,top_components=10,
        cut_name="sequence_fit_rank64_cw0"):
    torch.set_num_threads(1)
    decoder=Decoder();check_decoder(decoder)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    x=get_cuts()[cut_name]
    left,s,right=np.linalg.svd(x,full_matrices=False)
    rank=int(np.linalg.matrix_rank(x))
    a=left[:,:rank]*np.sqrt(s[:rank]);b=right[:rank].T*np.sqrt(s[:rank])
    fineweb=np.load(CORPUS,mmap_mode="r")
    parts={"selection":sample(state,fineweb,15000,17500,n_pairs,71,a,b,decoder),
           "fineweb_test":sample(state,fineweb,17500,20000,n_pairs,73,a,b,decoder)}
    for name,path in DATASETS.items():
        corpus=np.load(path,mmap_mode="r")
        parts[name]=sample(state,corpus,0,1200,n_pairs,73,a,b,decoder)
    del state
    sample_diagnostics={}
    for name,part in parts.items():
        energy=part["w"]*part["target"]**2
        by_label=np.bincount(part["label"],weights=energy,
                             minlength=len(CATEGORIES)**2)
        order=np.argsort(by_label)[::-1]
        sample_diagnostics[name]=dict(
            response_energy_ess=float(energy.sum()**2/np.sum(energy**2)),
            largest_pair_energy_fraction=float(energy.max()/energy.sum()),
            top_categories=[dict(label=label_name(int(index)),
                                 energy_fraction=float(by_label[index]/energy.sum()),
                                 sampled_pairs=int(np.sum(part["label"]==index)))
                            for index in order[:10]])
    suffix="" if cut_name=="sequence_fit_rank64_cw0" else f"_{cut_name}"
    rotations=np.load(ART/f"head0_exact_gauge_rotations{suffix}.npz")
    records=[];examples={}
    for gauge in rotations.files:
        rotation=rotations[gauge]
        selection,base,size,z=distribution(parts["selection"],rotation)
        eligible=base>=.02
        assert eligible.any()
        order=np.argsort(size)[::-1][:top_components]
        labels={}
        for component in order:
            enrichment=np.where(eligible,selection[component]/np.maximum(base,1e-30),-1)
            category=int(np.argmax(enrichment))
            labels[int(component)]=category
            if gauge in ("svd","random_0","activation_0",
                         "intervention_activation_0","rare_balanced_activation_0",
                         "factor_weight_0"):
                examples[f"{gauge}_{component}"]=dict(
                    label=label_name(category),
                    top_pairs=exemplars(parts["selection"],z,component,decoder))
        for corpus_name,part in parts.items():
            comp,base_test,_,_=distribution(part,rotation)
            for component,category in labels.items():
                fraction=float(comp[component,category])
                base_fraction=float(base_test[category])
                records.append(dict(gauge=gauge,corpus=corpus_name,
                    component=component,selection_rank=int(np.where(order==component)[0][0]),
                    proposed_label=label_name(category),
                    component_fraction=fraction,base_fraction=base_fraction,
                    enrichment=fraction/max(base_fraction,1e-30),
                    log2_enrichment=float(np.log2(max(fraction,1e-30)/
                                                   max(base_fraction,1e-30))),
                    selection_label=bool(corpus_name=="selection")))
    result=dict(head=0,rank=rank,cut=cut_name,n_pairs_per_corpus=n_pairs,
        top_components=top_components,label_categories=list(CATEGORIES),
        label_rule="max selection enrichment among categories with >=2% original-response energy",
        selection_fineweb_documents=[15000,17500],
        test_fineweb_documents=[17500,20000],
        scope="post-fit token-form labels; no labels in gauge fitting; not full-model causal validation",
        sample_diagnostics=sample_diagnostics,
        records=records,examples=examples)
    (ART/f"head0_component_label_transfer{suffix}.json").write_text(
        json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-pairs",type=int,default=32768)
    parser.add_argument("--top-components",type=int,default=10)
    parser.add_argument("--cut-name",default="sequence_fit_rank64_cw0",
                        choices=tuple(get_cuts()))
    args=parser.parse_args()
    run(args.n_pairs,args.top_components,args.cut_name)
