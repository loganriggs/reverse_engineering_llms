"""Post-fit held-out token-pair examples for the pre-RoPE response group."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_component_labels import CATEGORIES,Decoder,check_decoder
from flagship_matched_rope_fit import pre_features
from flagship_matrix_free import CORPUS,SNAPSHOT,rows,sample_corpus_pairs
from flagship_prerope_group_ablation import candidates
from flagship_transfer import ART


def run(n_pairs=65536,seed=73,doc_begin=17500,doc_end=18750):
    torch.set_num_threads(4)
    decoder=Decoder();check_decoder(decoder)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    base,effects,_=candidates()
    effect=effects["identity_0"]
    train,_,_=rows(state,0,seed,n_pairs,1,128,
        distribution="corpus",corpus=corpus,
        train_docs=(doc_begin,doc_end),val_docs=(doc_begin,doc_end),
        position_mode="cross")
    qid,kid,ipos,jpos=sample_corpus_pairs(np.random.default_rng(seed),corpus,
        n_pairs,128,doc_begin,doc_end,position_mode="cross")
    u,v,w,_,_=train
    pu,pv,gate=pre_features(u,v,ipos)
    group_score=np.sum((pu@effect)*pv,axis=1)
    full_score=np.sum((pu@base)*pv,axis=1)
    group_energy=w*group_score**2
    full_energy=w*full_score**2
    qcat=np.array([decoder.category(token) for token in qid])
    kcat=np.array([decoder.category(token) for token in kid])
    category=[]
    for qi,qname in enumerate(CATEGORIES):
        for ki,kname in enumerate(CATEGORIES):
            mask=(qcat==qi)&(kcat==ki)
            if mask.sum()<10:continue
            category.append(dict(query_category=qname,key_category=kname,
                n=int(mask.sum()),
                group_energy_share=float(group_energy[mask].sum()/group_energy.sum()),
                baseline_energy_share=float(full_energy[mask].sum()/full_energy.sum()),
                energy_share_ratio=float((group_energy[mask].sum()/group_energy.sum())/
                    max(full_energy[mask].sum()/full_energy.sum(),1e-30))))
    top=np.argsort(group_energy)[-40:][::-1]
    examples=[dict(query_id=int(qid[i]),key_id=int(kid[i]),
        query_token=decoder.decode(qid[i]),key_token=decoder.decode(kid[i]),
        key_pos=int(jpos[i]),query_pos=int(ipos[i]),
        group_score=float(group_score[i]),
        baseline_score=float(full_score[i]),
        weighted_group_energy=float(group_energy[i])) for i in top]
    output=dict(layer=0,head=0,rank=96,n_pairs=n_pairs,seed=seed,
        corpus_documents=[doc_begin,doc_end],
        pre_post_identity_gate=gate,
        group="largest corpus-response group of identity-start factor-weight gauge",
        limitation="post-fit examples and coarse token forms do not establish a task or semantic label",
        category=category,top_examples=examples)
    (ART/"head0_prerope_group_exemplars.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
