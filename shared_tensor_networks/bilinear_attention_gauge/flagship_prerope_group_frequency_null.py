"""Frequency- and token-form-matched null for the connective-key group label."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_component_labels import Decoder,check_decoder
from flagship_matched_rope_fit import pre_features
from flagship_matrix_free import CORPUS,SNAPSHOT,rows,sample_corpus_pairs
from flagship_ood_corpora import DATASETS
from flagship_prerope_group_ablation import candidates
from flagship_prerope_group_label_test import CONNECTIVES
from flagship_transfer import ART


def matched_sets(decoder,corpus,n_null=100,seed=910,pool_size=100,
                 target_ids=None):
    vocab=50304
    names=[decoder.decode(i).strip().lower() for i in range(vocab)]
    categories=np.array([decoder.category(i) for i in range(vocab)])
    connector_ids=(np.array([i for i,name in enumerate(names)
                             if name in CONNECTIVES],dtype=np.int64)
                   if target_ids is None else
                   np.array(sorted(set(target_ids)),dtype=np.int64))
    assert len(connector_ids)>0 and np.all((connector_ids>=0)&
                                           (connector_ids<vocab))
    counts=np.bincount(np.asarray(corpus[:15000,:512]).ravel(),
                       minlength=vocab)
    eligible=np.setdiff1d(np.arange(vocab),connector_ids)
    rng=np.random.default_rng(seed)
    pools=[]
    for token in connector_ids:
        same=eligible[categories[eligible]==categories[token]]
        distance=np.abs(np.log1p(counts[same])-np.log1p(counts[token]))
        nearest=same[np.argpartition(distance,min(pool_size,len(same)-1))[:pool_size]]
        pools.append(nearest)
    controls=[np.array([rng.choice(pool) for pool in pools],dtype=np.int64)
              for _ in range(n_null)]
    return connector_ids,controls,counts,categories


def run(n_pairs=65536,seeds=(97,101),n_null=100):
    torch.set_num_threads(4)
    decoder=Decoder();check_decoder(decoder)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    fineweb=np.load(CORPUS,mmap_mode="r")
    target_ids,controls,counts,categories=matched_sets(
        decoder,fineweb,n_null)
    base,effects,_=candidates()
    selected={name:effect for name,effect in effects.items()
              if name=="identity_0" or name.startswith("randomstart_")}
    datasets={"fineweb_validation":(CORPUS,18750,20000),
              **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[]
    for corpus_name,(path,begin,end) in datasets.items():
        corpus=np.load(path,mmap_mode="r")
        for seed in seeds:
            train,_,_=rows(state,0,seed,n_pairs,1,128,
                distribution="corpus",corpus=corpus,
                train_docs=(begin,end),val_docs=(begin,end),
                position_mode="cross")
            _,kid,ipos,_=sample_corpus_pairs(np.random.default_rng(seed),
                corpus,n_pairs,128,begin,end,position_mode="cross")
            u,v,w,_,_=train
            pu,pv,gate=pre_features(u,v,ipos)
            assert gate<1e-10
            exact=np.sum((pu@base)*pv,axis=1)
            base_energy=w*exact**2
            masks=[np.isin(kid,target_ids)]+[np.isin(kid,ids)
                                             for ids in controls]
            base_shares=np.array([base_energy[mask].sum()/base_energy.sum()
                                  for mask in masks])
            for group,effect in selected.items():
                score=np.sum((pu@effect)*pv,axis=1)
                energy=w*score**2
                shares=np.array([energy[mask].sum()/energy.sum()
                                 for mask in masks])
                enrich=shares/np.maximum(base_shares,1e-30)
                records.append(dict(corpus=corpus_name,seed=seed,group=group,
                    n_target_pairs=int(masks[0].sum()),
                    target_energy_share=float(shares[0]),
                    target_baseline_share=float(base_shares[0]),
                    target_enrichment=float(enrich[0]),
                    null_energy_share_mean=float(shares[1:].mean()),
                    null_energy_share_max=float(shares[1:].max()),
                    null_enrichment_mean=float(enrich[1:].mean()),
                    null_enrichment_max=float(enrich[1:].max()),
                    null_rank_energy=int(np.sum(shares[1:]>=shares[0])),
                    null_rank_enrichment=int(np.sum(enrich[1:]>=enrich[0]))))
            print(corpus_name,seed,flush=True)
    output=dict(layer=0,head=0,rank=96,
        n_pairs_per_sample=n_pairs,seeds=list(seeds),n_null=n_null,
        discovery_fineweb_documents=[17500,18750],
        validation_fineweb_documents=[18750,20000],
        target_token_ids=target_ids.tolist(),
        control_rule="for each target token ID, sample an ID with the same coarse token-form category from its 100 nearest training-frequency neighbors",
        train_frequency_documents=[0,15000],
        scope="post-fit held-out pair-energy label test; no full-model label-specific intervention",
        records=records)
    (ART/"head0_prerope_group_frequency_null.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
