"""Held-out test of a connective-token hypothesis suggested by group exemplars."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_component_labels import Decoder,check_decoder
from flagship_matched_rope_fit import pre_features
from flagship_matrix_free import CORPUS,SNAPSHOT,rows,sample_corpus_pairs
from flagship_prerope_group_ablation import candidates
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


CONNECTIVES=frozenset(("as","so","if","when","where","what","how",
                      "because","whether","but"))
COMMON_CONTROLS=frozenset(("the","a","of","to","and","in","for",
                           "on","at","with"))


def run(n_pairs=65536,seeds=(97,101)):
    torch.set_num_threads(4)
    decoder=Decoder();check_decoder(decoder)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    base,effects,details=candidates()
    records=[]
    datasets={"fineweb_validation":(CORPUS,18750,20000),
              **{name:(path,0,1200) for name,path in DATASETS.items()}}
    for corpus_name,(path,begin,end) in datasets.items():
      corpus=np.load(path,mmap_mode="r")
      for seed in seeds:
        train,_,_=rows(state,0,seed,n_pairs,1,128,
            distribution="corpus",corpus=corpus,
            train_docs=(begin,end),val_docs=(begin,end),
            position_mode="cross")
        qid,kid,query_pos,key_pos=sample_corpus_pairs(
            np.random.default_rng(seed),corpus,n_pairs,128,begin,end,
            position_mode="cross")
        u,v,w,_,_=train
        pu,pv,gate=pre_features(u,v,query_pos)
        assert gate<1e-10
        names={int(token):decoder.decode(token).strip().lower()
               for token in np.unique(kid)}
        labels=np.array([names[int(token)] for token in kid],dtype=object)
        masks={"connective_key":np.isin(labels,list(CONNECTIVES)),
               "common_control_key":np.isin(labels,list(COMMON_CONTROLS)),
               "short_gap":(query_pos-key_pos)<=4}
        strata={"short_gap":masks["short_gap"],
                "long_gap":~masks["short_gap"]}
        baseline=np.sum((pu@base)*pv,axis=1)
        base_energy=w*baseline**2
        for group,effect in effects.items():
            score=np.sum((pu@effect)*pv,axis=1)
            energy=w*score**2
            for label,mask in masks.items():
                group_share=float(energy[mask].sum()/energy.sum())
                baseline_share=float(base_energy[mask].sum()/base_energy.sum())
                records.append(dict(corpus=corpus_name,seed=seed,group=group,
                    group_kind=details[group]["kind"],label=label,
                    n_pairs=int(mask.sum()),group_energy_share=group_share,
                    baseline_energy_share=baseline_share,
                    enrichment_over_baseline=group_share/baseline_share))
            for stratum,subset in strata.items():
                mask=subset&masks["connective_key"]
                group_share=float(energy[mask].sum()/energy[subset].sum())
                baseline_share=float(base_energy[mask].sum()/
                                     base_energy[subset].sum())
                records.append(dict(corpus=corpus_name,seed=seed,group=group,
                    group_kind=details[group]["kind"],
                    label=f"connective_key_given_{stratum}",
                    n_pairs=int(mask.sum()),
                    group_energy_share=group_share,
                    baseline_energy_share=baseline_share,
                    enrichment_over_baseline=group_share/baseline_share))
        print(corpus_name,seed,flush=True)
    output=dict(layer=0,head=0,rank=96,n_pairs_per_seed=n_pairs,
        validation_seeds=list(seeds),
        discovery_fineweb_documents=[17500,18750],
        validation_fineweb_documents=[18750,20000],
        external_corpora=list(DATASETS),
        connective_key_tokens=sorted(CONNECTIVES),
        common_control_key_tokens=sorted(COMMON_CONTROLS),
        scope="post-fit label test of group pair energy; not a full-model or semantic-circuit proof",
        records=records)
    (ART/"head0_prerope_group_label_test.json").write_text(
        json.dumps(output,indent=2)+"\n")
    return output


if __name__=="__main__":
    run()
