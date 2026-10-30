"""Out-of-fit-measure checks for width-128 first-layer QK1 reductions.

Cuts were fit on four- or 128-position windows. Evaluate on disjoint corpus
documents, longer windows, token-frequency strata, and a QK2-constant
intervention. This remains a head-level pair-function metric, not model CE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from flagship_matrix_free import CORPUS,SNAPSHOT,rows
from matrix_free_environment import SampledBondEnvironment


HERE=Path(__file__).parent
ART=HERE/"full_width_results"


def load_cuts():
    cuts={}
    for fit_length in (4,128):
        for fit_measure in ("uniform","corpus"):
            suffix="_corpus" if fit_measure=="corpus" else ""
            stem=f"head0_seed17_n65536{suffix}"
            if fit_length!=4:stem+=f"_l{fit_length}"
            base=np.load(ART/(stem+".npz"))
            direct=np.load(ART/(stem+"_fet.npz"))
            prefix=f"{fit_measure}_fit_l{fit_length}"
            cuts[f"{prefix}_wtg16"]=base["wtg_rank16"]
            cuts[f"{prefix}_local16"]=base["local_rank16"]
            cuts[f"{prefix}_direct_wtg_start16"]=direct["wtg_rank16"]
            cuts[f"{prefix}_direct_local_start16"]=direct["local_rank16"]
            if fit_length==128:
                for rank in (32,64,96):
                    cuts[f"{prefix}_wtg{rank}"]=base[f"wtg_rank{rank}"]
    for mixture in ("uniform_corpus_half",
                    "uniform_corpus_intervention_thirds"):
        arrays=np.load(ART/f"head0_l128_{mixture}.npz")
        for rank in (32,64):
            cuts[f"{mixture}_wtg{rank}"]=arrays[f"rank{rank}"]
    return cuts


def run(n_test=32768,seeds=(23,29)):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    corpus=np.load(CORPUS,mmap_mode="r")
    vocab=state["transformer.wte.weight"].shape[0]
    frequency=np.bincount(corpus[:15000,:512].ravel(),minlength=vocab)
    pools={"rare_1_to_10":np.flatnonzero((frequency>=1)&(frequency<=10)),
           "common_1000_plus":np.flatnonzero(frequency>=1000),
           "unseen_zero":np.flatnonzero(frequency==0)}
    conditions=[
        ("uniform_l4_all","uniform",4,"all",None),
        ("uniform_l4_self","uniform",4,"self",None),
        ("uniform_l4_cross","uniform",4,"cross",None),
        ("uniform_l128_all","uniform",128,"all",None),
        ("uniform_l128_self","uniform",128,"self",None),
        ("uniform_l128_cross","uniform",128,"cross",None),
        ("corpus_l4_all","corpus",4,"all",None),
        ("corpus_l4_self","corpus",4,"self",None),
        ("corpus_l4_cross","corpus",4,"cross",None),
        ("corpus_l128_all","corpus",128,"all",None),
        ("corpus_l128_self","corpus",128,"self",None),
        ("corpus_l128_cross","corpus",128,"cross",None),
        ("rare_l128_all","uniform",128,"all",pools["rare_1_to_10"]),
        ("common_l128_all","uniform",128,"all",pools["common_1000_plus"]),
        ("unseen_l128_all","uniform",128,"all",pools["unseen_zero"])]
    cuts=load_cuts()
    records=[]
    condition_stats=[]
    for seed in seeds:
        for name,distribution,length,position_mode,pool in conditions:
            _,validation,coverage=rows(state,0,seed,1,n_test,length,
                distribution=distribution,
                corpus=corpus if distribution=="corpus" else None,
                token_pool=pool,position_mode=position_mode)
            u,v,w,b,c_norm=validation
            identity_score=np.sum(u*v,axis=1)
            condition_stats.append(dict(seed=seed,condition=name,
                target_pair_energy=float(np.mean(w*identity_score**2)),
                mean_qk2_ov_weight=float(np.mean(w)),
                coverage=coverage))
            natural=SampledBondEnvironment(u,v,w/w.sum())
            constant=SampledBondEnvironment(u,v,c_norm/c_norm.sum())
            for cut_name,x in cuts.items():
                records.append(dict(seed=seed,condition=name,length=length,
                    distribution=distribution,position_mode=position_mode,
                    cut=cut_name,pair_error=natural.relative_error(x),
                    qk2_constant_error=constant.relative_error(x)))
            print(seed,name,"coverage",coverage["unique_query_tokens"],
                  coverage["unique_key_tokens"],flush=True)
    output=ART/"head0_context_transfer.json"
    result=dict(head=0,ranks=[16,32,64,96],n_test_per_condition_seed=n_test,
        seeds=list(seeds),fit_lengths=[4,128],fit_seed=17,fit_n=65536,
        corpus_fit_documents=[0,15000],corpus_test_documents=[15000,20000],
        corpus_path=str(CORPUS),
        frequency_definition="counts in fit documents, first 512 tokens per document",
        token_pool_sizes={name:int(len(pool)) for name,pool in pools.items()},
        result_scope="weighted first-layer head pair contribution only; not full sequence output or CE",
        condition_stats=condition_stats,
        records=records)
    output.write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":run()
