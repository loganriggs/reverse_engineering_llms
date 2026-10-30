"""Independent held-out evaluation for sequence-fitted head-0 QK1 cuts."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_matrix_free import CORPUS,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_sequence_transfer import check_direct_long,sequence_head
from flagship_transfer import ART,load_cuts


def run(n_docs=256,seeds=(23,29)):
    torch.set_num_threads(1)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    exact_gate=check_direct_long(state)
    saved=load_cuts()
    cuts={name:saved[name] for name in
          ("uniform_corpus_half_wtg64",
           "uniform_corpus_intervention_thirds_wtg64")}
    for weight in ("0","0.25"):
        name=f"sequence_fit_rank64_cw{weight}"
        cuts[name]=np.load(ART/f"head0_l128_{name}.npz")["cut"]
    cuts["pair_refit_rank64"]=np.load(ART/"head0_l128_pair_refit_rank64.npz")["cut"]
    fits={weight:json.loads((ART/f"head0_l128_sequence_fit_rank64_cw{weight}.json").read_text())
          for weight in ("0","0.25")}
    for fit in fits.values():
        assert fit["fit_documents"]==[0,15000]
        assert fit["validation_documents"]==[15000,17500]
        assert not set(fit["fit_doc_ids"])&set(fit["validation_doc_ids"])
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    modes={"natural":"natural","qk2_constant":"constant",
           "qk2_key_shift":"key_shift"}
    records=[]
    for corpus_name,(path,begin,end) in corpora.items():
        corpus=np.load(path,mmap_mode="r")
        for seed in seeds:
            rng=np.random.default_rng(seed+128000)
            docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
            if corpus_name=="fineweb_test":
                for fit in fits.values():
                    assert not set(docs)&set(fit["validation_doc_ids"])
            totals={name:{mode:dict(sse=0.,energy=0.) for mode in modes}
                    for name in cuts}
            for doc in docs:
                start=int(rng.integers(0,corpus.shape[1]-128+1))
                tokens=np.asarray(corpus[doc,start:start+128],dtype=np.int64)
                for label,mode in modes.items():
                    outputs=sequence_head(state,tokens,0,cuts,qk2_mode=mode)
                    target=outputs["identity"]
                    energy=float(np.sum(target**2))
                    for name in cuts:
                        delta=outputs[name]-target
                        totals[name][label]["sse"]+=float(np.sum(delta**2))
                        totals[name][label]["energy"]+=energy
            for name in cuts:
                record=dict(corpus=corpus_name,seed=seed,n_docs=n_docs,cut=name)
                for label in modes:
                    stat=totals[name][label]
                    record[f"{label}_error"]=stat["sse"]/stat["energy"]
                    record[f"{label}_sse"]=stat["sse"]
                    record[f"{label}_energy"]=stat["energy"]
                records.append(record)
            print(corpus_name,seed,flush=True)
    result=dict(head=0,rank=64,length=128,seeds=list(seeds),
        n_docs_per_seed=n_docs,exact_gate_max_abs=exact_gate,
        fit_documents=[0,15000],selection_documents=[15000,17500],
        fineweb_test_documents=[17500,20000],
        other_corpus_documents=[0,1200],
        scope="layer-0 head output after OV; no later layers, KL, or CE",
        records=records)
    (ART/"head0_l128_sequence_fit_eval.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--n-docs",type=int,default=256)
    parser.add_argument("--seeds",type=int,nargs="+",default=[23,29])
    args=parser.parse_args()
    run(args.n_docs,tuple(args.seeds))
