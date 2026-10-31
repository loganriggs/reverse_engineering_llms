"""Held-out natural adjacent-pair test for full-vocabulary function bases."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


def sample_pairs(corpus,begin,end,n,seed,exclude):
    rng=np.random.default_rng(seed)
    candidates=rng.choice(np.arange(begin,end),size=min(end-begin,n*4),
                          replace=False)
    chosen=[]
    for doc in candidates:
        start=int(rng.integers(0,corpus.shape[1]-128+1))
        key=int(corpus[doc,start+7]);query=int(corpus[doc,start+8])
        if query in exclude:continue
        chosen.append(dict(document_id=int(doc),start=start,
                           query_id=query,key_id=key))
        if len(chosen)>=n:break
    if len(chosen)<n:raise RuntimeError((len(chosen),n))
    return chosen


def run(n_pairs=512):
    torch.set_num_threads(4)
    prior=json.loads((ART/"head0_full_function_metric.json").read_text())
    assert prior["n_query_anchors_per_law"]==16
    bases=np.load(ART/"head0_full_function_key_bases.npz")
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    vocab=state["transformer.wte.weight"].shape[0]
    train_frequency=np.bincount(np.asarray(np.load(CORPUS,mmap_mode="r")[:15000]).ravel(),
                                 minlength=vocab)
    assert hashlib.sha256(train_frequency.tobytes()).hexdigest()==prior["frequency_sha256"]
    ratio=train_frequency/train_frequency.mean()
    weights={"uniform":np.ones(vocab),"half_corpus":.5+.5*ratio,
             "mostly_corpus":.1+.9*ratio}
    exclude=set(prior["query_ids"]["train_uniform"]+
                prior["query_ids"]["train_corpus"])
    corpora={"fineweb_test":(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    samples={name:sample_pairs(np.load(path,mmap_mode="r"),begin,end,n_pairs,
                               91013+index,exclude)
             for index,(name,(path,begin,end)) in enumerate(corpora.items())}
    all_query=np.unique([r["query_id"] for case in samples.values() for r in case])
    query=project(state,all_query,0,("c_q","c_q2"))
    key=project(state,np.arange(vocab,dtype=np.int64),0,
                ("c_k","c_k2","c_v"))
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64).numpy()
    eigen,vector=np.linalg.eigh(output.T@output)
    c=(key["c_v"]@(vector@np.diag(np.sqrt(np.maximum(eigen,0)))@vector.T))
    bond=np.load(ART/"head0_l128_sequence_fit_rank64_cw0.npz")["cut"]
    cos,sin=rotary(9)
    q1=rotate(query["c_q"],np.full(len(all_query),8),cos,sin)
    q2=rotate(query["c_q2"],np.full(len(all_query),8),cos,sin)
    k1=rotate(key["c_k"],np.full(vocab,7),cos,sin)
    k2=rotate(key["c_k2"],np.full(vocab,7),cos,sin)
    prepared={name:dict(root=np.sqrt(weight),right=bases[name][:64])
              for name,weight in weights.items()}
    all_records=[]
    for token_index,token in enumerate(all_query):
        rows=[(corpus_name,r) for corpus_name,case in samples.items()
              for r in case if r["query_id"]==token]
        if not rows:continue
        score=(q1[token_index]@k1.T)/HD*(q2[token_index]@k2.T)/HD
        bond_score=(q1[token_index]@bond@k1.T)/HD*(q2[token_index]@k2.T)/HD
        # Shared across all occurrences of this query; all key IDs enter
        # the projection, even though only natural key pairs are scored.
        for corpus_name,row in rows:
            k=row["key_id"]
            reference=score[k]*c[k]
            outputs={"bond_rank64":bond_score[k]*c[k]}
            for name,part in prepared.items():
                root=part["root"]
                right=part["right"]
                factor=(c.T*(score*root)[None,:])@right.T
                outputs[name+"_rank64"]=factor@(right[:,k]/root[k])
            for name,prediction in outputs.items():
                all_records.append(dict(corpus=corpus_name,
                    document_id=row["document_id"],start=row["start"],
                    query_id=int(token),key_id=k,method=name,
                    squared_error=float(np.sum((prediction-reference)**2)),
                    reference_energy=float(np.sum(reference**2)),
                    query_train_frequency=int(train_frequency[token]),
                    key_train_frequency=int(train_frequency[k])))
        if (token_index+1)%32==0:print("query",token_index+1,"of",len(all_query),flush=True)
    result=dict(n_pairs_per_corpus=n_pairs,n_unique_query_ids=len(all_query),
        fit_query_ids_excluded=True,position_pair=[7,8],
        fit_corpus_documents=[0,15000],
        test_corpora={name:[begin,end] for name,(_,begin,end) in corpora.items()},
        scope="exact head pair-output metric on actual adjacent tokens at fixed positions, no sequence sum or full-model propagation",
        records=all_records)
    (ART/"head0_full_function_natural_pairs.json").write_text(json.dumps(result,indent=2)+"\n")
    for corpus_name in corpora:
        print(corpus_name,flush=True)
        for method in ("uniform_rank64","half_corpus_rank64",
                       "mostly_corpus_rank64","bond_rank64"):
            selected=[r for r in all_records if r["corpus"]==corpus_name and
                      r["method"]==method]
            ratio=sum(r["squared_error"] for r in selected)/sum(
                r["reference_energy"] for r in selected)
            print(method,ratio,flush=True)
    return result


if __name__=="__main__":run()
