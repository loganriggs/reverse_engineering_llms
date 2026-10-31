"""Score-only function cuts on the exact same natural pairs as flat cuts."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from flagship_effective_score_metric import score_table
from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project
from flagship_transfer import ART


def run():
    torch.set_num_threads(4)
    flat=json.loads((ART/"head0_full_function_natural_pairs.json").read_text())
    cases=[r for r in flat["records"] if r["method"]=="bond_rank64"]
    query_ids=np.unique([r["query_id"] for r in cases])
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    vocab=state["transformer.wte.weight"].shape[0]
    fit_frequency=np.bincount(np.asarray(np.load(CORPUS,mmap_mode="r")[:15000]).ravel(),
                               minlength=vocab)
    ratio=fit_frequency/fit_frequency.mean()
    query=project(state,query_ids,0,("c_q","c_q2"))
    key=project(state,np.arange(vocab),0,("c_k","c_k2","c_v"))
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64).numpy()
    value_weight=np.einsum("ki,ij,kj->k",key["c_v"],output.T@output,key["c_v"])
    score=score_table(query,key,7,8)
    by_id={int(token):index for index,token in enumerate(query_ids)}
    rows=np.array([by_id[r["query_id"]] for r in cases],dtype=np.int64)
    keys=np.array([r["key_id"] for r in cases],dtype=np.int64)
    target=score[rows,keys].astype(np.float64)
    energy=(target**2)*value_weight[keys]
    reference=np.array([r["reference_energy"] for r in cases])
    gate=float(np.max(np.abs(energy-reference)/np.maximum(reference,1e-20)))
    assert gate<1e-4,gate
    configurations={
        "strict_uniform":("head0_effective_score_key_bases_uniform_queries.npz",
                          "uniform",np.ones(vocab)),
        "strict_half_corpus":("head0_effective_score_key_bases_uniform_queries.npz",
                              "half_corpus",.5+.5*ratio),
        "strict_mostly_corpus":("head0_effective_score_key_bases_uniform_queries.npz",
                                "mostly_corpus",.1+.9*ratio),
        "mixed_uniform":("head0_effective_score_key_bases.npz",
                         "uniform",np.ones(vocab)),
        "mixed_mostly_corpus":("head0_effective_score_key_bases.npz",
                               "mostly_corpus",.1+.9*ratio),
        "multi_mostly_corpus":("head0_effective_score_multi_position_bases.npz",
                               "mostly_corpus",.1+.9*ratio)}
    records=[]
    for name,(path,key_name,frequency_weight) in configurations.items():
        basis=np.load(ART/path)[key_name]
        root=np.sqrt(value_weight*frequency_weight).astype(np.float32)
        scaled=score*root[None,:]
        projection=scaled@basis.T
        for rank in (64,128):
            predicted=np.sum(projection[rows,:rank]*basis[:rank][:,keys].T,
                             axis=1)/root[keys]
            error=(predicted.astype(np.float64)-target)**2*value_weight[keys]
            for case,err,e in zip(cases,error,energy):
                records.append(dict(corpus=case["corpus"],
                    document_id=case["document_id"],start=case["start"],
                    query_id=case["query_id"],key_id=case["key_id"],
                    method=f"{name}_rank{rank}",squared_error=float(err),
                    reference_energy=float(e)))
        print(name,flush=True)
    result=dict(n_pairs_per_corpus=flat["n_pairs_per_corpus"],
        source_pair_file="head0_full_function_natural_pairs.json",
        reference_energy_max_relative_gate=gate,
        fit_frequency_sha256=hashlib.sha256(fit_frequency.tobytes()).hexdigest(),
        scope="same 512 actual adjacent-token pairs/corpus as flat-function evaluation; exact OV value map; no sequence sum or full-model KL",
        records=records)
    (ART/"head0_effective_score_natural_pairs.json").write_text(json.dumps(result,indent=2)+"\n")
    for corpus_name in ("fineweb_test","wikitext103","tinystories"):
        print(corpus_name,flush=True)
        for name in configurations:
            r=[x for x in records if x["corpus"]==corpus_name and
               x["method"]==f"{name}_rank64"]
            print(name,sum(x["squared_error"] for x in r)/
                  sum(x["reference_energy"] for x in r),flush=True)
    return result


if __name__=="__main__":run()
