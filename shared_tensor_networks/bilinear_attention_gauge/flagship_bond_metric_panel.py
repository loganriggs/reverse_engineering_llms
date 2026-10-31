"""Common full-vocabulary and natural-pair panel for saved QK1 bond cuts."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from flagship_effective_score_metric import fidelity,score_table
from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary
from flagship_transfer import ART,load_cuts


def selected_cuts():
    saved=load_cuts()
    names=("uniform_fit_l128_wtg64","corpus_fit_l128_wtg64",
           "uniform_corpus_half_wtg64",
           "uniform_corpus_intervention_thirds_wtg64",
           "uniform_fit_l128_wtg96","corpus_fit_l128_wtg96")
    cuts={name:saved[name] for name in names}
    for weight in ("0","0.25"):
        name=f"sequence_fit_rank64_cw{weight}"
        cuts[name]=np.load(ART/f"head0_l128_{name}.npz")["cut"]
    cuts["pair_refit_rank64"]=np.load(
        ART/"head0_l128_pair_refit_rank64.npz")["cut"]
    return cuts


def natural_pairs(cases,state,cuts,value_weight):
    ids=np.array([r["query_id"] for r in cases],dtype=np.int64)
    keys=np.array([r["key_id"] for r in cases],dtype=np.int64)
    q_unique,qinv=np.unique(ids,return_inverse=True)
    k_unique,kinv=np.unique(keys,return_inverse=True)
    q_proj=project(state,q_unique,0,("c_q","c_q2"))
    k_proj=project(state,k_unique,0,("c_k","c_k2"))
    cos,sin=rotary(9)
    q1=rotate(q_proj["c_q"][qinv],np.full(len(cases),8),cos,sin)
    q2=rotate(q_proj["c_q2"][qinv],np.full(len(cases),8),cos,sin)
    k1=rotate(k_proj["c_k"][kinv],np.full(len(cases),7),cos,sin)
    k2=rotate(k_proj["c_k2"][kinv],np.full(len(cases),7),cos,sin)
    other=np.sum(q2*k2,axis=1)/HD
    reference=np.sum(q1*k1,axis=1)/HD*other
    energy=reference**2*value_weight[keys]
    old=np.array([r["reference_energy"] for r in cases])
    gate=float(np.max(np.abs(energy-old)/np.maximum(old,1e-20)))
    assert gate<1e-4,gate
    records=[]
    for name,x in cuts.items():
        changed=np.sum((q1@x)*k1,axis=1)/HD*other
        error=(changed-reference)**2*value_weight[keys]
        for case,e,denom in zip(cases,error,energy):
            records.append(dict(corpus=case["corpus"],
                document_id=case["document_id"],start=case["start"],
                query_id=case["query_id"],key_id=case["key_id"],
                cut=name,squared_error=float(e),reference_energy=float(denom)))
    return records,gate


def run():
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    vocab=state["transformer.wte.weight"].shape[0]
    corpus=np.load(CORPUS,mmap_mode="r")
    test_frequency=np.bincount(np.asarray(corpus[17500:20000]).ravel(),minlength=vocab)
    test_ratio=test_frequency/test_frequency.mean()
    panel_files={"strict_uniform":ART/"head0_effective_score_metric_uniform_queries.json",
                 "mixed":ART/"head0_effective_score_metric.json"}
    panels={name:json.loads(path.read_text()) for name,path in panel_files.items()}
    qids={name:panel["query_ids"]["test_heldout_corpus"]
          for name,panel in panels.items()}
    all_query_ids=np.unique([id for ids in qids.values() for id in ids])
    query=project(state,all_query_ids,0,("c_q","c_q2"))
    key=project(state,np.arange(vocab),0,("c_k","c_k2","c_v"))
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64).numpy()
    value_weight=np.einsum("ki,ij,kj->k",key["c_v"],output.T@output,key["c_v"])
    value_scale=float(value_weight.mean())
    value_weight/=value_scale
    cuts=selected_cuts()
    results=[]
    for panel_name,ids in qids.items():
        indices=np.searchsorted(all_query_ids,ids)
        chosen={name:z[indices] for name,z in query.items()}
        for key_pos,query_pos in ((7,8),(32,64),(16,64)):
            original=score_table(chosen,key,key_pos,query_pos)
            for name,x in cuts.items():
                reduced=score_table(chosen,key,key_pos,query_pos,x)
                results.append(dict(query_panel=panel_name,cut=name,
                    position_pair=[key_pos,query_pos],
                    **fidelity(original,reduced,value_weight,test_ratio)))
            print("full vocab",panel_name,key_pos,query_pos,flush=True)
    source=json.loads((ART/"head0_full_function_natural_pairs.json").read_text())
    cases=[r for r in source["records"] if r["method"]=="bond_rank64"]
    pairs,gate=natural_pairs(cases,state,cuts,value_weight*value_scale)
    # value_weight was normalized for full-vocab metric; the source pair
    # energies use the unnormalized native OV metric. Undo normalization.
    result=dict(layer=0,head=0,full_key_vocabulary=vocab,
        cuts={name:dict(rank=int(np.linalg.matrix_rank(x,tol=1e-8)),
                        frobenius=float(np.linalg.norm(x))) for name,x in cuts.items()},
        query_panels={name:ids for name,ids in qids.items()},
        test_position_pairs=[[7,8],[32,64],[16,64]],
        fit_fineweb_documents=[0,15000],heldout_fineweb_documents=[17500,20000],
        test_frequency_sha256=hashlib.sha256(test_frequency.tobytes()).hexdigest(),
        natural_pair_reference_gate=gate,
        scope="saved QK1 bond cuts under identical complete-head pair-output metric, full key vocabulary and heldout query IDs; natural-pair records use same prior 512-document samples; no new full-model run",
        full_vocab_results=results,natural_pair_results=pairs)
    (ART/"head0_bond_metric_panel.json").write_text(json.dumps(result,indent=2)+"\n")
    for name in cuts:
        selected=[r for r in results if r["query_panel"]=="mixed" and
                  r["position_pair"]==[7,8] and r["cut"]==name][0]
        print(name,selected["uniform_head_pair_mse"],
              selected["corpus_head_pair_mse"],flush=True)
    return result


if __name__=="__main__":run()
