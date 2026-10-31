"""Position coverage versus rank budget for effective-score canonicalization."""

from __future__ import annotations

import json

import numpy as np
import torch

from flagship_effective_score_metric import exact_basis,evaluate,fidelity,score_table
from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project
from flagship_transfer import ART


FIT_OFFSETS=(1,2,4,8,16,32)
TEST_OFFSETS=(1,24,32,48)


def run():
    torch.set_num_threads(4)
    earlier=json.loads((ART/"head0_effective_score_metric.json").read_text())
    n=earlier["n_query_anchors_per_law"]
    query_ids=earlier["query_ids"]
    train=query_ids["train_uniform"]+query_ids["train_corpus"]
    test=query_ids["test_heldout_corpus"]
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    vocab=state["transformer.wte.weight"].shape[0]
    corpus=np.load(CORPUS,mmap_mode="r")
    fit_freq=np.bincount(np.asarray(corpus[:15000]).ravel(),minlength=vocab)
    test_freq=np.bincount(np.asarray(corpus[17500:20000]).ravel(),minlength=vocab)
    key=project(state,np.arange(vocab),0,("c_k","c_k2","c_v"))
    q_train=project(state,np.array(train),0,("c_q","c_q2"))
    q_test=project(state,np.array(test),0,("c_q","c_q2"))
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64).numpy()
    value_weight=np.einsum("ki,ij,kj->k",key["c_v"],output.T@output,key["c_v"])
    value_weight/=value_weight.mean()
    fit_ratio=fit_freq/fit_freq.mean();test_ratio=test_freq/test_freq.mean()
    measures={"uniform":value_weight,
              "half_corpus":value_weight*(.5+.5*fit_ratio),
              "mostly_corpus":value_weight*(.1+.9*fit_ratio)}
    bond=np.load(ART/"head0_l128_sequence_fit_rank64_cw0.npz")["cut"]
    fit=np.concatenate([score_table(q_train,key,64-offset,64)
                        for offset in FIT_OFFSETS],axis=0)
    fitted={};bases={}
    for name,weight in measures.items():
        right,optimal,orth,spectrum=exact_basis(fit,weight,128)
        bases[name]=right
        fitted[name]=dict(exact_optimal_weighted_residual=optimal,
            right_basis_orthogonality_error=orth,
            train=evaluate(fit,right,weight,value_weight,fit_ratio,(64,128)))
        print("fit",name,optimal["64"],flush=True)
    del fit
    original_bases=np.load(ART/"head0_effective_score_key_bases.npz")
    results={};bond_results={}
    for offset in TEST_OFFSETS:
        target=score_table(q_test,key,64-offset,64)
        edited=score_table(q_test,key,64-offset,64,bond)
        bond_results[str(offset)]=fidelity(target,edited,value_weight,test_ratio)
        for name,weight in measures.items():
            results[f"multi_{name}_{offset}"]=evaluate(target,bases[name],weight,
                value_weight,test_ratio,(64,128))
            results[f"single_{name}_{offset}"]=evaluate(target,original_bases[name],
                weight,value_weight,test_ratio,(64,128))
        print("test",offset,flush=True)
    result=dict(layer=0,head=0,full_key_vocabulary=vocab,
        fit_query_ids=train,heldout_query_ids=test,
        fit_query_position=64,fit_relative_offsets=list(FIT_OFFSETS),
        test_query_position=64,test_relative_offsets=list(TEST_OFFSETS),
        fit_fineweb_documents=[0,15000],heldout_fineweb_documents=[17500,20000],
        scope="exact effective QK1*QK2 score SVD across multiple RoPE relative offsets; full key vocabulary and exact OV output metric; no full-model propagation",
        fitted=fitted,tests=results,bond_tests=bond_results)
    (ART/"head0_effective_score_positions.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(ART/"head0_effective_score_multi_position_bases.npz",**bases)
    return result


if __name__=="__main__":run()
