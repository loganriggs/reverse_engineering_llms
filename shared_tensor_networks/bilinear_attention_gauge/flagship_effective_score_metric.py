"""Branch-swap-invariant effective-score SVD with exact value/OV map.

The score table is S(q,k)=QK1(q,k)*QK2(q,k), at declared positions.
Its key-mode rank is tested while retaining all 50,304 key IDs and the
original value/OV function. Data weighting is a measure, not task labels.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import scipy.linalg as sla
import torch

from flagship_full_function_metric import choose_queries
from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary
from flagship_transfer import ART


def score_table(query,key,key_pos,query_pos,bond=None):
    cos,sin=rotary(max(query_pos,key_pos)+1)
    q1=rotate(query["c_q"],np.full(len(query["c_q"]),query_pos),cos,sin)
    q2=rotate(query["c_q2"],np.full(len(query["c_q2"]),query_pos),cos,sin)
    k1=rotate(key["c_k"],np.full(len(key["c_k"]),key_pos),cos,sin)
    k2=rotate(key["c_k2"],np.full(len(key["c_k2"]),key_pos),cos,sin)
    if bond is not None:q1=q1@bond
    return (((q1@k1.T)/HD)*((q2@k2.T)/HD)).astype(np.float32)


def exact_basis(table,weight,rank):
    matrix=table*np.sqrt(weight)[None,:].astype(np.float32)
    gram=matrix@matrix.T
    eigen,left=sla.eigh(gram,check_finite=False)
    eigen=eigen[::-1];left=left[:,::-1]
    assert np.all(eigen[:rank]>0)
    right=(left[:,:rank].T@matrix)/np.sqrt(eigen[:rank,None])
    orth=float(np.linalg.norm(right@right.T-np.eye(rank)))
    assert orth<1e-3,orth
    total=float(np.sum(matrix.astype(np.float64)**2))
    optimal={str(k):1-float(np.sum(eigen[:k]))/total
             for k in (16,32,64,128)}
    return right,optimal,orth,np.sqrt(eigen[:128]).tolist()


def fidelity(original,reduced,value_weight,corpus_ratio):
    squared=(original.astype(np.float64)-reduced.astype(np.float64))**2
    reference=original.astype(np.float64)**2
    numerator=squared*value_weight[None,:]
    denominator=reference*value_weight[None,:]
    return dict(uniform_head_pair_mse=float(numerator.sum()/denominator.sum()),
        corpus_head_pair_mse=float((numerator*corpus_ratio[None,:]).sum()/
                                   (denominator*corpus_ratio[None,:]).sum()),
        uniform_score_mse=float(squared.sum()/reference.sum()))


def evaluate(table,right,fit_weight,value_weight,corpus_ratio,ranks):
    root=np.sqrt(fit_weight).astype(np.float32)
    weighted=table*root[None,:]
    out={}
    for rank in ranks:
        basis=right[:rank]
        reduced=((weighted@basis.T)@basis)/root[None,:]
        out[str(rank)]=fidelity(table,reduced,value_weight,corpus_ratio)
    return out


def run(n_each=128,seed=3029,ranks=(16,32,64,128),query_panel="mixed"):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    vocab=state["transformer.wte.weight"].shape[0]
    corpus=np.load(CORPUS,mmap_mode="r")
    fit_frequency=np.bincount(np.asarray(corpus[:15000]).ravel(),minlength=vocab)
    test_frequency=np.bincount(np.asarray(corpus[17500:20000]).ravel(),minlength=vocab)
    if query_panel=="mixed":
        train,test,query_ids=choose_queries(fit_frequency,vocab,n_each,seed)
        test_names=("uniform","corpus")
    elif query_panel=="uniform":
        rng_uniform=np.random.default_rng(seed)
        chosen=rng_uniform.choice(vocab,size=4*n_each,replace=False)
        train=chosen[:2*n_each];test=chosen[2*n_each:]
        query_ids=dict(train_uniform=train.tolist(),
                       test_uniform=test[:n_each].tolist(),
                       test_uniform2=test[n_each:].tolist())
        test_names=("uniform","uniform2")
    else:
        raise ValueError(query_panel)
    rng=np.random.default_rng(seed+1)
    allowed=np.ones(vocab,dtype=bool)
    allowed[np.r_[train,test]]=False
    probability=np.where(allowed,test_frequency,0).astype(float)
    probability/=probability.sum()
    heldout=rng.choice(vocab,size=n_each,replace=False,p=probability)
    query_ids["test_heldout_corpus"]=heldout.tolist()
    query=project(state,np.r_[train,test,heldout],0,("c_q","c_q2"))
    key=project(state,np.arange(vocab),0,("c_k","c_k2","c_v"))
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64).numpy()
    gram=output.T@output
    value_weight=np.einsum("ki,ij,kj->k",key["c_v"],gram,key["c_v"])
    assert np.all(value_weight>0)
    norm_gate=max(abs(value_weight[k]-np.sum((key["c_v"][k]@output.T)**2))
                  for k in (0,1,17,vocab-1))
    assert norm_gate<1e-6,norm_gate
    value_weight/=np.mean(value_weight)
    bond=np.load(ART/"head0_l128_sequence_fit_rank64_cw0.npz")["cut"]
    fit_ratio=fit_frequency/fit_frequency.mean()
    test_ratio=test_frequency/test_frequency.mean()
    measures={"uniform":value_weight,
              "half_corpus":value_weight*(.5+.5*fit_ratio),
              "mostly_corpus":value_weight*(.1+.9*fit_ratio)}
    train_q={name:z[:len(train)] for name,z in query.items()}
    fit=score_table(train_q,key,7,8)
    swapped_query={"c_q":train_q["c_q2"],"c_q2":train_q["c_q"]}
    swapped_key={"c_k":key["c_k2"],"c_k2":key["c_k"]}
    branch_swap_gate=float(np.max(np.abs(fit-score_table(
        swapped_query,swapped_key,7,8))))
    assert branch_swap_gate<1e-6,branch_swap_gate
    fit_bond=score_table(train_q,key,7,8,bond)
    bond_fit=fidelity(fit,fit_bond,value_weight,fit_ratio)
    del fit_bond
    fits={};bases={}
    for name,weight in measures.items():
        right,optimal,orth,spectrum=exact_basis(fit,weight,max(ranks))
        bases[name]=right
        fits[name]=dict(exact_optimal_weighted_residual=optimal,
                        right_basis_orthogonality_error=orth,
                        leading_singular_values=spectrum,
                        train=evaluate(fit,right,weight,value_weight,fit_ratio,ranks))
        print("fit",name,optimal["64"],flush=True)
    del fit
    tests={};bond_tests={}
    query_test={name:z[len(train):] for name,z in query.items()}
    for query_kind,sl in ((test_names[0],slice(0,n_each)),
                          (test_names[1],slice(n_each,2*n_each)),
                          ("heldout_corpus",slice(2*n_each,3*n_each))):
        selected={name:z[sl] for name,z in query_test.items()}
        for key_pos,query_pos in ((7,8),(32,64)):
            target=score_table(selected,key,key_pos,query_pos)
            bond_target=score_table(selected,key,key_pos,query_pos,bond)
            tag=f"{query_kind}_q_{key_pos}_{query_pos}"
            bond_tests[tag]=fidelity(target,bond_target,value_weight,test_ratio)
            for name,weight in measures.items():
                tests[tag+"_"+name]=evaluate(target,bases[name],weight,
                    value_weight,test_ratio,ranks)
            print("test",tag,flush=True)
    result=dict(layer=0,head=0,full_key_vocabulary=vocab,
        query_panel=query_panel,
        n_query_anchors_per_law=n_each,query_ids=query_ids,
        fit_position=[7,8],test_positions=[[7,8],[32,64]],
        fit_fineweb_documents=[0,15000],heldout_fineweb_documents=[17500,20000],
        fit_frequency_sha256=hashlib.sha256(fit_frequency.tobytes()).hexdigest(),
        heldout_frequency_sha256=hashlib.sha256(test_frequency.tobytes()).hexdigest(),
        object="complete QK1*QK2 scalar score over declared query anchors and all key IDs; original value/OV map retained exactly; invariant to swapping QK branches",
        measure="squared head-output pair response gives each key score error weight ||OV(key)||^2; uniform versus FineWeb key-frequency mixtures",
        decomposition="exact leading SVD of weighted score table via row Gram eigensolve",
        value_output_norm_gate=norm_gate,
        qk_branch_swap_max_abs=branch_swap_gate,
        structured_control="existing sequence-fitted rank-64 QK1 bond cut; resource and function class differ",
        fits=fits,bond_fit=bond_fit,tests=tests,bond_tests=bond_tests)
    suffix="" if query_panel=="mixed" else "_uniform_queries"
    (ART/f"head0_effective_score_metric{suffix}.json").write_text(
        json.dumps(result,indent=2)+"\n")
    np.savez_compressed(ART/f"head0_effective_score_key_bases{suffix}.npz",**bases)
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--query-panel",choices=("mixed","uniform"),default="mixed")
    args=p.parse_args()
    run(query_panel=args.query_panel)
