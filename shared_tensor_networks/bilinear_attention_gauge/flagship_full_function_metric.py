"""Full-key-vocabulary first-layer attention response reductions.

For fixed query/key positions, form the *complete head response* rather than
one physical QK bond. All 50,304 key IDs and all 128 value/output directions
are retained. Query token anchors and positions are declared limits.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import scipy.linalg as sla
import torch

from flagship_matrix_free import CORPUS,HD,SNAPSHOT,project,rotate,rotary
from flagship_transfer import ART


def choose_queries(frequency,vocab,n_each,seed):
    rng=np.random.default_rng(seed)
    uniform=rng.choice(vocab,size=2*n_each,replace=False)
    remaining=np.ones(vocab,dtype=bool)
    remaining[uniform]=False
    probability=np.where(remaining,frequency,0).astype(float)
    probability/=probability.sum()
    corpus=rng.choice(vocab,size=2*n_each,replace=False,p=probability)
    assert not set(uniform)&set(corpus)
    return np.r_[uniform[:n_each],corpus[:n_each]],\
           np.r_[uniform[n_each:],corpus[n_each:]],dict(
               train_uniform=uniform[:n_each].tolist(),
               train_corpus=corpus[:n_each].tolist(),
               test_uniform=uniform[n_each:].tolist(),
               test_corpus=corpus[n_each:].tolist())


def response_matrix(query_projection,key_projection,c,key_pos,query_pos,cut=None):
    cos,sin=rotary(max(query_pos,key_pos)+1)
    q1=rotate(query_projection["c_q"],
              np.full(len(query_projection["c_q"]),query_pos),cos,sin)
    q2=rotate(query_projection["c_q2"],
              np.full(len(query_projection["c_q2"]),query_pos),cos,sin)
    k1=rotate(key_projection["c_k"],
              np.full(len(key_projection["c_k"]),key_pos),cos,sin)
    k2=rotate(key_projection["c_k2"],
              np.full(len(key_projection["c_k2"]),key_pos),cos,sin)
    if cut is not None:
        q1=q1@cut
    score=((q1@k1.T)/(HD))*((q2@k2.T)/(HD))
    # Row index is (query anchor, orthonormalized head-output coordinate).
    matrix=np.empty((len(q1)*HD,len(c)),dtype=np.float32)
    ct=c.T.astype(np.float32)
    for i in range(len(q1)):
        matrix[i*HD:(i+1)*HD]=ct*score[i][None,:]
    return matrix


def exact_right(matrix,rank,spectrum_rank=256):
    """Exact leading right singular vectors via the smaller row Gram matrix."""
    gram=matrix@matrix.T
    lo=gram.shape[0]-spectrum_rank
    eigen,left=sla.eigh(gram,subset_by_index=(lo,gram.shape[0]-1),
                        check_finite=False)
    eigen=eigen[::-1];left=left[:,::-1]
    assert np.all(eigen[:rank]>0)
    right=(left[:,:rank].T@matrix)/np.sqrt(eigen[:rank,None])
    orth=float(np.linalg.norm(right@right.T-np.eye(rank)))
    assert orth<1e-3,orth
    total=float(np.sum(matrix.astype(np.float64)**2))
    optimal={str(k):1-float(np.sum(eigen[:k]))/total
             for k in (16,32,64,128,256)}
    return right,np.sqrt(eigen).tolist(),optimal,orth


def score(matrix,right,fit_weight,test_weight,ranks):
    root=np.sqrt(fit_weight).astype(np.float32)
    m=matrix*root[None,:]
    denominator_uniform=np.sum(matrix.astype(np.float64)**2)
    denominator_test=np.sum((matrix.astype(np.float64)**2)*test_weight[None,:])
    scored={}
    for rank in ranks:
        basis=right[:rank]
        reconstructed=(m@basis.T)@basis
        delta=(m-reconstructed)/root[None,:]
        squared=delta.astype(np.float64)**2
        scored[str(rank)]=dict(
            uniform_relative_mse=float(np.sum(squared)/denominator_uniform),
            corpus_weighted_relative_mse=float(np.sum(squared*test_weight[None,:])/
                                      denominator_test))
    return scored


def direct_score(original,reduced,test_weight):
    squared=(original.astype(np.float64)-reduced.astype(np.float64))**2
    energy=original.astype(np.float64)**2
    return dict(uniform_relative_mse=float(squared.sum()/energy.sum()),
        corpus_weighted_relative_mse=float(
            np.sum(squared*test_weight[None,:])/
            np.sum(energy*test_weight[None,:])))


def run(n_each=16,ranks=(16,32,64),seed=1029):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/"pytorch_model.bin",map_location="cpu",
                     weights_only=True,mmap=True)
    bond_cut=np.load(ART/"head0_l128_sequence_fit_rank64_cw0.npz")["cut"]
    assert np.linalg.matrix_rank(bond_cut,tol=1e-8)==64
    vocab=state["transformer.wte.weight"].shape[0]
    corpus=np.load(CORPUS,mmap_mode="r")
    frequency=np.bincount(np.asarray(corpus[:15000]).ravel(),minlength=vocab)
    test_frequency=np.bincount(np.asarray(corpus[17500:20000]).ravel(),
                               minlength=vocab)
    train,test,query_ids=choose_queries(frequency,vocab,n_each,seed)
    rng=np.random.default_rng(seed+1)
    excluded=np.ones(vocab,dtype=bool)
    excluded[np.r_[train,test]]=False
    probability=np.where(excluded,test_frequency,0).astype(float)
    probability/=probability.sum()
    heldout_queries=rng.choice(vocab,size=n_each,replace=False,p=probability)
    query_ids["test_heldout_corpus"]=heldout_queries.tolist()
    all_ids=np.arange(vocab,dtype=np.int64)
    key=project(state,all_ids,0,("c_k","c_k2","c_v"))
    query=project(state,np.r_[train,test,heldout_queries],0,("c_q","c_q2"))
    output=state["transformer.h.0.attn.c_proj.weight"][:,:HD].to(torch.float64).numpy()
    gram=output.T@output
    eigen,vector=np.linalg.eigh(gram)
    root=vector@np.diag(np.sqrt(np.maximum(eigen,0)))@vector.T
    c=key["c_v"]@root
    # Exact output-norm metric: ||score * c_k||^2 equals the native
    # 1152-dimensional OV output norm for each pair.
    for k in (0,1,17,50303):
        native=key["c_v"][k]@output.T
        assert abs(np.sum(native**2)-np.sum(c[k]**2))<1e-10*np.sum(native**2)
    query_train={name:value[:len(train)] for name,value in query.items()}
    query_test={name:value[len(train):] for name,value in query.items()}
    del state,query
    fit=response_matrix(query_train,key,c,7,8)
    fit_bond=response_matrix(query_train,key,c,7,8,bond_cut)
    del query_train
    mean_frequency=frequency.mean()
    ratio=frequency/mean_frequency
    test_ratio=test_frequency/test_frequency.mean()
    measures={"uniform":np.ones(vocab),
              "half_corpus":.5+.5*ratio,
              "mostly_corpus":.1+.9*ratio}
    max_rank=max(ranks)
    fits={};bases={}
    bond_fit=direct_score(fit,fit_bond,ratio)
    del fit_bond
    for name,weight in measures.items():
        right,singular,optimal,orth=exact_right(
            fit*np.sqrt(weight)[None,:].astype(np.float32),max_rank)
        bases[name]=right
        fits[name]=dict(singular_values=singular,
            exact_optimal_weighted_residual=optimal,
            right_basis_orthogonality_error=orth,
            train=score(fit,right,weight,ratio,ranks))
        print("fit",name,optimal["64"],flush=True)
    del fit
    tests={};bond_tests={}
    for query_kind,sl in (("uniform",slice(0,n_each)),
                          ("corpus",slice(n_each,2*n_each)),
                          ("heldout_corpus",slice(2*n_each,3*n_each))):
        selected={name:value[sl] for name,value in query_test.items()}
        for key_pos,query_pos in ((7,8),(32,64)):
            matrix=response_matrix(selected,key,c,key_pos,query_pos)
            bond_matrix=response_matrix(selected,key,c,key_pos,query_pos,bond_cut)
            bond_tests[f"{query_kind}_q_{key_pos}_{query_pos}"]=direct_score(
                matrix,bond_matrix,test_ratio)
            del bond_matrix
            for name,weight in measures.items():
                tests[f"{query_kind}_q_{key_pos}_{query_pos}_{name}"]=score(
                    matrix,bases[name],weight,test_ratio,ranks)
            print("test",query_kind,key_pos,query_pos,flush=True)
    frequency_hash=hashlib.sha256(frequency.tobytes()).hexdigest()
    test_frequency_hash=hashlib.sha256(test_frequency.tobytes()).hexdigest()
    result=dict(head=0,layer=0,full_key_vocabulary=vocab,
        head_output_metric="exact 1152-dimensional OV output norm, represented in 128 value coordinates",
        fit_position=[7,8],test_positions=[[7,8],[32,64]],
        n_query_anchors_per_law=n_each,query_ids=query_ids,
        fit_documents=[0,15000],test_documents=[17500,20000],
        frequency_sha256=frequency_hash,
        heldout_frequency_sha256=test_frequency_hash,
        decomposition="exact leading SVD of each weighted fixed-position training table via 4096x4096 row Gram eigensolve; float32 matrix with float64 error accumulation",
        measures="uniform all-token key weight; half uniform/FineWeb key frequency; 10% uniform/90% frequency; no labels",
        structured_bond_control="existing sequence-fitted rank-64 QK1 bond matrix, evaluated on identical pair-response domain; unlike flat key rank it preserves QK2 multiplication and value dependence; parameter counts and function classes differ",
        scope="complete first-layer head pair response over all key IDs, but only declared query anchors and two position pairs; heldout corpus documents choose evaluation queries/key weighting only; no full-model edit yet",
        fits=fits,bond_fit=bond_fit,tests=tests,bond_tests=bond_tests)
    (ART/"head0_full_function_metric.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(ART/"head0_full_function_key_bases.npz",**bases)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--n-each",type=int,default=16)
    p.add_argument("--seed",type=int,default=1029)
    args=p.parse_args()
    run(args.n_each,(16,32,64),args.seed)
