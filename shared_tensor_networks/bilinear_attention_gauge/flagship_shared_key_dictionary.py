"""Weight-only versus usage-weighted key dictionaries shared by nine heads.

The dictionary is a low-rank factorization of the complete pre-RoPE layer-0
QK1 key table on all vocabulary tokens. Evaluation includes QK2, OV, RoPE,
and summed 9-head output on held-out token pairs; it is not a full-model test.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import scipy.linalg as sla
import torch
from torch.nn import functional as F

from flagship_matrix_free import (
    CORPUS,HD,SNAPSHOT,project,rotary,sample_pairs,sample_corpus_pairs,
)
from flagship_ood_corpora import DATASETS
from flagship_transfer import ART


HEADS=9
WIDTH=HEADS*HD
SHARED_RANKS=(128,256,384,576)
INDEPENDENT_RANK=64


def project_all_heads(state,ids,names,chunk=2048):
    ids=np.asarray(ids,dtype=np.int64)
    lambdas=state['transformer.h.0.lambdas'].to(torch.float64)
    weights={name:state[f'transformer.h.0.attn.{name}.weight'].
             to(torch.float64).T.contiguous() for name in names}
    result={name:np.empty((len(ids),HEADS,HD),dtype=np.float32)
            for name in names}
    for start in range(0,len(ids),chunk):
        stop=min(start+chunk,len(ids))
        emb=state['transformer.wte.weight'][
            torch.from_numpy(ids[start:stop])].to(torch.float64)
        x0=F.rms_norm(emb,(WIDTH,))
        h=F.rms_norm((lambdas[0]+lambdas[1])*x0,(WIDTH,))
        for name,weight in weights.items():
            z=(h@weight).reshape(-1,HEADS,HD)
            if name!='c_v':z=F.rms_norm(z,(HD,))
            result[name][start:stop]=z.to(torch.float32).numpy()
    return result


def leading(gram,rank):
    dim=gram.shape[0]
    value,vector=sla.eigh(gram,subset_by_index=(dim-rank,dim-1),
                          check_finite=False)
    vector=vector[:,::-1].copy()
    return vector.astype(np.float32),value[::-1]


def rotate_all(z,pos,cos,sin):
    left,right=np.split(z,2,axis=-1)
    c=cos[pos,None,:];s=sin[pos,None,:]
    return np.concatenate((left*c+right*s,-left*s+right*c),axis=-1)


def sampled_pairs(corpus_name,path,begin,end,n,seed,vocab):
    rng=np.random.default_rng(seed)
    if corpus_name=='uniform':
        return sample_pairs(rng,vocab,n,128,position_mode='cross')
    corpus=np.load(path,mmap_mode='r')
    return sample_corpus_pairs(rng,corpus,n,128,begin,end,
                               position_mode='cross')


def evaluate(state,key_table,bases,independent,n_pairs=2048,
             eval_seed=631):
    vocab=len(key_table)
    laws={'uniform':(None,0,0),
          'fineweb_test':(CORPUS,17500,20000),
          **{name:(path,0,1200) for name,path in DATASETS.items()}}
    samples={name:sampled_pairs(name,path,begin,end,n_pairs,eval_seed+index,
                                vocab)
             for index,(name,(path,begin,end)) in enumerate(laws.items())}
    qids=np.unique(np.concatenate([sample[0] for sample in samples.values()]))
    kids=np.unique(np.concatenate([sample[1] for sample in samples.values()]))
    query=project_all_heads(state,qids,('c_q','c_q2'))
    key=project_all_heads(state,kids,('c_k2','c_v'))
    # Check the vectorized projection against the pre-existing per-head path.
    for head in (0,4,8):
        exact=project(state,qids[:4],head,('c_q','c_q2'))
        for name in exact:
            error=np.max(np.abs(query[name][:4,head].astype(np.float64)-
                                exact[name]))
            assert error<2e-5,(head,name,error)
    key1=key_table[kids]
    output=state['transformer.h.0.attn.c_proj.weight'].to(
        torch.float32).numpy()
    cos,sin=rotary(128)
    records=[]
    for law,(qid,kid,ipos,jpos) in samples.items():
        qi=np.searchsorted(qids,qid);ki=np.searchsorted(kids,kid)
        q1=rotate_all(query['c_q'][qi],ipos,cos,sin)
        q2=rotate_all(query['c_q2'][qi],ipos,cos,sin)
        k2=rotate_all(key['c_k2'][ki],jpos,cos,sin)
        value=key['c_v'][ki]
        kflat=key1[ki].reshape(n_pairs,WIDTH)
        exact_k1=rotate_all(key1[ki],jpos,cos,sin)
        s1=np.sum(q1*exact_k1,axis=-1)/HD
        s2=np.sum(q2*k2,axis=-1)/HD
        native=((s1*s2)[...,None]*value).reshape(n_pairs,WIDTH)
        reference=native.astype(np.float32)@output.T
        denominator=float(np.sum(reference.astype(np.float64)**2))
        assert denominator>0
        for method,basis in bases.items():
            for rank in SHARED_RANKS:
                u=basis[:,:rank]
                reduced=(kflat@u)@u.T
                reduced=rotate_all(reduced.reshape(n_pairs,HEADS,HD),
                                   jpos,cos,sin)
                changed=np.sum(q1*reduced,axis=-1)/HD
                delta=(((changed-s1)*s2)[...,None]*value).reshape(
                    n_pairs,WIDTH)
                out=delta.astype(np.float32)@output.T
                records.append(dict(law=law,method=method,
                    representation='shared',rank=rank,
                    relative_pair_output_mse=float(np.sum(
                        out.astype(np.float64)**2)/denominator)))
        for method,parts in independent.items():
            reduced=np.empty_like(key1[ki])
            for head,basis in enumerate(parts):
                x=key1[ki,head]
                reduced[:,head]=(x@basis)@basis.T
            reduced=rotate_all(reduced,jpos,cos,sin)
            changed=np.sum(q1*reduced,axis=-1)/HD
            delta=(((changed-s1)*s2)[...,None]*value).reshape(n_pairs,WIDTH)
            out=delta.astype(np.float32)@output.T
            records.append(dict(law=law,method=method,
                representation='independent',rank=INDEPENDENT_RANK,
                relative_pair_output_mse=float(np.sum(
                    out.astype(np.float64)**2)/denominator)))
        print('pair test',law,flush=True)
    return records,dict(n_query_ids=len(qids),n_key_ids=len(kids),
                        projection_gate='three heads and four token IDs <2e-5')


def run(n_pairs=2048,eval_seed=631):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    vocab=state['transformer.wte.weight'].shape[0]
    corpus=np.load(CORPUS,mmap_mode='r')
    frequency=np.bincount(np.asarray(corpus[:15000]).ravel(),
                          minlength=vocab).astype(np.float64)
    weights={'uniform':np.ones(vocab),
             'half_corpus':.5+.5*frequency/frequency.mean()}
    key_table=project_all_heads(state,np.arange(vocab),('c_k',))['c_k']
    flat=key_table.reshape(vocab,WIDTH)
    bases={};independent={};geometry=[]
    eval_grams={name:(flat.astype(np.float64)*w[:,None]).T@
                flat.astype(np.float64) for name,w in weights.items()}
    for name,gram in eval_grams.items():
        basis,eigen=leading(gram,max(SHARED_RANKS))
        bases[name]=basis
        independent[name]=[]
        w=weights[name]
        for head in range(HEADS):
            part=key_table[:,head].astype(np.float64)
            hgram=(part*w[:,None]).T@part
            h_basis,_=leading(hgram,INDEPENDENT_RANK)
            independent[name].append(h_basis)
        print('fit',name,flush=True)
    for method,basis in bases.items():
        for rank in SHARED_RANKS:
            u=basis[:,:rank].astype(np.float64)
            for law,gram in eval_grams.items():
                geometry.append(dict(method=method,representation='shared',
                    rank=rank,eval_metric=law,
                    relative_key_mse=float(1-np.trace(u.T@gram@u)/
                                           np.trace(gram))))
    for method,parts in independent.items():
        for law,w in weights.items():
            denominator=0.;residual=0.
            for head,basis in enumerate(parts):
                part=key_table[:,head].astype(np.float64)
                gram=(part*w[:,None]).T@part
                u=basis.astype(np.float64)
                denominator+=np.trace(gram)
                residual+=np.trace(gram)-np.trace(u.T@gram@u)
            geometry.append(dict(method=method,representation='independent',
                rank=INDEPENDENT_RANK,eval_metric=law,
                relative_key_mse=float(residual/denominator)))
    pairs,coverage=evaluate(state,key_table,bases,independent,n_pairs,
                            eval_seed)
    result=dict(layer=0,heads=list(range(HEADS)),vocab=vocab,
        key='complete pre-RoPE, normalized layer-0 QK1 key table',
        fit_corpus_documents=[0,15000],
        heldout_fineweb_documents=[17500,20000],
        n_pairs_per_law=n_pairs,eval_seed=eval_seed,
        shared_ranks=list(SHARED_RANKS),
        independent_rank_per_head=INDEPENDENT_RANK,
        parameter_count_definition='shared V*m+m*(9*128); independent 9*(V*r+r*128)',
        frequency_sha256=hashlib.sha256(frequency.tobytes()).hexdigest(),
        coverage=coverage,geometry=geometry,pair_tests=pairs,
        scope='all-vocabulary weight-only/usage-weighted shared key dictionary; held-out pair output across nine heads, no full-model edit or semantic labels')
    stem=('all_head_shared_key_dictionary' if (n_pairs,eval_seed)==(2048,631)
          else f'all_head_shared_key_dictionary_n{n_pairs}_s{eval_seed}')
    (ART/f'{stem}.json').write_text(
        json.dumps(result,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}_bases.npz',
        **{f'shared_{k}':v for k,v in bases.items()},
        **{f'independent_{k}':np.stack(v) for k,v in independent.items()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-pairs',type=int,default=2048)
    p.add_argument('--eval-seed',type=int,default=631)
    a=p.parse_args()
    run(a.n_pairs,a.eval_seed)
