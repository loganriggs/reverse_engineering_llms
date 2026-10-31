"""Fit a shared first-layer key subspace for the nine-head pair response.

The fit changes only an orthonormal rank-256 projection of native pre-RoPE
QK1 keys. All query, QK2, value, and output weights remain frozen. Training
and selection use disjoint FineWeb documents; test includes external corpora.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from flagship_matrix_free import CORPUS,HD,SNAPSHOT,rotary
from flagship_ood_corpora import DATASETS
from flagship_shared_key_dictionary import (
    HEADS,WIDTH,project_all_heads,rotate_all,sampled_pairs,
)
from flagship_transfer import ART


def make_panels(state,n_train,n_val,n_test,test_seed_offset=0):
    vocab=state['transformer.wte.weight'].shape[0]
    laws={
        'train_uniform':('uniform',None,0,0,n_train,1901),
        'train_fineweb':('corpus',CORPUS,0,15000,n_train,1902),
        'validation_uniform':('uniform',None,0,0,n_val,2901),
        'validation_fineweb':('corpus',CORPUS,15000,17500,n_val,2902),
        'test_uniform':('uniform',None,0,0,n_test,3901+test_seed_offset),
        'test_fineweb':('corpus',CORPUS,17500,20000,n_test,
                        3902+test_seed_offset),
        **{f'test_{name}':('corpus',path,0,1200,n_test,
                          3903+i+test_seed_offset)
           for i,(name,path) in enumerate(DATASETS.items())},
    }
    sampled={name:sampled_pairs(distribution,path,begin,end,n,seed,vocab)
             for name,(distribution,path,begin,end,n,seed) in laws.items()}
    qids=np.unique(np.concatenate([part[0] for part in sampled.values()]))
    kids=np.unique(np.concatenate([part[1] for part in sampled.values()]))
    query=project_all_heads(state,qids,('c_q','c_q2'))
    key=project_all_heads(state,kids,('c_k','c_k2','c_v'))
    out=state['transformer.h.0.attn.c_proj.weight'].to(
        torch.float32).numpy()
    gram=np.empty((len(kids),HEADS,HEADS),dtype=np.float32)
    for start in range(0,len(kids),512):
        stop=min(start+512,len(kids))
        value=key['c_v'][start:stop]
        ov=np.stack([value[:,head]@out[:,head*HD:(head+1)*HD].T
                     for head in range(HEADS)],axis=1)
        gram[start:stop]=np.einsum('nho,njo->nhj',ov,ov,
                                  optimize=True)
    cos,sin=rotary(128)
    panels={};formula_gate=None
    for law,(qid,kid,ipos,jpos) in sampled.items():
        qi=np.searchsorted(qids,qid);ki=np.searchsorted(kids,kid)
        qpost=rotate_all(query['c_q'][qi],ipos,cos,sin)
        qeff=rotate_all(qpost,jpos,cos,-sin)
        q2post=rotate_all(query['c_q2'][qi],ipos,cos,sin)
        k2post=rotate_all(key['c_k2'][ki],jpos,cos,sin)
        kpre=key['c_k'][ki]
        s1=np.sum(qeff*kpre,axis=-1)/HD
        s2=np.sum(q2post*k2post,axis=-1)/HD
        a=s1*s2
        g=gram[ki]
        reference=np.einsum('nh,nhj,nj->n',a,g,a,optimize=True)
        assert np.min(reference)>-1e-2
        if formula_gate is None:
            kpost=rotate_all(kpre[:32],jpos[:32],cos,sin)
            check=np.sum(qpost[:32]*kpost,axis=-1)/HD
            score_error=float(np.max(np.abs(check-s1[:32])))
            assert score_error<1e-4,score_error
            v=key['c_v'][ki[:32]]
            explicit=((a[:32,...,None]*v).reshape(32,WIDTH)@
                      out.T)
            norm=np.sum(explicit.astype(np.float64)**2,axis=1)
            norm_error=float(np.max(np.abs(norm-reference[:32])/
                                    np.maximum(norm,1e-8)))
            assert norm_error<2e-4,norm_error
            formula_gate=dict(score_max_abs=score_error,
                              output_norm_max_relative=norm_error)
        panels[law]=dict(
            k=torch.from_numpy(kpre.reshape(-1,WIDTH).copy()),
            q=torch.from_numpy(qeff.astype(np.float32)),
            s1=torch.from_numpy(s1.astype(np.float32)),
            s2=torch.from_numpy(s2.astype(np.float32)),
            gram=torch.from_numpy(g.copy()),
            ref=torch.from_numpy(np.maximum(reference,0).astype(np.float32)))
        print('prepared',law,len(qid),flush=True)
    return panels,formula_gate,dict(query_tokens=len(qids),key_tokens=len(kids))


def error(panel,basis,index=None):
    if index is None:index=slice(None)
    k=panel['k'][index]
    q=panel['q'][index]
    s1=panel['s1'][index]
    s2=panel['s2'][index]
    gram=panel['gram'][index]
    reduced=((k@basis)@basis.T).reshape(-1,HEADS,HD)
    changed=(q*reduced).sum(-1)/HD
    delta=(changed-s1)*s2
    squared=torch.einsum('nh,nhj,nj->n',delta,gram,delta)
    return squared.sum()/panel['ref'][index].sum()


def scores(panels,basis):
    with torch.no_grad():
        return {law:float(error(panel,basis)) for law,panel in panels.items()}


def qk2_constant(panel):
    """Counterfactual pair panel with the QK2 score set to one."""
    changed=panel.copy()
    changed['s2']=torch.ones_like(panel['s2'])
    changed['ref']=torch.einsum('nh,nhj,nj->n',
                  panel['s1'],panel['gram'],panel['s1']).clamp_min(0)
    return changed


def run(n_train=8192,n_val=2048,n_test=4096,steps=160,
        batch_per_law=512,lr=.0001,seed=641,branch_weight=0.):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    panels,formula_gate,coverage=make_panels(state,n_train,n_val,n_test)
    assert 0<=branch_weight<=1
    if branch_weight:
        panels.update({f'{law}_qk2_constant':qk2_constant(panel)
                       for law,panel in list(panels.items())})
    del state
    prior=np.load(ART/'all_head_shared_key_dictionary_bases.npz')
    initial=torch.from_numpy(prior['shared_half_corpus'][:,:256].copy())
    uniform=torch.from_numpy(prior['shared_uniform'][:,:256].copy())
    initial_scores=scores(panels,initial)
    uniform_scores=scores(panels,uniform)
    parameter=torch.nn.Parameter(initial.clone())
    optimizer=torch.optim.Adam([parameter],lr=lr)
    rng=np.random.default_rng(seed)
    best=initial.clone()
    def objective(basis,split,index=None):
        uniform_index=None if index is None else index['uniform']
        fineweb_index=None if index is None else index['fineweb']
        natural=.5*(error(panels[f'{split}_uniform'],basis,uniform_index)+
                    error(panels[f'{split}_fineweb'],basis,fineweb_index))
        if not branch_weight:return natural
        edited=.5*(error(panels[f'{split}_uniform_qk2_constant'],basis,uniform_index)+
                   error(panels[f'{split}_fineweb_qk2_constant'],basis,fineweb_index))
        return (1-branch_weight)*natural+branch_weight*edited
    best_value=float(objective(initial,'validation'))
    best_step=0
    trace=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        basis=torch.linalg.qr(parameter,mode='reduced').Q
        idx={law:torch.from_numpy(rng.integers(0,n_train,size=batch_per_law))
             for law in ('uniform','fineweb')}
        loss=objective(basis,'train',idx)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter],1.)
        optimizer.step()
        if (step+1)%10==0 or step==steps-1:
            with torch.no_grad():
                basis=torch.linalg.qr(parameter,mode='reduced').Q
                val=float(objective(basis,'validation'))
                train=float(objective(basis,'train'))
                trace.append(dict(step=step+1,train=train,validation=val))
                if val<best_value:
                    best=basis.clone();best_value=val;best_step=step+1
            print('step',step+1,'train',round(train,6),
                  'val',round(val,6),flush=True)
    best_scores=scores(panels,best)
    orth=float(torch.linalg.norm(best.T@best-torch.eye(256)))
    assert orth<1e-3,orth
    output=dict(layer=0,heads=list(range(9)),rank=256,
        fit_corpus_documents=[0,15000],
        selection_corpus_documents=[15000,17500],
        test_fineweb_documents=[17500,20000],
        n_train_per_law=n_train,n_val_per_law=n_val,
        n_test_per_law=n_test,steps=steps,batch_per_law=batch_per_law,
        learning_rate=lr,seed=seed,formula_gate=formula_gate,
        branch_weight=branch_weight,
        coverage=coverage,orthogonality_error=orth,
        objective='equal uniform/FineWeb relative nine-head pair-output squared error, with stated weight on QK2-constant branch edit',
        initial_scores=initial_scores,uniform_scores=uniform_scores,
        best_step=best_step,best_validation=best_value,
        best_scores=best_scores,trace=trace,
        scope='function-aware orthogonal rank-256 shared key subspace; no held-out task labels or full-model propagation')
    stem=f'all_head_function_aware_shared_key_s{seed}_n{n_train}_t{steps}'
    if branch_weight:stem+=f'_branch{round(100*branch_weight)}'
    (ART/f'{stem}.json').write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}.npz',basis=best.numpy())
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-train',type=int,default=8192)
    p.add_argument('--n-val',type=int,default=2048)
    p.add_argument('--n-test',type=int,default=4096)
    p.add_argument('--steps',type=int,default=160)
    p.add_argument('--batch-per-law',type=int,default=512)
    p.add_argument('--lr',type=float,default=.0001)
    p.add_argument('--seed',type=int,default=641)
    p.add_argument('--branch-weight',type=float,default=0.)
    a=p.parse_args()
    run(a.n_train,a.n_val,a.n_test,a.steps,a.batch_per_law,a.lr,a.seed,
        a.branch_weight)
