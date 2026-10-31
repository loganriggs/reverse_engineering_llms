"""Exact-gauge weight versus data-response sparsity on an all-head key cut.

Only orthogonal rotations of a fixed rank-256 basis are fitted. The key
projector, nine-head attention function, and full-model logits are unchanged.
All metrics are concentration/stability proxies, not semantic labels.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import scipy.optimize as opt
import torch

from flagship_function_aware_shared_key import error,make_panels
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_dictionary import HEADS,WIDTH,project_all_heads
from flagship_transfer import ART


def rotation(seed,m):
    rng=np.random.default_rng(seed)
    q,r=np.linalg.qr(rng.normal(size=(m,m)))
    return q*np.sign(np.diag(r))[None,:]


def varimax(x,start,max_iter=40,tol=1e-7):
    """Orthogonal varimax with an explicit multi-start stationarity trace."""
    n,m=x.shape
    r=start.copy();previous=None;trace=[]
    for step in range(max_iter):
        lam=x@r
        column_energy=np.sum(lam**2,axis=0)
        grad=x.T@(lam**3-lam*(column_energy/n)[None,:])
        u,s,vt=np.linalg.svd(grad,full_matrices=False)
        candidate=u@vt
        value=float(np.sum(s))
        trace.append(value)
        if previous is not None and abs(value-previous)<=tol*max(abs(previous),1e-12):
            r=candidate;break
        r=candidate;previous=value
    return r,dict(iterations=len(trace),
                  gradient_nuclear_norm_trace=trace,
                  final_gradient_nuclear_norm=trace[-1])


def direct_response_l1(train_z,validation_z,start,steps=160,seed=0):
    """Minimize normalized token-coordinate L1 under an orthogonal gauge."""
    parameter=torch.nn.Parameter(torch.from_numpy(start.astype(np.float32)))
    train=torch.from_numpy(train_z.astype(np.float32))
    validation=torch.from_numpy(validation_z.astype(np.float32))
    optimizer=torch.optim.Adam([parameter],lr=0.0003)
    rng=np.random.default_rng(7100+seed)
    best_value=float('inf');best=None;best_step=0;trace=[]
    for step in range(steps):
        idx=torch.from_numpy(rng.integers(0,len(train),size=512))
        optimizer.zero_grad(set_to_none=True)
        q=torch.linalg.qr(parameter,mode='reduced').Q
        projected=train[idx]@q
        loss=torch.sqrt(projected.square()+1e-6).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([parameter],1.)
        optimizer.step()
        if (step+1)%10==0 or step==steps-1:
            with torch.no_grad():
                q=torch.linalg.qr(parameter,mode='reduced').Q
                score=float(torch.sqrt((validation@q).square()+1e-6).sum(-1).mean())
                trace.append(dict(step=step+1,validation_l1=score))
                if score<best_value:
                    best=q.numpy().astype(np.float64).copy()
                    best_value=score;best_step=step+1
    best=np.linalg.qr(best)[0]
    return best,dict(best_step=best_step,best_validation_l1=best_value,
                     trace=trace)


def effective_coordinates(z):
    z=np.asarray(z,dtype=np.float64)
    norm2=np.sum(z*z,axis=-1)
    return float(np.mean(np.sum(np.abs(z),axis=-1)**2/
                         np.maximum(norm2,1e-20)))


def head_participation(u):
    energy=np.sum(u.reshape(HEADS,HD,-1)**2,axis=1)
    fraction=energy/np.maximum(np.sum(energy,axis=0),1e-20)
    return float(np.mean(1/np.sum(fraction**2,axis=0)))


def pair_concentration(panel,u,constant=False):
    q=panel['q'].numpy().astype(np.float64)
    k=panel['k'].numpy().astype(np.float64)
    s2=np.ones_like(panel['s2'].numpy()) if constant else panel['s2'].numpy()
    gram=panel['gram'].numpy().astype(np.float64)
    coef=k@u
    query=np.einsum('nhd,hdr->nhr',q,u.reshape(HEADS,HD,-1),optimize=True)
    score=(query*coef[:,None,:]*s2[:,:,None]/HD).transpose(0,2,1)
    reduced=(coef[:8]@u.T).reshape(-1,HEADS,HD)
    direct=np.sum(q[:8]*reduced,axis=-1)*s2[:8]/HD
    assembled=np.sum(score[:8],axis=1)
    assert np.max(np.abs(direct-assembled))<2e-5
    energy=np.einsum('nrh,nhj,nrj->nr',score,gram,score,optimize=True)
    magnitude=np.sqrt(np.maximum(energy,0))
    return effective_coordinates(magnitude)


def match(a,b):
    similarity=np.abs(a.T@b)
    i,j=opt.linear_sum_assignment(-similarity)
    return float(np.mean(similarity[i,j]))


def token_panels(vocab,train_size=2048,test_size=2048):
    frequency=np.bincount(np.asarray(np.load(CORPUS,mmap_mode='r')[:15000]).ravel(),
                          minlength=vocab).astype(np.float64)
    probability=.5/vocab+.5*frequency/max(frequency.sum(),1)
    rng=np.random.default_rng(2061)
    train=rng.choice(vocab,size=train_size,replace=True,p=probability)
    laws={'uniform':rng.integers(0,vocab,size=test_size),
          'fineweb':np.asarray(np.load(CORPUS,mmap_mode='r')[17500:20000]).ravel(),
          **{name:np.asarray(np.load(path,mmap_mode='r')[:1200]).ravel()
             for name,path in DATASETS.items()}}
    for name,tokens in laws.items():
        if name=='uniform':continue
        laws[name]=rng.choice(tokens,size=test_size,replace=True)
    validation_tokens=np.asarray(np.load(CORPUS,mmap_mode='r')[15000:17500]).ravel()
    validation=np.concatenate((rng.integers(0,vocab,size=train_size//2),
                    rng.choice(validation_tokens,size=train_size//2,replace=True)))
    return train,validation,laws


def run(base='function',train_size=2048,test_size=2048,
        pair_size=2048,max_iter=120,direct_steps=160):
    torch.set_num_threads(4)
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    vocab=state['transformer.wte.weight'].shape[0]
    table=project_all_heads(state,np.arange(vocab),('c_k',))['c_k'].reshape(
          vocab,WIDTH).astype(np.float64)
    if base=='function':
        source='all_head_function_aware_shared_key_s642_n32768_t160.npz'
        basis=np.load(ART/source)['basis'].astype(np.float64)
    elif base=='usage':
        source='all_head_shared_key_dictionary_bases.npz'
        basis=np.load(ART/source)['shared_half_corpus'][:,:256].astype(np.float64)
    else:raise ValueError(base)
    assert np.linalg.norm(basis.T@basis-np.eye(256))<1e-3
    train,validation,laws=token_panels(vocab,train_size,test_size)
    train_z=table[train]@basis
    train_z=train_z/np.maximum(np.linalg.norm(train_z,axis=1,keepdims=True),1e-12)
    val_z=table[validation]@basis
    val_z=val_z/np.maximum(np.linalg.norm(val_z,axis=1,keepdims=True),1e-12)
    weight_x=basis.copy()
    rotations={'native':np.eye(256)}
    fit={}
    for seed in (0,1):
        start=rotation(9100+seed,256)
        rotations[f'random_{seed}']=start
        for kind,x in (('weight',weight_x),('response',train_z)):
            r,info=varimax(x,start,max_iter)
            rotations[f'{kind}_{seed}']=r
            fit[f'{kind}_{seed}']=info
        if direct_steps:
            r,info=direct_response_l1(train_z,val_z,np.eye(256),
                                      direct_steps,seed)
            rotations[f'direct_l1_native_{seed}']=r
            fit[f'direct_l1_native_{seed}']=info
            r,info=direct_response_l1(train_z,val_z,start,
                                      direct_steps,100+seed)
            rotations[f'direct_l1_random_{seed}']=r
            fit[f'direct_l1_random_{seed}']=info
    pair_panels,formula_gate,_=make_panels(state,128,128,pair_size,
                                            test_seed_offset=20000)
    del state
    reference_probe={law:float(error(panel,torch.from_numpy(
                     basis.astype(np.float32)),slice(0,64)))
                     for law,panel in pair_panels.items()
                     if law.startswith('test_')}
    records=[]
    for name,r in rotations.items():
        u=basis@r
        orth=float(np.linalg.norm(r.T@r-np.eye(256)))
        projector=float(np.linalg.norm(u@u.T-basis@basis.T)/
                        np.linalg.norm(basis@basis.T))
        assert max(orth,projector)<1e-7,(name,orth,projector)
        function_probe=max(abs(float(error(panel,torch.from_numpy(
                           u.astype(np.float32)),slice(0,64)))-
                           reference_probe[law])
                           for law,panel in pair_panels.items()
                           if law.startswith('test_'))
        assert function_probe<1e-6,(name,function_probe)
        common=dict(gauge=name,projector_relative_error=projector,
                    orthogonality_error=orth,
                    function_probe_max_abs_error=function_probe,
                    basis_entry_l1=float(np.sum(np.abs(u))),
                    mean_effective_heads=head_participation(u))
        for law,ids in laws.items():
            records.append(dict(**common,metric='token',law=law,
                effective_coordinates=effective_coordinates(table[ids]@u)))
        for law,panel in pair_panels.items():
            if not law.startswith('test_'):continue
            for constant in (False,True):
                records.append(dict(**common,metric='pair',law=law,
                    qk2_constant=constant,
                    effective_coordinates=pair_concentration(panel,u,constant)))
        print('evaluated',name,flush=True)
    kinds=('random','weight','response')+(
          ('direct_l1_native','direct_l1_random') if direct_steps else ())
    matches={kind:match(rotations[f'{kind}_0'],rotations[f'{kind}_1'])
             for kind in kinds}
    output=dict(base=base,source_basis=source,rank=256,
                max_varimax_iterations=max_iter,
                direct_l1_steps=direct_steps,
                train_token_law='half uniform, half FineWeb docs 0:15000 frequency',
                train_token_samples=train_size,test_token_samples=test_size,
                pair_samples_per_law=pair_size,pair_test_seed_offset=20000,
                formula_gate=formula_gate,fit=fit,seed_matches=matches,
                records=records,
                scope='exact orthogonal gauges of fixed all-head key subspace; concentration and stability proxies only')
    stem=(f'all_head_shared_key_gauge_priors_{base}_n{train_size}'
          f'_p{pair_size}_i{max_iter}_d{direct_steps}')
    (ART/f'{stem}.json').write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}.npz',**rotations)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',choices=('function','usage'),default='function')
    p.add_argument('--train-size',type=int,default=2048)
    p.add_argument('--test-size',type=int,default=2048)
    p.add_argument('--pair-size',type=int,default=2048)
    p.add_argument('--max-iter',type=int,default=120)
    p.add_argument('--direct-steps',type=int,default=160)
    a=p.parse_args()
    run(a.base,a.train_size,a.test_size,a.pair_size,a.max_iter,
        a.direct_steps)
