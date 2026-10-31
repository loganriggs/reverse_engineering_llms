"""Exact output-isometric gauges of the frozen L9H7 rank-96 cut."""

import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from flagship_all_head_rank_frontier import top_basis
from flagship_full_model_eval import GPT, GPTConfig
from flagship_later_head_rank_pilot import HeadCut, final_logits
from flagship_matrix_free import SNAPSHOT
from flagship_transfer import ART

SEED=20261103
RANDOM_CONTROLS=32


def varimax(matrix,max_iter=200,tolerance=1e-9):
    """Orthomax gamma=1 rotation of matrix columns, preserving their span."""
    n,k=matrix.shape
    r=np.eye(k)
    previous=-np.inf
    for _ in range(max_iter):
        z=matrix@r
        gradient=matrix.T@(z**3-z@np.diag(np.sum(z*z,axis=0)/n))
        u,s,vh=np.linalg.svd(gradient,full_matrices=False)
        r=u@vh
        value=float(np.sum(s))
        if value-previous<max(tolerance,abs(value)*tolerance):break
        previous=value
    return r


def type_signature(z,types,names):
    cov=[]
    for name in names:
        x=z[types==name]
        c=x.T@x/len(x)
        cov.append(c/max(np.trace(c),1e-30))
    cov=np.stack(cov)
    # Distinct fixed type codes break the exact degeneracy of a sum of
    # squared contrasts for equally strong, perfectly separated types.
    codes=np.random.default_rng(179).normal(size=len(names))
    signature=np.einsum('t,tij->ij',codes,cov)
    return top_basis(signature),cov


def axis_match(a,b):
    overlap=np.abs(a.T@b)
    row,col=linear_sum_assignment(-overlap)
    return float(overlap[row,col].mean())


def metrics(z,d,types,names):
    energy=np.mean(z*z,axis=0)
    sample=z*z
    participation=np.mean((np.sum(sample,axis=1)**2)/
                          np.sum(sample*sample,axis=1).clip(1e-30))
    type_energy=np.stack([np.mean(z[types==name]**2,axis=0) for name in names])
    shares=type_energy/type_energy.sum(axis=0,keepdims=True).clip(1e-30)
    weighted_purity=float(np.sum(energy*np.max(shares,axis=0))/
                          np.sum(energy).clip(1e-30))
    equalized=type_energy/type_energy.sum(axis=1,keepdims=True).clip(1e-30)
    equalized_axis_energy=equalized.sum(axis=0)
    balanced_purity=float(np.sum(np.max(equalized,axis=0))/
                          np.sum(equalized_axis_energy).clip(1e-30))
    return dict(output_dictionary_mean_abs=float(np.mean(np.abs(d))),
                edit_effective_axes=float(participation),
                type_energy_weighted_purity=weighted_purity,
                equal_type_energy_purity=balanced_purity,
                top_type_axis={name:int(np.argmax(type_energy[i]/
                    (np.mean(np.delete(type_energy,i,axis=0),axis=0)+1e-30)))
                    for i,name in enumerate(names)})


def capture_new(model,probe,cases):
    layer,head=9,7
    out=np.empty((len(cases),128),dtype=np.float32)
    groups={}
    for i,r in enumerate(cases):groups.setdefault(len(r['base_ids']),[]).append(i)
    with torch.no_grad():
        for indices in groups.values():
            for start in range(0,len(indices),8):
                group=indices[start:start+8]
                before=torch.tensor([cases[i]['base_ids'] for i in group],dtype=torch.long)
                after=torch.tensor([cases[i]['edited_ids'] for i in group],dtype=torch.long)
                final_logits(model,before,stop_at=layer)
                base=probe.last_input[:,-1,head*128:(head+1)*128].clone()
                final_logits(model,after,stop_at=layer)
                out[group]=(probe.last_input[:,-1,head*128:(head+1)*128]-base).numpy()
    return out


def check():
    rng=np.random.default_rng(23)
    raw=np.zeros((400,4));raw[np.arange(400),rng.integers(0,4,400)]=rng.normal(size=400)
    q,_=np.linalg.qr(rng.normal(size=(4,4)))
    r=varimax(raw@q)
    assert axis_match(q@r,np.eye(4))>.95
    types=np.repeat(np.arange(4),100)
    z=np.zeros((400,4));z[np.arange(400),types]=rng.normal(size=400)
    r,_=type_signature(z@q,types,list(range(4)))
    assert axis_match(q@r,np.eye(4))>.95


def run():
    check()
    torch.set_num_threads(4)
    rng=np.random.default_rng(SEED)
    gate=json.loads((ART/'train_finite_fidelity_gate.json').read_text())
    assert (gate['layer'],gate['head'],gate['selected']['rank'],
            gate['selected']['method'])==(9,7,96,'pooled_edit')
    scan=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    first=json.loads((ART/'new_family_path_transfer_rank96.json').read_text())
    second=json.loads((ART/'gate_second_family_stress.json').read_text())
    train=scan['layer9__train'][:,7].astype(np.float64)
    val=scan['layer9__validation'][:,7].astype(np.float64)
    train_types=np.array([r['type'] for r in meta['panels']['train']])
    val_types=np.array([r['type'] for r in meta['panels']['validation']])
    names=sorted(set(train_types))
    u=top_basis(train.T@train/len(train))[:,:96]
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    probe=HeadCut(model.transformer.h[9].attn)
    projection=model.transformer.h[9].attn.c_proj.weight.detach().numpy()
    w=projection[:,7*128:8*128].astype(np.float64)
    g=u.T@w.T@w@u
    eig,v=np.linalg.eigh((g+g.T)/2)
    assert eig.min()>1e-8,eig.min()
    root=(v*np.sqrt(eig))@v.T
    invroot=(v/np.sqrt(eig))@v.T
    d=w@u@invroot
    assert np.max(np.abs(d.T@d-np.eye(96)))<1e-9
    a_train=train@u@root
    a_val=val@u@root
    cases=first['cases']+second['cases']
    novel=capture_new(model,probe,cases).astype(np.float64)
    novel_types=np.array([r['type'] for r in cases])
    novel_names=sorted(set(novel_types))
    a_novel=novel@u@root
    np.savez_compressed(ART/'rank96_exact_gauge_novel_deltas.npz',
                        novel=novel.astype(np.float32))
    balanced=a_train.copy()
    for name in names:
        mask=train_types==name
        balanced[mask]/=np.sqrt(np.mean(np.sum(a_train[mask]**2,axis=1))).clip(1e-30)
    rotations={'output_weight_varimax':varimax(d),
               'edit_activation_varimax':varimax(balanced),
               'edit_type_signature':type_signature(a_train,train_types,names)[0],
               'pooled_edit_pca':top_basis(a_train.T@a_train/len(a_train)),
               'identity':np.eye(96)}
    for i in range(RANDOM_CONTROLS):
        rotations[f'random_{i}']=np.linalg.qr(rng.normal(size=(96,96)))[0]
    rows=[]
    baseline=a_val@d.T
    native_reduced=val@u@u.T@w.T
    assert np.linalg.norm(baseline-native_reduced)/np.linalg.norm(native_reduced)<1e-10
    for name,r in rotations.items():
        assert np.max(np.abs(r.T@r-np.eye(96)))<1e-8
        reconstruction=(a_val@r)@(d@r).T
        invariant=float(np.linalg.norm(reconstruction-baseline)/np.linalg.norm(baseline))
        assert invariant<1e-10,invariant
        row=dict(method=name,exact_output_relative_error=invariant,
                 train=metrics(a_train@r,d@r,train_types,names),
                 validation=metrics(a_val@r,d@r,val_types,names),
                 unseen=metrics(a_novel@r,d@r,novel_types,novel_names))
        rows.append(row)
        print('gauge',name,'exact err',invariant,'new effective axes',
              row['unseen']['edit_effective_axes'],flush=True)
    result=dict(description='Exact output-isometric rotations of frozen L9H7 rank96 pooled-edit cut. Weight varimax, balanced-edit activation varimax, fixed-code edit-type signature eigengauge, pooled edit PCA, identity, and 32 random controls. Every rotation reconstructs the same immediate c_proj output; all stage-one final-logit/path/KL fidelity is identical. Train known edit types fit gauges; independent same-type edits and ten whole-new-family types evaluate. Type signature uses known type labels; other learned gauges do not. No task answer labels or named-rule recovery.',
        layer=9,head=7,rank=96,seed=SEED,n_random=RANDOM_CONTROLS,
        train_types=names,unseen_types=novel_names,
        output_gram_condition_number=float(eig.max()/eig.min()),rows=rows)
    (ART/'rank96_exact_gauge_priors.json').write_text(json.dumps(result,
                                                     separators=(',',':'))+'\n')
    np.savez_compressed(ART/'rank96_exact_gauge_rotations.npz',**rotations)


if __name__=='__main__':run()
