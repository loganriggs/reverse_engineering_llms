"""Split and independent-panel axis stability of fixed rank-96 gauges."""

import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from flagship_all_head_rank_frontier import top_basis
from flagship_matrix_free import SNAPSHOT
from flagship_rank96_exact_gauge_priors import varimax,type_signature,axis_match
from flagship_transfer import ART

SEED=20261104


def top_energy_match(a,b,z,k=20):
    energy=np.mean((z@a)**2,axis=0)
    selected=np.argsort(energy)[::-1][:k]
    overlap=np.abs(a[:,selected].T@b)
    row,col=linear_sum_assignment(-overlap)
    return float(overlap[row,col].mean())


def run():
    saved=np.load(ART/'all_head_unlabeled_edit_scan.npz')
    meta=json.loads((ART/'later_head_unlabeled_edits.json').read_text())
    train=saved['layer9__train'][:,7].astype(np.float64)
    val=saved['layer9__validation'][:,7].astype(np.float64)
    train_types=np.array([r['type'] for r in meta['panels']['train']])
    val_types=np.array([r['type'] for r in meta['panels']['validation']])
    names=sorted(set(train_types))
    u=top_basis(train.T@train/len(train))[:,:96]
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    w=state['transformer.h.9.attn.c_proj.weight'].numpy()[:,7*128:8*128].astype(np.float64)
    g=u.T@w.T@w@u
    eig,v=np.linalg.eigh((g+g.T)/2)
    root=(v*np.sqrt(eig))@v.T
    z=train@u@root
    y=val@u@root
    first=[];second=[]
    for name in names:
        indices=np.flatnonzero(train_types==name)
        first+=indices[:40].tolist();second+=indices[40:].tolist()
    first=np.array(first);second=np.array(second)
    def balanced(values,types):
        out=values.copy()
        for name in names:
            mask=types==name
            out[mask]/=np.sqrt(np.mean(np.sum(values[mask]**2,axis=1))).clip(1e-30)
        return out
    def fit(values,types):
        return {'edit_activation_varimax':varimax(balanced(values,types)),
                'edit_type_signature':type_signature(values,types,names)[0],
                'pooled_edit_pca':top_basis(values.T@values/len(values))}
    a=fit(z[first],train_types[first]);b=fit(z[second],train_types[second]);c=fit(y,val_types)
    full=np.load(ART/'rank96_exact_gauge_rotations.npz')
    records=[]
    for name in a:
        records.append(dict(method=name,
            split_axis_match=axis_match(a[name],b[name]),
            train_validation_axis_match=axis_match(full[name],c[name]),
            split_top20_energy_match=top_energy_match(a[name],b[name],z[first]),
            train_validation_top20_energy_match=top_energy_match(full[name],c[name],z)))
    rng=np.random.default_rng(SEED)
    random=[]
    for _ in range(32):
        ra=np.linalg.qr(rng.normal(size=(96,96)))[0]
        rb=np.linalg.qr(rng.normal(size=(96,96)))[0]
        random.append(dict(axis_match=axis_match(ra,rb),
                           top20_energy_match=top_energy_match(ra,rb,z)))
    result=dict(description='Independent edit-panel and half-training-panel refits of exact rank96 gauges within the same frozen output-isometric subspace. Hungarian absolute-cosine axis matching, overall and for twenty axes with largest training edit energy. No downstream task labels. Validation refit used only to audit stability, not to select the prior.',
        methods=records,random_control_mean={key:float(np.mean([r[key] for r in random]))
                                            for key in random[0]},random_controls=random)
    (ART/'rank96_gauge_stability.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in records:print(row)
    print('random',result['random_control_mean'])


if __name__=='__main__':run()
