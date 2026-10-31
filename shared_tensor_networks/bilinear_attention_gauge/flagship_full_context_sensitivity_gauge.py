"""Full-model QK1-key sensitivity signatures and exact-gauge comparison.

No reduction is refitted: all rotations preserve one fixed rank-256 projector.
The behavioral readout is the last-position next-token log probability.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_labels import CATEGORIES, Decoder
from flagship_full_model_eval import GPT, GPTConfig, logits
from flagship_matrix_free import CORPUS, HD, SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_full_model import KeyDictionaryEdit
from flagship_shared_key_gauge_priors import varimax
from flagship_transfer import ART


class SensitivityEdit(KeyDictionaryEdit):
    def __init__(self,model,length,basis):
        super().__init__(model,length)
        self.set_shared(basis)
        self.last_pre=None
        self.head_equivalence_max_abs=None
        self.injection=None

    def forward(self,q,k,v,q2,k2):
        rotation=self.rot.to(device=k.device,dtype=k.dtype)
        inverse=self.inv.to(device=k.device,dtype=k.dtype)
        pre=torch.einsum('bthd,tde->bthe',k,inverse).detach().requires_grad_(True)
        self.last_pre=pre
        if self.injection is not None:
            position,direction,amplitude=self.injection
            mask=torch.zeros((1,pre.shape[1],1,1),device=pre.device,dtype=pre.dtype)
            mask[:,position]=1.
            pre=pre+amplitude*mask*direction.reshape(1,1,pre.shape[2],pre.shape[3])
        basis=self.shared.to(device=k.device,dtype=k.dtype)
        flat=pre.reshape(-1,pre.shape[-2]*pre.shape[-1])
        reduced=((flat@basis)@basis.T).reshape_as(pre)
        post=torch.einsum('bthd,tde->bthe',reduced,rotation)
        output=self.native(q,post,v,q2,k2)
        if self.head_equivalence_max_abs is None and self.injection is None:
            with torch.no_grad():
                reference=KeyDictionaryEdit.forward(self,q,k,v,q2,k2)
                self.head_equivalence_max_abs=float((output-reference).abs().max())
                assert self.head_equivalence_max_abs<1e-4
        return output


def signatures(model,editor,basis,corpus,begin,end,n_docs,seed,length):
    data=np.load(corpus,mmap_mode='r')
    rng=np.random.default_rng(seed)
    docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
    vectors=[];keys=[];queries=[];positions=[];document_ids=[]
    readouts=[];norms=[];starts=[];finite_difference_gate=None
    basis_t=torch.from_numpy(basis.astype(np.float32))
    for doc in docs:
        start=int(rng.integers(0,data.shape[1]-length-1))
        tokens=np.asarray(data[doc,start:start+length+1],dtype=np.int64)
        idx=torch.from_numpy(tokens[:-1].copy())[None,:]
        target=int(tokens[-1])
        output=logits(model,idx)
        value=F.log_softmax(output[0,-1].float(),dim=-1)[target]
        grad=torch.autograd.grad(value,editor.last_pre)[0][0]
        projected=grad.reshape(length,-1)@basis_t
        arr=projected.detach().numpy().astype(np.float32)
        assert np.isfinite(arr).all()
        if finite_difference_gate is None:
            position=length-2
            axis=int(np.argmax(np.abs(arr[position])))
            derivative=float(arr[position,axis])
            editor.injection=(position,basis_t[:,axis],.1)
            plus=F.log_softmax(logits(model,idx)[0,-1].float(),dim=-1)[target]
            editor.injection=(position,basis_t[:,axis],-.1)
            minus=F.log_softmax(logits(model,idx)[0,-1].float(),dim=-1)[target]
            editor.injection=None
            finite=float(((plus-minus)/.2).detach())
            finite_difference_gate=dict(position=position,axis=axis,
                autograd_derivative=derivative,central_difference=finite,
                absolute_error=abs(derivative-finite))
            assert abs(derivative-finite)<max(2e-4,.05*abs(derivative)),finite_difference_gate
        vectors.append(arr)
        keys.extend(map(int,tokens[:-1]))
        queries.extend([int(tokens[-2])]*length)
        positions.extend(range(length))
        document_ids.extend([int(doc)]*length)
        starts.extend([start]*length)
        readouts.append(float(value.detach()))
        norms.append(float(np.linalg.norm(arr)))
    return dict(vectors=np.concatenate(vectors),keys=np.asarray(keys),
        queries=np.asarray(queries),positions=np.asarray(positions),
        document_ids=np.asarray(document_ids),starts=np.asarray(starts),
        readout_logprob=readouts,document_projected_gradient_norm=norms,
        sampled_documents=list(map(int,docs)),finite_difference_gate=finite_difference_gate)


def normalize_rows(x):
    norms=np.linalg.norm(x,axis=1)
    threshold=float(np.quantile(norms[norms>0],.1))
    return (x/np.maximum(norms[:,None],threshold)).astype(np.float64),threshold


def effective_coordinates(x):
    power=np.sum(x*x,axis=1)
    return float(np.mean(np.sum(np.abs(x),axis=1)**2/np.maximum(power,1e-30)))


def category_scores(train,test,rotation,decoder,top=20,row_normalized=False):
    """Choose key-token-form labels on train; score unchanged on test."""
    train_energy=(train['vectors']@rotation)**2
    if row_normalized:
        total=np.sum(train_energy,axis=1)
        train_energy/=np.maximum(total[:,None],1e-30)
    order=np.argsort(train_energy.sum(axis=0))[::-1][:top]
    train_cat=np.asarray([decoder.category(int(t)) for t in train['keys']])
    test_cat=np.asarray([decoder.category(int(t)) for t in test['keys']])
    base=np.bincount(train_cat,weights=np.sum(train_energy,axis=1),
                     minlength=len(CATEGORIES)).astype(np.float64)
    base/=max(base.sum(),1e-30)
    eligible=base>=.02
    test_energy=(test['vectors']@rotation)**2
    if row_normalized:
        total=np.sum(test_energy,axis=1)
        test_energy/=np.maximum(total[:,None],1e-30)
    out=[]
    for rank,axis in enumerate(order):
        energy=train_energy[:,axis]
        by=np.bincount(train_cat,weights=energy,minlength=len(CATEGORIES))
        fractions=by/max(by.sum(),1e-30)
        label=int(np.argmax(np.where(eligible,fractions/np.maximum(base,1e-30),-1)))
        held=test_energy[:,axis]
        held_by=np.bincount(test_cat,weights=held,minlength=len(CATEGORIES))
        test_fraction=float(held_by[label]/max(held_by.sum(),1e-30))
        test_base_by=np.bincount(test_cat,weights=np.sum(test_energy,axis=1),
                                minlength=len(CATEGORIES))
        test_base=float(test_base_by[label]/max(test_base_by.sum(),1e-30))
        out.append(dict(selection_rank=rank,axis=int(axis),
            key_category=CATEGORIES[label],train_fraction=float(fractions[label]),
            train_baseline=float(base[label]),test_fraction=test_fraction,
            test_baseline=test_base,
            heldout_log2_enrichment=float(np.log2(max(test_fraction,1e-30)/
                                                   max(test_base,1e-30)))))
    return out


def run(train_docs=8,test_docs=4,length=64,seed=1031):
    torch.set_num_threads(4)
    basis=np.load(ART/'all_head_function_aware_shared_key_s642_n32768_t160.npz')['basis']
    assert basis.shape==(9*HD,256)
    assert np.max(np.abs(basis.T@basis-np.eye(256)))<1e-3
    rotations=np.load(ART/'all_head_shared_key_gauge_priors_function_n2048_p2048_i120_d600.npz')
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    model.requires_grad_(False)
    editor=SensitivityEdit(model,length,basis)
    panels={}
    settings={
        'train_fineweb':(CORPUS,0,15000,train_docs,seed),
        'validation_fineweb':(CORPUS,15000,17500,test_docs,seed+1),
        'test_fineweb':(CORPUS,17500,20000,test_docs,seed+2),
        'test_wikitext103':(DATASETS['wikitext103'],0,1200,test_docs,seed+3),
        'test_tinystories':(DATASETS['tinystories'],0,1200,test_docs,seed+4),
    }
    for name,args in settings.items():
        panels[name]=signatures(model,editor,basis,*args,length)
        print(name,'documents',len(panels[name]['sampled_documents']),flush=True)
    train=panels['train_fineweb'];normalized,threshold=normalize_rows(train['vectors'])
    starts={name:np.asarray(rotations[name],dtype=np.float64)
            for name in ('native','random_0','weight_0','response_0')}
    full_context,trace=varimax(normalized,np.eye(256),max_iter=240)
    starts['full_context_varimax']=full_context
    decoder=Decoder()
    results={}
    for name,rotation in starts.items():
        assert np.max(np.abs(rotation.T@rotation-np.eye(256)))<2e-3,name
        results[name]={}
        for law,panel in panels.items():
            coordinates=panel['vectors'].astype(np.float64)@rotation
            results[name][law]=dict(
                effective_coordinates=effective_coordinates(coordinates),
                mean_gradient_norm=float(np.mean(np.linalg.norm(coordinates,axis=1))),
                key_category_labels=category_scores(train,panel,rotation,decoder),
                row_normalized_key_category_labels=category_scores(
                    train,panel,rotation,decoder,row_normalized=True))
    output=dict(layer=0,heads=list(range(9)),rank=256,
        length=length,train_docs=train_docs,test_docs=test_docs,seed=seed,
        readout='last-position next-token log probability through full 18-layer model',
        intervention='infinitesimal additive pre-RoPE QK1 key perturbation, projected into fixed rank-256 basis',
        row_normalization='unit norm with tenth-percentile floor on train; only used to fit full-context varimax',
        train_normalization_floor=threshold,varimax_trace=trace,
        head_equivalence_max_abs=editor.head_equivalence_max_abs,
        finite_difference_gate=panels['train_fineweb']['finite_difference_gate'],
        corpus_splits={name:panel['sampled_documents'] for name,panel in panels.items()},
        results=results,
        caveat='gradient and key-form enrichment proxy; finite causal edits and task-level behavior still required')
    stem=f'all_head_full_context_sensitivity_gauge_train{train_docs}_test{test_docs}_l{length}_s{seed}'
    (ART/f'{stem}.json').write_text(json.dumps(output,indent=2)+'\n')
    np.savez_compressed(ART/f'{stem}.npz',
        **{f'{law}_vectors':panel['vectors'] for law,panel in panels.items()},
        **{f'{law}_keys':panel['keys'] for law,panel in panels.items()},
        **{f'rotation_{name}':rotation for name,rotation in starts.items()})
    return output


def rescore_row_normalized_labels(stem):
    """Add robust label proxy to a saved run without repeating model passes."""
    path=ART/f'{stem}.json'
    output=json.loads(path.read_text())
    arrays=np.load(ART/f'{stem}.npz')
    train=dict(vectors=arrays['train_fineweb_vectors'],
               keys=arrays['train_fineweb_keys'])
    decoder=Decoder()
    for name in output['results']:
        r=arrays[f'rotation_{name}']
        for law in output['results'][name]:
            panel=dict(vectors=arrays[f'{law}_vectors'],keys=arrays[f'{law}_keys'])
            output['results'][name][law]['row_normalized_key_category_labels']=\
                category_scores(train,panel,r,decoder,row_normalized=True)
    path.write_text(json.dumps(output,indent=2)+'\n')
    return output


def refit_from_saved(stem,max_iter=240):
    """Improve gauge convergence without rerunning the full model."""
    path=ART/f'{stem}.json'
    output=json.loads(path.read_text())
    npz_path=ART/f'{stem}.npz'
    with np.load(npz_path) as existing:
        arrays={name:existing[name] for name in existing.files}
    train=arrays['train_fineweb_vectors'].astype(np.float64)
    norm=np.linalg.norm(train,axis=1)
    train/=np.maximum(norm[:,None],output['train_normalization_floor'])
    fitted,trace=varimax(train,np.eye(256),max_iter=max_iter)
    arrays['rotation_full_context_varimax']=fitted
    output['varimax_trace']=trace
    decoder=Decoder()
    source=dict(vectors=arrays['train_fineweb_vectors'],keys=arrays['train_fineweb_keys'])
    for law in output['results']['full_context_varimax']:
        panel=dict(vectors=arrays[f'{law}_vectors'],keys=arrays[f'{law}_keys'])
        coordinates=panel['vectors'].astype(np.float64)@fitted
        row=output['results']['full_context_varimax'][law]
        row['effective_coordinates']=effective_coordinates(coordinates)
        row['mean_gradient_norm']=float(np.mean(np.linalg.norm(coordinates,axis=1)))
        row['key_category_labels']=category_scores(source,panel,fitted,decoder)
        row['row_normalized_key_category_labels']=category_scores(
            source,panel,fitted,decoder,row_normalized=True)
    np.savez_compressed(npz_path,**arrays)
    path.write_text(json.dumps(output,indent=2)+'\n')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-docs',type=int,default=8)
    parser.add_argument('--test-docs',type=int,default=4)
    parser.add_argument('--length',type=int,default=64)
    parser.add_argument('--seed',type=int,default=1031)
    parser.add_argument('--rescore-stem')
    parser.add_argument('--refit-stem')
    parser.add_argument('--max-iter',type=int,default=240)
    args=parser.parse_args()
    if args.refit_stem:refit_from_saved(args.refit_stem,args.max_iter)
    elif args.rescore_stem:rescore_row_normalized_labels(args.rescore_stem)
    else:run(args.train_docs,args.test_docs,args.length,args.seed)
