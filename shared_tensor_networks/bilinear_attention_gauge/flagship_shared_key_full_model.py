"""Full-model test of joint versus separate first-layer QK1 key dictionaries."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_ablation import kl_per_document
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_prerope_cut_transfer import row_rotation
from flagship_transfer import ART


class KeyDictionaryEdit:
    def __init__(self,model,length):
        self.attn=model.transformer.h[0].attn
        self.native=self.attn.squared_attention
        self.attn.squared_attention=self.forward
        rot=np.stack([row_rotation(p) for p in range(length)])
        self.rot=torch.from_numpy(rot).to(torch.float32)
        self.inv=torch.from_numpy(np.linalg.inv(rot)).to(torch.float32)
        self.shared=None
        self.independent=None
        self.roundtrip_max=0.

    def set_shared(self,basis):
        self.shared=torch.from_numpy(np.asarray(basis)).to(torch.float32)
        self.independent=None

    def set_independent(self,bases):
        self.independent=torch.from_numpy(np.asarray(bases)).to(torch.float32)
        self.shared=None

    def reset(self):
        self.shared=None;self.independent=None

    def forward(self,q,k,v,q2,k2):
        if self.shared is None and self.independent is None:
            return self.native(q,k,v,q2,k2)
        rotation=self.rot[:k.shape[1]].to(device=k.device,dtype=k.dtype)
        inverse=self.inv[:k.shape[1]].to(device=k.device,dtype=k.dtype)
        pre=torch.einsum('bthd,tde->bthe',k,inverse)
        if self.shared is not None:
            basis=self.shared.to(device=k.device,dtype=k.dtype)
            flat=pre.reshape(-1,pre.shape[-2]*pre.shape[-1])
            new=((flat@basis)@basis.T).reshape_as(pre)
        else:
            basis=self.independent.to(device=k.device,dtype=k.dtype)
            new=torch.einsum('bthr,hdr->bthd',
                torch.einsum('bthd,hdr->bthr',pre,basis),basis)
        post=torch.einsum('bthd,tde->bthe',new,rotation)
        return self.native(q,post,v,q2,k2)


def run(n_docs=8,seeds=(23,29),batch_size=4,length=128,
        exclude_artifacts=(),function_basis=None,
        extra_function_bases=()):
    torch.set_num_threads(4)
    arrays=np.load(ART/'all_head_shared_key_dictionary_bases.npz')
    methods={
        'shared_uniform_256':('shared',arrays['shared_uniform'][:,:256]),
        'shared_uniform_384':('shared',arrays['shared_uniform'][:,:384]),
        'shared_half_corpus_256':('shared',arrays['shared_half_corpus'][:,:256]),
        'shared_half_corpus_384':('shared',arrays['shared_half_corpus'][:,:384]),
        'independent_uniform_64':('independent',arrays['independent_uniform']),
        'independent_half_corpus_64':('independent',arrays['independent_half_corpus']),
    }
    if function_basis is not None:
        learned=np.load(ART/function_basis)['basis']
        assert learned.shape==(9*HD,256)
        methods['function_aware_mixed_256']=('shared',learned)
    for i,name in enumerate(extra_function_bases):
        learned=np.load(ART/name)['basis']
        assert learned.shape==(9*HD,256)
        methods[f'function_aware_extra{i}_256']=('shared',learned)
    config=json.loads((SNAPSHOT/'config.json').read_text())
    config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    editor=KeyDictionaryEdit(model,length)
    corpora={'fineweb_test':(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    excluded={}
    for artifact in exclude_artifacts:
        prior=json.loads((ART/artifact).read_text())
        for row in prior['records']:
            excluded.setdefault(row['corpus'],set()).add(row['document_id'])
    records=[];identity_gate=None
    with torch.no_grad():
        for corpus_name,(path,begin,end) in corpora.items():
            corpus=np.load(path,mmap_mode='r')
            for seed in seeds:
                rng=np.random.default_rng(seed+length*1000)
                choices=np.setdiff1d(np.arange(begin,end),
                                     list(excluded.get(corpus_name,set())))
                docs=rng.choice(choices,size=n_docs,replace=False)
                for offset in range(0,n_docs,batch_size):
                    spans=[]
                    for doc in docs[offset:offset+batch_size]:
                        start=int(rng.integers(0,corpus.shape[1]-length))
                        spans.append(np.asarray(
                            corpus[doc,start:start+length+1],dtype=np.int64))
                    idx=torch.from_numpy(np.stack(spans)[:,:-1].copy())
                    targets=torch.from_numpy(np.stack(spans)[:,1:].copy())
                    editor.reset()
                    reference=logits(model,idx)
                    native_ce=F.cross_entropy(
                        reference.flatten(0,1),targets.ravel(),
                        reduction='none').reshape(len(idx),length).mean(-1)
                    if identity_gate is None:
                        editor.set_shared(np.eye(9*HD,dtype=np.float32))
                        identity_gate=float((logits(model,idx)-reference).
                                            abs().max())
                        assert identity_gate<1e-3,identity_gate
                    for method,(kind,basis) in methods.items():
                        if kind=='shared':editor.set_shared(basis)
                        else:editor.set_independent(basis)
                        candidate=logits(model,idx)
                        per_doc=kl_per_document(reference,candidate)
                        edited_ce=F.cross_entropy(
                            candidate.flatten(0,1),targets.ravel(),
                            reduction='none').reshape(len(idx),length).mean(-1)
                        flips=((candidate.argmax(-1)!=reference.argmax(-1))
                               .to(torch.float32).mean(-1))
                        for local,doc in enumerate(docs[offset:offset+batch_size]):
                            records.append(dict(corpus=corpus_name,seed=seed,
                                document_id=int(doc),position=offset+local,
                                method=method,kl_per_token=float(per_doc[local]),
                                native_ce=float(native_ce[local]),
                                edited_ce=float(edited_ce[local]),
                                ce_delta=float(edited_ce[local]-native_ce[local]),
                                argmax_flip_fraction=float(flips[local])))
                print(corpus_name,seed,flush=True)
    result=dict(layer=0,heads=list(range(9)),length=length,
        n_docs_per_seed=n_docs,seeds=list(seeds),
        source_bases='all_head_shared_key_dictionary_bases.npz',
        methods={name:dict(kind=kind,rank=basis.shape[-1])
                 for name,(kind,basis) in methods.items()},
        identity_basis_max_abs_logit=identity_gate,
        excluded_document_artifacts=list(exclude_artifacts),
        excluded_document_counts={k:len(v) for k,v in excluded.items()},
        function_basis=function_basis,
        extra_function_bases=list(extra_function_bases),
        fineweb_test_documents=[17500,20000],
        scope='all 18 layers, first-layer QK1 key projection for all nine heads, KL relative to native model',
        records=records)
    suffix=f'n{n_docs}'
    if tuple(seeds)!=(23,29):suffix+='_s'+'-'.join(map(str,seeds))
    if exclude_artifacts:suffix+='_newdocs'
    if function_basis is not None or extra_function_bases:
        suffix+='_function'
    if extra_function_bases:suffix+=f'_extra{len(extra_function_bases)}'
    (ART/f'all_head_shared_key_full_model_{suffix}.json').write_text(
        json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-docs',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[23,29])
    p.add_argument('--exclude-artifacts',nargs='+',default=[])
    p.add_argument('--function-basis',default=None)
    p.add_argument('--extra-function-bases',nargs='+',default=[])
    a=p.parse_args()
    run(a.n_docs,tuple(a.seeds),
        exclude_artifacts=tuple(a.exclude_artifacts),
        function_basis=a.function_basis,
        extra_function_bases=tuple(a.extra_function_bases))
