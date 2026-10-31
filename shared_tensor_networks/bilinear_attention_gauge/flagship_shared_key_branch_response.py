"""Full-546M small-QK2-edit response fidelity of matched rank-256 bases."""

import argparse
import json

import numpy as np
import torch

from flagship_component_ablation import kl_per_document
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_full_model import KeyDictionaryEdit
from flagship_transfer import ART


class BranchResponseEdit(KeyDictionaryEdit):
    def __init__(self,model,length):
        self.dose=0.
        super().__init__(model,length)

    def forward(self,q,k,v,q2,k2):
        if self.shared is None and self.independent is None and self.dose==0:
            return self.native(q,k,v,q2,k2)
        if self.independent is not None:
            raise ValueError('this comparison uses matched shared rank only')
        if self.shared is not None:
            rotation=self.rot.to(device=k.device,dtype=k.dtype)
            inverse=self.inv.to(device=k.device,dtype=k.dtype)
            pre=torch.einsum('bthd,tde->bthe',k,inverse)
            basis=self.shared.to(device=k.device,dtype=k.dtype)
            flat=pre.reshape(-1,pre.shape[-2]*pre.shape[-1])
            changed=((flat@basis)@basis.T).reshape_as(pre)
            k=torch.einsum('bthd,tde->bthe',changed,rotation)
        score1=torch.einsum('bihd,bjhd->bhij',q,k)/HD
        score2=torch.einsum('bihd,bjhd->bhij',q2,k2)/HD
        score2=(1-self.dose)*score2+self.dose
        pattern=(score1*score2).tril()
        return torch.einsum('bhij,bjhd->bhid',pattern,v)


def load_methods():
    prior=np.load(ART/'all_head_shared_key_dictionary_bases.npz')
    methods={'uniform_key_256':prior['shared_uniform'][:,:256],
             'usage_key_256':prior['shared_half_corpus'][:,:256]}
    for label,stem in (
        ('function_aware_256','all_head_function_aware_shared_key_s642_n32768_t160'),
        ('branch_aware_256','all_head_function_aware_shared_key_s644_n32768_t160_branch50')):
        methods[label]=np.load(ART/f'{stem}.npz')['basis']
    for name,basis in methods.items():
        assert basis.shape==(9*HD,256),(name,basis.shape)
        assert np.max(np.abs(basis.T@basis-np.eye(256)))<1e-3,name
    return methods


def run(n_docs=2,seed=891,length=128,dose=.05):
    torch.set_num_threads(4)
    config=json.loads((SNAPSHOT/'config.json').read_text())
    config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state)
    del state
    editor=BranchResponseEdit(model,length)
    methods=load_methods()
    corpora={'fineweb_test':(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    records=[];identity_gate=None;ordinary_gate=None
    with torch.no_grad():
        for corpus_index,(corpus_name,(path,begin,end)) in enumerate(corpora.items()):
            corpus=np.load(path,mmap_mode='r')
            rng=np.random.default_rng(seed+corpus_index*100000)
            docs=rng.choice(np.arange(begin,end),size=n_docs,replace=False)
            for doc in docs:
                start=int(rng.integers(0,corpus.shape[1]-length))
                idx=torch.from_numpy(np.asarray(corpus[doc,start:start+length],dtype=np.int64).copy())[None,:]
                editor.reset();editor.dose=0.
                native=logits(model,idx)
                if ordinary_gate is None:
                    editor.set_shared(np.eye(9*HD,dtype=np.float32))
                    ordinary_gate=float((logits(model,idx)-native).abs().max())
                    assert ordinary_gate<1e-3,ordinary_gate
                    editor.reset()
                editor.dose=dose
                native_edit=logits(model,idx)
                native_response=native_edit-native
                response_energy=float(native_response.square().sum())
                assert response_energy>0
                if identity_gate is None:
                    editor.set_shared(np.eye(9*HD,dtype=np.float32))
                    identity_gate=float((logits(model,idx)-native_edit).abs().max())
                    assert identity_gate<1e-3,identity_gate
                    editor.reset()
                for name,basis in methods.items():
                    editor.set_shared(basis)
                    editor.dose=0.
                    reduced=logits(model,idx)
                    editor.dose=dose
                    reduced_edit=logits(model,idx)
                    response=reduced_edit-reduced
                    error=float((response-native_response).square().sum())
                    records.append(dict(corpus=corpus_name,document_id=int(doc),
                        start=start,method=name,response_sse=error,
                        native_response_energy=response_energy,
                        response_relative_mse=error/response_energy,
                        ordinary_kl=float(kl_per_document(native,reduced)[0]),
                        edited_kl=float(kl_per_document(native_edit,reduced_edit)[0]),
                        native_response_rms=float(native_response.square().mean().sqrt())))
            print(corpus_name,'docs',len(docs),flush=True)
    result=dict(layer=0,heads=list(range(9)),length=length,dose=dose,
        n_docs_per_corpus=n_docs,seed=seed,methods=list(methods),
        first_layer_edit='QK2 score becomes (1-dose)*score+dose before multiplication with QK1',
        response='full 18-layer final logits; compare edited-minus-unedited output within each basis',
        ordinary_identity_max_abs_logit=ordinary_gate,
        edited_identity_max_abs_logit=identity_gate,
        records=records)
    name=f'all_head_shared_key_branch_response_n{n_docs}_s{seed}_l{length}_d{round(dose*1000)}.json'
    (ART/name).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n-docs',type=int,default=2)
    p.add_argument('--seed',type=int,default=891)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--dose',type=float,default=.05)
    a=p.parse_args();run(a.n_docs,a.seed,a.length,a.dose)
