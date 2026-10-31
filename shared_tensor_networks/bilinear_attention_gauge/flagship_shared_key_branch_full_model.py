"""Full-model fidelity of shared QK1 key cuts under a QK2-constant edit."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from flagship_component_ablation import kl_per_document
from flagship_full_model_eval import GPT,GPTConfig,logits
from flagship_matrix_free import CORPUS,HD,SNAPSHOT
from flagship_ood_corpora import DATASETS
from flagship_shared_key_full_model import KeyDictionaryEdit
from flagship_transfer import ART


class BranchKeyEdit(KeyDictionaryEdit):
    """Interpolate QK2 scores toward one, then project QK1 keys."""

    def __init__(self,model,length,strength):
        super().__init__(model,length)
        self.strength=strength

    def forward(self,q,k,v,q2,k2):
        if self.shared is None:
            changed=k
        else:
            rotation=self.rot[:k.shape[1]].to(device=k.device,dtype=k.dtype)
            inverse=self.inv[:k.shape[1]].to(device=k.device,dtype=k.dtype)
            pre=torch.einsum('bthd,tde->bthe',k,inverse)
            basis=self.shared.to(device=k.device,dtype=k.dtype)
            flat=pre.reshape(-1,pre.shape[-2]*pre.shape[-1])
            changed=((flat@basis)@basis.T).reshape_as(pre)
            changed=torch.einsum('bthd,tde->bthe',changed,rotation)
        d=q.shape[-1]
        score1=torch.einsum('bthd,bshd->bhts',q,changed)/d
        if self.strength==1.:
            score2=1.
        else:
            native_score2=torch.einsum('bthd,bshd->bhts',q2,k2)/d
            score2=(1-self.strength)*native_score2+self.strength
        pattern=(score1*score2).tril()
        return torch.einsum('bhts,bshd->bhtd',pattern,v)


def bases():
    prior=np.load(ART/'all_head_shared_key_dictionary_bases.npz')
    stems={
        'natural_function_256':'all_head_function_aware_shared_key_s642_n32768_t160.npz',
        'branch_half_256':'all_head_function_aware_shared_key_s644_n32768_t160_branch50.npz',
        'branch_only_256':'all_head_function_aware_shared_key_s645_n32768_t160_branch100.npz',
    }
    out={'usage_svd_256':prior['shared_half_corpus'][:,:256].copy()}
    for name,stem in stems.items():
        out[name]=np.load(ART/stem)['basis'].copy()
    assert all(v.shape==(9*HD,256) for v in out.values())
    assert all(np.linalg.norm(v.T@v-np.eye(256))/256<1e-5
               for v in out.values())
    return out


def exclusions():
    excluded={}
    sources=[]
    for path in sorted(ART.glob('all_head_shared_key_full_model_*.json')):
        prior=json.loads(path.read_text())
        if 'records' not in prior:continue
        sources.append(path.name)
        for row in prior['records']:
            excluded.setdefault(row['corpus'],set()).add(row['document_id'])
    return excluded,sources


def run(n_docs=8,seeds=(91,93),length=128,batch_size=2,
        branch_strength=1.):
    torch.set_num_threads(4)
    assert 0<=branch_strength<=1
    methods=bases()
    config=json.loads((SNAPSHOT/'config.json').read_text());config.pop('step',None)
    model=GPT(GPTConfig(**config)).eval()
    state=torch.load(SNAPSHOT/'pytorch_model.bin',map_location='cpu',
                     weights_only=True,mmap=True)
    model.load_state_dict(state);del state
    model.requires_grad_(False)
    editor=BranchKeyEdit(model,length,branch_strength)
    corpora={'fineweb_test':(CORPUS,17500,20000),
             **{name:(path,0,1200) for name,path in DATASETS.items()}}
    excluded,sources=exclusions()
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
                    spans=[];starts=[]
                    for doc in docs[offset:offset+batch_size]:
                        start=int(rng.integers(0,corpus.shape[1]-length))
                        starts.append(start)
                        spans.append(np.asarray(corpus[doc,start:start+length+1],
                                                dtype=np.int64))
                    idx=torch.from_numpy(np.stack(spans)[:,:-1].copy())
                    targets=torch.from_numpy(np.stack(spans)[:,1:].copy())
                    editor.attn.squared_attention=editor.native
                    native_reference=logits(model,idx)
                    editor.attn.squared_attention=editor.forward
                    editor.reset()
                    branch_reference=logits(model,idx)
                    if identity_gate is None:
                        editor.set_shared(np.eye(9*HD,dtype=np.float32))
                        identity_gate=float((logits(model,idx)-branch_reference).
                                            abs().max())
                        assert identity_gate<1e-3,identity_gate
                    branch_ce=F.cross_entropy(branch_reference.flatten(0,1),
                        targets.ravel(),reduction='none').reshape(len(idx),length).mean(-1)
                    native_ce=F.cross_entropy(native_reference.flatten(0,1),
                        targets.ravel(),reduction='none').reshape(len(idx),length).mean(-1)
                    native_to_branch=kl_per_document(native_reference,branch_reference)
                    for name,basis in methods.items():
                        editor.set_shared(basis)
                        candidate=logits(model,idx)
                        per_doc=kl_per_document(branch_reference,candidate)
                        edited_ce=F.cross_entropy(candidate.flatten(0,1),
                            targets.ravel(),reduction='none').reshape(len(idx),length).mean(-1)
                        flips=(candidate.argmax(-1)!=branch_reference.argmax(-1)).float().mean(-1)
                        for local,doc in enumerate(docs[offset:offset+batch_size]):
                            records.append(dict(corpus=corpus_name,seed=seed,
                                document_id=int(doc),position=offset+local,
                                start_position=starts[local],
                                method=name,kl_per_token=float(per_doc[local]),
                                native_ce=float(native_ce[local]),
                                native_to_branch_kl_per_token=float(native_to_branch[local]),
                                branch_reference_ce=float(branch_ce[local]),
                                edited_ce=float(edited_ce[local]),
                                ce_delta=float(edited_ce[local]-branch_ce[local]),
                                argmax_flip_fraction=float(flips[local])))
                print(corpus_name,seed,flush=True)
    output=dict(scope='all 18 layers; first-layer all-nine-head QK2 scores interpolated toward one; compare same-rank shared QK1 key cuts to the unprojected branch-edited model',
        branch_strength=branch_strength,
        n_docs_per_seed=n_docs,seeds=list(seeds),length=length,
        methods=list(methods),identity_basis_max_abs_logit=identity_gate,
        excluded_document_artifacts=sources,
        excluded_document_counts={k:len(v) for k,v in excluded.items()},
        records=records)
    name=f'all_head_shared_key_branch_full_model_n{n_docs}_s'+'-'.join(map(str,seeds))
    if branch_strength!=1.:
        name+='_a'+str(branch_strength).replace('.','p')
    name+='.json'
    (ART/name).write_text(json.dumps(output,indent=2)+'\n')
    for corpus in corpora:
        for method in methods:
            values=[r['kl_per_token'] for r in records if r['corpus']==corpus
                    and r['method']==method]
            print(corpus,method,'mean KL',float(np.mean(values)),flush=True)
    print('saved',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--n-docs',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[91,93])
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--branch-strength',type=float,default=1.)
    a=p.parse_args()
    run(a.n_docs,tuple(a.seeds),a.length,a.batch_size,a.branch_strength)
